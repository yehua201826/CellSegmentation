import collections
import os
import traceback
import glob
import shutil
import argparse
import time
import csv
import simplejson
from collections import OrderedDict
from easydict import EasyDict as edict
from tqdm import tqdm

import numpy as np
import cv2
import imageio
from skimage import io
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tf

from dataset import get_tiles, MaskTestset, PointTestset
from model import nets
from inference import inference_seg
from metrics import dice_coef, euclid_dist, precision_recall
from utils import (dotting,
                   crop_wsi,
                   locate_cells,
                   sort_files,
                   overlap_mask,
                   remove_small_regions,
                   save_images_with_masks)

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_seg.py", description='Segmentation evaluation')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model')
parser.add_argument('--model_path', type=str, help='path to pretrained model')
parser.add_argument('--soft_mask', action='store_true', help='output soft masks to output_path/soft')
parser.add_argument('--area_type', action='store_true',     # true
                    help='split test data by area type, conflict with --cancer_type')
parser.add_argument('--cancer_type', action='store_true',
                    help='split test data by cancer type, conflict with --area_type')
parser.add_argument('-r', '--reg_limit', action='store_true',
                    help='whether or not setting limitation on artifact patches by counting')
parser.add_argument('-D', '--data_path', type=str, default='./data/test.h5',
                    help='path to testing data (default: ./data/test.h5)')
parser.add_argument('-B', '--image_batch_size', type=int, default=128,
                    help='batch size of images (default: 128)')
parser.add_argument('-c', '--threshold', type=float, default=0.5,
                    help='minimal prob of pixels for generating segmentation masks')
parser.add_argument('-w', '--workers', type=int, default=8,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0)')
parser.add_argument('--save_image', action='store_true', help='save model prediction images')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now), metavar='OUTPUT/PATH',
                    help='path of output masked images (default: ./output/<timestamp>)')
parser.add_argument('--resume_from', type=str, default=None, metavar='IMAGE_FILE_NAME.<EXT>',
                    help='ROI image file name (path set in --data_path) to continue testing '
                         'if workers are killed halfway')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
args = parser.parse_args()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n

    @property
    def val(self):
        return self._val.item() if torch.is_tensor(self._val) else self._val

    @property
    def avg(self):
        avg = self._sum / self._count
        return avg.item() if torch.is_tensor(avg) else avg


class MetricGroup:
    def __init__(self):
        self.P = AverageMeter()
        self.R = AverageMeter()
        self.F1 = AverageMeter()
        self.dice = AverageMeter()

    def avg(self):
        return self.P.avg, self.R.avg, self.F1.avg, self.dice.avg

    def val(self):
        return self.P.val, self.R.val, self.F1.val, self.dice.val

    def update(self, vals):
        self.P.update(vals[0])
        self.R.update(vals[1])
        self.F1.update(vals[2])
        self.dice.update(vals[3])


def test(mode="lysto", categorize_by=None, save_image=True):

    if save_image:
        if not os.path.exists(os.path.join(args.output, 'centered')):
            os.makedirs(os.path.join(args.output, 'centered'))
        if not os.path.exists(os.path.join(args.output, 'predict_mask')):
            os.makedirs(os.path.join(args.output, 'predict_mask'))
        if not os.path.exists(os.path.join(args.output, 'predict_mask_binary')):
            os.makedirs(os.path.join(args.output, 'predict_mask_binary'))

    assert mode in {"lysto", "ihc"}
    filename_pattern = "(?<=test_)\d*" if mode == "lysto" else None
    testset = PointTestset(args.data_path, filename_pattern, num_of_imgs=1 if args.debug else 0)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers,
                             pin_memory=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)

    m = torch.load(args.model, map_location=device)
    model = nets[m['encoder']]
    # load all params
    model.load_state_dict(
        OrderedDict({k: v for k, v in m['state_dict'].items()
                     if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                     model.image_module_prefix + model.seg_module_prefix)}),
        strict=False)
    model.setmode("segment")
    model.to(device)
    model.eval()

    f = open(os.path.join(args.output, "center.csv"), 'w', newline="")
    w = csv.writer(f, delimiter=",")
    w.writerow(["id", "count", "tp", "fp", "fn", "p", "r", "f1", "dice"])

    if mode == "lysto" and categorize_by == "cancer_type":
        metrics = {
            "breast": MetricGroup(),
            "colon": MetricGroup(),
            "prostate": MetricGroup()
        }
    elif categorize_by == "area_type":
        metrics = {
            "regular": MetricGroup(),
            "clustered": MetricGroup(),
            "artifact": MetricGroup()
        }
    else:
        metrics = MetricGroup()

    with torch.no_grad():
        for i, (image, mask, points, cancer, area) in enumerate(tqdm(test_loader, desc="testing")):
            points = points[0]
            mask = mask[0].cpu().to(dtype=torch.float32)
            mask_hat = model(image.to(device)).to(dtype=torch.float32)
            mask_hat = F.softmax(mask_hat, dim=1)[:, 1][0].cpu().numpy()

            model.setmode("image")
            output_reg = model(image.to(device))[1].detach()[:, 0].clone().cpu()
            count = np.round(output_reg[0].item()).astype(int)
            model.setmode("segment")
            # cell count limitation
            if args.reg_limit and count == 0:
                mask_hat = 0 * mask_hat

            classes = mask_hat > args.threshold

            classes = remove_small_regions(classes, min_object_size=300, hole_area_threshold=100)
            dice = dice_coef(torch.from_numpy(classes).to(dtype=torch.float32), mask / 255)

            p, r, f1, tp, fp, fn = [0] * 6
            if mode == "lysto" and categorize_by == "cancer_type":
                metrics[cancer[0]].update([p, r, f1, dice])
            elif categorize_by == "area_type":
                metrics[area[0]].update([p, r, f1, dice])
            else:
                metrics.update([p, r, f1, dice])

            if save_image:

                # cover masks on images
                original_img = testset.get_image(i).copy()
                io.imsave(os.path.join(args.output, 'predict_mask_binary', testset.names[i]), classes)
                overlap_mask(original_img, classes, postprocess=False,
                             save=os.path.join(args.output, 'predict_mask',
                                               os.path.splitext(testset.names[i])[0] + "_{}.png".format(str(count))))

            w.writerow([testset.names[i], str(count), str(tp), str(fp), str(fn),
                        str(p), str(r), str(f1), str(dice)])

    if save_image:
        print("Test results saved in \'{}\' and \'{}\'.".format(os.path.join(args.output, 'centered'),
                                                                os.path.join(args.output, 'predict_mask')))
    if mode == "lysto" and categorize_by == "cancer_type":
        print("Breast: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['breast'].avg()))
        print("Colon: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['colon'].avg()))
        print("Prostate: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['prostate'].avg()))
        w.writerow(['Breast---', 'Precision', metrics['breast'].avg()[0], 'Recall', metrics['breast'].avg()[1],
                    'F1', metrics['breast'].avg()[2], 'Dice', metrics['breast'].avg()[3]])
        w.writerow(['Colon---', 'Precision', metrics['colon'].avg()[0], 'Recall', metrics['colon'].avg()[1],
                    'F1', metrics['colon'].avg()[2], 'Dice', metrics['colon'].avg()[3]])
        w.writerow(['Prostate---', 'Precision', metrics['prostate'].avg()[0], 'Recall', metrics['prostate'].avg()[1],
                    'F1', metrics['prostate'].avg()[2], 'Dice', metrics['prostate'].avg()[3]])
        f.close()

    elif categorize_by == "area_type":
        print("Regular areas: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['regular'].avg()))
        print("Clustered cells: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['clustered'].avg()))
        print("Artifacts: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['artifact'].avg()))
        w.writerow(['Regular areas---', 'Precision', metrics['regular'].avg()[0], 'Recall', metrics['regular'].avg()[1],
             'F1', metrics['regular'].avg()[2], 'Dice', metrics['regular'].avg()[3]])
        w.writerow(['Clustered cells---', 'Precision', metrics['clustered'].avg()[0], 'Recall', metrics['clustered'].avg()[1],
             'F1', metrics['clustered'].avg()[2], 'Dice', metrics['clustered'].avg()[3]])
        w.writerow(['Artifacts---', 'Precision', metrics['artifact'].avg()[0], 'Recall', metrics['artifact'].avg()[1],
             'F1', metrics['artifact'].avg()[2], 'Dice', metrics['artifact'].avg()[3]])
        f.close()

    else:
        print("Average Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics.avg()))
        res = open("out.csv", 'a')
        resw = csv.writer(res, delimiter=',')
        resw.writerow([str(args.threshold)] + list(map(str, metrics.avg())))
        res.close()
        return metrics.avg()


if __name__ == "__main__":

    print("Testing settings: ")
    print("Device: {} | Model: {} | Data directory: {} | Image batch size: {}"
          .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.data_path, args.image_batch_size))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.area_type:
        categorize_by = "area_type"
    elif args.cancer_type:
        categorize_by = "cancer_type"
    else:
        categorize_by = None
    print("Mode: {} | Categorize by: {}\nThreshold: {}".format(os.path.basename(args.data_path),
                  categorize_by if categorize_by is not None else "", args.threshold))
    test(os.path.basename(args.data_path), categorize_by=categorize_by, save_image=args.save_image)