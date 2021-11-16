import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import csv
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

import model.resnet as models

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_count.py", description='Cell count evaluation')
parser.add_argument('-m', '--model', type=str, default='checkpoint/checkpoint_10epochs.pth',
                    help='path to pretrained model (default: checkpoint/checkpoint_10epochs.pth)')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='image batch size (default: 64)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now),
                    help='path of output details .csv file (default: ./output/<timestamp>)')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

print("Testing settings: ")
print("Model: {} | Image batch size: {} | Output directory: {}"
      .format(args.model, args.batch_size, args.output))
if not os.path.exists(args.output):
    os.mkdir(args.output)

model = models.encoders[torch.load(args.model)['encoder']]
model.fc_tile = nn.Linear(model.fc_tile.in_features, 2)
epoch = torch.load(args.model)['epoch']
model.load_state_dict(torch.load(args.model)['state_dict'])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.ToTensor(), normalize])
# trans = transforms.ToTensor()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
model.to(device)


def test_count(testset, batch_size, workers, output_path):

    global epoch, model

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    fconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
        now, epoch)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count'])

    print('Start testing ...')

    testset.setmode("count")
    output = predict_counts(test_loader)
    for i, count in enumerate(output):
        w.writerow([i, count])

    fconv.close()


def predict_counts(loader):
    """前馈推导一次模型，获取实例分类概率。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    """
    global device, model

    model.setmode("image")
    model.eval()

    with torch.no_grad():
        image_bar = tqdm(loader, desc="cell counting")
        output = [model(input.to(device))[1].squeeze() for input in image_bar]

    print("output.size = ", np.array(output).shape)
    return output


if __name__ == "__main__":
    from dataset.dataset import LystoTestset

    print('Loading Dataset ...')
    imageSet_test = LystoTestset(filepath="data/testing.h5", transform=trans)

    test_count(imageSet_test, batch_size=args.batch_size, workers=args.workers, output_path=args.output)
