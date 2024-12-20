import warnings
import os
import sys
import configparser
import argparse
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Maskset
from model import nets
from train import train_seg, val_seg

warnings.filterwarnings("ignore")
now = int(time.time())

# Training settings
parser = argparse.ArgumentParser(prog="train_seg.py", description='pt.3: cell segmentation branch training.')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model in pt.2')
parser.add_argument('--skip_draw', action='store_true',
                    help='skip binary mask generating step, using the images from data/<pseudomask_dir> instead')
parser.add_argument('-p', '--preprocess', action='store_true',
                    help='whether or not processing result masks by morphological approaches '
                         '(no use if --skip_draw is chosen)')
parser.add_argument('-P', '--pseudomask_dir', type=str, default='pseudomask', metavar='DIRECTORY',
                    help='dir name to save pseudo masks (default: pseudomask)')
parser.add_argument('-b', '--tile_batch_size', type=int, default=40960,     # 10240
                    help='batch size of tiles (default: 40960, no use if --skip_draw is chosen)')
parser.add_argument('-i', '--interval', type=int, default=5,    # 16
                    help='sample interval of tiles (default: 5, no use if --skip_draw is chosen)')
parser.add_argument('-t', '--tile_size', type=int, default=16,      # 32
                    help='size of each tile (default: 16, no use if --skip_draw is chosen)')
parser.add_argument('-c', '--threshold', type=float, default=0.95,
                    help='minimal prob for tiles to show in generating segmentation masks '
                         '(default: 0.95, no use if --skip_draw is chosen)')
parser.add_argument('-B', '--image_batch_size', type=int, default=32,
                    help='batch size of images (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='total number of epochs to train (default: 30)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-s', '--scheduler', type=str, default=None,
                    help='learning rate scheduler if necessary, '
                         '{\'OneCycleLR\', \'ExponentialLR\', \'CosineAnnealingWarmRestarts\'} (default: None)')
parser.add_argument('-a', '--augment', action="store_true", help='apply data augmentation')
parser.add_argument('-w', '--workers', default=8, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('--scratch', action="store_true",
                    help='[ABLATION] encoder is trained if set')
parser.add_argument('--distributed', action="store_true",
                    help='if distributed parallel training is enabled (seems to be no avail)')
parser.add_argument('-d', '--device', type=int, default=1,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint/{}'.format(now), metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint/<timestamp>)')
parser.add_argument('-r', '--resume', type=str, default=None, metavar='MODEL/FILE/PATH',
                    help='continue training from a checkpoint.pth')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()


def train(total_epochs, last_epoch, model, device, optimizer, scheduler, output_path):
    """pt.3: cell segmentation training.

    :param total_epochs:    total number of training epochs
    :param last_epoch:      previous number of training epochs (if resuming training)
    :param model:           nn.Module
    :param device:          cpu or cuda
    :param optimizer:       gradient optimizer of model training
    :param scheduler:       learning rate scheduler
    :param output_path:     directory of model files and training data results
    """

    fconv = open(os.path.join(output_path, '{}-seg-training.csv'.format(now)), 'w')
    fconv.write('epoch,image_seg_loss,val_loss\n')
    fconv.close()

    start = int(time.time())
    with SummaryWriter(comment=output_path.rsplit('/', maxsplit=1)[-1]) as writer:

        print("PT.III - cell segmentation branch training ...")
        best_val = float('inf')
        for epoch in range(1 + last_epoch, total_epochs + 1):
            try:
                # This will lead to problem
                # if device.type == 'cuda':
                #     torch.cuda.manual_seed(epoch)
                # else:
                #     torch.manual_seed(epoch)

                loss = train_seg(train_loader, epoch, total_epochs, model, device, optimizer,
                                 scheduler)

                loss_val = val_seg(val_loader, epoch, total_epochs, model, device)

                print("image seg train loss: {:.4f}, val loss: {:.4f}".format(loss, loss_val))
                fconv = open(os.path.join(output_path, '{}-seg-training.csv'.format(now)), 'a')
                fconv.write('{},{},{}\n'.format(epoch, loss, loss_val))
                fconv.close()

                add_scalar_loss(writer, epoch, loss)

                if loss_val < best_val:
                    best_val = loss_val
                    if not os.path.exists(output_path+'/val'):
                        os.makedirs(output_path+'/val')
                    save_model(epoch, model, optimizer, scheduler, output_path+'/val')

                save_model(epoch, model, optimizer, scheduler, output_path)

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path)
                print("\nTraining interrupted at epoch {}. Model saved in \'{}\'.".format(epoch, output_path))
                sys.exit(0)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def save_model(epoch, model, optimizer, scheduler, output_path, prefix='pt3'):
    """save model as a .pth file. """
    # save all params
    state_dict = OrderedDict({k: v for k, v in model.state_dict().items()
                              if k.startswith(model.encoder_prefix +
                                              model.image_module_prefix +
                                              model.tile_module_prefix +
                                              model.seg_module_prefix)})
    obj = {
        'mode': 'seg',
        'epoch': epoch,
        'state_dict': state_dict,
        'encoder': model.encoder_name,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(obj, os.path.join(output_path, '{}_{}epochs.pth'.format(prefix, epoch)))


def add_scalar_loss(writer, epoch, loss):
    writer.add_scalar("image seg loss", loss, epoch)


if __name__ == "__main__":

    # time.sleep(16800)

    print("Training settings: ")
    print("Training Mode: {} | Device: {} | Model: {} | {} epoch(s) in total\n"
          "No vidation | Initial LR: {} | Output directory: {}"
          .format('segmentation (pt.3)', 'GPU' if torch.cuda.is_available() else 'CPU',
                  args.resume if args.resume else args.model, args.epochs, args.lr, args.output))
    if args.skip_draw:
        print("Read pseudomasks from storage | Image batch size: {}".format(args.image_batch_size))
    else:
        print("Tile batch size: {} | Tile size: {} | Stride: {} | Image batch size: {}".format(
            args.tile_batch_size, args.tile_size, args.interval, args.image_batch_size))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # model setup
    def to_device(model, device):
        if args.distributed:
            import torch.distributed as dist
            if dist.is_nccl_available():
                print('\nNCCL is available. Setup distributed parallel training with {} devices...\n'
                      .format(torch.cuda.device_count()))
                dist.init_process_group(backend='nccl', world_size=1)
                device = torch.device("cuda", args.local_rank)
                model.to(device)
                model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
            else:
                model.to(device)
        else:
            model.to(device)
        return model

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)

    if args.resume:
        cp = torch.load(args.resume, map_location=device)
        model = nets[cp['encoder']]
        # load all params
        model.load_state_dict(
            OrderedDict({k: v for k, v in cp['state_dict'].items()
                         if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                         model.image_module_prefix + model.seg_module_prefix)}),
            strict=False)
        model = to_device(model, device)
        last_epoch = cp['epoch']
        last_epoch_for_scheduler = cp['scheduler']['last_epoch'] if cp['scheduler'] is not None else -1
    elif args.scratch:
        model = nets['resnet50']
        model = to_device(model, device)
        last_epoch = 0
        last_epoch_for_scheduler = -1
        args.skip_draw = True
    else:   # True
        f = torch.load(args.model, map_location=device)
        model = nets[f['encoder']]
        model.load_state_dict(
            OrderedDict({k: v for k, v in f['state_dict'].items()
                         if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                         model.image_module_prefix)}),
            strict=False)
        model = to_device(model, device)
        # load params of resnet encoder, tile head and image head only
        last_epoch = 0
        last_epoch_for_scheduler = -1

    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    training_data_path = config.get("data", "data_path")

    if not args.skip_draw:
        from dataset import LystoDataset
        from inference import inference_tiles

        print('Generating masks using the pretrained model \'{}\' ...'.format(args.model))

        dataset = LystoDataset(os.path.join(training_data_path, "training.h5"), tile_size=args.tile_size,
                               interval=args.interval, augment=False, kfold=None, num_of_imgs=100 if args.debug else 0)
        loader = DataLoader(dataset, batch_size=args.tile_batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=False)
        dataset.setmode(1)
        model.setmode("tile")

        probs = inference_tiles(loader, model, device)

        def rank(dataset, probs, threshold):
            """Sort tiles by inference probabilities. """

            groups = np.array(dataset.tileIDX)
            tiles = np.array(dataset.tiles_grid)

            order = np.lexsort((probs, groups))
            groups = groups[order]
            probs = probs[order]
            tiles = tiles[order]

            index = [prob > threshold for prob in probs]

            return tiles[index], probs[index], groups[index]

        tiles, _, groups = rank(dataset, probs, args.threshold)

        # if args.preprocess:
        from dataset import LystoTestset
        from inference import inference_image
        from utils import generate_masks

        # clear artifact images
        limit_set = LystoTestset(os.path.join(training_data_path, "training.h5"),
                                 num_of_imgs=100 if args.debug else 0)
        limit_loader = DataLoader(limit_set, batch_size=args.image_batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

        limit_set.setmode("image")
        model.setmode("image")

        counts = inference_image(limit_loader, model, device, mode='test')[1]

        img_indices = np.select([counts != 0], [counts]).nonzero()[0]
        indices = [i for i, g in enumerate(groups) if g in img_indices]
        tiles = tiles[indices]
        groups = groups[indices]

        generate_masks(dataset, tiles, groups, preprocess=args.preprocess,
                       output_path=os.path.join(training_data_path, args.pseudomask_dir))

    trainset = Maskset(os.path.join(training_data_path, "training.h5"),
                       os.path.join(training_data_path, args.pseudomask_dir),
                       augment=args.augment, num_of_imgs=100 if args.debug else 0, train=True)
    valset = Maskset(os.path.join(training_data_path, "training.h5"),
                       os.path.join(training_data_path, args.pseudomask_dir),
                       augment=args.augment, num_of_imgs=100 if args.debug else 0, train=False)

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(trainset) if dist.is_nccl_available() and args.distributed else None
        val_sampler = DistributedSampler(valset) if dist.is_nccl_available() and args.distributed else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(trainset, batch_size=args.image_batch_size, shuffle=True, num_workers=args.workers,
                              sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                              sampler=val_sampler, pin_memory=True)

    # optimization settings
    optimizer_params = {'params': model.parameters(),
                        'initial_lr': args.lr}
    optimizers = {
        'SGD': optim.SGD([optimizer_params], lr=args.lr, momentum=0.9, weight_decay=1e-4),
        'Adam': optim.Adam([optimizer_params], lr=args.lr, weight_decay=1e-4)
    }
    schedulers = {
        'OneCycleLR': OneCycleLR,  # note that last_epoch means last iteration number here
        'ExponentialLR': ExponentialLR,
        'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts,
    }
    scheduler_kwargs = {
        'OneCycleLR': {
            'max_lr': args.lr,  # note that input lr means max_lr here
            'epochs': args.epochs,
            'steps_per_epoch': len(train_loader),
        },
        'ExponentialLR': {
            'gamma': 0.9,
        },
        'CosineAnnealingWarmRestarts': {
            'T_0': 5,
        }
    }

    optimizer = optimizers['SGD'] if args.scheduler is not None else optimizers['Adam']
    scheduler = schedulers[args.scheduler](optimizer,
                                           last_epoch=last_epoch_for_scheduler,
                                           **scheduler_kwargs[args.scheduler]) \
        if args.scheduler is not None else None
    if args.resume:
        optimizer.load_state_dict(cp['optimizer'])
        if cp['scheduler'] is not None and scheduler is not None:
            scheduler.load_state_dict(cp['scheduler'])

    model.setmode("segment")
    if args.scratch:
        model.set_encoder_grads(True)

    train(total_epochs=args.epochs,
          last_epoch=last_epoch,
          model=model,
          device=device,
          optimizer=optimizer,
          scheduler=scheduler,
          output_path=args.output
          )
