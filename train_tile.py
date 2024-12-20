import warnings
import os
import sys
import configparser
import argparse
import time
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import LystoDataset
from model import nets
from inference import inference_tiles, sample
from train import train_tile
from evaluate import evaluate_tile

warnings.filterwarnings("ignore")
now = int(time.time())

# Training settings
parser = argparse.ArgumentParser(prog="train_tile.py", description='pt.2: tile classifier training.')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model in pt.1')
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='total number of epochs to train (default: 30)')
parser.add_argument('-b', '--tile_batch_size', type=int, default=40960,     # 10240
                    help='batch size of tiles (default: 40960)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-s', '--scheduler', type=str, default=None,
                    help='learning rate scheduler if necessary, '
                         '{\'OneCycleLR\', \'ExponentialLR\', \'CosineAnnealingWarmRestarts\'} (default: None)')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int,
                    help='validate every (default: 1) epoch(s). To use all data for training, '
                         'set this greater than --epochs')
parser.add_argument('-t', '--tile_size', type=int, default=32,
                    help='size of a certain tile (default: 32)')
parser.add_argument('-i', '--interval', type=int, default=20,   # 16
                    help='interval between adjacent tiles (default: 20)')
parser.add_argument('-k', '--tiles_per_pos', default=1, type=int,
                    help='k tiles are from a single positive cell (default: 1)')
parser.add_argument('-n', '--topk_neg', default=30, type=int,
                    help='top k tiles from a negative image (default: 30)')
parser.add_argument('-R', '--pos_neg_ratio', default=0.5, type=float,
                    help='ratio of sample instances \'pos/neg\' (default: 0.5, unfix this by setting it to None)')
parser.add_argument('-c', '--threshold', type=float, default=0.95,
                    help='minimal prob for tiles to show in generating heatmaps (default: 0.95)')
parser.add_argument('--scratch', action="store_true",
                    help='[ABLATION] encoder is trained if set')
parser.add_argument('--distributed', action='store_true',
                    help='if distributed parallel training is enabled (seems to be no avail)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint/{}'.format(now), metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint/<timestamp>)')
parser.add_argument('-r', '--resume', type=str, default=None, metavar='MODEL/FILE/PATH',
                    help='continue training from a checkpoint.pth')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()


def train(total_epochs, last_epoch, test_every, model, device, crit_cls, optimizer, scheduler,
          threshold, tiles_per_pos, topk_neg, output_path):
    """pt.2: tile classifier training.

    :param total_epochs:    total number of training epochs
    :param last_epoch:      previous number of training epochs (if resuming training)
    :param test_every:      epochs per validation
    :param model:           nn.Module
    :param device:          cpu or cuda
    :param crit_cls:        loss function of classification
    :param optimizer:       gradient optimizer of model training
    :param scheduler:       learning rate scheduler
    :param threshold:       confidence used in validation
    :param tiles_per_pos:   k_p, the number of tiles selected on ``a single pos cell``
                            ($topk_pos = tiles_per_pos * label$)
    :param topk_neg:        k_n, the number of tiles selected on ``the entire neg image``
    :param output_path:     directory of model files and training data results
    """

    # open output file
    fconv = open(os.path.join(output_path, '{}-tile-training.csv'.format(now)), 'w')
    fconv.write('epoch,tile_loss\n')
    fconv.close()
    # training results will be saved in 'output_path/<timestamp>-tile-training.csv'
    if test_every <= args.epochs:
        fconv = open(os.path.join(output_path, '{}-tile-validation.csv'.format(now)), 'w')
        fconv.write('epoch,tile_error,tile_fpr,tile_fnr\n')
        fconv.close()
    # validation results will be saved in 'output_path/<timestamp>-tile-validation.csv'

    validate = lambda epoch, test_every: (epoch + 1) % test_every == 0
    start = int(time.time())
    with SummaryWriter(comment=output_path.rsplit('/', maxsplit=1)[-1]) as writer:
        gamma = 1.

        print("PT.II - tile classifier training ...")

        for epoch in range(1 + last_epoch, total_epochs + 1):
            try:

                # if device.type == 'cuda':
                #     torch.cuda.manual_seed(epoch)
                # else:
                #     torch.manual_seed(epoch)

                trainset.setmode(1)

                probs = inference_tiles(train_loader, model, device, epoch, total_epochs)
                sample(trainset, probs, tiles_per_pos, topk_neg, pos_neg_ratio=args.pos_neg_ratio)

                torch.cuda.empty_cache()  #########

                trainset.setmode(3)
                loss = train_tile(train_loader, epoch, total_epochs, model, device, crit_cls, optimizer,
                                  scheduler, gamma)

                torch.cuda.empty_cache()  #########

                print("tile loss: {:.4f}".format(loss))
                fconv = open(os.path.join(output_path, '{}-tile-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                add_scalar_loss(writer, epoch, loss)

                # Validating step
                if validate(epoch, test_every):
                    valset.setmode(1)
                    print('Validating ...')

                    probs_t = inference_tiles(val_loader, model, device, epoch, total_epochs)
                    metrics_t = evaluate_tile(valset, probs_t, tiles_per_pos, threshold)

                    torch.cuda.empty_cache()    #########

                    print('tile error: {} | tile FPR: {} | tile FNR: {}\n'.format(*metrics_t))

                    fconv = open(os.path.join(output_path, '{}-tile-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{}\n'.format(epoch, *metrics_t))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_t)

                # --------------------------
                # --------------------------
                # --------------------------
                if epoch >= args.epochs:
                    save_model(epoch, model, optimizer, scheduler, output_path)

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path)
                print("\nTraining interrupted at epoch {}. Model saved in \'{}\'.".format(epoch, output_path))
                sys.exit(0)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def save_model(epoch, model, optimizer, scheduler, output_path, prefix='pt2'):
    """Save model as a .pth file. """
    # save params of resnet encoder, image head and tile head only
    state_dict = OrderedDict({k: v for k, v in model.state_dict().items()
                              if k.startswith(model.encoder_prefix +
                                              model.image_module_prefix +
                                              model.tile_module_prefix)})
    obj = {
        'mode': 'tile',
        'epoch': epoch,
        'state_dict': state_dict,
        'encoder': model.encoder_name,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(obj, os.path.join(output_path, '{}_{}epochs.pth'.format(prefix, epoch)))


def add_scalar_loss(writer, epoch, loss):
    writer.add_scalar("tile loss", loss, epoch)


def add_scalar_metrics(writer, epoch, metrics):
    metrics = list(metrics)

    assert len(metrics) == 3, "Tile metrics should include 3 items: error rate, FPR and FNR. "
    writer.add_scalar('tile error rate', metrics[0], epoch)
    writer.add_scalar('tile false positive rate', metrics[1], epoch)
    writer.add_scalar('tile false negative rate', metrics[2], epoch)


if __name__ == "__main__":

    print("Training settings: ")
    print("Training Mode: {} | Device: {} | Model: {} | {} epoch(s) in total\n"
          "{} | Initial LR: {} | Output directory: {}"
          .format('tile + image (pt.2)', 'GPU' if torch.cuda.is_available() else 'CPU',
                  args.resume if args.resume else args.model, args.epochs, 'Validate every {} epoch(s)'
                  .format(args.test_every) if args.test_every <= args.epochs else 'No validation',
                  args.lr, args.output))
    print("Tile batch size: {} | Tile size: {} | Stride: {} | Negative top-k: {}"
          .format(args.tile_batch_size, args.tile_size, args.interval, args.topk_neg))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    training_data_path = config.get("data", "data_path")

    # data loading
    kfold = None if args.test_every > args.epochs else 10
    trainset = LystoDataset(os.path.join(training_data_path, "training.h5"), tile_size=args.tile_size, interval=args.interval, kfold=kfold,
                            num_of_imgs=100 if args.debug else 0)
    valset = LystoDataset(os.path.join(training_data_path, "training.h5"), tile_size=args.tile_size, interval=args.interval, train=False,
                          kfold=kfold, num_of_imgs=100 if args.debug else 0)

    # TODO: how can I split the training step for distributed parallel training?
    trainset.setmode(1)
    train_sampler = DistributedSampler(trainset) if dist.is_nccl_available() and args.distributed else None
    val_sampler = DistributedSampler(valset) if dist.is_nccl_available() and args.distributed else None
    train_loader = DataLoader(trainset, batch_size=args.tile_batch_size, shuffle=True,
                              num_workers=args.workers, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.tile_batch_size, shuffle=False, num_workers=args.workers,
                            sampler=val_sampler, pin_memory=True)

    # model setup
    def to_device(model, device):
        if dist.is_nccl_available() and args.distributed:
            print('\nNCCL is available. Setup distributed parallel training with {} devices...\n'
                  .format(torch.cuda.device_count()))
            dist.init_process_group(backend='nccl', world_size=1)
            device = torch.device("cuda", args.local_rank)
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
        else:
            model.to(device)
        return model

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    if args.resume:
        cp = torch.load(args.resume, map_location=device)
        model = nets[cp['encoder']]
        model = to_device(model, device)
        # load params of resnet encoder, tile head and image head only
        model.load_state_dict(
            OrderedDict({k: v for k, v in cp['state_dict'].items()
                         if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                         model.image_module_prefix)}),
            strict=False)
        model.load_state_dict(cp['state_dict'], strict=False)
        last_epoch = cp['epoch']
        last_epoch_for_scheduler = cp['scheduler']['last_epoch'] if cp['scheduler'] is not None else -1
    elif args.scratch:
        model = nets['resnet50']
        model = to_device(model, device)
        last_epoch = 0
        last_epoch_for_scheduler = -1
    else:   # True
        f = torch.load(args.model, map_location=device)
        model = nets[f['encoder']]
        model = to_device(model, device)
        # load params of resnet encoder and image head only
        model.load_state_dict(
            OrderedDict({k: v for k, v in f['state_dict'].items()
                         if k.startswith(model.encoder_prefix + model.image_module_prefix)}),
            strict=False)
        last_epoch = 0
        last_epoch_for_scheduler = -1
    model.setmode("tile")
    if args.scratch:
        model.set_encoder_grads(True)

    crit_cls = nn.CrossEntropyLoss()

    # optimization settings
    optimizer_params = {'params': filter(lambda m: m.requires_grad, model.parameters()),
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
            'max_lr': args.lr,
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

    optimizer = optimizers['SGD'] if args.scheduler is not None else optimizers['Adam']     # adam
    scheduler = schedulers[args.scheduler](optimizer,
                                           last_epoch=last_epoch_for_scheduler,
                                           **scheduler_kwargs[args.scheduler]) \
        if args.scheduler is not None else None
    if args.resume:
        optimizer.load_state_dict(cp['optimizer'])
        if cp['scheduler'] is not None and scheduler is not None:
            scheduler.load_state_dict(cp['scheduler'])

    train(total_epochs=args.epochs,
          last_epoch=last_epoch,
          test_every=args.test_every,
          model=model,
          device=device,
          crit_cls=crit_cls,
          optimizer=optimizer,
          scheduler=scheduler,
          threshold=args.threshold,
          tiles_per_pos=args.tiles_per_pos,
          topk_neg=args.topk_neg,
          output_path=args.output)