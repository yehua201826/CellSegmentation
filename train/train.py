from tqdm import tqdm
import numpy as np

import torch
from torch.nn import CrossEntropyLoss as CELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import *

from .losses import DiceLoss


def train_tile(loader, epoch, total_epochs, model, device, criterion, optimizer, scheduler, gamma):
    """Tile training for one epoch.

    :param loader:          DataLoader of training set
    :param epoch:           current number of epochs
    :param total_epochs:    total number of epochs
    :param model:           nn.Module
    :param criterion:       loss of classification (criterion_cls)
    :param optimizer:       gradient optimizer of model training
    """

    # tile training, dataset.mode = 3
    model.train()

    tile_num = 0
    train_loss = 0.
    train_bar = tqdm(loader, desc="tile training")
    for i, (data, label) in enumerate(train_bar):
        train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))

        optimizer.zero_grad()
        output = model(data.to(device), freeze_bn=True)
        loss = criterion(output, label.to(device)) * gamma  # no need of softmax() with pytorch CrossEntropy

        loss.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        tile_num += data.size(0)
        train_loss += loss.item() * data.size(0)

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    train_loss /= tile_num
    return train_loss


def train_image(loader, epoch, total_epochs, model, device, crit_cls, crit_reg, optimizer, scheduler, alpha, beta):
    """tile + image training for one epoch. image mode = image_cls + image_reg + image_seg

    :param loader:          DataLoader of training set
    :param epoch:           current number of epochs
    :param total_epochs:    total number of epochs
    :param model:           nn.Module
    :param crit_cls:        loss of classification
    :param crit_reg:        loss of regression
    :param optimizer:       gradient optimizer of model training
    :param scheduler:       learning rate scheduler
    :param alpha:           image_cls_loss ratio
    :param beta:            image_reg_loss ratio
    """

    # image training, dataset.mode = 5
    model.train()

    image_cls_loss = 0.
    image_reg_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, desc="image training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (data, label_cls, label_num) in enumerate(train_bar):

        optimizer.zero_grad()
        output = model(data.to(device))

        image_cls_loss_i = crit_cls(output[0], label_cls.to(device))
        image_reg_loss_i = crit_reg(output[1].squeeze(), label_num.to(device, dtype=torch.float32))

        image_loss_i = alpha * image_cls_loss_i + beta * image_reg_loss_i
        # image_loss_i = image_reg_loss_i
        image_loss_i.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_cls_loss += image_cls_loss_i.item() * data.size(0)
        image_reg_loss += image_reg_loss_i.item() * data.size(0)
        image_loss += image_loss_i.item() * data.size(0)

        # print("image data size:", data.size(0))

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    # print("Total images:", len(loader.dataset))

    image_loss /= len(loader.dataset)
    image_cls_loss /= len(loader.dataset)
    image_reg_loss /= len(loader.dataset)

    return image_cls_loss, image_reg_loss, image_loss
    # return 0, image_reg_loss, image_loss


def train_image_cls(loader, epoch, total_epochs, model, device, crit_cls, optimizer, scheduler):
    # image training, dataset.mode = 5
    model.train()

    image_cls_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, desc="image training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (data, label_cls, label_num) in enumerate(train_bar):

        optimizer.zero_grad()
        output = model(data.to(device))

        image_cls_loss_i = crit_cls(output[0], label_cls.to(device))
        image_cls_loss_i.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_cls_loss += image_cls_loss_i.item() * data.size(0)

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    image_loss /= len(loader.dataset)
    image_cls_loss /= len(loader.dataset)

    return image_cls_loss


def train_image_reg(loader, epoch, total_epochs, model, device, crit_reg, optimizer, scheduler):
    # image training, dataset.mode = 5
    model.train()

    image_reg_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, desc="image training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (data, label_cls, label_num) in enumerate(train_bar):

        optimizer.zero_grad()
        output = model(data.to(device))

        image_reg_loss_i = crit_reg(output[1].squeeze(), label_num.to(device, dtype=torch.float32))
        image_reg_loss_i.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_reg_loss += image_reg_loss_i.item() * data.size(0)
        image_loss += image_reg_loss_i.item() * data.size(0)

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    image_loss /= len(loader.dataset)
    image_reg_loss /= len(loader.dataset)

    return image_reg_loss


def train_seg(loader, epoch, total_epochs, model, device, optimizer, scheduler):
    # segmentation training
    model.train()

    image_seg_loss = 0.

    train_bar = tqdm(loader, desc="segmentation training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (image, mask, label) in enumerate(train_bar):

        mask = (mask / 255).to(device, dtype=torch.float32)
        # label = label.to(device)
        optimizer.zero_grad()
        output = model(image.to(device)).to(dtype=torch.float32)
        # output: [n, 2, 299, 299]
        # mask:   [n,    299, 299]
        ce = CELoss()(output, mask.to(dtype=torch.long))
        dice = DiceLoss()(F.softmax(output)[:, 1], mask)
        print(f"ce:{ce}"
              f", dice:{dice}"
             )
        # loss = dice + ce
        # loss = ce
        loss = dice
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_seg_loss += loss.item() * image.size(0)

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    image_seg_loss /= len(loader.dataset)
    return image_seg_loss

def val_seg(loader, epoch, total_epochs, model, device):
    model.eval()
    loss_val = 0.
    val_bar = tqdm(loader, desc="segmentation validating")
    val_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    with torch.no_grad():
        for i, (image, mask, label) in enumerate(val_bar):
            mask = (mask / 255).to(device, dtype=torch.float32)
            output = model(image.to(device)).to(dtype=torch.float32)
            ce = CELoss()(output, mask.to(dtype=torch.long))
            dice = DiceLoss()(F.softmax(output)[:, 1], mask)
            # loss_v = ce + dice
            # loss_v = ce
            loss_v = dice
            loss_val += loss_v.item() * image.size(0)
    loss_val /= len(loader.dataset)

    model.train()

    return loss_val

def train_alternative(loader, epoch, total_epochs, model, device, crit_cls, crit_reg, optimizer,
                      scheduler, threshold, alpha, beta, gamma, delta):
    """tile + image training for one epoch. image mode = image_cls + image_reg + image_seg

    :param loader:          DataLoader of training set
    :param epoch:           current number of epochs
    :param total_epochs:    total number of epochs
    :param model:           nn.Module
    :param crit_cls:        loss of classification
    :param crit_reg:        loss of regression
    :param optimizer:       gradient optimizer of model training
    :param scheduler:       learning rate scheduler
    :param alpha:           tile_loss ratio
    :param beta:            image_cls_loss ratio
    :param gamma:           image_reg_loss ratio
    :param delta:           image_seg_loss ratio
    """

    # alternative training, dataset.mode = 2

    tile_num = 0
    tile_loss = 0.
    image_cls_loss = 0.
    image_reg_loss = 0.
    image_seg_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, desc="alternative training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (data, labels) in enumerate(train_bar):

        # pt.1: tile training
        model.setmode("tile")
        model.train()

        # print("images pack size:", data[0].size())
        # print("tiles pack size:", data[1].size())

        optimizer.zero_grad()
        output = model(data[1].to(device))

        tile_loss_i = gamma * crit_cls(output, labels[2].to(device))
        tile_loss_i.backward()
        optimizer.step()

        tile_loss += tile_loss_i.item() * data[1].size(0)
        tile_num += data[1].size(0)

        # pt.2: image training
        model.setmode("image")
        model.train()
        optimizer.zero_grad()
        output = model(data[0].to(device))

        image_cls_loss_i = crit_cls(output[0], labels[0].to(device))
        image_reg_loss_i = crit_reg(output[1].squeeze(), labels[1].to(device, dtype=torch.float32))
        # image_seg_loss_i = crit_seg(output[?], labels[?].to(device))

        # total_loss_i = gamma * tile_loss_i + alpha * image_cls_loss_i + \
        #                beta * image_reg_loss_i + delta * image_seg_loss_i
        image_loss_i = alpha * image_cls_loss_i + beta * image_reg_loss_i
        image_loss_i.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_cls_loss += image_cls_loss_i.item() * data[0].size(0)
        image_reg_loss += image_reg_loss_i.item() * data[0].size(0)
        # image_seg_loss += image_seg_loss_i.item() * image_data[0].size(0)
        image_loss += image_loss_i.item() * data[0].size(0)

        # print("image data size:", data[0].size(0))
        # print("tile data size:", data[1].size(0))

        # # train seg?
        # if (i + 1) % ((len(loader) + 1) // mini_epochs?) == 0:
        #     train_seg(train_loader_forward, epoch, total_epochs, model, device,
        #     optimizer, scheduler, delta)

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()
    # print("Total tiles:", tile_num)
    # print("Total images:", len(loader.dataset))

    tile_loss /= tile_num
    image_loss /= len(loader.dataset)
    image_cls_loss /= len(loader.dataset)
    image_reg_loss /= len(loader.dataset)
    # image_seg_loss /= len(loader.dataset)
    image_seg_loss = 0.
    return tile_loss, image_cls_loss, image_reg_loss, image_seg_loss, image_loss
