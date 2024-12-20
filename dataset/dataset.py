import collections
import copy
import os
import sys
import re
import csv
from tqdm import tqdm

import h5py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
from skimage import io
from openslide import OpenSlide

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from utils import sort_files

Image.MAX_IMAGE_PIXELS = None
patch_size = np.array([299, 299])


class LystoDataset(Dataset):

    def __init__(self, filepath=None, tile_size=None, interval=None, train=True, organ=None,
                 augment=False, kfold=10, shuffle=False, num_of_imgs=0, _ensemble_init=False):
        """
        :param filepath:    path of hdf5 data file
        :param tile_size:   the length of the side of a certain superpixel
        :param interval:    stride of sliding window
        :param train:       training set or development set
        :param organ:       if categorise data by organ
        :param augment:     if use data augmentation
        :param kfold:       k of k-fold cross-validation, 10 by default
        :param num_of_imgs: first n images are used only, set this to 0 to run with the whole data (for debugging)
        """

        super(LystoDataset, self).__init__()

        if not _ensemble_init:
            if os.path.exists(filepath):
                f = h5py.File(filepath, 'r')
            else:
                raise FileNotFoundError("Invalid data directory.")

            if kfold is not None and kfold <= 0:
                raise Exception("Invalid k-fold cross-validation argument.")
            else:
                self.kfold = kfold

        self.train = train
        self.organ = organ
        self.organs = []            # organ type of image. list ( 20000 )
        self.images = []            # array ( 20000 * 299 * 299 * 3 )
        self.labels = []            # class of image (pos / neg). list ( 20000 )
        self.cls_labels = []        # 7-classification of image. list ( 20000 )
        self.transformIDX = []      # method of augmentation. list ( 20000 )
        self.tileIDX = []           # image index for each tile. list ( 20000 * n )
        self.tiles_grid = []        # upper left coords of tiles on the image. list ( 20000 * n * 2 )
        self.interval = interval
        self.tile_size = tile_size

        self.augment = augment
        self.augment_transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1)
            ])
        ]
        self.transform = [transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])] + [transforms.Compose([
            transforms.ToTensor(),
            aug,
            # transforms.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.4,
            #     hue=0.05,
            # ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) for aug in self.augment_transforms]

        self.image_size = patch_size
        self.mode = None

        if not _ensemble_init:
            tileIDX = -1
            for i, (org, img, label) in enumerate(tqdm(zip(f['organ'], f['x'], f['y']), desc="loading images")):
                org = org.decode('utf-8')

                if num_of_imgs != 0 and i == num_of_imgs:
                    break

                if self.kfold is not None:
                    if (self.train and (i + 1) % self.kfold == 0) or (not self.train and (i + 1) % self.kfold != 0):
                        continue

                if self.organ is None or self.organ == org.partition('_')[0]:
                    tileIDX += 1
                    self.add_data(org, img, label, tileidx=tileIDX)
                    # # augmentation for images with more cells
                    if self.train and self.augment: # and categorize(label) >= 3:
                        for j in range(1, 4):
                            self.add_data(org, img, label, transidx=j)

            assert len(self.labels) == len(self.images), "Mismatched number of labels and images."
            if shuffle:
                idxes = np.random.choice(len(self.images), len(self.images), replace=False)
                self.organs = list(np.asarray(self.organs)[idxes])
                self.images = list(np.asarray(self.images)[idxes])
                self.labels = list(np.asarray(self.labels)[idxes])
                self.cls_labels = list(np.asarray(self.cls_labels)[idxes])
                self.transformIDX = list(np.asarray(self.transformIDX)[idxes])

    def add_data(self, organ, img, label, transidx=0, tileidx=None):

        assert transidx <= len(self.augment_transforms), "Not enough transformations for image augmentation. "

        self.organs.append(organ)
        self.images.append(img)
        self.labels.append(label)
        cls_label = categorize(label)
        self.cls_labels.append(cls_label)
        self.transformIDX.append(transidx)

        if self.interval is not None and self.tile_size is not None and tileidx:
            t = get_tiles(img, self.interval, self.tile_size)
            self.tiles_grid.extend(t)  # 获取 tiles
            self.tileIDX.extend([tileidx] * len(t))  # 每个 tile 对应的 image 标签

        return cls_label

    def random_delete(self, num):

        idxes = np.sort(np.random.choice(len(self.images), num, replace=False))
        for i in reversed(list(idxes)):
            del self.organs[i], self.images[i], self.labels[i], self.cls_labels[i], self.transformIDX[i]

    def setmode(self, mode):
        """
        mode 1: instance inference mode, for top-k sampling -> tile (sampled from images), label
        mode 2: alternative training mode (alternate tile training + image training per batch iteration)
                -> (image, tiles sampled from which), (class, number, binary tile labels)
        mode 3: tile-only training mode -> tile (from top-k training data), label
        mode 4: image validating mode -> 3d image, class, number
        mode 5: image-only training mode -> 4d image, class, number
        """
        self.mode = mode

    def make_train_data(self, idxs, pos_neg_ratio: float = None):
        # set tile label to 1 if counts of the coordinated image > 0 else 0
        self.train_data = np.array([(self.tileIDX[i], self.tiles_grid[i],
                                     0 if self.labels[self.tileIDX[i]] == 0 else 1) for i in idxs])

        pos = 0
        for _, _, label in self.train_data:
            pos += label
        neg = len(self.train_data) - pos

        # fix pos-neg ratio
        np.random.shuffle(self.train_data)
        if pos_neg_ratio is not None:
            if pos > int(neg * pos_neg_ratio):
                flag = 1
                n = pos - int(neg * pos_neg_ratio)
                pos = int(neg * pos_neg_ratio)
                print('Note: Positive superpixels are pruned to meet the pos_neg_ratio. ')
            elif neg > int(pos / pos_neg_ratio):
                flag = 0
                n = neg - int(pos / pos_neg_ratio)
                neg = int(pos / pos_neg_ratio)
                print('Note: Negative superpixels are pruned to meet the pos_neg_ratio. ')
            else:
                return pos, neg

            excess = []
            for i, (_, _, label) in enumerate(self.train_data):
                if label == flag:
                    excess.append(i)
                if len(excess) == n:
                    break

            self.train_data = np.delete(self.train_data, excess, 0)

        return pos, neg

    def __getitem__(self, idx):

        # top-k tile sampling mode
        if self.mode == 1:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile inference. "

            (x, y) = self.tiles_grid[idx]
            tile = self.images[self.tileIDX[idx]][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform[self.transformIDX[self.tileIDX[idx]]](tile)

            label = self.labels[self.tileIDX[idx]]
            return tile, label

        # alternative training mode
        elif self.mode == 2:
            assert len(self.tiles_grid) > 0, \
                "Dataset tile size and interval have to be settled for alternative training. "

            # Get images
            image = self.images[idx]
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            # Get tiles
            tiles = []
            tile_grids = []
            tile_labels = []
            for tileIDX, (x, y), label in self.train_data:
                if tileIDX == idx:
                    tile = self.images[tileIDX][x:x + self.tile_size, y:y + self.tile_size]
                    tiles.append(tile)
                    tile_grids.append((x, x + self.tile_size, y, y + self.tile_size))
                    tile_labels.append(label)

            tiles = [self.transform[self.transformIDX[self.tileIDX[idx]]](tile) for tile in tiles]
            image = self.transform[self.transformIDX[idx]](image)

            return (image.unsqueeze(0), torch.stack(tiles, dim=0)), \
                   (label_cls, label_reg, torch.tensor(tile_labels))

        # tile-only training mode
        elif self.mode == 3:
            assert len(
                self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile-mode training. "

            tileIDX, (x, y), label = self.train_data[idx]
            tile = self.images[tileIDX][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform[self.transformIDX[self.tileIDX[idx]]](tile)   # tileIDX
            return tile, label

        # image validating mode
        elif self.mode == 4:
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            image = self.transform[self.transformIDX[idx]](image)
            return image, label_cls, label_reg

        # image-only training mode
        elif self.mode == 5:
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            image = self.transform[self.transformIDX[idx]](image)
            # for image-only training, images need to be unsqueezed
            return image.unsqueeze(0), label_cls, label_reg

        else:
            raise Exception("Something wrong in setmode.")

    def __len__(self):

        assert self.mode is not None, "Something wrong in setmode."

        if self.mode == 1:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "
            return len(self.tileIDX)
        elif self.mode == 2:
            return len(self.images)
        elif self.mode == 3:
            return len(self.train_data)
        else:
            return len(self.labels)


class EnsembleSet:

    def __init__(self, filepath=None, augment=False, k: int = 10):

        self.data = LystoDataset(filepath, kfold=None, augment=augment)
        self.training_set_frame = LystoDataset(kfold=None, _ensemble_init=True)
        self.validating_set_frame = LystoDataset(kfold=None, _ensemble_init=True)

        self.k = k
        self.training_idxes = []
        self.validating_idxes = []
        self.training_set = None
        self.validating_set = None

        size, extra = divmod(len(self.data.labels), self.k)
        split_size = np.full(self.k, size)
        split_size[:extra] += 1
        for i in range(1, len(split_size)):
            split_size[i] += split_size[i - 1]
        split_size = [0] + list(split_size)

        for i in range(self.k):
            self.training_idxes.append([idx for idx in range(split_size[i])] +
                                       [idx for idx in range(split_size[i + 1], split_size[-1])])
            self.validating_idxes.append([idx for idx in range(split_size[i], split_size[i + 1])])

    def get_loader(self, train, idx, **kwargs):
        if train:
            self.training_set = copy.deepcopy(self.training_set_frame)
            for i in self.training_idxes[idx]:
                self.training_set.images.append(self.data.images[i])
                self.training_set.labels.append(self.data.labels[i])
                self.training_set.organs.append(self.data.organs[i])
                self.training_set.cls_labels.append(self.data.cls_labels[i])
                self.training_set.transformIDX.append(self.data.transformIDX[i])
            return DataLoader(self.training_set, shuffle=True, pin_memory=True, **kwargs)

        else:
            self.validating_set = copy.deepcopy(self.validating_set_frame)
            for i in self.validating_idxes[idx]:
                self.validating_set.images.append(self.data.images[i])
                self.validating_set.labels.append(self.data.labels[i])
                self.validating_set.organs.append(self.data.organs[i])
                self.validating_set.cls_labels.append(self.data.cls_labels[i])
                self.validating_set.transformIDX.append(self.data.transformIDX[i])
            return DataLoader(self.validating_set, shuffle=False, pin_memory=True, **kwargs)

    def setmode(self, train, mode):
        if train:
            self.training_set_frame.setmode(mode)
        else:
            self.validating_set_frame.setmode(mode)


class LystoTestset(Dataset):

    def __init__(self, filepath, tile_size=None, interval=None, organ=None, num_of_imgs=0):
        """
        :param filepath:    path of hdf5 data file
        :param interval:    stride of sliding window
        :param tile_size:   the length of the side of a certain superpixel
        :param num_of_imgs: first n images are used only, set this to 0 to run with the whole data (for debugging)
        """

        super(LystoTestset, self).__init__()

        if os.path.exists(filepath):
            f = h5py.File(filepath, 'r')
        else:
            raise FileNotFoundError("Invalid data directory.")

        self.organ = organ
        self.id = []
        self.organs = []  # organ type of image. list ( 20000 )
        self.images = []  # list ( 20000 * 299 * 299 * 3 )
        self.tileIDX = []  # image index for each tile. list ( 20000 * n )
        self.tiles_grid = []  # upper left coords of tiles on the image. list ( 20000 * n * 2 )
        self.interval = interval
        self.tile_size = tile_size

        tileIDX = -1
        for i, (org, img) in enumerate(tqdm(zip(f['organ'], f['x']), desc="loading images")):
            org = org.decode('utf-8')

            if num_of_imgs != 0 and i == num_of_imgs:
                break

            if self.organ is None or self.organ == org.partition('_')[0]:
                tileIDX += 1
                self.id.append(i)
                self.organs.append(org)
                self.images.append(img)

            if self.interval is not None and self.tile_size is not None:
                t = get_tiles(img, self.interval, self.tile_size)
                self.tiles_grid.extend(t)
                self.tileIDX.extend([tileIDX] * len(t))

        self.image_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.mode = None

    def setmode(self, mode):
        """
        mode "tile":  instance mode, used in pseudo-mask heatmap generation -> tile (sampled from images)
        mode "image": image assessment mode, used in cell counting -> image
        """
        self.mode = mode

    def __getitem__(self, idx):
        # test_tile
        if self.mode == "tile":
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "

            (x, y) = self.tiles_grid[idx]
            tile = self.images[self.tileIDX[idx]][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform(tile)

            return tile

        # test_count
        elif self.mode == "image":
            image = self.images[idx]
            image = self.transform(image)

            return self.id[idx], image

        else:
            raise Exception("Something wrong in setmode.")

    def __len__(self):
        if self.mode == "tile":
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "
            return len(self.tileIDX)
        elif self.mode == "image":
            return len(self.images)
        else:
            raise Exception("Something wrong in setmode.")


class Maskset(Dataset):

    def __init__(self, filepath, mask_data, augment=False, num_of_imgs=0, train=True):

        super(Maskset, self).__init__()
        assert type(mask_data) in [np.ndarray, str], "Invalid data type. "

        self.filepath = filepath
        if self.filepath:
            f = h5py.File(self.filepath, 'r')
        else:
            raise FileNotFoundError("Invalid data file.")

        self.organs = []
        self.images = []
        self.masks = []
        self.labels = []

        for i, (org, img, label) in enumerate(tqdm(zip(f['organ'], f['x'], f['y']), desc="loading images")):
            org = org.decode('utf-8')
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            if (train and (i + 1) % 10 == 0) or (not train and (i + 1) % 10 != 0):
                continue

            self.organs.append(org)
            self.images.append(img)
            self.labels.append(label)

        if isinstance(mask_data, str):

            for i, file in enumerate(sorted(os.listdir(os.path.join(mask_data, 'mask_145_o_3')))):
                if num_of_imgs != 0 and len(self.masks) == len(self.images):
                    break

                if (train and (i + 1) % 10 == 0) or (not train and (i + 1) % 10 != 0):
                    continue

                img = io.imread(os.path.join(mask_data, 'mask_145_o_3', file))
                self.masks.append(img)

        else:
            self.masks = [torch.from_numpy(np.uint8(md)) for i, md in enumerate(mask_data) if (train and (i + 1) % 10 != 0) or (not train and (i + 1) % 10 == 0)]
            if num_of_imgs != 0:
                self.masks = self.masks[:num_of_imgs]

        assert len(self.masks) == len(self.images), "Mismatched number of masks and RGB images."

        # for i in range(len(self.images)):
        #     io.imsave("ts/rgb_{}.png".format(i + 1), np.uint8(self.images[i]))
        #     io.imsave("ts/mask_{}.png".format(i + 1), np.uint8(self.masks[i]))

        if augment:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.3,
                    saturation=0.4,
                    hue=0.05,
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):

        image = self.transform(self.images[idx])
        mask = self.masks[idx]
        label = self.labels[idx]

        return image, mask, label

    def __len__(self):
        return len(self.images)


class MaskTestset(Dataset):

    def __init__(self, filepath, num_of_imgs=0, resume_from=None):

        super(MaskTestset, self).__init__()

        self.filepath = filepath
        self.patch_size = patch_size

        if os.path.isdir(self.filepath):
            self.images_grid = []   # list ( n * 2 )
            self.imageIDX = []      # list ( n )
            self.image_size = []    # list ( ? )

            self.files = [f for f in sorted(os.listdir(self.filepath))
                          if os.path.isfile(os.path.join(self.filepath, f))]
            if resume_from is not None:
                self.files[:self.files.index(resume_from)] = []
            for i, file in enumerate(tqdm(self.files, desc="loading images")):
                if num_of_imgs != 0 and i == num_of_imgs:
                    break
                if file.endswith((".svs", ".tiff")):
                    self.mode = "WSI"
                    slide = OpenSlide(os.path.join(self.filepath, file))
                    patches_grid = self.sample_patches(slide.dimensions, self.patch_size - 16)
                    self.images_grid.extend(patches_grid)
                    self.imageIDX.extend([i] * len(patches_grid))
                    self.image_size.append(slide.dimensions)
                    slide.close()
                elif file.endswith((".jpg", ".png")):
                    self.mode = "ROI"
                    img = io.imread(os.path.join(self.filepath, file)).astype(np.uint8)
                    patches_grid = self.sample_patches(img.shape, self.patch_size - 16)
                    self.images_grid.extend(patches_grid)
                    self.imageIDX.extend([i] * len(patches_grid))
                    self.image_size.append(img.shape)
                else:
                    raise FileNotFoundError("Invalid data directory.")

        elif (os.path.exists(self.filepath)
              and os.path.isfile(self.filepath)
              and self.filepath.endswith(("h5", "hdf5"))):
            self.mode = "patch"
            self.images = []
            f = h5py.File(self.filepath, 'r')
            for i, img in enumerate(f['x']):
                if num_of_imgs != 0 and i == num_of_imgs:
                    break
                self.images.append(img)

        else:
            raise FileNotFoundError("Invalid data directory.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def sample_patches(self, size, interval):

        patches_grid = []
        xborder = size[0] - self.patch_size[0]
        yborder = size[1] - self.patch_size[1]

        if self.mode == "WSI":
            for x in np.arange(0, xborder + 1, interval[0]):
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((x, y))
                if patches_grid[-1][1] != yborder:
                    patches_grid.append((x, yborder))

            if patches_grid[-1][0] != xborder:
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((xborder, y))

                if patches_grid[-1][1] != yborder:
                    patches_grid.append((xborder, yborder))

        elif self.mode == "ROI":
            for x in np.arange(0, xborder + 1, interval[0]):
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((x, y))
                if patches_grid[-1][1] != yborder:
                    patches_grid.append((x, yborder))

            if patches_grid[-1][0] != xborder:
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((xborder, y))

                if patches_grid[-1][1] != yborder:
                    patches_grid.append((xborder, yborder))
        else:
            raise TypeError("Invalid image type.")
        return patches_grid

    def get_a_patch(self, idx):

        if self.mode == "WSI":
            image_file = os.path.join(self.filepath, self.files[self.imageIDX[idx]])
            slide = OpenSlide(image_file)
            x, y = self.images_grid[idx]
            patch = np.asarray(slide.read_region((x, y), level=0, size=tuple(self.patch_size)).convert('RGB'))
            slide.close()
            return patch, self.imageIDX[idx]

        elif self.mode == "ROI":
            image_file = os.path.join(self.filepath, self.files[self.imageIDX[idx]])
            image = io.imread(image_file).astype(np.uint8)
            x, y = self.images_grid[idx]
            patch = image[x:x + self.patch_size[0], y:y + self.patch_size[1]]
            return patch, self.imageIDX[idx]

        else:
            patch = self.images[idx]
            return [patch]


    def __getitem__(self, idx):

        patch = self.get_a_patch(idx)[0]
        patch = self.transform(patch)
        return patch

    def __len__(self):

        if self.mode == "patch":
            return len(self.images)
        else:
            return len(self.images_grid)


class PointTestset(Dataset):
    def __init__(self, data_path="data/qupath/lysto", filename_pattern="(?<=test_)\d*", num_of_imgs=0):

        """
        filename_pattern: regex pattern used to extract numbers from file name strings
        """

        super(PointTestset, self).__init__()

        self.image_path = os.path.join(data_path, "images")
        self.mask_path = os.path.join(data_path, "masks")
        self.point_path = os.path.join(data_path, "points")
        self.names = []
        self.images = []
        self.masks = []
        self.points = []
        self.area_types = []
        self.cancer_types = []

        for i, file in enumerate(sort_files(os.listdir(self.image_path), filename_pattern)):
            if num_of_imgs != 0 and i == num_of_imgs:
                break
            self.names.append(file)
            self.images.append(io.imread(os.path.join(self.image_path, file)).astype(np.uint8))
        for file in sort_files(os.listdir(self.mask_path), filename_pattern):
            if num_of_imgs != 0 and len(self.masks) == len(self.images):
                break
            mask = io.imread(os.path.join(self.mask_path, file)).astype(np.uint8)
            self.masks.append(mask[..., 0] if mask.ndim > 1 else mask)
        for file in sort_files(os.listdir(self.point_path), filename_pattern):
            if num_of_imgs != 0 and len(self.points) == len(self.images):
                break
            coord_data = pd.read_csv(os.path.join(self.point_path, file), sep='\t', usecols=[0, 1])
            self.points.append(np.round(coord_data.values).astype(int))
        w = csv.reader(open(os.path.join(data_path, 'image_type.csv'), 'r'), delimiter=',')
        for line in w:
            if line[0] in os.listdir(self.image_path):
                self.cancer_types.append(re.findall('[A-Za-z]*', line[1])[0])
                self.area_types.append(re.findall('[A-Za-z]*', line[2])[0])

        assert len(self.masks) == len(self.images), "Mismatched number of masks and RGB images."

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def get_image(self, idx):
        return self.images[idx]

    def __getitem__(self, idx):

        image = self.transform(self.images[idx])
        mask = self.masks[idx]
        points = self.points[idx]
        area_type = self.area_types[idx]
        cancer_type = self.cancer_types[idx]
        # count = self.points[idx].shape[0]

        # return image, mask, points, count
        return image, mask, points, cancer_type, area_type

    def __len__(self):
        return len(self.images)


def get_tiles(image, interval, size):
    """
    get tiles from an image by sliding window.
    :param image:       input image matrix, 299 x 299 x 3
    :param interval:    stride of sliding window
    :param size:        the length of the side of a certain tile
    :return:            list of upper left coords of tiles
    """

    tiles = []
    for x in np.arange(0, image.shape[0] - size + 1, interval):
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            tiles.append((x, y))

        if tiles[-1][1] + size != image.shape[1]:
            tiles.append((x, image.shape[1] - size))

    if tiles[-1][0] + size != image.shape[0]:
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            tiles.append((image.shape[0] - size, y))

        if tiles[-1][1] + size != image.shape[1]:
            tiles.append((image.shape[0] - size, image.shape[1] - size))

    return tiles


def categorize(x):
    """Split counts into 7 classes. Same as LYSTO scoring criterion. """
    if x == 0:
        label = 0
    elif x <= 5:
        label = 1
    elif x <= 10:
        label = 2
    elif x <= 20:
        label = 3
    elif x <= 50:
        label = 4
    elif x <= 200:
        label = 5
    else:
        label = 6
    return label


def de_categorize(label):
    """Transform 7 classes into count ranges. """
    if label == 0:
        xmin, xmax = 0, 0
    elif label == 1:
        xmin, xmax = 1, 5
    elif label == 2:
        xmin, xmax = 6, 10
    elif label == 3:
        xmin, xmax = 11, 20
    elif label == 4:
        xmin, xmax = 21, 50
    elif label == 5:
        xmin, xmax = 51, 200
    else:
        xmin, xmax = 201, 100000
    return xmin, xmax


if __name__ == '__main__':

    batch_size = 2
    imageSet = LystoDataset("data/training.h5", tile_size=32, interval=150, num_of_imgs=51)
    imageSet_val = LystoDataset("data/training.h5", train=False, tile_size=32, interval=150, num_of_imgs=51)
    train_loader = DataLoader(imageSet, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(imageSet_val, batch_size=batch_size, shuffle=False)

    imageSet.setmode(1)
    imageSet_val.setmode(1)
    for idx, data in enumerate(train_loader):
        print('Dry run : [{}/{}]\r'.format(idx + 1, len(train_loader)))
    print("Length of dataset: {}".format(len(train_loader.dataset)))
    for idx, data in enumerate(val_loader):
        print('Dry run : [{}/{}]\r'.format(idx + 1, len(val_loader)))
    print("Length of dataset: {}".format(len(val_loader.dataset)))

    # Output the very first image
    print("The first training image: ")
    plt.imshow(imageSet.images[0])
    plt.show()
    print("Slide Patch: {0}\nLabel: {1}".format(imageSet.organs[0], imageSet.labels[0]))
    print("Grids of tiles: {}".format(imageSet.tiles_grid[0]))

    print("The first validation image: ")
    plt.imshow(imageSet_val.images[0])
    plt.show()
    print("Slide Patch: {0}\nLabel: {1}".format(imageSet_val.organs[0], imageSet_val.labels[0]))
    print("Grids of tiles: {}".format(imageSet_val.tiles_grid[0]))
