import sys

import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils

class LystoDataset(Dataset):

    def __init__(self, filepath, tile_size=None, interval=None, train=True, augment=True, kfold=10, num_of_imgs=0):
        """
        :param filepath:    hdf5数据文件路径
        :param train:       训练集 / 验证集，默认为训练集
        :param kfold:       k 折交叉验证的参数，数据集每隔 k 份抽取 1 份作为验证集，默认值为 10
        :param interval:    选取 tile 的间隔步长
        :param tile_size:   一个 tile 的边长
        :param num_of_imgs: 调试程序用参数，表示用数据集的前 n 张图片构造数据集，设为 0 使其不起作用
        """

        if filepath:
            f = h5py.File(filepath, 'r')
        else:
            raise Exception("Invalid data file.")

        if kfold is not None and kfold <= 0:
            raise Exception("Invalid k-fold cross-validation argument.")

        self.train = train
        self.kfold = kfold
        # self.visualize = False
        self.organs = []            # 全切片来源，array ( 20000 )
        self.images = []            # array ( 20000 * 299 * 299 * 3 )
        self.labels = []            # 图像中的阳性细胞数目，array ( 20000 )
        self.cls_labels = []        # 按数目把图像分为 7 类，存为类别标签
        self.transformIDX = []      # 数据增强的类别，array (  )
        self.tileIDX = []           # 每个 tile 对应的图像编号，array ( 20000 * n )
        self.tiles_grid = []        # 每张图像中选取的像素 tile 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = interval
        self.tile_size = tile_size

        augment_transforms = [
            transforms.ColorJitter(),
            transforms.Compose([
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(p=1)
            ]),
            transforms.Compose([
                transforms.ColorJitter(),
                transforms.RandomVerticalFlip(p=1)
            ]),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1)
            ]),
            transforms.Compose([
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1)
            ])
        ]
        self.transforms = [transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])] + [transforms.Compose([
                transforms.ToTensor(),
                augment,
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        ]) for augment in augment_transforms]

        def store_data(transidx=0):

            nonlocal organ, img, label, tileIDX, augment_transforms
            assert transidx <= len(augment_transforms), "Not enough transformations for image augmentation. "

            self.organs.append(organ)
            self.images.append(img)
            self.labels.append(label)
            cls_label = categorize(label)
            self.cls_labels.append(cls_label)
            self.transformIDX.append(transidx)

            if self.interval is not None and self.tile_size is not None:
                t = get_tiles(img, self.interval, self.tile_size)
                self.tiles_grid.extend(t)  # 获取 tiles
                self.tileIDX.extend([tileIDX] * len(t))  # 每个 tile 对应的 image 标签

            return cls_label

        tileIDX = -1
        for i, (organ, img, label) in enumerate(zip(f['organ'], f['x'], f['y'])):

            # 调试用代码
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            if self.kfold is not None:
                if (self.train and (i + 1) % self.kfold == 0) or (not self.train and (i + 1) % self.kfold != 0):
                    continue

            tileIDX += 1

            store_data()
            if self.train and augment:
                for i in range(8):
                    store_data(i)

        assert len(self.labels) == len(self.images), "Mismatched number of labels and images."

        self.image_size = self.images[0].shape[0:2]
        self.mode = None

    def setmode(self, mode):
        self.mode = mode

    # def visualize_bboxes(self):
    #     self.visualize = True

    def make_train_data(self, idxs):
        # 制作 tile mode 训练用数据集，当 tile 对应的图像的 label 为 n 时标签为 1 ，否则为 0
        self.train_data = [(self.tileIDX[i], self.tiles_grid[i],
                           0 if self.labels[self.tileIDX[i]] == 0 else 1) for i in idxs]
        # if shuffle:
        #     self.train_data = random.sample(self.train_data, len(self.train_data))

        pos = 0
        for _, _, label in self.train_data:
            pos += label

        # 返回正负样本数目
        return pos, len(self.train_data) - pos

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        # top-k 选取模式 (tile mode)
        if self.mode == 1:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile inference. "

            (x, y) = self.tiles_grid[idx]
            tile = self.images[self.tileIDX[idx]][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transforms[self.transformIDX[self.tileIDX[idx]]](tile)

            label = self.labels[self.tileIDX[idx]]
            return tile, label

        # alternative training mode
        elif self.mode == 2:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for alternative training. "

            # Get images
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            # if self.visualize:
            #     plt.imshow(image)

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

                    # if self.visualize:
                    #     plt.gca().add_patch(
                    #         plt.Rectangle((x, y), x + self.size, y + self.size,
                    #                       fill=False, edgecolor='red' if label == 0 else 'deepskyblue', linewidth=1)
                    #     )

            # # tile visualize testing
            # if self.visualize:
            #     plt.savefig('test/img{}.png'.format(idx))
            #     plt.close()

            # # 画边界框（有问题）
            #     image_tensor = torch.from_numpy(tile.transpose((2, 0, 1))).contiguous()
            #     utils.draw_bounding_boxes(image_tensor, torch.tensor(tile_grids),
            #                               labels=['neg' if lbl == 0 else 'pos' for lbl in tile_labels],
            #                               colors=list(cycle('red')))
            #     utils.save_image(image, "test/img{}.png".format(idx))
            #     print("Image is saved.")

            tiles = [self.transforms[self.transformIDX[self.tileIDX[idx]]](tile) for tile in tiles]
            image = self.transforms[self.transformIDX[idx]](image)

            return (image.unsqueeze(0), torch.stack(tiles, dim=0)), \
                   (label_cls, label_reg, torch.tensor(tile_labels))

        # tile-only training mode
        elif self.mode == 3:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile-mode training. "

            tileIDX, (x, y), label = self.train_data[idx]
            tile = self.images[tileIDX][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transforms[self.transformIDX[self.tileIDX[idx]]](tile)
            return tile, label

        # image validating mode
        elif self.mode == 4:
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            image = self.transforms[self.transformIDX[idx]](image)
            return image, label_cls, label_reg

        # image-only training mode
        elif self.mode == 5:
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            image = self.transforms[self.transformIDX[idx]](image)
            # for image-only training, images need to be unsqueezed
            return image.unsqueeze(0), label_cls, label_reg

        else:
            raise Exception("Something wrong in setmode.")

    def __len__(self):

        if self.mode == 1:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "
            return len(self.tileIDX)
        elif self.mode == 2:
            return len(self.images)
        elif self.mode == 3:
            return len(self.train_data)
        else:
            return len(self.labels)


class LystoTestset(Dataset):

    def __init__(self, filepath, tile_size=None, interval=None, num_of_imgs=0):

        """
        :param filepath:    hdf5数据文件路径
        :param transform:   数据预处理方式
        :param interval:    选取 tile 的间隔步长
        :param tile_size:   一个 tile 的边长
        :param num_of_imgs: 调试程序用参数，表示用数据集的前 n 张图片构造数据集，设为 0 使其不起作用
        """

        if filepath:
            f = h5py.File(filepath, 'r')
        else:
            raise Exception("Invalid data file.")

        self.organs = []            # 全切片来源，array ( 20000 )
        self.images = []            # array ( 20000 * 299 * 299 * 3 )
        self.tileIDX = []           # 每个 tile 对应的图像编号，array ( 20000 * n )
        self.tiles_grid = []        # 每张图像中选取的像素 tile 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = interval
        self.tile_size = tile_size

        tileIDX = -1
        for i, (organ, img) in enumerate(zip(f['organ'], f['x'])):

            # TODO: 调试用代码，实际代码不包含 num_of_imgs 参数及以下两行
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            tileIDX += 1
            self.organs.append(organ)
            self.images.append(img)

            if self.interval is not None and self.tile_size is not None:

                t = get_tiles(img, self.interval, self.tile_size)
                self.tiles_grid.extend(t)
                self.tileIDX.extend([tileIDX] * len(t))

        self.image_size = self.images[0].shape[0:2]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.mode = None

    def setmode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        # test_tile
        if self.mode == "tile":
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "

            # organ = self.organs[idx]
            (x, y) = self.tiles_grid[idx]
            tile = self.images[self.tileIDX[idx]][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform(tile)

            return tile

        # test_count
        elif self.mode == "count":
            image = self.images[idx]
            image = self.transform(image)

            return image

        else:
            raise Exception("Something wrong in setmode.")

    def __len__(self):
        if self.mode == "tile":
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "
            return len(self.tileIDX)
        elif self.mode == "count":
            return len(self.images)
        else:
            raise Exception("Something wrong in setmode.")


class MaskSet(Dataset):

    def __init__(self):
        pass


def get_tiles(image, interval, size):
    """
    在每张图片上生成 tile 实例。
    :param image: 输入图片矩阵，299 x 299 x 3
    :param interval: 取 tile 坐标点的间隔
    :param size: 单个 tile 的大小
    """

    tiles = []
    for x in np.arange(0, image.shape[0] - size + 1, interval):
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            tiles.append((x, y))  # n x 2

    return tiles


def categorize(x):
    """按 LYSTO 划分的 7 个细胞数目类别划分分类标签。"""
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
    """给出每个 label 对应的范围。"""
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

    # 查看第一张图片
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
