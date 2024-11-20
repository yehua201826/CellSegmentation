import random
import numpy as np
from PIL import Image
# from scipy import misc
import imageio
import torch

# use
def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img

# use
def random_scaling(image, label=None, scale_range=None):
    min_ratio, max_ratio = scale_range
    assert min_ratio <= max_ratio

    ratio = random.uniform(min_ratio, max_ratio)  # 0.5-2.0

    return _img_rescaling(image, label, scale=ratio)

# use
def _img_rescaling(image, label=None, scale=None):
    # scale = random.uniform(scales)
    h, w, _ = image.shape

    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    if label is None:
        return new_image

    new_label = Image.fromarray(label).resize(new_scale, resample=Image.NEAREST)
    new_label = np.asarray(new_label)

    return new_image, new_label


# use
def random_fliplr(image, label=None):
    p = random.random()

    if label is None:
        if p > 0.5:
            image = np.fliplr(image)
        return image
    else:
        if p > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        return image, label


# use
def random_crop(image, label=None, crop_size=None, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = image.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = image

    def get_random_cropbox(_label, cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)  # =0
            H_end = H_start + crop_size  # 512
            W_start = random.randrange(0, W - crop_size + 1, 1)  # =0
            W_end = W_start + crop_size  # 512

            if _label is None:
                return H_start, H_end, W_start, W_end,

            temp_label = _label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]

            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox(label)  # 0，512，0，512

    image = pad_image[H_start:H_end, W_start:W_end, :]

    img_H_start = max(H_pad - H_start, 0)  # H_pad-H_start
    img_W_start = max(W_pad - W_start, 0)  # W_pad-W_start
    img_H_end = min(H_end, H_pad + h)  # 512
    img_W_end = min(W_end, W_pad + w)  # 512
    img_box = np.asarray([img_H_start, img_H_end, img_W_start, img_W_end], dtype=np.int16)

    if label is None:
        return image, img_box

    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label
    label = pad_label[H_start:H_end, W_start:W_end]

    return image, label, img_box

def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0

# def img_transforms(image):
#     img_box = None
#     '''
#     if self.resize_range:
#         image, label = transforms.random_resize(
#             image, label, size_range=self.resize_range)
#     '''
#     rescale_range = [0.5, 2.0]
#     image = transforms.random_scaling(
#         image,
#         scale_range=rescale_range)
#
#     image = transforms.random_fliplr(image)
#
#     crop_size = 512
#     image, img_box = transforms.random_crop(
#         image,
#         crop_size=crop_size,
#         mean_rgb=[0, 0, 0],  # [123.675, 116.28, 103.53],
#         ignore_index=255)
#
#     '''
#     if self.stage != "train":
#         image = transforms.img_resize_short(image, min_size=min(self.resize_range))
#     '''
#     image = transforms.normalize_img(image)
#     # to chw
#     image = np.transpose(image, (2, 0, 1))
#
#     return image, img_box