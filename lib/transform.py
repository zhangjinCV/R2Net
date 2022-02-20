#!/usr/bin/python3
# coding=utf-8

import cv2
import paddle
import numpy as np
from PIL import ImageEnhance
import random
import paddle


# random.seed(1234)
# np.random.seed(1234)
# paddle.seed(1234)

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask):
        for op in self.ops:
            image, mask = op(image, mask)
        return image, mask


class RandomVorizontalFlip(object):
    def __call__(self, image, mask, body=None, detail=None):
        if np.random.randint(5) == 1:
            image = image[::-1, :, :].copy()
            mask = mask[::-1, :, :].copy()
        #   body = body[::-1, :, :].copy()
        #   detail = detail[::-1, :, :].copy()
        return image.copy(), mask.copy()  # , body.copy(), detail.copy()


class RandomHorizontalFlip(object):
    def __call__(self, image, mask, body=None, detail=None):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        # body = body[:, ::-1, :].copy()
        # detail = detail[:, ::-1, :].copy()
        return image, mask  # , body, detail


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, body=None, detail=None):
        image = (image - self.mean) / self.std
        mask /= 255
        if body is None:
            return image, mask
        else:
            body /= 255
            detail /= 255
            return image, mask, body, detail


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if body is None:
            return image, mask
        else:
            body = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            detail = cv2.resize(detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask, body, detail


class RandomBrightness(object):
    def __call__(self, image, mask, body=None, detail=None):
        contrast = np.random.rand(1) + 0.5
        light = np.random.randint(-15, 15)
        inp_img = contrast * image + light
        return np.clip(inp_img, 0, 255), mask  # , body, detail


class RandomCrop(object):
    def __call__(self, image, mask, body=None, detail=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3, :]  # , body[p0:p1,p2:p3, :], detail[p0:p1,p2:p3, :]


class RandomBlur:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, label, body=None, detail=None):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
        return im, label  # , body, detail


class RandomScaleAspect:
    """
    Crop a sub-image from an original image with a range of area ratio and aspect and
    then scale the sub-image back to the size of the original image.

    Args:
        min_scale (float, optional): The minimum area ratio of cropped image to the original image. Default: 0.5.
        aspect_ratio (float, optional): The minimum aspect ratio. Default: 0.33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    im = cv2.resize(
                        im, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    if label is not None:
                        label = label[h1:(h1 + dh), w1:(w1 + dw)]
                        label = cv2.resize(
                            label, (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return (im,)
        else:
            return (im, label)


class ToTensor(object):
    def __call__(self, image, mask, body=None, detail=None):
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        image, mask = image.astype(np.float32), mask.astype(np.float32)
        mask = mask.mean(axis=0, keepdims=True)
        if body is None:
            return image, mask
        else:
            body = body.transpose((2, 0, 1)).astype(np.float32)
            body = body.mean(axis=0, keepdims=True)
            detail = detail.transpose((2, 0, 1)).astype(np.float32)
            detail = detail.mean(axis=0, keepdims=True)
            return image, mask, body, detail


