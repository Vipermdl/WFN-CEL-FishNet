# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image


import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return {'LF': PIL.ImageOps.autocontrast(img['LF']),
            'HF': PIL.ImageOps.autocontrast(img['HF'])}


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return {'LF': PIL.ImageEnhance.Brightness(img['LF']).enhance(v),
            'HF': PIL.ImageEnhance.Brightness(img['HF']).enhance(v)}


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return {'LF': PIL.ImageEnhance.Color(img['LF']).enhance(v),
            'HF': PIL.ImageEnhance.Color(img['HF']).enhance(v)}


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return {'LF': PIL.ImageEnhance.Contrast(img['LF']).enhance(v),
            'HF': PIL.ImageEnhance.Contrast(img['HF']).enhance(v)}


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img['LF'].size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img['LF'].size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    LF = img['LF'].copy()
    PIL.ImageDraw.Draw(LF).rectangle(xy, color)
    HF = img['HF'].copy()
    PIL.ImageDraw.Draw(HF).rectangle(xy, color)
    return {'LF': LF, 'HF': HF}


def Equalize(img, **kwarg):
    return {'LF': PIL.ImageOps.equalize(img['LF']),
            'HF': PIL.ImageOps.equalize(img['HF'])}


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return {'LF': PIL.ImageOps.invert(img['LF']),
            'HF': PIL.ImageOps.invert(img['HF'])}


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return {'LF': PIL.ImageOps.posterize(img['LF'], v),
            'HF': PIL.ImageOps.posterize(img['HF'], v)}


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return {'HF': img['HF'].rotate(v), 'LF': img['LF'].rotate(v)}


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return {'HF':PIL.ImageEnhance.Sharpness(img['HF']).enhance(v),
            'LF':PIL.ImageEnhance.Sharpness(img['LF']).enhance(v)}


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return {'HF': img['HF'].transform(img['HF'].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)),
            'LF': img['LF'].transform(img['LF'].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))}


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return {'HF': img['HF'].transform(img['HF'].size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)),
            'LF': img['LF'].transform(img['LF'].size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))}


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return {'LF': PIL.ImageOps.solarize(img['LF'], 256 - v),
            'HF': PIL.ImageOps.solarize(img['HF'], 256 - v)}


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    
    img_np_HF = np.array(img['HF']).astype(np.int)
    img_np_HF = img_np_HF + v
    img_np_HF = np.clip(img_np_HF, 0, 255)
    img_np_HF = img_np_HF.astype(np.uint8)
    img_HF = Image.fromarray(img_np_HF)

    img_np_LF = np.array(img['LF']).astype(np.int)
    img_np_LF = img_np_LF + v
    img_np_LF = np.clip(img_np_LF, 0, 255)
    img_np_LF = img_np_LF.astype(np.uint8)
    img_LF = Image.fromarray(img_np_LF)

    return {'LF': PIL.ImageOps.solarize(img_LF, threshold),
            'HF': PIL.ImageOps.solarize(img_HF, threshold)}


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img['HF'].size[0])
    return {'HF': img['HF'].transform(img['HF'].size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)),
            'LF': img['LF'].transform(img['LF'].size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))}
    


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img['HF'].size[1])
    return {'HF': img['HF'].transform(img['HF'].size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)),
            'LF': img['LF'].transform(img['LF'].size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))}


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class Resize(transforms.Resize):
    def forward(self, img):
        return {
            'LF': F.resize(img['LF'], self.size, self.interpolation, self.max_size, self.antialias),
            'HF': F.resize(img['HF'], self.size, self.interpolation, self.max_size, self.antialias),
            
        }

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        if torch.rand(1) < self.p:        
            return {
                'LF': F.hflip(img['LF']),
                'HF': F.hflip(img['HF'])}
        return img

class RandomCrop(transforms.RandomCrop):
    def forward(self, img):
        if self.padding is not None:
            img['LF'] = F.pad(img['LF'], self.padding, self.fill, self.padding_mode)
            img['HF'] = F.pad(img['HF'], self.padding, self.fill, self.padding_mode)
        
        width, height = F.get_image_size(img['HF'])

        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1]-width, 0]
            img['HF'] = F.pad(img['HF'], padding, self.fill, self.padding_mode)
            img['LF'] = F.pad(img['LF'], padding, self.fill, self.padding_mode)
        
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0]-height]
            img['HF'] = F.pad(img['HF'], padding, self.fill, self.padding_mode)
            img['LF'] = F.pad(img['LF'], padding, self.fill, self.padding_mode)
        
        i,j,h,w = self.get_params(img['LF'], self.size)

        return {'LF': F.crop(img['LF'], i, j, h, w), 'HF': F.crop(img['HF'], i, j, h, w),}

class ToTensor(transforms.ToTensor):
    def __call__(self, pic):
        return {'LF': F.to_tensor(pic['LF']), 'HF': F.to_tensor(pic['HF'])}

class Normalize(transforms.Normalize):
    def forward(self, pic):
        return {'LF': F.normalize(pic['LF'], self.mean, self.std, self.inplace), 
        'HF': F.normalize(pic['HF'], self.mean, self.std, self.inplace)}

class CenterCrop(transforms.CenterCrop):
    def forward(self, img):
        return {'LF': F.center_crop(img['LF'], self.size), 
        'HF': F.center_crop(img['HF'], self.size)}
