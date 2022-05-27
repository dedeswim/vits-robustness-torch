"""Random Erasing implementation to work on inputs in [0, 1].

Parts of this file are adapted from PyTorch Image Models by Ross Wightman

The original ones can be found at https://github.com/rwightman/pytorch-image-models/

The original license can be found at this link:
https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
"""

import math
import random

import torch
from torch import nn
from timm.data.random_erasing import RandomErasing


def _get_pixels(per_pixel, rand_color, patch_size, a, b, dtype=torch.float32, device='cuda'):
    mean = (b - a) / 2
    std = mean / 2
    if per_pixel:
        pixels = torch.empty(patch_size, dtype=dtype, device=device)
        nn.init.trunc_normal_(pixels, mean, std, a, b)
    elif rand_color:
        pixels = torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device)
        nn.init.trunc_normal_(pixels, mean, std, a, b)
    else:
        pixels = torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
    return pixels


class NotNormalizedRandomErasing(RandomErasing):

    def __init__(self,
                 a=0,
                 b=1,
                 probability=0.5,
                 min_area=0.02,
                 max_area=1 / 3,
                 min_aspect=0.3,
                 max_aspect=None,
                 mode='const',
                 count=1,
                 num_splits=0):
        """Adaptation of timm's RandomErasing that works for images in [0, 1].

        It uses a normal truncated in [0, 1] instead of a regular normal"""
        super().__init__(probability, min_area, max_area, min_aspect, max_aspect, mode, count, num_splits)
        self.b = b
        self.a = a

    def _erase(self, img, chan, img_h, img_w, dtype):
        device = img.device
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = random.randint(1, self.count) if self.count > 1 else self.count
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(self.per_pixel,
                                                                     self.rand_color, (chan, h, w),
                                                                     self.a,
                                                                     self.b,
                                                                     dtype=dtype,
                                                                     device=device)
                    break
