"""From
https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/wide_resnet.py"""

import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1',
        'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'wide_resnet28_10': _cfg(),
    'wide_resnet34_10': _cfg(),
    'wide_resnet34_20': _cfg(),
    'wide_resnet70_16': _cfg(),
    'wide_resnet106_16': _cfg(),
}


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self,
                 depth=28,
                 num_classes=10,
                 widen_factor=10,
                 sub_block1=False,
                 drop_rate=0.0,
                 bias_last=True,
                 in_chans=3,
                 img_size=224):
        super().__init__()
        self.num_classes = num_classes
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_chans, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes, bias=bias_last)
        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)
        return self.fc(out)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.n_channels, num_classes) if num_classes > 0 else nn.Identity()


def _create_wide_resnet(variant, pretrained=False, default_cfg=None, **kwargs):
    model = build_model_with_cfg(WideResNet, variant, pretrained, **kwargs)
    return model


@register_model
def wide_resnet28_10(pretrained=False, **kwargs):
    model_args = dict(depth=28, widen_factor=10, **kwargs)
    return _create_wide_resnet('wide_resnet28_10', pretrained, **model_args)


@register_model
def wide_resnet34_10(pretrained=False, **kwargs):
    model_args = dict(depth=34, widen_factor=10, **kwargs)
    return _create_wide_resnet('wide_resnet34_10', pretrained, **model_args)


@register_model
def wide_resnet34_20(pretrained=False, **kwargs):
    model_args = dict(depth=34, widen_factor=20, **kwargs)
    return _create_wide_resnet('wide_resnet34_20', pretrained, **model_args)


@register_model
def wide_resnet70_16(pretrained=False, **kwargs):
    model_args = dict(depth=70, widen_factor=16, **kwargs)
    return _create_wide_resnet('wide_resnet70_16', pretrained, **model_args)


@register_model
def wide_resnet106_16(pretrained=False, **kwargs):
    model_args = dict(depth=106, widen_factor=16, **kwargs)
    return _create_wide_resnet('wide_resnet106_16', pretrained, **model_args)
