import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.utils import load_state_dict_from_url
from functools import partial

import functools

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
    'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
]


class GhostBN2D_Old(nn.Module):

    def __init__(self,
                 num_features,
                 *args,
                 virtual2actual_batch_size_ratio=2,
                 affine=False,
                 sync_stats=False,
                 **kwargs):
        super().__init__()
        self.virtual2actual_batch_size_ratio = virtual2actual_batch_size_ratio
        self.affine = affine
        self.num_features = num_features
        self.sync_stats = sync_stats
        self.proxy_bn = nn.BatchNorm2d(num_features * virtual2actual_batch_size_ratio,
                                       *args,
                                       **kwargs,
                                       affine=False)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        # for mimic the behavior that different GPUs use different stats when eval
        self.eval_use_different_stats = False

    def reset_parameters(self) -> None:
        self.proxy_bn.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_actual_running_stats(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if not self.proxy_bn.track_running_stats:
            return None, None
        else:
            select_fun = {False: lambda x: x[0], True: lambda x: torch.mean(x, dim=0)}[self.sync_stats]
            return tuple(
                select_fun(var.reshape(self.virtual2actual_batch_size_ratio, self.num_features))
                for var in [self.proxy_bn.running_mean, self.proxy_bn.running_var])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = (self.proxy_bn.running_mean is None) and (self.proxy_bn.running_var is None)

        if bn_training or self.eval_use_different_stats:
            n, c, h, w = input.shape
            if n % self.virtual2actual_batch_size_ratio != 0:
                raise RuntimeError()
            proxy_input = input.reshape(int(n / self.virtual2actual_batch_size_ratio),
                                        self.virtual2actual_batch_size_ratio * c, h, w)
            proxy_output = self.proxy_bn(proxy_input)
            proxy_output = proxy_output.reshape(n, c, h, w)

            if self.affine:
                weight = self.weight
                bias = self.bias
                weight = weight.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                #                 print('proxy_output', proxy_output.shape)
                return proxy_output * weight + bias
            else:
                return proxy_output
        else:
            #             print('running_mean', running_mean.shape)
            running_mean, running_var = self.get_actual_running_stats()

            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                # won't update running_mean & running_var
                0.0,
                self.proxy_bn.eps)


class NoOpAttacker():

    def attack(self, image, label, model):
        return image, -torch.ones_like(label)


class FourBN(nn.Module):

    def __init__(self,
                 num_features,
                 *args,
                 virtual2actual_batch_size_ratio=2,
                 affine=False,
                 sync_stats=False,
                 **kwargs):
        super(FourBN, self).__init__()

        self.bn0 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn1 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn2 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn3 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)

        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)

    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)

        input = self.aff(input)
        return input


class EightBN(nn.Module):

    def __init__(self,
                 num_features,
                 *args,
                 virtual2actual_batch_size_ratio=2,
                 affine=False,
                 sync_stats=False,
                 **kwargs):
        super(EightBN, self).__init__()

        self.bn0 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn1 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn2 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn3 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn4 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn5 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn6 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)
        self.bn7 = GhostBN2D_Old(num_features=num_features,
                                 *args,
                                 virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio,
                                 affine=affine,
                                 sync_stats=sync_stats,
                                 **kwargs)

        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)

    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        elif self.bn_type == 'bn4':
            input = self.bn4(input)
        elif self.bn_type == 'bn5':
            input = self.bn5(input)
        elif self.bn_type == 'bn6':
            input = self.bn6(input)
        elif self.bn_type == 'bn7':
            input = self.bn7(input)

        input = self.aff(input)
        return input


def eval_use_different_stats(model, val=False):

    def aux(m):
        if isinstance(m, GhostBN2D_Old):
            m.eval_use_different_stats = val

    model.apply(aux)


to_clean = functools.partial(eval_use_different_stats, val=False)
to_adv = functools.partial(eval_use_different_stats, val=True)


def to_bn(m, status):
    if hasattr(m, 'bn_type'):
        m.bn_type = status


to_0 = partial(to_bn, status='bn0')
to_1 = partial(to_bn, status='bn1')
to_2 = partial(to_bn, status='bn2')
to_3 = partial(to_bn, status='bn3')


class Affine(nn.Module):

    def __init__(self, width, *args, k=1, **kwargs):
        super(Affine, self).__init__()
        self.bnconv = nn.Conv2d(width, width, k, padding=(k - 1) // 2, groups=width, bias=True)

    def forward(self, x):
        return self.bnconv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.gelu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# Zero-initialize the last BN in each residual branch,
# so that the residual branch starts with zeros, and each residual block behaves like an identity.
# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        for m in self.modules():
            if isinstance(m, Affine):
                assert m.bnconv.weight is not None
                assert m.bnconv.bias is not None
                nn.init.constant_(m.bnconv.weight, 1)
                nn.init.constant_(m.bnconv.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                  norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class AdvResNet(ResNet):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 attacker=NoOpAttacker()):
        super().__init__(block,
                         layers,
                         num_classes=num_classes,
                         zero_init_residual=zero_init_residual,
                         groups=groups,
                         width_per_group=width_per_group,
                         replace_stride_with_dilation=replace_stride_with_dilation,
                         norm_layer=norm_layer)
        self.attacker = attacker
        self.mix = False
        self.sing = False
        self.mixup_fn = False
        self.bn_num = 0
#         self.iter = 0

    def set_mixup_fn(self, mixup):
        self.mixup_fn = mixup

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_bn_num(self, bn_num):
        self.bn_num = bn_num

    def set_mix(self, mix):
        self.mix = mix

    def set_sing(self, sing):
        self.sing = sing

    def forward(self, x, labels):
        if self.sing:
            # Adversarial training.
            training = self.training
            input_len = len(x)
            if training:
                self.eval()
                self.apply(to_adv)
                if self.bn_num == 0:
                    self.apply(to_0)
                elif self.bn_num == 1:
                    self.apply(to_1)
                elif self.bn_num == 2:
                    self.apply(to_2)
                elif self.bn_num == 3:
                    self.apply(to_3)


#                 elif self.bn_num == 4:
#                     self.apply(to_4)
#                 elif self.bn_num == 5:
#                     self.apply(to_5)
#                 elif self.bn_num == 6:
#                     self.apply(to_6)
#                 elif self.bn_num == 7:
#                     self.apply(to_7)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, True,
                                                         self.mixup_fn)
                    images = aux_images
                    targets = labels
                self.train()
                self.apply(to_clean)
                return self._forward_impl(images), targets
            else:
                self.apply(to_0)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, False, False)
                    images = aux_images
                    targets = labels
                return self._forward_impl(images), targets


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError('do not set pretrained as True, since we aim at training from scratch')
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
