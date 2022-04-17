from timm.models import cait, resnet, xcit
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from torch import nn

from src import utils

default_cfgs = {
    'cait_s12_224': cait._cfg(input_size=(3, 224, 224)),
    'xcit_medium_12_p16_224': xcit._cfg(),
    'xcit_large_12_p16_224': xcit._cfg(),
    'xcit_large_12_h8_p16_224': xcit._cfg(),
    'xcit_small_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
    'xcit_medium_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
    'xcit_large_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
    'resnet18_gelu': resnet._cfg(),
    'resnet50_gelu': resnet._cfg(interpolation='bicubic', crop_pct=0.95),
    'resnext152_32x8d': resnet._cfg(input_size=(3, 380, 380))
}


@register_model
def cait_s12_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=8, init_values=1.0, **kwargs)
    return build_model_with_cfg(cait.Cait,
                                'cait_s12_224',
                                pretrained,
                                pretrained_filter_fn=cait.checkpoint_filter_fn,
                                **model_kwargs)


@register_model
def xcit_medium_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_medium_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_large_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=16,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_large_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_large_12_h8_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_large_12_h8_p16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def xcit_small_12_p8_32(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,  # 16 because the pre-trained model has 16
        embed_dim=384,
        depth=12,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p4_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.XCiT)
    model = utils.adapt_model_patches(model, 8)
    return model


@register_model
def xcit_small_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,  # 16 because the pre-trained model has 16
        embed_dim=384,
        depth=12,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p4_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.XCiT)
    model = utils.adapt_model_patches(model, 4)
    return model


@register_model
def xcit_medium_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_medium_12_p4_32', pretrained=pretrained, **model_kwargs)
    # TODO: make this a function
    assert isinstance(model, xcit.XCiT)
    model = utils.adapt_model_patches(model, 4)
    return model


@register_model
def xcit_large_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=16,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_large_12_p16_224', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.XCiT)
    model = utils.adapt_model_patches(model, 4)
    return model


@register_model
def xcit_small_12_p2_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p2_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.XCiT)
    model = utils.adapt_model_patches(model, 2)
    return model


@register_model
def resnet50_gelu(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model wit GELU."""
    model_args = dict(block=resnet.Bottleneck,
                      layers=[3, 4, 6, 3],
                      act_layer=lambda inplace: nn.GELU(),
                      **kwargs)
    return resnet._create_resnet('resnet50_gelu', pretrained, **model_args)


@register_model
def resnet18_gelu(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model wit GELU."""
    model_args = dict(block=resnet.BasicBlock,
                      layers=[2, 2, 2, 2],
                      act_layer=lambda inplace: nn.GELU(),
                      **kwargs)
    return resnet._create_resnet('resnet18_gelu', pretrained, **model_args)


@register_model
def resnext152_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt152-32x8d model. Added to compare to https://arxiv.org/abs/2006.14536"""
    model_args = dict(block=resnet.Bottleneck,
                      layers=[3, 4, 36, 3],
                      cardinality=32,
                      base_width=8,
                      act_layer=nn.SiLU,
                      **kwargs)
    return resnet._create_resnet('resnext152_32x8d', pretrained, **model_args)
