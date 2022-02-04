from timm.models import cait, xcit
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model


default_cfgs = {
    'xcit_small_12_p8_32': xcit._cfg(input_size=(3, 32, 32)),
    'cait_xs24_224': cait._cfg(input_size=(3, 224, 224)),
}


@register_model
def cait_xs24_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=288, depth=24, num_heads=8, init_values=1e-5, **kwargs)
    return build_model_with_cfg(cait.Cait,
                                'cait_s24_224',
                                pretrained,
                                pretrained_filter_fn=cait.checkpoint_filter_fn,
                                **model_kwargs)


@register_model
def xcit_small_12_p16_160(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    default_cfg = xcit._cfg("", input_size=(3, 160, 160))
    return build_model_with_cfg(xcit.XCiT,
                                'xcit_small_12_p16_160',
                                pretrained,
                                default_cfg=default_cfg,
                                pretrained_filter_fn=xcit.checkpoint_filter_fn,
                                **model_kwargs)


@register_model
def xcit_small_12_p16_128(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    default_cfg = xcit._cfg("", input_size=(3, 128, 128))
    return build_model_with_cfg(xcit.XCiT,
                                'xcit_small_12_p16_128',
                                pretrained,
                                default_cfg=default_cfg,
                                pretrained_filter_fn=xcit.checkpoint_filter_fn,
                                **model_kwargs)


@register_model
def xcit_small_12_p16_80(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    default_cfg = xcit._cfg("", input_size=(3, 80, 80))
    return build_model_with_cfg(xcit.XCiT,
                                'xcit_small_12_p16_80',
                                pretrained,
                                default_cfg=default_cfg,
                                pretrained_filter_fn=xcit.checkpoint_filter_fn,
                                **model_kwargs)


@register_model
def xcit_small_12_p8_32(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=8,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    return build_model_with_cfg(xcit.XCiT,
                                'xcit_small_12_p8_32',
                                pretrained,
                                pretrained_filter_fn=xcit.checkpoint_filter_fn,
                                **model_kwargs)
