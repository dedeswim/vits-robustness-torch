from timm.models import cait, xcit
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model

default_cfgs = {
    'cait_s12_224': cait._cfg(input_size=(3, 224, 224)),
    'xcit_medium_12_p16_224': xcit._cfg(),
    'xcit_large_12_p16_224': xcit._cfg()
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
