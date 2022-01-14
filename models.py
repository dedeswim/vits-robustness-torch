import timm
from timm.models import cait
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model


@register_model
def cait_xs24_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=288, depth=24, num_heads=8, init_scale=1e-5, **kwargs)
    default_cfg = cait._cfg("", input_size=(3, 224, 224))
    return build_model_with_cfg(
        cait.Cait, 'cait_s24_224', pretrained,
        default_cfg=default_cfg,
        pretrained_filter_fn=cait.checkpoint_filter_fn,
        **model_args)
