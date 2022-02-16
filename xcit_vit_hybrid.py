import math
from collections import OrderedDict
from functools import partial

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import xcit, vision_transformer as vit, build_model_with_cfg
from timm.models.helpers import named_apply
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from torch import nn


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs
    }


class XCiTViTHybrid(nn.Module):
    """Vision Transformer with XCA instead of SA.
    
    From timm.models.vision_transformer"""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 representation_size=None,
                 global_pool='',
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 weight_init='',
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            xcit.XCABlock(dim=embed_dim,
                          num_heads=num_heads,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[i],
                          norm_layer=norm_layer,
                          act_layer=act_layer) for i in range(depth)
        ])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if 'jax' not in mode:
            # init cls token to truncated normal if not following jax impl, jax impl is zero
            trunc_normal_(self.cls_token, std=.02)
        named_apply(partial(vit._init_vit_weights, head_bias=head_bias, jax_impl='jax' in mode), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        vit._init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        vit._load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool == 'avg':
            x = x[:, self.num_tokens:].mean(dim=1)
        else:
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x
