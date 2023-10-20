import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
from model.backbone.deit import DistilledVisionTransformer
from model.backbone.vit import VisionTransformer
import math
from timm.models.layers import trunc_normal_
"""
Borrow from https://github.com/facebookresearch/dino
"""


class PTVIT(DistilledVisionTransformer):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.num_classes = num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.cls_head = nn.Sequential(
          nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.AdaptiveAvgPool2d(1),
          nn.ReLU(),
        )
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights_i)

        return x, attn_weights

    def forward(self, x):
        x, attn_weights = self.forward_features(x)
        B, np, C = x.shape
        x = x.reshape(B, int(math.sqrt(np)), int(math.sqrt(np)), C).permute(0, 3, 1, 2).contiguous()
        x = self.cls_head(x)
        x = x.squeeze(3).squeeze(2)

        return x


class VIT_MOD(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        embeds = []
        weights_list = []
        for blk in self.blocks:
            x, weights = blk(x, )
            embeds.append(x)
            weights_list.append(weights)

        x = self.norm(x)
        embeds[-1] = x
        return x[:, 0], x[:, 1:], weights_list

    def forward(self, x, visual_attmap_f=False):
        x, p_x, weights_list = self.forward_features(x)
        x = self.head(x)
        if visual_attmap_f:
            return weights_list

        return x
