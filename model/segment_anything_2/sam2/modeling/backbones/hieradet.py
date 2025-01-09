# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.segment_anything_2.sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from model.segment_anything_2.sam2.modeling.sam2_utils import DropPath, MLP


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.N = 8 
        self.prompt_proj = nn.Linear(dim, self.N * dim)

        
        # === Adapter module ===
        # A small, trainable bottleneck MLP. In typical adapter style:
        #   dim -> adapter_dim -> dim
        adapter_dim = 128
        self.adapter_down = nn.Linear(dim, adapter_dim, bias=False)
        self.adapter_act = nn.ReLU()  # or GELU, etc.
        self.adapter_up = nn.Linear(adapter_dim, dim, bias=False)
        self.adapter_norm = nn.LayerNorm(dim)


        # Initialize weights
        self._reset_parameters()

    def forward(self, x_img, prompt_features):
        """
        x_img: (B, H, W, C) - Image feature
        x_clip: (B, C) - Clip feature
        """

        B, H, W, C = x_img.shape
        L_img = H * W

        # Reshape image features
        # q = self.q_proj(x_img.reshape(B, L_img, C))  # (B, L_img, C)
        # k = self.k_proj(x_clip)  # (B, L_clip, C)
        # v = self.v_proj(x_clip)  # (B, L_clip, C)
        x_img = x_img.reshape(B, L_img, C)

        prompt_features = nn.GELU()(prompt_features)
        prompt_features = self.prompt_proj(prompt_features)
        # prompt_features = prompt_features.expand(B, L_img, -1)
        prompt_features = prompt_features.reshape(B, self.N, C)

        if torch.isnan(prompt_features).any() or torch.isinf(prompt_features).any():
            print(f"CrossAttention: prompt_features has NaN or Inf values")

        q = x_img.transpose(0, 1) # (L_img, B, C)
        k = prompt_features.transpose(0, 1)  # (N, B, C)
        v = prompt_features.transpose(0, 1)  # (N, B, C)

        # Apply multi-head attention
        x, _ = F.multi_head_attention_forward(
            query=q, key=k, value=v, 
            embed_dim_to_check=C,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None, 
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"CrossAttention: x has NaN or Inf values")

        x_adapter = self.adapter_down(x)
        x_adapter = self.adapter_act(x_adapter)
        x_out = self.adapter_up(x_adapter)
        x_out = self.adapter_norm(x_out)
        # x_out = x

        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
            print(f"adapter: x_out has NaN or Inf values")

        # Reshape back to image format (B, H, W, C)        
        x_out = x_out.transpose(0, 1)

        x_out = x_img + x_out
        x_out = x_out.view(B, H, W, C)
        return x_out

    def _reset_parameters(self):
        """
        Initializes the weights of the layers with reasonable values to avoid
        NaN losses or poor convergence.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.prompt_proj.weight)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        

        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.prompt_proj.bias)
        

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        use_cross_attention: bool = False,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        if use_cross_attention:
            self.cross_attn = CrossAttention(dim, num_heads)
            self.cross_attn_norm = norm_layer(dim)
        else:
            self.cross_attn = None
            self.cross_attn_norm = None
        # self.cross_attn = CrossAttention(dim, num_heads) if use_cross_attention else None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor, prompt_features) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)



        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))


        if self.cross_attn and prompt_features is not None:
            x = shortcut + self.drop_path(x)  # Add self-attention residual connection

            x = x + self.drop_path(self.cross_attn(self.cross_attn_norm(x), prompt_features))

            # x = shortcut + self.drop_path(x)
            # MLP
            x = x + self.drop_path(self.mlp(self.norm2(x))) # this is frozen

        else:
            x = shortcut + self.drop_path(x)
            # MLP
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
        cross_attn: bool = False,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        if cross_attn:
            self.cross_attn_blocks = self.global_att_blocks
            print(f"Using cross attention in layers : {self.cross_attn_blocks}")
        else:
            print("Not using cross attention")
            self.cross_attn_blocks = []

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                use_cross_attention=(i in self.cross_attn_blocks),
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor, prompted_feats: torch.Tensor) -> List[torch.Tensor]:
        
        
        if prompted_feats is None:
            use_adapter = False
        else:
            use_adapter = True
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]

        for i, blk in enumerate(self.blocks):

            # add prompted features 
            current_gpu = blk.attn.qkv.weight.device
            x = x.to(current_gpu)

            if use_adapter:
                prompted_feats[i] = prompted_feats[i].to(current_gpu)
                x = prompted_feats[i].reshape(B, 1, 1, -1) + x
                x = blk(x, prompted_feats[i])
            else:
                x = blk(x, None)

            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        
        # update outputs's device to current device
        outputs = [output.to(current_gpu) for output in outputs]
        return outputs
