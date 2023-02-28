from math import sqrt
from typing import Callable

import torch
from einops import rearrange
from tqdm import tqdm


def raise_if_not_batched_2d_tensor(tensor: torch.Tensor):
    if len(tensor.shape) != 3:
        raise ValueError(
            f'Expected tensor to be 2D batched tensor (N, Tokens, Embeddings), but got shape {tensor.shape}'
        )


def raise_if_not_batched_3d_tensor(tensor: torch.Tensor):
    if len(tensor.shape) != 4:
        raise ValueError(f'Expected tensor to be 3D batched tensor (N, C, H, W), but got shape: {tensor.shape}')


def split_to_patches(tensor: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    raise_if_not_batched_3d_tensor(tensor)
    b, c, h, w = tensor.shape
    pad_h = -h % patch_size
    pad_w = -w % patch_size
    pad = (0, pad_w, 0, pad_h)
    padded = torch.nn.functional.pad(tensor, pad, mode='reflect')
    patches = padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    _, _, p_rows, p_cols, _, _ = patches.shape
    patches = rearrange(patches, 'b c p_rows p_cols h w -> (p_rows p_cols) b c h w')
    return patches, p_rows, p_cols


def merge_patches(
    patches,
    out_shape: tuple[int, int],
    patches_rows,
    patches_cols,
) -> torch.Tensor:
    if len(patches.shape) != 5:
        raise ValueError(
            'Expected patches tensor to be 3D batched tensor with P patches '
            f'(P, N, C, H, W), but got shape: {patches.shape}'
        )
    oh, ow = out_shape
    p, b, c, h, w = patches.shape
    p_r, p_c = patches_rows, patches_cols

    patches = rearrange(patches, '(p_r p_c) b c h w -> b c p_r p_c h w', p_r=p_r, p_c=p_c)
    merged = patches.permute(0, 1, 2, 4, 3, 5).reshape(b, c, p_r * h, p_c * w)
    return merged[..., :oh, :ow]


def process_in_patches(
    tensor: torch.Tensor, patch_size: int, transform: Callable[[torch.Tensor], torch.Tensor], progress: bool = False
) -> torch.Tensor:
    raise_if_not_batched_3d_tensor(tensor)
    b, c, h, w = tensor.shape
    patches, patches_rows, patches_cols = split_to_patches(tensor, patch_size)
    assert len(patches.shape) == 5, f'{patches.shape=}'

    outs = []
    for p in tqdm(patches, disable=not progress):
        outs.append(transform(p))
    merged = merge_patches(torch.stack(outs), (h, w), patches_rows, patches_cols)
    return merged


class SmoothFusionBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels

        self.small_conv = torch.nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))
        self.small_bn = torch.nn.BatchNorm2d(channels)
        self.medium_conv = torch.nn.Conv2d(channels, channels, (5, 5), padding=(2, 2))
        self.medium_bn = torch.nn.BatchNorm2d(channels)
        self.big_conv = torch.nn.Conv2d(channels, channels, (9, 9), padding=(4, 4))
        self.large_bn = torch.nn.BatchNorm2d(channels)

        self.out_relu = torch.nn.ReLU(channels)

    def forward(self, x):
        x_s = self.small_bn(self.small_conv(x))
        x_m = self.medium_bn(self.medium_conv(x))
        x_b = self.large_bn(self.big_conv(x))
        return self.out_relu(x + x_s + x_m + x_b)


class SpectralSpatialCrossTransformer(torch.nn.Module):
    def __init__(self, channels: int, spatial_size: int, num_heads: int, channel_embed_size: int):
        super().__init__()

        self.channels = channels
        self.spatial_size = spatial_size

        assert sqrt(channel_embed_size).is_integer()
        channel_embed_hw = int(sqrt(channel_embed_size))
        assert (spatial_size / channel_embed_hw).is_integer()
        ch_scale = int(spatial_size / channel_embed_hw)

        self.channel_embed_hw = channel_embed_hw
        self.ch_scale = ch_scale

        self.sep_conv_spatial = torch.nn.Conv2d(channels, channels, (3, 3), padding=(1, 1), groups=channels)

        self.down_conv_in_self_attn_channel = torch.nn.Conv2d(
            channels, channels, (ch_scale, ch_scale), stride=(ch_scale, ch_scale), padding='valid'
        )
        self.up_out_self_attn_channel = torch.nn.Upsample(scale_factor=ch_scale)

        # Spatial and channel self attention
        self.self_attn_spatial = torch.nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.self_attn_channel = torch.nn.MultiheadAttention(channel_embed_size, num_heads, batch_first=True)

        self.down_conv_in_cross_attn_channel = torch.nn.Conv2d(
            channels, channels, (ch_scale, ch_scale), stride=(ch_scale, ch_scale), padding='valid'
        )
        self.up_out_cross_attn_channel = torch.nn.Upsample(scale_factor=ch_scale)

        # Spatial-channel and channel-spatial cross attention
        self.cross_attn_spatial = torch.nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_attn_channel = torch.nn.MultiheadAttention(channel_embed_size, num_heads, batch_first=True)

        self.final_conv = torch.nn.Conv2d(channels * 2, channels, (3, 3), padding=(1, 1))

    def forward(self, x):
        raise_if_not_batched_3d_tensor(x)
        b, c, h, w = x.shape
        if h != w != self.spatial_size:
            raise ValueError()
        if c != self.channels:
            raise ValueError()

        t_sp = self.sep_conv_spatial(x)
        t_sp = rearrange(x, 'b c h w -> b (h w) c')

        t_ch = self.down_conv_in_self_attn_channel(x)
        t_ch = rearrange(t_ch, 'b c h w -> b c (h w)')

        os_sp = self.self_attn_spatial(t_sp, t_sp, t_sp)[0]
        os_ch = self.self_attn_channel(t_ch, t_ch, t_ch)[0]

        os_ch_up = rearrange(os_ch, 'b c (h w) -> b c h w', h=self.channel_embed_hw, w=self.channel_embed_hw)
        os_ch_up = self.up_out_self_attn_channel(os_ch_up)
        os_ch_up = rearrange(os_ch_up, 'b c h w -> b (h w) c', h=self.spatial_size, w=self.spatial_size)

        os_sp_down = rearrange(os_sp, 'b (h w) c -> b c h w', h=self.spatial_size, w=self.spatial_size)
        os_sp_down = self.down_conv_in_cross_attn_channel(os_sp_down)
        os_sp_down = rearrange(os_sp_down, 'b c h w -> b c (h w)', h=self.channel_embed_hw, w=self.channel_embed_hw)

        oc_sp = self.cross_attn_spatial(os_ch_up, os_sp, os_sp)[0]
        oc_ch = self.cross_attn_channel(os_sp_down, os_ch, os_ch)[0]

        oc_ch = rearrange(oc_ch, 'b c (h w) -> b c h w', h=self.channel_embed_hw, w=self.channel_embed_hw)
        oc_ch = self.up_out_cross_attn_channel(oc_ch)

        oc_sp = rearrange(oc_sp, 'b (h w) c -> b c h w', h=self.spatial_size, w=self.spatial_size)

        concated = torch.concat([oc_sp, oc_ch], axis=1)
        out = self.final_conv(concated)
        return out


class IceTransformer(torch.nn.Module):
    def __init__(self, channels: int, patch_size: int, channel_embed_size: int):
        super().__init__()

        self.channels = channels
        self.patch_size = patch_size
        self.out_conv_ch = 32

        self.spc_spt_tf = SpectralSpatialCrossTransformer(channels, patch_size, 4, channel_embed_size)

        self.smooth_conv_1 = SmoothFusionBlock(channels)
        self.smooth_conv_2 = SmoothFusionBlock(channels)

        self.conv_sic = torch.nn.Conv2d(self.channels, self.out_conv_ch, kernel_size=(3, 3), padding=(1, 1))
        self.conv_sod = torch.nn.Conv2d(self.channels, self.out_conv_ch, kernel_size=(3, 3), padding=(1, 1))
        self.conv_floe = torch.nn.Conv2d(self.channels, self.out_conv_ch, kernel_size=(3, 3), padding=(1, 1))

        self.output_conv_sic = torch.nn.Conv2d(self.out_conv_ch, 12, kernel_size=(1, 1), stride=(1, 1))
        self.output_conv_sod = torch.nn.Conv2d(self.out_conv_ch, 7, kernel_size=(1, 1), stride=(1, 1))
        self.output_conv_floe = torch.nn.Conv2d(self.out_conv_ch, 8, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, patch_progress: bool = False):
        raise_if_not_batched_3d_tensor(x)
        b, c, h, w = x.shape

        x = process_in_patches(x, self.patch_size, lambda p: self.spc_spt_tf(p), progress=patch_progress)

        x = self.smooth_conv_1(x)
        x = self.smooth_conv_2(x)

        return {
            'SIC': self.output_conv_sic(self.conv_sic(x)),
            'SOD': self.output_conv_sod(self.conv_sod(x)),
            'FLOE': self.output_conv_floe(self.conv_floe(x)),
        }
