from typing import Callable

import torch
from einops import rearrange


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
    tensor: torch.Tensor, patch_size: int, transform: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    raise_if_not_batched_3d_tensor(tensor)
    b, c, h, w = tensor.shape
    patches, patches_rows, patches_cols = split_to_patches(tensor, patch_size)
    assert len(patches.shape) == 5, f'{patches.shape=}'
    patches = torch.stack([transform(p) for p in patches])
    merged = merge_patches(patches, (h, w), patches_rows, patches_cols)
    return merged


class SpectralSpatialCrossTransformer(torch.nn.Module):
    def __init__(self, channels: int, spatial_size: int, num_heads: int):
        super().__init__()

        self.channels = channels
        self.spatial_size = spatial_size

        self.down_conv_1 = torch.nn.Conv2d(channels, channels, (8, 8), stride=(8, 8), padding='valid')
        self.up_1 = torch.nn.Upsample(scale_factor=8)
        self.down_conv_2 = torch.nn.Conv2d(channels, channels, (8, 8), stride=(8, 8), padding='valid')
        self.up_2 = torch.nn.Upsample(scale_factor=8)

        # Spectral, spatial self attention
        self.as_ch = torch.nn.MultiheadAttention(channels, num_heads)
        self.as_sp = torch.nn.MultiheadAttention(64, num_heads)
        # Spectral-spatial, spatial-spectral cross attention
        self.ac_ch = torch.nn.MultiheadAttention(channels, num_heads)
        self.ac_sp = torch.nn.MultiheadAttention(64, num_heads)

        self.final_conv = torch.nn.Conv2d(channels * 2, channels, (3, 3), padding=(1, 1))

    def forward(self, x):
        raise_if_not_batched_3d_tensor(x)
        b, c, h, w = x.shape
        if h != w != self.spatial_size:
            raise ValueError()
        if c != self.channels:
            raise ValueError()

        t_ch = rearrange(x, 'b c h w -> b (h w) c')

        t_sp = self.down_conv_1(x)
        t_sp = rearrange(t_sp, 'b c h w -> b c (h w)')

        os_ch = self.as_ch(t_ch, t_ch, t_ch)[0]
        os_sp = self.as_sp(t_sp, t_sp, t_sp)[0]

        os_sp_up = rearrange(os_sp, 'b c (h w) -> b c h w', h=8, w=8)
        os_sp_up = self.up_1(os_sp_up)
        os_sp_up = rearrange(os_sp_up, 'b c h w -> b (h w) c', h=self.spatial_size, w=self.spatial_size)

        os_ch_down = rearrange(os_ch, 'b (h w) c -> b c h w', h=self.spatial_size, w=self.spatial_size)
        os_ch_down = self.down_conv_2(os_ch_down)
        os_ch_down = rearrange(os_ch_down, 'b c h w -> b c (h w)', h=8, w=8)

        oc_ch = self.ac_ch(os_sp_up, os_ch, os_ch)[0]
        oc_sp = self.ac_sp(os_ch_down, os_sp, os_sp)[0]

        oc_sp = rearrange(oc_sp, 'b c (h w) -> b c h w', h=8, w=8)
        oc_sp = self.up_2(oc_sp)

        oc_ch = rearrange(oc_ch, 'b (h w) c -> b c h w', h=self.spatial_size, w=self.spatial_size)

        concated = torch.concat([oc_sp, oc_ch], axis=1)
        out = self.final_conv(concated)
        return out


class IceTransformer(torch.nn.Module):
    def __init__(self, channels: int, patch_size: int):
        super().__init__()

        self.channels = channels
        self.patch_size = patch_size

        self.spc_spt_tf_1 = SpectralSpatialCrossTransformer(channels, patch_size, 4)
        self.spc_spt_tf_2 = SpectralSpatialCrossTransformer(channels, patch_size, 4)
        self.spc_spt_tf_3 = SpectralSpatialCrossTransformer(channels, patch_size, 4)

        self.final_conv_sic = torch.nn.Conv2d(self.channels, 24, (3, 3), padding=(1, 1))
        self.final_conv_sod = torch.nn.Conv2d(self.channels, 24, (3, 3), padding=(1, 1))
        self.final_conv_floe = torch.nn.Conv2d(self.channels, 24, (3, 3), padding=(1, 1))
        self.output_conv_sic = torch.nn.Conv2d(24, 12, kernel_size=(1, 1), stride=(1, 1))
        self.output_conv_sod = torch.nn.Conv2d(24, 7, kernel_size=(1, 1), stride=(1, 1))
        self.output_conv_floe = torch.nn.Conv2d(24, 8, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        raise_if_not_batched_3d_tensor(x)
        b, c, h, w = x.shape

        x = process_in_patches(
            x,
            self.patch_size,
            lambda p: self.spc_spt_tf_3(self.spc_spt_tf_2(self.spc_spt_tf_1(p))))

        return {
            'SIC': self.output_conv_sic(self.final_conv_sic(x)),
            'SOD': self.output_conv_sod(self.final_conv_sod(x)),
            'FLOE': self.output_conv_floe(self.final_conv_floe(x))
        }
