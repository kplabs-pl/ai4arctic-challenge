from typing import Callable

import torch
from einops import rearrange


def raise_if_not_batched_3d_tensor(tensor: torch.Tensor):
    if len(tensor.shape) != 4:
        raise ValueError('Expected tensor to be 3D batched tensor (N, C, H, W), ' f'but got shape: {tensor.shape}')


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
    def __init__(self, in_channels: int, out_channels: int, spatial_size: int, num_heads: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        # Spectral, spatial self attention
        self.as_ch = torch.nn.MultiheadAttention(in_channels, num_heads)
        self.as_sp = torch.nn.MultiheadAttention(spatial_size**2, num_heads)
        # Spectral-spatial, spatial-spectral cross attention
        self.ac_ch = torch.nn.MultiheadAttention(in_channels, num_heads)
        self.ac_sp = torch.nn.MultiheadAttention(spatial_size**2, num_heads)

        self.final_conv = torch.nn.Conv2d(in_channels * 2, out_channels, (3, 3), padding=(1, 1))
        self.bn_final_conv = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        raise_if_not_batched_3d_tensor(x)
        b, c, h, w = x.shape
        if h != w != self.spatial_size:
            raise ValueError()
        if c != self.in_channels:
            raise ValueError()

        t_flat = x.reshape(b, c, h * w)
        t_ch = t_flat.moveaxis(1, 2)
        t_sp = t_flat

        os_ch = self.as_ch(t_ch, t_ch, t_ch)[0]
        os_sp = self.as_sp(t_sp, t_sp, t_sp)[0]

        oc_ch = self.ac_ch(os_sp.moveaxis(1, 2), os_ch, os_ch)[0]
        oc_sp = self.ac_sp(os_ch.moveaxis(1, 2), os_sp, os_sp)[0]

        oc_ch_unflat = oc_ch.moveaxis(1, 2).reshape(x.shape)
        oc_sp_unflat = oc_sp.reshape(x.shape)

        concated = torch.concat([oc_ch_unflat, oc_sp_unflat], axis=1)
        out = self.bn_final_conv(self.final_conv(concated))

        return out


class DoubleConvResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self._channels = channels

        self.conv_1 = torch.nn.Conv2d(self._channels, self._channels, (3, 3), padding=(1, 1))
        self.bn_1 = torch.nn.BatchNorm2d(self._channels)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(self._channels, self._channels, (3, 3), padding=(1, 1))
        self.bn_2 = torch.nn.BatchNorm2d(self._channels)
        self.relu_2 = torch.nn.ReLU()

    def forward(self, x):
        raise_if_not_batched_3d_tensor(x)
        residual = x
        x = self.bn_1(self.conv_1(x))
        x = self.relu_1(x)
        x = self.bn_2(self.conv_2(x))
        x = x + residual
        x = self.relu_2(x)
        return x


class IceTransformer(torch.nn.Module):
    def __init__(self, channels: int, patch_size: int):
        super().__init__()

        # Parameters
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = 4
        self.num_channels_conv = 32

        # Layers
        self.spc_spt_tf = SpectralSpatialCrossTransformer(channels, self.num_channels_conv, patch_size, self.num_heads)

        self.double_conv_res_block = DoubleConvResidualBlock(self.num_channels_conv)

        self.output_conv_sic = torch.nn.Conv2d(self.num_channels_conv, 12, kernel_size=(1, 1), stride=(1, 1))
        self.output_conv_sod = torch.nn.Conv2d(self.num_channels_conv, 7, kernel_size=(1, 1), stride=(1, 1))
        self.output_conv_floe = torch.nn.Conv2d(self.num_channels_conv, 8, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        raise_if_not_batched_3d_tensor(x)
        b, c, h, w = x.shape

        x = process_in_patches(x, self.patch_size, lambda p: self.spc_spt_tf(p))
        x = self.double_conv_res_block(x)
        return {'SIC': self.output_conv_sic(x), 'SOD': self.output_conv_sod(x), 'FLOE': self.output_conv_floe(x)}
