import json
from math import ceil
from pathlib import Path
from typing import Callable
from warnings import warn

import clize
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import xarray as xr
from nnunet.dataset_conversion.utils import generate_dataset_json
from skimage.transform import resize
from skimage.util import view_as_blocks

from loaders import get_variable_options
from utils import (
    FLOE_GROUPS,
    FLOE_LOOKUP,
    SCENE_VARIABLES,
    SIC_GROUPS,
    SIC_LOOKUP,
    SOD_GROUPS,
    SOD_LOOKUP,
)

EXPORT_OPTIONS = {
    'patch_size': 256,  # Size of patches sampled. Used for both Width and Height (default 256).
    'charts': ['SIC', 'SOD', 'FLOE'],  # Charts to train on.
    'pixel_spacing': 80,  # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
    'train_variables': SCENE_VARIABLES,  # Contains the relevant variables in the scenes.
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },
    'spacing': (999.0, 1.0, 1.0),
}


def warn_if_not_chw(array: np.ndarray):
    if array.ndim != 3:
        raise ValueError(f'Expected array in CHW format but got one with shape: {array.shape}')
    c, h, w = array.shape
    if c > h or c > w:
        warn(
            'Expected array in CHW format, but channel dimension is larger than spatial; are you sure you have '
            f'passed the array in the correct format? Received one with shape: {array.shape}'
        )


def apply_in_hwc(array_in_chw: np.ndarray, transform: Callable) -> np.ndarray:
    warn_if_not_chw(array_in_chw)
    return np.moveaxis(transform(np.moveaxis(array_in_chw, 0, 2)), 2, 0)


def split_to_patches(scene: np.ndarray, patch_size: int, fill_value: float) -> np.ndarray:
    warn_if_not_chw(scene)
    patch_shape = (scene.shape[0], patch_size, patch_size)
    pad_size = tuple(ceil(ss / ps) * ps - ss for ss, ps in zip(scene.shape[1:], patch_shape[1:]))
    pad_shape = (
        (0, 0),
        (0, pad_size[0]),
        (0, pad_size[1]),
    )
    padded = np.pad(scene, pad_shape, constant_values=fill_value)
    patches_grid = view_as_blocks(padded, patch_shape)
    patches = np.reshape(patches_grid, (-1, *patch_shape))
    return patches


def extract_file_name_from_file_desc(file_desc: str) -> str:
    sample_datetime = file_desc[17:32]
    data_source_name = file_desc[77:80]
    return sample_datetime + '_' + data_source_name + '_prep'


def extract_sample_x(scene_ds: xr.Dataset, sar_variables: list[str], amsrenv_variables: list[str]) -> np.ndarray:
    sar = scene_ds[sar_variables].to_array().values
    amsrenv = apply_in_hwc(scene_ds[amsrenv_variables].to_array().values, lambda x: resize(x, sar.shape[1:], order=0))
    x = np.concatenate([sar, amsrenv], axis=0)
    return x


def extract_sample_y(scene: xr.Dataset, charts: list[str]) -> np.ndarray:
    y = {chart: np.expand_dims(scene[chart].values, 0) for chart in charts}
    return y


def save_for_nnunet(array: np.ndarray, output_dir: Path, output_name: str, spacing: int, band_num: int | None = None):
    if band_num:
        array = sitk.GetImageFromArray(array.astype(np.float32))
        array.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(array, str(output_dir / output_name) + f'_{band_num:04}.nii.gz')
    else:
        array = sitk.GetImageFromArray(array.astype(np.uint8))
        array.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(array, str(output_dir / output_name) + '.nii.gz')


def save_x_patches_for_nnunet(x_array_patches: np.ndarray, output_dir_path: Path, scene_name: str, spacing: tuple):
    if x_array_patches.ndim != 4:
        raise ValueError(f'Expected array to have format PATCH x CHW but got {x_array_patches.shape}')
    warn_if_not_chw(x_array_patches[0])
    for i, x_array in enumerate(x_array_patches):
        for b, band in enumerate(x_array):
            save_for_nnunet(band, output_dir_path, f'{scene_name}_P{i:04}', spacing, band_num=b)


def save_y_chart_patches_for_nnunet(
    y_array_patches: dict[str, np.ndarray], output_dir_paths: dict[str, Path], scene_name: str, spacing: tuple
):
    charts = y_array_patches.keys() & output_dir_paths.keys()
    for chart in charts:
        if y_array_patches[chart].ndim != 4:
            raise ValueError(f'Expected array to have format PATCH x CHW but got {y_array_patches.shape}')
        warn_if_not_chw(y_array_patches[chart][0])
        for i, y_array in enumerate(y_array_patches[chart]):
            save_for_nnunet(y_array[0], output_dir_paths[chart], f'{scene_name}_P{i:04}', spacing)


def generate_nnunet_ds_metadata(
    output_json_path: Path, training_dir_path: Path, test_dir_path: Path | None, bands: list[str], labels: dict[int, str]
):
    generate_dataset_json(
        output_file=str(output_json_path),
        imagesTr_dir=str(training_dir_path),
        imagesTs_dir=str(test_dir_path) if test_dir_path is not None else None,
        modalities=tuple(bands),
        labels=labels,
        dataset_name='AI4Arctic',
        license='Apache-2.0 License',
        dataset_description='AI4Arctic Ice Detection Challenge Dataset',
        dataset_reference='https://platform.ai4eo.eu/auto-ice/data',
        dataset_release='0.0',
    )


def main(*, limit: int = None):
    input_dir = Path('./data/ai4arctic_challenge')
    output_dir = Path('./exports/nnunet_ds')
    datalists_path = Path('./datalists/dataset.json')

    options = get_variable_options(EXPORT_OPTIONS.copy())

    output_dir_train_path = output_dir / 'imagesTr'
    output_dir_labels_paths = {chart: output_dir / f'labelsTr{chart}' for chart in options['charts']}
    output_dir_train_path.mkdir(parents=True, exist_ok=True)
    for chart in output_dir_labels_paths:
        output_dir_labels_paths[chart].mkdir(parents=True, exist_ok=True)

    with open(datalists_path) as f:
        scene_files = json.loads(f.read())

    for scene_file in tqdm(scene_files[:limit]):
        scene_name = extract_file_name_from_file_desc(scene_file)
        scene_ds = xr.open_dataset(input_dir / f'{scene_name}.nc')
        x = extract_sample_x(scene_ds, options['sar_variables'], options['amsrenv_variables'])
        y = extract_sample_y(scene_ds, options['charts'])
        x_patches = split_to_patches(x, options['patch_size'], options['train_fill_value'])
        y_patches = {
            chart: split_to_patches(y[chart], options['patch_size'], options['class_fill_values'][chart]) for chart in y
        }
        save_x_patches_for_nnunet(x_patches, output_dir_train_path, scene_name, options['spacing'])
        save_y_chart_patches_for_nnunet(y_patches, output_dir_labels_paths, scene_name, options['spacing'])

    generate_nnunet_ds_metadata(
        output_dir / 'dataset_SIC.json', output_dir_train_path, None, SCENE_VARIABLES, SIC_GROUPS
    )
    generate_nnunet_ds_metadata(
        output_dir / 'dataset_SOD.json', output_dir_train_path, None, SCENE_VARIABLES, SOD_GROUPS
    )
    generate_nnunet_ds_metadata(
        output_dir / 'dataset_FLOE.json', output_dir_train_path, None, SCENE_VARIABLES, FLOE_GROUPS
    )


if __name__ == '__main__':
    clize.run(main)
