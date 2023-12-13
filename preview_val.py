# -- Built-in modules -- #
import gc

# -- Third-part modules -- #
import json
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import clize
import matplotlib
import mlflow
import numpy as np
import torch
from dvc.repo.get_url import get_url
from matplotlib import pyplot as plt
from torchinfo import summary
from tqdm import tqdm  # Progress bar

# --Proprietary modules -- #
from functions import (  # Functions to calculate metrics and show the relevant chart colorbar.
    chart_cbar,
    compute_metrics,
)
from ice_transformer import IceTransformer
from loaders import (  # Custom dataloaders for regular training and validation.
    AI4ArcticChallengeDataset,
    AI4ArcticChallengeTestDataset,
    get_variable_options,
)
from train import setup_dataset, setup_device, setup_model, setup_options
from train_options import TRAIN_OPTIONS, TRANSFORMER_MODEL_OPTIONS, UNET_MODEL_OPTIONS
from unet import UNet  # Convolutional Neural Network model
from utils import ICE_STRINGS, colour_str

MLFLOW_URI = 'http://misc.mlflow.kplabs.pl'
EXPERIMENT_NAME = 'AI4Artctic challenge'

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    }
)


def make_preview(
    x: np.ndarray, output: np.ndarray, masks: np.ndarray, scene_name: str, output_dir: Path, train_options
):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 2))
    axs = axs.flatten()

    ax = axs[0]
    inp = x[0][0].detach().numpy()
    inp[masks['SIC']] = np.nan
    imshow_out = ax.imshow(inp, cmap='gray', vmin=np.nanquantile(inp, q=0.025), vmax=np.nanquantile(inp, q=0.975))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Sentinel-1 SAR primary')
    plt.colorbar(imshow_out, fraction=0.043, pad=0.049, ax=ax)

    for idx, chart in enumerate(train_options['charts'], start=1):
        ax = axs[idx]
        output[chart] = output[chart].astype(float)
        output[chart][masks[chart]] = np.nan
        ax.imshow(
            output[chart], vmin=0, vmax=train_options['n_classes'][chart] - 2, cmap='jet', interpolation='nearest'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ICE_STRINGS[chart])
        chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
    fig.savefig(output_dir / f'{scene_name}.png', format='png', dpi=180, bbox_inches='tight')
    plt.close('all')


def train(
    train_options: dict,
    model_options: dict,
    datalader_val: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    preview_out_dir: Path,
):
    model = model.to(device)
    in_shape = (
        train_options['batch_size'],
        len(train_options['train_variables']),
        train_options['patch_size'],
        train_options['patch_size'],
    )
    summary(model, input_size=in_shape)

    # -- Validation Loop -- #

    # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
    outputs_flat = {chart: np.array([], dtype=np.float32) for chart in train_options['charts']}
    inf_ys_flat = {chart: np.array([], dtype=np.float32) for chart in train_options['charts']}

    model.eval()  # Set network to evaluation mode.
    # - Loops though scenes in queue.
    for inf_x, inf_y, masks, name in tqdm(
        iterable=datalader_val,
        total=len(train_options['validate_list']),
        colour='green',
        position=0,
        desc='Validation',
    ):
        torch.cuda.empty_cache()

        # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if device == 'cuda' else False):
            inf_x = inf_x.to(device, non_blocking=True)
            output = model(inf_x)

        # - Final output layer, and storing of non masked pixels.
        for chart in train_options['charts']:
            output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy().astype(np.float32)
            outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
            inf_y[chart] = inf_y[chart].cpu().numpy().astype(np.float32)

        make_preview(inf_x, output, masks, name, preview_out_dir, train_options)
        make_preview(inf_x, inf_y, masks, name + '_gt', preview_out_dir, train_options)

        del inf_x, inf_y, masks, output  # Free memory.


def main(preview_out_dir: Path, ckpt_url: str, *, force_cpu_device: bool = False):
    train_options = setup_options(TRAIN_OPTIONS)
    print('Options initialised')

    device = setup_device(train_options) if not force_cpu_device else torch.device('cpu')
    print('Device initialised')

    _, dataloader_val = setup_dataset(train_options)
    print('Data setup complete')

    preview_out_dir.mkdir(exist_ok=True, parents=True)
    model, model_options = setup_model(train_options)

    with tempfile.NamedTemporaryFile() as ckpt_file:
        get_url(ckpt_url, ckpt_file.name, force=True)
        state_dict = torch.load(ckpt_file.name, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])

    print('Model setup complete')
    train(train_options, model_options, dataloader_val, model, device, preview_out_dir),


if __name__ == '__main__':
    clize.run(main)
