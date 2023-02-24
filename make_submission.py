import json
import os
import tempfile
import time
from pathlib import Path

import clize
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from dvc import repo
from git import Repo
from tqdm import tqdm

from functions import chart_cbar
from loaders import AI4ArcticChallengeTestDataset, get_variable_options
from train import setup_device, setup_model
from train_options import TRAIN_OPTIONS


def setup_dataset(train_options) -> torch.utils.data.DataLoader:
    with open(train_options['path_to_env'] + 'datalists/testset.json') as file:
        train_options['test_list'] = json.loads(file.read())

    train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in train_options['test_list']]
    train_options['path_to_processed_data'] += '_test'

    dataset = AI4ArcticChallengeTestDataset(options=train_options, files=train_options['test_list'], test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False
    )
    return dataloader


def make_submission(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    train_options: dict,
    previews_dir: Path | None = None,
) -> xr.Dataset:
    upload_package = xr.Dataset()

    os.makedirs('inference', exist_ok=True)
    model.eval()

    for x, _, masks, scene_name in tqdm(
        iterable=dataloader, total=len(train_options['test_list']), colour='green', position=0
    ):
        scene_name = scene_name[:19]  # Removes the _prep.nc from the name.

        torch.cuda.empty_cache()
        x = x.to(device, non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(x)

        for chart in train_options['charts']:
            output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
            upload_package[f"{scene_name}_{chart}"] = xr.DataArray(
                name=f"{scene_name}_{chart}",
                data=output[chart].astype('uint8'),
                dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"),
            )

        if previews_dir is not None:
            make_preview(output, masks, scene_name, previews_dir, train_options)

    return upload_package


def make_preview(output: np.ndarray, masks: np.ndarray, scene_name: str, output_dir: Path, train_options):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    for idx, chart in enumerate(train_options['charts']):
        ax = axs[idx]
        output[chart] = output[chart].astype(float)
        output[chart][masks] = np.nan
        ax.imshow(
            output[chart], vmin=0, vmax=train_options['n_classes'][chart] - 2, cmap='jet', interpolation='nearest'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

    plt.suptitle(f'Scene: {scene_name}', y=0.65)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
    fig.savefig(output_dir / f'{scene_name}.png', format='png', dpi=128, bbox_inches='tight')
    plt.close('all')


def dump_meta(output_dir: Path, model_url: str):
    gitrepo = Repo()
    meta = {
        'model_url': model_url,
        'commit': str(gitrepo.rev_parse('HEAD')),
        'repo_dirty': gitrepo.is_dirty(),
        'timestamp': time.time(),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f)


def main(
    model_url: str, *, output_dir: Path = Path('exports'), no_previews: bool = False, force_cpu_device: bool = False
):
    output_dir.mkdir(exist_ok=True, parents=True)

    train_options = get_variable_options(TRAIN_OPTIONS)
    print('Options initialised')

    model, model_options = setup_model(train_options)
    print('Model initialised')
    device = setup_device(train_options) if not force_cpu_device else torch.device('cpu')
    print('Device initialised')
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_file = Path(tmp_dir) / 'model_weights.pt'
        repo.get_url.get_url(model_url, out=weights_file)
        model.load_state_dict(torch.load(weights_file, map_location=device)['model_state_dict'])
        model.to(device)
    print('Model weights loaded')

    dataloader = setup_dataset(train_options)
    print('Dataset initialized')

    if no_previews:
        previews_dir = None
    else:
        previews_dir = output_dir / 'inference_previews'
        previews_dir.mkdir(exist_ok=True, parents=True)

    print('Compiling submission')
    upload_package = make_submission(model, device, dataloader, train_options, previews_dir=previews_dir)
    print('Submission compiled')

    print('Saving submission')
    compression = {'zlib': True, 'complevel': 1}
    encoding = {var: compression for var in upload_package.data_vars}
    nc_file_path = output_dir / 'upload_package.nc'
    upload_package.to_netcdf(nc_file_path, mode='w', format='netcdf4', engine='netcdf4', encoding=encoding)
    print(f'Saved to {nc_file_path}')

    dump_meta(output_dir, model_url)
    print('Submission creation succeeded')


if __name__ == '__main__':
    clize.run(main)
