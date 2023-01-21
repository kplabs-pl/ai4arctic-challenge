# -- Built-in modules -- #
import gc

# -- Third-part modules -- #
import json
import os
from tempfile import TemporaryDirectory

import clize
import mlflow
import numpy as np
import torch
from tqdm import tqdm  # Progress bar

# --Proprietary modules -- #
from functions import (  # Functions to calculate metrics and show the relevant chart colorbar.
    compute_metrics,
)
from ice_transformer import IceTransformer
from loaders import (  # Custom dataloaders for regular training and validation.
    AI4ArcticChallengeDataset,
    AI4ArcticChallengeTestDataset,
    get_variable_options,
)
from train_options import TRAIN_OPTIONS, TRANSFORMER_MODEL_OPTIONS, UNET_MODEL_OPTIONS
from unet import UNet  # Convolutional Neural Network model
from utils import colour_str

MLFLOW_URI = 'http://misc.mlflow.kplabs.pl'
EXPERIMENT_NAME = 'AI4Artctic challenge'


def setup_device(train_options: dict) -> torch.device:
    if torch.has_cuda:
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
        device = torch.device(f"cuda:{train_options['gpu_id']}")
    elif torch.has_mps:
        print(colour_str('MPS available!', 'green'))
        device = torch.device('mps')
    else:
        print(colour_str('GPU not available', 'red'))
        device = torch.device('cpu')
    return device


def setup_options(options: dict, short_training: bool = False) -> dict:
    options = options.copy()

    # Overwrite options for short training
    if short_training:
        options['epochs'] = 1
        options['epoch_len'] = 5
        options['num_val_scenes'] = 1

    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(options.copy())

    # Load training list.
    with open(train_options['path_to_env'] + 'datalists/dataset.json') as file:
        train_options['train_list'] = json.loads(file.read())
    # Convert the original scene names to the preprocessed names.
    train_options['train_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in train_options['train_list']]
    # Select a random number of validation scenes with the same seed. Feel free to change the seed.et
    np.random.seed(0)
    train_options['validate_list'] = np.random.choice(
        np.array(train_options['train_list']), size=train_options['num_val_scenes'], replace=False
    )
    # Remove the validation scenes from the train list.
    train_options['train_list'] = [
        scene for scene in train_options['train_list'] if scene not in train_options['validate_list']
    ]
    return train_options


def setup_dataset(train_options) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns a tuple of train and validation dataloaders."""
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(files=train_options['train_list'], options=train_options)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True
    )
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.
    dataset_val = AI4ArcticChallengeTestDataset(options=train_options, files=train_options['validate_list'])
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False
    )
    return dataloader, dataloader_val


def setup_model(train_options: dict) -> tuple[torch.nn.Module, dict]:
    model: torch.nn.Module
    model_options: dict
    match train_options['model']:
        case 'unet':
            model_options = UNET_MODEL_OPTIONS.copy()
            model = UNet(options=train_options | model_options)
        case 'ice_transformer':
            model_options = TRANSFORMER_MODEL_OPTIONS.copy()
            model = IceTransformer(len(train_options['train_variables']), model_options['internal_patch_size'])
    return model, model_options


def prepare_train_options_for_logging(train_options: dict) -> dict:
    ret = train_options.copy()
    del ret['train_variables']
    return ret


def main(run_name: str, *, remote_mlflow: bool = False, force_cpu_device: bool = False, short_training: bool = False):
    """Train ice detection network.

    :param run_name: Name of the experiment run (will be used for logging, artifacts, etc.)
    :param remote_mlflow: Enable remote mlflow logging
    :param force_cpu_device: Use CPU device (default behaviour is autodetect GPU/MPS/fallback to CPU)
    :param short_training: Overwrite settings for a smaller num of epochs (useful for sanity runs)
    """
    train_options = setup_options(TRAIN_OPTIONS, short_training=short_training)
    print('Options initialised')

    device = setup_device(train_options) if not force_cpu_device else torch.device('cpu')
    print('Device initialised')

    dataloader, dataloader_val = setup_dataset(train_options)
    print('Data setup complete')

    # Setup U-Net model, adam optimizer, loss function and dataloader.
    net, model_options = setup_model(train_options)
    net = net.to(device)
    optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
    torch.backends.cudnn.benchmark = (
        True  # Selects the kernel with the best performance for the GPU and given input size.
    )
    print('Model setup complete')

    # Loss functions to use for each sea ice parameter.
    # The ignore_index argument discounts the masked values, ensuring that the model is not using these pixels to train
    # on. It is equivalent to multiplying the loss of the relevant masked pixel with 0.
    loss_functions = {
        chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart])
        for chart in train_options['charts']
    }
    print('Training setup complete')

    best_combined_score = float('-inf')  # Best weighted model score.

    if remote_mlflow:
        mlflow.set_tracking_uri(MLFLOW_URI)

    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name), TemporaryDirectory() as artifacts_tmp_dir:
        mlflow.log_params(prepare_train_options_for_logging(TRAIN_OPTIONS) | model_options)
        best_model_val_epoch = 0
        best_model_artifact_path = os.path.join(artifacts_tmp_dir, 'ice_best_model.pt')

        # -- Training Loop -- #
        for epoch in tqdm(range(train_options['epochs']), position=0, desc='Epoch'):
            gc.collect()  # Collect garbage to free memory.
            loss_sum = torch.tensor([0.0])  # To sum the batch losses during the epoch.
            net.train()  # Set network to evaluation mode.
            net.to(device)

            # Loops though batches in queue.
            for i, (batch_x, batch_y) in enumerate(
                (
                    progress := tqdm(
                        iterable=dataloader, total=train_options['epoch_len'], colour='red', position=0, desc='Batch'
                    )
                )
            ):
                torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
                loss_batch = torch.Tensor([0]).to(device)  # Reset from previous batch.

                # - Transfer to device.
                batch_x = batch_x.to(device, non_blocking=True)

                # - Mixed precision training. (Saving memory)
                with torch.cuda.amp.autocast(enabled=True if device == 'cuda' else False):
                    # - Forward pass.
                    output = net(batch_x)

                    # - Calculate loss.
                    for chart in train_options['charts']:
                        loss_batch += loss_functions[chart](input=output[chart], target=batch_y[chart].to(device))

                # - Reset gradients from previous pass.
                optimizer.zero_grad()

                # - Backward pass.
                loss_batch.backward()

                # - Optimizer step
                optimizer.step()

                # - Add batch loss.
                loss_sum += loss_batch.detach().item()

                # - Average loss for displaying
                loss_epoch = torch.true_divide(loss_sum, i + 1).detach().item()
                progress.set_description(f'Mean training loss: {loss_epoch:.3f}, batch')
                mlflow.log_metric('batch_loss', loss_batch, step=epoch * train_options['epoch_len'] + i)
                mlflow.log_metric('mean_batch_loss_in_epoch', loss_epoch, step=epoch * train_options['epoch_len'] + i)
                del output, batch_x, batch_y  # Free memory

            loss_batch_float = loss_batch.detach().item()  # For printing after the validation loop.
            print(f'Epoch last batch loss: {loss_batch_float:.3f}')
            print(f'Mean epoch loss: {loss_epoch:.3f}')
            mlflow.log_metric('epoch_last_batch_loss', loss_batch_float, step=epoch)
            mlflow.log_metric('mean_epoch_loss', loss_epoch, step=epoch)
            del loss_sum

            # -- Validation Loop -- #

            # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
            outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
            inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

            net.eval()  # Set network to evaluation mode.
            # - Loops though scenes in queue.
            for inf_x, inf_y, masks, name in tqdm(
                iterable=dataloader_val,
                total=len(train_options['validate_list']),
                colour='green',
                position=0,
                desc='Validation',
            ):
                torch.cuda.empty_cache()

                # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if device == 'cuda' else False):
                    inf_x = inf_x.to(device, non_blocking=True)
                    output = net(inf_x)

                # - Final output layer, and storing of non masked pixels.
                for chart in train_options['charts']:
                    output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
                    outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
                    inf_ys_flat[chart] = np.append(inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())

                del inf_x, inf_y, masks, output  # Free memory.

            # - Compute the relevant scores.
            combined_score, scores = compute_metrics(
                true=inf_ys_flat,
                pred=outputs_flat,
                charts=train_options['charts'],
                metrics=train_options['chart_metric'],
            )

            print(f'\nEpoch {epoch} score:')
            for chart in train_options['charts']:
                print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")
            print(f'Combined score: {combined_score}%\n')
            mlflow.log_metrics({f'{chart}_val': scores[chart] for chart in scores}, step=epoch)
            mlflow.log_metric('epoch_combined_score_val', combined_score, step=epoch)

            # If the scores is better than the previous epoch, then save the u and rename the image to best_validation.
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_model_val_epoch = epoch
                torch.save(
                    {
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                    },
                    best_model_artifact_path,
                )
            del inf_ys_flat, outputs_flat  # Free memory.
        mlflow.log_metric('best_model_val_epoch', best_model_val_epoch)
        mlflow.log_artifact(best_model_artifact_path)


if __name__ == '__main__':
    clize.run(main)
