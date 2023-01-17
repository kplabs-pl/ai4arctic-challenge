# -- Built-in modules -- #
import gc

# -- Third-part modules -- #
import json
import os

import numpy as np
import torch
from tqdm.notebook import tqdm  # Progress bar

# --Proprietary modules -- #
from functions import (  # Functions to calculate metrics and show the relevant chart colorbar.
    compute_metrics,
    f1_metric,
    r2_metric,
)
from loaders import (  # Custom dataloaders for regular training and validation.
    AI4ArcticChallengeDataset,
    AI4ArcticChallengeTestDataset,
    get_variable_options,
)
from unet import UNet  # Convolutional Neural Network model
from utils import (
    CHARTS,
    FLOE_LOOKUP,
    SCENE_VARIABLES,
    SIC_LOOKUP,
    SOD_LOOKUP,
    colour_str,
)

# -- Environmental variables -- #
os.environ['AI4ARCTIC_DATA'] = './data/ai4arctic_challenge'  # Fill in directory for data location.
os.environ['AI4ARCTIC_ENV'] = './'  # Fill in directory for environment with Ai4Arctic get-started package.


def main():
    train_options = {
        # -- Training options -- #
        'path_to_processed_data': os.environ['AI4ARCTIC_DATA'],  # Replace with data directory path.
        'path_to_env': os.environ['AI4ARCTIC_ENV'],  # Replace with environmment directory path.
        'lr': 0.0001,  # Optimizer learning rate.
        'epochs': 50,  # Number of epochs before training stop (default 50).
        'epoch_len': 500,  # Number of batches for each epoch (default 500).
        'patch_size': 256,  # Size of patches sampled. Used for both Width and Height (default 256).
        'batch_size': 8,  # Number of patches for each batch.
        'loader_upsampling': 'nearest',  # How to upscale low resolution variables to high resolution.
        # -- Data prepraration lookups and metrics.
        'train_variables': SCENE_VARIABLES,  # Contains the relevant variables in the scenes.
        'charts': CHARTS,  # Charts to train on.
        'n_classes': {  # number of total classes in the reference charts, including the mask.
            'SIC': SIC_LOOKUP['n_classes'],
            'SOD': SOD_LOOKUP['n_classes'],
            'FLOE': FLOE_LOOKUP['n_classes'],
        },
        'pixel_spacing': 80,  # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
        'train_fill_value': 0,  # Mask value for SAR training data.
        'class_fill_values': {  # Mask value for class/reference data.
            'SIC': SIC_LOOKUP['mask'],
            'SOD': SOD_LOOKUP['mask'],
            'FLOE': FLOE_LOOKUP['mask'],
        },
        # -- Validation options -- #
        'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
            'SIC': {
                'func': r2_metric,
                'weight': 2,
            },
            'SOD': {
                'func': f1_metric,
                'weight': 2,
            },
            'FLOE': {
                'func': f1_metric,
                'weight': 1,
            },
        },
        'num_val_scenes': 10,  # Number of scenes randomly sampled from train_list to use in validation.
        # -- GPU/cuda options -- #
        'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
        'num_workers': 4,  # Number of parallel processes to fetch data.
        'num_workers_val': 1,  # Number of parallel processes during validation.
        # -- U-Net Options -- #
        'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
        'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
        'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
        'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
        'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
        'conv_padding_style': 'zeros',  # Style of padding.
    }
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)

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
    print('Options initialised')

    # Get GPU resources.
    if torch.has_cuda:
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
        device = torch.device(f"cuda:{train_options['gpu_id']}")
    elif torch.has_mps:
        print(colour_str('MPS available!', 'green'))
        device = torch.device('mps')
    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')

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

    print('GPU and data setup complete.')

    # Setup U-Net model, adam optimizer, loss function and dataloader.
    net = UNet(options=train_options).to(device)
    optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
    torch.backends.cudnn.benchmark = (
        True  # Selects the kernel with the best performance for the GPU and given input size.
    )

    # Loss functions to use for each sea ice parameter.
    # The ignore_index argument discounts the masked values, ensuring that the model is not using these pixels to train
    # on. It is equivalent to multiplying the loss of the relevant masked pixel with 0.
    loss_functions = {
        chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart])
        for chart in train_options['charts']
    }
    print('Model setup complete')

    best_combined_score = 0  # Best weighted model score.

    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(train_options['epochs']), position=0):
        gc.collect()  # Collect garbage to free memory.
        loss_sum = torch.tensor([0.0])  # To sum the batch losses during the epoch.
        net.train()  # Set network to evaluation mode.

        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(
            tqdm(iterable=dataloader, total=train_options['epoch_len'], colour='red', position=0)
        ):
            torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            loss_batch = 0  # Reset from previous batch.

            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():
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
            print('\rMean training loss: ' + f'{loss_epoch:.3f}', end='\r')
            del output, batch_x, batch_y  # Free memory
        del loss_sum

        # -- Validation Loop -- #
        loss_batch = loss_batch.detach().item()  # For printing after the validation loop.

        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
        inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

        net.eval()  # Set network to evaluation mode.
        # - Loops though scenes in queue.
        for inf_x, inf_y, masks, name in tqdm(
            iterable=dataloader_val, total=len(train_options['validate_list']), colour='green', position=0
        ):
            torch.cuda.empty_cache()

            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
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
            true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'], metrics=train_options['chart_metric']
        )

        print("")
        print(f"Final batch loss: {loss_batch:.3f}")
        print(f"Epoch {epoch} score:")
        for chart in train_options['charts']:
            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")
        print(f"Combined score: {combined_score}%")

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save(
                obj={
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                },
                f='best_model',
            )
        del inf_ys_flat, outputs_flat  # Free memory.


if __name__ == '__main__':
    main()
