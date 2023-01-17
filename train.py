# -- Built-in modules -- #
import gc

# -- Third-part modules -- #
import json
import os

# -- Environmental variables -- #
os.environ['AI4ARCTIC_DATA'] = './data/ai4arctic_challenge'  # Fill in directory for data location.
os.environ['AI4ARCTIC_ENV'] = './'  # Fill in directory for environment with Ai4Arctic get-started package.

import numpy as np
import torch
from tqdm.notebook import tqdm  # Progress bar

# --Proprietary modules -- #
from functions import (  # Functions to calculate metrics and show the relevant chart colorbar.
    compute_metrics,
)
from loaders import (  # Custom dataloaders for regular training and validation.
    AI4ArcticChallengeDataset,
    AI4ArcticChallengeTestDataset,
    get_variable_options,
)
from train_options import TRAIN_OPTIONS
from unet import UNet  # Convolutional Neural Network model
from utils import colour_str


def setup_device(train_options: dict) -> torch.device:
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
    return device


def setup_options(options: dict) -> dict:
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(options)

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


def setup_model(train_options: dict) -> torch.nn.Module:
    model = UNet(options=train_options)
    return model


def main():
    train_options = setup_options(TRAIN_OPTIONS)
    print('Options initialised')

    device = setup_device(train_options)
    print('Device initialised')

    dataloader, dataloader_val = setup_dataset(train_options)
    print('Data setup complete')

    # Setup U-Net model, adam optimizer, loss function and dataloader.
    net = setup_model(train_options).to(device)
    optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
    torch.backends.cudnn.benchmark = (
        True  # Selects the kernel with the best performance for the GPU and given input size.
    )
    print('Model setup complete.')

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
