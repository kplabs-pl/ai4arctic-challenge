from functions import f1_metric, r2_metric
from utils import CHARTS, FLOE_LOOKUP, SCENE_VARIABLES, SIC_LOOKUP, SOD_LOOKUP

TRAIN_OPTIONS = {
    # -- Training options -- #
    'path_to_processed_data': './data/ai4arctic_challenge/',  # Replace with data directory path.
    'path_to_env': './',  # Replace with environmment directory path.
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
    # -- Model selection -- #
    'model': 'unet',  # Model architecture ('unet' or `ice_transformer')
}

UNET_MODEL_OPTIONS = {
    'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
}

TRANSFORMER_MODEL_OPTIONS = {
    'internal_patch_size': 64,
}
