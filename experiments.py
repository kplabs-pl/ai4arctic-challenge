import itertools

import clize
from tqdm import tqdm

from train import setup_dataset, setup_device, setup_model, setup_options, train
from train_options import TRAIN_OPTIONS

ALLOWED_EXPERIMENTS = clize.parameters.one_of('baseline_grid')


def generate_experiments_options_grid(options_base: dict, options_overrides: dict) -> list[dict]:
    options_base = options_base.copy()
    keys, values = zip(*options_overrides.items())
    options_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    experiments = []
    for combination in options_combinations:
        exp = options_base.copy()
        exp.update(combination)
        experiments.append(exp)
    return experiments


def baseline_grid():
    options_grid = {
        'charts': [['SIC', 'SOD', 'FLOE'], ['SIC'], ['SOD'], ['FLOE']],
        'model': ['unet', 'ice_transformer'],
    }
    experiments_options_list = generate_experiments_options_grid(TRAIN_OPTIONS, options_grid)

    print(f'Grid evaluates into {len(experiments_options_list)} experiments')
    for train_options in (progress := tqdm(experiments_options_list)):
        run_name = f'Baseline {train_options["model"]} with charts: {train_options["charts"]}'
        print(f'Running experiment: {run_name}')
        progress.set_description(run_name)

        train_options = setup_options(TRAIN_OPTIONS)
        print('Options initialised')

        device = setup_device(train_options)
        print('Device initialised')

        dataloader, dataloader_val = setup_dataset(train_options)
        print('Data setup complete')

        model, model_options = setup_model(train_options)
        print('Model setup complete')
        train(
            run_name,
            train_options,
            model_options,
            dataloader,
            dataloader_val,
            model,
            device,
            remote_mlflow=True,
        ),


def main(experiment_name: ALLOWED_EXPERIMENTS):
    match experiment_name:
        case 'baseline_grid':
            baseline_grid()


if __name__ == '__main__':
    clize.run(main)
