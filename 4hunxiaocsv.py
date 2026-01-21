import os
import argparse
import multiprocessing as mp
import pprint
import yaml

from src.utils.distributed import init_distributed
from evals.scaffold_混淆矩阵 import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='hxjz.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Automatically fill dataset_train and dataset_val
    current_directory = os.getcwd()
    params['data']['dataset_train'] = os.path.join(current_directory, 'train_video.csv')
    params['data']['dataset_val'] = os.path.join(current_directory, 'test_video.csv')

    # Set the folder for checkpoints
    params['pretrain']['folder'] = os.path.join(current_directory, 'checkpoints')

    # Log the updated parameters
    logger.info('Updated params with dataset paths and checkpoints folder:')
    pp.pprint(params)

    # Init distributed
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()