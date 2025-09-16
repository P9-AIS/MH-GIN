import sys
import os
import copy
import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
import pathlib
import datetime
import yaml
import psutil
import pynvml
import time
from threading import Thread, Event

current_path = os.path.dirname(os.path.abspath(__file__))
parent_current_path = os.path.dirname(current_path)
root_path = parent_current_path
sys.path.append(root_path)
from src.models.MTSHGNN import MtsHGnn
# from src.data.danish_data_process_module import process_ais_multi_csv_dataset, \
#     hyperparameter_DataProcess
from src.data.american_data_process_module import process_ais_multi_csv_dataset, \
    hyperparameter_DataProcess
from src.pipeline.experiment import Experiment
import src
from src.pipeline.predict import predict
from src.pipeline.train import train
from src.data.AISDataset import AISDataset
from src.utils.file_util import split_dict
from src.utils.utils import seed_everything, count_trainable_parameters
from src.utils import parser_utils
import src.data.preprocess_data as pcd
from src.models.M2MutipleScaleMining import ESNModel
from src.logging.logger import setup_logger

def monitor_cpu_memory(stop_event, peak_memory):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        current_mem = process.memory_info().rss
        if current_mem > peak_memory[0]:
            peak_memory[0] = current_mem
        time.sleep(0.001) 

def monitor_gpu_utilization(device=0, interval=0.1):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    utils = []
    stop_event = Event()
    
    def monitor():
        while not stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            utils.append(util)
            time.sleep(interval)
    
    monitor_thread = Thread(target=monitor)
    monitor_thread.start()
    return utils, stop_event, monitor_thread

def configure_parser():
    """Configure command line arguments for the experiment"""
    parser = argparse.ArgumentParser(description="MTS-HGNN")
    
    # Base configuration
    base_group = parser.add_argument_group('Base Configuration')
    base_group.add_argument('--seed', type=int, default=-1,
                          help='Random seed for reproducibility')

    # Dataset parameters
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group = hyperparameter_DataProcess(data_group)  # Inherit data processing params
    data_group.add_argument('--max-len', type=int, default=8000,
                          help='Maximum sequence length for trajectory data')
    data_group.add_argument('--mask-rate', type=float, default=0.1)
    data_group.add_argument('--workers', type=int, default=0)
    data_group.add_argument('--batch-size', type=int, default=32)
    data_group.add_argument('--batch-test', type=int, default=16)
    data_group.add_argument('--dataset_name', type=str, default='danish_ais')

    # Model architecture parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_name', type=str, default='MTS-HGNN',
                           help='Name of the model architecture')
    model_group.add_argument('--d-data-model', type=int, default=128)
    model_group.add_argument('--d-mining-model', type=int, default=128)
    model_group.add_argument('--output-size', type=int, default=128)
    model_group.add_argument('--graph-mask-values', type=list, default=[4, 8])

    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training from')
    train_group.add_argument('--device', type=str, default='cuda:0',
                           help='Device for training (cpu/cuda)')
    train_group.add_argument('--epochs', type=int, default=100)
    train_group.add_argument('--lr', type=float, default=0.001)
    train_group.add_argument('--weight-decay', type=float, default=1e-6)
    train_group.add_argument('--use-lr-schedule', type=parser_utils.str_to_bool, nargs='?',
                        const=True, default=True)
    train_group.add_argument('--grad-clip-val', type=float, default=5.)

    train_group.add_argument('--valid-epoch-interval', type=int, default=50)
    train_group.add_argument('--patience', type=int, default=5) # 2025-02-23_05-20-57_42
    train_group.add_argument('--pretrained_model', type=str, default='')

    # Add ESN specific parameters
    ESNModel.add_model_specific_args(parser)
    return parser

def run_experiment(args):
    """Main experiment execution flow"""
    # Deep copy arguments to prevent side effects
    args = copy.deepcopy(args)  
    
    # Initialize random seeds
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(max(args.workers, 1))
    seed_everything(args.seed)
    
    # Configure experiment paths
    if args.pretrained_model == '':
        # Training mode setup
        exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
        logdir = os.path.join(src.config['logs_dir'],
                            args.dataset_name,
                            args.model_name,
                            exp_name)
        pathlib.Path(logdir).mkdir(parents=True)
        print(f"logdir: {logdir}")
        
        # Save experiment configuration
        with open(os.path.join(logdir, 'exp_config.yaml'), 'w') as fp:
            yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4,
                    sort_keys=True)
        logger = setup_logger(log_dir=logdir)
        logger.info(f'SEED: {args.seed}')
        logger.info(args)
    else:
        # Inference mode setup
        logdir = os.path.join(src.config['logs_dir'],
                          args.dataset_name,
                          args.model_name,
                          args.pretrained_model)
        logger = setup_logger(log_dir=logdir, train_mode=False)

    # Data preparation pipeline
    traj_data, stats = process_ais_multi_csv_dataset(args)
    for traj_id, traj in traj_data.items():
        adjusted_timestamps = np.array(traj['timestamp']) - 1704067200
        if np.any(adjusted_timestamps < 0):
            logger.warning(f"Negative timestamps found in trajectory {traj_id} after adjustment")
        traj_data[traj_id]['timestamp'] = adjusted_timestamps
        
    train_traj_data, test_traj_data = split_dict(traj_data)
    
    # Process training data
    processed_data, labels, masks, padding_masks = pcd.preprocess_traj_data_v2(
        train_traj_data, stats, args.max_len, args.mask_rate
    )
    train_loader = DataLoader(
        AISDataset(processed_data, labels, masks, padding_masks),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    
    # Process test data
    processed_data, labels, masks, padding_masks = pcd.preprocess_traj_data_v2(
        test_traj_data, stats, args.max_len, args.mask_rate
    )
    test_loader = DataLoader(
        AISDataset(processed_data, labels, masks, padding_masks),
        batch_size=args.batch_test,
        shuffle=True,
        num_workers=args.workers
    )
    
    # Model initialization
    model = MtsHGnn(
        logger=logger,
        device=args.device,
        dataset_name=args.dataset_name,
        d_data_model=args.d_data_model,
        data_status=stats,
        d_mining_model=args.d_mining_model,
        hidden_size=args.reservoir_size,
        output_size=args.output_size,
        horizon=args.horizon,
        num_layers=args.reservoir_layers,
        graph_mask_values=args.graph_mask_values
    ).to(args.device)

    # Execution flow control
    if args.pretrained_model == '':
        total_params = count_trainable_parameters(model)
        logger.info(f"Total trainable parameters: {total_params}")
        print(f"exp_name: {exp_name}")
        logger.info(f"exp_name: {exp_name}")
        result_path = os.path.join(src.config['result_dir'],
                               args.dataset_name,
                               args.model_name,
                               exp_name)
        train(model, args, logger, train_loader, test_loader, folder_name=result_path)
    else:
        # Inference phase
        pretrained_path = os.path.join(src.config['result_dir'],
                                     args.dataset_name,
                                     args.model_name)
        model.load_state_dict(torch.load(
            f"{pretrained_path}/{args.pretrained_model}/model.pth",
            map_location=args.device
        ))
        peak_cpu_memory = [0]
        stop_cpu_event = Event()
        cpu_thread = Thread(target=monitor_cpu_memory, args=(stop_cpu_event, peak_cpu_memory))
        cpu_thread.start()

        gpu_utils, gpu_stop_event, gpu_thread = monitor_gpu_utilization(device=0)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_mem = torch.cuda.memory_allocated()

        predict(model, test_loader, logger)
        stop_cpu_event.set()
        cpu_thread.join()

        peak_cpu_mb = peak_cpu_memory[0] / (1024 ** 2)
        logger.info(f"Peak CPU Memory Usage: {peak_cpu_mb:.2f} MB")

        if torch.cuda.is_available():
            gpu_stop_event.set()
            gpu_thread.join()
            peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"Peak GPU Memory Usage: {peak_gpu_mem:.2f} MB")
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
            logger.info(f"Average GPU Utilization: {avg_gpu_util:.1f}%")


if __name__ == "__main__":
    experiment_parser = configure_parser()
    exp = Experiment(run_fn=run_experiment, parser=experiment_parser,
                     config_path=src.config['config_dir'])
    exp.run()


