"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

from trolo.utils import dist_utils
from trolo.loaders import YAMLConfig, yaml_utils
from trolo.trainers import TASKS
from trolo.utils.smart_defaults import infer_device
from trolo.loaders.maps import get_dataset_config_path, get_model_config_path

debug=False

if debug:
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def init_distributed_mode(device='cpu'):
    """Initialize distributed training based on available hardware."""
    if not torch.distributed.is_available():
        print("Warning: Distributed package not available. Running in non-distributed mode.")
        return

    try:
        if torch.cuda.is_available():
            backend = "nccl"  # NCCL backend for GPU
        else:
            backend = "gloo"  # Gloo backend for CPU

        # Initialize process group
        torch.distributed.init_process_group(
            backend=backend,
            init_method="tcp://localhost:12345",
            rank=0,
            world_size=1
        )
    except Exception as e:
        print(f"Warning: Distributed training initialization failed: {e}")
        print("Running in non-distributed mode.")

def train_model(config: str,
         resume: str = None,
         tuning: str = None,
         device: str = None,
         seed: int = None,
         use_amp: bool = False,
         output_dir: str = None,
         summary_dir: str = None,
         test_only: bool = False,
         update: list = None,
         print_method: str = 'builtin',
         print_rank: int = 0,
         local_rank: int = None) -> None:
    """Main training function for D-FINE models.
    
    Args:
        config: Path to YAML config file
        resume: Path to checkpoint to resume training from
        tuning: Path to checkpoint to tune from
        device: Device to run on ('cpu', 'cuda:0', etc)
        seed: Random seed for reproducibility
        use_amp: Whether to use automatic mixed precision training
        output_dir: Directory to save outputs
        summary_dir: Directory for tensorboard summary
        test_only: If True, only run validation
        update: List of YAML config updates
        print_method: Print method to use ('builtin', etc)
        print_rank: Rank ID to print from
        local_rank: Local rank ID for distributed training
    """
    # Only setup distributed training if CUDA is available
    if torch.cuda.is_available():
        dist_utils.setup_distributed(print_rank, print_method, seed=seed)

    assert not all([tuning, resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # Infer device if not specified
    if device is None:
        device = infer_device()

    init_distributed_mode(device)

    update_dict = yaml_utils.parse_cli(update) if update else {}
    update_dict.update({k: v for k, v in locals().items() \
        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(config, **update_dict)

    if resume or tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    try:
        if test_only:
            solver.val()
        else:
            solver.fit()
    except Exception as e:
        print(f"Training error: {e}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(**vars(args))