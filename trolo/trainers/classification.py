import time
import json
import datetime
from pathlib import Path
from typing import Optional, Union, Dict

import torch
import torch.nn as nn

from ..utils import dist_utils
from .base import BaseTrainer
from .clas_engine import train_one_epoch, evaluate


class ClassificationTrainer(BaseTrainer):
    """Classification specific trainer implementation"""
    def __init__(
        self,
        config: Optional[Union[str, Path, Dict]] = None,  # Combined config path or dict
        model: Optional[Union[str, Path, Dict]] = None,  # Model name, config path, or config dict
        dataset: Optional[Union[str, Path, Dict]] = None,  # Dataset name, config path, or config dict
        pretrained_model: Optional[Union[str, Path]] = None,  # Path to pretrained model or model name
        **kwargs
    ):
        """Initialize classification trainer.
        
        Args:
            config: Combined config - can be:
                    - Path to complete config file
                    - Complete config dictionary
            model: Model specification - can be:
                    - Model name (e.g. "resnet50")
                    - Path to model config
                    - Model config dictionary
            dataset: Dataset specification - can be:
                    - Dataset name (e.g. "imagenet")
                    - Path to dataset config
                    - Dataset config dictionary
            pretrained_model: Path to pretrained model or model name - can be:
                    - Absolute path to checkpoint file
                    - Model name to load from default location
            **kwargs: Additional config overrides
        """
        super().__init__(
            config=config,
            model=model,
            dataset=dataset,
            pretrained_model=pretrained_model,
            **kwargs
        )
        
        if not self.cfg.task == "classification":
            raise ValueError("ClassificationTrainer requires task='classification' in config")

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params:', n_parameters)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(self.model,
                                        self.criterion,
                                        self.train_dataloader,
                                        self.optimizer,
                                        self.ema,
                                        epoch=epoch,
                                        device=self.device)
            self.lr_scheduler.step()
            self.last_epoch += 1

            if output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(module, self.criterion, self.val_dataloader, self.device)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if output_dir and dist_utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
