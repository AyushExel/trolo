import time
import json
import datetime
import subprocess
import os
import sys
import yaml
import socket
import torch

from ..utils import dist_utils, stats

from .base import BaseTrainer
from .det_engine import train_one_epoch, evaluate

from pathlib import Path
from typing import Union, Optional, Dict, List, Any
from trolo.utils.logging import WandbLogger, ExperimentLogger
from datetime import datetime, timedelta

class DetectionTrainer(BaseTrainer):
    """Detection specific trainer implementation"""
    def __init__(
        self, 
        config: Optional[Union[str, Path, Dict]] = None,
        model: Optional[Union[str, Path, Dict]] = None,
        dataset: Optional[Union[str, Path, Dict]] = None,
        pretrained_model: Optional[Union[str, Path]] = None,
        loggers: Optional[List[ExperimentLogger]] = None,
        **overrides
    ):
        """Initialize detection trainer.
        
        Args:
            config: Combined config - can be:
                    - Path to complete config file
                    - Complete config dictionary
            model: Model specification - can be:
                    - Model name (e.g. "dfine_n_coco")
                    - Path to model config
                    - Model config dictionary
            dataset: Dataset specification - can be:
                    - Dataset name (e.g. "coco", "dummy_coco") 
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
            loggers=loggers,
            **overrides
        )
        
        if not self.cfg.task == "detection":
            raise ValueError("DetectionTrainer requires task='detection' in config")


    def fit(self, device: str = None):
        self.train(device)
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)
        top1 = 0
        best_stat = {'epoch': -1, }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train/{k}': v for k, v in train_stats.items()},
                **{f'test/{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            # Add COCO evaluation metrics
            if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                metric_names = [
                    'AP@[IoU=0.50:0.95]',
                    'AP@[IoU=0.50]',
                    'AP@[IoU=0.75]',
                    'AP@[IoU=0.50:0.95]|small',
                    'AP@[IoU=0.50:0.95]|medium', 
                    'AP@[IoU=0.50:0.95]|large',
                    'AR@[IoU=0.50:0.95]|1',
                    'AR@[IoU=0.50:0.95]|10',
                    'AR@[IoU=0.50:0.95]|100',
                    'AR@[IoU=0.50:0.95]|small',
                    'AR@[IoU=0.50:0.95]|medium',
                    'AR@[IoU=0.50:0.95]|large'
                ]
                log_stats.update({
                    f'metrics/{name}': value 
                    for name, value in zip(metric_names, coco_evaluator.coco_eval["bbox"].stats.tolist())
                })

            if self.loggers:
                for logger in self.loggers:
                    logger.log_metrics(log_stats, epoch)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, device: str = None):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, device or self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
    
    def execute_ddp(self, device: str, task="train"):
        """Execute distributed training on specified devices"""
        # Create output directories
        tmp_dir = Path(self.cfg.output_dir) / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique training file
        timestamp = datetime.now().isoformat().replace(':', '-')
        tmp_file = tmp_dir / f'train_{timestamp}.py'
        tmp_config = tmp_dir / f'{tmp_file.stem}_config.yml'
        
        # Save config
        with open(tmp_config, 'w') as f:
            if "__include__" in self.cfg.yaml_cfg:
                cfg_dir = Path(self.cfg_path).parent
                paths = [str((cfg_dir / p).resolve()) for p in self.cfg.yaml_cfg["__include__"]]
                self.cfg.yaml_cfg["__include__"] = paths
            yaml.dump(self.cfg.yaml_cfg, f)
        
        # Create training script with improved process initialization
        with open(tmp_file, 'w', encoding='utf-8') as f:
            f.write(f'''
import os
import torch
from pathlib import Path
from trolo.trainers import DetectionTrainer
from trolo.utils import dist_utils

def main():
    # Set multiprocessing start method
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Setup distributed environment
    dist_utils.setup_distributed(seed=0)
    
    # Load config from the saved yaml
    config_path = "{tmp_config.absolute()}"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {{config_path}}")

    # Initialize trainer with the saved config
    trainer = DetectionTrainer(config=config_path)
    trainer._setup()
    trainer._prepare_training()
    
    try:
        {"trainer.fit()" if task == "train" else "trainer.val()"}
    finally:
        dist_utils.cleanup()

if __name__ == "__main__":
    main()
''')

        # Set CUDA devices and run DDP training with additional environment variables
        num_gpus = len(device.split(','))
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': device,
            'PYTHONPATH': os.pathsep.join(sys.path),
            'OMP_NUM_THREADS': '1',  # Prevent OpenMP runtime issues
            'NCCL_DEBUG': 'INFO'     # Help debug NCCL issues
        })
        
        # Get free port
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()

        cmd = f'torchrun --nproc_per_node={num_gpus} --master_port {port} {tmp_file}'
        
        try:
            print(f"Executing DDP command: {cmd}")
            subprocess.run(cmd, env=env, shell=True, check=True)
        finally:
            # Cleanup temporary files
            print("Cleaning up temporary files...")
            tmp_file.unlink(missing_ok=True)
            tmp_config.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
