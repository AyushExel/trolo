import time
import json
import datetime

import torch

from ..utils import dist_utils, stats

from .base import BaseTrainer
from .det_engine import train_one_epoch, evaluate

from pathlib import Path
from typing import Union, Optional, Dict, List
from trolo.utils.logging import WandbLogger, ExperimentLogger

class DetectionTrainer(BaseTrainer):
    """Detection specific trainer implementation"""
    def __init__(
        self, 
        config: Optional[Union[str, Path, Dict]] = None,
        model: Optional[Union[str, Path, Dict]] = None,
        dataset: Optional[Union[str, Path, Dict]] = None,
        pretrained_model: Optional[Union[str, Path]] = None,
        loggers: Optional[List[ExperimentLogger]] = None,
        **kwargs
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
            **kwargs
        )
        
        if not self.cfg.task == "detection":
            raise ValueError("DetectionTrainer requires task='detection' in config")


    def fit(self, ):
        self.train()
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
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
