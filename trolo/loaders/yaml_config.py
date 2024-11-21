import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy
from typing import Union, Dict

from ._config import BaseConfig
from .registry import create_from_config
from .yaml_utils import load_config, merge_config, merge_dict

class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "YAMLConfig":
        """Create YAMLConfig instance from state dictionary
        
        Args:
            state_dict: Dictionary containing config state
            
        Returns:
            YAMLConfig instance initialized from state_dict
        """
        config = cls.__new__(cls)
        super(YAMLConfig, config).__init__()
        
        # Initialize yaml_cfg first
        config.yaml_cfg = copy.deepcopy(state_dict.get('yaml_cfg', {}))
        
        # Load remaining state
        config.load_state_dict(state_dict)
        return config

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create_from_config(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create_from_config(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create_from_config(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create_from_config('optimizer', self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create_from_config('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            print(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create_from_config('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create_from_config('ema', self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create_from_config('scaler', self.global_cfg)
        return super().scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create_from_config('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from trolo.utils import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        print(f'building {name} with batch_size={bs}...')
        loader = create_from_config(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader

    @classmethod
    def merge_configs(cls, model_cfg: Union[str, Dict], dataset_cfg: Union[str, Dict], **kwargs) -> 'YAMLConfig':
        """Merge separate model and dataset configs."""
        # Load configs if paths provided
        if isinstance(model_cfg, str):
            model_cfg = load_config(model_cfg)
        if isinstance(dataset_cfg, str):
            dataset_cfg = load_config(dataset_cfg)
        
        # Deep copy to avoid modifying originals
        model_cfg = copy.deepcopy(model_cfg)
        dataset_cfg = copy.deepcopy(dataset_cfg)
        
        # Start with model config as base
        merged = model_cfg
        
        # Merge dataset config underneath (won't override model settings)
        merged = merge_dict(dataset_cfg, merged, inplace=False)
        
        # Finally apply any additional kwargs overrides
        merged = merge_dict(merged, kwargs, inplace=False)
        
        # Create config object
        cfg = cls.__new__(cls)
        super(YAMLConfig, cfg).__init__()
        cfg.yaml_cfg = copy.deepcopy(merged)
        
        # Copy attributes from base config
        for k in super(YAMLConfig, cfg).__dict__:
            if not k.startswith('_') and k in merged:
                cfg.__dict__[k] = merged[k]
                
        return cfg

    def state_dict(self):
        """Return serializable state dictionary including yaml config"""
        state = super().state_dict()
        state['yaml_cfg'] = self.yaml_cfg
        return state

    def load_state_dict(self, state_dict):
        """Load state including yaml config"""
        super().load_state_dict(state_dict)
        if 'yaml_cfg' in state_dict:
            self.yaml_cfg = state_dict['yaml_cfg']
