
import torch.cuda.amp as amp

from ..registry import register


__all__ = ['GradScaler']

GradScaler = register()(amp.grad_scaler.GradScaler)
