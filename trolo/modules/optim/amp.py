
import torch.cuda.amp as amp

from trolo.loaders import register


__all__ = ['GradScaler']

GradScaler = register()(amp.grad_scaler.GradScaler)
