

import torch.nn as nn
from trolo.loaders.registry import register

from .det_criterion import DetCriterion

CrossEntropyLoss = register()(nn.CrossEntropyLoss)
