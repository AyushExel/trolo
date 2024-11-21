

from .base import BaseTrainer
from .classification import ClassificationTrainer
from .detection import DetectionTrainer



from typing import Dict

TASKS :Dict[str, BaseTrainer] = {
    'classification': ClassificationTrainer,
    'detection': DetectionTrainer,
}
