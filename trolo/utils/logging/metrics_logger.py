from abc import ABC, abstractmethod
from typing import Dict, Any

class ExperimentLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        pass
    
    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        pass

    def log_model(self, model) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass



