try:
    import wandb
except ImportError:
    wandb = None

from typing import Optional, Dict, Any
from trolo.utils.logging.metrics_logger import ExperimentLogger

class WandbLogger(ExperimentLogger):
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if wandb is None:
            raise ImportError("Wandb is not installed. Please install it with `pip install wandb`.")
        wandb.init(project=project, name=name, config=config, **kwargs)
        
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        wandb.log(metrics)
        
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params)
        
    def close(self) -> None:
        wandb.finish()
