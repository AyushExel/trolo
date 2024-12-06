from typing import Union, Dict, Optional
from pathlib import Path

import torch
import torchvision.transforms as T
from torch import nn

from ..loaders import YAMLConfig
from ..utils.smart_defaults import infer_model_config_path, infer_pretrained_model, infer_device
from ..loaders.maps import get_model_config_path

from ..utils.logging import LOGGER

class DetExporter:
    def __init__(
        self,
        model: Union[str, Path] = None,  # Model name or checkpoint path
        config: Union[str, Path] = None,  # Config name or path
        device: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """Initialize detection predictor

        Args:
            model: Model name (e.g. 'dfine-n') or path to checkpoint
            config: Optional config name or path. If None, will try to:
                   1. Load from checkpoint if available
                   2. Infer from model name
            device: Device to run inference on
        """
        if model is None:
            raise ValueError("Must specify model name or checkpoint path")

        # Convert model to path if it's a name
        model_path = infer_pretrained_model(model)

        # Load checkpoint first to check for config
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if config is None:
            if "cfg" in checkpoint:
                LOGGER.info("Loading config from checkpoint")
                self.config = YAMLConfig.from_state_dict(checkpoint["cfg"])
            else:
                LOGGER.warning("Config not found in checkpoint, inferring from model name")
                config = infer_model_config_path(model)
                self.config = self.load_config(config)
        else:
            # Convert config to path if it's a name
            if isinstance(config, str) and not Path(config).exists():
                config = get_model_config_path(config)
            self.config = self.load_config(config)

        self.device = torch.device(infer_device(device))
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.output_path = output_path

    def __call__(self, *args, **kwargs):
        pass

    def export_onnx(self, simplify: bool = False, opset_version: int = 11, **kwargs) -> None:

        class Model(nn.Module):
            def __init__(self, model, postprocessor) -> None:
                super().__init__()
                self.model = model
                self.postprocessor = postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        """Export model to ONNX format"""
        # Export model to ONNX
        LOGGER.info(f"Exporting model to ONNX: {self.output_path}")
        postprocessor = self.config.postprocessor
        model = Model(model=self.model, postprocessor=postprocessor)
        model = model.to(self.device)
        size = tuple(self.config.yaml_cfg["eval_spatial_size"])  # [H, W]
        data = torch.rand(1, 3, size[0], size[1])
        data = data.to(self.device)
        size = torch.tensor([size[0], size[1]])
        size = size.to(self.device)
        _ = model(data, size)

        torch.onnx.export(
            model,
            (data, size),
            self.output_path,
            input_names=['images', 'orig_target_sizes'],
            output_names=["labels", 'boxes', "scores"],
            dynamic_axes=None,
            opset_version=16,
            verbose=False,
        )
        check = True
        if check:
            import onnx
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            LOGGER.info('Check export onnx model done...')

    def load_config(self, config_path: str) -> Dict:
        """Load config from YAML"""
        LOGGER.info(f"Loading config from {config_path}")
        cfg = YAMLConfig(config_path)
        return cfg

    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load detection model using config"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if "HGNetv2" in self.config.yaml_cfg:
            self.config.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Load model state
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Load state into config.model
        self.config.model.load_state_dict(state)

        # Create deployment model wrapper
        model = self.config.model.deploy()
        return model





