from typing import Dict, Union, Optional, List, Tuple
import torch
import os
import torch.nn as nn 
import onnx
from pathlib import Path
from .base import BaseExporter
from ..loaders import YAMLConfig
from ..loaders.maps import get_model_config_path

from ..utils.smart_defaults import infer_pretrained_model, infer_model_config_path
from ..utils.logging.glob_logger import LOGGER

class ModelExporter(BaseExporter):

    def __init__(
        self, 
        model : Union[str, Path] = None,
        config : Union[str, Path] =  None, 
        device : Optional[str] = None,
    ):
        """Initialize detection predictor

        Args:
            model: Model name (e.g. 'dfine-n') or path to checkpoint
            config: Optional config name or path. If None, will try to:
                   1. Load from checkpoint if available
                   2. export from model name
            device: device on export will be run
        """
        if model is None:
            raise ValueError("Must specify model name or checkpoint path")
        self.model_path = model
        # Convert model to path if it's a name
        model = infer_pretrained_model(model)

        # Load checkpoint first to check for config
        checkpoint = torch.load(model, map_location="cpu", weights_only=False)

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

        super().__init__(model, device)

    def load_config(self, config_path: str) -> Dict:
        """Load config from YAML"""
        LOGGER.info(f"Loading config from {config_path}")
        cfg = YAMLConfig(config_path)
        return cfg
    
    def load_model(self, model: str) -> torch.nn.Module:
        """Load detection model using config"""
        # Load checkpoint
        checkpoint = torch.load(model, map_location="cpu", weights_only=False)

        if "HGNetv2" in self.config.yaml_cfg:
            self.config.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Load model state
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Load state into config.model
        self.config.model.load_state_dict(state)

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images):
                # Run base model
                outputs = self.model(images)
                logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
                # Post-process outputs
                probs = logits.softmax(-1)
                scores, labels = probs.max(-1)
                return labels, boxes, scores

        # Create deployment model wrapper
        model = self.config.model.deploy()
        wrapped_model = ModelWrapper(model)
        return wrapped_model

    def export(
        self, 
        input_size : Union[List, Tuple[int, int], int] =  (640, 640),
        export_format : str = "onnx"
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        # check the model format
        if export_format is None:
            raise ValueError(f"Export format is missing!")

        if export_format.lower().strip() == "onnx":
            self._export2onnx(
                input_size=torch.tensor(input_size)
            )

    def _export2onnx(
        self,
        input_size : Union[List, Tuple] = None,
        dynamic_axes : Optional [dict] =  {'images': {0: 'N'}, 'orig_target_sizes': {0: 'N'}},
        batch_size : Optional[int] =  1,
        opset_version : Optional[int] = 16,
        simplify : Optional[bool] = False
    ) -> None: 
        input_size  = torch.tensor(input_size)
        input_data = torch.rand(batch_size, 3, *input_size)
        filename, file_ext = os.path.splitext(self.model_path)
        exported_path  =  f"{filename}.onnx"
        dynamic_axes = dynamic_axes  or {'images': {0: 'N', },'orig_target_sizes': {0: 'N'}}

        input_names = ['images', 'orig_target_sizes']
        output_names = ['labels', 'boxes', 'scores']

        dynamic_axes = None
        torch.onnx.export(
            self.model.cpu(),
            input_data,
            exported_path,
            input_names = input_names, 
            output_names = output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,
            do_constant_folding=True,
        )

        # TODO -  Exceptional Handler 
        onnx_model  = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)
        LOGGER.info(f"Model exported to ONNX: {exported_path}")

        if simplify:
            import onnxsim
            onnx_model_simplified, check = onnxsim.simplify(exported_path)
            onnx.save(onnx_model_simplified, exported_path)        
            onnx_model  = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)
        