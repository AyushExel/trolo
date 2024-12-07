from typing import Dict, Union, Optional, List, Tuple
import torch 
import onnx
from pathlib import Path
from .base import BaseExporter
from ..loaders import YAMLConfig
from ..loaders.maps import get_model_config_path

from ..utils.smart_defaults import infer_pretrained_model
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

        # Create deployment model wrapper
        model = self.config.model.deploy()

        return model

    def export(
        self, 
        input_size : Union[List, Tuple] =  [640, 640], 
        export_format : str = "onnx"
    ):
        # check the model format
        if export_format is None:
            raise ValueError(f"Export format is missing!")
        
        # check for export format
        accepted = self._filter_format(export_format)

        if export_format.lower().strip() == "onnx":
            self._export2onnx(
                input_size=torch.tensor(input_size)
            )

    def _export2onnx(
        self,
        input_size : Union[List, Tuple] = None,
        input_names : Optional[list] =  ['images', 'orig_target_sizes'], 
        output_names : Optional[list] =  ['pred_logits', 'pred_boxes'],
        dynamic_axes : Optional [dict] =  {'images': {0: 'N'}, 'orig_target_sizes': {0: 'N'}},
        batch_size : int =  1,
        opset_version : int = 16,
        simplify : bool = False
    ) -> None: 
        input_size  = torch.tensor(input_size)
        input_data = torch.rand(batch_size, 3, *input_size)
        exported_path  =  f"{self.model_path.replace('pth', "onnx")}"

        dynamic_axes = dynamic_axes  or {'images': {0: 'N', },'orig_target_sizes': {0: 'N'}}
        torch.onnx.export(
            self.model,
            (input_data, input_size),
            exported_path,
            input_names = input_names, 
            output_names = output_names,
            dynamic_axes=dynamic_axes,
            opset_version=16,
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

        