from typing import Dict, Union, Optional, List, Tuple

import numpy as np
import torch
import os
import torch.nn as nn
from pathlib import Path
from ..loaders import YAMLConfig
from ..loaders.maps import get_model_config_path

from ..utils.smart_defaults import infer_pretrained_model, infer_model_config_path, infer_device
from ..utils.logging.glob_logger import LOGGER


class ModelExporter:

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
        model_path = model
        # Convert model to path if it's a name
        self.model_path = infer_pretrained_model(model_path)
        if os.path.exists(model):
            LOGGER.error(f"{model} not found")

        # Load checkpoint first to check for config
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

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
        self.model = self.load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

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

        # Create deployment model wrapper so post process is included
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
            self.export2onnx(
                input_size=torch.tensor(input_size)
            )
        elif export_format.lower().strip() == "openvino":
            self.export_openvino(
                input_size=input_size
            )

    def export2onnx(
        self,
        input_size : Union[List, Tuple, torch.Tensor] = None,
        dynamic : Optional [bool] = False,
        batch_size : Optional[int] =  1,
        opset_version : Optional[int] = 16,
        simplify : Optional[bool] = False
    ) -> Path:
        import onnx
        input_size  = torch.tensor(input_size)
        input_data = torch.rand(batch_size, 3, *input_size)
        input_data = input_data.to(self.device)

        filename, file_ext = os.path.splitext(self.model_path)
        exported_path  =  f"{filename}.onnx"
        if dynamic:
            dynamic_axes = {'images': {0: 'N', },'orig_target_sizes': {0: 'N'}}

        input_names = ['images']
        output_names = ['labels', 'boxes', 'scores']

        # dynamic only compatible with cpu do not use it with gpu
        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,
            input_data.cpu() if dynamic else input_data,
            exported_path,
            input_names = input_names, 
            output_names = output_names,
            dynamic_axes=dynamic_axes if dynamic else None,
            opset_version=opset_version,
            verbose=False,
            do_constant_folding=True,
        )
        if not os.path.exists(exported_path):
            LOGGER.error(f"Failed to export model to ONNX: {exported_path}")

        # Check the model
        onnx_model  = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)
        LOGGER.info(f"Model exported to ONNX: {exported_path}")

        if simplify:
            LOGGER.info("Simplifying the onnx model")
            import onnxsim
            onnx_model_simplified, check = onnxsim.simplify(exported_path)
            onnx.save(onnx_model_simplified, exported_path)        
            onnx_model  = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)
            LOGGER.info(f"Simplified model exported to ONNX: {exported_path}")


    def export_openvino(
        self,
        input_size : Union[List, Tuple] = None,
        dynamic : Optional [bool] = False,
        batch_size : Optional[int] =  1,
    ) -> None:

        import openvino as ov
        # input_data = np.random.randn(batch_size, 3, *input_size).astype(np.float32) / 255.0
        #
        filename, file_ext = os.path.splitext(self.model_path)
        output_path = f"{filename}.xml"
        #
        # ov_model = ov.convert_model(
        #     self.model.cpu(),
        #     input=[1, 3, *input_size],
        #     example_input=input_data,
        # )
        #
        # print(f"Exporting model to OpenVINO: {output_path}")
        # ov.runtime.save_model(ov_model, output_path, compress_to_fp16=False)

        dummy_input = torch.randn(1, 3, 640, 640)
        traced_model = torch.jit.trace(self.model.cpu(), dummy_input)
        ov_model = ov.convert_model(
            traced_model,  # Use traced model
            share_weights=False
        )

        # Save the OpenVINO model
        ov.save_model(ov_model, output_path)
