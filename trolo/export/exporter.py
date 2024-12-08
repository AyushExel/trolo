from typing import Dict, Union, Optional, List, Tuple, Any
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
        input_names : Optional[list] =  None, 
        output_names : Optional[list] =  None,
        dynamic_axes : Optional [dict] =  None,
        batch_size : int =  1,
        opset_version : int = 16,
        simplify : bool = False
    ) -> None: 
        # Default input and output names with post-processing
        input_names = ['images', 'orig_target_sizes']
        output_names =['boxes', 'scores', 'labels']
        
        # Prepare dynamic axes
        dynamic_axes = dynamic_axes or {
            'images': {0: 'N'}, 
            'orig_target_sizes': {0: 'N'},
            'boxes': {0: 'N', 1: 'M'},
            'scores': {0: 'N'},
            'labels': {0: 'N'}
        }

        input_size = torch.tensor(input_size)
        input_data = torch.rand(batch_size, 3, *input_size)
        letterbox_sizes = torch.tensor([[input_size[0], input_size[1]]] * batch_size, dtype=torch.float32)
        original_sizes = torch.tensor([[input_size[0], input_size[1]]] * batch_size, dtype=torch.float32)

        # Define a wrapper function that includes post-processing
        class PostProcessWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images, orig_target_sizes):
                # Get model outputs
                outputs = self.model(images)
                
                # Perform softmax on logits
                if isinstance(outputs, dict):
                    logits = outputs["pred_logits"]
                    boxes = outputs["pred_boxes"]
                else:
                    logits, boxes = outputs

                probs = logits.softmax(-1)
                scores, labels = probs.max(-1)

                # Placeholder for letterbox adjustments (you might need to implement this)
                # This is a simplified version and might need customization
                boxes_adjusted = boxes.clone()
                
                return boxes_adjusted, scores, labels

        # Create the wrapper model
        wrapped_model = PostProcessWrapper(self.model)

        # Exported path
        exported_path = f"{self.model_path.replace('pth', 'onnx')}"

        # Export to ONNX with post-processing
        torch.onnx.export(
            wrapped_model,
            (input_data, original_sizes),
            exported_path,
            input_names=input_names, 
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,
            do_constant_folding=True,
        )

        # Validate the ONNX model
        onnx_model = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)
        LOGGER.info(f"Model exported to ONNX: {exported_path}")

        # Optional model simplification
        if simplify:
            import onnxsim
            onnx_model_simplified, check = onnxsim.simplify(exported_path)
            onnx.save(onnx_model_simplified, exported_path)        
            onnx_model = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)
            LOGGER.info(f"Simplified Model exported to ONNX: {exported_path}")

    def postprocess(
        self, outputs: torch.Tensor, letterbox_sizes: List[Tuple[int, int]], original_sizes: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """Convert model outputs to boxes, scores, labels

        Returns:
            List of dictionaries, one per image, each containing:
                - boxes: tensor of shape (N, 4) in [cx, cy, w, h] format
                - scores: tensor of shape (N,)
                - labels: tensor of shape (N,)
        """
        if isinstance(outputs, dict):
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]  # [cx, cy, w, h] format
        else:
            logits, boxes = outputs

        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)

        # Scale normalized coordinates to image size
        boxes = boxes.clone()
        boxes = letterbox_adjust_boxes(boxes, letterbox_sizes, original_sizes)

        # Convert batch tensors to list of individual predictions
        predictions = []
        for i in range(len(original_sizes)):
            predictions.append({"boxes": boxes[i].cpu(), "scores": scores[i].cpu(), "labels": labels[i].cpu()})

        return predictions