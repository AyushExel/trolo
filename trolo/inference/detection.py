from typing import Union, List, Dict, Any
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from .base import BasePredictor
from ..loaders import YAMLConfig
from ..data.transforms import Compose
from ..utils.smart_defaults import infer_model_config_path
from ..loaders.maps import get_model_config_path

class DetectionPredictor(BasePredictor):
    def __init__(self, 
                 model: Union[str, Path] = None,  # Model name or checkpoint path
                 config: Union[str, Path] = None,  # Config name or path
                 device: str = 'cpu'):
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
        if isinstance(model, str) and not Path(model).exists():
            print(f"Warning: Model path {model} does not exist, inferring config from model name")
            model = get_model_config_path(model)
        
        # Load checkpoint first to check for config
        checkpoint = torch.load(model, map_location='cpu')
        
        if config is None:
            if 'cfg' in checkpoint:
                print("Loading config from checkpoint")
                self.config = YAMLConfig.from_state_dict(checkpoint['cfg'])
            else:
                print("Config not found in checkpoint, inferring from model name")
                config = infer_model_config_path(model)
                self.config = self.load_config(config)
        else:
            # Convert config to path if it's a name
            if isinstance(config, str) and not Path(config).exists():
                config = get_model_config_path(config)
            self.config = self.load_config(config)

        self.transforms = self.build_transforms()
        super().__init__(model, device)
        
    def load_config(self, config_path: str) -> Dict:
        """Load config from YAML"""
        print(f"Loading config from {config_path}")
        cfg = YAMLConfig(config_path)
        return cfg
        
    def build_transforms(self) -> T.Compose:
        """Build preprocessing transforms for inference"""
        # Get image size from config or use default
        if hasattr(self.config, 'yaml_cfg') and 'eval_spatial_size' in self.config.yaml_cfg:
            size = tuple(self.config.yaml_cfg['eval_spatial_size'])  # [H, W]
        else:
            size = (640, 640)  # Default size
        
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load detection model using config"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'HGNetv2' in self.config.yaml_cfg:
            self.config.yaml_cfg['HGNetv2']['pretrained'] = False

        # Load model state
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # Load state into config.model
        self.config.model.load_state_dict(state)
        
        # Create deployment model wrapper
        model = self.config.model.deploy()
        return model
        
    def preprocess(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess images for detection model"""
        if isinstance(inputs, (str, Image.Image)):
            inputs = [inputs]
            
        images = []
        for img in inputs:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            images.append(self.transforms(img))
            
        return torch.stack(images).to(self.device)
        
    def postprocess(self, outputs: torch.Tensor, original_sizes: List[tuple]) -> Dict[str, Any]:
        """Convert model outputs to boxes, scores, labels"""
        if isinstance(outputs, dict):
            logits = outputs['pred_logits']
            boxes = outputs['pred_boxes']
        else:
            logits, boxes = outputs
        
        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)
        
        # Convert relative [0,1] boxes to absolute coordinates using original sizes
        boxes = boxes.clone()
        for i, (h, w) in enumerate(original_sizes):
            boxes[i, :, [0, 2]] *= w
            boxes[i, :, [1, 3]] *= h
        
        return {
            'boxes': boxes.cpu(),
            'scores': scores.cpu(),
            'labels': labels.cpu()
        }
        
    def predict(self, images, conf_threshold=0.5):
        """
        Predict on images with confidence thresholding
        
        Args:
            images: PIL Image or list of PIL Images
            conf_threshold: Confidence threshold for detections (default: 0.5)
        """
        # Get original image sizes before preprocessing
        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        original_sizes = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            original_sizes.append((img.height, img.width))
        
        # Process input
        batch = self.preprocess(images)
        batch_size = batch.shape[0]
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch)
            # Pass original sizes to postprocess
            outputs = self.postprocess(outputs, original_sizes)
        
        # Filter by confidence while maintaining batch dimension
        if 'scores' in outputs:
            mask = outputs['scores'] >= conf_threshold  # Shape: (batch_size, num_queries)
            filtered_outputs = {}
            for k in outputs:
                filtered_outputs[k] = []
                for b in range(batch_size):
                    filtered_outputs[k].append(outputs[k][b][mask[b]])
                filtered_outputs[k] = torch.stack(filtered_outputs[k])
            outputs = filtered_outputs
        
        return outputs