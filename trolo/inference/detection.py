from typing import Union, List, Dict, Any, Tuple, Iterator, Optional
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from .base import BasePredictor
from ..loaders import YAMLConfig
from ..data.transforms import Compose
from ..utils.smart_defaults import infer_model_config_path
from ..loaders.maps import get_model_config_path
from ..inference.video import VideoStream
from ..utils.smart_defaults import infer_input_type, infer_input_path, get_images_from_folder

class DetectionPredictor(BasePredictor):
    def __init__(self, 
                 model: Union[str, Path] = None,  # Model name or checkpoint path
                 config: Union[str, Path] = None,  # Config name or path
                 device: Optional[str] = None):
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
        checkpoint = torch.load(model, map_location='cpu', weights_only=False)
        
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
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

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
        
    def postprocess(self, outputs: torch.Tensor, original_sizes: List[tuple]) -> List[Dict[str, Any]]:
        """Convert model outputs to boxes, scores, labels
        
        Returns:
            List of dictionaries, one per image, each containing:
                - boxes: tensor of shape (N, 4) in [cx, cy, w, h] format
                - scores: tensor of shape (N,)
                - labels: tensor of shape (N,)
        """
        if isinstance(outputs, dict):
            logits = outputs['pred_logits']
            boxes = outputs['pred_boxes']  # [cx, cy, w, h] format
        else:
            logits, boxes = outputs
        
        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)
        
        # Scale normalized coordinates to image size
        boxes = boxes.clone()
        for i, (h, w) in enumerate(original_sizes):
            boxes[i, :, [0, 2]] *= w  # cx, w
            boxes[i, :, [1, 3]] *= h  # cy, h
        
        # Convert batch tensors to list of individual predictions
        predictions = []
        for i in range(len(original_sizes)):
            predictions.append({
                'boxes': boxes[i].cpu(),
                'scores': scores[i].cpu(),
                'labels': labels[i].cpu()
            })
            
        return predictions
        
    def predict(
        self, 
        inputs: Optional[Union[str, List[str], Image.Image, List[Image.Image]]] = None, 
        conf_threshold: float = 0.5,
        return_inputs: bool = False,
        batch_size: int = 1,
        stream: Optional[bool] = None
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Image.Image]], Iterator[Tuple[List[Dict[str, Any]], List[Image.Image]]]]:
        """
        Predict on images or video with confidence thresholding
        
        Args:
            inputs: PIL Image, list of PIL Images, path to image/video, or list of paths
            conf_threshold: Confidence threshold for detections (default: 0.5)
            return_inputs: Whether to return the original inputs
            batch_size: Batch size for video processing
            stream: If True and input is video, yield predictions for each batch
            
        Returns:
            If input is image(s):
                List of prediction dictionaries, one per image, each containing:
                    - boxes: tensor of shape (N, 4)
                    - scores: tensor of shape (N,)
                    - labels: tensor of shape (N,)
                If return_inputs=True, also returns list of PIL images
            
            If input is video and stream=True:
                Iterator yielding (predictions, frames) tuples for each batch
        """
        # Handle string input paths
        if inputs is None:
            inputs = infer_input_path()
        if isinstance(inputs, (str, Path)):
            input_type = infer_input_type(inputs)
            
            if input_type in ['video', 'webcam']:
                source = 0 if input_type == 'webcam' else inputs
                return self._predict_video(
                    source=source,
                    batch_size=batch_size,
                    conf_threshold=conf_threshold,
                    return_inputs=return_inputs,
                    stream=True if stream is None else stream
                )
            if stream:
                raise ValueError("stream is only supported for video inputs")
            if input_type == 'folder':
                inputs = get_images_from_folder(inputs)
        # Handle image inputs
        if isinstance(inputs, (str, Image.Image)):
            inputs = [inputs]

        original_sizes = []
        input_images = []
        for img in inputs:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            input_images.append(img)
            original_sizes.append((img.height, img.width))
        # Process input
        batch = self.preprocess(input_images)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch)
            predictions = self.postprocess(outputs, original_sizes)
        
        # Filter by confidence while maintaining batch dimension
        filtered_predictions = []
        for pred in predictions:
            mask = pred['scores'] >= conf_threshold
            filtered_predictions.append({
                'boxes': pred['boxes'][mask],
                'scores': pred['scores'][mask],
                'labels': pred['labels'][mask]
            })
        
        if return_inputs:
            return filtered_predictions, input_images
        return filtered_predictions

    def _predict_video(
        self,
        source: Union[str, int],
        batch_size: int = 1,
        conf_threshold: float = 0.5,
        return_inputs: bool = True,
        stream: bool = True
    ) -> Iterator[Tuple[List[Dict[str, Any]], List[Image.Image]]]:
        """Internal method to process video streams"""
        
        with VideoStream(source, batch_size=batch_size) as video_stream:
            if not stream:
                # Collect all predictions if not streaming
                all_predictions = []
                all_frames = []
                
            for batch in video_stream:
                frames = batch['frames']  # List of RGB numpy arrays
                
                # Convert frames to PIL Images
                pil_frames = [Image.fromarray(frame) for frame in frames]
                
                # Get predictions for batch
                predictions = self.predict(
                    pil_frames, 
                    conf_threshold=conf_threshold,
                    return_inputs=False
                )
                
                if stream:
                    yield predictions, pil_frames if return_inputs else predictions
                else:
                    all_predictions.extend(predictions)
                    if return_inputs:
                        all_frames.extend(pil_frames)
                        
            if not stream:
                return all_predictions, all_frames if return_inputs else all_predictions