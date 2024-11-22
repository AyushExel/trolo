from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import json
import numpy as np
from PIL import ImageDraw, ImageFont


import torch
from PIL import Image
import cv2

class BasePredictor(ABC):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    @abstractmethod
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from path"""
        pass
        
    @abstractmethod
    def preprocess(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess inputs to model input format"""
        pass
        
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Convert model outputs to final predictions"""
        pass

    @abstractmethod
    def predict(
        self, 
        input: Union[str, List[str], Image.Image, List[Image.Image]],
        return_inputs: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Image.Image]]]:
        """Run inference on input"""
        pass
    
    def visualize(
        self,
        input: Union[str, List[str], Image.Image, List[Image.Image]],
        show: bool = False,
        save: bool = False,
        save_dir: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Visualize predictions

        Args:
            input: Input image or list of images
            show: Whether to display results on screen
            save: Whether to save results to disk
            save_dir: Directory to save results. If None, will use ./outputs/
        """
        # Run prediction
        predictions, inputs = self.predict(input, return_inputs=True)
        
        # Debug prints
        print("Predictions type:", type(predictions))
        print("First prediction type:", type(predictions[0]))
        print("First prediction:", predictions[0])
        
        # Visualize predictions
        viz_images = self._visualize_predictions(inputs, predictions)
        
        # Show if requested
        if show:
            if isinstance(viz_images, list):
                for img in viz_images:
                    img.show()
            else:
                viz_images.show()
                
        # Save if requested
        if save:
            save_dir = Path(save_dir or './outputs')
            save_dir.mkdir(exist_ok=True, parents=True)
            
            if isinstance(viz_images, list):
                for i, img in enumerate(viz_images):
                    img.save(save_dir / f'pred_{i}.jpg')
            else:
                viz_images.save(save_dir / 'pred.jpg')
                
        return viz_images


    def _visualize_predictions(
        self, 
        image: Union[Image.Image, List[Image.Image]], 
        predictions: List[Dict[str, Any]],
        class_names: Optional[List[str]] = None
    ) -> List[Image.Image]:
        """Internal method to visualize predictions
        
        Args:
            image: Single image or list of images
            predictions: List of prediction dictionaries with boxes in [cx, cy, w, h] format
            class_names: Optional list of class names
        Returns:
            List of PIL Images with visualized predictions
        """
        # Ensure inputs are lists
        images = [image] if isinstance(image, Image.Image) else image
        
        # Default color palette and font
        colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(80)}
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        result_images = []
        
        # Process each image-prediction pair
        for img, preds in zip(images, predictions):
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            
            # Draw each detection
            for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
                # Convert from [cx, cy, w, h] to [x0, y0, x1, y1]
                cx, cy, w, h = box.tolist()
                x0 = int(cx - w/2)
                y0 = int(cy - h/2)
                x1 = int(cx + w/2)
                y1 = int(cy + h/2)
                
                # Ensure within image bounds
                x0 = max(0, min(x0, img.width))
                y0 = max(0, min(y0, img.height))
                x1 = max(0, min(x1, img.width))
                y1 = max(0, min(y1, img.height))
                
                # Skip invalid boxes
                if x1 <= x0 or y1 <= y0:
                    continue
                    
                # Get color for current label
                color = colors[int(label.item())]
                
                # Draw box with thicker width
                draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                
                # Prepare label text
                if class_names:
                    label_text = class_names[label.item()]
                else:
                    label_text = f"{label.item()}"
                conf_text = f"{score.item():.2f}"
                
                # Get text sizes
                label_bbox = draw.textbbox((0, 0), label_text, font=font)
                conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]
                conf_width = conf_bbox[2] - conf_bbox[0]
                
                # Draw label background
                label_bg_color = tuple(int(c * 0.7) for c in color)
                draw.rectangle(
                    [x0, max(0, y0 - label_height - 4), 
                     x0 + label_width + 4, y0],
                    fill=label_bg_color
                )
                
                # Draw confidence background
                draw.rectangle(
                    [x0 + label_width + 4, max(0, y0 - label_height - 4),
                     x0 + label_width + conf_width + 8, y0],
                    fill=(50, 50, 50)
                )
                
                # Draw text
                draw.text(
                    (x0 + 2, max(0, y0 - label_height - 2)),
                    label_text,
                    fill=(255, 255, 255),
                    font=font
                )
                draw.text(
                    (x0 + label_width + 6, max(0, y0 - label_height - 2)),
                    conf_text,
                    fill=(255, 255, 255),
                    font=font
                )
                
            result_images.append(draw_img)
            
        return result_images