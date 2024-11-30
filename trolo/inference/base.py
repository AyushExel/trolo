from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import json
import numpy as np
from PIL import ImageDraw, ImageFont
from trolo.utils.smart_defaults import infer_input_type, infer_output_path
from trolo.inference.video import VideoStream


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
        return_inputs: bool = False,
        conf_threshold: float = 0.5,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Image.Image]]]:
        """Run inference on input"""
        pass
    
    def visualize(
        self,
        input: Union[str, List[str], Image.Image, List[Image.Image]],
        conf_threshold: float = 0.5,
        show: bool = False,
        save: bool = False,
        save_dir: Optional[str] = None,
        batch_size: int = 1
    ) -> Optional[Union[Image.Image, List[Image.Image]]]:
        """
        Visualize predictions on different input types
        
        Args:
            input: Path to image/video/webcam, or PIL Image(s)
            conf_threshold: Confidence threshold for detections
            show: Whether to show results in window
            save: Whether to save results to disk
            save_dir: Directory to save results (if None, uses default)
            batch_size: Batch size for video processing
        """
        # Handle string input paths
        if isinstance(input, str):
            input_type = infer_input_type(input)
            
            if input_type in ['video', 'webcam']:
                source = 0 if input_type == 'webcam' else input
                self._process_video(
                    source=source,
                    batch_size=batch_size,
                    conf_threshold=conf_threshold,
                    show=show,
                    save=save,
                    output_path=save_dir
                )
                return None  # Video processing handles its own visualization
            
        # Run prediction and visualization for images
        predictions, inputs = self.predict(input, return_inputs=True, conf_threshold=conf_threshold)
        
        # Try to get class names from model config
        class_names = self.config.yaml_cfg.get('class_names', None)
        
        # Visualize predictions
        viz_images = self._visualize_predictions(inputs, predictions, class_names=class_names)
        
        # Show if requested
        if show:
            if isinstance(viz_images, list):
                for img in viz_images:
                    img.show()
            else:
                viz_images.show()
                
        # Save if requested
        if save:
            save_dir = save_dir or infer_output_path()
            save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
            print(f'Saving to {save_dir}')
            
            if isinstance(viz_images, list):
                for i, img in enumerate(viz_images):
                    img.save(save_dir / f'pred_{i}.jpg')
            else:
                viz_images.save(save_dir / 'pred.jpg')
                
        return viz_images

    def _process_video(
        self,
        source: Union[str, int],
        batch_size: int = 1,
        conf_threshold: float = 0.5,
        show: bool = True,
        save: bool = True,
        output_path: Optional[str] = None
    ) -> None:
        """Internal method to process video streams"""
        class_names = self.config.yaml_cfg.get('class_names', None)

        with VideoStream(source, batch_size=batch_size) as stream:
            # Get video properties
            cap = stream.cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer if saving
            if save:
                output_path = output_path or infer_output_path()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(Path(output_path) / 'output.mp4'), 
                    fourcc, fps, (width, height)
                )
            
            # Process stream in batches
            for batch in stream:
                frames = batch['frames']  # List of RGB numpy arrays
                
                # Convert frames to PIL Images
                pil_frames = [Image.fromarray(frame) for frame in frames]
                
                # Run prediction and visualization
                predictions, _ = self.predict(pil_frames, return_inputs=True, conf_threshold=conf_threshold)
                viz_frames = self._visualize_predictions(pil_frames, predictions, class_names=class_names)
                
                # Convert back to BGR for OpenCV
                for viz_frame in viz_frames:
                    bgr_frame = cv2.cvtColor(np.array(viz_frame), cv2.COLOR_RGB2BGR)
                    
                    if save:
                        out.write(bgr_frame)
                    
                    if show:
                        cv2.imshow('Video Stream', bgr_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return
            
            if save:
                out.release()
            
            if show:
                cv2.destroyAllWindows()

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