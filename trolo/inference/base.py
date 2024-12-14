from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import supervision as sv
import torch
from PIL import Image

from trolo.utils.smart_defaults import infer_input_type, infer_output_path, infer_device
from trolo.utils.draw_utils import draw_predictions
from trolo.utils.logging import LOGGER


class BasePredictor(ABC):
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(infer_device(device))
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.fps_monitor = sv.FPSMonitor() # Monitor for FPS calculation

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
        batch_size: int = 1,
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

            if input_type in ["video", "webcam"]:
                source = 0 if input_type == "webcam" else input
                self._process_video(
                    source=source,
                    batch_size=batch_size,
                    conf_threshold=conf_threshold,
                    show=show,
                    save=save,
                    output_path=save_dir,
                )
                return None  # Video processing handles its own visualization

        # Run prediction and visualization for images
        predictions, inputs = self.predict(input, return_inputs=True, conf_threshold=conf_threshold)
        LOGGER.info(f"Average FPS: {self.fps_monitor.fps:.2f}")

        # Try to get class names from model config
        class_names = self.config.yaml_cfg.get("class_names", None)

        # Visualize predictions
        viz_images = draw_predictions(inputs, predictions, class_names=class_names)

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
            LOGGER.info(f"Saving to {save_dir}")

            if isinstance(viz_images, list):
                for i, img in enumerate(viz_images):
                    img.save(save_dir / f"pred_{i}.jpg")
            else:
                viz_images.save(save_dir / "pred.jpg")

        return viz_images
