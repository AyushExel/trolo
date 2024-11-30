from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
import supervision as sv

from trolo.utils.smart_defaults import infer_input_type, infer_output_path, infer_device
from trolo.inference.video import VideoStream
from trolo.utils.box_ops import to_sv
from trolo.utils.draw_utils import draw_predictions


import torch
from PIL import Image
import cv2


class BasePredictor(ABC):
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(infer_device(device))
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
            print(f"Saving to {save_dir}")

            if isinstance(viz_images, list):
                for i, img in enumerate(viz_images):
                    img.save(save_dir / f"pred_{i}.jpg")
            else:
                viz_images.save(save_dir / "pred.jpg")

        return viz_images

    def _process_video(
        self,
        source: Union[str, int],
        batch_size: int = 1,
        conf_threshold: float = 0.5,
        show: bool = True,
        save: bool = True,
        output_path: Optional[str] = None,
    ) -> None:
        """Internal method to process video streams"""
        class_names = self.config.yaml_cfg.get("class_names", None)

        with VideoStream(source, batch_size=batch_size) as stream:
            # Get video properties
            cap = stream.cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize video writer if saving
            if save:
                output_path = output_path or infer_output_path()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(Path(output_path) / "output.mp4"), fourcc, fps, (width, height))

            # Process stream in batches
            for batch in stream:
                frames = batch["frames"]  # List of RGB numpy arrays

                # Convert frames to PIL Images
                pil_frames = [Image.fromarray(frame) for frame in frames]

                # Run prediction and visualization
                predictions, _ = self.predict(pil_frames, return_inputs=True, conf_threshold=conf_threshold)
                viz_frames = draw_predictions(pil_frames, predictions, class_names=class_names)

                # Convert back to BGR for OpenCV
                for viz_frame in viz_frames:
                    bgr_frame = cv2.cvtColor(np.array(viz_frame), cv2.COLOR_RGB2BGR)

                    if save:
                        out.write(bgr_frame)

                    if show:
                        cv2.imshow("Video Stream", bgr_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return

            if save:
                out.release()

            if show:
                cv2.destroyAllWindows()
