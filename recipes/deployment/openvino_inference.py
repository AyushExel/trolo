from typing import Optional, Tuple, Dict
import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import openvino
import cv2
from openvino.runtime import Core
import supervision as sv


"""
Dependencies:
pip install openvino supervision
"""


class OpenVinoInfer:
    """
    A class for performing inference with an OpenVino detection model.

    Attributes:
        model_path (str): Path to the OpenVino model file.
        infer_resolution (Tuple[int, int]): Resolution to resize the input image for inference.
        device (str): Device to use for inference (e.g., 'AUTO', 'cpu', 'cuda').
    """

    def __init__(self, model_path: str, infer_resolution: Tuple[int, int] = (640, 640), device: str = "AUTO"):
        """
        Initializes the OpenVino class.

        Args:
            model_path (str): Path to the ONNX model file.
            infer_resolution (Tuple[int, int]): Inference resolution (width, height). Default is (640, 640).
            device (str): Inference device ('AUTO', 'cpu' or 'cuda'). Default is 'AUTO'.
        """
        self.model_path = model_path
        if isinstance(infer_resolution, int):
            self.infer_resolution = (infer_resolution, infer_resolution)
        self.infer_resolution = infer_resolution
        self.device = device
        self.core = Core()
        self.ov_model = self.core.read_model(model=str(model_path), weights=Path(model_path).with_suffix(".bin"))
        self.available_device = self.core.available_devices
        self.compile_model = self.core.compile_model(self.model_path, device)

    def infer(self, input_path: str,
              conf_threshold: Optional[float] = 0.5,
              vis: Optional[bool]=True,
              output_path: Optional[str] = None) -> sv.Detections:
        """
        Perform inference on an input image.

        Args:
            input_path (str): Path to the input image file.
            conf_threshold (Optional[float]): Confidence threshold for detections. Default is 0.5.
            vis (Optional[bool]): Whether to visualize the results. Default is True.
            output_path (Optional[str]): Path to save the annotated output image. Default is None.

        Returns:
            sv.Detections: Detected bounding boxes, labels, and confidence scores.
        """
        image = Image.open(input_path)
        input_data = self.preprocess(image)
        output = self._infer(input_data)

        labels = output['labels']
        boxes = output['boxes']
        scores = output['scores']
        labels = labels.squeeze()
        boxes = boxes.squeeze()
        scores = scores.squeeze()

        boxes = self.post_process(image.size, boxes)
        detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)
        detections = detections[detections.confidence > conf_threshold]
        annotated_image = self.annotate(image=image, detections=detections)
        if vis:
            annotated_image.show()
        if output_path:
            annotated_image.save(output_path)
        return detections

    def _infer(self, inputs: dict):
        infer_request = self.compile_model.create_infer_request()
        for input_name, input_data in inputs.items():
            input_tensor = openvino.Tensor(input_data)
            infer_request.set_tensor(input_name, input_tensor)
        infer_request.infer()
        outputs = {'labels': infer_request.get_tensor(self.compile_model.outputs[0]).data,
                   'boxes': infer_request.get_tensor(self.compile_model.outputs[1]).data,
                   'scores': infer_request.get_tensor(self.compile_model.outputs[2]).data}
        return outputs

    def preprocess(self, image: Image) -> Dict:
        """
        Preprocess the input image for inference.

        Args:
            image (Image): Input PIL image.

        Returns:
            np.ndarray: Preprocessed image tensor ready for model input.
        """
        resized_im_pil = sv.letterbox_image(image, resolution_wh=self.infer_resolution)
        resized_im_pil = resized_im_pil.convert("RGB")
        im_data = np.asarray(resized_im_pil).astype(np.float32) / 255.0
        blob_image = cv2.dnn.blobFromImage(im_data, 1.0)
        inputs = {
            'images': blob_image,
        }
        return inputs

    def post_process(self, resolution_wh: Tuple[int, int], boxes: sv.Detections) -> Image:
        """
        Adjust bounding boxes to match the original image size.

        Args:
            resolution_wh (Tuple[int, int]): Original image resolution (width, height).
            boxes (sv.Detections): Detected bounding boxes in model output format.

        Returns:
            np.ndarray: Adjusted bounding boxes in xyxy format.
        """
        boxes_np = boxes.copy()

        boxes_xyxy = sv.xcycwh_to_xyxy(boxes_np)
        input_w, input_h = resolution_wh
        letterbox_w, letterbox_h = self.infer_resolution

        boxes_xyxy[:, [0, 2]] *= letterbox_w
        boxes_xyxy[:, [1, 3]] *= letterbox_h

        target_ratio = letterbox_w / letterbox_h
        image_ratio = input_w / input_h
        if image_ratio >= target_ratio:
            width_new = letterbox_w
            height_new = int(letterbox_w / image_ratio)
        else:
            height_new = letterbox_h
            width_new = int(letterbox_h * image_ratio)

        scale = input_w / width_new

        padding_top = (letterbox_h - height_new) // 2
        padding_left = (letterbox_w - width_new) // 2

        boxes_xyxy[:, [0, 2]] -= padding_left
        boxes_xyxy[:, [1, 3]] -= padding_top

        boxes_xyxy[:, [0, 2]] *= scale
        boxes_xyxy[:, [1, 3]] *= scale

        return boxes_xyxy

    def annotate(self, image: Image, detections: sv.Detections):
        """
        Annotate the image with detections.

        Args:
            image (Image): Input PIL image.
            detections (sv.Detections): Detections to annotate.

        Returns:
            Image: Annotated PIL image.
        """
        box_an = sv.BoxAnnotator()
        im_pil = box_an.annotate(image.copy(), detections)
        labels = [f"{int(class_id)}: {conf}" for class_id, conf in zip(detections.class_id, detections.confidence)]
        label_an = sv.LabelAnnotator()
        im_pil = label_an.annotate(im_pil, detections, labels)
        return im_pil


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trolo onnx model inference.")
    parser.add_argument("--input_path", type=str, default="bus.jpg", help="Path to the input image file.")
    parser.add_argument("--model_name", type=str, default="../../dfine_m.xml", help="Name of the onnx detection model.")
    parser.add_argument("--output_path", type=str, default="output.jpg",
                        help="Path to save the output annotated image.")
    parser.add_argument("--vis", type=bool, default=True,
                        help="Whether to visualize the output frames (default: True).")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for detection (default: 0.35).")
    args = parser.parse_args()
    infer = OpenVinoInfer(model_path=args.model_name)
    infer.infer(input_path=args.input_path, vis=args.vis, output_path=args.output_path, conf_threshold=args.conf_threshold)
