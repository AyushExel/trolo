from typing import Optional, Tuple
import argparse

from PIL import Image
import numpy as np
import onnxruntime as rt
import supervision as sv


class OnnxInfer:

    def __init__(self, model_path: str, infer_resolution: Tuple[int, int] = (640, 640), device: str = "cpu"):
        self.model_path = model_path
        if isinstance(infer_resolution, int):
            self.infer_resolution = (infer_resolution, infer_resolution)
        self.infer_resolution = infer_resolution
        self.device = device
        self.session = rt.InferenceSession(self.model_path)

    def infer(self, input_path: str,
              conf_threshold: Optional[float] = 0.5,
              vis: Optional[bool]=True,
              output_path: Optional[str] = None) -> sv.Detections:
        image = Image.open(input_path)
        image_processed = self.preprocess(image)
        input_feed = {"images": image_processed}
        output = self.session.run(["labels", "boxes", "scores"], input_feed=input_feed)
        labels, boxes, scores = output
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

    def preprocess(self, image: Image):
        resized_im_pil = sv.letterbox_image(image, resolution_wh=self.infer_resolution)
        resized_im_pil = resized_im_pil.convert("RGB")
        im_data = np.asarray(resized_im_pil).astype(np.float32) / 255.0
        im_data = np.expand_dims(im_data, axis=0)
        im_data = np.transpose(im_data, (0, 3, 1, 2))
        return im_data

    def post_process(self, resolution_wh: Tuple[int, int], boxes: sv.Detections) -> Image:
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
        box_an = sv.BoxAnnotator()
        im_pil = box_an.annotate(image.copy(), detections)
        labels = [f"{int(class_id)}: {conf}" for class_id, conf in zip(detections.class_id, detections.confidence)]
        label_an = sv.LabelAnnotator()
        im_pil = label_an.annotate(im_pil, detections, labels)
        return im_pil


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trolo onnx model inference.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--model_name", type=str, default="dfine-n.onnx", help="Name of the onnx detection model.")
    parser.add_argument("--output_path", type=str, default="output.jpg",
                        help="Path to save the output annotated image.")
    parser.add_argument("--vis", type=bool, default=True,
                        help="Whether to visualize the output frames (default: True).")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for detection (default: 0.35).")
    args = parser.parse_args()
    infer = OnnxInfer(model_path=args.model_name)
    infer.infer(input_path=args.input_path, vis=args.vis, output_path=args.output_path, conf_threshold=args.conf_threshold)
