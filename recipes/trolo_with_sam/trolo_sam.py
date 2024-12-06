from typing import Optional
import argparse

import cv2
import torch
import supervision as sv
from PIL import Image
from trolo import DetectionPredictor, to_sv

from segment_anything_hq import SamPredictor, sam_model_registry


def detect_objects(model_predictor: DetectionPredictor,
                   image: Image,
                   conf_threshold: Optional[float]=0.5) -> sv.Detections:
    """
    Detect objects in an image using the provided model predictor.
    """

    results = model_predictor.predict([image], conf_threshold=conf_threshold)
    detections = to_sv(results[0])
    detections = detections[detections.class_id == 0]
    detections = detections.with_nms(threshold=0.5)
    return detections


def main(image_path: str,
         model_name: Optional[str]="dfine-m" ,
         output_path: Optional[str]=None,
         vis: Optional[bool]=True,
         conf_threshold: Optional[float]=0.35) -> int:

    predictor = DetectionPredictor(model=model_name)

    # Change as per your requirement
    color_lookup = sv.ColorLookup.INDEX
    # annotators
    box_annotator = sv.BoxAnnotator(color_lookup=color_lookup)
    label_annotator = sv.LabelAnnotator(color_lookup=color_lookup)
    mask_ann = sv.MaskAnnotator(color_lookup=color_lookup)

    image = Image.open(image_path)
    detection = detect_objects(model_predictor=predictor, image=image, conf_threshold=conf_threshold)

    labels = [
        f"{class_id[0]}"
        for class_id in zip(detection.class_id)
    ]
    img = box_annotator.annotate(image.copy(), detection)
    img = label_annotator.annotate(img, detection, labels)

    # remove the detection model to save memory
    del predictor

    sam_checkpoint = "sam_hq_vit_tiny.pth"
    model_type = "vit_tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    input_boxes = detection.xyxy
    image_bgr = sv.pillow_to_cv2(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    detection_list = []
    for box in input_boxes:
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=False,
            box=box,
            return_logits=False,
        )
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detection_list.append(detections)

    mask_det = sv.Detections.merge(detection_list)

    img = mask_ann.annotate(img, mask_det)
    if vis:
        img.show()
    if output_path:
        img.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection and tracking in video with line zone counting.")
    parser.add_argument("--image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--model_name", type=str, default="dfine-m", help="Name of the detection model.")
    parser.add_argument("--output_path", type=str, default="demo.mp4", help="Path to save the output annotated image.")
    parser.add_argument("--vis", type=bool, default=True,
                        help="Whether to visualize the output frames (default: True).")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for detection (default: 0.5).")

    args = parser.parse_args()
    main(
        image_path=args.image_path,
        model_name=args.model_name,
        output_path=args.output_path,
        vis=args.vis,
        conf_threshold=float(args.conf_threshold)
    )



