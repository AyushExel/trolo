from typing import List, Dict, Union, Any, Optional

from PIL import Image
import supervision as sv
import numpy as np

from trolo.utils.box_ops import to_sv


def draw_predictions(
        image: Union[Image.Image, List[Image.Image]],
        predictions: List[Dict[str, Any]],
        class_names: Optional[List[str]] = None,
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

    color_lookup = sv.ColorLookup.CLASS
    box_annotator = sv.BoxAnnotator(color_lookup=color_lookup)
    label_annotator = sv.RichLabelAnnotator(color_lookup=color_lookup)

    result_images = []

    for img, preds in zip(images, predictions):
        detections = to_sv(preds)

        if class_names:
            class_names = np.asarray(class_names)
            detections.data = {"class_name": class_names}

            labels = [
                f"{detections['class_name'][class_id]} - {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
        else:
            labels = [
                f"{class_id} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

        img = box_annotator.annotate(img, detections)
        img = label_annotator.annotate(img, detections, labels)
        result_images.append(img)

    return result_images
