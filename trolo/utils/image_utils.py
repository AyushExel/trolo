from typing import Tuple

import cv2
import numpy as np
import supervision as sv
from PIL import Image


def letterbox_image(
    image: Image,
    resolution_wh: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    cv_image = sv.pillow_to_cv2(image)
    target_w, target_h = resolution_wh
    old_h, old_w = image.height, image.width
    ratio = min(target_w / image.width, target_h / image.height)

    new_w = int(old_w * ratio)
    new_h = int(old_h * ratio)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = int(round(pad_h - 0.1))
    pad_bottom = int(round(pad_h + 0.1))
    pad_left = int(round(pad_w - 0.1))
    pad_right = int(round(pad_w + 0.1))

    # add border
    cv_image = cv2.copyMakeBorder(cv_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)

    return cv_image, (ratio, ratio), (pad_left, pad_top)




    image = sv.pillow_to_cv2(image)
    h_original, w_original = image.shape[:2]

    image_ratio = w_original / h_original
    target_ratio = resolution_wh[0] / resolution_wh[1]
    if image_ratio >= target_ratio:
        width_new = resolution_wh[0]
        height_new = int(resolution_wh[0] / image_ratio)
    else:
        height_new = resolution_wh[1]
        width_new = int(resolution_wh[1] * image_ratio)

    scale_w = width_new / w_original
    scale_h = height_new / h_original

    image_resized = cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR)
    h_new, w_new = image_resized.shape[:2]

    pad_x = (target_w - w_new) // 2
    pad_y = (target_h - h_new) // 2
    pad_top, pad_bottom = int(round(pad_y - 0.1)),  int(round(pad_y + 0.1))
    pad_left, pad_right = int(round(pad_x - 0.1)),  int(round(pad_x + 0.1))

    image_letterbox = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)

    return image_letterbox, (scale_w, scale_h), (pad_left, pad_top)


def letterbox_adjust_boxes(boxes, letterbox_sizes, scales, paddings):
    """
    Adjust bounding boxes for letterboxing while maintaining aspect ratio.

    Args:
        boxes (torch.Tensor): Bounding boxes in [N, num_boxes, 4] (cx, cy, w, h) normalized [0, 1].
        original_sizes (list of tuples): List of (height, width) for each image.
        target_size (tuple): Desired (target_height, target_width) for letterboxing.

    Returns:
        torch.Tensor: Adjusted bounding boxes in pixel coordinates for the original image.
    """
    adjusted_boxes = boxes.clone()
    for i, (target_size) in enumerate(letterbox_sizes):
        h_original, w_original = 1080, 810
        pad_left, pad_top = paddings[i]
        scale_w, scale_h = scales[i]

        adjusted_boxes[i, :, [0, 2]] *= w_original  # Scale x-coordinates
        adjusted_boxes[i, :, [1, 3]] *= h_original  # Scale y-coordinates

        # # Adjust boxes for letterboxing
        adjusted_boxes[i, :, [0, 2]] -= (pad_left * 640 / w_original)    # Scale x-coordinates
        adjusted_boxes[i, :, [1, 3]] -= pad_top   # Scale y-coordinates

        # # Scale normalized boxes to resized dimensions
        # adjusted_boxes[i, :, [0, 2]] *= scale_w    # Scale x-coordinates
        # adjusted_boxes[i, :, [1, 3]] *= scale_h    # Scale y-coordinates

    return adjusted_boxes
