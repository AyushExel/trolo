from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchvision
from torch import Tensor
import supervision as sv


def xcycwh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Args:
        boxes, [N, 4], (cx, cy, w, h)
    Returns:
        boxes, [N, 4], (x1, y1, x2, y2)
    """
    x, y, w, h = torch.split(boxes, 1, dim=-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.concat([x1, y1, x2, y2], dim=-1)


def to_sv(results: Dict) -> sv.Detections:
    detections = sv.Detections.empty()
    if "boxes" in results:
        if isinstance(results["boxes"], torch.Tensor):
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)
        elif isinstance(results["boxes"], np.ndarray):
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]
            detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)
    return detections


def letterbox_adjust_boxes(boxes: Union[torch.Tensor, np.ndarray], letterbox_sizes: Tuple[int, int], original_sizes: Tuple[int, int]):
    """
    Adjust bounding boxes for letterboxing while maintaining aspect ratio.

    Args:
        boxes (torch.Tensor): Bounding boxes in [N, num_boxes, 4] (cx, cy, w, h) normalized [0, 1].
        letterbox_sizes (list of tuples): List of (height, width) for each image.
        original_sizes (list of tuples): List of (height, width) for each image.

    Returns:
        torch.Tensor: Adjusted bounding boxes in pixel coordinates for the original image.
    """
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = boxes.copy()
    for i, (letterbox_size) in enumerate(letterbox_sizes):
        _boxes = boxes_np[i]
        boxes_xyxy = sv.xcycwh_to_xyxy(_boxes)
        input_w, input_h = original_sizes[i]
        letterbox_w, letterbox_h = letterbox_size

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
        boxes_np[i] = boxes_xyxy

    if isinstance(boxes, torch.Tensor):
        return torch.from_numpy(boxes_np)
    return boxes_np


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


# elementwise
def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Args:
        boxes1, [N, 4]
        boxes2, [N, 4]
    Returns:
        iou, [N, ]
        union, [N, ]
    """
    area1 = torchvision.ops.box_area(boxes1)  # [N, ]
    area2 = torchvision.ops.box_area(boxes2)  # [N, ]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N, ]
    union = area1 + area2 - inter
    iou = inter / union
    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Args:
        boxes1, [N, 4] with [x1, y1, x2, y2]
        boxes2, [N, 4] with [x1, y1, x2, y2]
    Returns:
        giou, [N, ]
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    area = wh[:, 0] * wh[:, 1]
    return iou - (area - union) / area


def check_point_inside_box(points: Tensor, boxes: Tensor, eps=1e-9) -> Tensor:
    """
    Args:
        points, [K, 2], (x, y)
        boxes, [N, 4], (x1, y1, y2, y2)
    Returns:
        Tensor (bool), [K, N]
    """
    x, y = [p.unsqueeze(-1) for p in points.unbind(-1)]
    x1, y1, x2, y2 = [x.unsqueeze(0) for x in boxes.unbind(-1)]

    l = x - x1
    t = y - y1
    r = x2 - x
    b = y2 - y

    ltrb = torch.stack([l, t, r, b], dim=-1)
    mask = ltrb.min(dim=-1).values > eps

    return mask


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    """
    Args:
        boxes, [N, 4], (x1, y1, x2, y2)
        points, [N, 2], (x, y)
    Returns:
        Tensor (N, 4), (l, t, r, b)
    """
    x1y1, x2y2 = torch.split(boxes, 2, dim=-1)
    lt = points - x1y1
    rb = x2y2 - points
    return torch.concat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    """
    Args:
        points (Tensor), [N, 2], (x, y)
        distances (Tensor), [N, 4], (l, t, r, b)
    Returns:
        boxes (Tensor),  (N, 4), (x1, y1, x2, y2)
    """
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = -lt + points
    x2y2 = rb + points
    boxes = torch.concat([x1y1, x2y2], dim=-1)
    return boxes
