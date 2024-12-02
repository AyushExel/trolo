from typing import Dict

import torch
import torchvision
from torch import Tensor
import supervision as sv
from torchvision.ops.boxes import box_area



def to_sv(results: Dict) -> sv.Detections:
    detections = sv.Detections.empty()
    if "boxes" in results:
        boxes_xcycwh = results["boxes"].numpy()
        # Convert from [cx, cy, w, h] to [x0, y0, x1, y1]
        boxes = sv.xcycwh_to_xyxy(boxes_xcycwh)
        scores = results["scores"].numpy()
        labels = results["labels"].numpy()
        detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)
    return detections


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



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
