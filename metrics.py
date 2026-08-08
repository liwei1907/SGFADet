"""Pixel, boundary, topology, and distance metrics for binary crack masks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    foreground_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    return {
        "Pr": precision,
        "Re": recall,
        "F1": f1,
        "mIoU": (foreground_iou + background_iou) / 2.0,
        "foreground_IoU": foreground_iou,
        "background_IoU": background_iou,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def update_counts(counts: list[int], prediction: np.ndarray, target: np.ndarray) -> None:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    counts[0] += int(np.logical_and(prediction, target).sum())
    counts[1] += int(np.logical_and(prediction, ~target).sum())
    counts[2] += int(np.logical_and(~prediction, target).sum())
    counts[3] += int(np.logical_and(~prediction, ~target).sum())


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        return np.zeros_like(mask)
    return np.logical_xor(mask, binary_erosion(mask, structure=np.ones((3, 3)), border_value=0))


def boundary_fscore(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> tuple[float, float, float]:
    pred_edge, target_edge = _boundary(prediction), _boundary(target)
    pred_count, target_count = int(pred_edge.sum()), int(target_edge.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0, 0.0, 0.0
    structure = np.ones((3, 3), dtype=bool)
    target_band = binary_dilation(target_edge, structure=structure, iterations=tolerance)
    pred_band = binary_dilation(pred_edge, structure=structure, iterations=tolerance)
    precision = float(np.logical_and(pred_edge, target_band).sum()) / pred_count
    recall = float(np.logical_and(target_edge, pred_band).sum()) / target_count
    fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, fscore


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_edge, target_edge = _boundary(prediction), _boundary(target)
    if not pred_edge.any() and not target_edge.any():
        return 0.0
    if not pred_edge.any() or not target_edge.any():
        return float(math.hypot(*prediction.shape))
    to_target = distance_transform_edt(~target_edge)[pred_edge]
    to_prediction = distance_transform_edt(~pred_edge)[target_edge]
    return float(np.percentile(np.concatenate((to_target, to_prediction)), 95))


def cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    pred_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    topology_precision = float(np.logical_and(pred_skeleton, target).sum()) / max(int(pred_skeleton.sum()), 1)
    topology_sensitivity = float(np.logical_and(target_skeleton, prediction).sum()) / max(int(target_skeleton.sum()), 1)
    return 2.0 * topology_precision * topology_sensitivity / max(
        topology_precision + topology_sensitivity, 1e-12
    )


def thin_skeleton_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    max_width: float = 3.0,
    tolerance: int = 2,
) -> tuple[int, int]:
    """Return matched/total target skeleton pixels whose local width is small."""
    target = target.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)
    if not target.any():
        return 0, 0
    target_skeleton = skeletonize(target)
    local_width = 2.0 * distance_transform_edt(target)
    thin = np.logical_and(target_skeleton, local_width <= max_width)
    total = int(thin.sum())
    if total == 0:
        return 0, 0
    band = binary_dilation(prediction, structure=np.ones((3, 3), dtype=bool), iterations=tolerance)
    return int(np.logical_and(thin, band).sum()), total


def per_image_metrics(prediction: np.ndarray, target: np.ndarray, name: str, tolerance: int = 2) -> dict:
    counts = [0, 0, 0, 0]
    update_counts(counts, prediction, target)
    row = {"image": name, **metrics_from_counts(*counts)}
    bp, br, bf = boundary_fscore(prediction, target, tolerance=tolerance)
    hd = hd95(prediction, target)
    thin_hits, thin_total = thin_skeleton_counts(prediction, target, max_width=3.0, tolerance=tolerance)
    row.update(
        {
            "BoundaryPr": bp,
            "BoundaryRe": br,
            "BoundaryF1": bf,
            "HD95": hd,
            "NormalizedHD95": hd / max(math.hypot(*prediction.shape), 1e-12),
            "clDice": cldice(prediction, target),
            "ThinSkeletonHits": thin_hits,
            "ThinSkeletonPixels": thin_total,
            "ThinSkeletonRecall": thin_hits / thin_total if thin_total else float("nan"),
        }
    )
    return row


def save_detailed_results(rows: list[dict], output_dir: Path, summary: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_mask(mask: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
