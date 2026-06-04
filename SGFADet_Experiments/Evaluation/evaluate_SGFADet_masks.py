"""Evaluate saved binary crack masks using Precision, Recall, F1 and binary mIoU.

The script compares PNG masks by filename stem. Ground-truth pixels with value 255 are ignored.
For binary crack segmentation, mIoU is computed as mean(IoU_background, IoU_crack), matching common paper reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binary crack segmentation masks.")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing predicted PNG masks.")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory containing ground-truth PNG masks.")
    parser.add_argument("--threshold", type=int, default=127, help="Foreground threshold for 0..255 masks.")
    parser.add_argument("--ignore-label", type=int, default=255, help="Ignore label in GT masks.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save aggregate metrics JSON.")
    parser.add_argument("--save-csv", type=str, default="", help="Optional path to save per-image metrics CSV.")
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    return mask


def to_binary(mask: np.ndarray, threshold: int) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(bool)
    if int(mask.max()) <= 1:
        return mask == 1
    return mask > threshold


def image_metrics(pred: np.ndarray, gt: np.ndarray, threshold: int, ignore_label: int) -> dict[str, float | int]:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = gt != ignore_label
    p = to_binary(pred, threshold)[valid]
    g = to_binary(gt, threshold)[valid]
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return counts_to_metrics(tp, fp, fn, tn) | {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def counts_to_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    crack_iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    miou = (crack_iou + bg_iou) / 2.0
    pix_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mIoU": miou,
        "crack_IoU": crack_iou,
        "background_IoU": bg_iou,
        "pixel_acc": pix_acc,
    }


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pred_files = {p.stem: p for p in pred_dir.glob("*.png")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.png")}
    stems = sorted(pred_files.keys() & gt_files.keys())
    if not stems:
        raise FileNotFoundError(f"No matching PNG mask stems found between {pred_dir} and {gt_dir}.")

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows = []
    for stem in stems:
        metrics = image_metrics(read_mask(pred_files[stem]), read_mask(gt_files[stem]), args.threshold, args.ignore_label)
        for k in totals:
            totals[k] += int(metrics[k])
        rows.append({"image": stem, **metrics})

    aggregate = counts_to_metrics(**totals) | totals | {"images": len(stems)}
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image", "Precision", "Recall", "F1", "mIoU", "crack_IoU", "background_IoU", "pixel_acc", "tp", "fp", "fn", "tn"]
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
