"""Evaluate SGFADet on the test split and optionally save predicted masks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

DEFAULT_DATA = ROOT / "SGFADet_Configs/Datasets/Generic_Crack_Semantic.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SGFADet.")
    parser.add_argument("--weights", type=str, required=True, help="Trained SGFADet .pt checkpoint.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Test image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="CUDA device or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--project", type=str, default="runs/SGFADet", help="Output project directory.")
    parser.add_argument("--name", type=str, default="test", help="Run name.")
    parser.add_argument("--save-masks", action="store_true", help="Save predicted PNG masks to the results directory.")
    parser.add_argument("--plots", action="store_true", help="Save metric plots and confusion matrix.")
    parser.add_argument("--sam-checkpoint", type=str, default="", help="SAM checkpoint path if the model YAML is evaluated directly.")
    parser.add_argument("--sam-backend", choices=["auto", "sam", "edge"], default="auto", help="SAM prior backend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sam_checkpoint:
        os.environ["SGFADET_SAM_CKPT"] = args.sam_checkpoint
    os.environ["SGFADET_SAM_BACKEND"] = args.sam_backend

    model = YOLO(args.weights, task="semantic")
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        save_json=args.save_masks,
        plots=args.plots,
        task="semantic",
    )
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
