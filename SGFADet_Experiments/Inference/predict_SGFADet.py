"""Run SGFADet inference and save raw semantic mask PNG files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crack masks with SGFADet.")
    parser.add_argument("--weights", type=str, required=True, help="Trained SGFADet .pt checkpoint.")
    parser.add_argument("--source", type=str, required=True, help="Image/video/dir/glob source accepted by Ultralytics.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help="CUDA device or cpu.")
    parser.add_argument("--project", type=str, default="runs/SGFADet", help="Output project directory.")
    parser.add_argument("--name", type=str, default="predict", help="Run name.")
    parser.add_argument("--save-overlay", action="store_true", help="Also save Ultralytics overlay visualizations.")
    parser.add_argument("--sam-checkpoint", type=str, default="", help="SAM checkpoint path if the model YAML is used directly.")
    parser.add_argument("--sam-backend", choices=["auto", "sam", "edge"], default="auto", help="SAM prior backend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sam_checkpoint:
        os.environ["SGFADET_SAM_CKPT"] = args.sam_checkpoint
    os.environ["SGFADET_SAM_BACKEND"] = args.sam_backend

    save_dir = Path(args.project) / args.name / "masks"
    save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights, task="semantic")

    for result in model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save_overlay,
        stream=True,
        task="semantic",
    ):
        if result.semantic_mask is None:
            continue
        mask = result.semantic_mask.data
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask, dtype=np.uint8)
        Image.fromarray(mask).save(save_dir / (Path(result.path).stem + ".png"))
    print(f"Raw masks saved to {save_dir}")


if __name__ == "__main__":
    main()
