"""Train SGFADet on a binary crack semantic segmentation dataset.

Example:
    python SGFADet_Experiments/Train/train_SGFADet.py \
        --data SGFADet_Configs/Datasets/CUAV_Crack500.yaml \
        --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
        --device 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

DEFAULT_MODEL = ROOT / "SGFADet_Configs/Network/SGFADetn_Semantic.yaml"
DEFAULT_DATA = ROOT / "SGFADet_Configs/Datasets/Generic_Crack_Semantic.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SGFADet for UAV road crack semantic segmentation.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="SGFADet model YAML or checkpoint.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs; paper setting is 100.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size; paper setting is 4.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size; paper setting is 640.")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer; paper setting is Adam.")
    parser.add_argument("--lr0", type=float, default=5e-4, help="Initial learning rate; paper setting is 5e-4.")
    parser.add_argument("--weight-decay", type=float, default=3e-5, help="Weight decay; paper setting is 3e-5.")
    parser.add_argument("--device", type=str, default=None, help="CUDA device, e.g. 0 or 0,1. Use cpu for CPU.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--project", type=str, default="runs/SGFADet", help="Output project directory.")
    parser.add_argument("--name", type=str, default="train", help="Run name.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint in the run directory.")
    parser.add_argument("--pretrained", type=str, default=None, help="Optional pretrained checkpoint to load.")
    parser.add_argument("--sam-checkpoint", type=str, default="", help="SAM checkpoint path. Also accepted via SGFADET_SAM_CKPT.")
    parser.add_argument("--sam-backend", choices=["auto", "sam", "edge"], default="auto", help="SAM prior backend.")
    parser.add_argument("--degrees", type=float, default=180.0, help="Rotation augmentation range in degrees.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability.")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic probability. Kept 0 for paper-style simple augmentation.")
    parser.add_argument("--plots", action="store_true", help="Save training/validation plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sam_checkpoint:
        os.environ["SGFADET_SAM_CKPT"] = args.sam_checkpoint
    os.environ["SGFADET_SAM_BACKEND"] = args.sam_backend

    model = YOLO(args.model, task="semantic")
    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        degrees=args.degrees,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
        task="semantic",
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
