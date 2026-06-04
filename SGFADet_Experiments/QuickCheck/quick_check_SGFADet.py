"""Build SGFADet and run one random forward pass. No dataset is required."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import SemanticSegmentationModel  # noqa: E402

DEFAULT_MODEL = ROOT / "SGFADet_Configs/Network/SGFADetn_Semantic.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGFADet build/forward sanity check.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="SGFADet model YAML.")
    parser.add_argument("--imgsz", type=int, default=256, help="Random input size. Use multiples of 32.")
    parser.add_argument("--device", type=str, default="cpu", help="Device.")
    parser.add_argument("--sam-backend", choices=["auto", "sam", "edge"], default="edge", help="Use edge backend for lightweight checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["SGFADET_SAM_BACKEND"] = args.sam_backend
    device = torch.device(args.device)
    model = SemanticSegmentationModel(args.model, ch=3, nc=1, verbose=False).to(device).eval()
    x = torch.rand(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape[1] == 1, "Binary SGFADet should output one foreground logit channel."


if __name__ == "__main__":
    main()
