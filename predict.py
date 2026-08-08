"""Predict binary crack masks for one image or a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam_adapter import load_mobile_sam
from sgfadet import SGFADet


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(source)


def sam_predictor(model):
    try:
        from mobile_sam import SamPredictor
    except ImportError:
        from mobilesam_lite.mobile_sam import SamPredictor
    return SamPredictor(model)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="validation-selected SGFADet checkpoint")
    parser.add_argument("--sam-checkpoint", required=True, help="paper-aligned mobile_sam.pt")
    parser.add_argument("--source", required=True, help="image file or non-recursive image directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=640, choices=(640,))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    config = checkpoint.get("config", {})
    if config.get("sam_mode", "cache") != "cache":
        raise ValueError("predict.py expects a main cached-prior decoder checkpoint")
    model = SGFADet(
        variant=config.get("variant", "full"),
        fusion_stages=config.get("fusion_stages", "345"),
        backbone=config.get("backbone", "custom"),
    ).to(device).eval()
    state = checkpoint.get("ema", checkpoint.get("model"))
    if state is None:
        raise RuntimeError("Checkpoint contains neither EMA nor model weights")
    model.load_state_dict(state)

    sam, runtime = load_mobile_sam(Path(args.sam_checkpoint))
    sam = sam.to(device).eval()
    predictor = sam_predictor(sam)
    mean, std = MEAN.to(device), STD.to(device)
    output_dir = Path(args.output_dir).resolve()
    mask_dir, overlay_dir = output_dir / "masks", output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(Path(args.source).resolve())
    if not paths:
        raise RuntimeError(f"No supported images found under {args.source}")
    for path in paths:
        original = Image.open(path).convert("RGB")
        resized = original.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
        array = np.ascontiguousarray(np.asarray(resized))
        predictor.set_image(array)
        prior = predictor.get_image_embedding().to(device)
        rgb = torch.from_numpy(array.transpose(2, 0, 1)).float().to(device)[None] / 255.0
        rgb = (rgb - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(rgb, prior)["logits"][:, 0].float())[0]
        mask_640 = (probability >= args.threshold).cpu().numpy().astype(np.uint8) * 255
        mask = Image.fromarray(mask_640, mode="L").resize(original.size, Image.Resampling.NEAREST)
        mask.save(mask_dir / f"{path.stem}.png")

        base = np.asarray(original).copy()
        selected = np.asarray(mask) > 0
        overlay = base.astype(np.float32)
        overlay[selected] = 0.55 * overlay[selected] + 0.45 * np.array([255, 40, 40], dtype=np.float32)
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_dir / f"{path.stem}.png")
        print(f"predicted {path.name}", flush=True)

    print(f"saved {len(paths)} masks and overlays to {output_dir} using {runtime}")


if __name__ == "__main__":
    main()
