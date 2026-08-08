"""Post-freeze synthetic robustness tests with MobileSAM features recomputed per condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from metrics import metrics_from_counts, update_counts
from sam_adapter import verify_mobile_sam_checkpoint
from segmentation_common import DATASETS, load_grouped_split
from sgfadet import SGFADet


PROJECT = Path(__file__).resolve().parent
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def transform(image: Image.Image, condition: str, size: int) -> Image.Image:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    if condition == "clean":
        return image
    if condition.startswith("resolution_"):
        scale = float(condition.split("_")[1])
        small = max(16, round(size * scale))
        return image.resize((small, small), Image.Resampling.BILINEAR).resize((size, size), Image.Resampling.BILINEAR)
    if condition == "gaussian_blur_1p5":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    if condition == "motion_blur_9":
        array = np.asarray(image)
        kernel = np.zeros((9, 9), dtype=np.float32)
        kernel[4, :] = 1.0 / 9.0
        filtered = cv2.filter2D(array, -1, kernel)
        return Image.fromarray(filtered)
    if condition == "low_light_gamma_1p6":
        array = np.asarray(image).astype(np.float32) / 255.0
        return Image.fromarray(np.clip(array**1.6 * 255.0, 0, 255).astype(np.uint8))
    if condition == "bright_gamma_0p65":
        array = np.asarray(image).astype(np.float32) / 255.0
        return Image.fromarray(np.clip(array**0.65 * 255.0, 0, 255).astype(np.uint8))
    if condition == "contrast_0p6":
        return ImageEnhance.Contrast(image).enhance(0.6)
    raise ValueError(condition)


def load_mobile_sam(vendor: Path | None, checkpoint: Path):
    verify_mobile_sam_checkpoint(checkpoint)
    if vendor is not None:
        sys.path.insert(0, str(vendor))
    try:
        from mobile_sam import SamPredictor, sam_model_registry
        runtime = "mobile_sam"
    except ImportError:
        from mobilesam_lite.mobile_sam import SamPredictor, sam_model_registry
        runtime = "mobilesam_lite.mobile_sam"
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).cuda().eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    return SamPredictor(sam), runtime


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--sam-vendor", help="optional path to a local MobileSAM source checkout")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    split_path = Path(args.split_file).resolve() if args.split_file else PROJECT / "splits" / f"{args.dataset}_grouped_v2.json"
    _train, _validation, test, split_info = load_grouped_split(split_path)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cuda")
    config = checkpoint.get("config", {})
    model = SGFADet(variant=config.get("variant", "full")).cuda().eval()
    model.load_state_dict(checkpoint.get("ema", checkpoint.get("model")))
    vendor = Path(args.sam_vendor).resolve() if args.sam_vendor else None
    predictor, runtime = load_mobile_sam(vendor, Path(args.sam_checkpoint).resolve())
    data_root = Path(args.data_root).resolve()
    conditions = (
        "clean",
        "resolution_0.75",
        "resolution_0.50",
        "resolution_0.25",
        "gaussian_blur_1p5",
        "motion_blur_9",
        "low_light_gamma_1p6",
        "bright_gamma_0p65",
        "contrast_0p6",
    )
    results = {}
    mean, std = MEAN.cuda(), STD.cuda()
    for condition in conditions:
        counts = [0, 0, 0, 0]
        for pair in test:
            pil = transform(Image.open(data_root / pair.image).convert("RGB"), condition, args.image_size)
            array = np.ascontiguousarray(np.asarray(pil)).copy()
            predictor.set_image(array)
            prior = predictor.get_image_embedding()
            rgb = torch.from_numpy(array.transpose(2, 0, 1)).float().cuda()[None] / 255.0
            rgb = (rgb - mean) / std
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                probability = torch.sigmoid(model(rgb, prior)["logits"][:, 0].float())
            prediction = (probability[0] >= args.threshold).cpu().numpy()
            mask = Image.open(data_root / pair.mask).convert("L").resize(
                (args.image_size, args.image_size), Image.Resampling.NEAREST
            )
            target = np.asarray(mask) > 0
            update_counts(counts, prediction, target)
        metrics = metrics_from_counts(*counts)
        results[condition] = metrics
        print(f"{condition}: mIoU={metrics['mIoU']:.4f} F1={metrics['F1']:.4f}", flush=True)
    clean = results["clean"]
    for condition, metrics in results.items():
        metrics["delta_mIoU"] = metrics["mIoU"] - clean["mIoU"]
        metrics["delta_F1"] = metrics["F1"] - clean["F1"]
    report = {
        "dataset": args.dataset,
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "image_count": len(test),
        "split_file": split_path.name,
        "split_sha256": split_info["sha256"],
        "threshold": args.threshold,
        "mobile_sam_runtime": runtime,
        "results": results,
        "scope_note": "Resolution reduction is a controlled proxy for loss of spatial detail, not a substitute for metadata-stratified flight-altitude testing. Blur and photometric perturbations are synthetic; pavement-type generalization remains untested because the public files lack reliable strata.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
