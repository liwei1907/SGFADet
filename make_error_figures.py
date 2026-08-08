"""Create top false-positive/false-negative overlays from final frozen predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/calibri.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def prediction_name(image_name: str) -> str:
    suffix = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:8]
    return f"{Path(image_name).stem}_{suffix}.png"


def overlay(image: Image.Image, prediction: np.ndarray, target: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    pred_only = np.logical_and(prediction, ~target)
    missed = np.logical_and(~prediction, target)
    correct = np.logical_and(prediction, target)
    colors = ((pred_only, np.array([255, 32, 32])), (missed, np.array([32, 96, 255])), (correct, np.array([32, 230, 80])))
    for mask, color in colors:
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--per-image-metrics", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scene-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    prediction_dir = Path(args.predictions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.per_image_metrics).open("r", encoding="utf-8-sig", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    with Path(args.scene_labels).open("r", encoding="utf-8-sig", newline="") as handle:
        label_rows = {row["image"]: row for row in csv.DictReader(handle)}

    for metric_key, title, filename in (
        ("FP", "Largest false-positive pixel counts", "top_false_positives.png"),
        ("FN", "Largest false-negative pixel counts", "top_false_negatives.png"),
    ):
        selected = sorted(metric_rows, key=lambda row: float(row[metric_key]), reverse=True)[: args.top_k]
        columns, tile, label_height, title_height = 4, 330, 64, 70
        rows = (len(selected) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * tile, title_height + rows * (tile + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), title, fill="black", font=font(28))
        draw.text((16, 43), "green=true positive; red=false positive; blue=false negative", fill="#333333", font=font(18))
        for index, row in enumerate(selected):
            image_name = row["image"]
            rgb = Image.open(data_root / image_name).convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
            target_path = image_name.replace("leftImg8bit", "gtFine") if "leftImg8bit" in image_name else None
            if target_path:
                target_path = str(Path(target_path).with_suffix(".png"))
            else:
                target_path = image_name.replace("CrackTree260/CrackTree260", "CrackTree260/gt")
                target_path = str(Path(target_path).with_suffix(".bmp"))
            target = np.asarray(Image.open(data_root / target_path).convert("L").resize((640, 640), Image.Resampling.NEAREST)) > 0
            prediction = np.asarray(Image.open(prediction_dir / prediction_name(image_name)).convert("L")) > 0
            visual = overlay(rgb, prediction, target).resize((tile, tile), Image.Resampling.LANCZOS)
            y, x = divmod(index, columns)
            left, top = x * tile, title_height + y * (tile + label_height)
            canvas.paste(visual, (left, top))
            categories = [key for key, value in label_rows[image_name].items() if key not in {"index", "image", "group"} and value == "1"]
            draw.text((left + 5, top + tile + 4), f"{Path(image_name).stem[:31]}", fill="black", font=font(15))
            draw.text((left + 5, top + tile + 25), f"FP={int(float(row['FP'])):,}  FN={int(float(row['FN'])):,}", fill="black", font=font(15))
            draw.text((left + 5, top + tile + 45), ", ".join(categories[:3]) or "ordinary", fill="#444444", font=font(13))
        canvas.save(output_dir / filename)
    print(f"saved error figures to {output_dir}")


if __name__ == "__main__":
    main()
