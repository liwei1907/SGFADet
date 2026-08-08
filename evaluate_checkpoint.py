"""One-shot detailed evaluation of a frozen validation-selected checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SGFADataset, verify_cache
from metrics import metrics_from_counts, per_image_metrics, save_detailed_results, save_mask, update_counts
from sam_adapter import build_model_from_config
from segmentation_common import DATASETS, load_grouped_split


PROJECT = Path(__file__).resolve().parent


def prediction_name(image_name: str) -> str:
    suffix = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:8]
    return f"{Path(image_name).stem}_{suffix}.png"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--sam-checkpoint", help="required when evaluating an online MobileSAM control")
    parser.add_argument("--split-file")
    parser.add_argument("--subset", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    split_path = Path(args.split_file).resolve() if args.split_file else PROJECT / "splits" / f"{args.dataset}_grouped_v2.json"
    _train, validation, test, split_info = load_grouped_split(split_path)
    pairs = validation if args.subset == "validation" else test
    verify_cache(Path(args.cache_root), args.dataset, pairs)
    dataset = SGFADataset(Path(args.data_root), pairs, args.dataset, args.image_size, False, Path(args.cache_root))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    config = checkpoint.get("config", {})
    model = build_model_from_config(config, sam_checkpoint=args.sam_checkpoint).to(device).eval()
    state = checkpoint.get("ema", checkpoint.get("model"))
    if state is None:
        raise RuntimeError("Checkpoint contains neither EMA nor model weights")
    model.load_state_dict(state)

    output_dir = Path(args.output_dir).resolve()
    rows = []
    counts = [0, 0, 0, 0]
    for image, target, sam, names in loader:
        image = image.to(device, non_blocking=True)
        sam = sam.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
            probability = torch.sigmoid(model(image, sam)["logits"][:, 0].float())
        predictions = (probability >= args.threshold).cpu().numpy()
        targets = target.numpy().astype(bool)
        for prediction, truth, name in zip(predictions, targets, names):
            update_counts(counts, prediction, truth)
            rows.append(per_image_metrics(prediction, truth, name, args.boundary_tolerance))
            if args.save_predictions:
                save_mask(prediction, output_dir / "predictions" / prediction_name(name))

    summary = metrics_from_counts(*counts)
    summary.update(
        {
            "Dice": summary["F1"],
            "FDR": 1.0 - summary["Pr"],
            "FNR": 1.0 - summary["Re"],
            "BoundaryF1": float(np.mean([row["BoundaryF1"] for row in rows])),
            "BoundaryF1_SD": float(np.std([row["BoundaryF1"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "HD95": float(np.mean([row["HD95"] for row in rows])),
            "HD95_median": float(np.median([row["HD95"] for row in rows])),
            "NormalizedHD95": float(np.mean([row["NormalizedHD95"] for row in rows])),
            "clDice": float(np.mean([row["clDice"] for row in rows])),
            "ThinSkeletonRecall": sum(row["ThinSkeletonHits"] for row in rows)
            / max(sum(row["ThinSkeletonPixels"] for row in rows), 1),
            "ThinSkeletonPixels": int(sum(row["ThinSkeletonPixels"] for row in rows)),
            "thin_width_pixels": 3.0,
            "threshold": args.threshold,
            "boundary_tolerance_pixels": args.boundary_tolerance,
            "dataset": args.dataset,
            "subset": args.subset,
            "image_count": len(rows),
            "checkpoint": Path(args.checkpoint).name,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "training_dataset": config.get("dataset"),
            "variant": config.get("variant", "full"),
            "sam_mode": config.get("sam_mode", "cache"),
            "sam_feature_level": config.get("sam_feature_level", "neck"),
            "fusion_stages": config.get("fusion_stages", "345"),
            "backbone": config.get("backbone", "custom"),
            "split_file": split_path.name,
            "split_sha256": split_info["sha256"],
        }
    )
    save_detailed_results(rows, output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
