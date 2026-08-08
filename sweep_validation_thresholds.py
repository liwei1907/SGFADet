"""Diagnostic threshold sweep restricted to validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SGFADataset, verify_cache
from metrics import metrics_from_counts
from sam_adapter import build_model_from_config
from segmentation_common import DATASETS, load_grouped_split


PROJECT = Path(__file__).resolve().parent


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--sam-checkpoint", help="required for online MobileSAM controls")
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    args = parser.parse_args()
    _train, validation, _test, split_info = load_grouped_split(PROJECT / "splits" / f"{args.dataset}_grouped_v2.json")
    verify_cache(Path(args.cache_root), args.dataset, validation)
    loader = DataLoader(SGFADataset(Path(args.data_root), validation, args.dataset, 640, False, Path(args.cache_root)), batch_size=4, shuffle=False)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cuda")
    config = checkpoint.get("config", {})
    model = build_model_from_config(config, sam_checkpoint=args.sam_checkpoint).cuda().eval()
    model.load_state_dict(checkpoint.get("ema", checkpoint.get("model")))
    thresholds = torch.tensor([float(value) for value in args.thresholds.split(",")], device="cuda")
    counts = torch.zeros((len(thresholds), 4), dtype=torch.long, device="cuda")
    for image, target, sam, _names in loader:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            probability = torch.sigmoid(model(image.cuda(), sam.cuda())["logits"][:, 0].float())
        prediction = probability[:, None] >= thresholds[None, :, None, None]
        truth = target.cuda().bool()[:, None]
        counts[:, 0] += (prediction & truth).sum((0, 2, 3))
        counts[:, 1] += (prediction & ~truth).sum((0, 2, 3))
        counts[:, 2] += (~prediction & truth).sum((0, 2, 3))
        counts[:, 3] += (~prediction & ~truth).sum((0, 2, 3))
    rows = []
    for index, threshold in enumerate(thresholds.tolist()):
        row = {"threshold": threshold, **metrics_from_counts(*(int(value) for value in counts[index]))}
        rows.append(row)
    report = {
        "dataset": args.dataset,
        "subset": "validation only",
        "checkpoint": Path(args.checkpoint).name,
        "split_sha256": split_info["sha256"],
        "selection_by_mIoU": max(rows, key=lambda row: row["mIoU"]),
        "selection_by_F1": max(rows, key=lambda row: row["F1"]),
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
