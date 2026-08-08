"""Precompute frozen MobileSAM embeddings for every public image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import cache_key
from sam_adapter import MOBILE_SAM_CHECKPOINT_SHA256, verify_mobile_sam_checkpoint
from segmentation_common import DATASETS, inventory_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--sam-vendor", help="optional path to a local MobileSAM source checkout")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM feature precomputation")
    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    checkpoint = Path(args.sam_checkpoint).resolve()
    checkpoint_sha256 = verify_mobile_sam_checkpoint(checkpoint)
    if args.sam_vendor:
        sys.path.insert(0, str(Path(args.sam_vendor).resolve()))
    try:
        from mobile_sam import SamPredictor, sam_model_registry
        runtime_source = "mobile_sam"
    except ImportError:
        from mobilesam_lite.mobile_sam import SamPredictor, sam_model_registry
        runtime_source = "mobilesam_lite.mobile_sam"

    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predictor = SamPredictor(model)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        pairs = inventory_pairs(data_root, dataset)
        output_dir = cache_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for index, pair in enumerate(pairs, 1):
            destination = output_dir / f"{cache_key(pair.image)}.pt"
            if destination.exists() and not args.force:
                continue
            image = Image.open(data_root / pair.image).convert("RGB")
            image = image.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
            predictor.set_image(np.ascontiguousarray(np.asarray(image)))
            feature = predictor.get_image_embedding().detach().cpu().half()
            torch.save(feature, destination)
            if index % 25 == 0 or index == len(pairs):
                print(f"[{dataset}] {index}/{len(pairs)} embeddings", flush=True)
        manifest = {
            "dataset": dataset,
            "feature_count": len(pairs),
            "image_size": args.image_size,
            "feature_shape": [1, 256, 64, 64],
            "feature_dtype": "float16",
            "encoder": "MobileSAM TinyViT (vit_t), frozen",
            "runtime_source": runtime_source,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "expected_checkpoint_sha256": MOBILE_SAM_CHECKPOINT_SHA256,
            "coverage": "all inventoried public images; split-independent cache",
            "seconds": time.time() - started,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{dataset}] cache ready in {manifest['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
