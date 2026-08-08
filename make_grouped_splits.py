"""Create conservative group-disjoint 70/15/15 splits with mask-ratio balancing."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation_common import DATASETS, Pair, inventory_pairs, serialize_pairs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_ratio(path: Path) -> float:
    values = np.asarray(Image.open(path).convert("L"))
    return float((values > 0).mean())


def choose_split(data_root: Path, pairs: list[Pair], seed: int, attempts: int):
    ratios = {pair.image: mask_ratio(data_root / pair.mask) for pair in pairs}
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.group].append(pair)
    groups = sorted(grouped)
    total = len(pairs)
    target_counts = {"train": 0.70 * total, "validation": 0.15 * total, "test": 0.15 * total}
    global_ratio = sum(ratios[pair.image] for pair in pairs) / total
    best = None
    rng = random.Random(seed)
    for _ in range(attempts):
        order = groups[:]
        rng.shuffle(order)
        buckets = {"train": [], "validation": [], "test": []}
        counts = {key: 0 for key in buckets}
        for group in order:
            size = len(grouped[group])
            candidates = sorted(
                buckets,
                key=lambda key: (counts[key] + size) / target_counts[key],
            )
            chosen = candidates[0]
            buckets[chosen].append(group)
            counts[chosen] += size
        score = 0.0
        split_ratios = {}
        for key in buckets:
            members = [pair for group in buckets[key] for pair in grouped[group]]
            current_ratio = sum(ratios[pair.image] for pair in members) / max(len(members), 1)
            split_ratios[key] = current_ratio
            score += 3.0 * abs(len(members) - target_counts[key]) / total
            score += abs(current_ratio - global_ratio) / max(global_ratio, 1e-8)
        if best is None or score < best[0]:
            best = (score, buckets, counts, split_ratios)
    _, buckets, counts, split_ratios = best
    split_pairs = {
        key: sorted((pair for group in buckets[key] for pair in grouped[group]), key=lambda pair: pair.image)
        for key in buckets
    }
    return split_pairs, grouped, ratios, counts, split_ratios, global_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generated_splits"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--attempts", type=int, default=20000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        pairs = inventory_pairs(args.data_root, dataset)
        split_pairs, grouped, ratios, counts, split_ratios, global_ratio = choose_split(
            args.data_root, pairs, args.seed, args.attempts
        )
        all_hashes = {pair.image: file_sha256(args.data_root / pair.image) for pair in pairs}
        duplicate_hashes = defaultdict(list)
        for image, digest in all_hashes.items():
            duplicate_hashes[digest].append(image)
        duplicate_sets = [images for images in duplicate_hashes.values() if len(images) > 1]
        payload = {
            "dataset": dataset,
            "protocol": "group-disjoint fixed 70/15/15 split",
            "grouping_rule": (
                "All crops from neighboring DJI frames stay in the same 10-frame temporal scene block"
                if dataset == "uav_crack500"
                else "Conservative filename-sequence blocks visually audited against road/background scene characteristics; no official scene IDs were available"
            ),
            "seed": args.seed,
            "total_count": len(pairs),
            "train_count": len(split_pairs["train"]),
            "validation_count": len(split_pairs["validation"]),
            "test_count": len(split_pairs["test"]),
            "group_counts": {key: len({pair.group for pair in value}) for key, value in split_pairs.items()},
            "global_positive_pixel_ratio": global_ratio,
            "split_positive_pixel_ratio": split_ratios,
            "exact_duplicate_sets": duplicate_sets,
            "note": "The test split is never used for checkpoint or threshold selection.",
            "train": serialize_pairs(split_pairs["train"]),
            "validation": serialize_pairs(split_pairs["validation"]),
            "test": serialize_pairs(split_pairs["test"]),
        }
        output = args.output_dir / f"{dataset}_grouped_v2.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"{dataset}: train={payload['train_count']} val={payload['validation_count']} "
            f"test={payload['test_count']} groups={payload['group_counts']} "
            f"ratios={payload['split_positive_pixel_ratio']} output={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
