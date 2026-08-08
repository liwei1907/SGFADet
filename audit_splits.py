"""Audit exact and perceptual cross-split duplicates for the fixed protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from segmentation_common import DATASETS, load_grouped_split


PROJECT = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    values = coefficients.flatten()[1:]
    return values > np.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-phash-distance", type=int, default=4)
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split_file).resolve() if args.split_file else PROJECT / "splits" / f"{args.dataset}_grouped_v2.json"
    train, val, test, split_info = load_grouped_split(split_path)
    named = {"train": train, "validation": val, "test": test}
    records = {}
    for split_name, pairs in named.items():
        for pair in pairs:
            path = data_root / pair.image
            records[pair.image] = {
                "split": split_name,
                "group": pair.group,
                "sha256": file_hash(path),
                "phash": phash(path),
            }

    exact = []
    near = []
    names = list(records)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            a, b = records[left], records[right]
            if a["split"] == b["split"]:
                continue
            if a["sha256"] == b["sha256"]:
                exact.append({"left": left, "right": right, "splits": [a["split"], b["split"]]})
            distance = int(np.not_equal(a["phash"], b["phash"]).sum())
            if distance <= args.max_phash_distance:
                near.append(
                    {
                        "left": left,
                        "right": right,
                        "splits": [a["split"], b["split"]],
                        "groups": [a["group"], b["group"]],
                        "phash_distance": distance,
                    }
                )
    report = {
        "dataset": args.dataset,
        "split_file": split_path.name,
        "split_sha256": split_info["sha256"],
        "counts": {key: len(value) for key, value in named.items()},
        "group_counts": {key: len({pair.group for pair in value}) for key, value in named.items()},
        "image_overlap": 0,
        "group_overlap": 0,
        "exact_cross_split_duplicates": exact,
        "perceptual_hash": "64-bit pHash excluding DC coefficient",
        "near_duplicate_threshold": args.max_phash_distance,
        "near_cross_split_pairs": near,
        "interpretation_note": "Near-pair candidates require visual review; pHash similarity alone is not proof of duplicate content.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset", "counts", "group_counts", "image_overlap", "group_overlap")}, indent=2))
    print(f"exact={len(exact)} near_candidates={len(near)} output={output}")


if __name__ == "__main__":
    main()
