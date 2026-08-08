"""Materialize the pre-prediction single-reader scene audit labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent

# Indices were assigned from the contact sheets before model predictions were inspected.
LABELS = {
    "uav_crack500": {
        "pavement_marking": {1, 3, 4, 6, 7, 8, 11, 13, 16, 17, 18, 19, 22, 24, 26, 27, 28, 29, 31, 33, 34, 35, 36, 38, 39, 40, 42, 43, 45, 46, 47},
        "shadow": {1, 2, 4, 8, 9, 11, 12, 14, 19, 20, 21, 25, 28, 30, 32, 34, 38, 43},
        "vehicle": {11, 16, 21, 30, 32},
        "vegetation": {11, 21},
        "stain_or_dark_patch": set(),
        "joint_or_seam": set(),
        "coarse_texture": set(range(1, 49)),
        "dense_crack_network": {1, 4, 5, 6, 8, 10, 13, 15, 18, 22, 24, 26, 27, 28, 29, 31, 33, 35, 36, 40, 45, 46, 47},
    },
    "cracktree260": {
        "pavement_marking": {39},
        "shadow": {1, 2, 3, 4, 5, 6, 29, 30, 31, 34, 35, 36},
        "vehicle": set(),
        "vegetation": set(),
        "stain_or_dark_patch": {1, 2, 3, 4, 5, 35},
        "joint_or_seam": set(),
        "coarse_texture": set(range(19, 29)),
        "dense_crack_network": {28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets", default=str(PROJECT / "artifacts" / "scene_audit" / "sheets"))
    parser.add_argument("--output-dir", default=str(PROJECT / "artifacts" / "scene_audit"))
    args = parser.parse_args()
    sheets = Path(args.sheets).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, category_indices in LABELS.items():
        mapping = json.loads((sheets / f"{dataset}_test_mapping.json").read_text(encoding="utf-8"))
        rows = []
        for item in mapping["images"]:
            row = {"index": item["index"], "image": item["image"], "group": item["group"]}
            for category, indices in category_indices.items():
                row[category] = int(item["index"] in indices)
            row["ordinary_or_uncategorized"] = int(not any(row[category] for category in category_indices))
            rows.append(row)
        csv_path = output_dir / f"{dataset}_scene_labels.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "dataset": dataset,
            "split_sha256": mapping["split_sha256"],
            "annotation_timing": "completed from RGB contact sheets before any final-model prediction was inspected",
            "annotation_type": "single-reader image-level multi-label visual audit",
            "category_counts": {category: len(indices) for category, indices in category_indices.items()},
            "limitations": "Labels describe visible scene content, not pixel-level nuisance masks. Empty categories mean that no clear eligible example was visible in the fixed test split; they must not be interpreted as zero model error.",
            "csv": csv_path.name,
        }
        (output_dir / f"{dataset}_scene_labels.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
