"""Join blinded scene labels with final per-image errors and summarize FP/FN evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics import metrics_from_counts


IDENTIFIERS = {"index", "image", "group"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-labels", required=True)
    parser.add_argument("--per-image-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    label_rows = read_csv(Path(args.scene_labels))
    metric_rows = read_csv(Path(args.per_image_metrics))
    labels = {row["image"]: row for row in label_rows}
    metrics = {row["image"]: row for row in metric_rows}
    if set(labels) != set(metrics):
        raise RuntimeError(f"Label/metric image mismatch: labels_only={len(set(labels)-set(metrics))}, metrics_only={len(set(metrics)-set(labels))}")
    categories = [key for key in label_rows[0] if key not in IDENTIFIERS]
    category_results = []
    for category in categories:
        selected = [name for name, row in labels.items() if int(row[category]) == 1]
        if not selected:
            category_results.append({"category": category, "n": 0, "status": "no eligible images in fixed test split"})
            continue
        counts = [sum(int(float(metrics[name][key])) for name in selected) for key in ("TP", "FP", "FN", "TN")]
        result = {"category": category, "n": len(selected), "status": "evaluated", **metrics_from_counts(*counts)}
        result.update(
            {
                "FDR": 1.0 - result["Pr"],
                "FNR": 1.0 - result["Re"],
                "BoundaryF1_mean": float(np.mean([float(metrics[name]["BoundaryF1"]) for name in selected])),
                "HD95_mean": float(np.mean([float(metrics[name]["HD95"]) for name in selected])),
                "clDice_mean": float(np.mean([float(metrics[name]["clDice"]) for name in selected])),
            }
        )
        category_results.append(result)

    def ranking(key: str, descending: bool = True):
        return sorted(
            (
                {
                    "image": name,
                    key: float(row[key]),
                    "Pr": float(row["Pr"]),
                    "Re": float(row["Re"]),
                    "F1": float(row["F1"]),
                    "categories": [category for category in categories if int(labels[name][category]) == 1],
                }
                for name, row in metrics.items()
            ),
            key=lambda item: item[key],
            reverse=descending,
        )[:10]

    report = {
        "category_results": category_results,
        "largest_false_positive_counts": ranking("FP"),
        "largest_false_negative_counts": ranking("FN"),
        "annotation_note": "Scene categories were assigned from RGB contact sheets before final predictions were viewed. They are image-level nuisance tags, not pixel-level causal attribution.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = sorted({key for row in category_results for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(category_results)
    print(json.dumps(category_results, indent=2))


if __name__ == "__main__":
    main()
