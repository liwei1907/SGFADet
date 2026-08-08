"""Aggregate detailed per-seed metrics with sample SD and t confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


DEFAULT_FIELDS = (
    "Pr", "Re", "F1", "mIoU", "foreground_IoU", "background_IoU",
    "BoundaryF1", "HD95", "NormalizedHD95", "clDice", "ThinSkeletonRecall",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407, 2025, 2026])
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    args = parser.parse_args()

    records = []
    root = Path(args.input_root).resolve()
    for seed in args.seeds:
        path = root / f"seed_{seed}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append({"seed": seed, **{field: float(payload[field]) for field in args.fields}})
    summary = {"n": len(records), "seeds": args.seeds, "records": records, "metrics": {}}
    critical = float(t.ppf(0.975, df=len(records) - 1)) if len(records) > 1 else 0.0
    for field in args.fields:
        values = np.asarray([record[field] for record in records], dtype=np.float64)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
        summary["metrics"][field] = {
            "mean": mean,
            "sample_sd": sd,
            "ci95": [mean - half, mean + half],
            "values": values.tolist(),
        }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record", "seed", *args.fields])
        writer.writeheader()
        for record in records:
            writer.writerow({"record": "seed", **record})
        writer.writerow({"record": "mean", **{field: summary["metrics"][field]["mean"] for field in args.fields}})
        writer.writerow({"record": "sample_sd", **{field: summary["metrics"][field]["sample_sd"] for field in args.fields}})
        writer.writerow({"record": "ci95_low", **{field: summary["metrics"][field]["ci95"][0] for field in args.fields}})
        writer.writerow({"record": "ci95_high", **{field: summary["metrics"][field]["ci95"][1] for field in args.fields}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
