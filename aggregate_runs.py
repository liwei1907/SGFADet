"""Aggregate independent-seed test results and confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


METRICS = ("Pr", "Re", "F1", "mIoU", "foreground_IoU", "background_IoU")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records = []
    for item in args.runs:
        path = Path(item).resolve()
        payload = json.loads((path / "final_metrics.json").read_text(encoding="utf-8"))
        test = payload.get("test")
        if not test:
            raise RuntimeError(f"No frozen-test result in {path}")
        config = payload["config"]
        records.append(
            {
                "run": str(path),
                "dataset": config["dataset"],
                "variant": config["variant"],
                "seed": config["seed"],
                "best_epoch": payload["best_validation"]["epoch"],
                **{metric: test[metric] for metric in METRICS},
            }
        )
    summary = {"n": len(records), "dataset": records[0]["dataset"], "variant": records[0]["variant"], "metrics": {}}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in records], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half = float(t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))) if len(values) > 1 else 0.0
        summary["metrics"][metric] = {
            "mean": mean,
            "sd": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "values": values.tolist(),
        }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(json.dumps({"runs": records, "summary": summary}, indent=2), encoding="utf-8")
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
