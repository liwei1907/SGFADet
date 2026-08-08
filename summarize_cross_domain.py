"""Combine within-domain and zero-shot aggregates into a domain-gap table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ("Pr", "Re", "F1", "mIoU", "foreground_IoU", "BoundaryF1", "HD95", "clDice", "ThinSkeletonRecall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-root", required=True)
    parser.add_argument("--cross-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    structural = Path(args.structural_root).resolve()
    cross = Path(args.cross_root).resolve()
    rows = []
    for source, target in (("uav_crack500", "cracktree260"), ("cracktree260", "uav_crack500")):
        # A domain gap must compare two models on the same target test set:
        # target-trained (within-domain) versus source-trained (zero-shot).
        # Comparing against the source-domain test result would mix datasets and
        # make the reported difference uninterpretable.
        target_within = json.loads((structural / target / "aggregate.json").read_text(encoding="utf-8"))
        target_cross = json.loads((cross / f"{source}_to_{target}" / "aggregate.json").read_text(encoding="utf-8"))
        row = {"source": source, "target": target}
        for field in FIELDS:
            within = float(target_within["metrics"][field]["mean"])
            transferred = float(target_cross["metrics"][field]["mean"])
            row[f"within_{field}"] = within
            row[f"cross_{field}"] = transferred
            row[f"absolute_change_{field}"] = transferred - within
        rows.append(row)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "domain_gap.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "domain_gap.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
