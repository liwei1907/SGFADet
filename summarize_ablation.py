"""Summarize the single-seed component controls into CSV and JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "Pr",
    "Re",
    "F1",
    "mIoU",
    "foreground_IoU",
    "background_IoU",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", default="uav_crack500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    full_payload = json.loads(Path(args.full_metrics).read_text(encoding="utf-8"))
    full = full_payload.get("test", full_payload)
    rows.append({"variant": "full", **{field: full[field] for field in FIELDS}})

    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        path = variant_dir / args.dataset / f"seed_{args.seed}" / "final_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed control: {path}")
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        payload = raw_payload.get("test", raw_payload)
        rows.append(
            {"variant": variant_dir.name, **{field: payload[field] for field in FIELDS}}
        )

    full_miou = float(rows[0]["mIoU"])
    for row in rows:
        row["delta_mIoU_vs_full"] = float(row["mIoU"]) - full_miou

    with (output / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", *FIELDS, "delta_mIoU_vs_full"))
        writer.writeheader()
        writer.writerows(rows)
    (output / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
