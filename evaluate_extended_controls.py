"""Detailed evaluation and summary for the extended controls."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--component-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    formal_root = Path(args.formal_root).resolve()
    component_root = Path(args.component_root).resolve()

    checkpoints: list[tuple[str, str, Path]] = []
    for control_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        for dataset_dir in sorted(path for path in control_dir.iterdir() if path.is_dir()):
            checkpoints.append((control_dir.name, dataset_dir.name, dataset_dir / "seed_42" / "best.pt"))
    for dataset in ("uav_crack500", "cracktree260"):
        checkpoints.append(("full_cached", dataset, formal_root / dataset / "seed_42" / "best.pt"))
    checkpoints.extend(
        [
            ("p_plain_head", "uav_crack500", component_root / "plain_head" / "uav_crack500" / "seed_42" / "best.pt"),
            ("p_atah_no_dcn", "uav_crack500", component_root / "atah_no_dcn" / "uav_crack500" / "seed_42" / "best.pt"),
        ]
    )

    rows = []
    for control, dataset, checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        destination = output_root / control / dataset / "seed_42"
        metrics_path = destination / "metrics.json"
        if not metrics_path.exists() or "ThinSkeletonRecall" not in metrics_path.read_text(encoding="utf-8"):
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "evaluate_checkpoint.py"),
                    "--dataset", dataset,
                    "--checkpoint", str(checkpoint),
                    "--data-root", args.data_root,
                    "--cache-root", args.cache_root,
                    "--sam-checkpoint", args.sam_checkpoint,
                    "--output-dir", str(destination),
                    "--batch-size", "2",
                ],
                check=True,
                cwd=PROJECT,
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint_payload = __import__("torch").load(checkpoint, map_location="cpu")
        config = checkpoint_payload.get("config", {})
        rows.append(
            {
                "control": control,
                "dataset": dataset,
                "Pr": metrics["Pr"],
                "Re": metrics["Re"],
                "F1": metrics["F1"],
                "mIoU": metrics["mIoU"],
                "foreground_IoU": metrics["foreground_IoU"],
                "BoundaryF1": metrics["BoundaryF1"],
                "HD95": metrics["HD95"],
                "NormalizedHD95": metrics["NormalizedHD95"],
                "clDice": metrics["clDice"],
                "ThinSkeletonRecall": metrics["ThinSkeletonRecall"],
                "parameter_count": config.get("parameter_count"),
                "trainable_parameter_count": config.get("trainable_parameter_count", config.get("parameter_count")),
                "best_epoch": checkpoint_payload.get("epoch"),
                "sam_mode": config.get("sam_mode", "cache"),
                "sam_feature_level": config.get("sam_feature_level", "neck"),
                "fusion_stages": config.get("fusion_stages", "345"),
                "backbone": config.get("backbone", "custom"),
            }
        )
    full_by_dataset = {row["dataset"]: row["mIoU"] for row in rows if row["control"] == "full_cached"}
    for row in rows:
        row["delta_mIoU_vs_full_cached"] = row["mIoU"] - full_by_dataset[row["dataset"]]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
