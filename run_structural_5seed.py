"""Evaluate all frozen main checkpoints with structural metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
SEEDS = (42, 123, 3407, 2025, 2026)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    for dataset in ("uav_crack500", "cracktree260"):
        dataset_output = output_root / dataset
        for seed in SEEDS:
            destination = dataset_output / f"seed_{seed}"
            metrics = destination / "metrics.json"
            if metrics.exists() and "ThinSkeletonRecall" in metrics.read_text(encoding="utf-8"):
                print(f"[structural] already complete: {metrics}", flush=True)
                continue
            command = [
                sys.executable,
                str(PROJECT / "evaluate_checkpoint.py"),
                "--dataset", dataset,
                "--checkpoint", str(run_root / dataset / f"seed_{seed}" / "best.pt"),
                "--data-root", args.data_root,
                "--cache-root", args.cache_root,
                "--output-dir", str(destination),
                "--batch-size", "4",
            ]
            subprocess.run(command, check=True, cwd=PROJECT)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "aggregate_detailed_runs.py"),
                "--input-root", str(dataset_output),
                "--output", str(dataset_output / "aggregate.json"),
            ],
            check=True,
            cwd=PROJECT,
        )


if __name__ == "__main__":
    main()
