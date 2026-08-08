"""Bidirectional zero-fine-tuning evaluation of validation-selected checkpoints."""

from __future__ import annotations

import argparse
import json
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
    protocol = {
        "directions": ["uav_crack500->cracktree260", "cracktree260->uav_crack500"],
        "seeds": list(SEEDS),
        "target_policy": "fixed grouped target test only; no target fine-tuning, validation, threshold calibration, or normalization adaptation",
        "threshold": 0.5,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for source, target in (("uav_crack500", "cracktree260"), ("cracktree260", "uav_crack500")):
        direction = output_root / f"{source}_to_{target}"
        for seed in SEEDS:
            destination = direction / f"seed_{seed}"
            metrics = destination / "metrics.json"
            if metrics.exists() and "ThinSkeletonRecall" in metrics.read_text(encoding="utf-8"):
                print(f"[cross] already complete: {metrics}", flush=True)
                continue
            command = [
                sys.executable,
                str(PROJECT / "evaluate_checkpoint.py"),
                "--dataset", target,
                "--checkpoint", str(run_root / source / f"seed_{seed}" / "best.pt"),
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
                "--input-root", str(direction),
                "--output", str(direction / "aggregate.json"),
            ],
            check=True,
            cwd=PROJECT,
        )


if __name__ == "__main__":
    main()
