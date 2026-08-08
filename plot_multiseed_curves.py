"""Plot validation mIoU curves (mean and standard deviation across seeds)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_curve(path: Path) -> np.ndarray:
    by_epoch: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_epoch[int(row["Epoch"])] = float(row["mIoU"])
    if not by_epoch:
        raise ValueError(f"No rows in {path}")
    epochs = sorted(by_epoch)
    if epochs != list(range(1, max(epochs) + 1)):
        raise ValueError(f"Non-contiguous epochs in {path}")
    return np.asarray([by_epoch[epoch] for epoch in epochs], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407, 2025, 2026])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).resolve() / args.dataset
    curves = [load_curve(root / f"seed_{seed}" / "epoch_metrics.csv") for seed in args.seeds]
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        raise ValueError(f"Run lengths disagree: {lengths}")
    values = np.stack(curves)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    epochs = np.arange(1, len(mean) + 1)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(epochs, mean, color="#165D9C", linewidth=1.4, label="mean validation mIoU")
    plt.fill_between(epochs, mean - sd, mean + sd, color="#6BAED6", alpha=0.26, label="±1 SD")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.xlim(1, len(mean))
    plt.grid(alpha=0.22, linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    summary = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "epochs": len(mean),
        "peak_mean_epoch": int(np.argmax(mean) + 1),
        "peak_mean_validation_mIoU": float(mean.max()),
        "final_mean_validation_mIoU": float(mean[-1]),
        "final_sd_validation_mIoU": float(sd[-1]),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
