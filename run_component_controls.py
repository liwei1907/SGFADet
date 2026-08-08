"""Run the predeclared component and static-fusion control experiments.

The full model is trained separately by ``run_matrix.py``.  These controls use
the identical grouped split, optimiser, 200-epoch budget, fixed 0.5 threshold,
and seed 42. They isolate the three controlled comparisons reported in the paper:

* SFC versus a conventional squeeze-and-excitation attention block;
* SAF versus element-wise addition and channel concatenation;
* ATAH versus a plain head and a head without deformable convolution.

The legacy component controls (RGB-only, SFC-only, SAF-only) are included so
that the revised table can replace the earlier, non-auditable ablation table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
VARIANTS = (
    "rgb_plain",
    "sfc_only",
    "saf_only",
    "se_attention",
    "sam_add",
    "sam_concat",
    "plain_head",
    "atah_no_dcn",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", default="uav_crack500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "purpose": "paper component and static-fusion controlled alternatives",
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "threshold": 0.5,
        "selection_metric": "validation mIoU",
        "test_policy": "one evaluation after the best validation checkpoint is frozen",
        "variants": list(VARIANTS),
    }
    (root / "ablation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    for index, variant in enumerate(VARIANTS, start=1):
        variant_root = root / variant
        command = [
            sys.executable,
            str(PROJECT / "run_matrix.py"),
            "--data-root", args.data_root,
            "--cache-root", args.cache_root,
            "--run-root", str(variant_root),
            "--datasets", args.dataset,
            "--seeds", str(args.seed),
            "--variant", variant,
            "--epochs", str(args.epochs),
            "--workers", str(args.workers),
            "--batch-size", str(args.batch_size),
            "--eval-batch-size", str(args.eval_batch_size),
        ]
        print(f"[ablation {index}/{len(VARIANTS)}] {variant}", flush=True)
        subprocess.run(command, check=True, cwd=PROJECT)

    print(f"[ablation] completed under {root}", flush=True)


if __name__ == "__main__":
    main()
