"""Run or resume the frozen independent-seed SGFADet experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["uav_crack500", "cracktree260"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407, 2025, 2026])
    parser.add_argument("--variant", default="full")
    parser.add_argument("--resnet34-init")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "variant": args.variant,
        "epochs": args.epochs,
        "configuration_status": "frozen before any formal test evaluation",
        "resnet34_init": args.resnet34_init,
    }
    (root / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = len(args.datasets) * len(args.seeds)
    index = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            index += 1
            run_dir = root / dataset / f"seed_{seed}"
            final = run_dir / "final_metrics.json"
            if final.exists():
                payload = json.loads(final.read_text(encoding="utf-8"))
                if payload.get("test_evaluated") == (not args.skip_test):
                    print(f"[matrix {index}/{total}] already complete: {run_dir}", flush=True)
                    continue
            command = [
                sys.executable,
                str(PROJECT / "train.py"),
                "--dataset", dataset,
                "--data-root", args.data_root,
                "--cache-root", args.cache_root,
                "--run-dir", str(run_dir),
                "--variant", args.variant,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--workers", str(args.workers),
                "--seed", str(seed),
                "--save-every", "25",
                "--resume",
                "--keep-resume-checkpoint",
                "--nondeterministic",
            ]
            if args.resnet34_init:
                command.extend(["--resnet34-init", args.resnet34_init])
            if args.skip_test:
                command.append("--skip-test")
            print(f"[matrix {index}/{total}] starting: {dataset} seed={seed}", flush=True)
            subprocess.run(command, check=True, cwd=PROJECT)
    print(f"[matrix] completed {total} runs under {root}", flush=True)


if __name__ == "__main__":
    main()
