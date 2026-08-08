"""Predeclared semantic-prior and architecture-control matrix."""

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
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    controls = [
        # Semantic-prior controls on both datasets with the same decoder and epoch budget.
        {"name": "m_no_sam", "datasets": ["uav_crack500", "cracktree260"], "variant": "no_saf", "sam_mode": "cache"},
        {"name": "m_zero_prior", "datasets": ["uav_crack500", "cracktree260"], "variant": "full", "sam_mode": "zero_cache"},
        {"name": "m_frozen_online", "datasets": ["uav_crack500", "cracktree260"], "variant": "full", "sam_mode": "frozen_online"},
        {"name": "m_finetune_last", "datasets": ["uav_crack500", "cracktree260"], "variant": "full", "sam_mode": "finetune_last"},
        # One-factor architecture controls on the primary UAV dataset.
        {"name": "p_no_sfc", "datasets": ["uav_crack500"], "variant": "no_sfc", "sam_mode": "cache"},
        {"name": "p_fusion_p3", "datasets": ["uav_crack500"], "variant": "full", "sam_mode": "cache", "fusion_stages": "3"},
        {"name": "p_fusion_p4", "datasets": ["uav_crack500"], "variant": "full", "sam_mode": "cache", "fusion_stages": "4"},
        {"name": "p_fusion_p5", "datasets": ["uav_crack500"], "variant": "full", "sam_mode": "cache", "fusion_stages": "5"},
        {"name": "p_sam_layer1", "datasets": ["uav_crack500"], "variant": "full", "sam_mode": "frozen_online", "sam_feature_level": "layer1"},
        {"name": "p_sam_layer2", "datasets": ["uav_crack500"], "variant": "full", "sam_mode": "frozen_online", "sam_feature_level": "layer2"},
        {"name": "p_resnet18", "datasets": ["uav_crack500"], "variant": "full", "sam_mode": "cache", "backbone": "resnet18"},
    ]
    manifest = {
        "protocol": "semantic-prior and architecture controls",
        "seed": 42,
        "epochs": 200,
        "controls": controls,
        "selection": "validation mIoU only",
        "test_policy": "one fixed-test evaluation after checkpoint freeze",
        "status": "predeclared before these control runs",
    }
    (root / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tasks = [(control, dataset) for control in controls for dataset in control["datasets"]]
    for index, (control, dataset) in enumerate(tasks, start=1):
        run_dir = root / control["name"] / dataset / "seed_42"
        final = run_dir / "final_metrics.json"
        if final.exists() and json.loads(final.read_text(encoding="utf-8")).get("test_evaluated"):
            print(f"[extended {index}/{len(tasks)}] already complete: {run_dir}", flush=True)
            continue
        online = control["sam_mode"] in {"frozen_online", "finetune_last"}
        command = [
            sys.executable,
            str(PROJECT / "train.py"),
            "--dataset", dataset,
            "--data-root", args.data_root,
            "--cache-root", args.cache_root,
            "--sam-checkpoint", args.sam_checkpoint,
            "--run-dir", str(run_dir),
            "--variant", control["variant"],
            "--sam-mode", control["sam_mode"],
            "--sam-feature-level", control.get("sam_feature_level", "neck"),
            "--fusion-stages", control.get("fusion_stages", "345"),
            "--backbone", control.get("backbone", "custom"),
            "--epochs", "200",
            "--batch-size", "2" if online else "6",
            "--accumulate", "3" if online else "1",
            "--eval-batch-size", "2" if online else "4",
            "--workers", str(args.workers),
            "--seed", "42",
            "--save-every", "25",
            "--resume",
            "--keep-resume-checkpoint",
        ]
        print(f"[extended {index}/{len(tasks)}] starting {control['name']} on {dataset}", flush=True)
        subprocess.run(command, check=True, cwd=PROJECT)


if __name__ == "__main__":
    main()
