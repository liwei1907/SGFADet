"""Audit completeness and test-use compliance of the extended-control matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "matrix_manifest.json").read_text(encoding="utf-8"))
    records: list[dict] = []
    for control in manifest["controls"]:
        for dataset in control["datasets"]:
            run = root / control["name"] / dataset / "seed_42"
            csv_path = run / "epoch_metrics.csv"
            config_path = run / "config.json"
            final_path = run / "final_metrics.json"
            checkpoint_path = run / "best.pt"
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig"))) if csv_path.exists() else []
            epochs = [int(row["Epoch"]) for row in rows]
            final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            checks = {
                "epoch_count_200": len(rows) == manifest["epochs"],
                "epochs_unique_1_to_200": sorted(set(epochs)) == list(range(1, manifest["epochs"] + 1)),
                "best_checkpoint_exists": checkpoint_path.exists(),
                "final_metrics_exists": final_path.exists(),
                "test_evaluated_once_after_freeze": final.get("test_evaluated") is True
                and config.get("test_use_policy")
                == "not evaluated during optimization; one evaluation after best validation checkpoint is frozen",
                "selection_validation_miou": config.get("selection_metric") == "mIoU",
                "seed_42": config.get("seed") == manifest["seed"],
            }
            records.append(
                {
                    "control": control["name"],
                    "dataset": dataset,
                    "run_dir": str(run),
                    "epochs": len(rows),
                    "best_epoch": final.get("best_validation", {}).get("epoch"),
                    "test_mIoU": final.get("test", {}).get("mIoU"),
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    report = {
        "expected_runs": sum(len(control["datasets"]) for control in manifest["controls"]),
        "audited_runs": len(records),
        "all_pass": all(record["pass"] for record in records),
        "records": records,
    }
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Extended-control run audit",
        "",
        f"- Expected/audited runs: {report['expected_runs']}/{report['audited_runs']}",
        f"- Overall result: {'PASS' if report['all_pass'] else 'FAIL'}",
        "",
        "| Control | Dataset | Epochs | Best epoch | Test mIoU | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        metric = record["test_mIoU"]
        lines.append(
            f"| {record['control']} | {record['dataset']} | {record['epochs']} | "
            f"{record['best_epoch']} | {metric:.6f} | {'PASS' if record['pass'] else 'FAIL'} |"
        )
    (output / "run_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
