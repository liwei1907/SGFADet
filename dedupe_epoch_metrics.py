"""Remove duplicated epoch rows introduced by an interrupted/resumed run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for item in args.paths:
        path = Path(item).resolve()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        by_epoch = {int(row["Epoch"]): row for row in rows}
        cleaned = [by_epoch[epoch] for epoch in sorted(by_epoch)]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(cleaned)
        print(f"{path}: {len(rows)} -> {len(cleaned)} rows")


if __name__ == "__main__":
    main()
