"""Dataset inventory and auditable grouped train/validation/test splits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


DATASETS = ("uav_crack500", "cracktree260")


@dataclass(frozen=True)
class Pair:
    image: str
    mask: str
    group: str = ""


def _uav_group(filename: str) -> str:
    """Keep neighboring DJI frames in one 10-frame temporal scene block."""
    upper = filename.upper()
    marker = ".JPG_"
    if marker in upper:
        frame = upper.split(marker, 1)[0] + ".JPG"
        match = re.search(r"_(\d{4})_Z\.JPG$", frame)
        if match:
            block = int(match.group(1)) // 10
            parts = frame.split("_")
            timestamp = parts[2] if len(parts) > 2 else "UNKNOWN"
            flight = "_".join(parts[:2] + [timestamp[:8]])
            return f"{flight}_TEMPORAL_{block:03d}X"
        return frame
    return Path(filename).stem


def _cracktree_group(filename: str) -> str:
    """Conservative acquisition-sequence groups inferred from public filenames."""
    stem = Path(filename).stem
    dscn = re.fullmatch(r"DSCN(\d+)", stem, flags=re.IGNORECASE)
    if dscn:
        return f"DSCN_{int(dscn.group(1)) // 10:03d}x"
    liu = re.fullmatch(r"LiuHuan_SXX_(\d+)([a-z]?)", stem, flags=re.IGNORECASE)
    if liu:
        return f"LIU_{int(liu.group(1)) // 10:03d}x"
    numeric = re.fullmatch(r"(\d+)", stem)
    if numeric:
        return f"NUM_{int(numeric.group(1)) // 10:03d}x"
    return stem.upper()


def inventory_pairs(data_root: Path, dataset: str) -> list[Pair]:
    data_root = Path(data_root)
    if dataset == "uav_crack500":
        image_dir = data_root / "UAV-Crack500" / "leftImg8bit" / "train" / "UAV-CrackX4"
        mask_dir = data_root / "UAV-Crack500" / "gtFine" / "train" / "UAV-CrackX4"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
        pairs = []
        for image in images:
            mask = mask_dir / f"{image.stem}.png"
            if not mask.exists():
                raise FileNotFoundError(f"Mask missing for {image}: {mask}")
            pairs.append(
                Pair(
                    image=image.relative_to(data_root).as_posix(),
                    mask=mask.relative_to(data_root).as_posix(),
                    group=_uav_group(image.name),
                )
            )
    elif dataset == "cracktree260":
        image_dir = data_root / "CrackTree260" / "CrackTree260"
        mask_dir = data_root / "CrackTree260" / "gt"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
        pairs = []
        for image in images:
            mask = mask_dir / f"{image.stem}.bmp"
            if not mask.exists():
                raise FileNotFoundError(f"Mask missing for {image}: {mask}")
            pairs.append(
                Pair(
                    image=image.relative_to(data_root).as_posix(),
                    mask=mask.relative_to(data_root).as_posix(),
                    group=_cracktree_group(image.name),
                )
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    expected = 400 if dataset == "uav_crack500" else 260
    if len(pairs) != expected:
        raise RuntimeError(f"{dataset} inventory mismatch: expected {expected}, found {len(pairs)}")
    return pairs


def split_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grouped_split(path: Path) -> tuple[list[Pair], list[Pair], list[Pair], dict]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = [Pair(**item) for item in payload["train"]]
    val = [Pair(**item) for item in payload["validation"]]
    test = [Pair(**item) for item in payload["test"]]
    for left_name, left, right_name, right in (
        ("train", train, "validation", val),
        ("train", train, "test", test),
        ("validation", val, "test", test),
    ):
        left_images = {pair.image for pair in left}
        right_images = {pair.image for pair in right}
        overlap = left_images & right_images
        if overlap:
            raise RuntimeError(f"Image overlap between {left_name} and {right_name}: {next(iter(overlap))}")
        left_groups = {pair.group for pair in left}
        right_groups = {pair.group for pair in right}
        group_overlap = left_groups & right_groups
        if group_overlap:
            raise RuntimeError(f"Group overlap between {left_name} and {right_name}: {next(iter(group_overlap))}")
    payload["sha256"] = split_sha256(path)
    return train, val, test, payload


def serialize_pairs(pairs: list[Pair]) -> list[dict]:
    return [asdict(pair) for pair in pairs]
