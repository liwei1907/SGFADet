"""Dataset and synchronized RGB/mask/SAM-feature augmentation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from sam_adapter import MOBILE_SAM_CHECKPOINT_SHA256


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def cache_key(image_path: str) -> str:
    return hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:20]


class SGFADataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        pairs,
        dataset: str,
        image_size: int,
        augment: bool,
        cache_root: Path,
        return_name: bool = True,
    ):
        self.data_root = Path(data_root)
        self.pairs = pairs
        self.dataset = dataset
        self.image_size = image_size
        self.augment = augment
        self.cache_root = Path(cache_root)
        self.return_name = return_name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        image = Image.open(self.data_root / pair.image).convert("RGB")
        mask = Image.open(self.data_root / pair.mask).convert("L")
        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)

        feature_path = self.cache_root / self.dataset / f"{cache_key(pair.image)}.pt"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing SAM feature: {feature_path}. Run precompute_sam.py first.")
        with feature_path.open("rb") as handle:
            sam = torch.load(handle, map_location="cpu", weights_only=True).float()
        if sam.ndim == 4:
            sam = sam[0]

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                sam = torch.flip(sam, dims=(-1,))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                sam = torch.flip(sam, dims=(-2,))
            turns = random.randrange(4)
            if turns:
                image = image.rotate(90 * turns, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(90 * turns, resample=Image.Resampling.NEAREST)
                sam = torch.rot90(sam, turns, dims=(-2, -1))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        target = (np.asarray(mask) > 0).astype(np.float32)
        sample = ((torch.from_numpy(rgb) - MEAN) / STD, torch.from_numpy(target), sam)
        return (*sample, pair.image) if self.return_name else sample


def verify_cache(cache_root: Path, dataset: str, pairs) -> None:
    missing = [p.image for p in pairs if not (cache_root / dataset / f"{cache_key(p.image)}.pt").exists()]
    if missing:
        raise RuntimeError(f"SAM cache is incomplete for {dataset}: {len(missing)} missing; first={missing[0]}")
    manifest = cache_root / dataset / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"SAM cache manifest is missing: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset:
        raise RuntimeError(f"SAM manifest dataset mismatch: expected={dataset}, actual={payload.get('dataset')}")
    if int(payload.get("image_size", -1)) != 640:
        raise RuntimeError(f"SAM cache image_size must be 640 for the paper protocol: {manifest}")
    if payload.get("feature_shape") != [1, 256, 64, 64]:
        raise RuntimeError(f"Unexpected SAM feature shape in {manifest}: {payload.get('feature_shape')}")
    if str(payload.get("checkpoint_sha256", "")).lower() != MOBILE_SAM_CHECKPOINT_SHA256:
        raise RuntimeError(f"SAM checkpoint hash in cache manifest does not match the paper protocol: {manifest}")
    cached_count = int(payload.get("feature_count", 0))
    if cached_count < len({pair.image for pair in pairs}):
        raise RuntimeError(
            f"SAM manifest count is too small for {dataset}: manifest={cached_count}, "
            f"requested={len({pair.image for pair in pairs})}"
        )
