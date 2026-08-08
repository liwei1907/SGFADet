"""Cached, online-frozen, zero-prior, and fine-tuned MobileSAM adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sgfadet import SGFADet


FEATURE_CHANNELS = {"layer1": 160, "layer2": 320, "layer3": 320, "neck": 256}
MOBILE_SAM_CHECKPOINT_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mobile_sam_checkpoint(checkpoint: Path) -> str:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists() or checkpoint.stat().st_size < 30_000_000:
        raise FileNotFoundError(f"Valid MobileSAM checkpoint not found: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest.lower() != MOBILE_SAM_CHECKPOINT_SHA256:
        raise RuntimeError(
            "MobileSAM checkpoint SHA-256 does not match the paper protocol: "
            f"expected={MOBILE_SAM_CHECKPOINT_SHA256}, actual={digest}"
        )
    return digest


def load_mobile_sam(checkpoint: Path):
    verify_mobile_sam_checkpoint(checkpoint)
    try:
        from mobile_sam import sam_model_registry
        runtime = "mobile_sam"
    except ImportError:
        from mobilesam_lite.mobile_sam import sam_model_registry
        runtime = "mobilesam_lite.mobile_sam"
    sam = sam_model_registry["vit_t"](checkpoint=str(Path(checkpoint).resolve()))
    return sam, runtime


class ZeroPriorSystem(nn.Module):
    """Parameter-matched decoder control in which every SAM prior is zero."""

    def __init__(self, decoder: SGFADet):
        super().__init__()
        self.decoder = decoder

    def forward(self, image, sam_prior):
        return self.decoder(image, torch.zeros_like(sam_prior))


class OnlineSAMSystem(nn.Module):
    """SGFADet with MobileSAM features computed from the synchronized RGB tensor."""

    def __init__(
        self,
        decoder: SGFADet,
        checkpoint: Path,
        mode: str,
        feature_level: str = "neck",
    ):
        super().__init__()
        if mode not in {"frozen_online", "finetune_last"}:
            raise ValueError(f"Unsupported online SAM mode: {mode}")
        if feature_level not in FEATURE_CHANNELS:
            raise ValueError(f"Unsupported MobileSAM feature level: {feature_level}")
        if mode == "finetune_last" and feature_level != "neck":
            raise ValueError("finetune_last is defined for the final neck feature only")
        sam, self.runtime = load_mobile_sam(checkpoint)
        self.decoder = decoder
        self.image_encoder = sam.image_encoder
        self.sam_mode = mode
        self.feature_level = feature_level
        self.encoder_input_size = int(getattr(self.image_encoder, "img_size", 1024))
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("sam_mean", torch.tensor(SAM_PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("sam_std", torch.tensor(SAM_PIXEL_STD).view(1, 3, 1, 1), persistent=False)

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)
        if mode == "finetune_last":
            for parameter in self.image_encoder.layers[3].parameters():
                parameter.requires_grad_(True)
            for parameter in self.image_encoder.neck.parameters():
                parameter.requires_grad_(True)
        self.image_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Preserve pretrained normalization statistics; gradients still flow
        # through the explicitly unfrozen last stage and neck.
        self.image_encoder.eval()
        return self

    def _preprocess(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        rgb01 = normalized_rgb.float() * self.imagenet_std + self.imagenet_mean
        rgb255 = rgb01.clamp(0.0, 1.0) * 255.0
        resized = F.interpolate(
            rgb255,
            size=(self.encoder_input_size, self.encoder_input_size),
            mode="bilinear",
            align_corners=False,
        )
        return (resized - self.sam_mean) / self.sam_std

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        batch, length, channels = tokens.shape
        side = int(round(length ** 0.5))
        if side * side != length:
            raise RuntimeError(f"MobileSAM token count is not square: {length}")
        return tokens.view(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()

    def _frozen_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.image_encoder.patch_embed(self._preprocess(image))
            x = self.image_encoder.layers[0](x)
            x = self.image_encoder.layers[1](x)
            if self.feature_level == "layer1":
                return self._tokens_to_map(x)
            x = self.image_encoder.layers[2](x)
            if self.feature_level == "layer2":
                return self._tokens_to_map(x)
            x = self.image_encoder.layers[3](x)
            mapped = self._tokens_to_map(x)
            if self.feature_level == "layer3":
                return mapped
            return self.image_encoder.neck(mapped)

    def _finetuned_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.image_encoder.patch_embed(self._preprocess(image))
            x = self.image_encoder.layers[0](x)
            x = self.image_encoder.layers[1](x)
            x = self.image_encoder.layers[2](x)
        x = self.image_encoder.layers[3](x.detach())
        return self.image_encoder.neck(self._tokens_to_map(x))

    def forward(self, image, _cached_prior):
        if self.sam_mode == "finetune_last" and self.training:
            prior = self._finetuned_features(image)
        elif self.sam_mode == "finetune_last":
            # Evaluation does not require gradients but uses the tuned weights.
            with torch.no_grad():
                prior = self._finetuned_features(image)
        else:
            prior = self._frozen_features(image)
        return self.decoder(image, prior)


def build_model(
    *,
    variant: str = "full",
    sam_mode: str = "cache",
    sam_feature_level: str = "neck",
    sam_checkpoint: str | Path | None = None,
    fusion_stages: str = "345",
    backbone: str = "custom",
) -> nn.Module:
    sam_channels = FEATURE_CHANNELS[sam_feature_level] if sam_mode in {"frozen_online", "finetune_last"} else 256
    decoder = SGFADet(
        sam_channels=sam_channels,
        variant=variant,
        fusion_stages=fusion_stages,
        backbone=backbone,
    )
    if sam_mode == "cache":
        return decoder
    if sam_mode == "zero_cache":
        return ZeroPriorSystem(decoder)
    if sam_checkpoint is None:
        raise ValueError(f"sam_checkpoint is required for {sam_mode}")
    return OnlineSAMSystem(decoder, Path(sam_checkpoint), sam_mode, sam_feature_level)


def build_model_from_config(config: dict, sam_checkpoint: str | Path | None = None) -> nn.Module:
    return build_model(
        variant=config.get("variant", "full"),
        sam_mode=config.get("sam_mode", "cache"),
        sam_feature_level=config.get("sam_feature_level", "neck"),
        sam_checkpoint=sam_checkpoint or config.get("sam_checkpoint"),
        fusion_stages=config.get("fusion_stages", "345"),
        backbone=config.get("backbone", "custom"),
    )
