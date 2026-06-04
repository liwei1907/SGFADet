# SGFADet modules. Based on the bundled AGPL-3.0 semantic segmentation engine.
"""SGFADet modules for SAM-guided UAV road crack semantic segmentation.

This file adds a lightweight implementation of the modules described in the
SGFADet paper: SFC, SAF and ATAH.  The SAM branch is implemented as an optional
frozen SAM image encoder when ``segment_anything`` and a checkpoint are
available; otherwise it falls back to a deterministic frozen edge/RGB prior so
that the model remains buildable for unit tests and CPU-only dry runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .conv import Conv

__all__ = "SAMFrozenPyramid", "SFC", "SAF", "ATAH", "ATAHSegment"


def _as_tuple3(channels: int | Iterable[int]) -> tuple[int, int, int]:
    """Normalize an int/list channel specification to a 3-level pyramid tuple."""
    if isinstance(channels, int):
        return (channels, channels, channels)
    values = tuple(int(c) for c in channels)
    if len(values) != 3:
        raise ValueError(f"SAMFrozenPyramid expects exactly 3 output channels, got {values}.")
    return values


def _num_groups(channels: int, max_groups: int = 16) -> int:
    """Return a GroupNorm group count that divides channels."""
    for g in (max_groups, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    """Convolution + GroupNorm + SiLU used inside ATAH."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_num_groups(c2), c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class SAMFrozenPyramid(nn.Module):
    """Frozen SAM/edge prior pyramid for SGFADet.

    Args:
        c1: Input channels, normally 3.
        out_channels: Three output channel sizes for P3/8, P4/16 and P5/32 priors.
        model_type: SAM model type used by ``segment_anything.sam_model_registry``.
        checkpoint: SAM checkpoint path. If empty, ``SGFADET_SAM_CKPT`` is used.
        freeze: Freeze the SAM image encoder. This should stay True to match the paper.
        backend: ``auto`` loads SAM when possible and otherwise uses the edge prior; ``sam`` requires SAM;
            ``edge`` always uses the deterministic fallback prior.
        sam_img_size: Spatial size fed into SAM image encoder.

    Notes:
        The fallback prior is intentionally deterministic and non-pretrained. It is meant to keep the code runnable
        without distributing SAM weights. For reproducing the paper, install ``segment-anything`` and provide a SAM
        checkpoint via ``--sam-checkpoint`` or ``SGFADET_SAM_CKPT``.
    """

    def __init__(
        self,
        c1: int = 3,
        out_channels: Iterable[int] | int = (128, 256, 512),
        model_type: str = "vit_b",
        checkpoint: str | None = "",
        freeze: bool = True,
        backend: str = "auto",
        sam_img_size: int = 1024,
    ):
        super().__init__()
        self.c1 = int(c1)
        self.out_channels = _as_tuple3(out_channels)
        self.model_type = str(model_type)
        self.checkpoint = str(checkpoint or os.getenv("SGFADET_SAM_CKPT", ""))
        self.freeze = bool(freeze)
        self.backend = str(os.getenv("SGFADET_SAM_BACKEND", backend or "auto")).lower()
        self.sam_img_size = int(sam_img_size)
        self.image_encoder = None
        self.sam_dim = 256
        self._warned_fallback = False

        self.register_buffer(
            "pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )

        self.sam_adapters = nn.ModuleList(Conv(self.sam_dim, c, 1, 1) for c in self.out_channels)
        self.edge_adapters = nn.ModuleList(Conv(max(self.c1, 3) + 1, c, 1, 1) for c in self.out_channels)
        self._try_load_sam()

    def _try_load_sam(self) -> None:
        """Load SAM encoder if requested and possible."""
        if self.backend == "edge":
            return
        if not self.checkpoint:
            if self.backend == "sam":
                raise FileNotFoundError("SAM backend was requested but no checkpoint path was provided.")
            return
        checkpoint = Path(self.checkpoint).expanduser()
        if not checkpoint.exists():
            if self.backend == "sam":
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
            LOGGER.warning(f"SAM checkpoint not found: {checkpoint}; SGFADet will use the edge-prior fallback.")
            return
        try:
            from segment_anything import sam_model_registry  # type: ignore

            sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
            self.image_encoder = sam.image_encoder
            self.sam_dim = getattr(self.image_encoder, "out_chans", self.sam_dim)
            if self.sam_dim != 256:
                self.sam_adapters = nn.ModuleList(Conv(self.sam_dim, c, 1, 1) for c in self.out_channels)
            if self.freeze:
                self.image_encoder.eval()
                for p in self.image_encoder.parameters():
                    p.requires_grad_(False)
            LOGGER.info(f"Loaded frozen SAM image encoder ({self.model_type}) from {checkpoint}.")
        except Exception as e:
            if self.backend == "sam":
                raise RuntimeError(f"Unable to load SAM backend: {e}") from e
            LOGGER.warning(f"Unable to load SAM backend ({e}); SGFADet will use the edge-prior fallback.")
            self.image_encoder = None

    def train(self, mode: bool = True):
        """Keep the frozen SAM encoder in eval mode while allowing adapters to follow train/eval mode."""
        super().train(mode)
        if self.image_encoder is not None and self.freeze:
            self.image_encoder.eval()
        return self

    @staticmethod
    def _pyramid_shapes(x: torch.Tensor) -> list[tuple[int, int]]:
        h, w = x.shape[-2:]
        return [(max(h // 8, 1), max(w // 8, 1)), (max(h // 16, 1), max(w // 16, 1)), (max(h // 32, 1), max(w // 32, 1))]

    def _forward_sam(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract a SAM feature map and resize it to P3/P4/P5."""
        assert self.image_encoder is not None
        # Ultralytics images are normalized to 0..1. SAM normalization expects RGB values in 0..255.
        x_sam = F.interpolate(x[:, :3].float(), (self.sam_img_size, self.sam_img_size), mode="bilinear", align_corners=False)
        x_sam = (x_sam * 255.0 - self.pixel_mean) / self.pixel_std
        with torch.no_grad() if self.freeze else torch.enable_grad():
            feat = self.image_encoder(x_sam)
        dtype = self.sam_adapters[0].conv.weight.dtype
        feat = feat.to(device=x.device, dtype=dtype)
        return [adapter(F.interpolate(feat, size=shape, mode="bilinear", align_corners=False)) for adapter, shape in zip(self.sam_adapters, self._pyramid_shapes(x))]

    def _forward_edge_prior(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Build a deterministic RGB + Sobel edge prior pyramid."""
        if not self._warned_fallback:
            LOGGER.info("SGFADet SAM branch is using the edge-prior fallback. Set SGFADET_SAM_CKPT for real SAM priors.")
            self._warned_fallback = True
        rgb = x[:, :3] if x.shape[1] >= 3 else x.repeat(1, 3 // x.shape[1] + 1, 1, 1)[:, :3]
        gray = rgb.mean(1, keepdim=True)
        kx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        ky = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        prior = torch.cat((rgb, edge), dim=1)
        return [adapter(F.interpolate(prior, size=shape, mode="bilinear", align_corners=False)) for adapter, shape in zip(self.edge_adapters, self._pyramid_shapes(x))]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return SAM/edge prior pyramid [P3, P4, P5]."""
        return self._forward_sam(x) if self.image_encoder is not None else self._forward_edge_prior(x)


class SFC(nn.Module):
    """Salient Feature Calibrator.

    Implements the paper's dual-branch, dual-gating and residual recalibration design for RGB backbone features.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16):
        super().__init__()
        c2 = int(c2 or c1)
        hidden = max(c2 // int(reduction), 8)
        self.align = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.f1 = Conv(c2, c2, 1, 1)
        self.f2 = Conv(c2, c2, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(4 * c2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Sequential(
            nn.Conv2d(2 * c2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.align(x)
        f1, f2 = self.f1(residual), self.f2(residual)
        fc = torch.cat((f1, f2), dim=1)
        gap = F.adaptive_avg_pool2d(fc, 1)
        gmp = F.adaptive_max_pool2d(fc, 1)
        w1, w2 = self.gate(torch.cat((gap, gmp), dim=1)).chunk(2, dim=1)
        s1, s2 = self.scale(gap).chunk(2, dim=1)
        out = f1 * w1 * s1 + f2 * w2 * s2 + residual
        return self.out(out)


class SAF(nn.Module):
    """SAM-guided Adaptive Fusion.

    Fuses an RGB feature and a SAM-prior feature using channel attention, cross-branch enhancement and an RGB residual.
    """

    def __init__(self, c_rgb: int, c_sam: int, c2: int, reduction: int = 16):
        super().__init__()
        hidden = max(c2 // int(reduction), 8)
        self.rgb_proj = Conv(c_rgb, c2, 1, 1)
        self.sam_proj = Conv(c_sam, c2, 1, 1)
        self.weight = nn.Sequential(
            nn.Conv2d(2 * c2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(2 * c2, c2, 1, 1)
        self.shortcut = Conv(c_rgb, c2, 1, 1) if c_rgb != c2 else nn.Identity()

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        rgb, sam = x
        if sam.shape[-2:] != rgb.shape[-2:]:
            sam = F.interpolate(sam, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        r = self.rgb_proj(rgb)
        s = self.sam_proj(sam)
        joint = torch.cat((r, s), dim=1)
        omega = self.weight(F.adaptive_avg_pool2d(joint, 1))
        fr, fs = (joint * omega).chunk(2, dim=1)
        r_enh, s_enh = r + fs, s + fr
        return self.out(torch.cat((r_enh, s_enh), dim=1)) + self.shortcut(rgb)


class DeformAlign(nn.Module):
    """Deformable alignment block with a Conv fallback when torchvision deform conv is unavailable."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = int(kernel_size)
        padding = self.kernel_size // 2
        self.offset = nn.Conv2d(channels, 2 * self.kernel_size * self.kernel_size, 3, padding=1)
        self.mask = nn.Conv2d(channels, self.kernel_size * self.kernel_size, 3, padding=1)
        self.deform = None
        try:
            from torchvision.ops import DeformConv2d  # type: ignore

            self.deform = DeformConv2d(channels, channels, self.kernel_size, padding=padding, bias=False)
        except Exception:
            self.deform = None
        self.fallback = nn.Conv2d(channels, channels, self.kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision DeformConv2d is CUDA-oriented and can be very slow on CPU. Use the fallback for CPU dry runs.
        if self.deform is None or not x.is_cuda or os.getenv("SGFADET_USE_DCN", "1") == "0":
            y = self.fallback(x)
        else:
            y = self.deform(x, self.offset(x), torch.sigmoid(self.mask(x)))
        return self.act(self.norm(y))


class ATAH(nn.Module):
    """Adaptive Task-Aware Alignment Head core.

    It aligns semantic discrimination and geometric localization features before producing dense logits.
    """

    def __init__(self, c1: int, c2: int, nc: int):
        super().__init__()
        self.shared = nn.Sequential(ConvGNAct(c1, c2, 3), ConvGNAct(c2, c2, 3))
        self.cls_fc = nn.Sequential(nn.Conv2d(c2, c2, 1), nn.Sigmoid())
        self.reg_fc = nn.Sequential(nn.Conv2d(c2, c2, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(Conv(c2, max(c2 // 2, 16), 3, 1), nn.Conv2d(max(c2 // 2, 16), 1, 1), nn.Sigmoid())
        self.geom = DeformAlign(c2)
        self.fuse = Conv(2 * c2, c2, 1, 1)
        self.pred = nn.Conv2d(c2, nc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fs = self.shared(x)
        z = F.adaptive_avg_pool2d(fs, 1) + F.adaptive_max_pool2d(fs, 1)
        f_cls = fs * self.cls_fc(z) + fs
        f_reg = fs * self.reg_fc(z) + fs
        f_cls = f_cls * self.spatial(f_cls)
        f_reg = self.geom(f_reg)
        return self.pred(self.fuse(torch.cat((f_cls, f_reg), dim=1)))


class ATAHSegment(nn.Module):
    """Semantic segmentation head with multi-scale fusion and ATAH prediction alignment.

    This head is a drop-in semantic head for Ultralytics.  It accepts P3/P4 or P3/P4/P5 neck features and returns
    logits at P3 stride during inference. During training it also returns one auxiliary P4 output for deep supervision.
    """

    export = False
    format = None

    def __init__(self, nc: int = 1, ch: tuple[int, ...] = (), c_mid: int | None = None):
        super().__init__()
        if not ch:
            raise ValueError("ATAHSegment requires at least one input feature channel.")
        self.nc = int(nc)
        self.nl = len(ch)
        self.stride = torch.zeros(self.nl)
        c_mid = int(c_mid or ch[0])
        self.lateral = nn.ModuleList(Conv(c, c_mid, 1, 1) for c in ch)
        self.fuse = Conv(c_mid * len(ch), c_mid, 3, 1)
        self.atah = ATAH(c_mid, c_mid, self.nc)
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3, 1), nn.Conv2d(c_mid, self.nc, 1)) if len(ch) > 1 else None

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = [x]
        target_shape = x[0].shape[-2:]
        feats = []
        for feat, proj in zip(x, self.lateral):
            y = proj(feat)
            if y.shape[-2:] != target_shape:
                y = F.interpolate(y, size=target_shape, mode="bilinear", align_corners=False)
            feats.append(y)
        logits = self.atah(self.fuse(torch.cat(feats, dim=1)))
        if self.training:
            return (logits, self.aux_head(x[1])) if self.aux_head is not None else logits
        if self.export and self.format != "coreml":
            return F.interpolate(logits, scale_factor=8, mode="bilinear", align_corners=False)
        return logits
