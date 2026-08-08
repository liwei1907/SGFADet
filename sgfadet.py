"""Paper-guided SGFADet for binary road-crack segmentation.

The network follows the paper's RGB backbone + SFC, frozen-SAM prior + SAF,
PAN/FPN neck, and task-aware semantic/boundary alignment head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


class Conv(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        layers = [
            nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.cv1 = Conv(channels, channels, 3)
        self.cv2 = Conv(channels, channels, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3K2(nn.Module):
    """Compact split-transform-merge block used by the RGB path and neck."""

    def __init__(self, c1, c2, depth=1):
        super().__init__()
        hidden = c2 // 2
        self.left = Conv(c1, hidden, 1)
        self.right = Conv(c1, hidden, 1)
        self.blocks = nn.Sequential(*(Bottleneck(hidden) for _ in range(depth)))
        self.out = Conv(hidden * 2, c2, 1)

    def forward(self, x):
        return self.out(torch.cat((self.blocks(self.left(x)), self.right(x)), dim=1))


class SFC(nn.Module):
    """Salient Feature Calibrator from equations (6)-(10) of the paper."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.branch1 = Conv(channels, channels, 1)
        self.branch2 = Conv(channels, channels, 1)
        hidden = max(16, channels // reduction)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1),
            nn.Sigmoid(),
        )
        self.out = Conv(channels, channels, 3)

    def forward(self, x):
        f1, f2 = self.branch1(x), self.branch2(x)
        joined = torch.cat((f1, f2), dim=1)
        avg = F.adaptive_avg_pool2d(joined, 1)
        maximum = F.adaptive_max_pool2d(joined, 1)
        w1, w2 = self.spatial_gate(torch.cat((avg, maximum), dim=1)).chunk(2, dim=1)
        s1, s2 = self.scale(avg).chunk(2, dim=1)
        return self.out(f1 * w1 * s1 + f2 * w2 * s2 + x)


class SEBlock(nn.Module):
    """Generic squeeze-and-excitation control used in replacement experiments."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x) + x


class SPPF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.cv2 = Conv(hidden * 4, c2, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat((x, y1, y2, self.pool(y2)), dim=1))


class PSABlock(nn.Module):
    """Lightweight position-sensitive attention at the P5 resolution."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self.norm1(tokens)
        tokens = tokens + self.attn(q, q, q, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class SAF(nn.Module):
    """SAM-guided Adaptive Fusion from equations (1)-(5) of the paper."""

    def __init__(self, rgb_channels, sam_channels=256, reduction=8):
        super().__init__()
        self.rgb_project = Conv(rgb_channels, rgb_channels, 1)
        self.sam_project = Conv(sam_channels, rgb_channels, 1)
        hidden = max(16, rgb_channels * 2 // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels * 2, 1),
            nn.Sigmoid(),
        )
        self.fuse = Conv(rgb_channels * 2, rgb_channels, 3)

    def forward(self, rgb, sam):
        sam = F.interpolate(sam.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        rgb = self.rgb_project(rgb)
        sam = self.sam_project(sam)
        wrgb, wsam = self.attention(torch.cat((rgb, sam), dim=1)).chunk(2, dim=1)
        attentive_rgb, attentive_sam = rgb * wrgb, sam * wsam
        enhanced_rgb = rgb + attentive_sam
        enhanced_sam = sam + attentive_rgb
        return self.fuse(torch.cat((enhanced_rgb, enhanced_sam), dim=1)) + rgb


class RGBOnlyFusion(nn.Module):
    def __init__(self, rgb_channels, sam_channels=256):
        super().__init__()
        self.project = Conv(rgb_channels, rgb_channels, 1)

    def forward(self, rgb, sam):
        return self.project(rgb)


class AddFusion(nn.Module):
    def __init__(self, rgb_channels, sam_channels=256):
        super().__init__()
        self.rgb_project = Conv(rgb_channels, rgb_channels, 1)
        self.sam_project = Conv(sam_channels, rgb_channels, 1)
        self.out = Conv(rgb_channels, rgb_channels, 3)

    def forward(self, rgb, sam):
        sam = F.interpolate(sam.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.rgb_project(rgb) + self.sam_project(sam))


class ConcatFusion(nn.Module):
    def __init__(self, rgb_channels, sam_channels=256):
        super().__init__()
        self.sam_project = Conv(sam_channels, rgb_channels, 1)
        self.out = Conv(rgb_channels * 2, rgb_channels, 3)

    def forward(self, rgb, sam):
        sam = F.interpolate(sam.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat((rgb, self.sam_project(sam)), dim=1))


class TaskAwareModulator(nn.Module):
    """Joint average/max-pooling channel modulation used by both ATAH tasks."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.out = Conv(channels, channels, 3)

    def forward(self, x):
        gate = torch.sigmoid(
            self.mlp(F.adaptive_avg_pool2d(x, 1))
            + self.mlp(F.adaptive_max_pool2d(x, 1))
        )
        return self.out(x * gate + x)


class GeometryAlignment(nn.Module):
    """DCNv2-style offset/mask alignment for the boundary-localization task."""

    def __init__(self, channels):
        super().__init__()
        self.offset_mask = nn.Conv2d(channels, 27, 3, padding=1)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.deform = DeformConv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        values = self.offset_mask(x)
        offset, mask = values[:, :18], values[:, 18:].sigmoid()
        # torchvision's CUDA deformable convolution is kept in fp32 for stability.
        with torch.cuda.amp.autocast(enabled=False):
            y = self.deform(x.float(), offset.float(), mask.float())
            y = self.project(y)
        return self.act(self.norm(y))


class ATAH(nn.Module):
    """Adaptive Task-Aware Head for semantic and crack-boundary alignment."""

    def __init__(self, channels, deformable=True):
        super().__init__()
        self.shared = nn.Sequential(Conv(channels, channels, 3), Conv(channels, channels, 3))
        self.semantic_mod = TaskAwareModulator(channels)
        self.semantic_spatial = nn.Sequential(nn.Conv2d(channels, 1, 7, padding=3), nn.Sigmoid())
        self.boundary_mod = TaskAwareModulator(channels)
        self.geometry = GeometryAlignment(channels) if deformable else nn.Sequential(
            Conv(channels, channels, 3, g=channels), Conv(channels, channels, 1)
        )
        self.semantic_head = nn.Conv2d(channels, 1, 1)
        self.boundary_head = nn.Conv2d(channels, 1, 1)
        self.fuse = nn.Sequential(Conv(channels * 2, channels, 3), nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        shared = self.shared(x)
        semantic = self.semantic_mod(shared)
        semantic = semantic * self.semantic_spatial(semantic) + semantic
        boundary = self.geometry(self.boundary_mod(shared))
        return self.fuse(torch.cat((semantic, boundary), dim=1)), self.semantic_head(semantic), self.boundary_head(boundary)


class PlainHead(nn.Module):
    """Shared convolutional head used as the non-task-aware control."""

    def __init__(self, channels):
        super().__init__()
        self.shared = nn.Sequential(Conv(channels, channels, 3), Conv(channels, channels, 3))
        self.main = nn.Conv2d(channels, 1, 1)
        self.semantic = nn.Conv2d(channels, 1, 1)
        self.boundary = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        feature = self.shared(x)
        return self.main(feature), self.semantic(feature), self.boundary(feature)


VARIANTS = {
    "full": ("sfc", "saf", "atah"),
    "rgb_plain": ("none", "none", "plain"),
    "sfc_only": ("sfc", "none", "plain"),
    "saf_only": ("none", "saf", "plain"),
    "se_attention": ("se", "none", "plain"),
    "sam_add": ("sfc", "add", "atah"),
    "sam_concat": ("sfc", "concat", "atah"),
    "no_sfc": ("none", "saf", "atah"),
    "no_saf": ("sfc", "none", "atah"),
    "plain_head": ("sfc", "saf", "plain"),
    "atah_no_dcn": ("sfc", "saf", "atah_no_dcn"),
}


class SGFADet(nn.Module):
    def __init__(self, sam_channels=256, variant="full", fusion_stages="345", backbone="custom"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown SGFADet variant: {variant}; choices={tuple(VARIANTS)}")
        if not fusion_stages or any(stage not in "345" for stage in fusion_stages):
            raise ValueError(f"fusion_stages must be a non-empty subset of 345, got {fusion_stages!r}")
        if backbone not in {"custom", "resnet18"}:
            raise ValueError(f"Unknown RGB backbone: {backbone}")
        self.variant = variant
        self.fusion_stages = "".join(stage for stage in "345" if stage in set(fusion_stages))
        self.backbone_kind = backbone
        sfc_kind, fusion_kind, head_kind = VARIANTS[variant]

        def make_sfc(channels):
            if sfc_kind == "sfc":
                return SFC(channels)
            if sfc_kind == "se":
                return SEBlock(channels)
            return nn.Identity()

        fusion_classes = {"saf": SAF, "add": AddFusion, "concat": ConcatFusion, "none": RGBOnlyFusion}

        def make_fusion(stage, channels):
            selected = fusion_kind if str(stage) in self.fusion_stages else "none"
            return fusion_classes[selected](channels, sam_channels)

        if backbone == "custom":
            self.stem = Conv(3, 32, 3, 2)
            self.stage2 = nn.Sequential(Conv(32, 64, 3, 2), C3K2(64, 64, 1))
            self.sfc2 = make_sfc(64)
            self.stage3 = nn.Sequential(Conv(64, 128, 3, 2), C3K2(128, 128, 2), make_sfc(128))
            self.stage4 = nn.Sequential(Conv(128, 256, 3, 2), C3K2(256, 256, 2), make_sfc(256))
            self.stage5 = nn.Sequential(Conv(256, 384, 3, 2), C3K2(384, 384, 1), SPPF(384, 384), PSABlock(384))
        else:
            from torchvision.models import resnet18

            rgb = resnet18(weights=None)
            self.res_stem = nn.Sequential(rgb.conv1, rgb.bn1, rgb.relu)
            self.res_stage2 = nn.Sequential(rgb.maxpool, rgb.layer1)
            self.res_stage3 = rgb.layer2
            self.res_stage4 = rgb.layer3
            self.res_stage5 = rgb.layer4
            self.res_stem_detail = Conv(64, 32, 1)
            self.sfc2 = make_sfc(64)
            self.res_sfc3 = make_sfc(128)
            self.res_sfc4 = make_sfc(256)
            self.res_p5_project = Conv(512, 384, 1)

        self.saf3 = make_fusion(3, 128)
        self.saf4 = make_fusion(4, 256)
        self.saf5 = make_fusion(5, 384)

        self.p5_lateral = Conv(384, 256, 1)
        self.top4 = C3K2(512, 256, 2)
        self.p4_lateral = Conv(256, 128, 1)
        self.top3 = C3K2(256, 128, 2)
        self.down4 = Conv(128, 128, 3, 2)
        self.bottom4 = C3K2(384, 256, 2)
        self.down5 = Conv(256, 256, 3, 2)
        self.bottom5 = C3K2(640, 384, 1)
        self.neck4_project = Conv(256, 128, 1)
        self.neck5_project = Conv(384, 128, 1)
        self.multiscale = C3K2(384, 128, 2)
        self.high_resolution = Conv(128, 64, 1)
        self.detail_project = Conv(64, 32, 1)
        self.detail_fuse = C3K2(96, 64, 1)
        if head_kind == "atah":
            self.head = ATAH(64, deformable=True)
        elif head_kind == "atah_no_dcn":
            self.head = ATAH(64, deformable=False)
        else:
            self.head = PlainHead(64)
        self.half_detail = Conv(32, 16, 1)
        self.half_refine = C3K2(17, 32, 1)
        self.half_head = nn.Conv2d(32, 1, 1)
        self.rgb_detail = nn.Sequential(Conv(3, 8, 3), Conv(8, 8, 3, g=8))
        self.full_refine = nn.Sequential(Conv(9, 16, 3), nn.Conv2d(16, 1, 1))
        self.side_stem = nn.Conv2d(32, 1, 1)
        self.side3 = nn.Conv2d(128, 1, 1)
        self.side4 = nn.Conv2d(256, 1, 1)
        self.side5 = nn.Conv2d(384, 1, 1)
        self.side_fuse = nn.Conv2d(5, 1, 1)
        with torch.no_grad():
            self.side_fuse.weight.copy_(torch.tensor([0.60, 0.15, 0.10, 0.08, 0.07]).view(1, 5, 1, 1))
            self.side_fuse.bias.zero_()

    def forward(self, image, sam_prior):
        input_size = image.shape[-2:]
        if self.backbone_kind == "custom":
            stem_detail = self.stem(image)
            x = self.sfc2(self.stage2(stem_detail))
            p3 = self.saf3(self.stage3(x), sam_prior)
            p4 = self.saf4(self.stage4(p3), sam_prior)
            p5 = self.saf5(self.stage5(p4), sam_prior)
        else:
            stem = self.res_stem(image)
            stem_detail = self.res_stem_detail(stem)
            x = self.sfc2(self.res_stage2(stem))
            p3 = self.saf3(self.res_sfc3(self.res_stage3(x)), sam_prior)
            p4 = self.saf4(self.res_sfc4(self.res_stage4(p3)), sam_prior)
            p5 = self.saf5(self.res_p5_project(self.res_stage5(p4)), sam_prior)

        n5 = self.p5_lateral(p5)
        up5 = F.interpolate(n5.float(), size=p4.shape[-2:], mode="nearest").to(p4.dtype)
        n4 = self.top4(torch.cat((up5, p4), dim=1))
        n4_small = self.p4_lateral(n4)
        up4 = F.interpolate(n4_small.float(), size=p3.shape[-2:], mode="nearest").to(p3.dtype)
        n3 = self.top3(torch.cat((up4, p3), dim=1))
        o4 = self.bottom4(torch.cat((self.down4(n3), n4), dim=1))
        o5 = self.bottom5(torch.cat((self.down5(o4), p5), dim=1))
        up_o4 = F.interpolate(
            self.neck4_project(o4).float(), size=n3.shape[-2:], mode="bilinear", align_corners=False
        ).to(n3.dtype)
        up_o5 = F.interpolate(
            self.neck5_project(o5).float(), size=n3.shape[-2:], mode="bilinear", align_corners=False
        ).to(n3.dtype)
        fused = self.multiscale(torch.cat((n3, up_o4, up_o5), dim=1))
        high = F.interpolate(
            self.high_resolution(fused).float(), size=x.shape[-2:], mode="bilinear", align_corners=False
        ).to(x.dtype)
        detail = self.detail_fuse(torch.cat((high, self.detail_project(x)), dim=1))
        logits, semantic, boundary = self.head(detail)
        half_coarse = F.interpolate(
            logits.float(), size=stem_detail.shape[-2:], mode="bilinear", align_corners=False
        ).to(stem_detail.dtype)
        half_feature = self.half_refine(torch.cat((half_coarse, self.half_detail(stem_detail)), dim=1))
        half_logits = self.half_head(half_feature) + half_coarse
        full_coarse = F.interpolate(
            half_logits.float(), size=input_size, mode="bilinear", align_corners=False
        )
        rgb_detail = self.rgb_detail(image)
        refined = self.full_refine(torch.cat((full_coarse.to(rgb_detail.dtype), rgb_detail), dim=1)) + full_coarse
        semantic = F.interpolate(semantic.float(), size=input_size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary.float(), size=input_size, mode="bilinear", align_corners=False)
        sides = [
            F.interpolate(layer(feature).float(), size=input_size, mode="bilinear", align_corners=False)
            for layer, feature in ((self.side_stem, stem_detail), (self.side3, p3), (self.side4, p4), (self.side5, p5))
        ]
        logits = self.side_fuse(torch.cat((refined.float(), *sides), dim=1))
        return {"logits": logits.float(), "semantic": semantic, "boundary": boundary, "sides": sides}
