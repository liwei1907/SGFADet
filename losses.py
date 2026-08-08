"""Class-balanced semantic and boundary objectives for thin cracks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0):
    probability = torch.sigmoid(logits.float())
    target = target.float()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def balanced_bce(logits, target, pos_power=0.5, pos_cap=20.0, fixed_pos_weight=None):
    target = target.float()
    if fixed_pos_weight is None:
        positive = target.sum()
        negative = target.numel() - positive
        ratio = (negative + 1.0) / (positive + 1.0)
        pos_weight = ratio.pow(pos_power).clamp(1.0, pos_cap)
    else:
        pos_weight = torch.as_tensor(fixed_pos_weight, device=logits.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pos_weight)


def focal_tversky(logits, target, alpha=0.3, beta=0.7, gamma=0.75):
    probability = torch.sigmoid(logits.float())
    target = target.float()
    dims = tuple(range(1, probability.ndim))
    tp = (probability * target).sum(dims)
    fp = (probability * (1.0 - target)).sum(dims)
    fn = ((1.0 - probability) * target).sum(dims)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return ((1.0 - score) ** gamma).mean()


def boundary_target(target, width=3):
    target = target[:, None].float()
    pad = width // 2
    dilated = F.max_pool2d(target, width, stride=1, padding=pad)
    eroded = -F.max_pool2d(-target, width, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)[:, 0]


def semantic_loss(logits, target, pos_power=0.5, pos_cap=20.0, fixed_pos_weight=None, dice_weight=0.5):
    bce = balanced_bce(logits, target, pos_power, pos_cap, fixed_pos_weight)
    return (1.0 - dice_weight) * bce + dice_weight * dice_loss(logits, target)


def sgfadet_loss(
    output,
    target,
    pos_power=0.5,
    pos_cap=20.0,
    fixed_pos_weight=None,
    dice_weight=0.5,
    semantic_weight=0.20,
    boundary_weight=0.10,
    side_weight=0.15,
    boundary_width=3,
):
    main = output["logits"][:, 0]
    semantic = output["semantic"][:, 0]
    boundary = output["boundary"][:, 0]
    edge = boundary_target(target, width=boundary_width)
    loss_args = (pos_power, pos_cap, fixed_pos_weight, dice_weight)
    boundary_loss = semantic_loss(boundary, edge, *loss_args)
    sides = output.get("sides", [])
    side_loss = sum(semantic_loss(side[:, 0], target, *loss_args) for side in sides) / max(len(sides), 1)
    return (
        semantic_loss(main, target, *loss_args)
        + semantic_weight * semantic_loss(semantic, target, *loss_args)
        + boundary_weight * boundary_loss
        + side_weight * side_loss
    )
