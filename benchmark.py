"""Reproducible decoder-only and end-to-end MobileSAM efficiency benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.ops import DeformConv2d

from sam_adapter import verify_mobile_sam_checkpoint
from sgfadet import SGFADet


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def profile_module(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict:
    """Count MACs for convolutions/linears and explicit attention products.

    Linear projections inside attention blocks are counted by ``linear_hook``;
    the attention hooks add only the two matrix products (QK^T and AV).
    """
    macs = 0
    handles = []

    def conv_hook(layer, args, output):
        nonlocal macs
        elements = output.numel()
        kernel = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,) * 2
        macs += elements * (layer.in_channels // layer.groups) * kernel[0] * kernel[1]

    def deform_hook(layer, args, output):
        nonlocal macs
        kernel = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,) * 2
        macs += output.numel() * (layer.in_channels // layer.groups) * kernel[0] * kernel[1]

    def linear_hook(layer, args, output):
        nonlocal macs
        macs += output.numel() * layer.in_features

    def mha_hook(layer, args, output):
        nonlocal macs
        query = args[0]
        if layer.batch_first:
            batch, tokens, channels = query.shape
        else:
            tokens, batch, channels = query.shape
        macs += batch * (4 * tokens * channels * channels + 2 * tokens * tokens * channels)

    def tinyvit_attention_hook(layer, args, output):
        nonlocal macs
        query = args[0]
        batch, tokens, _ = query.shape
        macs += batch * layer.num_heads * tokens * tokens * (layer.key_dim + layer.d)

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, DeformConv2d):
            handles.append(layer.register_forward_hook(deform_hook))
        elif isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))
        elif isinstance(layer, nn.MultiheadAttention):
            handles.append(layer.register_forward_hook(mha_hook))
        elif (
            layer.__class__.__name__ == "Attention"
            and hasattr(layer, "key_dim")
            and hasattr(layer, "d")
            and hasattr(layer, "num_heads")
        ):
            handles.append(layer.register_forward_hook(tinyvit_attention_hook))
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        module(*inputs)
    for handle in handles:
        handle.remove()
    return {"GMACs": macs / 1e9, "GFLOPs_2_per_MAC": 2.0 * macs / 1e9}


def timed_decoder(model, image, prior, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(image, prior)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(image, prior)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(times)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": median,
        "p95_ms": percentile(times, 95),
        "fps_from_median": 1000.0 / median,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / (1024**2),
        "warmup": warmup,
        "iterations": iterations,
    }


def load_mobile_sam(vendor: Path | None, checkpoint: Path):
    verify_mobile_sam_checkpoint(checkpoint)
    if vendor is not None:
        sys.path.insert(0, str(vendor))
    try:
        from mobile_sam import SamPredictor, sam_model_registry
        source = "mobile_sam"
    except ImportError:
        from mobilesam_lite.mobile_sam import SamPredictor, sam_model_registry
        source = "mobilesam_lite.mobile_sam"
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).cuda().eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    return sam, SamPredictor(sam), source


def timed_end_to_end(model, predictor, image_uint8: np.ndarray, warmup: int, iterations: int) -> dict:
    rgb = torch.from_numpy(image_uint8.transpose(2, 0, 1)).float().cuda()[None] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    rgb = (rgb - mean) / std

    def once():
        predictor.set_image(image_uint8)
        prior = predictor.get_image_embedding()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(rgb, prior)

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(times)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": median,
        "p95_ms": percentile(times, 95),
        "fps_from_median": 1000.0 / median,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / (1024**2),
        "warmup": warmup,
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam-vendor", help="optional path to a local MobileSAM source checkout")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--end-to-end-warmup", type=int, default=10)
    parser.add_argument("--end-to-end-iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(Path(args.checkpoint), map_location="cuda")
    config = checkpoint.get("config", {})
    model = SGFADet(variant=config.get("variant", "full")).cuda().eval()
    model.load_state_dict(checkpoint.get("ema", checkpoint.get("model")))
    image = torch.randn(1, 3, args.image_size, args.image_size, device="cuda")
    prior = torch.randn(1, 256, 64, 64, device="cuda")
    decoder_ops = profile_module(model, (image, prior))
    decoder_timing = timed_decoder(model, image, prior, args.warmup, args.iterations)

    vendor = Path(args.sam_vendor).resolve() if args.sam_vendor else None
    sam, predictor, runtime = load_mobile_sam(vendor, Path(args.sam_checkpoint).resolve())
    encoder_size = int(getattr(sam.image_encoder, "img_size", 1024))
    encoder_input = torch.randn(1, 3, encoder_size, encoder_size, device="cuda")
    encoder_ops = profile_module(sam.image_encoder, (encoder_input,))
    end_to_end_ops = {
        "GMACs": decoder_ops["GMACs"] + encoder_ops["GMACs"],
        "GFLOPs_2_per_MAC": decoder_ops["GFLOPs_2_per_MAC"] + encoder_ops["GFLOPs_2_per_MAC"],
    }
    random_image = np.random.default_rng(2026).integers(0, 256, (args.image_size, args.image_size, 3), dtype=np.uint8)
    end_to_end_timing = timed_end_to_end(
        model, predictor, random_image, args.end_to_end_warmup, args.end_to_end_iterations
    )
    result = {
        "checkpoint": Path(args.checkpoint).name,
        "mobile_sam_checkpoint": Path(args.sam_checkpoint).name,
        "hardware": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "input": [1, 3, args.image_size, args.image_size],
        "batch_size": 1,
        "precision": "bfloat16 autocast; deformable convolution forced to float32",
        "decoder_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "mobile_sam_image_encoder_parameter_count": sum(parameter.numel() for parameter in sam.image_encoder.parameters()),
        "mobile_sam_full_checkpoint_parameter_count": sum(parameter.numel() for parameter in sam.parameters()),
        "end_to_end_used_parameter_count": sum(parameter.numel() for parameter in model.parameters())
        + sum(parameter.numel() for parameter in sam.image_encoder.parameters()),
        "decoder_operations": decoder_ops,
        "mobile_sam_image_encoder_operations": encoder_ops,
        "end_to_end_operations": end_to_end_ops,
        "decoder_cached_prior": decoder_timing,
        "end_to_end_mobile_sam_plus_decoder": end_to_end_timing,
        "mobile_sam_runtime": runtime,
        "operation_counting_convention": "Convolution, deformable convolution, linear projection, nn.MultiheadAttention, and TinyViT QK^T/AV matrix products; one multiply-add is two FLOPs. Normalization, activation, softmax, interpolation, and image preprocessing are excluded.",
        "deployment_interpretation": "The cached-prior mode is an offline two-stage pipeline; end-to-end latency is the relevant practical value when every image must be encoded by MobileSAM.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
