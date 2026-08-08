"""Leakage-safe SGFADet training on fixed group-disjoint splits."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SGFADataset, verify_cache
from losses import semantic_loss, sgfadet_loss
from metrics import metrics_from_counts
from sam_adapter import OnlineSAMSystem, build_model
from segmentation_common import DATASETS, load_grouped_split
from sgfadet import SGFADet, VARIANTS


PROJECT = Path(__file__).resolve().parent
FIELDS = [
    "Epoch", "TrainLoss", "ValLoss", "Pr", "Re", "F1", "mIoU",
    "foreground_IoU", "background_IoU", "LR", "Seconds",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, base: Path = PROJECT) -> str:
    """Record a repository-relative path or a basename, never a local absolute path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.name


def initialize_rgb_downsampling_from_resnet34(model: SGFADet, weights_path: Path) -> dict:
    """Initialize only shape-compatible RGB entry convolutions/BNs; topology is unchanged."""
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    targets = (
        (model.stem, "conv1", 32),
        (model.stage2[0], "layer1.0.conv1", 64),
        (model.stage3[0], "layer2.0.conv1", 128),
        (model.stage4[0], "layer3.0.conv1", 256),
        (model.stage5[0], "layer4.0.conv1", 384),
    )
    copied = []
    with torch.no_grad():
        for target, source_key, out_channels in targets:
            source = state[source_key + ".weight"][:out_channels, : target[0].weight.shape[1]]
            if source.shape[-2:] != target[0].weight.shape[-2:]:
                start_h = (source.shape[-2] - target[0].weight.shape[-2]) // 2
                start_w = (source.shape[-1] - target[0].weight.shape[-1]) // 2
                source = source[
                    :,
                    :,
                    start_h : start_h + target[0].weight.shape[-2],
                    start_w : start_w + target[0].weight.shape[-1],
                ]
            if source.shape != target[0].weight.shape:
                raise RuntimeError(f"Initialization shape mismatch for {source_key}: {source.shape} vs {target[0].weight.shape}")
            target[0].weight.copy_(source)
            bn_key = "bn1" if source_key == "conv1" else source_key.rsplit("conv1", 1)[0] + "bn1"
            for attribute in ("weight", "bias", "running_mean", "running_var"):
                getattr(target[1], attribute).copy_(state[f"{bn_key}.{attribute}"][:out_channels])
            copied.append(source_key)
    return {
        "source": Path(weights_path).name,
        "sha256": file_sha256(weights_path),
        "mapping": copied,
        "scope": "five shape-compatible RGB downsampling entry convolutions and batch-normalization layers only",
    }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, updates: int) -> None:
        decay = self.decay * (1.0 - math.exp(-updates / 2000.0))
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1.0 - decay)
            else:
                value.copy_(source[key])


def learning_rate(epoch: int, schedule_epochs: int, base_lr: float, warmup: int, minimum_ratio: float) -> float:
    if epoch <= warmup:
        return base_lr * (0.2 + 0.8 * epoch / max(warmup, 1))
    progress = (epoch - warmup) / max(schedule_epochs - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def autocast_context(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled, dtype=torch.bfloat16)


@torch.inference_mode()
def evaluate(model, loader, device, amp: bool, loss_kwargs: dict) -> dict:
    model.eval()
    counts = [0, 0, 0, 0]
    loss_sum = 0.0
    samples = 0
    for image, target, sam, _names in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sam = sam.to(device, non_blocking=True)
        with autocast_context(amp):
            logits = model(image, sam)["logits"][:, 0]
        loss = semantic_loss(
            logits,
            target,
            pos_power=loss_kwargs["pos_power"],
            pos_cap=loss_kwargs["pos_cap"],
            fixed_pos_weight=loss_kwargs["fixed_pos_weight"],
            dice_weight=loss_kwargs["dice_weight"],
        )
        prediction = logits.float() > 0.0
        truth = target.bool()
        counts[0] += int((prediction & truth).sum())
        counts[1] += int((prediction & ~truth).sum())
        counts[2] += int((~prediction & truth).sum())
        counts[3] += int((~prediction & ~truth).sum())
        loss_sum += float(loss) * image.shape[0]
        samples += image.shape[0]
    return {"ValLoss": loss_sum / max(samples, 1), **metrics_from_counts(*counts)}


def compact_checkpoint(path: Path, epoch: int, ema: ModelEMA, config: dict, best_metrics: dict) -> None:
    torch.save(
        {"epoch": epoch, "ema": ema.model.state_dict(), "config": config, "best_validation": best_metrics},
        path,
    )


def resume_checkpoint(path: Path, epoch: int, model, ema, optimizer, updates: int, best_value: float, config: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "updates": updates,
            "best_value": best_value,
            "config": config,
        },
        path,
    )


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )


def train_dataset(args, dataset_name: str, device: torch.device) -> dict:
    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    split_path = Path(args.split_file).resolve() if args.split_file else PROJECT / "splits" / f"{dataset_name}_grouped_v2.json"
    train_pairs, val_pairs, test_pairs, split_info = load_grouped_split(split_path)
    if split_info.get("dataset") != dataset_name:
        raise RuntimeError(f"Split dataset mismatch: requested={dataset_name}, file={split_info.get('dataset')}")
    verify_cache(cache_root, dataset_name, train_pairs + val_pairs + test_pairs)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "epoch_metrics.csv"
    jsonl_path = run_dir / "training.jsonl"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if csv_path.exists() and args.fresh:
        csv_path.unlink()
    if jsonl_path.exists() and args.fresh:
        jsonl_path.unlink()

    model = build_model(
        variant=args.variant,
        sam_mode=args.sam_mode,
        sam_feature_level=args.sam_feature_level,
        sam_checkpoint=args.sam_checkpoint,
        fusion_stages=args.fusion_stages,
        backbone=args.backbone,
    )
    initialization = {"type": "random"}
    if args.resnet34_init:
        if args.backbone != "custom":
            raise ValueError("--resnet34-init is only supported by the custom RGB backbone")
        decoder = getattr(model, "decoder", model)
        initialization = {
            "type": "ImageNet ResNet-34 partial initialization",
            **initialize_rgb_downsampling_from_resnet34(decoder, Path(args.resnet34_init).resolve()),
        }
    model = model.to(device)
    ema = ModelEMA(model, args.ema_decay)
    if isinstance(model, OnlineSAMSystem) and args.sam_mode == "finetune_last":
        decoder_parameters = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
        sam_parameters = [parameter for parameter in model.image_encoder.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": decoder_parameters, "lr": args.lr, "lr_scale": 1.0},
                {"params": sam_parameters, "lr": args.sam_lr, "lr_scale": args.sam_lr / args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": args.lr, "lr_scale": 1.0}],
            weight_decay=args.weight_decay,
        )
    loss_kwargs = {
        "pos_power": args.pos_power,
        "pos_cap": args.pos_cap,
        "fixed_pos_weight": args.fixed_pos_weight,
        "dice_weight": args.dice_weight,
        "semantic_weight": args.semantic_weight,
        "boundary_weight": args.boundary_weight,
        "side_weight": args.side_weight,
        "boundary_width": args.boundary_width,
    }
    config = {
        "model": "SGFADet",
        "variant": args.variant,
        "sam_mode": args.sam_mode,
        "sam_feature_level": args.sam_feature_level,
        "sam_checkpoint": Path(args.sam_checkpoint).name if args.sam_checkpoint else None,
        "sam_lr": args.sam_lr if args.sam_mode == "finetune_last" else None,
        "fusion_stages": args.fusion_stages,
        "backbone": args.backbone,
        "dataset": dataset_name,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.accumulate,
        "effective_batch_size": args.batch_size * args.accumulate,
        "eval_batch_size": args.eval_batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "minimum_lr_ratio": args.minimum_lr_ratio,
        "schedule_epochs": args.schedule_epochs or args.epochs,
        "loss": loss_kwargs,
        "ema_decay": args.ema_decay,
        "ema_policy": "exponential warm-up ramp to the configured maximum decay",
        "selection_metric": args.selection_metric,
        "evaluation_threshold": 0.5,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "train_count": len(train_pairs),
        "validation_count": len(val_pairs),
        "test_count": len(test_pairs),
        "split_file": portable_path(split_path),
        "split_sha256": split_info["sha256"],
        "test_use_policy": "not evaluated during optimization; one evaluation after best validation checkpoint is frozen",
        "sam_encoder": "not used by the decoder (SAF/SAM path removed)" if args.variant == "no_saf" else {
            "cache": "frozen MobileSAM TinyViT (vit_t), precomputed embeddings",
            "zero_cache": "parameter-matched all-zero prior control",
            "frozen_online": "frozen MobileSAM TinyViT (vit_t), online synchronized RGB encoding",
            "finetune_last": "MobileSAM TinyViT last stage and neck fine-tuned online",
        }[args.sam_mode],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "initialization": initialization,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start_epoch, updates, best_value, best_validation = 1, 0, -1.0, {}
    if last_path.exists() and args.resume and not args.fresh:
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        updates = int(checkpoint.get("updates", 0))
        best_value = float(checkpoint.get("best_value", -1.0))
        if best_path.exists():
            best_validation = torch.load(best_path, map_location="cpu").get("best_validation", {})

    train_set = SGFADataset(data_root, train_pairs, dataset_name, args.image_size, True, cache_root)
    val_set = SGFADataset(data_root, val_pairs, dataset_name, args.image_size, False, cache_root)
    train_loader = make_loader(train_set, args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(val_set, args.eval_batch_size, args.workers, False, args.seed)

    if start_epoch == 1:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
        jsonl_path.write_text("", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        lr = learning_rate(epoch, args.schedule_epochs or args.epochs, args.lr, args.warmup_epochs, args.minimum_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (image, target, sam, _names) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sam = sam.to(device, non_blocking=True)
            with autocast_context(args.amp):
                output = model(image, sam)
                loss = sgfadet_loss(output, target, **loss_kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {dataset_name} epoch {epoch}")
            (loss / args.accumulate).backward()
            if batch_index % args.accumulate == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                ema.update(model, updates)
            losses.append(float(loss.detach()))

        validation = evaluate(ema.model, val_loader, device, args.amp, loss_kwargs)
        row = {
            "Epoch": epoch,
            "TrainLoss": sum(losses) / max(len(losses), 1),
            **validation,
            "LR": lr,
            "Seconds": time.time() - started,
        }
        with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow({field: row[field] for field in FIELDS})
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        selection_value = float(validation[args.selection_metric])
        if selection_value > best_value:
            best_value = selection_value
            best_validation = {"epoch": epoch, **validation}
            compact_checkpoint(best_path, epoch, ema, config, best_validation)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            resume_checkpoint(last_path, epoch, model, ema, optimizer, updates, best_value, config)
        print(
            f"[{dataset_name}/{args.variant}/{args.sam_mode}/seed{args.seed}] {epoch:03d}/{args.epochs} "
            f"loss={row['TrainLoss']:.5f} val_loss={row['ValLoss']:.5f} "
            f"P={row['Pr']:.4f} R={row['Re']:.4f} F1={row['F1']:.4f} "
            f"mIoU={row['mIoU']:.4f} lr={lr:.2e} time={row['Seconds']:.1f}s",
            flush=True,
        )

    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["ema"])
    test_metrics = None
    if not args.skip_test:
        test_set = SGFADataset(data_root, test_pairs, dataset_name, args.image_size, False, cache_root)
        test_loader = make_loader(test_set, args.eval_batch_size, args.workers, False, args.seed)
        test_metrics = evaluate(model, test_loader, device, args.amp, loss_kwargs)
    result = {
        "config": config,
        "best_validation": best_validation,
        "test": test_metrics,
        "test_evaluated": not args.skip_test,
        "best_checkpoint": best_path.name,
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if last_path.exists() and not args.keep_resume_checkpoint:
        last_path.unlink()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--sam-mode", choices=("cache", "zero_cache", "frozen_online", "finetune_last"), default="cache")
    parser.add_argument("--sam-feature-level", choices=("layer1", "layer2", "layer3", "neck"), default="neck")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-lr", type=float, default=1e-5)
    parser.add_argument("--fusion-stages", choices=("3", "4", "5", "34", "35", "45", "345"), default="345")
    parser.add_argument("--backbone", choices=("custom", "resnet18"), default="custom")
    parser.add_argument("--resnet34-init")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.002)
    parser.add_argument("--schedule-epochs", type=int)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--selection-metric", choices=("mIoU", "F1"), default="mIoU")
    parser.add_argument("--pos-power", type=float, default=0.5)
    parser.add_argument("--pos-cap", type=float, default=20.0)
    parser.add_argument("--fixed-pos-weight", type=float)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=0.20)
    parser.add_argument("--boundary-weight", type=float, default=0.10)
    parser.add_argument("--side-weight", type=float, default=0.15)
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--keep-resume-checkpoint", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--nondeterministic", dest="deterministic", action="store_false")
    parser.set_defaults(amp=True, deterministic=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the declared SGFADet experiments")
    seed_everything(args.seed, args.deterministic)
    result = train_dataset(args, args.dataset, torch.device("cuda:0"))
    print(json.dumps({"best_validation": result["best_validation"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
