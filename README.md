# SGFADet: Semantic-Prior-Guided Feature-Adaptive Learning for Aerial Road-Crack Segmentation

Official PyTorch implementation and reproducibility package for **SGFADet**. The repository follows the revised paper: an RGB detail pathway is calibrated by SFC, a frozen MobileSAM TinyViT encoder supplies a reusable semantic prior, SAF transfers that prior at P3/P4/P5, and ATAH aligns semantic-region and boundary-geometry prediction.

> RGB detail hierarchy -> SFC calibration -> frozen MobileSAM prior -> three-scale SAF -> PAN/FPN aggregation -> ATAH -> dense crack mask

The code in this branch supersedes the earlier Ultralytics prototype. It does not use an edge/Sobel substitute for the paper model and does not retain CrackLS315 as experimental evidence.

## Paper-code contract

| Revised-paper item | Implementation |
|---|---|
| RGB backbone, C3K2, SPPF, C2PSA-like attention | `sgfadet.py`: `SGFADet`, `C3K2`, `SPPF`, `PSABlock` |
| SFC, Eqs. (6)-(10) | `sgfadet.py`: `SFC` |
| SAF at P3/P4/P5, Eqs. (1)-(5) | `sgfadet.py`: `SAF`, `SGFADet.saf3/saf4/saf5` |
| Frozen MobileSAM TinyViT semantic prior | `precompute_sam.py`, `sam_adapter.py` |
| ATAH, Eqs. (11)-(14) | `sgfadet.py`: `ATAH`, `TaskAwareModulator`, `GeometryAlignment` |
| Main, semantic, boundary, and four side-output losses, Eq. (15) | `losses.py`: `sgfadet_loss` |
| Fixed grouped train/validation/test protocol | `splits/*.json`, `segmentation_common.py`, `audit_splits.py` |
| Five-seed training and validation-only selection | `run_matrix.py`, `train.py`, `aggregate_runs.py` |
| Boundary F1, HD95/nHD95, clDice, Thin-Re | `metrics.py`, `evaluate_checkpoint.py` |
| Efficiency and robustness analysis | `benchmark.py`, `robustness.py` |

See [`docs/PAPER_CODE_ALIGNMENT.md`](docs/PAPER_CODE_ALIGNMENT.md) for the operator-level mapping and [`configs/paper_protocol.json`](configs/paper_protocol.json) for the machine-readable protocol.

## Repository layout

```text
.
|-- sgfadet.py                    # network: RGB path, SFC, SAF, neck, ATAH
|-- sam_adapter.py                # cached/online frozen MobileSAM and controls
|-- losses.py                     # Eq. (15)
|-- train.py                      # single-dataset/single-seed training
|-- run_matrix.py                 # two datasets x five seeds
|-- evaluate_checkpoint.py        # region and structural metrics
|-- predict.py                    # masks and overlays for new images
|-- benchmark.py                  # parameters, operations, latency and memory
|-- robustness.py                 # post-freeze perturbation analysis
|-- segmentation_common.py        # dataset inventory and split loading
|-- splits/                       # frozen grouped manifests
|-- configs/paper_protocol.json   # paper-aligned constants
|-- docs/                         # data and reproducibility notes
`-- tests/                        # repository-contract checks
```

The scripts whose names include `controls`, `cross_dataset`, or `structural` reproduce the component, semantic-prior, architecture, topology, and transfer analyses added during revision. They are not required for a standard single-model inference workflow.

## Installation

The paper experiments used an NVIDIA RTX 3090 (24 GB), 640 x 640 inputs, and CUDA-enabled PyTorch. Python 3.10 or 3.11 is recommended.

1. Create and activate an isolated environment.
2. Install a CUDA build of PyTorch and TorchVision using the command generated at <https://pytorch.org/get-started/locally/>.
3. Install the remaining dependencies and the official MobileSAM runtime:

```bash
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/ChaoningZhang/MobileSAM.git"
```

Download the official `mobile_sam.pt` checkpoint separately. The checkpoint used for the paper has SHA-256:

```text
6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
```

The repository does not redistribute third-party datasets or checkpoints.

## Datasets and fixed splits

The public sources are:

- [UAV-CrackX-Datasets](https://github.com/SHAN-JH/UAV-CrackX-Datasets), using the 400 annotated x4 images referred to as **UAV-Crack500** in the paper.
- [DeepCrack](https://github.com/qinnzou/DeepCrack), using **CrackTree260**.

Expected local layout:

```text
<data-root>/
|-- UAV-Crack500/
|   |-- leftImg8bit/train/UAV-CrackX4/*.jpg
|   `-- gtFine/train/UAV-CrackX4/*.png
`-- CrackTree260/
    |-- CrackTree260/*.jpg
    `-- gt/*.bmp
```

The committed manifests define:

| Dataset | Train | Validation | Test | Isolation rule |
|---|---:|---:|---:|---|
| UAV-Crack500 | 282 | 70 | 48 | neighboring crops remain in ten-frame temporal blocks; the held-out test set is one continuous block |
| CrackTree260 | 182 | 39 | 39 | conservative acquisition/scene sequence groups visually audited against road/background characteristics; no group crosses a split |

UAV-Crack500's public x4 images originate from one flight sequence, so the protocol is described as **temporal-block isolation**, not flight-disjoint testing. CrackTree260 has no official scene identifiers; the manifest therefore records the conservative audited grouping used in the revision.

CrackLS315 is excluded because the available local copy did not provide traceable acquisition/scene identifiers sufficient for the same leakage-audited grouped protocol, and no code-aligned grouped five-run result was available. Including it would apply a weaker validation standard than the two retained datasets.

Audit the frozen manifests before training:

```bash
python audit_splits.py --dataset uav_crack500 --data-root <data-root> --output artifacts/audits/uav_crack500.json
python audit_splits.py --dataset cracktree260 --data-root <data-root> --output artifacts/audits/cracktree260.json
```

Additional details are in [`docs/DATASETS.md`](docs/DATASETS.md).

## Reproducing the paper protocol

### 1. Precompute the frozen MobileSAM prior

The main runs cache the frozen TinyViT neck feature once for every image. Spatial augmentation is applied synchronously to RGB, mask, and cached prior during training.

```bash
python precompute_sam.py \
  --dataset all \
  --data-root <data-root> \
  --cache-root <cache-root> \
  --sam-checkpoint <path/to/mobile_sam.pt>
```

The generated manifest records the runtime, input size, feature shape, checkpoint filename, and checkpoint SHA-256.

### 2. Run the two-dataset, five-seed matrix

```bash
python run_matrix.py \
  --data-root <data-root> \
  --cache-root <cache-root> \
  --run-root runs/paper_main
```

Frozen defaults:

- seeds: 42, 123, 3407, 2025, 2026;
- the v2 matrix seeds every run but retains its original non-bitwise-deterministic CUDA/cuDNN execution; uncertainty is reported across five runs;
- input: 640 x 640; batch size: 6; evaluation batch size: 4;
- 200 epochs; AdamW; initial learning rate 5e-4; weight decay 3e-5;
- five warm-up epochs, cosine decay, gradient-norm clipping at 5;
- mean of class-balanced BCE and Dice for every segmentation term;
- loss weights: semantic 0.20, boundary 0.10, four-side-output mean 0.15;
- maximum EMA decay 0.999 with the v2 warm-up ramp; fixed threshold 0.5;
- checkpoint selected exclusively by validation mIoU; test evaluated once after selection.

For one dataset and seed, call `train.py` directly. Run `python train.py --help` for every option.

### 3. Aggregate the five runs

```bash
python aggregate_runs.py \
  --runs \
    runs/paper_main/uav_crack500/seed_42 \
    runs/paper_main/uav_crack500/seed_123 \
    runs/paper_main/uav_crack500/seed_3407 \
    runs/paper_main/uav_crack500/seed_2025 \
    runs/paper_main/uav_crack500/seed_2026 \
  --output-dir artifacts/uav_crack500_summary
```

Repeat with the five CrackTree260 directories. `plot_multiseed_curves.py` generates the mean validation-mIoU curve with sample-standard-deviation shading.

### 4. Evaluate structural metrics

```bash
python evaluate_checkpoint.py \
  --dataset uav_crack500 \
  --checkpoint runs/paper_main/uav_crack500/seed_42/best.pt \
  --data-root <data-root> \
  --cache-root <cache-root> \
  --subset test \
  --output-dir artifacts/uav_seed42_test \
  --save-predictions
```

The evaluator reports foreground Precision/Recall/F1, foreground/background mIoU, Boundary F1 with two-pixel tolerance, HD95, diagonal-normalized HD95, clDice, and recall of target-skeleton pixels whose local width is at most three pixels.

For prediction on new images, the script computes the frozen MobileSAM prior online and applies the validation-selected decoder:

```bash
python predict.py \
  --checkpoint runs/paper_main/uav_crack500/seed_42/best.pt \
  --sam-checkpoint <path/to/mobile_sam.pt> \
  --source <image-or-directory> \
  --output-dir outputs/predictions
```

### 5. Run the controlled analyses

```bash
python run_component_controls.py --data-root <data-root> --cache-root <cache-root> --run-root runs/component_controls --dataset uav_crack500
python run_structural_5seed.py --run-root runs/paper_main --output-root artifacts/structural --data-root <data-root> --cache-root <cache-root>
python run_cross_dataset.py --run-root runs/paper_main --output-root artifacts/cross_dataset --data-root <data-root> --cache-root <cache-root>
python run_prior_and_architecture_controls.py --data-root <data-root> --cache-root <cache-root> --sam-checkpoint <path/to/mobile_sam.pt> --run-root runs/extended_controls
```

These scripts preserve the same split, optimizer, epoch budget, validation-only selection, threshold, and one-shot test policy unless a script explicitly labels an analysis as a post-freeze evaluation.

## Reported revised-paper results

Mean +/- sample standard deviation over five independent runs:

| Dataset | Precision | Recall | F1 | mIoU |
|---|---:|---:|---:|---:|
| UAV-Crack500 | 0.7482 +/- 0.0113 | 0.8190 +/- 0.0092 | 0.7673 +/- 0.0046 | 0.7838 +/- 0.0033 |
| CrackTree260 | 0.5505 +/- 0.0041 | 0.7614 +/- 0.0069 | 0.6322 +/- 0.0018 | 0.7292 +/- 0.0010 |

Structural results from the same five test checkpoints:

| Dataset | Boundary F1 | HD95 | nHD95 | clDice | Thin-Re |
|---|---:|---:|---:|---:|---:|
| UAV-Crack500 | 0.8052 +/- 0.0068 | 52.40 +/- 2.37 | 0.0579 +/- 0.0026 | 0.8240 +/- 0.0065 | 0.8110 +/- 0.0174 |
| CrackTree260 | 0.9622 +/- 0.0017 | 3.16 +/- 0.77 | 0.0035 +/- 0.0009 | 0.6652 +/- 0.0017 | 0.9504 +/- 0.0041 |

On the paper's RTX 3090 benchmark, the complete model contains 25.59 M parameters and requires 136.52 GFLOPs per 640 x 640 forward pass under the stated two-FLOPs-per-MAC convention. End-to-end MobileSAM-plus-SGFADet inference reached 28.11 FPS with 439.55 MiB peak allocated memory; cached-prior decoder inference reached 76.43 FPS with 331.97 MiB. Re-run `benchmark.py` on the target system rather than treating these hardware-dependent values as universal.

## Quick checks

The static repository contract does not require PyTorch:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q .
```

After installing PyTorch/TorchVision, run a small random forward pass:

```bash
python quick_check.py --device cpu --image-size 128
```

CUDA is required for the declared 640 x 640 training and benchmark protocol.

## Checkpoints, logs, and provenance

Model weights, datasets, and local run directories are intentionally excluded from Git. A publication release should attach the five selected checkpoints per dataset, `epoch_metrics.csv`, `config.json`, `final_metrics.json`, split-audit reports, and aggregate summaries as release assets or in an archival record. Every generated configuration records the split SHA-256, seed, runtime, parameter count, and validation/test policy.

### Checkpoint-compatibility note

The numerical model and training path are intentionally kept compatible with the uploaded v2 source because the revised results were not retrained. The released ATAH stem uses the v2 `Conv` block (Conv-BN-SiLU), EMA uses a warm-up ramp whose configured maximum is 0.999, and `run_matrix.py` retains the seeded but non-bitwise-deterministic CUDA setting used for the reported runs. Changing these settings requires retraining and re-benchmarking.

## Citation

The paper is under revision. Use [`CITATION.cff`](CITATION.cff) for the current title and author list; update the venue, year, pages, and DOI after acceptance.

## License

This repository is released under the [GNU Affero General Public License v3.0](LICENSE). MobileSAM and the two datasets retain their own licenses and terms; users must obtain them from their original sources.
