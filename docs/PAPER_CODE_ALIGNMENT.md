# Paper-code alignment

This document maps the revised manuscript to the executable implementation. The manuscript is the scientific specification; control variants are isolated behind explicit command-line options and are not silently substituted for the complete model.

## Overall architecture

`SGFADet.forward` implements the following sequence:

1. The RGB hierarchy produces 64 x 160 x 160, 128 x 80 x 80, 256 x 40 x 40, and 384 x 20 x 20 principal features for a 640 x 640 input.
2. SFC calibrates shallow/intermediate RGB features without spatial downsampling.
3. A frozen 256 x 64 x 64 MobileSAM TinyViT neck feature is resized and projected separately at P3, P4, and P5.
4. SAF performs gated cross-guidance and retains an RGB residual path.
5. The PAN/FPN-style neck aggregates top-down and bottom-up features.
6. ATAH forms semantic and geometry-aligned features, then predicts a dense crack logit.
7. Four side logits provide deep supervision and participate in the v2 learned main-logit fusion.

## Equations

| Equations | Code | Contract |
|---|---|---|
| (1) | `SAF.rgb_project`, `SAF.sam_project` | channel alignment after spatial resizing |
| (2)-(3) | `SAF.attention` | global descriptor, excitation, equal split, branch gates |
| (4)-(5) | `SAF.forward` | cross-guidance, concatenation/projection, RGB residual |
| (6) | `SFC.branch1`, `SFC.branch2` | complementary feature bases |
| (7)-(8) | `SFC.forward`, `SFC.statistical_gate`, `SFC.scale` | average/maximum evidence and branch-specific gates |
| (9)-(10) | `SFC.forward`, `SFC.out` | gated branches, residual aggregation, 3 x 3 projection |
| (11) | `ATAH.shared` | checkpoint-compatible v2 Conv-BN-SiLU feature stem |
| (12) | `TaskAwareModulator`, `ATAH.semantic_spatial` | channel and spatial semantic alignment |
| (13) | `GeometryAlignment` | learned DCNv2 offsets, masks, and deformable sampling |
| (14) | `ATAH.fuse` | semantic/geometry concatenation and dense prediction |
| (15) | `losses.sgfadet_loss` | main + 0.20 semantic + 0.10 boundary + 0.15 mean of four side terms |

## Complete model versus controls

The default constructor is `SGFADet(variant="full", fusion_stages="345", backbone="custom")`. The following alternatives exist only for controlled comparisons:

- `rgb_plain`, `sfc_only`, `saf_only`, `no_sfc`, `no_saf`;
- static SAM addition and concatenation;
- a plain head and ATAH without deformable alignment;
- P3/P4/P5-only fusion and ResNet18 RGB-backbone controls;
- zero, online-frozen, and last-stage-fine-tuned MobileSAM controls.

`train.py` writes every selected variant into `config.json`, preventing a control run from being reported as the complete model.

## Wording that must remain checkpoint-compatible

The uploaded v2 implementation is the numerical authority for the reported, already-completed runs. It uses BatchNorm in the ATAH shared `Conv` blocks, an EMA warm-up ramp capped at 0.999, and seeded but non-bitwise-deterministic CUDA/cuDNN execution in `run_matrix.py`. These details must not be changed to GroupNorm, constant-from-step-one EMA, or deterministic kernels without retraining and re-benchmarking. Clarify any conflicting manuscript shorthand before final submission.

## Evaluation contract

- Foreground Precision, Recall, and F1 are accumulated from global test-set TP/FP/FN counts.
- mIoU is the mean of foreground and background IoU.
- Boundary F1 uses a two-pixel matching tolerance.
- HD95 is measured in the 640 x 640 evaluation space; nHD95 divides by the image diagonal.
- clDice measures centerline agreement.
- Thin-Re is recall on target-skeleton pixels whose estimated local width is at most three pixels.
- Threshold selection and checkpoint selection use validation data only. The frozen test split is evaluated once per selected seed checkpoint.

## Efficiency contract

`benchmark.py` reports decoder-only cached-prior and end-to-end MobileSAM-plus-decoder paths separately. Operation counts use one multiply-add = two FLOPs and exclude normalization, activation, softmax, interpolation, and image preprocessing. Hardware, precision, warm-up, iterations, and peak allocated memory are written to the JSON report.
