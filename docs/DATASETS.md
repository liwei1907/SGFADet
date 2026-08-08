# Dataset preparation and split policy

## Sources

- UAV-CrackX-Datasets: <https://github.com/SHAN-JH/UAV-CrackX-Datasets>
- CrackTree260 in DeepCrack: <https://github.com/qinnzou/DeepCrack>

Download each dataset from its original maintainer and comply with its license or usage terms. No dataset image or mask is redistributed here.

## Inventory rules

`segmentation_common.inventory_pairs` requires exactly 400 UAV-Crack500 x4 image/mask pairs and 260 CrackTree260 image/mask pairs. It fails on missing masks or unexpected counts instead of silently training on a partial download.

## Frozen manifests

The committed JSON manifests contain only relative paths, group identifiers, split metadata, and aggregate mask statistics. Images never cross splits, and every group is assigned wholly to train, validation, or test.

UAV-Crack500 is split by temporal blocks derived from neighboring DJI frame names. The public x4 subset is one flight sequence, so this is temporal-block isolation rather than flight-disjoint evaluation.

CrackTree260 does not publish reliable scene/acquisition IDs. The revision uses conservative acquisition/scene sequence groups and audits exact and perceptual cross-split similarity. This limitation should remain visible when interpreting the results.

## Leakage audit

`audit_splits.py` records:

- split and group counts;
- SHA-256 exact-duplicate matches across splits;
- 64-bit perceptual-hash near-pair candidates;
- the manifest SHA-256 and visual-review caveat.

A perceptual-hash candidate is not automatically a duplicate; visually inspect every candidate before changing a frozen manifest.

## Why CrackLS315 is not included

The available local copy lacked traceable scene/acquisition identifiers needed to construct and independently audit a grouped split comparable to the two retained datasets. It also lacked a code-aligned grouped five-run record under the revised protocol. Reporting it would therefore mix validation standards and weaken the reproducibility claim.
