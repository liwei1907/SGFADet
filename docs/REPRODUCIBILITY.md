# Reproducibility checklist

Before creating a public release, verify that it contains:

- the exact Git commit hash;
- `configs/paper_protocol.json`;
- both frozen split manifests and their audit reports;
- the MobileSAM source revision and checkpoint SHA-256;
- one `config.json`, `epoch_metrics.csv`, `best.pt`, and `final_metrics.json` per dataset/seed;
- five-run aggregate JSON/CSV and convergence plots;
- structural, component, cross-dataset, robustness, and benchmark reports used in the paper;
- the GPU model, CUDA, PyTorch, TorchVision, and precision mode;
- a statement that validation alone selected checkpoints and that each fixed test subset was evaluated once after selection.

Do not upload local absolute paths, credentials, raw datasets, or third-party checkpoints unless redistribution is permitted. Release assets can contain model weights and logs after their provenance and licensing have been checked.
