# Motor Anomaly Classification

Standalone PyTorch Lightning + Hydra project for offline and online motor anomaly classification experiments.

See the full usage guide in USER_MANUAL.md.

## Current scope

- Offline single-split training
- Offline cross-validation orchestration
- Online sequential fine-tuning
- Staged offline-then-online workflow
- Grouped backbone/head architecture with a shared Conv1D + cosine-classifier baseline
