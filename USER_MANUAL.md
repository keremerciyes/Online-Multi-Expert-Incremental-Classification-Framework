# User Manual

## Purpose

This project is a standalone PyTorch Lightning + Hydra workspace for online, class-incremental motor anomaly classification.

The workspace is now focused on multi-head designs. It supports:

- `online_oneclass_multihead`: shared frozen backbone + one one-class cosine head per class, feature-map 1x1 conv projector (default)
- `online_oneclass_multihead_latent`: same one-class design with a pooled linear projector
- `online_oneclass_multihead_etf`: shared single-layer projector aligning each class to a fixed simplex-ETF target

Every design shares one frozen **feature-extractor backbone** chosen from four deployable bundles (see *Feature Extractor Backbones*). Each one-class head is exactly two learnable layers: a projector (1x1 conv or linear) followed by a cosine prototype.

You can vary:

- the dataset root and per-class folders
- selected input channels
- the selected speed labels (default `1000`-`1600`)
- the feature-extractor backbone (four bundles)
- the head family and its calibration
- loss switches, weights, and margins
- optimizer, logging, and trainer settings

The offline single-split, offline CNN, offline cross-validation, online sequential, and the multi-head streaming designs.

## Project Layout

- `configs/`: Hydra configuration groups
- `src/motor_anomaly_classification/data/`: dataset loading, indexing, splits, standardization, datamodules
- `src/motor_anomaly_classification/models/`: feature-extractor backbones, heads, `OneClassMultiHeadNet`, `EtfMultiHeadNet`, model/optimizer factory
- `src/motor_anomaly_classification/systems/`: Lightning modules and callbacks
- `src/motor_anomaly_classification/train.py`: main train entrypoint
- `src/motor_anomaly_classification/eval.py`: checkpoint evaluation entrypoint
- `tests/`: smoke tests on a synthetic mini-dataset
- `outputs/`: Hydra runs, logger outputs, checkpoints, plots, and reports

## Pipeline Overview

1. Hydra composes a config from `configs/config.yaml` plus any command-line overrides.
2. The multi-head datamodule loads each class from its own folder, splits per class balanced by speed, and concatenates the per-class validation partitions into one shared evaluation pool.
3. The model config chooses a composition schema, and Hydra injects one backbone config and one head config into it.
4. The Lightning system wraps the model, losses, calibration, replay, optimizer construction, and streaming behavior.
5. Callbacks handle dataset summaries, confusion matrix, classification report, per-head score diagnostics, and batch-loss plots.
6. The trainer runs `fit` (sequential class phases) then `test`.
7. Logs, checkpoints, and artifacts are written under `outputs/`.

## How Data Flows

### Dataset structure

```text
dataset_root/
  class_a/
    speed_1/*.npy
    speed_2/*.npy
  class_b/
    ...
```

Default dataset root: `C:\github_project\motor_anomaly_classification\Balanced_by_speed_and_classes`

Each `.npy` file contains a 3D array of windows. Channel selection happens before any split or standardization; the model input channel count is derived from the selection (for example `[0, 1]`). The default selection is the two accelerometer channels `[0, 1]`, matching the pretrained feature-extractor bundles.

### Speed selection

By default only the speed folders in the `1000`-`1600` range are loaded (`datamodule.config.selected_speeds: ["1000", "1200", "1400", "1600"]`). Set `datamodule.config.selected_speeds=null` to use every speed folder, or pass a custom quoted list to change the range. The shaft speed for each window is parsed from its folder name and is required by the speed-aware handcrafted descriptors.

### Online multi-head flow

1. load each selected class from its own configured folder path
2. keep only the selected speed folders (default `1000`-`1600`)
3. split each class independently into train and validation, balanced by speed
4. concatenate per-class validation partitions into one shared evaluation pool
5. fit input standardization on the concatenated multi-class training portion
6. for handcrafted / hybrid bundles, precompute the standardized feature vector per window (see *Feature Extractor Backbones*)
7. build one shared feature-extractor backbone plus one head per class
8. compute a one-time global feature standardization over the frozen backbone features (see below)
9. train heads sequentially by class phase, with explicit head indices in training batches
10. evaluate by scoring all heads and choosing the highest voting score

## Model Composition

The model is built from `backbone` + `head`.

- Backbone output is aggregated with global average pooling and global max pooling, then concatenated. The time axis is never flattened.
- Conv-projector heads consume the frozen backbone feature map `[B, C, T]` through a 1x1 conv, then pool. Projector conv + cosine prototype = **two learnable layers**.
- Linear-projector heads consume the pooled `[B, 2C]` vector through a linear projector, then a cosine prototype = **two learnable layers**.

Head families:

- `configs/head/one_class_cosine_head_conv_projector.yaml`
- `configs/head/one_class_cosine_head_linear_projector.yaml`
- `configs/head/etf_head.yaml`

Model schemas:

- `configs/model/oneclass_multihead.yaml`
- `configs/model/etf_multihead.yaml`

## Feature Extractor Backbones

All designs share one frozen feature-extractor backbone loaded from a deployable
`.pt` bundle under
`C:\github_project\motor_anomaly_classification\Avalanche_Models\feature_extractor_comparison\deployable\`.
Choose one with `backbone=<name>`. There are two families:

**Live AE-encoder backbones** (`BundleEncoderBackbone`) rebuild a frozen conv1d
autoencoder encoder from the bundle and emit an `[B, 8, T']` feature map. They run
inside the model, need no precompute, and pair with either projector head or ETF.

- `ae_raw_basicmotions` (default) — conv1d AE encoder transferred from BasicMotions, 8-channel feature map.
- `ae_raw_clap` — conv1d AE encoder distilled from CLAP, 8-channel feature map.

**Precomputed-vector backbones** (`PrecomputedFeatureBackbone`) are identity
pass-throughs over speed-aware feature vectors that the datamodule computes on the
standardized windows. Because the handcrafted descriptors are order-domain, they
require the shaft speed per window, so the datamodule does the precompute. To use
one you must point `oneclass.feature_bundle_path` at the **same** bundle so the
datamodule standardizes and precomputes the vectors:

- `handcrafted16` — 16 physical vibration descriptors (RMS, crest, kurtosis, 1x/2x/harmonic/sub-order energy ratios, line/mid/high/very-high band ratios, spectral centroid/entropy, XY correlation, axis ratio, high-frequency kurtosis), robust-standardized.
- `wdcnn_hybrid` — 48-d hybrid: the 16 handcrafted descriptors concatenated with a 32-d pooled WDCNN latent, robust-standardized.

The old raw `conv1d_small` / `deep_conv1d` backbones and their offline pretrained
`.pt` encoders have been removed. The `oneclass.pretrained_backbone_checkpoint_path`
key is only needed if you want to override a backbone's weights from a
Lightning checkpoint; the four bundles already carry their own frozen weights.

### Benchmark Handcrafted22 Features

The four-class reference-feature experiment uses the benchmark's 22 XYZ
descriptors, Hann-windowed spectral power, exact RPM-dependent bands, and the
saved robust median/IQR + clipping + block-L2 scaler:

```powershell
& 'C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe' -m motor_anomaly_classification.train experiment=online_oneclass_multihead_latent_handcrafted22_4class
```

Its `handcrafted22` backbone points to the bundle under
`Avalanche_Models/feature_extractor_comparison_final_handcrafted22/deployable/`.
The experiment sets `datamodule.config.feature_bundle_path=${backbone.bundle_path}`,
channels `[0,1,2]`, and `datamodule.config.normalization=bundle`. Bundle mode uses
the saved input mean/std without refitting; channel order, normalization values,
and descriptor names/order are validated. Other experiments retain training-only
input normalization by default. Precomputed vectors enter latent heads once
(22 inputs), while temporal maps retain GAP+GMP pooling.

This experiment retains the requested 16-dimensional linear projector, losses,
SGD settings, z-score voting, and 15-epoch schedule. Feature standardization inside
the classifier is disabled because the bundle already scales the descriptors.
Outputs include the confusion matrix, per-speed accuracy, raw and calibrated
per-head scores, and the classification report. Evaluation reuses the validation
pool; saved bundle calibration may overlap the evaluation recordings, so do not
report this as independent test-set performance.

## One-Class Losses

Only the active head is trained during its class phase; inactive heads are frozen. The active-head objective is:

```text
L = w_align   * alignment_loss
  + w_bound   * boundary_loss
  + w_old     * old_contrast_loss
  + w_replay  * replay_negative_suppression_loss
```

With unit embedding `u = normalize(embed(x))`, unit prototype `p`, `d = ||u - p||^2`, radius `r`:

- `alignment_loss = mean(1 - cos(u, p)) = mean(d/2)` — pulls positives toward the prototype.
- `boundary_loss = mean(relu(d/r^2 - 1))` — keeps positives inside the calibrated radius (SVDD-style hinge).
- `old_contrast_loss = mean(relu(max_{j<h} c_j(x) - c_h(x) + margin))`, old scores detached — the active head must beat older heads on the new class by a margin.
- `replay_negative_suppression_loss` — the active head must reject stored embeddings of other classes.

Replay negatives are stored per class and, by default, mined for hardness: a larger candidate pool is scored by the active head and only the highest-scoring (most intruding) negatives are penalized (`use_hard_negative_mining`, `hard_negative_pool_multiplier`).

## Calibration and Standardization

Training geometry stays in cosine/boundary space for stability, while **voting** (evaluation and prediction) uses a per-head statistical calibration so heads are comparable.

- `voting_calibration_mode: mahalanobis` (default): each head accumulates an EMA mean and covariance of its positive embeddings; the voting score is `1 - D_M^2 / chi2_thresh`, where `D_M^2` is the Mahalanobis distance and `chi2_thresh` is a chi-square acceptance radius. Threshold 0 = acceptance boundary; argmax picks the nearest prototype in the whitened space.
- `voting_calibration_mode: zscore`: voting score is the per-head standardized raw cosine score.
- `voting_calibration_mode: boundary`: falls back to the geometric `(r^2 - d)/r^2` margin.

Global feature standardization (`init_feature_standardization: true`): once, at fit start, the frozen backbone is run over the training data to compute a fixed per-channel mean/std buffer that standardizes the backbone feature map before every head. Batch/group/layer norm layers are intentionally not used.

Mahalanobis tuning: `head.cov_shrinkage` (default `0.1`) blends the estimated covariance toward a scaled identity — raise it for smoother/more regularized boundaries when a class has few or noisy samples, lower it to trust the raw covariance shape more. `head.mahalanobis_confidence` (default `0.95`) sets the chi-square acceptance radius — raise it (e.g. `0.99`) to accept more (wider region, fewer rejections), lower it to be stricter. Neither changes the argmax ranking; they only move the accept/reject boundary.

## ETF Experiment

`online_oneclass_multihead_etf` uses a single shared projector — a cosine (linear) layer by default that produces the class prototypes, or a 1x1 conv (`head.projector_arch=conv`) — and fixed simplex-ETF target vectors (pairwise cosine `-1/(K-1)`). Only the shared projector trains; targets are frozen buffers, so no prototype receives negative optimization while other classes stream. The only loss is alignment of the active class to its fixed target; separation is guaranteed by construction. All classes live in one embedding space, so voting is plain cosine to each fixed target. This is an intentionally minimal, shared-space alternative to the isolated per-head OCC design and is expected to show shared-projector drift/forgetting.

## Config System

Hydra groups: `experiment`, `model`, `backbone`, `head`, `optimizer`, `datamodule`, `trainer`, `callbacks`, `logging`, `paths`.

Main config file: `configs/config.yaml` (defaults now point at the multi-head profile).

## Installation

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe -m pip install -e .
```

If not installed editable, set `PYTHONPATH` per command:

```powershell
$env:PYTHONPATH = "src"
```

## Main Commands

### Default multi-head one-class (conv projector, Mahalanobis voting)

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
$env:PYTHONPATH = "src"
C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe -m motor_anomaly_classification.train experiment=online_oneclass_multihead backbone=ae_raw_basicmotions head=one_class_cosine_head_conv_projector trainer.max_epochs=30
```

Swap `backbone=ae_raw_basicmotions` for `ae_raw_clap` (live AE encoder), or for a
precomputed backbone by adding its bundle path, e.g.:

```powershell
C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe -m motor_anomaly_classification.train experiment=online_oneclass_multihead backbone=wdcnn_hybrid "oneclass.feature_bundle_path='C:/github_project/motor_anomaly_classification/Avalanche_Models/feature_extractor_comparison/deployable/wdcnn_hybrid.pt'" trainer.max_epochs=30
```

### Fully parametrized experiment (loss switches, weights, margins, calibration, replay)


ae_raw_basicmotions, ae_raw_clap;	init_feature_standardization : true (default, don't override)
handcrafted16, wdcnn_hybrid;	init_feature_standardization : false (must set explicitly)

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
$env:PYTHONPATH = "src"

$Epochs = 1

python.exe -m motor_anomaly_classification.train `
  experiment=online_oneclass_multihead `
  oneclass.feature_bundle_path=C:\github_project\motor_anomaly_classification\Avalanche_Models\feature_extractor_comparison\deployable\handcrafted16.pt `
  backbone=handcrafted16 `
  head=one_class_cosine_head_linear_projector `
  model.num_classes=3 `
  trainer.max_epochs=$Epochs `
  system.training_config.use_alignment_loss=true `
  system.training_config.use_boundary_loss=true `
  system.training_config.use_old_contrast_loss=true `
  system.training_config.use_replay_negative_suppression_loss=true `
  system.training_config.alignment_weight=1.0 `
  system.training_config.boundary_weight=1.0 `
  system.training_config.old_contrast_weight=1.0 `
  system.training_config.replay_negative_suppression_weight=1.0 `
  system.training_config.old_contrast_margin=0.15 `
  system.training_config.replay_negative_margin=-0.15 `
  system.training_config.replay_buffer_size_per_class=32 `
  system.training_config.replay_batch_size_per_class=8 `
  system.training_config.min_replay_samples_per_class=16 `
  system.training_config.use_hard_negative_mining=true `
  system.training_config.hard_negative_pool_multiplier=4 `
  system.training_config.init_feature_standardization=false `
  system.training_config.use_uncertainty_weighting=true `
  system.training_config.contrast_score_mode=calibrated `
  datamodule.config.train_batch_size=32 `
  "datamodule.config.selected_classes=[belt,vibration,magnetic]" `
  head.boundary_radius=0.4 `
  head.voting_calibration_mode=mahalanobis `
  head.cov_shrinkage=0.3 `
  head.mahalanobis_confidence=0.95
```

Tunable knobs added here:

- `datamodule.config.train_batch_size` — model batch size (streaming batch per class phase). Larger = smoother per-step gradients and covariance/EMA estimates; smaller = more frequent updates.
- `datamodule.config.selected_classes` — class order for the online stream (default `magnetic, vibration, belt`). Applies to every multi-head experiment, including ETF. Quote the list so PowerShell does not split it.
- `datamodule.config.selected_speeds` — which speed folders to load (default `1000, 1200, 1400, 1600`). Set to `null` for all speeds, or pass a quoted list. Speeds outside the list are ignored end to end.
- `oneclass.feature_bundle_path` — required for the `handcrafted16` / `wdcnn_hybrid` backbones; point it at the same bundle so the datamodule precomputes the standardized feature vectors. Leave `null` for the live AE backbones.
- `system.training_config.use_uncertainty_weighting` — when `true`, the four losses are balanced automatically with learned per-loss log-variances (Kendall-style homoscedastic weighting) and the manual `*_weight` values are ignored. When `false`, the manual weights are used.

### Second Exp

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
$env:PYTHONPATH = "src"
$Epochs = 3
python.exe -m motor_anomaly_classification.train `
experiment=online_oneclass_multihead `
backbone=ae_raw_basicmotions `
head=one_class_cosine_head_conv_projector `
model.num_classes=3 `
trainer.max_epochs=$Epochs `
system.training_config.use_alignment_loss=true `
system.training_config.use_boundary_loss=true `
system.training_config.use_old_contrast_loss=true `
system.training_config.use_replay_negative_suppression_loss=true `
system.training_config.alignment_weight=1.0 `
system.training_config.boundary_weight=0.3 `
system.training_config.old_contrast_weight=0.7 `
system.training_config.replay_negative_suppression_weight=1.0 `
system.training_config.old_contrast_margin=0.2 `
system.training_config.replay_negative_margin=-0.1 `
system.training_config.replay_buffer_size_per_class=64 `
system.training_config.replay_batch_size_per_class=8 `
system.training_config.min_replay_samples_per_class=32 `
system.training_config.use_hard_negative_mining=false `
system.training_config.init_feature_standardization=true `
system.training_config.contrast_score_mode=calibrated `
head.boundary_radius=0.5 `
head.voting_calibration_mode=zscore
```

Switch calibration to z-score with `head.voting_calibration_mode=zscore`, or to the geometric margin with `head.voting_calibration_mode=boundary`.

### Linear-projector one-class variant

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
$env:PYTHONPATH = "src"
C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe -m motor_anomaly_classification.train experiment=online_oneclass_multihead_latent backbone=ae_raw_basicmotions trainer.max_epochs=30
```

### ETF shared-projector experiment

The shared projector defaults to a single cosine (linear) layer producing the class prototypes; set `head.projector_arch=conv` for the 1x1-conv variant.

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
$env:PYTHONPATH = "src"
C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe -m motor_anomaly_classification.train `
  experiment=online_oneclass_multihead_etf `
  backbone=ae_raw_basicmotions `
  trainer.max_epochs=30 `
  datamodule.config.train_batch_size=32 `
  "datamodule.config.selected_classes=[magnetic,vibration,belt]" `
  head.projector_arch=cosine `
  head.embedding_dim=16
```

### Evaluate a saved checkpoint

Use the same experiment profile at eval time that you used at train time.

```powershell
Set-Location "C:\github_project\motor_anomaly_classification\motor_anomaly_classification"
$env:PYTHONPATH = "src"
C:\github_project\motor_anomaly_classification\.venv\Scripts\python.exe -m motor_anomaly_classification.eval experiment=online_oneclass_multihead "eval.checkpoint_path='C:/path/to/epoch=14-step=3270.ckpt'"
```

## Hydra Troubleshooting

Path overrides whose filenames contain `=` must be written as one fully quoted override, otherwise Hydra tries to parse the `=` as override syntax:

```powershell
# correct
"eval.checkpoint_path='C:/.../epoch=14-step=3600.ckpt'"
"oneclass.feature_bundle_path='C:/.../wdcnn_hybrid.pt'"
```

Normal scalar overrides such as `trainer.max_epochs=30` need no special quoting. For a full stack trace:

```powershell
$env:HYDRA_FULL_ERROR = "1"
```

Use `backbone=<name>` to pick a feature extractor. The `handcrafted16` and `wdcnn_hybrid` backbones additionally require `oneclass.feature_bundle_path` pointed at the matching bundle so the datamodule can precompute their speed-aware feature vectors.
