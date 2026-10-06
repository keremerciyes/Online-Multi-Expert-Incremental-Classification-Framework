from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from motor_anomaly_classification.data.datamodule_multihead import MultiHeadSequentialDataConfig, MultiHeadSequentialDataModule
from motor_anomaly_classification.models.factory import ModelFactoryConfig, build_model
from motor_anomaly_classification.models.feature_extractors import HANDCRAFTED22_FEATURE_NAMES, compute_bundle_features, handcrafted22_features
from motor_anomaly_classification.systems.online_oneclass_module import OneClassTrainingConfig, OnlineOneClassMultiHeadModule
from motor_anomaly_classification.train import _build_model_config, _build_optimizer_config, _build_system, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
HANDCRAFTED_BUNDLE = (
    "C:/github_project/motor_anomaly_classification/Avalanche_Models/"
    "feature_extractor_comparison/deployable/handcrafted16.pt"
)


def test_replay_pool_and_total_batch_limits() -> None:
    config = OneClassTrainingConfig(replay_pool_samples_per_class=12, replay_batch_size=12, replay_negative_margin=0.1)
    sampled = []

    def sample(class_indices, device, per_class_count):
        sampled.append((class_indices, per_class_count))
        return torch.arange(24, dtype=torch.float32, device=device).unsqueeze(1)

    module = SimpleNamespace(
        training_config=config,
        replay_buffers={0: None, 1: None, 2: None},
        _sample_replay_buffer=sample,
        _head_calibrated_scores=lambda head, inputs: inputs.squeeze(1),
    )
    loss = OnlineOneClassMultiHeadModule._compute_replay_negative_loss(module, 2, object(), torch.zeros(1))
    assert sampled == [([0, 1], 12)]
    torch.testing.assert_close(loss, (torch.arange(12, 24, dtype=torch.float32) - 0.1).mean())


def test_latent_singleton_features_are_not_duplicated(tmp_path: Path) -> None:
    bundle_path = tmp_path / "features.pt"
    torch.save({"feature_dim": 22}, bundle_path)
    model = build_model(ModelFactoryConfig(
        mode="one_class_multi_head",
        num_classes=4,
        in_channels=3,
        backbone={
            "_target_": "motor_anomaly_classification.models.feature_extractors.PrecomputedFeatureBackbone",
            "bundle_path": str(bundle_path),
        },
        head={
            "_target_": "motor_anomaly_classification.models.head.OneClassCosineHeadLinearProjector",
            "projector_dim": 16,
        },
    ))
    features = torch.randn(4, 22, 1)
    scores, pooled = model(features)
    assert scores.shape == (4, 4)
    assert model.get_head(0).projector.in_features == 22
    torch.testing.assert_close(pooled, features.squeeze(2))
    temporal = torch.randn(4, 8, 5)
    torch.testing.assert_close(model.pooled_features_from_map(temporal), torch.cat([temporal.mean(2), temporal.amax(2)], dim=1))


def test_feature_standardization_warmup_limit() -> None:
    batches = [(torch.full((2, 1, 3), value),) for value in (1.0, 3.0, 100.0)]
    statistics = []
    module = SimpleNamespace(
        trainer=SimpleNamespace(datamodule=SimpleNamespace(train_dataloader=lambda: batches)),
        training_config=OneClassTrainingConfig(feature_std_warmup=2),
        device=torch.device("cpu"),
        model=SimpleNamespace(
            backbone=torch.nn.Identity(),
            set_feature_standardization=lambda mean, std: statistics.append((mean, std)),
        ),
    )
    OnlineOneClassMultiHeadModule._initialize_feature_standardization(module)
    mean, std = statistics[0]
    torch.testing.assert_close(mean, torch.tensor([2.0]))
    torch.testing.assert_close(std, torch.tensor([1.0 + 1e-6]))


def test_handcrafted22_bundle_normalization(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    bundle = {
        "design": "handcrafted22", "feature_dim": 22, "handcrafted_dim": 22,
        "has_handcrafted": True, "channel_indices": [0, 1, 2],
        "handcrafted_feature_names": list(HANDCRAFTED22_FEATURE_NAMES),
        "input_normalization": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 3.0, 4.0]},
        "standardizer": {"block_sizes": [22], "median": np.zeros(22), "iqr": np.ones(22)},
    }
    bundle_path = tmp_path / "handcrafted22.pt"
    torch.save(bundle, bundle_path)
    config = MultiHeadSequentialDataConfig(
        class_paths={"normal": str(synthetic_dataset_dir / "healthy")},
        channel_indices=(0, 1, 2), selected_speeds=None,
        feature_bundle_path=str(bundle_path), normalization="bundle",
    )
    datamodule = MultiHeadSequentialDataModule(config)
    datamodule.setup()
    np.testing.assert_array_equal(datamodule.mean.reshape(-1), bundle["input_normalization"]["mean"])
    np.testing.assert_array_equal(datamodule.std.reshape(-1), bundle["input_normalization"]["std"])
    features = next(iter(datamodule.train_dataloader()))[0]
    assert features.shape[1:] == (22, 1)
    assert torch.isfinite(features).all()
    config.channel_indices = (2, 1, 0)
    with pytest.raises(ValueError, match="channel order"):
        MultiHeadSequentialDataModule(config).setup()
    bundle["handcrafted_feature_names"] = list(reversed(HANDCRAFTED22_FEATURE_NAMES))
    with pytest.raises(ValueError, match="schema"):
        compute_bundle_features(bundle, np.zeros((2, 64, 3)), np.full(2, 1000.0))


def test_handcrafted22_experiment_configuration() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["experiment=online_oneclass_multihead_latent_handcrafted22_4class"])
    assert cfg.datamodule.config.feature_bundle_path == cfg.backbone.bundle_path
    assert str(cfg.backbone.bundle_path).endswith("/handcrafted22.pt")
    assert list(cfg.datamodule.config.channel_indices) == [0, 1, 2]
    assert cfg.datamodule.config.normalization == "bundle"
    assert cfg.model.in_channels == 3
    assert not cfg.system.training_config.init_feature_standardization


def test_handcrafted22_requires_xyz_and_valid_speed() -> None:
    with pytest.raises(ValueError, match="X, Y, Z"):
        handcrafted22_features(np.zeros((2, 64, 2)), np.full(2, 1000.0))
    with pytest.raises(ValueError, match="RPM"):
        handcrafted22_features(np.zeros((2, 64, 3)), np.array([1000.0, np.nan]))
    features = handcrafted22_features(np.zeros((2, 64, 3)), np.full(2, 1000.0))
    assert features.shape == (2, 22)
    assert np.isfinite(features).all()


def test_multihead_setup_preserves_train_eval_split(synthetic_dataset_dir: Path) -> None:
    datamodule = MultiHeadSequentialDataModule(MultiHeadSequentialDataConfig(
        class_paths={"normal": str(synthetic_dataset_dir / "healthy"), "belt": str(synthetic_dataset_dir / "fault")},
        selected_speeds=None,
    ))
    datamodule.setup()
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset
    test_speeds = datamodule.test_speeds
    for stage in ("fit", "test"):
        datamodule.setup(stage)
        assert datamodule.train_dataset is train_dataset
        assert datamodule.val_dataset is val_dataset
        assert datamodule.test_dataset is test_dataset
        assert datamodule.test_speeds is test_speeds


def compose_cfg(output_root: Path, overrides: list[str] | None = None):
    base_overrides = [
        f"paths.root_dir={output_root.as_posix()}",
        "logging=csv",
        "backbone.load_pretrained=false",
        "datamodule.config.selected_speeds=null",
        "trainer.max_epochs=1",
        "trainer.enable_checkpointing=false",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.limit_test_batches=1",
        "callbacks.dataset_inspection.print_summary=false",
        "callbacks.dataset_inspection.print_mapping=false",
    ]
    if overrides:
        base_overrides.extend(overrides)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=base_overrides)


def _multihead_class_paths(synthetic_dataset_dir: Path, channel_indices: str = "[0,1]") -> list[str]:
    healthy = (synthetic_dataset_dir / "healthy").as_posix()
    fault = (synthetic_dataset_dir / "fault").as_posix()
    return [
        f"datamodule.config.channel_indices={channel_indices}",
        f"datamodule.config.class_paths.normal={healthy}",
        f"datamodule.config.class_paths.belt={fault}",
        f"datamodule.config.class_paths.magnetic={healthy}",
        f"datamodule.config.class_paths.vibration={fault}",
    ]


def test_online_oneclass_multihead_smoke(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    cfg = compose_cfg(
        tmp_path / "online_oneclass_root",
        overrides=["experiment=online_oneclass_multihead", *_multihead_class_paths(synthetic_dataset_dir)],
    )
    run_experiment(cfg)
    assert any((tmp_path / "online_oneclass_root").rglob("classification_report.json"))


def test_online_oneclass_multihead_latent_smoke(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    cfg = compose_cfg(
        tmp_path / "online_oneclass_latent_root",
        overrides=["experiment=online_oneclass_multihead_latent", *_multihead_class_paths(synthetic_dataset_dir)],
    )
    run_experiment(cfg)
    assert any((tmp_path / "online_oneclass_latent_root").rglob("classification_report.json"))


def test_latent_calibration_runs_after_optimizer_step(tmp_path: Path, monkeypatch) -> None:
    torch.manual_seed(42)
    cfg = compose_cfg(tmp_path, overrides=["experiment=online_oneclass_multihead_latent"])
    datamodule = SimpleNamespace(class_names=["normal", "belt"], config=cfg.datamodule.config)
    system = _build_system(cfg, _build_model_config(cfg, datamodule), _build_optimizer_config(cfg))
    monkeypatch.setattr(system, "log", lambda *args, **kwargs: None)
    optimizer = system.configure_optimizers()
    inputs = torch.randn(4, 2, 128)
    labels = torch.zeros(4, dtype=torch.long)
    head = system.model.get_head(0)
    inactive_before = {name: value.detach().clone() for name, value in system.model.get_head(1).named_parameters()}
    pooled, _ = system.model.extract_representations(inputs)
    scores_before = head.raw_score_from_embeddings(head.embed(pooled)).detach()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = system.training_step((inputs, labels, labels), 0)["loss"]
        assert not bool(head.stats_initialized)
        assert not system.replay_buffers
        loss.backward()
        return loss

    system.optimizer_step(0, 0, optimizer, closure)
    scores_after = head.raw_score_from_embeddings(head.embed(pooled)).detach()
    assert not torch.allclose(scores_before, scores_after)
    torch.testing.assert_close(head.running_raw_score_mean, scores_after.mean())
    torch.testing.assert_close(head.running_raw_score_std, scores_after.std(unbiased=False).clamp_min(head.eps))
    torch.testing.assert_close(system.replay_buffers[0], pooled.detach())
    assert not system._pending_head_updates
    for name, value in system.model.get_head(1).named_parameters():
        torch.testing.assert_close(value, inactive_before[name])


def test_online_oneclass_multihead_etf_smoke(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    cfg = compose_cfg(
        tmp_path / "online_oneclass_etf_root",
        overrides=["experiment=online_oneclass_multihead_etf", *_multihead_class_paths(synthetic_dataset_dir)],
    )
    run_experiment(cfg)
    assert any((tmp_path / "online_oneclass_etf_root").rglob("classification_report.json"))


def test_online_oneclass_multihead_handcrafted_smoke(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    cfg = compose_cfg(
        tmp_path / "online_oneclass_handcrafted_root",
        overrides=[
            "experiment=online_oneclass_multihead",
            "backbone=handcrafted16",
            f"oneclass.feature_bundle_path={HANDCRAFTED_BUNDLE}",
            *_multihead_class_paths(synthetic_dataset_dir),
        ],
    )
    run_experiment(cfg)
    assert any((tmp_path / "online_oneclass_handcrafted_root").rglob("classification_report.json"))
