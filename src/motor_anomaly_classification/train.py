from pathlib import Path
from typing import Any, cast

import hydra
import lightning.pytorch as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from motor_anomaly_classification.models.factory import OptimizerConfig
from motor_anomaly_classification.systems.base_classification import ModelConfig
from motor_anomaly_classification.systems.online_module import OnlineBehaviorConfig
from motor_anomaly_classification.utils.seed import set_seed


ONLINE_BEHAVIOR_EXPERIMENTS = {
    "online_oneclass_multihead",
    "online_oneclass_multihead_latent",
    "online_oneclass_multihead_reconstruction",
    "online_oneclass_multihead_etf",
}


def _build_callbacks(callbacks_cfg: DictConfig | None) -> list[pl.Callback]:
    if callbacks_cfg is None:
        return []
    callbacks: list[pl.Callback] = []
    for _, callback_cfg in callbacks_cfg.items():
        if callback_cfg is None:
            continue
        callbacks.append(instantiate(callback_cfg))
    return callbacks


def _build_model_config(cfg: DictConfig, datamodule: Any) -> ModelConfig:
    model_values = cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg = ModelConfig(**model_values)
    model_cfg.num_classes = len(datamodule.class_names)
    model_cfg.in_channels = len(tuple(datamodule.config.channel_indices))
    return model_cfg


def _build_optimizer_config(cfg: DictConfig) -> OptimizerConfig:
    optimizer_values = cast(dict[str, Any], OmegaConf.to_container(cfg.optimizer, resolve=True))
    return OptimizerConfig(**optimizer_values)


def _build_logger(cfg: DictConfig, run_name: str):
    if cfg.get("logging") is None:
        return None
    return instantiate(cfg.logging, name=run_name)


def _build_trainer(cfg: DictConfig, callbacks: list[pl.Callback], run_name: str, stage_name: str | None = None) -> pl.Trainer:
    root_dir = Path(cfg.paths.output_dir) / cfg.experiment.name
    if stage_name:
        root_dir = root_dir / stage_name
    logger = _build_logger(cfg, run_name)
    return instantiate(cfg.trainer, callbacks=callbacks, logger=logger, default_root_dir=str(root_dir))


def _build_system(cfg: DictConfig, model_cfg: ModelConfig, optimizer_cfg: OptimizerConfig):
    if cfg.experiment.name in ONLINE_BEHAVIOR_EXPERIMENTS:
        behavior_values = cast(dict[str, Any], OmegaConf.to_container(cfg.online, resolve=True))
        behavior_cfg = OnlineBehaviorConfig(**behavior_values)
        return instantiate(
            cfg.system,
            model_config=model_cfg,
            optimizer_config=optimizer_cfg,
            behavior_config=behavior_cfg,
            _recursive_=False,
        )
    return instantiate(cfg.system, model_config=model_cfg, optimizer_config=optimizer_cfg, _recursive_=False)


def _run_fit_and_test(cfg: DictConfig, system, datamodule, stage_name: str | None = None):
    callbacks = _build_callbacks(cfg.callbacks)
    run_name = cfg.experiment.name if stage_name is None else f"{cfg.experiment.name}_{stage_name}"
    trainer = _build_trainer(cfg, callbacks, run_name=run_name, stage_name=stage_name)
    trainer.fit(system, datamodule=datamodule)
    results = trainer.test(system, datamodule=datamodule, ckpt_path=None)
    if wandb.run is not None:
        wandb.finish()
    return results


def run_experiment(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()

    model_cfg = _build_model_config(cfg, datamodule)
    optimizer_cfg = _build_optimizer_config(cfg)
    system = _build_system(cfg, model_cfg, optimizer_cfg)
    _run_fit_and_test(cfg, system, datamodule)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    pl.seed_everything(int(cfg.seed), workers=True)
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
