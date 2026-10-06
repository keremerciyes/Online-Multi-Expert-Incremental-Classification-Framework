from pathlib import Path
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, cast

from motor_anomaly_classification.models.factory import OptimizerConfig
from motor_anomaly_classification.systems.base_classification import ModelConfig
from motor_anomaly_classification.systems.online_module import OnlineBehaviorConfig


ONLINE_BEHAVIOR_EXPERIMENTS = {
    "online_oneclass_multihead",
    "online_oneclass_multihead_latent",
    "online_oneclass_multihead_reconstruction",
    "online_oneclass_multihead_etf",
}


def _build_callbacks(callbacks_cfg: DictConfig | None):
    if callbacks_cfg is None:
        return []
    callbacks = []
    for _, callback_cfg in callbacks_cfg.items():
        if callback_cfg is None:
            continue
        callbacks.append(instantiate(callback_cfg))
    return callbacks


def _build_logger(cfg: DictConfig, run_name: str):
    if cfg.get("logging") is None:
        return None
    return instantiate(cfg.logging, name=run_name)


def _load_checkpoint_state(system, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    system.load_state_dict(state_dict, strict=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()
    model_values = cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg = ModelConfig(**model_values)
    model_cfg.num_classes = len(datamodule.class_names)
    model_cfg.in_channels = len(tuple(datamodule.config.channel_indices))
    optimizer_values = cast(dict[str, Any], OmegaConf.to_container(cfg.optimizer, resolve=True))
    optimizer_cfg = OptimizerConfig(**optimizer_values)
    if cfg.experiment.name in ONLINE_BEHAVIOR_EXPERIMENTS:
        behavior_values = cast(dict[str, Any], OmegaConf.to_container(cfg.online, resolve=True))
        behavior_cfg = OnlineBehaviorConfig(**behavior_values)
        system = instantiate(cfg.system, model_config=model_cfg, optimizer_config=optimizer_cfg, behavior_config=behavior_cfg, _recursive_=False)
    else:
        system = instantiate(cfg.system, model_config=model_cfg, optimizer_config=optimizer_cfg, _recursive_=False)
    callbacks = _build_callbacks(cfg.callbacks)
    root_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / "eval"
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=_build_logger(cfg, f"{cfg.experiment.name}_eval"), default_root_dir=str(root_dir))
    if cfg.eval.checkpoint_path:
        _load_checkpoint_state(system, str(cfg.eval.checkpoint_path))
    trainer.test(system, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()
