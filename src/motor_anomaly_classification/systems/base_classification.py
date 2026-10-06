from dataclasses import dataclass, field
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn

from motor_anomaly_classification.models.factory import ModelFactoryConfig, OptimizerConfig, build_model, build_optimizer


@dataclass
class ModelConfig:
    mode: str = "timeseries_net"
    in_channels: int = 2
    num_classes: int = 4
    backbone: dict[str, Any] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)


class BaseClassificationModule(pl.LightningModule):
    def __init__(self, model_config: ModelConfig, optimizer_config: OptimizerConfig):
        super().__init__()
        self.save_hyperparameters()
        #self.save_hyperparameters(ignore=["model_config", "optimizer_config"])
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.model = build_model(ModelFactoryConfig(
            mode=model_config.mode,
            in_channels=model_config.in_channels,
            num_classes=model_config.num_classes,
            backbone=model_config.backbone,
            head=model_config.head,
        ))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(inputs)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> dict[str, Any]:
        inputs, labels = batch
        logits, features = self.model(inputs)
        loss = self.criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=stage != "test", on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return {
            "loss": loss,
            "predictions": predictions.detach(),
            "labels": labels.detach(),
            "features": features.detach(),
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")["loss"]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        return self._shared_step(batch, "test")

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        inputs, _ = batch
        logits, _ = self.model(inputs)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return build_optimizer(self.model, self.optimizer_config)
