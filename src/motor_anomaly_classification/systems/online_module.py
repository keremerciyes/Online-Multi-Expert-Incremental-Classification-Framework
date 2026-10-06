from dataclasses import dataclass

import torch

from motor_anomaly_classification.models.factory import OptimizerConfig
from motor_anomaly_classification.systems.base_classification import BaseClassificationModule, ModelConfig


@dataclass
class OnlineBehaviorConfig:
    freeze_backbone: bool = False


class OnlineClassificationModule(BaseClassificationModule):
    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        behavior_config: OnlineBehaviorConfig,
    ):
        super().__init__(model_config=model_config, optimizer_config=optimizer_config)
        self.behavior_config = behavior_config
        self.previous_last_label: int | None = None

    def on_train_epoch_start(self) -> None:
        self.previous_last_label = None

    def on_fit_start(self) -> None:
        if not self.behavior_config.freeze_backbone:
            return
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad = False
        self.model.backbone.eval()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor | bool | str]:
        if self.behavior_config.freeze_backbone:
            self.model.backbone.eval()
        step_output = self._shared_step(batch, "train")
        _, labels = batch
        labels_np = labels.detach().cpu().numpy().astype(int)
        within_change = bool((labels_np[1:] != labels_np[:-1]).any()) if labels_np.size > 1 else False
        boundary_change = self.previous_last_label is not None and labels_np.size > 0 and int(labels_np[0]) != int(self.previous_last_label)
        change_to: list[str] = []
        if boundary_change:
            change_to.append(str(int(labels_np[0])))
        if labels_np.size > 1:
            for before, after in zip(labels_np[:-1], labels_np[1:]):
                if before != after:
                    change_to.append(str(int(after)))
        self.previous_last_label = int(labels_np[-1]) if labels_np.size else self.previous_last_label
        return {
            "loss": step_output["loss"],
            "batch_change": bool(within_change or boundary_change),
            "batch_change_to": "|".join(dict.fromkeys(change_to)),
        }
