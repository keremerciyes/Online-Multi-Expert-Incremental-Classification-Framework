from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from motor_anomaly_classification.models.factory import OptimizerConfig, build_optimizer
from motor_anomaly_classification.systems.base_classification import BaseClassificationModule, ModelConfig
from motor_anomaly_classification.systems.online_module import OnlineBehaviorConfig


@dataclass
class OneClassTrainingConfig:
    assume_single_class_batches: bool = True
    pretrained_backbone_checkpoint_path: str | None = None
    freeze_inactive_heads: bool = True
    use_alignment_loss: bool = True
    use_boundary_loss: bool = True
    use_old_contrast_loss: bool = True
    use_replay_negative_suppression_loss: bool = True
    alignment_weight: float = 1.0
    boundary_weight: float = 1.0
    old_contrast_weight: float = 1.0
    replay_negative_suppression_weight: float = 1.0
    old_contrast_margin: float = 0.2
    replay_negative_margin: float = 0.1
    replay_buffer_size_per_class: int = 64
    replay_batch_size_per_class: int = 8
    replay_pool_samples_per_class: int | None = None
    replay_batch_size: int | None = None
    min_replay_samples_per_class: int = 32
    use_hard_negative_mining: bool = True
    hard_negative_pool_multiplier: int = 4
    init_feature_standardization: bool = True
    feature_std_warmup: int | None = None
    use_uncertainty_weighting: bool = False
    contrast_score_mode: str = "calibrated"
    use_raw_scores_for_training: bool = True
    log_threshold_metrics: bool = True


class OnlineOneClassMultiHeadModule(BaseClassificationModule):
    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        behavior_config: OnlineBehaviorConfig,
        training_config: OneClassTrainingConfig | None = None,
    ):
        super().__init__(model_config=model_config, optimizer_config=optimizer_config)
        self.behavior_config = behavior_config
        self.training_config = training_config or OneClassTrainingConfig()
        self.previous_last_label: int | None = None
        self.replay_buffers: dict[int, torch.Tensor] = {}
        self.replay_seen_counts: dict[int, int] = {}
        self._pending_head_updates: list[tuple[int, torch.Tensor]] = []
        # Learnable homoscedastic log-variances for [alignment, boundary, old_contrast, replay_negative].
        self.loss_log_vars = torch.nn.Parameter(torch.zeros(4))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = build_optimizer(self.model, self.optimizer_config)
        if bool(self.training_config.use_uncertainty_weighting):
            optimizer.add_param_group({"params": [self.loss_log_vars], "lr": self.optimizer_config.learning_rate_head})
        return optimizer

    @staticmethod
    def _split_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(batch) == 2:
            inputs, labels = batch
            return inputs, labels, None
        if len(batch) == 3:
            inputs, labels, head_indices = batch
            return inputs, labels, head_indices
        else:
            raise ValueError("Expected batch to be (inputs, labels) or (inputs, labels, head_indices).")

    def on_train_epoch_start(self) -> None:
        self.previous_last_label = None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        for head_index, head_inputs in self._pending_head_updates:
            head = self.model.get_head(head_index)
            if hasattr(head, "update_running_stats"):
                head.update_running_stats(head_inputs)
            self._append_replay_buffer(head_index, head_inputs)
        self._pending_head_updates.clear()

    def _append_replay_buffer(self, class_index: int, representations: torch.Tensor) -> None:
        """Algorithm-R reservoir per class, so every class keeps an unbiased sample of its own stream."""
        capacity = int(self.training_config.replay_buffer_size_per_class)
        if capacity <= 0:
            return
        incoming = representations.detach().cpu()
        reservoir = self.replay_buffers.get(class_index)
        seen = int(self.replay_seen_counts.get(class_index, 0))
        if reservoir is None:
            reservoir = incoming.new_empty((0, *incoming.shape[1:]))

        free_slots = capacity - int(reservoir.shape[0])
        if free_slots > 0:
            fill_count = min(free_slots, int(incoming.shape[0]))
            reservoir = torch.cat([reservoir, incoming[:fill_count]], dim=0)
            seen += fill_count
            incoming = incoming[fill_count:]

        if incoming.shape[0] > 0:
            arrival_positions = seen + torch.arange(1, int(incoming.shape[0]) + 1)
            slots = (torch.rand(int(incoming.shape[0])) * arrival_positions).long()
            # Apply accepted swaps in arrival order so repeated slot hits resolve like the sequential algorithm.
            for offset in torch.nonzero(slots < capacity, as_tuple=False).flatten().tolist():
                reservoir[int(slots[offset])] = incoming[offset]
            seen += int(incoming.shape[0])

        self.replay_buffers[class_index] = reservoir
        self.replay_seen_counts[class_index] = seen

    def _sample_replay_buffer(self, class_indices: list[int], device: torch.device, per_class_count: int | None = None) -> torch.Tensor | None:
        samples: list[torch.Tensor] = []
        batch_size = int(per_class_count) if per_class_count is not None else int(self.training_config.replay_batch_size_per_class)
        min_samples = int(self.training_config.min_replay_samples_per_class)
        if batch_size <= 0:
            return None
        for class_index in class_indices:
            stored = self.replay_buffers.get(class_index)
            if stored is None or stored.shape[0] < max(1, min_samples):
                continue
            sample_count = min(batch_size, stored.shape[0])
            permutation = torch.randperm(stored.shape[0])[:sample_count]
            samples.append(stored[permutation])
        if not samples:
            return None
        return torch.cat(samples, dim=0).to(device)

    @staticmethod
    def _head_calibrated_scores(head: Any, head_inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(head, "embed") and hasattr(head, "calibrated_score_from_embeddings"):
            embeddings = head.embed(head_inputs)
            return head.calibrated_score_from_embeddings(embeddings)
        output = head(head_inputs)
        return output.squeeze(1) if output.ndim == 2 and output.shape[1] == 1 else output

    def _compute_replay_negative_loss(self, active_head_index: int, active_head: Any, active_head_inputs: torch.Tensor) -> torch.Tensor:
        zero_loss = torch.tensor(0.0, device=active_head_inputs.device)
        if active_head is None:
            return zero_loss
        margin = float(self.training_config.replay_negative_margin)
        target_count = int(self.training_config.replay_batch_size_per_class)
        replay_negative_indices = [index for index in sorted(self.replay_buffers) if index != active_head_index]
        if bool(self.training_config.use_hard_negative_mining):
            multiplier = max(int(self.training_config.hard_negative_pool_multiplier), 1)
            pool_count = getattr(self.training_config, "replay_pool_samples_per_class", None)
            pool_count = target_count * multiplier if pool_count is None else int(pool_count)
            pool_inputs = self._sample_replay_buffer(replay_negative_indices, active_head_inputs.device, per_class_count=pool_count)
            if pool_inputs is None:
                return zero_loss
            pool_scores = self._head_calibrated_scores(active_head, pool_inputs)
            replay_batch_size = getattr(self.training_config, "replay_batch_size", None)
            keep = max(1, pool_inputs.shape[0] // multiplier) if replay_batch_size is None else int(replay_batch_size)
            if keep <= 0:
                return zero_loss
            keep = min(pool_inputs.shape[0], keep)
            hardest_scores = torch.topk(pool_scores, keep).values
            return torch.relu(hardest_scores - margin).mean()
        replay_negative_inputs = self._sample_replay_buffer(replay_negative_indices, active_head_inputs.device)
        if replay_negative_inputs is None:
            return zero_loss
        negative_scores = self._head_calibrated_scores(active_head, replay_negative_inputs)
        return torch.relu(negative_scores - margin).mean()

    @torch.no_grad()
    def _initialize_feature_standardization(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "train_dataloader"):
            return
        loader = datamodule.train_dataloader()
        channel_sum: torch.Tensor | None = None
        channel_sq_sum: torch.Tensor | None = None
        total_count = 0
        self.model.backbone.eval()
        for batch_index, batch in enumerate(loader):
            warmup = getattr(self.training_config, "feature_std_warmup", None)
            if warmup is not None and batch_index >= int(warmup):
                break
            inputs = batch[0].to(self.device)
            feature_map = self.model.backbone(inputs)
            channels = feature_map.shape[1]
            flat = feature_map.permute(0, 2, 1).reshape(-1, channels)
            batch_sum = flat.sum(dim=0)
            batch_sq_sum = (flat * flat).sum(dim=0)
            channel_sum = batch_sum if channel_sum is None else channel_sum + batch_sum
            channel_sq_sum = batch_sq_sum if channel_sq_sum is None else channel_sq_sum + batch_sq_sum
            total_count += int(flat.shape[0])
        if channel_sum is None or total_count == 0:
            return
        mean = channel_sum / total_count
        variance = torch.clamp(channel_sq_sum / total_count - mean * mean, min=0.0)
        std = torch.sqrt(variance) + 1e-6
        self.model.set_feature_standardization(mean, std)

    def on_fit_start(self) -> None:
        checkpoint_path = self.training_config.pretrained_backbone_checkpoint_path
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            backbone_state = {
                key.removeprefix("model.backbone."): value
                for key, value in state_dict.items()
                if key.startswith("model.backbone.")
            }
            if backbone_state:
                self.model.backbone.load_state_dict(backbone_state, strict=True)
            elif hasattr(self.model.backbone, "encoder"):
                from motor_anomaly_classification.models.encoder_presets import load_encoder_weights
                load_encoder_weights(self.model.backbone.encoder, checkpoint_path)
            else:
                raise ValueError("No model.backbone.* weights found in pretrained checkpoint, and backbone has no encoder to load raw weights.")

        if not self.behavior_config.freeze_backbone:
            return
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad = False
        self.model.backbone.eval()
        if bool(self.training_config.init_feature_standardization) and hasattr(self.model, "set_feature_standardization"):
            self._initialize_feature_standardization()
            self.model.backbone.eval()

    def _shared_eval(self, batch: tuple[torch.Tensor, ...], stage: str) -> dict[str, Any]:
        inputs, labels, _ = self._split_batch(batch)
        scores, features = self.model(inputs)
        raw_scores = None
        if hasattr(self.model, "extract_representations") and hasattr(self.model, "raw_score_from_representations"):
            pooled_features, feature_map = self.model.extract_representations(inputs)
            raw_scores = self.model.raw_score_from_representations(pooled_features, feature_map)
        loss = F.cross_entropy(scores, labels)
        predictions = torch.argmax(scores, dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=stage != "test", on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        if bool(self.training_config.log_threshold_metrics):
            accepted_mask = scores.max(dim=1).values > 0.0
            accepted_rate = accepted_mask.float().mean()
            self.log(f"{stage}/accepted_rate", accepted_rate, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{stage}/all_below_threshold_rate", (~accepted_mask).float().mean(), prog_bar=False, on_epoch=True, on_step=False)
            if bool(accepted_mask.any().item()):
                gated_accuracy = (predictions[accepted_mask] == labels[accepted_mask]).float().mean()
                self.log(f"{stage}/gated_acc", gated_accuracy, prog_bar=False, on_epoch=True, on_step=False)
        return {
            "loss": loss,
            "predictions": predictions.detach(),
            "labels": labels.detach(),
            "features": features.detach(),
            "scores": scores.detach(),
            "raw_scores": raw_scores.detach() if raw_scores is not None else None,
        }

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict[str, torch.Tensor | bool | str]:
        inputs, labels, head_indices = self._split_batch(batch)
        if self.behavior_config.freeze_backbone:
            self.model.backbone.eval()

        if head_indices is None:
            raise ValueError("Online one-class multi-head training requires explicit head indices in train batches.")
        active_head_index = int(head_indices[0].item())
        if bool((head_indices != active_head_index).any().item()):
            raise ValueError("Train batch contains multiple active heads; expected one head per sequential phase.")

        if hasattr(self.model, "set_active_head"):
            if bool(self.training_config.freeze_inactive_heads):
                self.model.set_active_head(active_head_index)
            else:
                self.model.set_active_head(None)

        if hasattr(self.model, "extract_representations"):
            pooled_features, feature_map = self.model.extract_representations(inputs)
        else:
            pooled_features = self.model.extract_embeddings(inputs)
            feature_map = pooled_features.unsqueeze(-1)
        active_head = self.model.get_head(active_head_index) if hasattr(self.model, "get_head") else None
        active_head_inputs = feature_map if active_head is not None and bool(getattr(active_head, "expects_feature_map", False)) else pooled_features

        raw_scores = None
        if hasattr(self.model, "raw_score_from_representations"):
            raw_scores = self.model.raw_score_from_representations(pooled_features, feature_map)
        if hasattr(self.model, "score_from_representations"):
            calibrated_scores = self.model.score_from_representations(pooled_features, feature_map)
        else:
            calibrated_scores = self.model.score_from_features(pooled_features) if hasattr(self.model, "score_from_features") else self.model(inputs)[0]
        if bool(self.training_config.use_raw_scores_for_training) and raw_scores is not None:
            scores = raw_scores
        else:
            scores = calibrated_scores
        contrast_scores = raw_scores if str(self.training_config.contrast_score_mode).lower() == "raw" and raw_scores is not None else calibrated_scores
        active_scores = contrast_scores[:, active_head_index]
        zero_loss = torch.tensor(0.0, device=inputs.device)
        alignment_loss = zero_loss
        if bool(self.training_config.use_alignment_loss) and active_head is not None and hasattr(active_head, "alignment_loss"):
            alignment_loss = active_head.alignment_loss(active_head_inputs)

        boundary_loss = zero_loss
        if bool(self.training_config.use_boundary_loss) and active_head is not None and hasattr(active_head, "boundary_loss"):
            boundary_loss = active_head.boundary_loss(active_head_inputs)

        old_contrast_loss = zero_loss
        replay_negative_suppression_loss = zero_loss
        num_heads = contrast_scores.shape[1]
        if num_heads > 1:
            old_indices = [index for index in range(active_head_index)]
            if bool(self.training_config.use_old_contrast_loss) and old_indices:
                max_old = contrast_scores[:, old_indices].detach().max(dim=1).values
                margin = float(self.training_config.old_contrast_margin)
                old_contrast_loss = torch.relu(max_old - active_scores + margin).mean()

        if bool(self.training_config.use_replay_negative_suppression_loss):
            replay_negative_suppression_loss = self._compute_replay_negative_loss(active_head_index, active_head, active_head_inputs)

        loss_terms = [alignment_loss, boundary_loss, old_contrast_loss, replay_negative_suppression_loss]
        manual_weights = [
            float(self.training_config.alignment_weight),
            float(self.training_config.boundary_weight),
            float(self.training_config.old_contrast_weight),
            float(self.training_config.replay_negative_suppression_weight),
        ]
        if bool(self.training_config.use_uncertainty_weighting):
            loss = torch.zeros((), device=inputs.device)
            for index, term in enumerate(loss_terms):
                if not term.requires_grad:
                    continue
                precision = torch.exp(-self.loss_log_vars[index])
                loss = loss + precision * term + self.loss_log_vars[index]
                self.log(f"train/uncertainty_weight_{index}", precision.detach(), prog_bar=False, on_epoch=True, on_step=False)
        else:
            loss = (
                manual_weights[0] * alignment_loss
                + manual_weights[1] * boundary_loss
                + manual_weights[2] * old_contrast_loss
                + manual_weights[3] * replay_negative_suppression_loss
            )

        self._pending_head_updates.append((active_head_index, active_head_inputs.detach()))

        predictions = torch.argmax(scores, dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/alignment_loss", alignment_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/boundary_loss", boundary_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/old_contrast_loss", old_contrast_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/replay_negative_suppression_loss", replay_negative_suppression_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/weighted_alignment_loss", float(self.training_config.alignment_weight) * alignment_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/weighted_boundary_loss", float(self.training_config.boundary_weight) * boundary_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/weighted_old_contrast_loss", float(self.training_config.old_contrast_weight) * old_contrast_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train/weighted_replay_negative_suppression_loss", float(self.training_config.replay_negative_suppression_weight) * replay_negative_suppression_loss, prog_bar=False, on_epoch=True, on_step=False)

        labels_np = labels.detach().cpu().numpy().astype(int)
        if self.training_config.assume_single_class_batches and labels_np.size > 1:
            batch_is_mixed = bool((labels_np[1:] != labels_np[:-1]).any())
            if batch_is_mixed:
                self.log("train/mixed_batch_warning", 1.0, prog_bar=False, on_epoch=True, on_step=False)

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
            "loss": loss,
            "batch_change": bool(within_change or boundary_change),
            "batch_change_to": "|".join(dict.fromkeys(change_to)),
        }

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict[str, Any]:
        return self._shared_eval(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict[str, Any]:
        return self._shared_eval(batch, "test")

    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        inputs, _, _ = self._split_batch(batch)
        scores, _ = self.model(inputs)
        return torch.argmax(scores, dim=1)
