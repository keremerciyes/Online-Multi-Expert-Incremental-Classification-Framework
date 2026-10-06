from collections import OrderedDict

import torch
import torch.nn as nn

from motor_anomaly_classification.models.base import BaseTimeSeriesClassifier


class OneClassMultiHeadNet(BaseTimeSeriesClassifier):
    def __init__(self, backbone: nn.Module, heads: dict[str, nn.Module]):
        super().__init__()
        if not heads:
            raise ValueError("OneClassMultiHeadNet requires at least one head.")
        self.backbone = backbone
        self.heads = nn.ModuleDict(OrderedDict(sorted(heads.items(), key=lambda item: int(item[0]))))
        self.active_head: int | None = None
        feature_channels = int(getattr(backbone, "output_dim", 0))
        self.register_buffer("feature_mean", torch.zeros(max(feature_channels, 1)))
        self.register_buffer("feature_std", torch.ones(max(feature_channels, 1)))
        self.register_buffer("feature_standardization_ready", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def set_feature_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean = mean.detach().to(self.feature_mean.device, self.feature_mean.dtype)
        self.feature_std = torch.clamp(std.detach().to(self.feature_std.device, self.feature_std.dtype), min=1e-6)
        self.feature_standardization_ready = torch.tensor(True, device=self.feature_standardization_ready.device)

    def calibration_mode(self) -> str:
        return "voting"

    def calibrated_acceptance_threshold(self) -> float:
        return 0.0

    def set_active_head(self, head_index: int | None) -> None:
        self.active_head = head_index
        if head_index is None:
            for head in self.heads.values():
                for parameter in head.parameters():
                    parameter.requires_grad = True
            return

        # TODO: Consider soft freezing or adapter-based updates to reduce forgetting.
        active_key = str(head_index)
        for key, head in self.heads.items():
            trainable = key == active_key
            for parameter in head.parameters():
                parameter.requires_grad = trainable

    def get_head(self, head_index: int) -> nn.Module:
        return self.heads[str(head_index)]

    @staticmethod
    def pooled_features_from_map(feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.shape[2] == 1:
            return feature_map.squeeze(2)
        avg_pool = feature_map.mean(dim=2)
        max_pool = feature_map.amax(dim=2)
        return torch.cat([avg_pool, max_pool], dim=1)

    def extract_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(inputs)
        if bool(self.feature_standardization_ready.item()):
            feature_map = (feature_map - self.feature_mean.view(1, -1, 1)) / self.feature_std.view(1, -1, 1)
        return feature_map

    def extract_shared_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pooled_features_from_map(self.extract_feature_map(inputs))

    @staticmethod
    def _head_output(head: nn.Module, head_inputs: torch.Tensor, score_kind: str) -> torch.Tensor:
        if hasattr(head, "embed"):
            embeddings = head.embed(head_inputs)
            if score_kind == "voting" and hasattr(head, "voting_score_from_embeddings"):
                return head.voting_score_from_embeddings(embeddings)
            if score_kind == "calibrated" and hasattr(head, "calibrated_score_from_embeddings"):
                return head.calibrated_score_from_embeddings(embeddings)
            if score_kind == "raw" and hasattr(head, "raw_score_from_embeddings"):
                return head.raw_score_from_embeddings(embeddings)
        return head(head_inputs)

    def score_from_representations(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return self._stack_head_scores(pooled_features, feature_map, score_kind="calibrated")

    def raw_score_from_representations(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return self._stack_head_scores(pooled_features, feature_map, score_kind="raw")

    def voting_score_from_representations(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return self._stack_head_scores(pooled_features, feature_map, score_kind="voting")

    def _stack_head_scores(self, pooled_features: torch.Tensor, feature_map: torch.Tensor, score_kind: str) -> torch.Tensor:
        per_head_scores = []
        for key in sorted(self.heads.keys(), key=lambda text: int(text)):
            head = self.heads[key]
            head_inputs = feature_map if bool(getattr(head, "expects_feature_map", False)) else pooled_features
            head_output = self._head_output(head, head_inputs, score_kind=score_kind)
            if head_output.ndim == 2 and head_output.shape[1] == 1:
                head_output = head_output.squeeze(1)
            per_head_scores.append(head_output)
        return torch.stack(per_head_scores, dim=1)

    def score_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_from_representations(features, features.unsqueeze(-1))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.extract_feature_map(inputs)
        pooled_features = self.pooled_features_from_map(feature_map)
        return self.voting_score_from_representations(pooled_features, feature_map), pooled_features

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.extract_shared_features(inputs)

    def extract_representations(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.extract_feature_map(inputs)
        return self.pooled_features_from_map(feature_map), feature_map

    def forward_scores_dict(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        scores, _ = self.forward(inputs)
        return {key: scores[:, index] for index, key in enumerate(sorted(self.heads.keys(), key=lambda text: int(text)))}
