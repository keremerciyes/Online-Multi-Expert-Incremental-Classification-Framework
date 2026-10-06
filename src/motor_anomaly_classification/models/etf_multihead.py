import torch
import torch.nn as nn
import torch.nn.functional as F

from motor_anomaly_classification.models.base import BaseTimeSeriesClassifier


def simplex_etf_targets(num_classes: int, dim: int) -> torch.Tensor:
    """Return ``num_classes`` unit vectors of dimension ``dim`` forming a simplex
    equiangular tight frame (pairwise cosine = -1/(num_classes-1))."""
    if dim < num_classes:
        raise ValueError(f"Simplex ETF needs dim >= num_classes; got dim={dim}, num_classes={num_classes}.")
    identity = torch.eye(num_classes, dtype=torch.float32)
    centering = identity - torch.full((num_classes, num_classes), 1.0 / num_classes)
    targets = torch.zeros(num_classes, dim, dtype=torch.float32)
    targets[:, :num_classes] = centering
    return targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-12)


class _EtfHeadView:
    """Lightweight per-class view sharing the net projector and one fixed ETF target.

    Only the shared projector is trainable. Targets are frozen buffers, so no class
    prototype receives negative optimization while other classes stream.
    """

    def __init__(self, net: "EtfMultiHeadNet", class_index: int):
        self._net = net
        self.class_index = int(class_index)
        self.expects_feature_map = bool(getattr(net, "projector_arch", "cosine") == "conv")

    def _target(self) -> torch.Tensor:
        return self._net.targets[self.class_index]

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        return self._net.project(features)

    def alignment_loss(self, features: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(self.embed(features), dim=1)
        return (1.0 - torch.matmul(normalized, self._target())).mean()

    def raw_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings, dim=1)
        return self._net.scale * torch.matmul(normalized, self._target())

    def calibrated_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings, dim=1)
        return torch.matmul(normalized, self._target())

    def voting_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.calibrated_score_from_embeddings(embeddings)

    def calibration_mode(self) -> str:
        return "cosine_etf"

    def calibrated_acceptance_threshold(self) -> float:
        return 0.0


class EtfMultiHeadNet(BaseTimeSeriesClassifier):
    """Shared single-layer conv projector plus fixed simplex-ETF class targets.

    All classes live in one embedding space. Streaming a class only aligns the shared
    projector to that class's fixed target; targets never move and no negative push is
    applied to other targets.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        projector_arch: str = "cosine",
        embedding_dim: int = 16,
        projector_channels: int = 8,
        scale: float = 15.0,
        projector_bias: bool = False,
    ):
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("EtfMultiHeadNet requires num_classes >= 2.")
        self.backbone = backbone
        self.scale = float(scale)
        self.projector_arch = str(projector_arch).lower()
        feature_channels = int(getattr(backbone, "output_dim"))
        pooled_dim = 2 * feature_channels
        if self.projector_arch == "conv":
            # Single 1x1 conv over the frozen feature map, then pooling.
            self.head = nn.Sequential(nn.Conv1d(feature_channels, int(projector_channels), kernel_size=1, bias=bool(projector_bias)))
            target_dim = 2 * int(projector_channels)
        else:
            # Default: a single shared cosine (linear) layer on pooled features.
            self.projector_arch = "cosine"
            target_dim = int(embedding_dim)
            self.head = nn.Linear(pooled_dim, target_dim, bias=bool(projector_bias))
        self.register_buffer("targets", simplex_etf_targets(int(num_classes), target_dim))
        self.register_buffer("feature_mean", torch.zeros(max(feature_channels, 1)))
        self.register_buffer("feature_std", torch.ones(max(feature_channels, 1)))
        self.register_buffer("feature_standardization_ready", torch.tensor(False, dtype=torch.bool))
        self._head_views = {index: _EtfHeadView(self, index) for index in range(int(num_classes))}
        self.active_head: int | None = None

    @torch.no_grad()
    def set_feature_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean = mean.detach().to(self.feature_mean.device, self.feature_mean.dtype)
        self.feature_std = torch.clamp(std.detach().to(self.feature_std.device, self.feature_std.dtype), min=1e-6)
        self.feature_standardization_ready = torch.tensor(True, device=self.feature_standardization_ready.device)

    def set_active_head(self, head_index: int | None) -> None:
        # Shared projector is always trainable; fixed targets never receive gradients.
        self.active_head = head_index

    def get_head(self, head_index: int) -> _EtfHeadView:
        return self._head_views[int(head_index)]

    def calibration_mode(self) -> str:
        return "cosine_etf"

    def calibrated_acceptance_threshold(self) -> float:
        return 0.0

    @staticmethod
    def pooled_features_from_map(feature_map: torch.Tensor) -> torch.Tensor:
        return torch.cat([feature_map.mean(dim=2), feature_map.amax(dim=2)], dim=1)

    def extract_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(inputs)
        if bool(self.feature_standardization_ready.item()):
            feature_map = (feature_map - self.feature_mean.view(1, -1, 1)) / self.feature_std.view(1, -1, 1)
        return feature_map

    def project(self, features: torch.Tensor) -> torch.Tensor:
        if self.projector_arch == "conv":
            return self.pooled_features_from_map(self.head(features))
        return self.head(features)

    def extract_representations(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.extract_feature_map(inputs)
        return self.pooled_features_from_map(feature_map), feature_map

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pooled_features_from_map(self.extract_feature_map(inputs))

    def _projector_input(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map if self.projector_arch == "conv" else pooled_features

    def _stack_head_scores(self, pooled_features: torch.Tensor, feature_map: torch.Tensor, score_kind: str) -> torch.Tensor:
        embeddings = self.project(self._projector_input(pooled_features, feature_map))
        per_head_scores = []
        for index in range(len(self._head_views)):
            head = self._head_views[index]
            if score_kind == "raw":
                per_head_scores.append(head.raw_score_from_embeddings(embeddings))
            else:
                per_head_scores.append(head.calibrated_score_from_embeddings(embeddings))
        return torch.stack(per_head_scores, dim=1)

    def raw_score_from_representations(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return self._stack_head_scores(pooled_features, feature_map, score_kind="raw")

    def score_from_representations(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return self._stack_head_scores(pooled_features, feature_map, score_kind="calibrated")

    def voting_score_from_representations(self, pooled_features: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        return self._stack_head_scores(pooled_features, feature_map, score_kind="calibrated")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.extract_feature_map(inputs)
        pooled_features = self.pooled_features_from_map(feature_map)
        return self._stack_head_scores(pooled_features, feature_map, score_kind="calibrated"), pooled_features
