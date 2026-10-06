import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

from motor_anomaly_classification.models.base import BaseHead


def _resolve_num_classes(num_classes: int | None, selected_classes: tuple[str, ...]) -> int:
    return len(selected_classes) if num_classes is None else int(num_classes)


class _CosineLayer(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, scale: float = 15.0):
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = F.normalize(features, dim=1)
        normalized_weights = F.normalize(self.weight, dim=1)
        return self.scale * torch.matmul(normalized_features, normalized_weights.t())


class _ConvProjectorMixin:
    @staticmethod
    def _build_conv_projector(
        in_channels: int,
        projector_channels: int,
        projector_bias: bool,
        use_batch_norm: bool,
    ) -> nn.Module:
        layers: list[nn.Module] = [
            nn.Conv1d(
                int(in_channels),
                int(projector_channels),
                kernel_size=1,
                bias=bool(projector_bias and not use_batch_norm),
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(int(projector_channels)))
        return nn.Sequential(*layers)

    @staticmethod
    def _pool_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
        avg_features = feature_map.mean(dim=2)
        max_features = feature_map.amax(dim=2)
        return torch.cat([avg_features, max_features], dim=1)


class MultiClassCosineHeadConvProjector(BaseHead, _ConvProjectorMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int | None = None,
        projector_channels: int = 8,
        scale: float = 15.0,
        projector_bias: bool = False,
        use_batch_norm: bool = False,
        selected_classes: tuple[str, ...] = ("belt", "magnetic", "vibration"),
    ):
        super().__init__()
        self.expects_feature_map = True
        resolved_num_classes = _resolve_num_classes(num_classes, selected_classes)
        self.projector = self._build_conv_projector(in_channels, projector_channels, projector_bias, use_batch_norm)
        self.classifier = _CosineLayer(2 * int(projector_channels), resolved_num_classes, scale=scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projector(features)
        pooled_features = self._pool_feature_map(projected)
        return self.classifier(pooled_features)


class MultiClassDeepCosineHeadConvProjector(BaseHead, _ConvProjectorMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int | None = None,
        projector_channels: int = 8,
        hidden_dim: int = 16,
        scale: float = 15.0,
        projector_bias: bool = False,
        hidden_bias: bool = True,
        use_batch_norm: bool = False,
        selected_classes: tuple[str, ...] = ("belt", "magnetic", "vibration"),
    ):
        super().__init__()
        self.expects_feature_map = True
        resolved_num_classes = _resolve_num_classes(num_classes, selected_classes)
        self.projector = self._build_conv_projector(in_channels, projector_channels, projector_bias, use_batch_norm)
        self.hidden = nn.Sequential(
            nn.Linear(2 * int(projector_channels), int(hidden_dim), bias=bool(hidden_bias)),
            nn.ReLU(),
        )
        self.classifier = _CosineLayer(int(hidden_dim), resolved_num_classes, scale=scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projector(features)
        pooled_features = self._pool_feature_map(projected)
        hidden_features = self.hidden(pooled_features)
        return self.classifier(hidden_features)


class MultiClassCosineHeadLinearProjector(BaseHead):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int | None = None,
        projector_dim: int | None = None,
        scale: float = 15.0,
        projector_bias: bool = False,
        selected_classes: tuple[str, ...] = ("belt", "magnetic", "vibration"),
    ):
        super().__init__()
        resolved_num_classes = _resolve_num_classes(num_classes, selected_classes)
        embedding_dim = int(projector_dim or feat_dim)
        self.projector = nn.Linear(int(feat_dim), embedding_dim, bias=bool(projector_bias))
        self.classifier = _CosineLayer(embedding_dim, resolved_num_classes, scale=scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projector(features)
        return self.classifier(projected)


class _OneClassCosineMixin:
    prototype: nn.Parameter
    scale: float
    distance_weight: float
    threshold_std_multiplier: float
    stats_momentum: float
    boundary_radius: float
    eps: float
    running_raw_score_mean: torch.Tensor
    running_raw_score_std: torch.Tensor
    running_threshold: torch.Tensor
    stats_initialized: torch.Tensor
    embedding_mean: torch.Tensor
    embedding_cov: torch.Tensor
    embedding_dim: int
    voting_mode: str
    cov_shrinkage: float
    mahalanobis_confidence: float

    def _init_one_class_state(
        self,
        embedding_dim: int,
        scale: float,
        distance_weight: float,
        threshold_std_multiplier: float,
        stats_momentum: float,
        boundary_radius: float,
        eps: float,
        voting_calibration_mode: str = "mahalanobis",
        cov_shrinkage: float = 0.1,
        mahalanobis_confidence: float = 0.95,
    ) -> None:
        self.scale = float(scale)
        self.distance_weight = float(distance_weight)
        self.threshold_std_multiplier = float(threshold_std_multiplier)
        self.stats_momentum = float(stats_momentum)
        self.boundary_radius = float(boundary_radius)
        self.eps = float(eps)
        self.embedding_dim = int(embedding_dim)
        self.voting_mode = str(voting_calibration_mode)
        self.cov_shrinkage = float(cov_shrinkage)
        self.mahalanobis_confidence = float(mahalanobis_confidence)
        self.prototype = nn.Parameter(torch.empty(int(embedding_dim)))
        nn.init.normal_(self.prototype, mean=0.0, std=0.02)
        module = cast(nn.Module, self)
        module.register_buffer("running_raw_score_mean", torch.tensor(0.0))
        module.register_buffer("running_raw_score_std", torch.tensor(1.0))
        module.register_buffer("running_threshold", torch.tensor(0.0))
        module.register_buffer("stats_initialized", torch.tensor(False, dtype=torch.bool))
        module.register_buffer("embedding_mean", torch.zeros(int(embedding_dim)))
        module.register_buffer("embedding_cov", torch.eye(int(embedding_dim)))

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def normalized_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embed(features), dim=1)

    def normalized_prototype(self) -> torch.Tensor:
        return F.normalize(self.prototype.unsqueeze(0), dim=1).squeeze(0)

    def cosine_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        return self.scale * torch.matmul(normalized_embeddings, self.normalized_prototype())

    def angular_distance_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        prototype = self.normalized_prototype()
        return torch.sum((normalized_embeddings - prototype.unsqueeze(0)) ** 2, dim=1)

    def raw_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.cosine_score_from_embeddings(embeddings)

    def raw_acceptance_threshold(self) -> float:
        radius_sq = self.boundary_radius * self.boundary_radius
        cosine_threshold = 1.0 - 0.5 * radius_sq
        return float(self.scale * cosine_threshold)

    def calibrated_acceptance_threshold(self) -> float:
        return 0.0

    def calibration_mode(self) -> str:
        return self.voting_calibration_mode()

    def calibrated_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        distance = self.angular_distance_from_embeddings(embeddings)
        radius_sq = max(self.boundary_radius * self.boundary_radius, self.eps)
        return (radius_sq - distance) / radius_sq

    def voting_calibration_mode(self) -> str:
        if not bool(self.stats_initialized.item()):
            return "boundary_margin"
        mode = str(self.voting_mode).lower()
        if mode == "zscore":
            return "zscore"
        if mode == "boundary":
            return "boundary_margin"
        return "mahalanobis"

    def _regularized_precision(self) -> torch.Tensor:
        covariance = self.embedding_cov
        identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        trace_mean = torch.diagonal(covariance).mean().clamp_min(self.eps)
        shrinkage = float(self.cov_shrinkage)
        regularized = (1.0 - shrinkage) * covariance + shrinkage * trace_mean * identity
        try:
            return torch.linalg.inv(regularized)
        except RuntimeError:
            return identity / trace_mean

    def mahalanobis_distance_sq(self, embeddings: torch.Tensor) -> torch.Tensor:
        precision = self._regularized_precision().to(embeddings.dtype)
        diff = embeddings - self.embedding_mean.unsqueeze(0)
        return torch.einsum("bi,ij,bj->b", diff, precision, diff)

    def _chi2_threshold(self) -> float:
        dof = float(self.embedding_dim)
        probability = min(max(float(self.mahalanobis_confidence), 1e-4), 1.0 - 1e-4)
        z = float(torch.erfinv(torch.tensor(2.0 * probability - 1.0)).item()) * math.sqrt(2.0)
        term = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
        return max(dof * term ** 3, self.eps)

    def voting_score_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        mode = str(self.voting_mode).lower()
        if not bool(self.stats_initialized.item()) or mode == "boundary":
            return self.calibrated_score_from_embeddings(embeddings)
        if mode == "zscore":
            raw_score = self.raw_score_from_embeddings(embeddings)
            std = self.running_raw_score_std.clamp_min(self.eps)
            return (raw_score - self.running_raw_score_mean) / std + self.threshold_std_multiplier
        distance_sq = self.mahalanobis_distance_sq(embeddings)
        threshold = self._chi2_threshold()
        return 1.0 - distance_sq / threshold

    def alignment_loss(self, features: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = self.normalized_embeddings(features)
        cosine = torch.matmul(normalized_embeddings, self.normalized_prototype())
        return (1.0 - cosine).mean()

    def boundary_loss(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(features)
        distance = self.angular_distance_from_embeddings(embeddings)
        radius_sq = max(self.boundary_radius * self.boundary_radius, self.eps)
        return torch.relu(distance / radius_sq - 1.0).mean()

    @torch.no_grad()
    def update_running_stats(self, features: torch.Tensor) -> None:
        embeddings = self.embed(features).detach()
        batch_raw_score = self.raw_score_from_embeddings(embeddings)
        batch_raw_mean = batch_raw_score.mean()
        batch_raw_std = torch.clamp(batch_raw_score.std(unbiased=False), min=self.eps)
        batch_threshold = batch_raw_mean - self.threshold_std_multiplier * batch_raw_std
        batch_mean_vector = embeddings.mean(dim=0)
        centered = embeddings - batch_mean_vector.unsqueeze(0)
        denominator = max(int(embeddings.shape[0]) - 1, 1)
        batch_cov = (centered.t() @ centered) / denominator

        if not bool(self.stats_initialized.item()):
            self.running_raw_score_mean.copy_(batch_raw_mean)
            self.running_raw_score_std.copy_(batch_raw_std)
            self.running_threshold.copy_(batch_threshold)
            self.embedding_mean.copy_(batch_mean_vector)
            self.embedding_cov.copy_(batch_cov)
            self.stats_initialized.fill_(True)
            return

        momentum = self.stats_momentum
        self.running_raw_score_mean.mul_(1.0 - momentum).add_(momentum * batch_raw_mean)
        self.running_raw_score_std.mul_(1.0 - momentum).add_(momentum * batch_raw_std)
        self.running_threshold.mul_(1.0 - momentum).add_(momentum * batch_threshold)
        self.embedding_mean.mul_(1.0 - momentum).add_(momentum * batch_mean_vector)
        self.embedding_cov.mul_(1.0 - momentum).add_(momentum * batch_cov)

    def compactness_loss(self, features: torch.Tensor) -> torch.Tensor:
        return self.boundary_loss(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(features)
        return self.calibrated_score_from_embeddings(embeddings)


class OneClassCosineHeadConvProjector(BaseHead, _ConvProjectorMixin, _OneClassCosineMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        projector_channels: int = 8,
        scale: float = 15.0,
        distance_weight: float = 1.0,
        threshold_std_multiplier: float = 1.0,
        stats_momentum: float = 0.1,
        boundary_radius: float = 0.75,
        eps: float = 1e-6,
        projector_bias: bool = False,
        use_batch_norm: bool = False,
        voting_calibration_mode: str = "mahalanobis",
        cov_shrinkage: float = 0.1,
        mahalanobis_confidence: float = 0.95,
        selected_classes: tuple[str, ...] = ("belt", "magnetic", "vibration"),
    ):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("OneClassCosineHeadConvProjector expects num_classes=1.")
        self.expects_feature_map = True
        self.projector = self._build_conv_projector(in_channels, projector_channels, projector_bias, use_batch_norm)
        self._init_one_class_state(
            embedding_dim=2 * int(projector_channels),
            scale=scale,
            distance_weight=distance_weight,
            threshold_std_multiplier=threshold_std_multiplier,
            stats_momentum=stats_momentum,
            boundary_radius=boundary_radius,
            eps=eps,
            voting_calibration_mode=voting_calibration_mode,
            cov_shrinkage=cov_shrinkage,
            mahalanobis_confidence=mahalanobis_confidence,
        )

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projector(features)
        return self._pool_feature_map(projected)


class OneClassCosineHeadLinearProjector(BaseHead, _OneClassCosineMixin):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int = 1,
        projector_dim: int | None = None,
        scale: float = 15.0,
        distance_weight: float = 1.0,
        threshold_std_multiplier: float = 1.0,
        stats_momentum: float = 0.1,
        boundary_radius: float = 0.75,
        eps: float = 1e-6,
        projector_bias: bool = False,
        voting_calibration_mode: str = "mahalanobis",
        cov_shrinkage: float = 0.1,
        mahalanobis_confidence: float = 0.95,
        selected_classes: tuple[str, ...] = ("belt", "magnetic", "vibration"),
    ):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("OneClassCosineHeadLinearProjector expects num_classes=1.")
        self.expects_feature_map = False
        embedding_dim = int(projector_dim or feat_dim)
        self.projector = nn.Linear(int(feat_dim), embedding_dim, bias=bool(projector_bias))
        self._init_one_class_state(
            embedding_dim=embedding_dim,
            scale=scale,
            distance_weight=distance_weight,
            threshold_std_multiplier=threshold_std_multiplier,
            stats_momentum=stats_momentum,
            boundary_radius=boundary_radius,
            eps=eps,
            voting_calibration_mode=voting_calibration_mode,
            cov_shrinkage=cov_shrinkage,
            mahalanobis_confidence=mahalanobis_confidence,
        )

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)