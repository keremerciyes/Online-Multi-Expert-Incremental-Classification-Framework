import torch
import torch.nn as nn

from motor_anomaly_classification.models.base import BaseTimeSeriesClassifier


class TimeSeriesNet(BaseTimeSeriesClassifier):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.backbone(inputs)
        avg_pool = feature_map.mean(dim=2)
        max_pool = feature_map.amax(dim=2)
        features = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.head(features)
        return logits, features

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(inputs)
        avg_pool = feature_map.mean(dim=2)
        max_pool = feature_map.amax(dim=2)
        return torch.cat([avg_pool, max_pool], dim=1)


class TimeSeriesMapHeadNet(BaseTimeSeriesClassifier):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.backbone(inputs)
        avg_pool = feature_map.mean(dim=2)
        max_pool = feature_map.amax(dim=2)
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.head(feature_map)
        return logits, pooled_features

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(inputs)
        avg_pool = feature_map.mean(dim=2)
        max_pool = feature_map.amax(dim=2)
        return torch.cat([avg_pool, max_pool], dim=1)
