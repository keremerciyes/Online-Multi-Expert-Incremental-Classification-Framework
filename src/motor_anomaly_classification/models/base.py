from typing import Protocol

import torch
import torch.nn as nn

# Redundant with BaseClassificationModule, but keeping for now to avoid breaking changes in the training loop
class BackboneProtocol(Protocol):
    output_dim: int

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


class BaseTimeSeriesClassifier(nn.Module):
    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# Redundant with BaseClassificationModule, but keeping for now to avoid breaking changes in the training loop
class BaseBackbone(nn.Module):
    output_dim: int

# Redundant with BaseClassificationModule, but keeping for now to avoid breaking changes in the training loop
class BaseHead(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
