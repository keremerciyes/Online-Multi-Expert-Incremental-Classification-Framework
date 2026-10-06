from dataclasses import dataclass, field
from typing import Any
from itertools import chain

import torch
from hydra.utils import instantiate

from motor_anomaly_classification.models.encoder_presets import DEFAULT_ENCODER_KERNEL_SIZES
from motor_anomaly_classification.models.etf_multihead import EtfMultiHeadNet
from motor_anomaly_classification.models.oneclass_multihead import OneClassMultiHeadNet
from motor_anomaly_classification.models.timeseries_net import TimeSeriesMapHeadNet, TimeSeriesNet


FEATURE_MAP_HEAD_NAMES = {
    "MultiClassCosineHeadConvProjector",
    "MultiClassDeepCosineHeadConvProjector",
    "OneClassCosineHeadConvProjector",
}


def _uses_feature_map_head(head_target: str) -> bool:
    return any(head_name in head_target for head_name in FEATURE_MAP_HEAD_NAMES)


@dataclass
class OptimizerConfig:
    optimizer_name: str = "sgd"
    learning_rate_head: float = 1e-3
    backbone_lr_divisor: float = 5.0
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class ModelFactoryConfig:
    mode: str = "timeseries_net"
    in_channels: int = 2
    num_classes: int = 3
    selected_classes: tuple[str, ...] = ("belt", "magnetic", "vibration")
    backbone: dict[str, Any] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)
    encoder: dict[str, Any] = field(default_factory=dict)


def build_model(model_config: ModelFactoryConfig) -> torch.nn.Module:
    backbone_config = dict(model_config.backbone)
    backbone_config.setdefault("in_channels", model_config.in_channels)
    backbone = instantiate(backbone_config)
    backbone_output_dim = int(getattr(backbone, "output_dim"))

    if model_config.mode == "one_class_multi_head":
        head_template = dict(model_config.head)
        head_target = str(head_template.get("_target_", ""))
        if _uses_feature_map_head(head_target):
            head_template.setdefault("in_channels", backbone_output_dim)
        else:
            head_template.setdefault("feat_dim", int(getattr(backbone, "pooled_output_dim", 2 * backbone_output_dim)))
        head_template.setdefault("num_classes", 1)
        head_count = model_config.num_classes
        heads: dict[str, torch.nn.Module] = {}
        for class_index in range(head_count):
            heads[str(class_index)] = instantiate(dict(head_template))
        return OneClassMultiHeadNet(backbone=backbone, heads=heads)

    if model_config.mode == "etf_multi_head":
        etf_config = dict(model_config.head)
        return EtfMultiHeadNet(
            backbone=backbone,
            num_classes=model_config.num_classes,
            projector_arch=str(etf_config.get("projector_arch", "cosine")),
            embedding_dim=int(etf_config.get("embedding_dim", 16)),
            projector_channels=int(etf_config.get("projector_channels", 8)),
            scale=float(etf_config.get("scale", 15.0)),
            projector_bias=bool(etf_config.get("projector_bias", False)),
        )

    head_config = dict(model_config.head)
    head_target = str(head_config.get("_target_", ""))
    if model_config.mode == "timeseries_map_head" or (model_config.mode == "timeseries_net" and _uses_feature_map_head(head_target)):
        head_config.setdefault("in_channels", backbone_output_dim)
        if "selected_classes" in head_config and head_config.get("num_classes") is None:
            head_config["num_classes"] = len(head_config["selected_classes"])
        else:
            head_config.setdefault("num_classes", model_config.num_classes)
        head = instantiate(head_config)
        return TimeSeriesMapHeadNet(backbone=backbone, head=head)

    head_config.setdefault("feat_dim", 2 * backbone_output_dim)
    if "selected_classes" in head_config and head_config.get("num_classes") is None:
        head_config["num_classes"] = len(head_config["selected_classes"])
    else:
        head_config.setdefault("num_classes", model_config.num_classes)
    head = instantiate(head_config)
    return TimeSeriesNet(backbone=backbone, head=head)


def build_optimizer(model: torch.nn.Module, optimizer_config: OptimizerConfig) -> torch.optim.Optimizer:
    head_learning_rate = optimizer_config.learning_rate_head
    backbone_learning_rate = head_learning_rate / optimizer_config.backbone_lr_divisor

    if hasattr(model, "backbone"):
        backbone_params = [p for p in getattr(model, "backbone").parameters() if p.requires_grad]
    elif hasattr(model, "encoder"):
        # Reconstruction model has no backbone; encoder takes the lower LR slot.
        backbone_params = [p for p in getattr(model, "encoder").parameters() if p.requires_grad]
    else:
        backbone_params = []

    if hasattr(model, "head"):
        head_parameters = getattr(model, "head").parameters()
    elif hasattr(model, "heads"):
        head_parameters = chain.from_iterable(head.parameters() for head in getattr(model, "heads").values())
    elif hasattr(model, "encoder") and hasattr(model, "decoders"):
        head_parameters = chain.from_iterable(decoder.parameters() for decoder in getattr(model, "decoders").values())
    else:
        raise AttributeError("Model must expose one of: head, heads, or encoder+decoders for optimizer construction.")

    parameter_groups: list[dict[str, Any]] = []
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": backbone_learning_rate})
    parameter_groups.append({"params": head_parameters, "lr": head_learning_rate})
    optimizer_name = str(optimizer_config.optimizer_name).lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            momentum=float(optimizer_config.momentum),
            weight_decay=float(optimizer_config.weight_decay),
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameter_groups,
            weight_decay=float(optimizer_config.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer_name={optimizer_config.optimizer_name!r}. Expected one of: 'sgd', 'adam'.")
