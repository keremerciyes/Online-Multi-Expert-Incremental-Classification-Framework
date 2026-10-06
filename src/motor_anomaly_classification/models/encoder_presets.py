from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_ENCODER_KERNEL_SIZES: tuple[int, int, int] = (7, 5, 5)
PRETRAINED_ENCODER_FILENAMES: dict[tuple[int, int, int], str] = {
    (32, 16, 8): "best_32-16-8.pt",
    (16, 32, 8): "best_16-32-8.pt",
}


def normalize_encoder_tuple(values: tuple[int, ...] | list[int], expected_length: int = 3) -> tuple[int, int, int]:
    normalized = tuple(int(value) for value in values)
    if len(normalized) != expected_length:
        raise ValueError(f"Expected {expected_length} encoder values, got {len(normalized)}.")
    return (normalized[0], normalized[1], normalized[2])


def project_outputs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "outputs"


def resolve_default_pretrained_encoder_path(channels: tuple[int, ...] | list[int]) -> Path | None:
    channel_key = normalize_encoder_tuple(channels)
    filename = PRETRAINED_ENCODER_FILENAMES.get(channel_key)
    if filename is None:
        return None
    candidate = project_outputs_dir() / filename
    return candidate if candidate.exists() else None


def build_autoencoder_encoder(
    in_channels: int,
    channels: tuple[int, ...] | list[int],
    kernel_sizes: tuple[int, ...] | list[int] = DEFAULT_ENCODER_KERNEL_SIZES,
) -> nn.Sequential:
    normalized_channels = normalize_encoder_tuple(channels)
    normalized_kernels = normalize_encoder_tuple(kernel_sizes)
    layers: list[nn.Module] = []
    previous_channels = int(in_channels)
    for hidden_channels, kernel_size in zip(normalized_channels, normalized_kernels):
        layers.append(
            nn.Conv1d(previous_channels, hidden_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        )
        layers.append(nn.ReLU())
        previous_channels = hidden_channels
    return nn.Sequential(*layers)


def build_autoencoder_decoder(
    out_channels: int,
    channels: tuple[int, ...] | list[int],
    kernel_sizes: tuple[int, ...] | list[int] = DEFAULT_ENCODER_KERNEL_SIZES,
) -> nn.Sequential:
    normalized_channels = normalize_encoder_tuple(channels)
    normalized_kernels = normalize_encoder_tuple(kernel_sizes)
    decoder_channels = (normalized_channels[-1], normalized_channels[-2], normalized_channels[-3], int(out_channels))
    decoder_kernels = tuple(reversed(normalized_kernels))
    layers: list[nn.Module] = []
    for index, (in_channels, next_channels, kernel_size) in enumerate(
        zip(decoder_channels[:-1], decoder_channels[1:], decoder_kernels)
    ):
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(nn.Conv1d(in_channels, next_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
        if index < len(decoder_channels) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _extract_tensor_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
    if isinstance(raw_state, dict):
        state_dict = raw_state.get("state_dict", raw_state)
    else:
        state_dict = raw_state
    if not isinstance(state_dict, dict):
        raise ValueError("Expected checkpoint to contain a state_dict dictionary.")

    tensor_state = {key: value for key, value in state_dict.items() if isinstance(value, torch.Tensor)}
    for prefix in ("encoder.", "model.encoder.", "backbone.", "model.backbone."):
        extracted = {key.removeprefix(prefix): value for key, value in tensor_state.items() if key.startswith(prefix)}
        if extracted:
            return extracted
    return tensor_state


def load_encoder_weights(module: nn.Module, checkpoint_path: str | Path) -> None:
    raw_state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = _extract_tensor_state_dict(raw_state)
    module.load_state_dict(state_dict, strict=True)