import numpy as np


def select_channels(data: np.ndarray, channel_indices: tuple[int, ...]) -> np.ndarray:
    max_index = max(channel_indices)
    if data.shape[2] > max_index:
        return data[:, :, channel_indices]
    if data.shape[1] > max_index:
        return np.transpose(data[:, channel_indices, :], (0, 2, 1))
    raise ValueError(f"Channel indices {channel_indices} out of bounds for shape {data.shape}")


def compute_standardization_stats(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True) + 1e-8
    return mean, std


def standardize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean) / std
