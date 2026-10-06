from typing import Any

import numpy as np
import torch


def load_checkpoint_compatible(checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except Exception as error:
        error_text = str(error)
        if "Weights only load failed" in error_text or "Unsupported global" in error_text:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        raise


def checkpoint_array_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
