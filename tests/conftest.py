from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    class_names = ("healthy", "fault")
    speed_names = ("400", "800")
    num_windows = 6
    window_length = 64
    num_channels = 3

    rng = np.random.default_rng(42)
    for class_index, class_name in enumerate(class_names):
        for speed_index, speed_name in enumerate(speed_names):
            speed_dir = dataset_dir / class_name / speed_name
            speed_dir.mkdir(parents=True, exist_ok=True)
            data = rng.normal(loc=class_index + speed_index * 0.1, scale=0.2, size=(num_windows, window_length, num_channels)).astype(np.float32)
            np.save(speed_dir / "windows.npy", data)

    return dataset_dir
