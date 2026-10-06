from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motor_anomaly_classification.data.preprocessing import select_channels
from motor_anomaly_classification.utils.speed_utils import parse_speed_label, speed_sort_key


@dataclass
class DatasetIndex:
    data_by_class: dict[str, np.ndarray]
    speed_by_class: dict[str, np.ndarray]
    class_names: list[str]
    class_to_idx: dict[str, int]
    speed_to_idx: dict[str, int]


def load_dataset_index(dataset_dir: str, channel_indices: tuple[int, ...], selected_classes: tuple[str, ...] | list[str] | None = None) -> DatasetIndex:
    root = Path(dataset_dir)
    data_by_class: dict[str, list[np.ndarray]] = {}
    speed_by_class: dict[str, list[np.ndarray]] = {}

    for class_path in sorted(root.iterdir()):
        if not class_path.is_dir():
            continue
        class_name = class_path.name
        if selected_classes is not None and class_name not in selected_classes:
            continue
        data_by_class[class_name] = []
        speed_by_class[class_name] = []
        for speed_path in sorted(class_path.iterdir(), key=lambda item: speed_sort_key(item.name)):
            if not speed_path.is_dir():
                continue
            speed_label = parse_speed_label(speed_path.name)
            for file_path in sorted(speed_path.iterdir()):
                if file_path.suffix.lower() != ".npy":
                    continue
                loaded = np.load(file_path)
                selected = select_channels(loaded, channel_indices)
                data_by_class[class_name].append(selected)
                speed_by_class[class_name].append(np.full((selected.shape[0],), speed_label, dtype=object))

    merged_data: dict[str, np.ndarray] = {}
    merged_speed: dict[str, np.ndarray] = {}
    for class_name, arrays in data_by_class.items():
        if not arrays:
            continue
        merged_data[class_name] = np.concatenate(arrays, axis=0)
        merged_speed[class_name] = np.concatenate(speed_by_class[class_name], axis=0)

    class_names = sorted(merged_data.keys())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    speed_names = sorted({str(speed) for arr in merged_speed.values() for speed in arr.tolist()}, key=speed_sort_key)
    speed_to_idx = {name: index for index, name in enumerate(speed_names)}
    return DatasetIndex(
        data_by_class=merged_data,
        speed_by_class=merged_speed,
        class_names=class_names,
        class_to_idx=class_to_idx,
        speed_to_idx=speed_to_idx,
    )


def flatten_index(index: DatasetIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_list = []
    labels_list = []
    speeds_list = []
    groups_list = []
    for class_name in index.class_names:
        class_data = index.data_by_class[class_name]
        class_speeds = index.speed_by_class[class_name]
        class_label = index.class_to_idx[class_name]
        data_list.append(class_data)
        labels_list.append(np.full((class_data.shape[0],), class_label, dtype=np.int64))
        speeds_list.append(class_speeds)
        groups = np.array(
            [index.class_to_idx[class_name] * 10000 + index.speed_to_idx[str(speed)] for speed in class_speeds],
            dtype=np.int64,
        )
        groups_list.append(groups)

    return (
        np.concatenate(data_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(speeds_list, axis=0),
        np.concatenate(groups_list, axis=0),
    )


def merge_split(
    class_names: list[str],
    class_to_idx: dict[str, int],
    split_data_by_class: dict[str, np.ndarray],
    split_speed_by_class: dict[str, np.ndarray],
    preserve_class_speed_order: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_blocks = []
    label_blocks = []
    speed_blocks = []

    for class_name in class_names:
        class_data = split_data_by_class[class_name]
        class_speeds = split_speed_by_class[class_name]

        if preserve_class_speed_order and class_speeds.size > 0:
            sorted_indices = np.array(
                sorted(np.arange(class_speeds.shape[0]).tolist(), key=lambda index: speed_sort_key(str(class_speeds[index]))),
                dtype=np.int64,
            )
            class_data = class_data[sorted_indices]
            class_speeds = class_speeds[sorted_indices]

        data_blocks.append(class_data)
        label_blocks.append(np.full((class_data.shape[0],), class_to_idx[class_name], dtype=np.int64))
        speed_blocks.append(class_speeds)

    return np.concatenate(data_blocks, axis=0), np.concatenate(label_blocks, axis=0), np.concatenate(speed_blocks, axis=0)
