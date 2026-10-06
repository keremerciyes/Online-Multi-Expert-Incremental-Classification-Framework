import numpy as np

from motor_anomaly_classification.data.indexing import DatasetIndex
from motor_anomaly_classification.utils.speed_utils import speed_sort_key


def show_data_before_shuffle(
    labels: np.ndarray,
    class_names: list[str],
    speeds: np.ndarray | None = None,
    title: str = "Data Order Preview",
    limit: int = 48,
) -> None:
    amount = min(limit, labels.shape[0])
    print(f"\n{title}")
    print([class_names[int(value)] for value in labels[:amount]])
    if speeds is not None:
        print([str(value) for value in speeds[:amount]])


def print_class_label_mapping(class_names: list[str], class_to_idx: dict[str, int]) -> None:
    print("\nClass label mapping from folder names:")
    for class_name in class_names:
        print(f"  folder='{class_name}' -> label={class_to_idx[class_name]}")


def print_dataset_structure_summary(index: DatasetIndex) -> None:
    print("\nDataset summary before splitting:")
    for class_name in index.class_names:
        class_data = index.data_by_class[class_name]
        class_speeds = index.speed_by_class[class_name]
        unique_speeds, counts = np.unique(class_speeds, return_counts=True)
        speed_stats = ", ".join(
            f"{speed}:{int(count)}"
            for speed, count in sorted(zip(unique_speeds.tolist(), counts.tolist()), key=lambda item: speed_sort_key(item[0]))
        )
        print(
            f"  class='{class_name}' label={index.class_to_idx[class_name]} windows={class_data.shape[0]} shape={tuple(class_data.shape[1:])} speeds=[{speed_stats}]"
        )
