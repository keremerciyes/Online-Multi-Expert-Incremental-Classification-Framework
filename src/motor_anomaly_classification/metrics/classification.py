import numpy as np

from motor_anomaly_classification.utils.speed_utils import speed_sort_key


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(np.mean(predictions == labels))


def compute_per_class_accuracy(predictions: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        result[class_name] = float(np.mean(predictions[mask] == labels[mask])) if np.any(mask) else 0.0
    return result


def compute_per_speed_accuracy(predictions: np.ndarray, labels: np.ndarray, speeds: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    unique_speeds = sorted(set(speeds.tolist()), key=speed_sort_key)
    for speed in unique_speeds:
        speed_mask = speeds == speed
        result[str(speed)] = float(np.mean(predictions[speed_mask] == labels[speed_mask])) if np.any(speed_mask) else 0.0
    return result


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix
