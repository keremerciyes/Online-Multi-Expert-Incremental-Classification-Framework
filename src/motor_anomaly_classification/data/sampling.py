import numpy as np

from motor_anomaly_classification.utils.speed_utils import speed_sort_key


def split_train_test_balanced(
    rng: np.random.RandomState,
    class_names: list[str],
    data_by_class: dict[str, np.ndarray],
    speed_by_class: dict[str, np.ndarray],
    test_ratio: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    data_splits = {"train": {}, "test": {}}
    speed_splits = {"train": {}, "test": {}}

    for class_name in class_names:
        class_data = data_by_class[class_name]
        class_speeds = speed_by_class[class_name]
        train_indices: list[np.ndarray] = []
        test_indices: list[np.ndarray] = []

        for speed_name in sorted(set(class_speeds.tolist()), key=speed_sort_key):
            group_indices = np.where(class_speeds == speed_name)[0]
            shuffled = group_indices.copy()
            rng.shuffle(shuffled)

            test_count = int(np.floor(group_indices.size * test_ratio))
            if group_indices.size >= 2:
                test_count = max(1, test_count)
            if test_count >= group_indices.size:
                test_count = max(0, group_indices.size - 1)

            group_test = np.sort(shuffled[:test_count])
            group_train = np.sort(shuffled[test_count:])
            test_indices.append(group_test)
            train_indices.append(group_train)

        merged_train_idx = np.sort(np.concatenate(train_indices)) if train_indices else np.array([], dtype=np.int64)
        merged_test_idx = np.sort(np.concatenate(test_indices)) if test_indices else np.array([], dtype=np.int64)

        data_splits["train"][class_name] = class_data[merged_train_idx]
        data_splits["test"][class_name] = class_data[merged_test_idx]
        speed_splits["train"][class_name] = class_speeds[merged_train_idx]
        speed_splits["test"][class_name] = class_speeds[merged_test_idx]

    return data_splits, speed_splits


def split_online_train_val_from_train(
    rng: np.random.RandomState,
    class_names: list[str],
    train_data_by_class: dict[str, np.ndarray],
    train_speed_by_class: dict[str, np.ndarray],
    val_ratio: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    data_splits = {"train": {}, "val": {}}
    speed_splits = {"train": {}, "val": {}}

    for class_name in class_names:
        class_data = train_data_by_class[class_name]
        class_speeds = train_speed_by_class[class_name]
        keep_indices: list[np.ndarray] = []
        val_indices: list[np.ndarray] = []

        for speed_name in sorted(set(class_speeds.tolist()), key=speed_sort_key):
            speed_indices = np.where(class_speeds == speed_name)[0]
            shuffled = speed_indices.copy()
            rng.shuffle(shuffled)
            val_count = int(np.floor(speed_indices.size * val_ratio))
            if speed_indices.size >= 3:
                val_count = max(1, val_count)
            if val_count >= speed_indices.size:
                val_count = max(0, speed_indices.size - 1)

            selected_val = np.sort(shuffled[:val_count])
            selected_train = np.sort(shuffled[val_count:])
            val_indices.append(selected_val)
            keep_indices.append(selected_train)

        train_idx = np.sort(np.concatenate(keep_indices)) if keep_indices else np.array([], dtype=np.int64)
        val_idx = np.sort(np.concatenate(val_indices)) if val_indices else np.array([], dtype=np.int64)

        data_splits["train"][class_name] = class_data[train_idx]
        data_splits["val"][class_name] = class_data[val_idx]
        speed_splits["train"][class_name] = class_speeds[train_idx]
        speed_splits["val"][class_name] = class_speeds[val_idx]

    return data_splits, speed_splits


def build_group_stratified_folds(
    rng: np.random.RandomState,
    labels: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_val_indices: list[list[int]] = [[] for _ in range(n_folds)]
    unique_groups = np.unique(groups)

    for group_id in unique_groups:
        group_indices = np.where(groups == group_id)[0]
        shuffled = group_indices.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, n_folds)
        for fold_index in range(n_folds):
            fold_val_indices[fold_index].extend(chunks[fold_index].tolist())

    all_indices = np.arange(labels.shape[0])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_folds):
        val_indices = np.array(sorted(fold_val_indices[fold_index]), dtype=np.int64)
        is_val = np.zeros(labels.shape[0], dtype=bool)
        is_val[val_indices] = True
        train_indices = all_indices[~is_val]
        folds.append((train_indices, val_indices))
    return folds
