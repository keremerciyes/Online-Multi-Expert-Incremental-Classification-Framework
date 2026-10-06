from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import BatchSampler, DataLoader

from motor_anomaly_classification.data.datasets import LabeledChunkDataset, PhaseLabeledChunkDataset
from motor_anomaly_classification.data.indexing import DatasetIndex
from motor_anomaly_classification.data.preprocessing import compute_standardization_stats, select_channels, standardize
from motor_anomaly_classification.utils.speed_utils import parse_speed_label, speed_sort_key


@dataclass
class MultiHeadSequentialDataConfig:
    class_paths: dict[str, str] = field(default_factory=dict)
    selected_classes: tuple[str, ...] | None = None
    selected_speeds: tuple[str, ...] | None = ("1000", "1200", "1400", "1600")
    channel_indices: tuple[int, ...] = (0, 1)
    feature_bundle_path: str | None = None
    normalization: str = "train"
    seed: int = 42
    train_ratio: float = 0.8
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0
    persistent_workers: bool = False


class MultiHeadSequentialDataModule(pl.LightningDataModule):
    def __init__(self, config: MultiHeadSequentialDataConfig):
        super().__init__()
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.index: DatasetIndex | None = None
        self.class_names: list[str] = []
        self.train_dataset: PhaseLabeledChunkDataset | None = None
        self.val_dataset: LabeledChunkDataset | None = None
        self.test_dataset: LabeledChunkDataset | None = None
        self.val_speeds: np.ndarray | None = None
        self.test_speeds: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.train_phase_ranges: list[tuple[int, int]] = []

    def prepare_data(self) -> None:
        for class_path in self.config.class_paths.values():
            self._load_class_samples(class_path)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return
        if not self.config.class_paths:
            raise ValueError("MultiHeadSequentialDataConfig.class_paths cannot be empty")

        if self.config.selected_classes is None:
            class_names = list(self.config.class_paths.keys())
        else:
            class_names = [class_name for class_name in self.config.selected_classes if class_name in self.config.class_paths]
            missing = [class_name for class_name in self.config.selected_classes if class_name not in self.config.class_paths]
            if missing:
                raise ValueError(f"Selected classes are missing from class_paths: {missing}")
        if not class_names:
            raise ValueError("MultiHeadSequentialDataModule requires at least one selected class.")
        class_to_idx = {name: index for index, name in enumerate(class_names)}

        data_by_class: dict[str, np.ndarray] = {}
        speed_by_class: dict[str, np.ndarray] = {}
        speed_values: list[str] = []

        for class_name in class_names:
            class_path = self.config.class_paths[class_name]
            class_data, class_speeds = self._load_class_samples(class_path)
            data_by_class[class_name] = class_data
            speed_by_class[class_name] = class_speeds
            speed_values.extend(class_speeds.tolist())

        speed_names = sorted(set(speed_values), key=speed_sort_key)
        speed_to_idx = {name: index for index, name in enumerate(speed_names)}
        self.index = DatasetIndex(
            data_by_class=data_by_class,
            speed_by_class=speed_by_class,
            class_names=class_names,
            class_to_idx=class_to_idx,
            speed_to_idx=speed_to_idx,
        )
        self.class_names = class_names

        train_blocks: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        train_heads: list[np.ndarray] = []
        train_speeds: list[np.ndarray] = []
        val_blocks: list[np.ndarray] = []
        val_labels: list[np.ndarray] = []
        val_speeds: list[np.ndarray] = []

        for class_name in class_names:
            class_data = data_by_class[class_name]
            class_speed = speed_by_class[class_name]
            class_label = class_to_idx[class_name]
            train_idx, val_idx = self._split_indices_balanced_by_speed(class_speed)

            train_blocks.append(class_data[train_idx])
            train_labels.append(np.full((train_idx.shape[0],), class_label, dtype=np.int64))
            train_heads.append(np.full((train_idx.shape[0],), class_label, dtype=np.int64))
            train_speeds.append(class_speed[train_idx])

            val_blocks.append(class_data[val_idx])
            val_labels.append(np.full((val_idx.shape[0],), class_label, dtype=np.int64))
            val_speeds.append(class_speed[val_idx])

        self.train_phase_ranges = []
        running_offset = 0
        for block in train_blocks:
            next_offset = running_offset + int(block.shape[0])
            self.train_phase_ranges.append((running_offset, next_offset))
            running_offset = next_offset

        raw_train = np.concatenate(train_blocks, axis=0)
        raw_val = np.concatenate(val_blocks, axis=0)
        merged_train_labels = np.concatenate(train_labels, axis=0)
        merged_train_heads = np.concatenate(train_heads, axis=0)
        merged_val_labels = np.concatenate(val_labels, axis=0)
        merged_train_speeds = np.concatenate(train_speeds, axis=0) if train_speeds else np.array([], dtype=object)
        merged_val_speeds = np.concatenate(val_speeds, axis=0) if val_speeds else np.array([], dtype=object)

        if self.config.normalization == "bundle":
            from motor_anomaly_classification.models.feature_extractors import load_bundle

            if not self.config.feature_bundle_path:
                raise ValueError("Bundle normalization requires feature_bundle_path.")
            bundle = load_bundle(self.config.feature_bundle_path)
            if tuple(bundle.get("channel_indices", ())) != tuple(self.config.channel_indices):
                raise ValueError("Selected channels do not match the feature bundle channel order.")
            normalization = bundle.get("input_normalization")
            if not normalization or "mean" not in normalization or "std" not in normalization:
                raise ValueError("Feature bundle has no saved input normalization statistics.")
            mean = np.asarray(normalization["mean"], dtype=np.float32)
            std = np.asarray(normalization["std"], dtype=np.float32)
            expected_shape = (len(self.config.channel_indices),)
            if mean.shape != expected_shape or std.shape != expected_shape:
                raise ValueError("Feature bundle normalization dimensions do not match the selected channels.")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
                raise ValueError("Feature bundle normalization must be finite with positive standard deviations.")
            self.mean, self.std = mean.reshape(1, 1, -1), std.reshape(1, 1, -1)
        elif self.config.normalization == "train":
            self.mean, self.std = compute_standardization_stats(raw_train)
        else:
            raise ValueError(f"Unknown normalization mode: {self.config.normalization}")
        train_data = standardize(raw_train, self.mean, self.std)
        val_data = standardize(raw_val, self.mean, self.std)

        if self.config.feature_bundle_path:
            train_data, val_data = self._maybe_precompute_features(
                train_data, merged_train_speeds, val_data, merged_val_speeds
            )

        self.train_dataset = PhaseLabeledChunkDataset(train_data, merged_train_labels, merged_train_heads)
        self.val_dataset = LabeledChunkDataset(val_data, merged_val_labels)
        # TODO: Add an independent held-out test split if needed; current workflow validates on the merged validation pool.
        self.test_dataset = LabeledChunkDataset(val_data.copy(), merged_val_labels.copy())
        self.val_speeds = merged_val_speeds
        self.test_speeds = merged_val_speeds

    def _split_indices_balanced_by_speed(self, class_speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if class_speeds.size <= 1:
            return np.arange(class_speeds.size, dtype=np.int64), np.array([], dtype=np.int64)

        train_indices: list[np.ndarray] = []
        val_indices: list[np.ndarray] = []
        for speed_name in sorted(set(class_speeds.tolist()), key=speed_sort_key):
            speed_indices = np.where(class_speeds == speed_name)[0]
            shuffled = speed_indices.copy()
            self.rng.shuffle(shuffled)

            speed_train_count = int(np.floor(speed_indices.size * self.config.train_ratio))
            if speed_indices.size >= 2:
                speed_train_count = max(1, speed_train_count)
            if speed_train_count >= speed_indices.size:
                speed_train_count = max(0, speed_indices.size - 1)

            selected_train = np.sort(shuffled[:speed_train_count])
            selected_val = np.sort(shuffled[speed_train_count:])
            train_indices.append(selected_train)
            val_indices.append(selected_val)

        merged_train_idx = np.sort(np.concatenate(train_indices)) if train_indices else np.array([], dtype=np.int64)
        merged_val_idx = np.sort(np.concatenate(val_indices)) if val_indices else np.array([], dtype=np.int64)
        return merged_train_idx, merged_val_idx

    def _load_class_samples(self, class_path: str) -> tuple[np.ndarray, np.ndarray]:
        root = Path(class_path)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Class path does not exist or is not a directory: {class_path}")

        data_blocks: list[np.ndarray] = []
        speed_blocks: list[np.ndarray] = []
        selected_speeds = None
        if self.config.selected_speeds is not None:
            selected_speeds = {str(speed) for speed in self.config.selected_speeds}
        for speed_dir in sorted(root.iterdir(), key=lambda item: speed_sort_key(item.name)):
            if not speed_dir.is_dir():
                continue
            speed_label = parse_speed_label(speed_dir.name)
            if selected_speeds is not None and str(speed_label) not in selected_speeds:
                continue
            for file_path in sorted(speed_dir.iterdir()):
                if file_path.suffix.lower() != ".npy":
                    continue
                loaded = np.load(file_path)
                selected = select_channels(loaded, self.config.channel_indices)
                data_blocks.append(selected)
                speed_blocks.append(np.full((selected.shape[0],), speed_label, dtype=object))

        if not data_blocks:
            raise ValueError(f"No .npy windows found under class path: {class_path}")

        return np.concatenate(data_blocks, axis=0), np.concatenate(speed_blocks, axis=0)

    def _maybe_precompute_features(
        self,
        train_data: np.ndarray,
        train_speeds: np.ndarray,
        val_data: np.ndarray,
        val_speeds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute standardized feature vectors for handcrafted / hybrid bundles.

        AE-only bundles (``has_handcrafted`` false) run live inside the backbone, so
        the standardized raw windows are returned unchanged. Handcrafted / hybrid
        bundles produce an ``(N, 1, D)`` tensor whose ``__getitem__`` permute yields
        ``(D, 1)`` and batches to ``(B, D, 1)`` for ``PrecomputedFeatureBackbone``.
        """
        from motor_anomaly_classification.models.feature_extractors import (
            compute_bundle_features,
            load_bundle,
            speed_label_to_rpm,
        )

        bundle = load_bundle(self.config.feature_bundle_path)
        if not bool(bundle.get("has_handcrafted")):
            return train_data, val_data

        def _rpm(speeds: np.ndarray) -> np.ndarray:
            return np.array([speed_label_to_rpm(speed) for speed in speeds.tolist()], dtype=np.float32)

        train_features = compute_bundle_features(bundle, train_data, _rpm(train_speeds))
        val_features = compute_bundle_features(bundle, val_data, _rpm(val_speeds))
        return train_features[:, None, :], val_features[:, None, :]

    def train_dataloader(self) -> DataLoader:
        num_workers = int(self.config.num_workers)
        persistent_workers = bool(self.config.persistent_workers and num_workers > 0)
        return DataLoader(
            self.train_dataset,
            batch_sampler=_SequentialPhaseBatchSampler(self.train_phase_ranges, self.config.train_batch_size),
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        num_workers = int(self.config.num_workers)
        persistent_workers = bool(self.config.persistent_workers and num_workers > 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        num_workers = int(self.config.num_workers)
        persistent_workers = bool(self.config.persistent_workers and num_workers > 0)
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )


class _SequentialPhaseBatchSampler(BatchSampler):
    def __init__(self, phase_ranges: list[tuple[int, int]], batch_size: int):
        self.phase_ranges = phase_ranges
        self.batch_size = int(batch_size)

    def __iter__(self):
        for start, end in self.phase_ranges:
            cursor = int(start)
            while cursor < int(end):
                upper = min(cursor + self.batch_size, int(end))
                yield list(range(cursor, upper))
                cursor = upper

    def __len__(self) -> int:
        total = 0
        for start, end in self.phase_ranges:
            span = max(0, int(end) - int(start))
            total += int(np.ceil(span / self.batch_size)) if span else 0
        return total
