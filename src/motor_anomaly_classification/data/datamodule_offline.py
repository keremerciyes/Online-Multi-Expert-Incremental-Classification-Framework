from dataclasses import dataclass

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

from motor_anomaly_classification.data.datasets import LabeledChunkDataset
from motor_anomaly_classification.data.indexing import DatasetIndex, flatten_index, load_dataset_index, merge_split
from motor_anomaly_classification.data.preprocessing import compute_standardization_stats, standardize
from motor_anomaly_classification.data.sampling import build_group_stratified_folds, split_online_train_val_from_train, split_train_test_balanced


@dataclass
class OfflineDataConfig:
    dataset_dir: str
    channel_indices: tuple[int, ...] = (0, 1)
    selected_classes: tuple[str, ...] | None = None
    seed: int = 42
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    n_folds: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0
    persistent_workers: bool = False


class OfflineTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config: OfflineDataConfig, use_cross_val: bool = False, fold_index: int = 0):
        super().__init__()
        self.config = config
        self.use_cross_val = use_cross_val
        self.fold_index = fold_index
        self.rng = np.random.RandomState(config.seed)
        self.index: DatasetIndex | None = None
        self.class_names: list[str] = []
        self.train_dataset: LabeledChunkDataset | None = None
        self.val_dataset: LabeledChunkDataset | None = None
        self.test_dataset: LabeledChunkDataset | None = None
        self.test_speeds: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def prepare_data(self) -> None:
        load_dataset_index(self.config.dataset_dir, self.config.channel_indices, self.config.selected_classes)

    def setup(self, stage: str | None = None) -> None:
        self.index = load_dataset_index(self.config.dataset_dir, self.config.channel_indices, self.config.selected_classes)
        self.class_names = self.index.class_names
        split_data, split_speed = split_train_test_balanced(
            self.rng,
            self.index.class_names,
            self.index.data_by_class,
            self.index.speed_by_class,
            self.config.test_ratio,
        )
        test_data, test_labels, test_speeds = merge_split(
            self.index.class_names,
            self.index.class_to_idx,
            split_data["test"],
            split_speed["test"],
            preserve_class_speed_order=False,
        )
        self.test_speeds = test_speeds

        if self.use_cross_val:
            train_index = DatasetIndex(
                data_by_class=split_data["train"],
                speed_by_class=split_speed["train"],
                class_names=self.index.class_names,
                class_to_idx=self.index.class_to_idx,
                speed_to_idx=self.index.speed_to_idx,
            )
            train_data, train_labels, _, train_groups = flatten_index(train_index)
            folds = build_group_stratified_folds(self.rng, train_labels, train_groups, self.config.n_folds)
            train_indices, val_indices = folds[self.fold_index]
            raw_train = train_data[train_indices]
            raw_val = train_data[val_indices]
            full_train_labels = train_labels
            train_labels = full_train_labels[train_indices]
            val_labels = full_train_labels[val_indices]
        else:
            online_split_data, online_split_speed = split_online_train_val_from_train(
                self.rng,
                self.index.class_names,
                split_data["train"],
                split_speed["train"],
                self.config.val_ratio,
            )
            raw_train, train_labels, _ = merge_split(
                self.index.class_names,
                self.index.class_to_idx,
                online_split_data["train"],
                online_split_speed["train"],
                preserve_class_speed_order=False,
            )
            raw_val, val_labels, _ = merge_split(
                self.index.class_names,
                self.index.class_to_idx,
                online_split_data["val"],
                online_split_speed["val"],
                preserve_class_speed_order=False,
            )

        self.mean, self.std = compute_standardization_stats(raw_train)
        train_data = standardize(raw_train, self.mean, self.std)
        val_data = standardize(raw_val, self.mean, self.std)
        standardized_test = standardize(test_data, self.mean, self.std)
        self.train_dataset = LabeledChunkDataset(train_data, train_labels)
        self.val_dataset = LabeledChunkDataset(val_data, val_labels)
        self.test_dataset = LabeledChunkDataset(standardized_test, test_labels)

    def train_dataloader(self) -> DataLoader:
        num_workers = int(self.config.num_workers)
        persistent_workers = bool(self.config.persistent_workers and num_workers > 0)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
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

