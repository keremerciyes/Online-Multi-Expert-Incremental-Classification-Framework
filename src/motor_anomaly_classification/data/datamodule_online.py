from dataclasses import dataclass

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

from motor_anomaly_classification.data.datasets import LabeledChunkDataset
from motor_anomaly_classification.data.indexing import DatasetIndex, load_dataset_index, merge_split
from motor_anomaly_classification.data.preprocessing import compute_standardization_stats, standardize
from motor_anomaly_classification.data.sampling import split_online_train_val_from_train, split_train_test_balanced


@dataclass
class OnlineDataConfig:
    dataset_dir: str
    channel_indices: tuple[int, ...] = (0, 1)
    selected_classes: tuple[str, ...] | None = None
    seed: int = 42
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0
    persistent_workers: bool = False


class OnlineTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config: OnlineDataConfig):
        super().__init__()
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.index: DatasetIndex | None = None
        self.class_names: list[str] = []
        self.train_dataset: LabeledChunkDataset | None = None
        self.val_dataset: LabeledChunkDataset | None = None
        self.test_dataset: LabeledChunkDataset | None = None
        self.train_speeds: np.ndarray | None = None
        self.val_speeds: np.ndarray | None = None
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
        online_split_data, online_split_speed = split_online_train_val_from_train(
            self.rng,
            self.index.class_names,
            split_data["train"],
            split_speed["train"],
            self.config.val_ratio,
        )

        raw_train, train_labels, self.train_speeds = merge_split(
            self.index.class_names,
            self.index.class_to_idx,
            online_split_data["train"],
            online_split_speed["train"],
            preserve_class_speed_order=True,
        )
        raw_val, val_labels, self.val_speeds = merge_split(
            self.index.class_names,
            self.index.class_to_idx,
            online_split_data["val"],
            online_split_speed["val"],
            preserve_class_speed_order=True,
        )
        raw_test, test_labels, self.test_speeds = merge_split(
            self.index.class_names,
            self.index.class_to_idx,
            split_data["test"],
            split_speed["test"],
            preserve_class_speed_order=False,
        )

        self.mean, self.std = compute_standardization_stats(raw_train)
        train_data = standardize(raw_train, self.mean, self.std)
        val_data = standardize(raw_val, self.mean, self.std)
        test_data = standardize(raw_test, self.mean, self.std)

        self.train_dataset = LabeledChunkDataset(train_data, train_labels)
        self.val_dataset = LabeledChunkDataset(val_data, val_labels)
        self.test_dataset = LabeledChunkDataset(test_data, test_labels)

    def train_dataloader(self) -> DataLoader:
        num_workers = int(self.config.num_workers)
        persistent_workers = bool(self.config.persistent_workers and num_workers > 0)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
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
