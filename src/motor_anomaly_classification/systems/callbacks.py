from pathlib import Path
import json
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import torch

from motor_anomaly_classification.metrics.classification import compute_confusion_matrix, compute_per_class_accuracy, compute_per_speed_accuracy
from motor_anomaly_classification.utils.inspection import print_class_label_mapping, print_dataset_structure_summary
from motor_anomaly_classification.utils.plotting import (
    plot_batch_loss,
    plot_batch_metric,
    plot_calibration_margins,
    plot_confusion_matrix,
    plot_per_head_score_distributions,
    plot_per_speed_accuracy,
    plot_selected_windows_by_class,
)


class DatasetInspectionCallback(pl.Callback):
    def __init__(self, print_summary: bool = True, print_mapping: bool = True):
        super().__init__()
        self.print_summary = print_summary
        self.print_mapping = print_mapping

    def on_fit_start(self, trainer, pl_module) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        index = getattr(datamodule, "index", None)
        if index is None:
            return
        if self.print_mapping:
            print_class_label_mapping(index.class_names, index.class_to_idx)
        if self.print_summary:
            print_dataset_structure_summary(index)


class SelectedWindowsPlotCallback(pl.Callback):
    def __init__(self, selected_indices: dict[str, list[int]] | None = None, channel_names: tuple[str, ...] | None = None, output_dirname: str = "selected_windows"):
        super().__init__()
        self.selected_indices = selected_indices or {}
        self.channel_names = channel_names
        self.output_dirname = output_dirname

    def on_fit_start(self, trainer, pl_module) -> None:
        if not self.selected_indices:
            return
        datamodule = getattr(trainer, "datamodule", None)
        index = getattr(datamodule, "index", None)
        if index is None:
            return
        log_dir = Path(getattr(trainer, "log_dir", trainer.default_root_dir))
        output_dir = log_dir / self.output_dirname
        plot_selected_windows_by_class(
            index.data_by_class,
            index.class_to_idx,
            self.selected_indices,
            str(output_dir),
            self.channel_names,
        )


class ConfusionMatrixCallback(pl.Callback):
    def __init__(self, output_name: str = "confusion_matrix.png"):
        super().__init__()
        self.output_name = output_name
        self._predictions: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if not isinstance(outputs, dict):
            return
        prediction_tensor = outputs.get("predictions")
        label_tensor = outputs.get("labels")
        if prediction_tensor is None or label_tensor is None:
            return
        self._predictions.append(cast(torch.Tensor, prediction_tensor).cpu())
        self._labels.append(cast(torch.Tensor, label_tensor).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._predictions:
            return
        predictions = torch.cat(self._predictions).numpy()
        labels = torch.cat(self._labels).numpy()
        num_classes = int(getattr(pl_module.model_config, "num_classes", 0))
        matrix = compute_confusion_matrix(predictions, labels, num_classes)
        datamodule = getattr(trainer, "datamodule", None)
        class_names = getattr(datamodule, "class_names", [str(index) for index in range(num_classes)])
        output_dir = Path(getattr(trainer, "log_dir", trainer.default_root_dir))
        plot_confusion_matrix(matrix, class_names, str(output_dir / self.output_name))
        self._predictions.clear()
        self._labels.clear()


class ClassificationReportCallback(pl.Callback):
    def __init__(self, output_name: str = "classification_report.json"):
        super().__init__()
        self.output_name = output_name
        self._predictions: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if not isinstance(outputs, dict):
            return
        prediction_tensor = outputs.get("predictions")
        label_tensor = outputs.get("labels")
        if prediction_tensor is None or label_tensor is None:
            return
        self._predictions.append(cast(torch.Tensor, prediction_tensor).cpu())
        self._labels.append(cast(torch.Tensor, label_tensor).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._predictions:
            return
        predictions = torch.cat(self._predictions).numpy()
        labels = torch.cat(self._labels).numpy()
        datamodule = getattr(trainer, "datamodule", None)
        class_names = getattr(datamodule, "class_names", [str(index) for index in range(int(getattr(pl_module.model_config, "num_classes", 0)))])
        raw_test_speeds = np.asarray(getattr(datamodule, "test_speeds", np.array([], dtype=object)))
        test_speeds = raw_test_speeds[: len(labels)] if raw_test_speeds.size else raw_test_speeds
        prediction_histogram = {
            class_names[index] if index < len(class_names) else str(index): {
                "count": int((predictions == index).sum()),
                "ratio": float((predictions == index).mean()) if len(predictions) else 0.0,
            }
            for index in range(int(getattr(pl_module.model_config, "num_classes", 0)))
        }
        report = {
            "accuracy": float(np.mean(predictions == labels)) if labels.size else 0.0,
            "per_class_accuracy": compute_per_class_accuracy(predictions, labels, class_names),
            "prediction_histogram": prediction_histogram,
            "per_speed_accuracy": compute_per_speed_accuracy(predictions, labels, test_speeds) if len(test_speeds) == len(labels) else {},
        }
        output_path = Path(getattr(trainer, "log_dir", trainer.default_root_dir)) / self.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nTest summary:")
        print(f"  accuracy={report['accuracy']:.4f}")
        print("  prediction_histogram:")
        for class_name, stats in prediction_histogram.items():
            print(f"    {class_name}: count={stats['count']} ratio={stats['ratio']:.3f}")
        self._predictions.clear()
        self._labels.clear()


class SpeedAccuracyPlotCallback(pl.Callback):
    def __init__(self, output_name: str = "per_speed_accuracy.png"):
        super().__init__()
        self.output_name = output_name
        self._predictions: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if not isinstance(outputs, dict):
            return
        prediction_tensor = outputs.get("predictions")
        label_tensor = outputs.get("labels")
        if prediction_tensor is None or label_tensor is None:
            return
        self._predictions.append(cast(torch.Tensor, prediction_tensor).cpu())
        self._labels.append(cast(torch.Tensor, label_tensor).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._predictions:
            return
        predictions = torch.cat(self._predictions).numpy()
        labels = torch.cat(self._labels).numpy()
        datamodule = getattr(trainer, "datamodule", None)
        raw_test_speeds = np.asarray(getattr(datamodule, "test_speeds", np.array([], dtype=object)))
        test_speeds = raw_test_speeds[: len(labels)] if raw_test_speeds.size else raw_test_speeds
        if len(test_speeds) != len(labels):
            return
        per_speed = compute_per_speed_accuracy(predictions, labels, test_speeds)
        output_dir = Path(getattr(trainer, "log_dir", trainer.default_root_dir))
        plot_per_speed_accuracy(per_speed, str(output_dir / self.output_name))
        self._predictions.clear()
        self._labels.clear()


class PerHeadScoreDiagnosticsCallback(pl.Callback):
    def __init__(
        self,
        output_name: str = "per_head_score_diagnostics.json",
        plot_name: str = "per_head_scores.png",
        margin_plot_name: str = "score_margins.png",
        raw_plot_name: str = "per_head_raw_scores.png",
    ):
        super().__init__()
        self.output_name = output_name
        self.plot_name = plot_name
        self.margin_plot_name = margin_plot_name
        self.raw_plot_name = raw_plot_name
        self._scores: list[torch.Tensor] = []
        self._raw_scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if not isinstance(outputs, dict):
            return
        score_tensor = outputs.get("scores")
        raw_score_tensor = outputs.get("raw_scores")
        label_tensor = outputs.get("labels")
        if score_tensor is None or label_tensor is None:
            return
        self._scores.append(cast(torch.Tensor, score_tensor).cpu())
        if raw_score_tensor is not None:
            self._raw_scores.append(cast(torch.Tensor, raw_score_tensor).cpu())
        self._labels.append(cast(torch.Tensor, label_tensor).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._scores:
            return
        scores = torch.cat(self._scores).numpy()
        raw_scores = torch.cat(self._raw_scores).numpy() if self._raw_scores else None
        labels = torch.cat(self._labels).numpy()
        predictions = np.argmax(scores, axis=1)
        datamodule = getattr(trainer, "datamodule", None)
        class_names = getattr(datamodule, "class_names", [str(index) for index in range(scores.shape[1])])
        calibrated_thresholds: list[float] = []
        raw_thresholds: list[float] = []
        diagnostics: dict[str, Any] = {
            "score_spaces": {
                "calibrated": {
                    "meaning": "dimensionless decision margin; values > 0 mean the sample is inside that head's acceptance region",
                    "threshold": 0.0,
                },
                "raw": {
                    "meaning": "pre-calibration head score in the model's native space",
                    "threshold_source": "per-head",
                },
            },
            "per_head": {},
            "top1_top2_margin_mean": 0.0,
            "true_class_margin_mean": 0.0,
            "prediction_distribution": {},
        }
        if scores.shape[1] >= 2:
            sorted_scores = np.sort(scores, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
            diagnostics["top1_top2_margin_mean"] = float(margins.mean())

            true_scores = scores[np.arange(scores.shape[0]), labels]
            masked_scores = scores.copy()
            masked_scores[np.arange(scores.shape[0]), labels] = -np.inf
            best_wrong_scores = masked_scores.max(axis=1)
            diagnostics["true_class_margin_mean"] = float((true_scores - best_wrong_scores).mean())
        else:
            margins = np.array([], dtype=np.float32)

        for class_index in range(scores.shape[1]):
            diagnostics["prediction_distribution"][str(class_index)] = int((predictions == class_index).sum())

        model = getattr(pl_module, "model", None)
        model_raw_thresholds = None
        if model is not None and hasattr(model, "raw_score_thresholds"):
            model_raw_thresholds = cast(torch.Tensor, model.raw_score_thresholds()).detach().cpu().numpy()
        model_calibration_mode = None
        if model is not None and hasattr(model, "calibration_mode"):
            model_calibration_mode = str(model.calibration_mode())
        model_calibrated_threshold = None
        if model is not None and hasattr(model, "calibrated_acceptance_threshold"):
            model_calibrated_threshold = float(model.calibrated_acceptance_threshold())
        for head_index in range(scores.shape[1]):
            positive_mask = labels == head_index
            negative_mask = labels != head_index
            head_module = model.get_head(head_index) if model is not None and hasattr(model, "get_head") else None
            raw_threshold = 0.0
            if head_module is not None:
                if hasattr(head_module, "raw_acceptance_threshold"):
                    raw_threshold = float(head_module.raw_acceptance_threshold())
                elif hasattr(head_module, "running_threshold"):
                    raw_threshold = float(getattr(head_module, "running_threshold").item())
            elif model_raw_thresholds is not None and head_index < len(model_raw_thresholds):
                raw_threshold = float(model_raw_thresholds[head_index])
            raw_thresholds.append(raw_threshold)

            calibrated_positive = scores[positive_mask, head_index]
            calibrated_negative = scores[negative_mask, head_index]
            raw_positive = raw_scores[positive_mask, head_index] if raw_scores is not None else np.array([], dtype=np.float32)
            raw_negative = raw_scores[negative_mask, head_index] if raw_scores is not None else np.array([], dtype=np.float32)
            calibration_mode = "none"
            if head_module is not None and hasattr(head_module, "calibration_mode"):
                calibration_mode = str(head_module.calibration_mode())
            elif model_calibration_mode is not None:
                calibration_mode = model_calibration_mode
            calibrated_threshold = 0.0 if model_calibrated_threshold is None or np.isnan(model_calibrated_threshold) else model_calibrated_threshold
            calibrated_thresholds.append(calibrated_threshold)
            diagnostics["per_head"][str(head_index)] = {
                "class_name": class_names[head_index] if head_index < len(class_names) else str(head_index),
                "calibration_mode": calibration_mode,
                "calibrated_threshold": calibrated_threshold,
                "calibrated_positive_mean": float(calibrated_positive.mean()) if np.any(positive_mask) else 0.0,
                "calibrated_positive_std": float(calibrated_positive.std()) if np.any(positive_mask) else 0.0,
                "calibrated_negative_mean": float(calibrated_negative.mean()) if np.any(negative_mask) else 0.0,
                "calibrated_negative_std": float(calibrated_negative.std()) if np.any(negative_mask) else 0.0,
                "positive_accept_rate": float((calibrated_positive > 0.0).mean()) if np.any(positive_mask) else 0.0,
                "negative_accept_rate": float((calibrated_negative > 0.0).mean()) if np.any(negative_mask) else 0.0,
                "raw_threshold": raw_threshold,
                "raw_positive_mean": float(raw_positive.mean()) if raw_positive.size else 0.0,
                "raw_positive_std": float(raw_positive.std()) if raw_positive.size else 0.0,
                "raw_negative_mean": float(raw_negative.mean()) if raw_negative.size else 0.0,
                "raw_negative_std": float(raw_negative.std()) if raw_negative.size else 0.0,
            }

        output_dir = Path(getattr(trainer, "log_dir", trainer.default_root_dir))
        output_path = output_dir / self.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        plot_per_head_score_distributions(
            scores,
            labels,
            class_names,
            str(output_dir / self.plot_name),
            calibrated_thresholds if calibrated_thresholds and not any(np.isnan(threshold) for threshold in calibrated_thresholds) else None,
            title="Per-Head Calibrated Score Distributions",
            x_label="Calibrated decision margin",
            threshold_label="acceptance boundary",
        )
        if raw_scores is not None:
            plot_per_head_score_distributions(
                raw_scores,
                labels,
                class_names,
                str(output_dir / self.raw_plot_name),
                raw_thresholds,
                title="Per-Head Raw Score Distributions",
                x_label="Raw score",
                threshold_label="raw threshold",
            )
        plot_calibration_margins(margins, str(output_dir / self.margin_plot_name))
        self._scores.clear()
        self._raw_scores.clear()
        self._labels.clear()


class EvalBatchLossPlotCallback(pl.Callback):
    def __init__(self, output_name: str = "eval_batch_loss.png"):
        super().__init__()
        self.output_name = output_name
        self._batch_losses: list[float] = []

    def on_test_start(self, trainer, pl_module) -> None:
        self._batch_losses.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if not isinstance(outputs, dict):
            return
        loss_value = outputs.get("loss")
        if loss_value is None:
            return
        self._batch_losses.append(float(cast(torch.Tensor, loss_value).detach().cpu().item()))

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._batch_losses:
            return
        output_dir = Path(getattr(trainer, "log_dir", trainer.default_root_dir))
        plot_batch_metric(self._batch_losses, str(output_dir / self.output_name), title="Evaluation Batch Loss", ylabel="Loss")


class OnlineBatchLossPlotCallback(pl.Callback):
    def __init__(self, output_name: str = "online_batch_loss.png"):
        super().__init__()
        self.output_name = output_name
        self._batch_losses: list[float] = []
        self._batch_changes: list[bool] = []
        self._batch_change_to: list[str] = []

    def on_train_start(self, trainer, pl_module) -> None:
        self._batch_losses.clear()
        self._batch_changes.clear()
        self._batch_change_to.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not isinstance(outputs, dict):
            return
        loss_value = outputs.get("loss")
        if loss_value is None:
            return
        self._batch_losses.append(float(cast(torch.Tensor, loss_value).detach().cpu().item()))
        self._batch_changes.append(bool(outputs.get("batch_change", False)))
        self._batch_change_to.append(str(outputs.get("batch_change_to", "")))

    def on_train_end(self, trainer, pl_module) -> None:
        if not self._batch_losses:
            return
        datamodule = getattr(trainer, "datamodule", None)
        class_names = getattr(datamodule, "class_names", None)
        output_dir = Path(getattr(trainer, "log_dir", trainer.default_root_dir))
        plot_batch_loss(
            batch_losses=self._batch_losses,
            batch_changes=self._batch_changes,
            batch_change_to=self._batch_change_to,
            output_path=str(output_dir / self.output_name),
            class_names=class_names,
        )
