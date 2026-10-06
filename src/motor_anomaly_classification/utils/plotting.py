from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_epoch_loss(history: dict[str, list[float]], title: str, output_path: str | None = None) -> None:
    if not history.get("train"):
        return
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train"], marker="o", label="train")
    if history.get("val"):
        plt.plot(epochs, history["val"], marker="o", label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_batch_loss(
    batch_losses: list[float],
    batch_changes: list[bool],
    batch_change_to: list[str],
    output_path: str,
    title: str = "Online Batch Loss",
    class_names: list[str] | None = None,
) -> None:
    if not batch_losses:
        return

    x_axis = np.arange(1, len(batch_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, batch_losses, linewidth=1.2, label="batch loss")

    if len(batch_changes) == len(batch_losses):
        marked_steps = [index for index, value in enumerate(batch_changes, start=1) if value]
        if marked_steps:
            marked_losses = [batch_losses[index - 1] for index in marked_steps]
            plt.scatter(marked_steps, marked_losses, color="tab:red", s=14, label="class-change")
            for step in marked_steps[:35]:
                raw_text = str(batch_change_to[step - 1]) if len(batch_change_to) == len(batch_losses) else ""
                if raw_text and class_names is not None:
                    tokens = [token for token in raw_text.split("|") if token]
                    mapped = [class_names[int(token)] if token.isdigit() and int(token) < len(class_names) else token for token in tokens]
                    text = "to:" + "|".join(mapped)
                elif raw_text:
                    text = "to:" + raw_text
                else:
                    text = "change"
                plt.annotate(text, (step, batch_losses[step - 1]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, rotation=90)

    plt.title(title)
    plt.xlabel("Batch step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_batch_metric(
    values: list[float],
    output_path: str,
    title: str,
    ylabel: str,
) -> None:
    if not values:
        return
    x_axis = np.arange(1, len(values) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, values, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Batch step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: str | None = None) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            plt.text(col, row, str(value), ha="center", va="center", color="white" if value > threshold else "black")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_speed_accuracy(per_speed_accuracy: dict[str, float], output_path: str, title: str = "Per-Speed Accuracy") -> None:
    if not per_speed_accuracy:
        return
    speed_labels = list(per_speed_accuracy.keys())
    values = [per_speed_accuracy[label] for label in speed_labels]
    plt.figure(figsize=(9, 5))
    plt.bar(speed_labels, values, color="tab:blue")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("Speed")
    plt.ylabel("Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_head_score_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: str,
    thresholds: list[float] | None = None,
    title: str = "Per-Head Score Distributions",
    x_label: str = "Score",
    threshold_label: str = "threshold",
) -> None:
    if scores.size == 0:
        return
    head_count = scores.shape[1]
    fig, axes = plt.subplots(head_count, 1, figsize=(10, 3 * head_count), sharex=True)
    if head_count == 1:
        axes = [axes]
    for head_index, axis in enumerate(axes):
        positive_mask = labels == head_index
        negative_mask = labels != head_index
        axis.hist(scores[positive_mask, head_index], bins=30, alpha=0.6, label=f"true={class_names[head_index]}", color="tab:green")
        axis.hist(scores[negative_mask, head_index], bins=30, alpha=0.5, label="others", color="tab:gray")
        if thresholds is not None and head_index < len(thresholds):
            axis.axvline(thresholds[head_index], color="tab:red", linestyle="--", linewidth=1.2, label=threshold_label)
        axis.set_ylabel(f"Head {head_index}")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel(x_label)
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_margins(margins: np.ndarray, output_path: str, title: str = "Top-1 Minus Top-2 Score Margin") -> None:
    if margins.size == 0:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(margins, bins=40, color="tab:orange", alpha=0.8)
    plt.title(title)
    plt.xlabel("Margin")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_selected_windows_by_class(
    data_by_class: dict[str, np.ndarray],
    class_to_idx: dict[str, int],
    selected_indices: dict[str, list[int]],
    output_dir: str,
    channel_names: tuple[str, ...] | None = None,
) -> None:
    if not selected_indices:
        return

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for class_name, indices in selected_indices.items():
        if class_name not in data_by_class:
            continue
        class_data = data_by_class[class_name]
        class_label = class_to_idx[class_name]
        for window_index in indices:
            if window_index < 0 or window_index >= class_data.shape[0]:
                continue
            window = class_data[window_index]
            num_channels = int(window.shape[1])
            labels = channel_names if channel_names and len(channel_names) == num_channels else tuple(f"ch{index}" for index in range(num_channels))
            fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2.5 * num_channels), sharex=True)
            if num_channels == 1:
                axes = [axes]
            for channel_index, axis in enumerate(axes):
                axis.plot(window[:, channel_index], linewidth=1.2)
                axis.set_ylabel(labels[channel_index])
                axis.grid(True, alpha=0.3)
            axes[-1].set_xlabel("Time step")
            fig.suptitle(f"class={class_name} label={class_label} window={window_index}")
            fig.tight_layout()
            fig.savefig(destination / f"{class_name}_label{class_label}_window{window_index}.png", dpi=150)
            plt.close(fig)
