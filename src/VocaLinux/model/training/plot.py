import json

import matplotlib.pyplot as plt
import numpy as np


def create_training_plots(filepath: str):
    """
    Creates matplotlib plots from training history but doesn't show/save them.
    Returns the figure object and the data used for plotting.

    Args:
        filepath (str): Path to the JSON history file

    Returns:
        fig: matplotlib figure
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode JSON from {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

    epochs_data = data.get("epochs", [])
    if not epochs_data:
        raise ValueError("No epoch data found in the JSON file.")

    # Initialize data collection containers
    plot_data = {
        "batch": {"indices": [], "loss": [], "accuracy": [], "cer": [], "wer": []},
        "epoch": {
            "numbers": [],
            "sampling_prob": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_cer": [],
            "val_wer": [],
        },
    }

    current_batch_count = 0

    # Collect all data
    for epoch_idx, epoch_data_dict in enumerate(epochs_data):
        # Batch metrics
        for batch_info in epoch_data_dict.get("batch_metrics", []):
            plot_data["batch"]["indices"].append(current_batch_count)
            plot_data["batch"]["loss"].append(batch_info.get("loss"))
            plot_data["batch"]["accuracy"].append(batch_info.get("accuracy"))
            plot_data["batch"]["cer"].append(batch_info.get("cer"))
            plot_data["batch"]["wer"].append(batch_info.get("wer"))
            current_batch_count += 1

        # Epoch metrics
        epoch_metrics = epoch_data_dict.get("epoch_metrics", {})
        plot_data["epoch"]["numbers"].append(epoch_idx + 1)  # 1-based indexing
        plot_data["epoch"]["sampling_prob"].append(epoch_metrics.get("sampling_probability"))

        # Validation metrics (use None if not available)
        plot_data["epoch"]["val_loss"].append(epoch_metrics.get("val_loss"))
        plot_data["epoch"]["val_accuracy"].append(epoch_metrics.get("val_accuracy"))
        plot_data["epoch"]["val_cer"].append(epoch_metrics.get("val_cer"))
        plot_data["epoch"]["val_wer"].append(epoch_metrics.get("val_wer"))

    # Convert to numpy arrays
    for key in plot_data["batch"]:
        plot_data["batch"][key] = np.array(plot_data["batch"][key], dtype=np.float32)

    for key in plot_data["epoch"]:
        if key != "numbers":  # numbers should stay as integers
            plot_data["epoch"][key] = np.array(plot_data["epoch"][key], dtype=np.float32)

    # Calculate epoch boundaries for vertical lines
    batches_per_epoch = len(epochs_data[0]["batch_metrics"]) if epochs_data else 0
    epoch_boundaries = np.arange(
        batches_per_epoch, len(plot_data["batch"]["indices"]), batches_per_epoch
    )

    # Create figure with more subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3)

    # Batch metrics plots
    ax1 = fig.add_subplot(gs[0, 0])  # Batch loss
    ax2 = fig.add_subplot(gs[0, 1])  # Batch accuracy
    ax3 = fig.add_subplot(gs[0, 2])  # Batch CER
    ax4 = fig.add_subplot(gs[1, 0])  # Batch WER

    # Epoch metrics plots
    ax5 = fig.add_subplot(gs[1, 1])  # Sampling probability
    ax6 = fig.add_subplot(gs[1, 2])  # Validation loss
    ax7 = fig.add_subplot(gs[2, 0])  # Validation accuracy
    ax8 = fig.add_subplot(gs[2, 1])  # Validation CER
    ax9 = fig.add_subplot(gs[2, 2])  # Validation WER

    # Plot batch metrics
    def plot_with_epoch_lines(ax, x, y, title, ylabel, color):
        ax.plot(x, y, color=color, alpha=0.7)
        for boundary in epoch_boundaries:
            ax.axvline(x=boundary, color="black", linestyle="--", alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Batch Number")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    plot_with_epoch_lines(
        ax1, plot_data["batch"]["indices"], plot_data["batch"]["loss"], "Batch Loss", "Loss", "blue"
    )
    plot_with_epoch_lines(
        ax2,
        plot_data["batch"]["indices"],
        plot_data["batch"]["accuracy"],
        "Batch Accuracy",
        "Accuracy",
        "green",
    )
    plot_with_epoch_lines(
        ax3, plot_data["batch"]["indices"], plot_data["batch"]["cer"], "Batch CER", "CER", "orange"
    )
    plot_with_epoch_lines(
        ax4, plot_data["batch"]["indices"], plot_data["batch"]["wer"], "Batch WER", "WER", "red"
    )

    # Plot epoch metrics
    def plot_epoch_metric(ax, x, y, title, ylabel, color, marker="o"):
        valid_indices = ~np.isnan(y)
        if np.any(valid_indices):
            ax.plot(x, y, color=color, marker=marker, linestyle="-")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_xticks(plot_data["epoch"]["numbers"])
            ax.grid(True)
        else:
            ax.set_title(f"{title} (No Data)")
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)

    # Sampling probability
    plot_epoch_metric(
        ax5,
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["sampling_prob"],
        "Sampling Probability Over Epochs",
        "Probability",
        "purple",
    )

    # Validation metrics
    plot_epoch_metric(
        ax6,
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_loss"],
        "Validation Loss",
        "Loss",
        "darkred",
    )
    plot_epoch_metric(
        ax7,
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_accuracy"],
        "Validation Accuracy",
        "Accuracy",
        "darkgreen",
    )
    plot_epoch_metric(
        ax8,
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_cer"],
        "Validation CER",
        "CER",
        "darkorange",
    )
    plot_epoch_metric(
        ax9,
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_wer"],
        "Validation WER",
        "WER",
        "brown",
    )
    plt.tight_layout()
    return fig
