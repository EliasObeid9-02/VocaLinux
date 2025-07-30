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

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle("Training and Validation Metrics", fontsize=16)

    # Batch Loss
    axes[0, 0].plot(
        plot_data["batch"]["indices"],
        plot_data["batch"]["loss"],
        label="Batch Loss",
        color="orange",
        alpha=0.7,
    )
    axes[0, 0].set_xlabel("Batch Index")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Batch Loss over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Batch Accuracy
    axes[0, 1].plot(
        plot_data["batch"]["indices"],
        plot_data["batch"]["accuracy"],
        label="Batch Accuracy",
        color="green",
        alpha=0.7,
    )
    axes[0, 1].set_xlabel("Batch Index")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Batch Accuracy over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Validation Loss and Sampling Probability
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    ax1.plot(
        plot_data["epoch"]["numbers"], plot_data["epoch"]["val_loss"], "b-", label="Validation Loss"
    )
    ax2.plot(
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["sampling_prob"],
        "r--",
        label="Sampling Probability",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss", color="b")
    ax2.set_ylabel("Sampling Probability", color="r")
    ax1.set_title("Validation Loss and Sampling Probability")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)

    # Validation Accuracy
    axes[1, 1].plot(
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_accuracy"],
        label="Validation Accuracy",
        color="purple",
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Validation CER
    axes[2, 0].plot(
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_cer"],
        label="Validation CER",
        color="cyan",
    )
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("CER")
    axes[2, 0].set_title("Validation Character Error Rate (CER)")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Validation WER
    axes[2, 1].plot(
        plot_data["epoch"]["numbers"],
        plot_data["epoch"]["val_wer"],
        label="Validation WER",
        color="magenta",
    )
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("WER")
    axes[2, 1].set_title("Validation Word Error Rate (WER)")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
