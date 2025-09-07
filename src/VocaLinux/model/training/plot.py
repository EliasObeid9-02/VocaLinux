"""This module provides utilities for plotting training history from a Keras History object.

It includes functions to generate and save plots for loss, accuracy, Character Error Rate (CER),
and Word Error Rate (WER), including training, validation, and sampling probability curves.
"""

import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import tensorflow as tf


# --- Configuration ---
# Use a professional plot style
style.use("seaborn-v0_8-whitegrid")
# Define a color palette for consistency
colors = {
    "training": "#00529B",  # A strong, professional blue
    "validation": "#D81B60",  # A vibrant, attention-grabbing magenta/red
    "sampling_prob": "#4E4E4E",  # A dark, contrasting gray
    "text": "#000000",
}


def plot_and_save_metric(
    epochs: np.ndarray,
    history_dict: Dict[str, Any],
    train_key: str,
    val_key: str,
    title: str,
    y_label: str,
    output_dir: str,
) -> None:
    """Generic function to create, style, and save a single plot for a given metric.

    Includes a secondary y-axis for sampling probability if available in history_dict.

    Args:
        epochs (np.ndarray): Array of epoch numbers.
        history_dict (Dict[str, Any]): Dictionary containing training history data.
        train_key (str): Key for the training metric in history_dict (e.g., 'loss').
        val_key (str): Key for the validation metric in history_dict (e.g., 'val_loss').
        title (str): Title of the plot.
        y_label (str): Label for the primary y-axis.
        output_dir (str): Directory to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Primary axis for the main metric
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, color=colors["text"])
    ax.tick_params(axis="y", labelcolor=colors["training"])

    # Plot Training and Validation metrics
    if train_key in history_dict:
        ax.plot(
            epochs,
            history_dict[train_key],
            label=f"Training {y_label}",
            color=colors["training"],
            linestyle="-",
        )
    if val_key in history_dict:
        ax.plot(
            epochs,
            history_dict[val_key],
            label=f"Validation {y_label}",
            color=colors["validation"],
            linestyle="-",
        )
    ax.legend(loc="upper left")

    # Secondary axis for Sampling Probability
    if "sampling_probability" in history_dict:
        ax2 = ax.twinx()
        ax2.set_ylabel("Sampling Probability", fontsize=12, color=colors["text"])
        ax2.plot(
            epochs,
            history_dict["sampling_probability"],
            label="Sampling Probability",
            color=colors["sampling_prob"],
            linestyle=":",
        )
        ax2.tick_params(axis="y", labelcolor=colors["sampling_prob"])
        ax2.legend(loc="upper right")

    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.set_xlim(left=0, right=len(epochs) + 1)  # Ensure x-axis starts at 0 and ends correctly
    ax.set_xticks(epochs) # Set x-ticks to actual epoch numbers

    # Save the figure
    safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    file_name = f"{safe_title}.png"
    output_path = os.path.join(output_dir, file_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)  # Close the figure to free memory
    print(f"Plot saved to: {output_path}")


def plot_training_history(history: tf.keras.callbacks.History, output_dir: str) -> None:
    """Generates and saves plots for various training metrics from a Keras History object.

    Args:
        history (tf.keras.callbacks.History): The Keras History object containing training logs.
        output_dir (str): The directory where the plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    history_dict = history.history
    epochs = np.arange(1, len(history_dict["loss"]) + 1)

    # Plot and save each metric
    plot_and_save_metric(
        epochs,
        history_dict,
        "loss",
        "val_loss",
        "Model Loss",
        "Loss",
        output_dir,
    )
    plot_and_save_metric(
        epochs,
        history_dict,
        "accuracy",
        "val_accuracy",
        "Model Accuracy",
        "Accuracy",
        output_dir,
    )
    plot_and_save_metric(
        epochs,
        history_dict,
        "character_error_rate",
        "val_character_error_rate",
        "Character Error Rate (CER)",
        "CER",
        output_dir,
    )
    plot_and_save_metric(
        epochs,
        history_dict,
        "word_error_rate",
        "val_word_error_rate",
        "Word Error Rate (WER)",
        "WER",
        output_dir,
    )