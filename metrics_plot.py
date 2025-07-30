import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np

# --- Configuration ---
# Use a professional plot style
style.use("seaborn-v0_8-whitegrid")
# Define a color palette for consistency
colors = {
    "training": "#00529B",  # A strong, professional blue
    "validation": "#D81B60",  # A vibrant, attention-grabbing magenta/red
    "sampling_prob": "#4E4E4E",  # A dark, contrasting gray
    "std_fill": 0.2,
    "text": "#000000",
}

# --- Data Loading and Processing ---


def load_and_process_data(data_dir):
    """
    Loads all JSON files from a specified directory, sorts them numerically,
    and concatenates the epoch metrics into a single continuous timeline.
    """
    file_pattern = os.path.join(data_dir, "history_*.json")
    json_files = glob.glob(file_pattern)

    if not json_files:
        print(f"Error: No files matching '{file_pattern}' found. Please check the --data_dir path.")
        return None, 0

    # Sort files numerically based on the digits in their names
    json_files.sort()

    print("Processing files in the following order:")
    for f in json_files:
        print(f"- {os.path.basename(f)}")

    # Use the keys from the first file's first epoch as the template
    try:
        with open(json_files[0], "r") as f:
            first_data = json.load(f)
            metric_keys = first_data["epochs"][0]["epoch_metrics"].keys()
            combined_metrics = {key: [] for key in metric_keys}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error reading template metrics from {json_files[0]}: {e}")
        return None, 0

    # Iterate through sorted files and append data
    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for epoch in data.get("epochs", []):
                    epoch_metrics = epoch.get("epoch_metrics", {})
                    for key in combined_metrics.keys():
                        combined_metrics[key].append(epoch_metrics.get(key, np.nan))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}. Skipping.")

    if not combined_metrics or not combined_metrics.get("loss"):
        print("Error: No valid data could be loaded and combined from any files.")
        return None, 0

    num_epochs = len(combined_metrics["loss"])
    return combined_metrics, num_epochs


# --- Plotting Functions ---


def plot_and_save_metric(epochs, metrics, train_key, val_key, title, y_label, output_dir):
    """
    Generic function to create, style, and save a single plot for a given metric.
    Includes a secondary y-axis for sampling probability.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Primary axis for the main metric
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, color=colors["text"])
    ax.tick_params(axis="y", labelcolor=colors["training"])

    # Plot Training and Validation metrics
    ax.plot(
        epochs,
        metrics[train_key],
        label=f"Training {y_label}",
        color=colors["training"],
        linestyle="-",
    )
    ax.plot(
        epochs,
        metrics[val_key],
        label=f"Validation {y_label}",
        color=colors["validation"],
        linestyle="-",
    )
    ax.legend(loc="upper left")

    # Secondary axis for Sampling Probability
    if "sampling_probability" in metrics:
        ax2 = ax.twinx()
        ax2.set_ylabel("Sampling Probability", fontsize=12, color=colors["text"])
        ax2.plot(
            epochs,
            metrics["sampling_probability"],
            label="Sampling Probability",
            color=colors["sampling_prob"],
            linestyle=":",
        )
        ax2.tick_params(axis="y", labelcolor=colors["sampling_prob"])
        ax2.legend(loc="upper right")

    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.set_xlim(left=0, right=len(epochs))  # Ensure x-axis starts at 0

    # Save the figure
    safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    file_name = f"{safe_title}.png"
    output_path = os.path.join(output_dir, file_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)  # Close the figure to free memory
    print(f"Plot saved to: {output_path}")


# --- Main Execution ---


def main():
    """Main function to orchestrate data loading, processing, and plotting for all files combined."""
    parser = argparse.ArgumentParser(description="Plot combined training history from JSON files.")
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Directory containing the history JSON files."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save the output plot images."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    combined_metrics, total_epochs = load_and_process_data(args.data_dir)

    if combined_metrics is None or total_epochs == 0:
        print("Execution halted due to data loading errors.")
        return

    epochs = np.arange(1, total_epochs + 1)

    # Plot and save each metric for the combined data
    plot_and_save_metric(
        epochs,
        combined_metrics,
        "loss",
        "val_loss",
        "Model Loss",
        "Loss",
        args.output_dir,
    )
    plot_and_save_metric(
        epochs,
        combined_metrics,
        "accuracy",
        "val_accuracy",
        "Model Accuracy",
        "Accuracy",
        args.output_dir,
    )
    plot_and_save_metric(
        epochs,
        combined_metrics,
        "cer",
        "val_cer",
        "Character Error Rate (CER)",
        "CER",
        args.output_dir,
    )
    plot_and_save_metric(
        epochs,
        combined_metrics,
        "wer",
        "val_wer",
        "Word Error Rate (WER)",
        "WER",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
