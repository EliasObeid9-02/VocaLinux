import argparse
import glob
import json
import os

import numpy as np


def load_and_process_data(data_dir):
    """
    Loads all JSON files from a specified directory, sorts them numerically,
    and concatenates the epoch metrics into a list of dictionaries.
    """
    file_pattern = os.path.join(data_dir, "history_*.json")
    json_files = sorted(glob.glob(file_pattern))

    if not json_files:
        print(f"Error: No files matching '{file_pattern}' found. Please check the --data_dir path.")
        return None

    all_epoch_metrics = []
    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for epoch in data.get("epochs", []):
                    metrics = epoch.get("epoch_metrics")
                    if isinstance(metrics, dict):
                        all_epoch_metrics.append(metrics)
                    else:
                        print(
                            f"Warning: 'epoch_metrics' in {file_path} is not a dictionary. Skipping."
                        )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}. Skipping.")

    if not all_epoch_metrics:
        print("Error: No valid epoch metrics could be loaded from any files.")
        return None

    return all_epoch_metrics


def calculate_average_change(metrics_data):
    """
    Calculates the average change for each metric from a list of dictionaries,
    excluding 'sampling_probability'.
    """
    if not metrics_data or len(metrics_data) < 2:
        print("Warning: Not enough data points (less than 2 epochs) to calculate a change.")
        return None

    # Get all metric keys from the first epoch and exclude 'sampling_probability'
    metric_keys = [key for key in metrics_data[0].keys() if key != "sampling_probability"]
    changes = {key: [] for key in metric_keys}

    # Iterate from the second epoch to the end to calculate differences
    for i in range(1, len(metrics_data)):
        for key in metric_keys:
            if key in metrics_data[i] and key in metrics_data[i - 1]:
                diff = metrics_data[i][key] - metrics_data[i - 1][key]
                changes[key].append(diff)
            else:
                print(
                    f"Warning: Metric '{key}' missing in epoch {i-1} or {i}. Cannot calculate change for this step."
                )

    # Calculate the average of the collected changes for each metric
    average_changes = {}
    for key, value_changes in changes.items():
        if value_changes:
            average_changes[key] = np.mean(value_changes)
        else:
            average_changes[key] = "N/A (no changes calculated)"

    return average_changes


def main():
    """
    Main function to orchestrate data loading, processing, and calculation of average metric changes.
    """
    parser = argparse.ArgumentParser(
        description="Calculate the average change in training metrics from JSON files without using pandas."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing the history JSON files.",
    )
    args = parser.parse_args()

    metrics_list = load_and_process_data(args.data_dir)

    if not metrics_list:
        print("Execution halted due to data loading errors.")
        return

    average_changes = calculate_average_change(metrics_list)

    if average_changes:
        print("\nAverage Change per Epoch for Each Metric:")

        # Find the longest metric name for alignment
        max_key_length = max(len(key) for key in average_changes.keys())

        # Print with aligned formatting
        for metric, change in average_changes.items():
            if isinstance(change, str):
                print(f"{metric:<{max_key_length}} : {change}")
            else:
                # Format with a sign and fixed decimal places for alignment
                print(f"{metric:<{max_key_length}} : {change:+.6f}")


if __name__ == "__main__":
    main()
