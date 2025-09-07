"""Main entry point for training and evaluating the LAS model."""

import argparse
import os
from typing import Optional, Tuple

import tensorflow as tf

from VocaLinux.configs import training as training_config
from VocaLinux.data.dataset import LibriSpeechDatasetLoader
from VocaLinux.model.build.builder import (
    create_model_from_scratch,
    load_model_from_file,
    rebuild_model,
)
from VocaLinux.model.training.evaluate import evaluate_model
from VocaLinux.model.training.plot import plot_training_history
from VocaLinux.model.training.train import train_model


def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
    """
    Finds the latest model checkpoint in a directory and extracts its epoch number.
    """
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        try:
            epoch_num_str = os.path.basename(latest_checkpoint).split("_")[2]
            initial_epoch = int(epoch_num_str)
            return latest_checkpoint, initial_epoch
        except (IndexError, ValueError):
            print(
                f"Warning: Could not parse epoch from checkpoint name {latest_checkpoint}. Starting from epoch 0."
            )
            return latest_checkpoint, 0
    return None, 0


def main() -> None:
    """
    Main function to orchestrate the model training process.
    """
    parser = argparse.ArgumentParser(description="Train or evaluate the VocaLinux LAS model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Mode to run: 'train' or 'evaluate'.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="train",
        help="Base directory for all training-related outputs (checkpoints, logs, plots).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Directory containing the dataset for training and evaluation.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Path to a specific pre-trained model file (.keras) to load. If not provided, will look for latest checkpoint or create new model.",
    )
    parser.add_argument(
        "--recompile_model",
        action="store_true",
        help="If set, the loaded model will be recompiled. Useful if metrics/optimizer changed.",
    )
    parser.add_argument(
        "--restart_training",
        action="store_true",
        help="If set, training will start from epoch 0, ignoring any previous checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_config.EPOCHS,
        help=f"Total number of epochs to train for. Default: {training_config.EPOCHS}",
    )

    args = parser.parse_args()

    # --- Derived Paths ---
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    log_dir = os.path.join(args.base_dir, "logs")
    history_save_path = os.path.join(args.base_dir, "training_history.json")
    plot_output_dir = os.path.join(args.base_dir, "plots")
    evaluation_output_dir = os.path.join(args.base_dir, "evaluation_results")

    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs(evaluation_output_dir, exist_ok=True)

    # --- Setup Distribution Strategy ---
    # For single GPU or CPU training, use OneDeviceStrategy.
    # If multiple GPUs are available and desired, MirroredStrategy would be used.
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") if tf.config.list_physical_devices("GPU") else tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print(f"Using distribution strategy: {strategy.worker_devices}")

    with strategy.scope():
        # --- Model Loading/Building Logic ---
        model = None
        initial_epoch = 0

        if args.restart_training:
            print("Restarting training from scratch. Building a new model.")
            model = create_model_from_scratch()
        elif args.load_model_path:
            print(f"Loading model from specified path: {args.load_model_path}")
            model = load_model_from_file(args.load_model_path)
            if args.recompile_model:
                model = rebuild_model(model)
            # When loading a specific model, we don't automatically resume epoch count
            # unless history_save_path is explicitly used to determine it later.
        else:
            latest_checkpoint, checkpoint_epoch = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print(f"Loading latest checkpoint: {latest_checkpoint}")
                model = load_model_from_file(latest_checkpoint)
                if args.recompile_model:
                    model = rebuild_model(model)
                initial_epoch = checkpoint_epoch
            else:
                print("No checkpoint found. Building a new model from scratch.")
                model = create_model_from_scratch()

        if model is None:
            raise RuntimeError("Failed to load or create a model.")

        loader = LibriSpeechDatasetLoader(
            data_root=args.dataset_dir,
            batch_size=training_config.BATCH_SIZE,
        )

        # --- Data Loading ---
        print(f"Loading datasets from: {args.dataset_dir}")
        train_ds_unbatched, val_ds_unbatched = loader.get_partitioned_datasets(
            split="train-clean-100",
            partitions=[0.8, 0.2],
            shuffle=True,
        )

        test_ds_unbatched = loader.get_dataset(
            split="test-clean",
            shuffle=False,
        )

        # Distribute datasets using the strategy
        train_ds = strategy.experimental_distribute_dataset(train_ds_unbatched)
        val_ds = strategy.experimental_distribute_dataset(val_ds_unbatched)
        test_ds = strategy.experimental_distribute_dataset(test_ds_unbatched)

    if args.mode == "train":
        print("--- Starting Training ---")
        history = train_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            initial_epoch=initial_epoch,
            epochs=args.epochs,
            history_save_path=history_save_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

        # --- Plotting ---
        if history:
            print("Plotting training history...")
            plot_training_history(history, plot_output_dir)
        tf.keras.backend.clear_session()  # Clear session to free memory

    elif args.mode == "evaluate":
        print("--- Starting Evaluation ---")
        evaluate_model(model, test_ds, evaluation_output_dir)
        tf.keras.backend.clear_session()  # Clear session to free memory


if __name__ == "__main__":
    main()
