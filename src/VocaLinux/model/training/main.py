"""Main entry point for training and evaluating the LAS model."""

import os
from typing import Optional, Tuple

import tensorflow as tf

from VocaLinux.configs import training as training_config
from VocaLinux.model.build.metrics import CharacterErrorRate, WordErrorRate
from VocaLinux.model.las_model import LASModel
from VocaLinux.model.training.evaluate import evaluate_model
from VocaLinux.model.training.plot import plot_training_history
from VocaLinux.model.training.train import train_model

# Ensure custom objects are registered for loading models
tf.keras.utils.get_custom_objects().update(
    {
        "LASModel": LASModel,
        "CharacterErrorRate": CharacterErrorRate,
        "WordErrorRate": WordErrorRate,
    }
)


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
            return None, 0
    return None, 0


def main() -> None:
    """
    Main function to orchestrate the model training process.
    """
    # --- Configuration ---
    base_dir = "train"
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    history_save_path = os.path.join(base_dir, "training_history.json")
    plot_output_dir = os.path.join(base_dir, "plots")

    os.makedirs(base_dir, exist_ok=True)

    # --- Setup ---
    latest_checkpoint, initial_epoch = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint)
    else:
        print("Building a new model...")
        # This is a placeholder for your model building logic
        # from VocaLinux.model.build import build_model
        # model = build_model()
        raise NotImplementedError("Model building from scratch is not implemented.")

    # --- Data Loading ---
    # Placeholder for your data loading logic
    # from VocaLinux.data.loader import load_datasets
    # train_ds, val_ds, test_ds = load_datasets(...)
    def dummy_generator():
        for _ in range(100):
            yield (
                tf.random.normal([10, 80, 1]),
                tf.random.uniform([10, 5], maxval=10, dtype=tf.int32),
            )

    train_ds = tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 80, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
        ),
    ).batch(10)

    val_ds = tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 80, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
        ),
    ).batch(10)

    test_ds = tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 80, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
        ),
    ).batch(10)

    # --- Training ---
    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        initial_epoch=initial_epoch,
        epochs=training_config.EPOCHS,
        history_save_path=history_save_path,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # --- Plotting ---
    if history:
        print("Plotting training history...")
        plot_training_history(history, plot_output_dir)

    # --- Evaluation ---
    evaluate_model(model, test_ds, log_dir)


if __name__ == "__main__":
    main()
