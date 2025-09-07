"""This module defines and handles the model training process for the Listen, Attend, and Spell (LAS) model.

It includes utilities to set up the training environment, manage callbacks,
and run the training loop.
"""

import json
import os
from typing import List

import tensorflow as tf

from VocaLinux.configs import training as training_config
from VocaLinux.model.las_model import LASModel
from VocaLinux.model.training.callbacks import (
    CyclicalLearningRateCallback,
    HistoryCallback,
    ScheduledSamplingCallback,
)


def setup_callbacks(
    initial_epoch: int,
    history_save_path: str,
    checkpoint_dir: str,
    log_dir: str,
) -> List[tf.keras.callbacks.Callback]:
    """Configures and returns a list of Keras callbacks for training."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks: List[tf.keras.callbacks.Callback] = []

    history_callback = HistoryCallback(filepath=history_save_path)
    if initial_epoch > 0 and os.path.exists(history_save_path):
        try:
            with open(history_save_path, "r") as f:
                loaded_history_data = json.load(f)
                history_callback.history = loaded_history_data.get("epochs", [])
        except (IOError, json.JSONDecodeError) as e:
            print(f"Could not load previous history: {e}")
    callbacks.append(history_callback)

    if training_config.USE_SCHEDULED_SAMPLING:
        callbacks.append(
            ScheduledSamplingCallback(
                start_prob=training_config.SCHEDULED_SAMPLING_START_PROB,
                end_prob=training_config.SCHEDULED_SAMPLING_END_PROB,
                ramp_epochs=training_config.SCHEDULED_SAMPLING_RAMP_EPOCHS,
            )
        )

    checkpoint_filepath = os.path.join(checkpoint_dir, "model_epoch_{epoch:04d}.keras")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    callbacks.append(model_checkpoint_callback)

    if training_config.USE_CYCLICAL_LR:
        clr_save_path = os.path.join(log_dir, "clr_state.json")
        clr_callback = CyclicalLearningRateCallback(
            min_lr=training_config.CYCLICAL_LR_MIN_LR,
            max_lr=training_config.CYCLICAL_LR_MAX_LR,
            lr_decay=training_config.CYCLICAL_LR_DECAY,
            cycle_length=training_config.CYCLICAL_LR_CYCLE_LENGTH,
            mult_factor=training_config.CYCLICAL_LR_MULT_FACTOR,
            save_file=clr_save_path,
            load_file=clr_save_path if initial_epoch > 0 else None,
        )
        callbacks.append(clr_callback)

    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    return callbacks


def train_model(
    model: LASModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    initial_epoch: int,
    epochs: int,
    history_save_path: str,
    checkpoint_dir: str,
    log_dir: str,
) -> tf.keras.callbacks.History:
    """Initiates and manages the training process for the LAS model."""
    callbacks = setup_callbacks(initial_epoch, history_save_path, checkpoint_dir, log_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # Save the final model state
    final_model_path = os.path.join(checkpoint_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    return history
