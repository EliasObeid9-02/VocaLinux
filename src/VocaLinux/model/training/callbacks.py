import json
import os
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from VocaLinux.configs import training as training_config


class HistoryCallback(tf.keras.callbacks.Callback):
    """A custom Keras callback to store training progress at both epoch and batch levels.

    Attributes:
        history (list): A list of dictionaries, where each dictionary represents an epoch.
                        Each epoch dictionary contains:
                        - 'epoch': The epoch number (0-indexed).
                        - 'epoch_metrics': A dictionary of epoch-level metrics (loss, accuracy, val_loss, val_accuracy).
                        - 'batch_metrics': A list of dictionaries, where each inner dictionary
                                           contains batch-level metrics (batch, loss, accuracy).
        current_epoch_data (dict): A reference to the dictionary for the currently
                                   active epoch in `self.history`.
    """

    def __init__(
        self,
        filepath: str,
        batch_period: int = training_config.BATCH_PERIOD,
    ):
        super().__init__()
        self.filepath = filepath
        self.history = []
        self.current_epoch_data = None
        self.total_training_time_seconds = 0.0
        self._epoch_start_time = 0.0
        self._total_training_start_time = 0.0
        self.batch_period = batch_period
        self.last_completed_epoch = -1

    def on_train_begin(self, logs=None):
        """Called at the beginning of training. Records the start time for total training duration."""
        self._total_training_start_time = time.time()

        print("Model training using model.fit() has started.\n")

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch. Initializes a new dictionary for the current epoch's data."""
        self._epoch_start_time = time.time()
        self.current_epoch_data = {
            "epoch": epoch,
            "epoch_metrics": {
                "loss": None,
                "accuracy": None,
                "cer": None,
                "wer": None,
                "val_loss": None,
                "val_accuracy": None,
                "val_cer": None,
                "val_wer": None,
                "sampling_probability": None,
            },
            "batch_metrics": [],
        }
        self.history.append(self.current_epoch_data)

        print(f"--- Start Epoch {epoch} ---")

    def on_train_end(self, logs=None):
        """Called at the end of training. Calculates the total training duration and saves the history."""
        self.total_training_time_seconds = time.time() - self._total_training_start_time
        self._save_history()
        print(
            f"Model training using model.fit() has ended. Total time: {self.total_training_time_seconds:.2f}s\n"
        )

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch. Populates the epoch-level metrics for the current epoch's data."""
        epoch_end_time = time.time()
        epoch_duration = f"{epoch_end_time - self._epoch_start_time:.2f}s"
        self.last_completed_epoch = epoch

        logs = logs or {}
        epoch_metrics = self.current_epoch_data["epoch_metrics"]
        epoch_metrics["sampling_probability"] = self.model.speller.sampling_probability

        epoch_metrics["loss"] = logs.get("loss")
        epoch_metrics["accuracy"] = logs.get("accuracy")
        epoch_metrics["cer"] = logs.get("cer")
        epoch_metrics["wer"] = logs.get("wer")

        epoch_metrics["val_loss"] = logs.get("val_loss")
        epoch_metrics["val_accuracy"] = logs.get("val_accuracy")
        epoch_metrics["val_cer"] = logs.get("val_cer")
        epoch_metrics["val_wer"] = logs.get("val_wer")

        self.current_epoch_data["epoch_metrics"] = epoch_metrics

        loss = f"{epoch_metrics['loss']:.4f}"
        acc = f"{epoch_metrics['accuracy']:.4f}"
        cer = f"{epoch_metrics['cer']:.4f}"
        wer = f"{epoch_metrics['wer']:.4f}"

        output_str = f"Time used: {epoch_duration}\n"
        output_str += "--- Training Metrics ---"
        output_str += f"\tLoss: {loss}, Accuracy: {acc}\n"
        output_str += f"\tCER:  {cer}, WER:      {wer}\n\n"

        if epoch_metrics["val_loss"] is not None:
            val_loss = f"{epoch_metrics['val_loss']:.4f}"
            val_acc = f"{epoch_metrics['val_accuracy']:.4f}"
            val_cer = f"{epoch_metrics['val_cer']:.4f}"
            val_wer = f"{epoch_metrics['val_wer']:.4f}"

            output_str += "--- Validation Metrics ---"
            output_str += f"\tLoss: {val_loss}, Accuracy: {val_acc}\n"
            output_str += f"\tCER:  {val_cer}, WER:      {val_wer}"
        print(output_str)
        print(f"--- End   Epoch {epoch} ---\n")

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of each training batch. Stores batch-level loss and accuracy."""
        logs = logs or {}
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        cer = logs.get("cer")
        wer = logs.get("wer")
        batch_info = {
            "batch": batch,
            "loss": loss,
            "accuracy": acc,
            "cer": cer,
            "wer": wer,
        }
        if self.current_epoch_data is not None:
            self.current_epoch_data["batch_metrics"].append(batch_info)

        if (batch + 1) % self.batch_period == 0:
            print(
                f"Processed {batch + 1} batches - Loss: {loss:.4f}, Accuracy: {acc:.4f}, CER: {cer:.4f}, WER: {wer:.4f}"
            )

    def _save_history(self):
        """Saves the collected history data to the filepath."""
        data_to_save = {
            "epochs": self.history,
            "total_training_time_seconds": self.total_training_time_seconds,
            "last_completed_epoch": self.last_completed_epoch,
        }
        try:
            with open(self.filepath, "w") as f:
                json.dump(data_to_save, f, indent=4)
            print(f"History saved successfully to {self.filepath}")
        except Exception as e:
            print(f"Error saving history to {self.filepath}: {e}")


class ScheduledSamplingCallback(tf.keras.callbacks.Callback):
    """Keras Callback to linearly increase the sampling probability (probability of using model's own prediction) during training."""

    def __init__(
        self,
        start_prob: float = training_config.SCHEDULED_SAMPLING_START_PROB,
        end_prob: float = training_config.SCHEDULED_SAMPLING_END_PROB,
        ramp_epochs: int = training_config.SCHEDULED_SAMPLING_RAMP_EPOCHS,
    ):
        super().__init__()
        self.current_prob = start_prob
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.ramp_epochs = ramp_epochs

        if not (0 <= self.start_prob <= 1 and 0 <= self.end_prob <= 1):
            raise ValueError("Probabilities must be between 0 and 1")
        if self.ramp_epochs <= 0:
            raise ValueError("ramp_epochs must be > 0")

    def on_epoch_begin(self, epoch, logs=None):
        """Update model's sampling probability, handling both ramp-up and ramp-down."""

        # This increment will be positive for ramp-up and negative for ramp-down.
        increment = (self.end_prob - self.start_prob) / self.ramp_epochs
        current_prob = self.model.speller.sampling_probability
        next_prob = current_prob + increment

        if increment > 0:
            final_prob = min(next_prob, self.end_prob)
        else:
            final_prob = max(next_prob, self.end_prob)

        self.model.speller.sampling_probability = final_prob
        print(f"Sampling Probability: {final_prob:.4f}")


class CyclicalLearningRateCallback(tf.keras.callbacks.Callback):
    """
    Stateful Cosine Annealing learning rate scheduler that can save and
    load its state to continue across multiple, separate training runs.
    """

    def __init__(
        self,
        min_lr: float = training_config.CYCLICAL_LR_MIN_LR,
        max_lr: float = training_config.CYCLICAL_LR_MAX_LR,
        lr_decay: int = training_config.CYCLICAL_LR_DECAY,
        cycle_length: int = training_config.CYCLICAL_LR_CYCLE_LENGTH,
        mult_factor: int = training_config.CYCLICAL_LR_MULT_FACTOR,
        load_file: Optional[str] = None,
        save_file="clr_state.json",
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.load_file = load_file
        self.save_file = save_file

        # Default state values
        self.batch_since_restart = 0
        self.next_restart = self.cycle_length
        self.history = {}

    def save_state(self):
        """Saves the critical state attributes to the save_file."""
        state = {
            "max_lr": self.max_lr,
            "batch_since_restart": self.batch_since_restart,
            "next_restart": self.next_restart,
            "cycle_length": self.cycle_length,
        }
        with open(self.save_file, "w") as f:
            json.dump(state, f, indent=4)
        print(f"CLR state saved to {self.save_file}")

    def load_state(self):
        """Loads state from the load_file if it exists."""
        if self.load_file and os.path.exists(self.load_file):
            with open(self.load_file, "r") as f:
                state = json.load(f)
                self.__dict__.update(state)
                print(f"CLR state loaded from {self.load_file}. Resuming schedule.")
        else:
            print("No CLR state load file specified or found. Starting a new schedule.")

    def clr(self):
        """Calculate the learning rate."""
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(fraction_to_restart * np.pi)
        )
        return lr

    def on_train_begin(self, logs={}):
        """Initialize the learning rate."""
        self.steps_per_epoch = (
            self.params["steps"]
            if self.params["steps"] is not None
            else round(self.params["samples"] / self.params["batch_size"])
        )
        self.load_state()
        self.model.optimizer.learning_rate.assign(self.clr())

    def on_batch_end(self, batch, logs={}):
        """Record stats and update learning rate."""
        self.history.setdefault("lr", []).append(self.model.optimizer.learning_rate.numpy())
        self.batch_since_restart += 1
        self.model.optimizer.learning_rate.assign(self.clr())

    def on_epoch_end(self, epoch, logs={}):
        """Check for end of cycle."""
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay

    def on_train_end(self, logs={}):
        """Save the callback's state for the next run."""
        self.save_state()
