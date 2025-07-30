import json
import time

import tensorflow as tf


class HistoryCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to store training progress at both epoch and batch levels.

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

    def __init__(self, batch_period: int = 100):
        super().__init__()
        self.history = []
        self.current_epoch_data = None
        self.total_training_time_seconds = 0.0
        self._epoch_start_time = 0.0
        self._total_training_start_time = 0.0
        self.batch_period = batch_period
        self.last_completed_epoch = -1

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training. Records the start time for total training duration.
        """
        self._total_training_start_time = time.time()

        print("Model training using model.fit() has started.\n")

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch. Initializes a new dictionary
        for the current epoch's data.
        """
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
        """
        Called at the end of training. Calculates the total training duration.
        """
        total_end_time = time.time()
        self.total_training_time_seconds = total_end_time - self._total_training_start_time

        print(
            f"Model training using model.fit() has ended. Total time: {self.total_training_time_seconds:.2f}s\n"
        )

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Populates the epoch-level metrics
        for the current epoch's data.
        """
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

        sampling_prob = f"{epoch_metrics['sampling_probability']:.4f}"
        loss = f"{epoch_metrics['loss']:.4f}"
        acc = f"{epoch_metrics['accuracy']:.4f}"
        cer = f"{epoch_metrics['cer']:.4f}"
        wer = f"{epoch_metrics['wer']:.4f}"

        output_str = f"Time used: {epoch_duration}, Sampling Probability: {sampling_prob}\n"
        output_str += "--- Training Metrics ---\n"
        output_str += f"\tLoss: {loss}, Accuracy: {acc}\n"
        output_str += f"\tCER:  {cer}, WER:      {wer}\n\n"

        if epoch_metrics["val_loss"] is not None:
            val_loss = f"{epoch_metrics['val_loss']:.4f}"
            val_acc = f"{epoch_metrics['val_accuracy']:.4f}"
            val_cer = f"{epoch_metrics['val_cer']:.4f}"
            val_wer = f"{epoch_metrics['val_wer']:.4f}"

            output_str += "--- Validation Metrics ---\n"
            output_str += f"\tLoss: {val_loss}, Accuracy: {val_acc}\n"
            output_str += f"\tCER:  {val_cer}, WER:      {val_wer}\n"
        print(output_str)
        print(f"--- End   Epoch {epoch} ---\n")

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch. Stores batch-level loss and accuracy.
        """
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

    def save_to_json(self, filepath: str):
        """
        Saves the collected history data to a JSON file.

        Args:
            filepath (str): The path to the file where the history should
                            be saved with JSON extention (e.g., "training_history.json").
        """
        data_to_save = {
            "epochs": self.history,
            "total_training_time_seconds": self.total_training_time_seconds,
            "last_completed_epoch": self.last_completed_epoch,
        }
        try:
            with open(filepath, "w") as f:
                json.dump(data_to_save, f, indent=4)
            print(f"History saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving history to {filepath}: {e}")


class ScheduledSamplingCallback(tf.keras.callbacks.Callback):
    """
    Keras Callback to linearly increase the sampling probability (probability of
    using model's own prediction) during training.
    """

    def __init__(
        self,
        start_prob: float = 0.05,
        end_prob: float = 0.8,
        warmup_epochs: int = 20,
        ramp_epochs: int = 260,
    ):
        super().__init__()
        self.current_prob = start_prob
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

        if not (0 <= self.start_prob <= 1 and 0 <= self.end_prob <= 1):
            raise ValueError("Probabilities must be between 0 and 1")
        # if self.start_prob > self.end_prob:
        #    raise ValueError("start_prob must be < end_prob")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.ramp_epochs <= 0:
            raise ValueError("ramp_epochs must be > 0")

    def on_epoch_begin(self, epoch, logs=None):
        """Update model's sampling probability"""
        progress = min(1.0, (epoch - self.warmup_epochs) / self.ramp_epochs)
        increment = (self.end_prob - self.start_prob) / self.ramp_epochs

        self.current_prob = min(self.current_prob + increment, self.end_prob)
        new_prob = max(self.model.speller.sampling_probability, self.current_prob)
        self.model.speller.sampling_probability = new_prob
