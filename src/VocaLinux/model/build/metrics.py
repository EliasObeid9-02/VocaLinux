"""Custom Keras metrics for evaluating the Listen, Attend, and Spell (LAS) model.

This module provides implementations for Character Error Rate (CER) and Word Error Rate (WER),
which are crucial for assessing the performance of speech recognition models.
"""

import jiwer
import tensorflow as tf

from VocaLinux.model.vocabulary import ids_to_text


class CharacterErrorRate(tf.keras.metrics.Metric):
    """Computes Character Error Rate (CER) during training/validation."""

    def __init__(self, name: str = "cer", **kwargs) -> None:
        """Initializes the CharacterErrorRate metric.

        Args:
            name (str): Name of the metric. Defaults to "cer".
            **kwargs: Additional keyword arguments for the Metric base class.
        """
        super().__init__(name=name, **kwargs)
        self.cer = self.add_weight(name="total_cer", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Updates the state of the Character Error Rate metric.

        Args:
            y_true (tf.Tensor): Ground truth token IDs. Shape: (batch_size, seq_len)
            y_pred (tf.Tensor): Predicted token probabilities. Shape: (batch_size, seq_len, vocab_size)
            sample_weight (tf.Tensor | None): Optional sample weights. Not used in this metric.
        """
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        def compute_cer(true_ids: tf.Tensor, pred_ids: tf.Tensor) -> tf.Tensor:
            """Computes CER for a single sample using numpy operations."""
            true_text = ids_to_text(true_ids.numpy())
            pred_text = ids_to_text(pred_ids.numpy())
            return tf.constant(jiwer.cer(true_text, pred_text), dtype=tf.float32)

        cer_scores = tf.map_fn(
            lambda x: tf.py_function(compute_cer, [x[0], x[1]], tf.float32),
            (y_true, y_pred_ids),
            fn_output_signature=tf.float32,
        )
        self.cer.assign_add(tf.reduce_sum(cer_scores))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self) -> tf.Tensor:
        """Computes and returns the current Character Error Rate.

        Returns:
            tf.Tensor: The current CER value.
        """
        return self.cer / tf.maximum(self.count, 1.0)

    def reset_state(self) -> None:
        """Resets the metric's state variables back to their initial values."""
        self.cer.assign(0.0)
        self.count.assign(0.0)


class WordErrorRate(tf.keras.metrics.Metric):
    """Computes Word Error Rate (WER) during training/validation."""

    def __init__(self, name: str = "wer", **kwargs) -> None:
        """Initializes the WordErrorRate metric.

        Args:
            name (str): Name of the metric. Defaults to "wer".
            **kwargs: Additional keyword arguments for the Metric base class.
        """
        super().__init__(name=name, **kwargs)
        self.wer = self.add_weight(name="total_wer", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Updates the state of the Word Error Rate metric.

        Args:
            y_true (tf.Tensor): Ground truth token IDs. Shape: (batch_size, seq_len)
            y_pred (tf.Tensor): Predicted token probabilities. Shape: (batch_size, seq_len, vocab_size)
            sample_weight (tf.Tensor | None): Optional sample weights. Not used in this metric.
        """
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        def compute_wer(true_ids: tf.Tensor, pred_ids: tf.Tensor) -> tf.Tensor:
            """Computes WER for a single sample using numpy operations."""
            true_text = ids_to_text(true_ids.numpy())
            pred_text = ids_to_text(pred_ids.numpy())
            return tf.constant(jiwer.wer(true_text, pred_text), dtype=tf.float32)

        wer_scores = tf.map_fn(
            lambda x: tf.py_function(compute_wer, [x[0], x[1]], tf.float32),
            (y_true, y_pred_ids),
            fn_output_signature=tf.float32,
        )
        self.wer.assign_add(tf.reduce_sum(wer_scores))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self) -> tf.Tensor:
        """Computes and returns the current Word Error Rate.

        Returns:
            tf.Tensor: The current WER value.
        """
        return self.wer / tf.maximum(self.count, 1.0)

    def reset_state(self) -> None:
        """Resets the metric's state variables back to their initial values."""
        self.wer.assign(0.0)
        self.count.assign(0.0)
