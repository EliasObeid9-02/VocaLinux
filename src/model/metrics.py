import jiwer
import tensorflow as tf

from model.vocabulary import ids_to_text


class CharacterErrorRate(tf.keras.metrics.Metric):
    """Computes Character Error Rate (CER) during training/validation."""

    def __init__(self, name="cer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cer = self.add_weight(name="total_cer", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true_ids: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """
        Args:
            y_true_ids: Ground truth token IDs (batch_size, seq_len)
            y_pred: Predicted IDs probailities (batch_size, seq_len)
        """
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        # Compute CER without using .numpy() on symbolic tensors
        def compute_cer(true_ids, pred_ids):
            true_text = ids_to_text(true_ids.numpy())
            pred_text = ids_to_text(pred_ids.numpy())
            return jiwer.cer(true_text, pred_text)

        cer_scores = tf.map_fn(
            lambda x: tf.py_function(compute_cer, [x[0], x[1]], tf.float32),
            (y_true_ids, y_pred_ids),
            fn_output_signature=tf.float32,
        )
        self.cer.assign_add(tf.reduce_sum(cer_scores))
        self.count.assign_add(tf.cast(tf.shape(y_true_ids)[0], tf.float32))

    def result(self):
        return self.cer / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.cer.assign(0.0)
        self.count.assign(0.0)


class WordErrorRate(tf.keras.metrics.Metric):
    """Computes Word Error Rate (WER) during training/validation."""

    def __init__(self, name="wer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.wer = self.add_weight(name="total_wer", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true_ids: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """
        Args:
            y_true_ids: Ground truth token IDs (batch_size, seq_len)
            y_pred: Predicted IDs probailities (batch_size, seq_len)
        """
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        # Compute WER without using .numpy() on symbolic tensors
        def compute_wer(true_ids, pred_ids):
            true_text = ids_to_text(true_ids.numpy())
            pred_text = ids_to_text(pred_ids.numpy())
            return jiwer.wer(true_text, pred_text)

        wer_scores = tf.map_fn(
            lambda x: tf.py_function(compute_wer, [x[0], x[1]], tf.float32),
            (y_true_ids, y_pred_ids),
            fn_output_signature=tf.float32,
        )
        self.wer.assign_add(tf.reduce_sum(wer_scores))
        self.count.assign_add(tf.cast(tf.shape(y_true_ids)[0], tf.float32))

    def result(self):
        return self.wer / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.wer.assign(0.0)
        self.count.assign(0.0)
