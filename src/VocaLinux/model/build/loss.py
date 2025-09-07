"""Custom loss functions for the Listen, Attend, and Spell (LAS) model."""

import tensorflow as tf

from VocaLinux.model.vocabulary import Vocabulary

vocabulary = Vocabulary()


@tf.keras.utils.register_keras_serializable()
def safe_sparse_categorical_crossentropy(
    y_true: tf.Tensor, y_pred_softmax_output: tf.Tensor
) -> tf.Tensor:
    """Computes Sparse Categorical Crossentropy for predictions that are already softmax outputs.

    Masks out padding tokens from the loss calculation.

    Args:
        y_true: True labels (integer IDs). Shape: (batch_size, sequence_length)
        y_pred_softmax_output: Softmax outputs from the model.
            Shape: (batch_size, sequence_length, vocab_size)

    Returns:
        A scalar loss value.
    """
    tf.debugging.check_numerics(y_pred_softmax_output, "y_pred_softmax_output contains NaN or Inf")
    y_true_int = tf.cast(y_true, tf.int32)

    # Convert softmax probabilities to log probabilities for numerical stability.
    y_pred_log_probs = tf.math.log(y_pred_softmax_output + 1e-10)
    true_class_log_probs = tf.gather(
        y_pred_log_probs,
        y_true_int,
        batch_dims=2,
    )

    negative_log_likelihood = -true_class_log_probs

    # Mask out padding tokens
    mask = tf.math.not_equal(y_true_int, vocabulary.PAD_TOKEN)
    mask = tf.cast(mask, dtype=negative_log_likelihood.dtype)
    masked_loss = negative_log_likelihood * mask

    # Sum over the masked values and normalize by the number of non-padded tokens
    sum_mask = tf.reduce_sum(mask)
    loss = tf.cond(
        tf.equal(sum_mask, 0),
        lambda: tf.constant(0.0, dtype=masked_loss.dtype),
        lambda: tf.reduce_sum(masked_loss) / sum_mask,
    )
    return loss
