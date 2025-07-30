import tensorflow as tf

from model.vocabulary import PAD_TOKEN, VOCAB_SIZE


@tf.keras.utils.register_keras_serializable()
def safe_sparse_categorical_crossentropy(y_true, y_pred_softmax_output):
    """
    Computes Sparse Categorical Crossentropy for predictions that are already softmax outputs.

    Args:
        y_true: True labels (integer IDs). Shape: (batch_size, sequence_length)
        y_pred_softmax_output: Softmax outputs from the model. Shape: (batch_size, sequence_length, vocab_size)

    Returns:
        A scalar loss value.
    """
    tf.debugging.check_numerics(y_pred_softmax_output, "y_pred_softmax_output contains NaN or Inf")

    # Ensure y_true is int32 for tf.gather
    y_true_int = tf.cast(y_true, tf.int32)

    # IMPORTANT: Check for valid token IDs
    y_true_min = tf.reduce_min(y_true_int)
    y_true_max = tf.reduce_max(y_true_int)
    tf.debugging.assert_greater_equal(y_true_min, tf.constant(0, dtype=y_true_int.dtype))
    tf.debugging.assert_less(y_true_max, tf.constant(VOCAB_SIZE, dtype=y_true_int.dtype))

    # Convert softmax probabilities to log probabilities for numerical stability in cross-entropy.
    # Add a small epsilon to avoid log(0) if any probabilities are exactly zero.
    y_pred_log_probs = tf.math.log(y_pred_softmax_output + 1e-10)  # Added epsilon

    # Get the log-probability of the true class for each time step and each sample in the batch
    true_class_log_probs = tf.gather(
        y_pred_log_probs,  # Use the log probabilities here
        y_true_int,
        batch_dims=2,  # This is crucial: (batch, seq_len) indexing into (batch, seq_len, vocab_size)
    )

    # Cross-entropy is -sum(p * log(q)). Here, p=1 for true class, p=0 for others.
    # So it simplifies to -log(q_true_class)
    negative_log_likelihood = -true_class_log_probs

    # Mask out padding tokens
    mask = tf.math.not_equal(y_true_int, PAD_TOKEN)
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
