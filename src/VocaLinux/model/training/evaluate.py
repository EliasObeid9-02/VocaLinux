"""This module handles the evaluation of the trained LAS model."""

import os

import tensorflow as tf

from VocaLinux.model.las_model import LASModel
from VocaLinux.model.vocabulary import Vocabulary


def evaluate_model(model: LASModel, test_ds: tf.data.Dataset, output_dir: str) -> None:
    """Evaluates the model on a single batch from the test dataset, comparing
    different decoding strategies, and saves the output to a file.

    Args:
        model (LASModel): The trained LAS model.
        test_ds (tf.data.Dataset): The test dataset.
        output_dir (str): The directory to save the sample outputs file.
    """
    print("Running evaluation on a sample batch...")

    # Get the first batch from the test dataset
    for sample_batch in test_ds.take(1):
        mel_spectrograms, target_sequences = sample_batch
        break
    else:
        print("Could not get a sample batch from the test dataset.")
        return

    # --- Perform Predictions ---
    # 1. Sampled prediction (teacher forcing is off during inference)
    # We need a dummy decoder input for the speller, matching the batch size.
    # The speller's internal logic will handle the rest.
    batch_size = tf.shape(mel_spectrograms)[0]
    dummy_decoder_input = tf.zeros((batch_size, 1), dtype=tf.int32)
    sampled_logits = model([mel_spectrograms, dummy_decoder_input], training=False)
    sampled_predictions = tf.argmax(sampled_logits, axis=-1, output_type=tf.int32)

    # 2. Greedy prediction
    greedy_predictions = model.greedy_predict(mel_spectrograms)

    # 3. Beam search prediction
    beam_search_predictions = model.beam_search_predict(mel_spectrograms)

    # --- Decode and Save Outputs ---
    vocab = Vocabulary()
    output_lines = []

    for i in range(batch_size):
        true_text = vocab.ids_to_text(target_sequences[i].numpy())
        sampled_text = vocab.ids_to_text(sampled_predictions[i].numpy())
        greedy_text = vocab.ids_to_text(greedy_predictions[i].numpy())
        beam_text = vocab.ids_to_text(beam_search_predictions[i].numpy())

        output_lines.append(f"--- Sample {i+1} ---")
        output_lines.append(f"Ground Truth: {true_text}")
        output_lines.append(f"Sampled Pred: {sampled_text}")
        output_lines.append(f"Greedy Pred:  {greedy_text}")
        output_lines.append(f"Beam Pred:    {beam_text}")
        output_lines.append("\n")

    output_path = os.path.join(output_dir, "sample_outputs.txt")
    try:
        with open(output_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"Sample outputs saved to {output_path}")
    except IOError as e:
        print(f"Error saving sample outputs to {output_path}: {e}")

