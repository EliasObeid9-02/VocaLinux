if evaluate_mode:
    print("--- Testing Results ---")
    res = model.evaluate(
        test_ds,
        return_dict=True,
        verbose=0,
    )

    loss = f"{res.get('loss'):.4f}"
    accuracy = f"{res.get('accuracy'):.4f}"
    cer = f"{res.get('cer'):.4f}"
    wer = f"{res.get('wer'):.4f}"
    print(f"Loss: {loss}, Accuracy: {accuracy}, CER: {cer}, WER: {wer}\n")

    print("Output for the first ten audio samples:")
    for inputs, outputs in test_ds.take(1):
        non_sampled_predictions = model.predict_on_batch(inputs)
        non_sampled_chars = tf.argmax(non_sampled_predictions, axis=-1, output_type=tf.int32)
        sampled_predictions = model(inputs, training=True)
        sampled_chars = tf.argmax(sampled_predictions, axis=-1, output_type=tf.int32)

        mel_spectrograms, _ = inputs
        greedy_chars = model.greedy_predict(mel_spectrograms)
        beam_search_chars = model.beam_search_predict(mel_spectrograms)

        printing_count = 16
        for i in range(printing_count):
            true_text = ids_to_text(outputs[i])
            sampled_text = ids_to_text(sampled_chars[i])
            non_sampled_text = ids_to_text(non_sampled_chars[i])
            greedy_text = ids_to_text(greedy_chars[i])
            beam_search_text = ids_to_text(beam_search_chars[i])

            print(f"--- Sample {i + 1} ---")
            print(f"True Output:        '{true_text}'")
            print(f"Sampled Output:     '{sampled_text}'")
            print(f"Non Sampled Output: '{non_sampled_text}'")
            print(f"Greedy Prediction:  '{greedy_text}'")
            print(f"Beam Search (k=32): '{beam_search_text}'\n")
else:
    print("Model is not in evaluation mode.")
