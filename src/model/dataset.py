import os
import random
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf

from model.vocabulary import CHAR_TO_ID


class LibriSpeechDatasetLoader:
    """
    A class to load and preprocess the LibriSpeech dataset using TensorFlow and SoundFile.

    The dataset is structured as:
    data_root/
    ├── split_name/ (e.g., dev-clean, test-clean, train-clean-100, train-clean-360)
    │   ├── speaker_id/
    │   │   ├── chapter_id/
    │   │   │   ├── speaker_id-chapter_id-utterance_id.flac
    │   │   │   └── speaker_id-chapter_id.trans.txt
    """

    def __init__(self, data_root: str, batch_size: int):
        """
        Initializes the LibriSpeechDatasetLoader.

        Args:
            data_root (str): The root directory where the LibriSpeech dataset is located.
            batch_size (int): The number of samples per batch.
        """
        if not os.path.isdir(data_root):
            raise ValueError(f"Data root directory not found: {data_root}")

        self.data_root = data_root
        self.batch_size = batch_size
        self.sample_rate = 16000

        # Configuration parameters
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 100

        # Pre-calculate linear_to_mel_matrix once, as sample_rate and n_fft are constant
        lower_edge_hertz = 80.0
        upper_edge_hertz = 7600.0

        # num_spectrogram_bins should be n_fft // 2 + 1 for Mel filterbank
        self._linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        )

    def _load_transcript_file(self, transcript_path: str) -> Dict[str, str]:
        """
        Loads and parses a .trans.txt file.

        Args:
            transcript_path (str): The full path to the .trans.txt file.

        Returns:
            Dict[str, str]: A dictionary mapping audio filenames (without extension)
                            to their corresponding transcripts.
        """
        transcripts = {}
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        # Filename in the transcript doesn't contain the flac extension
                        transcripts[parts[0]] = parts[1].lower()
        except FileNotFoundError:
            print(f"Warning: Transcript file not found: {transcript_path}")
        except Exception as e:
            print(f"Error reading transcript file {transcript_path}: {e}")
        return transcripts

    def _get_audio_transcript_paths(self, split: str) -> List[Tuple[str, str]]:
        """
        Recursively finds all .flac audio files and their corresponding transcripts
        within a specified data split.

        Args:
            split (str): The name of the data split (e.g., "dev-clean").

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains
                                    (full_audio_path, transcript_text).
        """
        split_path = os.path.join(self.data_root, split)
        if not os.path.isdir(split_path):
            raise ValueError(f"Split directory not found: {split_path}")

        all_data_paths = []
        for speaker_id_dir in os.listdir(split_path):
            speaker_path = os.path.join(split_path, speaker_id_dir)
            if not os.path.isdir(speaker_path):
                continue

            for chapter_id_dir in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_id_dir)
                if not os.path.isdir(chapter_path):
                    continue

                transcript_filename = f"{speaker_id_dir}-{chapter_id_dir}.trans.txt"
                transcript_filepath = os.path.join(chapter_path, transcript_filename)

                chapter_transcripts = self._load_transcript_file(transcript_filepath)

                for filename in os.listdir(chapter_path):
                    if filename.endswith(".flac"):
                        audio_id = filename.replace(".flac", "")
                        if audio_id in chapter_transcripts:
                            audio_path = os.path.join(chapter_path, filename)
                            transcript_text = chapter_transcripts[audio_id]
                            all_data_paths.append((audio_path, transcript_text))
                        else:
                            print(
                                f"Warning: Transcript not found for audio {audio_id} in {transcript_filepath}"
                            )
        return all_data_paths

    @tf.function(input_signature=[tf.TensorSpec((None,), tf.float32)])
    def _audio_to_mel_spectrogram(self, audio: tf.Tensor) -> tf.Tensor:
        """
        Converts a raw audio waveform into a Mel Spectrogram.

        Args:
            audio (tf.Tensor): The raw audio waveform tensor.

        Returns:
            tf.Tensor: The Mel Spectrogram of the audio.
        """
        # Ensure audio is float32 and has a single channel for STFT
        audio = tf.cast(audio, tf.float32)
        if audio.shape.ndims == 0:  # Handle scalar case if py_function returns scalar
            audio = tf.expand_dims(audio, axis=0)  # Make it a 1D tensor
        elif audio.shape.ndims == 2:  # If it's (frames, channels), take the first channel
            audio = audio[:, 0]

        stft = tf.signal.stft(
            audio,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
        )
        spectrogram = tf.abs(stft)

        # Use the pre-calculated linear_to_mel_matrix
        mel_spectrogram = tf.tensordot(spectrogram, self._linear_to_mel_matrix, 1)
        mel_spectrogram.set_shape([None, self.n_mels])

        # Apply log for better feature representation
        # Add a small epsilon to avoid log(0)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram

    def _text_to_char_ids(self, text: str) -> np.ndarray:
        """
        Converts a transcript string into a tensor of character IDs.
        Adds <sos> and <eos> tokens, and uses <unk> for unknown characters.

        Args:
            text (str): The input transcript string.

        Returns:
            tf.Tensor: A 1D tensor of integer character IDs.
        """
        char_ids = [CHAR_TO_ID["<sos>"]]
        for char in text:
            char_ids.append(CHAR_TO_ID.get(char, CHAR_TO_ID["<unk>"]))
        char_ids.append(CHAR_TO_ID["<eos>"])
        return np.array(char_ids, dtype=np.int32)

    def _load_and_preprocess_sample(
        self, audio_path_tensor: tf.Tensor, transcript_text_tensor: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Loads an audio file and its transcript, converts audio to Mel Spectrogram,
        and transcript to character IDs. This function is designed to be mapped
        over a tf.data.Dataset.

        Args:
            audio_path_tensor (tf.Tensor): Tensor containing the path to the audio file.
            transcript_text_tensor (tf.Tensor): Tensor containing the transcript text.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the Mel Spectrogram tensor
                                         and the character IDs tensor.
        """

        def _py_function_wrapper(audio_path_str_tensor, transcript_text_str_tensor):
            # Decode tensors to Python strings
            audio_path = audio_path_str_tensor.numpy().decode("utf-8")
            transcript_text = transcript_text_str_tensor.numpy().decode("utf-8")

            audio_data = np.array([0.0], dtype=np.float32)
            char_ids_np = np.array([0], dtype=np.int32)

            try:
                audio_data_loaded, sample_rate_read = sf.read(audio_path, dtype="float32")
                if audio_data_loaded.ndim > 1:
                    audio_data_loaded = audio_data_loaded[:, 0]  # Take first channel for stereo
                if sample_rate_read != self.sample_rate:
                    print(
                        f"Warning: Sample rate mismatch for {audio_path}. Expected {self.sample_rate}, got {sample_rate_read}. Resampling not implemented."
                    )
                audio_data = audio_data_loaded

            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")
                # If loading fails, audio_data remains dummy, and will be padded/handled downstream.

            # Convert transcript text to character IDs using the existing helper
            char_ids_np = self._text_to_char_ids(transcript_text)
            return audio_data, char_ids_np

        audio_data_tensor, char_ids_tensor = tf.py_function(
            func=_py_function_wrapper,
            inp=[audio_path_tensor, transcript_text_tensor],
            Tout=[tf.float32, tf.int32],
        )

        # Set shapes for the outputs of py_function because tf.py_function returns
        # tensors with unknown shapes by default
        audio_data_tensor.set_shape([None])  # Raw audio is a 1D sequence (time domain)
        char_ids_tensor.set_shape([None])  # Character IDs are a 1D sequence

        # Mel Spectrogram shape: [frames, mel_bins] (frames are variable, mel_bins fixed by self.n_mels)
        mel_spec = self._audio_to_mel_spectrogram(audio_data_tensor)
        mel_spec.set_shape([None, self.n_mels])
        return mel_spec, char_ids_tensor

    def _create_configured_dataset(
        self, data_pairs: List[Tuple[str, str]], shuffle: bool
    ) -> tf.data.Dataset:
        """
        Internal helper method to create and configure a tf.data.Dataset from a list of data pairs.
        This handles mapping, batching, and padding, always returning in the
        ((mel_spec, true_char_ids), true_char_ids) format.

        Args:
            data_pairs (List[Tuple[str, str]]): List of (audio_path, transcript_text) tuples.
            shuffle (bool): Whether to shuffle the dataset.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            ([pair[0] for pair in data_pairs], [pair[1] for pair in data_pairs])
        )

        if shuffle:
            ds = ds.shuffle(buffer_size=len(data_pairs) if len(data_pairs) > 0 else 1024)
        ds = ds.map(self._load_and_preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)

        padding_values_tuple = (
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(CHAR_TO_ID["<pad>"], dtype=tf.int32),
        )

        # Format the dataset as ((mel_spec, char_ids_input), char_ids_label)
        output_padded_shapes = (
            (
                tf.TensorShape([None, self.n_mels]),
                tf.TensorShape([None]),
            ),
            tf.TensorShape([None]),
        )
        ds = ds.map(
            lambda mel, char_ids: ((mel, char_ids), char_ids),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        final_padding_values = (
            (padding_values_tuple[0], padding_values_tuple[1]),
            padding_values_tuple[1],
        )

        ds = ds.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=output_padded_shapes,
            padding_values=final_padding_values,
            drop_remainder=True,
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def get_dataset(self, split: str, shuffle: bool) -> tf.data.Dataset:
        """
        Loads and preprocesses a specified split of the LibriSpeech dataset,
        returning a single dataset in the ((mel_spec, true_char_ids), true_char_ids) format.

        Args:
            split (str): The name of the data split (e.g., "dev-clean").
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            tf.data.Dataset: A single tf.data.Dataset object, formatted as
                             ((mel_spec, true_char_ids), true_char_ids).
        """
        all_data_pairs = self._get_audio_transcript_paths(split)
        return self._create_configured_dataset(all_data_pairs, shuffle)

    def get_partitioned_datasets(
        self, split: str, partitions: List[float], shuffle: bool
    ) -> List[tf.data.Dataset]:
        """
        Loads and preprocesses a specified split of the LibriSpeech dataset,
        returning a list of datasets partitioned using a list of floating point numbers.
        All returned datasets are in the ((mel_spec, true_char_ids), true_char_ids) format.

        Args:
            split (str): The name of the data split (e.g., "train-clean-100").
            partitions (List[float]): A list of floats (0, 1] representing the percentage
                                      of the original split for each partition. Sum must be approx 1.0.
            shuffle (bool): Whether to shuffle each partitioned dataset.

        Returns:
            List[tf.data.Dataset]: A list of tf.data.Dataset objects.
        """
        if not all(isinstance(p, (int, float)) and 0 < p <= 1 for p in partitions):
            raise ValueError("Partitions must be a list of floats (0, 1] representing percentages.")
        if abs(sum(partitions) - 1.0) > 1e-6:  # Check if sum is approximately 1.0
            raise ValueError("Sum of partitions must be approximately 1.0.")

        all_data_pairs = self._get_audio_transcript_paths(split)
        total_samples = len(all_data_pairs)
        partitioned_datasets = []
        current_idx = 0

        # Shuffle the *entire* list of data pairs once before partitioning to ensure
        # that samples are randomly distributed across partitions.
        if shuffle:
            random.shuffle(all_data_pairs)

        for i, percentage in enumerate(partitions):
            num_samples_for_partition = int(total_samples * percentage)
            if i == len(partitions) - 1:
                # Ensure the last partition gets all remaining samples due to potential rounding
                num_samples_for_partition = total_samples - current_idx

            partition_data_pairs = all_data_pairs[
                current_idx : current_idx + num_samples_for_partition
            ]
            current_idx += num_samples_for_partition

            if not partition_data_pairs:
                print(
                    f"Warning: Partition {i + 1} ({percentage * 100:.1f}%) resulted in 0 samples. "
                    "Consider adjusting batch_size or partition percentages for small datasets."
                )
                # Return an empty but correctly shaped dataset if a partition has no samples
                empty_ds = self._create_configured_dataset([], shuffle=False)
                partitioned_datasets.append(empty_ds)
                continue
            partitioned_datasets.append(
                self._create_configured_dataset(partition_data_pairs, shuffle)
            )
        return partitioned_datasets
