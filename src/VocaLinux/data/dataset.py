import os
import random
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf

from VocaLinux.model.vocabulary import CHAR_TO_ID
from VocaLinux.configs import dataset as dataset_config


class LibriSpeechDatasetLoader:
    """
    A class to load and preprocess the LibriSpeech dataset using TensorFlow and SoundFile.
    Includes on-the-fly, mutually exclusive SpecAugment functionality based on user-defined partitions.
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
        self.sample_rate = dataset_config.SAMPLE_RATE
        self.n_fft = dataset_config.N_FFT
        self.hop_length = dataset_config.HOP_LENGTH
        self.n_mels = dataset_config.N_MELS

        self._linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=dataset_config.LOWER_EDGE_HERTZ,
            upper_edge_hertz=dataset_config.UPPER_EDGE_HERTZ,
        )

    def _validate_aug_partitions(self, partitions: List[float]):
        """Validates the user-provided augmentation partition list."""
        if not isinstance(partitions, list) or len(partitions) != 3:
            raise ValueError("augmentation_partitions must be a list of three floats.")
        if not all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in partitions):
            raise ValueError(
                "All values in augmentation_partitions must be floats between 0.0 and 1.0."
            )
        if sum(partitions) > 1.0:
            raise ValueError("The sum of augmentation_partitions must not exceed 1.0.")

    @tf.function
    def _time_warping(self, mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Applies time warping to a Mel Spectrogram."""
        W = dataset_config.AUGMENTATION_PARAMS["W"]
        time_steps = tf.shape(mel_spectrogram)[0]
        freq_bins = tf.shape(mel_spectrogram)[1]

        if time_steps <= 2 * W or W == 0:
            return mel_spectrogram

        w0 = tf.random.uniform(shape=(), minval=W, maxval=time_steps - W, dtype=tf.int32)
        w = tf.random.uniform(shape=(), minval=-W, maxval=W, dtype=tf.int32)

        part1 = mel_spectrogram[:w0, :]
        part_to_warp = mel_spectrogram[w0 : w0 + W, :]
        part3 = mel_spectrogram[w0 + W :, :]

        part_to_warp_4d = tf.expand_dims(tf.expand_dims(part_to_warp, 0), -1)
        new_width = tf.cast(W, tf.int32) + w
        if new_width <= 0:
            return mel_spectrogram

        warped_slice_4d = tf.image.resize(part_to_warp_4d, [new_width, freq_bins])
        warped_slice = tf.squeeze(warped_slice_4d, [0, -1])

        concatenated = tf.concat([part1, warped_slice, part3], axis=0)
        concatenated_4d = tf.expand_dims(tf.expand_dims(concatenated, 0), -1)
        final_warped_4d = tf.image.resize(concatenated_4d, [time_steps, freq_bins])

        return tf.squeeze(final_warped_4d, [0, -1])

    @tf.function
    def _frequency_masking(self, mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Applies frequency masking to a Mel Spectrogram."""
        F = dataset_config.AUGMENTATION_PARAMS["F"]
        m_F = dataset_config.AUGMENTATION_PARAMS["m_F"]
        freq_bins = tf.shape(mel_spectrogram)[1]
        mean_value = tf.reduce_mean(mel_spectrogram)

        for _ in range(m_F):
            f = tf.random.uniform(shape=(), minval=0, maxval=F, dtype=tf.int32)
            f0 = tf.random.uniform(shape=(), minval=0, maxval=freq_bins - f, dtype=tf.int32)
            mask = tf.one_hot(tf.range(f0, f0 + f), depth=freq_bins, on_value=0.0, off_value=1.0)
            mask = tf.reduce_prod(mask, axis=0)
            mask = tf.expand_dims(mask, 0)
            mel_spectrogram = mel_spectrogram * mask + (1.0 - mask) * mean_value
        return mel_spectrogram

    @tf.function
    def _time_masking(self, mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Applies time masking to a Mel Spectrogram."""
        T = dataset_config.AUGMENTATION_PARAMS["T"]
        p = dataset_config.AUGMENTATION_PARAMS["p"]
        m_T = dataset_config.AUGMENTATION_PARAMS["m_T"]
        time_steps = tf.shape(mel_spectrogram)[0]
        mean_value = tf.reduce_mean(mel_spectrogram)

        for _ in range(m_T):
            max_mask_width = tf.cast(tf.cast(time_steps, tf.float32) * p, tf.int32)
            t = tf.random.uniform(
                shape=(), minval=0, maxval=tf.minimum(T, max_mask_width), dtype=tf.int32
            )
            t0 = tf.random.uniform(shape=(), minval=0, maxval=time_steps - t, dtype=tf.int32)
            mask = tf.one_hot(tf.range(t0, t0 + t), depth=time_steps, on_value=0.0, off_value=1.0)
            mask = tf.reduce_prod(mask, axis=0)
            mask = tf.expand_dims(mask, 1)
            mel_spectrogram = mel_spectrogram * mask + (1.0 - mask) * mean_value
        return mel_spectrogram

    @tf.function
    def _apply_augmentations(
        self, mel_spectrogram: tf.Tensor, char_ids: tf.Tensor, partitions: List[float]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies a single, randomly selected augmentation based on partition probabilities."""
        p_warp = tf.cast(partitions[0], tf.float32)
        p_freq = tf.cast(partitions[1], tf.float32)
        p_time = tf.cast(partitions[2], tf.float32)

        random_p = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

        # Use tf.case for mutually exclusive augmentations.
        # The default case is to return the spectrogram unmodified (clean).
        augmented_mel_spec = tf.case(
            [
                (tf.less(random_p, p_warp), lambda: self._time_warping(mel_spectrogram)),
                (
                    tf.logical_and(
                        tf.greater_equal(random_p, p_warp), tf.less(random_p, p_warp + p_freq)
                    ),
                    lambda: self._frequency_masking(mel_spectrogram),
                ),
                (
                    tf.logical_and(
                        tf.greater_equal(random_p, p_warp + p_freq),
                        tf.less(random_p, p_warp + p_freq + p_time),
                    ),
                    lambda: self._time_masking(mel_spectrogram),
                ),
            ],
            default=lambda: mel_spectrogram,
            exclusive=True,
        )
        return augmented_mel_spec, char_ids

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
        audio = tf.cast(audio, tf.float32)
        if audio.shape.ndims == 2:
            audio = audio[:, 0]
        stft = tf.signal.stft(
            audio,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
        )
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, self._linear_to_mel_matrix, 1)
        mel_spectrogram.set_shape([None, self.n_mels])
        return tf.math.log(mel_spectrogram + 1e-6)

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
        self,
        data_pairs: List[Tuple[str, str]],
        shuffle: bool,
        augmentation_partitions: List[float] = None,
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

        if augmentation_partitions is not None:
            self._validate_aug_partitions(augmentation_partitions)
            ds = ds.map(
                lambda mel, ids: self._apply_augmentations(mel, ids, augmentation_partitions),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

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

    def get_dataset(
        self, split: str, shuffle: bool, augmentation_partitions: List[float] = None
    ) -> tf.data.Dataset:
        """
        Loads a specified split, returning a single tf.data.Dataset.

        Args:
            split (str): The name of the data split (e.g., "dev-clean").
            shuffle (bool): Whether to shuffle the dataset.
            augmentation_partitions (List[float], optional): A list of three floats
                representing the probabilities for [time_warp, freq_mask, time_mask].
                Defaults to None (no augmentation).
        """
        all_data_pairs = self._get_audio_transcript_paths(split)
        return self._create_configured_dataset(all_data_pairs, shuffle, augmentation_partitions)

    def get_partitioned_datasets(
        self,
        split: str,
        partitions: List[float],
        shuffle: bool,
        augmentation_partitions: List[float] = None,
    ) -> List[tf.data.Dataset]:
        """
        Loads and partitions a split, returning a list of tf.data.Dataset objects.
        """
        if not all(isinstance(p, (int, float)) and 0 < p <= 1 for p in partitions):
            raise ValueError("Partitions must be a list of floats (0, 1].")
        if abs(sum(partitions) - 1.0) > 1e-6:
            raise ValueError("Sum of partitions must be approximately 1.0.")

        all_data_pairs = self._get_audio_transcript_paths(split)
        if shuffle:
            random.shuffle(all_data_pairs)

        partitioned_datasets = []
        current_idx = 0
        total_samples = len(all_data_pairs)

        for i, percentage in enumerate(partitions):
            num_samples = int(total_samples * percentage)
            if i == len(partitions) - 1:
                num_samples = total_samples - current_idx

            partition_data = all_data_pairs[current_idx : current_idx + num_samples]
            current_idx += num_samples

            if not partition_data:
                print(f"Warning: Partition {i+1} has 0 samples.")
                partitioned_datasets.append(self._create_configured_dataset([], shuffle=False))
                continue

            partitioned_datasets.append(
                self._create_configured_dataset(partition_data, shuffle, augmentation_partitions)
            )
        return partitioned_datasets
