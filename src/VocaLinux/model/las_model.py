from typing import List

import tensorflow as tf
from tensorflow.keras import Model

from VocaLinux.model.layers.listener import Listener
from VocaLinux.model.layers.speller import Speller
from VocaLinux.configs import model as model_config


class LASModel(Model):
    """
    The complete Listen, Attend and Spell (LAS) model.
    This model combines the Listener (encoder) and Speller (decoder) components
    for end-to-end speech recognition.
    """

    def __init__(
        self,
        name: str = "las_model",
        **kwargs,
    ):
        """
        Initializes the LASModel.

        Args:
            name (str): Name of the model.
            **kwargs: Additional keyword arguments for the Model base class.
        """
        super().__init__(name=name, **kwargs)

        self.listener = Listener(
            lstm_units=model_config.LISTENER_LSTM_UNITS,
            num_pblstm_layers=model_config.NUM_PBLSTM_LAYERS,
            name="las_listener",
        )

        self.speller = Speller(
            lstm_units=model_config.SPELLER_LSTM_UNITS,
            num_decoder_lstm_layers=model_config.NUM_DECODER_LSTM_LAYERS,
            attention_units=model_config.ATTENTION_UNITS,
            output_vocab_size=model_config.OUTPUT_VOCAB_SIZE,
            embedding_dim=model_config.EMBEDDING_DIM,
            sampling_probability=model_config.SAMPLING_PROBABILITY,
            beam_width=model_config.BEAM_WIDTH,
            name="las_speller",
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:  # type: ignore
        """
        Performs the forward pass of the LASModel.

        Args:
            inputs (List[tf.Tensor]): A list containing two tensors:
                                      - mel_spectrograms: Batches of audio samples.
                                        Shape: (batch_size, max_mel_frames, n_mels)
                                      - target_sequences: Ground truth character IDs (shifted for teacher forcing).
                                        Shape: (batch_size, max_target_len)
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: Logits for the character distribution at each time step.
                       Shape: (batch_size, max_target_len, output_vocab_size)
        """
        mel_spectrograms, target_sequences = inputs
        encoded_features = self.listener(mel_spectrograms, training=training)
        character_logits = self.speller([encoded_features, target_sequences], training=training)
        return character_logits

    def greedy_predict(self, mel_spectrograms):
        """
        Performs greedy decoding for inference with a dynamic length cap
        based on the input audio length.
        """
        input_len = tf.shape(mel_spectrograms)[1]

        # Heuristic: Set max length to be 1/4 of the input frames plus a fixed minimum.
        # This prevents extremely long outputs for noisy inputs while being generous.
        max_decode_len = (input_len // 4) + 20

        encoder_outputs = self.listener(mel_spectrograms, training=False)
        return self.speller._greedy_decode(encoder_outputs, max_decode_len)

    def beam_search_predict(self, mel_spectrograms):
        """
        Performs beam search decoding with a dynamic length cap
        based on the input audio length.
        """
        input_len = tf.shape(mel_spectrograms)[1]

        # Heuristic: Set max length to be 1/4 of the input frames plus a fixed minimum.
        # This prevents extremely long outputs for noisy inputs while being generous.
        max_decode_len = (input_len // 4) + 20

        encoder_outputs = self.listener(mel_spectrograms, training=False)
        return self.speller._beam_decode(encoder_outputs, max_decode_len)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "listener_lstm_units": self.listener.lstm_units,
                "num_pblstm_layers": self.listener.num_pblstm_layers,
                "speller_lstm_units": self.speller.lstm_units,
                "num_decoder_lstm_layers": self.speller.num_decoder_lstm_layers,
                "attention_units": self.speller.attention_units,
                "output_vocab_size": self.speller.output_vocab_size,
                "embedding_dim": self.speller.embedding_dim,
                "sampling_probability": self.speller.sampling_probability,
            }
        )
        return config
