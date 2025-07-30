from typing import List

import tensorflow as tf
from tensorflow.keras import Model

from model.layers.listener import Listener
from model.layers.speller import Speller
from model.vocabulary import VOCAB_SIZE


class LASModel(Model):
    """
    The complete Listen, Attend and Spell (LAS) model.
    This model combines the Listener (encoder) and Speller (decoder) components
    for end-to-end speech recognition.
    """

    def __init__(
        self,
        listener_lstm_units: int = 256,
        num_pblstm_layers: int = 3,
        speller_lstm_units: int = 512,
        num_decoder_lstm_layers: int = 2,
        attention_units: int = 512,
        output_vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = 256,
        sampling_probability: float = 0.1,
        beam_width: int = 32,
        name: str = "las_model",
        **kwargs,
    ):
        """
        Initializes the LASModel.

        Args:
            listener_lstm_units (int): Number of LSTM units per direction for Listener's LSTMs.
            num_pblstm_layers (int): Number of pBLSTM layers in the Listener.
            speller_lstm_units (int): Number of LSTM units for each Speller's LSTM layer.
            num_decoder_lstm_layers (int): Number of stacked LSTM layers in the Speller.
            attention_units (int): Dimension for the MLPs in the AttentionContext layer.
            output_vocab_size (int): Size of the output character vocabulary.
            embedding_dim (int): Dimension of the character embedding in the Speller.
            sampling_probability (float): Probability of using the model's own prediction
                                          instead of ground truth during training.
            name (str): Name of the model.
            **kwargs: Additional keyword arguments for the Model base class.
        """
        super().__init__(name=name, **kwargs)

        self.listener = Listener(
            lstm_units=listener_lstm_units,
            num_pblstm_layers=num_pblstm_layers,
            name="las_listener",
        )

        self.speller = Speller(
            lstm_units=speller_lstm_units,
            num_decoder_lstm_layers=num_decoder_lstm_layers,
            attention_units=attention_units,
            output_vocab_size=output_vocab_size,
            embedding_dim=embedding_dim,
            sampling_probability=sampling_probability,
            beam_width=beam_width,
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
