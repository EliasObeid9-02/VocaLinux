import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Layer, LayerNormalization

from model.layers.pblstm import PBLSTMLayer


class Listener(Layer):
    """
    The Listener (Encoder) component of the Listen, Attend and Spell (LAS) model.
    It is a pyramidal recurrent network encoder that accepts Mel Spectrograms as inputs.
    As described in the paper (Section 3.1), it consists of:
    1. A bottom BLSTM layer.
    2. Three stacked pBLSTM layers on top, each reducing time resolution by a factor of 2.
    """

    def __init__(self, lstm_units: int = 256, num_pblstm_layers: int = 3, name: str = "listener"):
        """
        Initializes the Listener.

        Args:
            lstm_units (int): Number of LSTM units in each direction for all BLSTM/pBLSTM layers.
                              The paper states "3 layers of 512 pBLSTM nodes (i.e., 256 nodes per direction)".
                              So, units here refers to units PER DIRECTION.
            num_pblstm_layers (int): Number of stacked pBLSTM layers. Paper uses 3.
            name (str): Name of the listener module.
        """
        super().__init__(name=name)
        self.lstm_units = lstm_units
        self.num_pblstm_layers = num_pblstm_layers
        self.kernel_init = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)
        self.recurrent_init = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the layer by creating its trainable weights and sub-layers.
        This method is called automatically once the input shape is known.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor to this layer.
                                          Expected: (batch_size, max_mel_frames, n_mels)
        """
        self.bottom_blstm = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=True,
                kernel_initializer=self.kernel_init,
                recurrent_initializer=self.recurrent_init,
                name="listener_bottom_blstm",
            ),
            name="listener_bottom_bidirectional",
        )
        self.bottom_layer_norm = LayerNormalization(name="listener_bottom_layer_norm")

        self.pblstm_layers = []
        for i in range(self.num_pblstm_layers):
            self.pblstm_layers.append(
                PBLSTMLayer(units=self.lstm_units, name=f"listener_pblstm_layer_{i+1}")
            )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Performs the forward pass of the Listener.

        Args:
            inputs (tf.Tensor): Batches of audio samples encoded as Mel Spectrograms.
                                Expected shape: (batch_size, max_mel_frames, n_mels)
            training (bool): Whether the layer is in training mode. (Used for dropout etc.,
                             though not explicitly used in this current implementation).

        Returns:
            tf.Tensor: The high-level feature representation `h` with reduced time resolution.
                       Shape: (batch_size, ceil(original_mel_frames / (2^num_pblstm_layers)), 2 * lstm_units)
        """
        h = self.bottom_blstm(inputs)
        h = self.bottom_layer_norm(h)

        for pblstm_layer in self.pblstm_layers:
            h = pblstm_layer(h)
        return h

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lstm_units": self.lstm_units,
                "num_pblstm_layers": self.num_pblstm_layers,
            }
        )
        return config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Computes the output shape of the layer.
        Handles dynamic time dimensions (None).
        """
        batch_size = input_shape[0]

        if input_shape[1] is None:
            output_time_steps = None  # If input is dynamic, output is dynamic
        else:
            # Time steps are halved and ceiling is applied due to padding)
            output_time_steps = input_shape[1]
            for _ in range(self.num_pblstm_layers):
                output_time_steps = output_time_steps // 2 + output_time_steps % 2  # type: ignore due to output_time_steps being int | None

        # Doubling due to bidirectional LSTM output
        output_feature_dim = 2 * self.lstm_units
        return tf.TensorShape([batch_size, output_time_steps, output_feature_dim])
