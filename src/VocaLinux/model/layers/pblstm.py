import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Layer, LayerNormalization


class PBLSTMLayer(Layer):
    """
    Implements a single Pyramidal Bidirectional LSTM (pBLSTM) layer as described
    in the "Listen, Attend and Spell" paper (Section 3.1, Equation 5).

    This layer takes an input sequence, concatenates pairs of consecutive time steps,
    and then feeds this reduced-resolution sequence through a Bidirectional LSTM.
    This effectively reduces the time resolution by a factor of 2.
    """

    def __init__(self, units: int, name: str = "pblstm_layer"):
        """
        Initializes the PBLSTMLayer.

        Args:
            units (int): Number of LSTM units in each direction (forward and backward).
            name (str): Name of the layer.
        """
        super().__init__(name=name)
        self.units = units
        self.kernel_init = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)
        self.recurrent_init = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the layer by creating its trainable weights and sub-layers.
        This method is called automatically once the input shape is known.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor to this layer.
                                          Expected: (batch_size, num_time_steps, feature_dim)
        """
        self.blstm = Bidirectional(
            LSTM(
                self.units,
                return_sequences=True,
                kernel_initializer=self.kernel_init,
                recurrent_initializer=self.recurrent_init,
                name=f"{self.name}_blstm_cell",
            ),
            name=f"{self.name}_blstm_bidirectional",
        )
        self.layer_norm = LayerNormalization(name=f"{self.name}_layer_norm")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the forward pass of the PBLSTMLayer.

        Args:
            inputs (tf.Tensor): The input tensor from the previous layer.
                                Expected shape: (batch_size, num_time_steps, feature_dim)

        Returns:
            tf.Tensor: The output tensor with reduced time resolution.
                       Shape: (batch_size, ceil(num_time_steps / 2), 2 * units)
        """
        input_shape = tf.shape(inputs)
        num_time_steps = input_shape[1]
        feature_dim = input_shape[2]

        # Use tf.cond for conditional logic with symbolic tensors
        inputs = tf.cond(
            tf.math.equal(tf.math.floormod(num_time_steps, 2), 1),
            true_fn=lambda: tf.concat(
                [inputs, tf.zeros(tf.stack([input_shape[0], 1, feature_dim]), dtype=inputs.dtype)],
                axis=1,
            ),
            false_fn=lambda: inputs,
        )
        num_time_steps = tf.shape(inputs)[1]

        # Split inputs into even and odd time steps
        even_steps = inputs[:, 0::2, :]
        odd_steps = inputs[:, 1::2, :]

        # Concatenate them along the feature dimension
        concatenated_inputs = tf.concat([even_steps, odd_steps], axis=-1)
        output = self.blstm(concatenated_inputs)
        output = self.layer_norm(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
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
            output_time_steps = input_shape[1] // 2 + input_shape[1] % 2  # type: ignore due to input_shape[1] being int | None

        # Doubling due to bidirectional LSTM output
        output_feature_dim = 2 * self.units
        return tf.TensorShape([batch_size, output_time_steps, output_feature_dim])
