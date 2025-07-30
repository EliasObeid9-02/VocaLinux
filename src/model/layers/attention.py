import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class AttentionContext(Layer):
    """
    Implements a multi-head attention mechanism.

    This layer computes a context vector by attending to the encoder's output
    states based on the decoder's current hidden state. It uses a scaled
    dot-product attention mechanism, split across multiple heads, allowing the
    model to jointly attend to information from different representation
    subspaces at different positions.
    """

    def __init__(self, attention_units, num_heads=4, name="attention_context"):
        """
        Initializes the AttentionContext layer.

        Args:
            attention_units (int): The total dimensionality of the attention space.
                                   This will be split among the heads.
            num_heads (int): The number of parallel attention heads.
            name (str): The name of the layer.
        """
        super().__init__(name=name)
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.head_dim = attention_units // num_heads
        self.kernel_init = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)

    def build(self, input_shape):
        """Builds the internal Dense layers for projections."""
        # Dense layers to project decoder state (query) and encoder states (key) for each head
        self.phi = [
            Dense(self.head_dim, kernel_initializer=self.kernel_init) for _ in range(self.num_heads)
        ]
        self.psi = [
            Dense(self.head_dim, kernel_initializer=self.kernel_init) for _ in range(self.num_heads)
        ]
        self.output_proj = Dense(self.attention_units, kernel_initializer=self.kernel_init)

    def call(self, inputs):
        """
        Performs the forward pass of the attention layer.

        Args:
            inputs (list): A list containing two tensors:
                - dec_state (tf.Tensor): The decoder hidden state from the previous time step.
                                         Shape: (batch_size, decoder_units)
                - enc_state (tf.Tensor): The output states from the encoder (Listener).
                                         Shape: (batch_size, U, encoder_units)

        Returns:
            tf.Tensor: The context vector, which is a weighted sum of the encoder states.
                       Shape: (batch_size, attention_units)
        """
        dec_state, enc_state = inputs
        contexts = []

        # Iterate over each attention head
        for head in range(self.num_heads):
            # Project decoder state (query) and encoder states (keys)
            phi_s = self.phi[head](dec_state)  # Shape: [B, head_dim]
            psi_h = self.psi[head](enc_state)  # Shape: [B, U, head_dim]

            # Calculate scaled dot-product energy scores
            energy = tf.einsum("bd,bud->bu", phi_s, psi_h) / tf.sqrt(
                tf.cast(self.head_dim, tf.float32)
            )
            attn = tf.nn.softmax(energy)

            # Compute the context vector for the current head
            context = tf.einsum("bu,bud->bd", attn, enc_state)
            context = tf.math.l2_normalize(context, axis=-1)
            contexts.append(context)
        combined = tf.concat(contexts, axis=-1)
        return self.output_proj(combined)
