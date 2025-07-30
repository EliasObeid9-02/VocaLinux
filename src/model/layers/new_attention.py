import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class LocationAwareAttention(Layer):
    """
    Implements a location-aware attention mechanism as described in
    "Attention-Based Models for Speech Recognition" by Chorowski et al.
    This is a multi-head dot-product attention mechanism with an added
    "location-awareness" component.
    """

    def __init__(self, attention_units, num_heads=4, name="location_aware_attention"):
        super().__init__(name=name)
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.head_dim = attention_units // num_heads
        self.kernel_init = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)

    def build(self, input_shape):
        """Builds the internal Dense layers for projections."""
        # Dense layers to project decoder state (query) and encoder states (key) for each head
        self.phi = [
            Dense(self.head_dim, kernel_initializer=self.kernel_init, name=f"phi_head_{i}")
            for i in range(self.num_heads)
        ]
        self.psi = [
            Dense(self.head_dim, kernel_initializer=self.kernel_init, name=f"psi_head_{i}")
            for i in range(self.num_heads)
        ]
        self.output_proj = Dense(self.attention_units, kernel_initializer=self.kernel_init)

        # Location-awareness components
        # A 1D convolution to extract features from previous attention weights
        self.location_conv = tf.keras.layers.Conv1D(
            filters=32, kernel_size=31, padding="same", use_bias=False, name="location_conv"
        )
        # A dense layer to project these features to a bias term for the energy calculation
        self.location_dense = Dense(self.num_heads, use_bias=False, name="location_dense")

    def call(self, inputs):
        """
        Performs the forward pass of the attention layer.

        Args:
            inputs (list): A list containing three tensors:
                - dec_state (tf.Tensor): The decoder hidden state from the previous time step.
                                         Shape: (batch_size, decoder_units)
                - enc_state (tf.Tensor): The output states from the encoder (Listener).
                                         Shape: (batch_size, U, encoder_units)
                - prev_attention_weights (tf.Tensor): The attention weights from the previous decoding step.
                                                      Shape: (batch_size, U)
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - context_vector (tf.Tensor): The context vector for the current time step.
                                              Shape: (batch_size, attention_units)
                - attention_weights (tf.Tensor): The new attention weights for the current step.
                                                 Shape: (batch_size, U)
        """
        dec_state, enc_state, prev_attention_weights = inputs

        # --- Location-Awareness Calculation ---
        # Reshape previous weights for convolution: (batch, U) -> (batch, U, 1)
        prev_weights_reshaped = tf.expand_dims(prev_attention_weights, axis=-1)
        # Extract features: (batch, U, 1) -> (batch, U, 32)
        location_features = self.location_conv(prev_weights_reshaped)
        # Project to a bias term for each head: (batch, U, 32) -> (batch, U, num_heads)
        location_bias = self.location_dense(location_features)
        # Transpose for broadcasting: (batch, U, num_heads) -> (num_heads, batch, U)
        location_bias_per_head = tf.transpose(location_bias, perm=[2, 0, 1])

        # --- Multi-Head Attention Calculation ---
        contexts = []
        head_attentions = []

        for head in range(self.num_heads):
            phi_s = self.phi[head](dec_state)  # Shape: (batch, head_dim)
            psi_h = self.psi[head](enc_state)  # Shape: (batch, U, head_dim)

            # Dot-product energy with location aware bias
            energy = tf.einsum("bd,bud->bu", phi_s, psi_h)
            energy += location_bias_per_head[head]

            # Compute attention weights
            attn = tf.nn.softmax(energy)
            head_attentions.append(attn)

            # Compute context vector for the current head
            context = tf.einsum("bu,bud->bd", attn, enc_state)
            contexts.append(context)

        # Concatenate context vectors from all heads
        combined_context = tf.concat(contexts, axis=-1)
        context_vector = self.output_proj(combined_context)

        # Average the attention weights across all heads for the next step's state
        attention_weights = tf.reduce_mean(tf.stack(head_attentions), axis=0)
        return context_vector, attention_weights
