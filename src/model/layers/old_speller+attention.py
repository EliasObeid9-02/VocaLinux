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
        self.kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.075, maxval=0.075)

    def build(self, input_shape):
        """Builds the internal Dense layers for projections."""
        # Dense layers to project decoder state (query) and encoder states (key) for each head
        self.phi = [
            Dense(self.head_dim, kernel_initializer=self.kernel_initializer)
            for _ in range(self.num_heads)
        ]
        self.psi = [
            Dense(self.head_dim, kernel_initializer=self.kernel_initializer)
            for _ in range(self.num_heads)
        ]
        self.output_proj = Dense(self.attention_units, kernel_initializer=self.kernel_initializer)

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


class Speller(Layer):
    """
    The Speller is an attention-based recurrent neural network decoder that
    emits characters as outputs, forming the second core component of the
    Listen, Attend and Spell (LAS) model. It takes the high-level features
    from the Listener and generates a probability distribution over character
    sequences, one character at a time.
    """

    def __init__(
        self,
        lstm_units: int = 512,
        num_decoder_lstm_layers: int = 2,
        attention_units: int = 512,
        output_vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = 256,
        sampling_probability: float = 0.1,
        beam_width: int = 32,
        name: str = "speller",
    ):
        """
        Initializes the Speller layer.

        Args:
            lstm_units (int): Number of units for each LSTM layer in the decoder.
            num_decoder_lstm_layers (int): Number of stacked LSTM layers in the decoder.
            attention_units (int): Number of units for the attention mechanism's internal MLPs.
            output_vocab_size (int): Size of the output character vocabulary.
            embedding_dim (int): Dimension of the embedding for output characters.
            sampling_probability (float): The probability of using the model's own
                                          prediction as the next input during training
                                          (scheduled sampling).
            name (str): The name of the layer.
        """
        super().__init__(name=name)
        self.lstm_units = lstm_units
        self.num_decoder_lstm_layers = num_decoder_lstm_layers
        self.attention_units = attention_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.beam_width = beam_width
        self.kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        self.recurrent_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

        self._sampling_probability = tf.Variable(
            sampling_probability, dtype=tf.float32, trainable=False, name="sampling_probability_var"
        )

    @property
    def sampling_probability(self):
        return self._sampling_probability.numpy().item()

    @sampling_probability.setter
    def sampling_probability(self, value):
        self._sampling_probability.assign(value)

    def build(self, input_shape: List[tf.TensorShape]):
        """
        Builds the Speller layer by creating its sub-layers and initializing weights.

        Args:
            input_shape (List[tf.TensorShape]): A list containing two TensorFlow TensorShapes:
                                                 - encoder_outputs_shape: Shape of the encoder's output features.
                                                                          Expected: (batch_size, U, encoder_feature_dim)
                                                 - target_sequences_shape: Shape of the target character sequences.
                                                                           Expected: (batch_size, max_target_len)
        Raises:
            ValueError: If the input_shape is not a list of two shapes.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Speller layer expects 2 input shapes: [encoder_outputs_shape, target_sequences_shape]"
            )

        encoder_outputs_shape, target_sequences_shape = input_shape
        encoder_feature_dim = encoder_outputs_shape[-1]

        self.embedding = Embedding(
            input_dim=self.output_vocab_size,
            output_dim=self.embedding_dim,
            name="speller_embedding",
            embeddings_initializer=self.kernel_initializer,
        )

        self.decoder_lstm_cells = []
        self.decoder_layer_norms = []
        for i in range(self.num_decoder_lstm_layers):
            # Input dimension for the first LSTM cell is a concatenation of the
            # previous character embedding and the previous context vector.
            # For subsequent LSTM cells, the input is the output of the previous LSTM cell.
            input_dim_for_lstm_cell = (
                self.embedding_dim + encoder_feature_dim if i == 0 else self.lstm_units  # type: ignore
            )
            # Each LSTM cell processes a single time step and returns its output
            # (h_curr) and its final hidden and cell states (h_curr, c_curr).
            self.decoder_lstm_cells.append(
                LSTM(
                    self.lstm_units,
                    name=f"speller_lstm_{i+1}",
                    kernel_initializer=self.kernel_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    return_sequences=False,
                    return_state=True,
                    dropout=0.25,
                    recurrent_dropout=0.18,
                    kernel_regularizer=tf.keras.regularizers.l2(7e-5),
                )
            )
            self.decoder_layer_norms.append(
                LayerNormalization(name=f"speller_lstm_layer_norm_{i+1}")
            )
        self.attention_context = AttentionContext(attention_units=self.attention_units)

        # MLP layer to predict the character distribution (logits) over the vocabulary.
        self.character_distribution_mlp = Dense(
            self.output_vocab_size,
            activation=None,
            name="character_distribution",
            kernel_initializer=self.kernel_initializer,
        )
        super().build(input_shape)

    def _decoder_step(
        self,
        prev_char_embedding: tf.Tensor,
        prev_context: tf.Tensor,
        decoder_lstm_hidden_states: tf.Tensor,
        decoder_lstm_cell_states: tf.Tensor,
        encoder_outputs: tf.Tensor,
    ):
        """
        Performs a single step of the Speller's decoding process.
        This function is designed to be used within tf.scan for iterative decoding.

        Args:
            prev_char_embedding (tf.Tensor): The embedding of the character from the previous time step.
                                             Shape: (batch_size, embedding_dim)
            prev_context (tf.Tensor): The context vector from the previous time step.
                                      Shape: (batch_size, encoder_feature_dim)
            decoder_lstm_hidden_states (tf.Tensor): Stacked hidden states of all decoder LSTM layers
                                                    from the previous time step.
                                                    Shape: (num_decoder_lstm_layers, batch_size, lstm_units)
            decoder_lstm_cell_states (tf.Tensor): Stacked cell states of all decoder LSTM layers
                                                  from the previous time step.
                                                  Shape: (num_decoder_lstm_layers, batch_size, lstm_units)
            encoder_outputs (tf.Tensor): The output features from the Listener (encoder).
                                         Shape: (batch_size, U, encoder_feature_dim)

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
                - logits (tf.Tensor): Logits (unnormalized log-probabilities) for the current
                                      character prediction. Shape: (batch_size, output_vocab_size)
                - current_context_output (tf.Tensor): The newly computed context vector for the current time step.
                                                      Shape: (batch_size, encoder_feature_dim)
                - new_decoder_lstm_hidden_states (tf.Tensor): Stacked new hidden states for all LSTM layers.
                                                              Shape: (num_decoder_lstm_layers, batch_size, lstm_units)
                - new_decoder_lstm_cell_states (tf.Tensor): Stacked new cell states for all LSTM layers.
                                                            Shape: (num_decoder_lstm_layers, batch_size, lstm_units)
        """
        decoder_lstm_states = []
        for i in range(self.num_decoder_lstm_layers):
            decoder_lstm_states.append((decoder_lstm_hidden_states[i], decoder_lstm_cell_states[i]))

        # Input to the first LSTM layer is a concatenation of the previous character embedding
        # and the previous context vector.
        current_input = tf.concat([prev_char_embedding, prev_context], axis=-1)
        new_decoder_lstm_hidden_states = []
        new_decoder_lstm_cell_states = []

        for i in range(self.num_decoder_lstm_layers):
            lstm_cell = self.decoder_lstm_cells[i]
            layer_norm_layer = self.decoder_layer_norms[i]
            h_prev, c_prev = decoder_lstm_states[i]

            # Pass the current input and previous states to the LSTM cell.
            # tf.expand_dims(current_input, 1) is used because LSTM expects a 3D input (batch, timesteps, features),
            # and here we are processing one timestep at a time.
            output, h_curr, c_curr = lstm_cell(
                tf.expand_dims(current_input, 1), initial_state=[h_prev, c_prev]
            )
            output = layer_norm_layer(output)

            new_decoder_lstm_hidden_states.append(h_curr)
            new_decoder_lstm_cell_states.append(c_curr)

            # For subsequent layers, the input is the output of the previous LSTM layer.
            # This output is (batch_size, lstm_units) as return_sequences=False
            current_input = output

        # The decoder state: equation (7) is typically the hidden state (h_state)
        # of the LAST LSTM layer in the stack for the current time step.
        current_decoder_hidden_state = new_decoder_lstm_hidden_states[-1]
        current_context = self.attention_context([current_decoder_hidden_state, encoder_outputs])
        prediction_input = tf.concat([current_decoder_hidden_state, current_context], axis=-1)
        raw_logits = self.character_distribution_mlp(prediction_input)
        softmax_probabilities = tf.nn.softmax(raw_logits, axis=-1)
        return (
            softmax_probabilities,
            current_context,
            tf.stack(new_decoder_lstm_hidden_states),
            tf.stack(new_decoder_lstm_cell_states),
        )

    def _greedy_decode(self, encoder_outputs, max_decode_len):
        """Performs greedy decoding for inference."""
        batch_size = tf.shape(encoder_outputs)[0]

        # Initialize with <sos> token
        decoded_sequence = tf.ones((batch_size, 1), dtype=tf.int32) * SOS_TOKEN

        # Initial decoder states and context
        decoder_states = [
            (tf.zeros((batch_size, self.lstm_units)), tf.zeros((batch_size, self.lstm_units)))
            for _ in range(self.num_decoder_lstm_layers)
        ]
        context_vector = tf.zeros((batch_size, tf.shape(encoder_outputs)[-1]))

        for t in range(max_decode_len):
            last_token = decoded_sequence[:, -1]
            char_embedding = self.embedding(last_token)

            (
                softmax_probs,
                context_vector,
                new_h_states,
                new_c_states,
            ) = self._decoder_step(
                char_embedding,
                context_vector,
                tf.stack([s[0] for s in decoder_states]),
                tf.stack([s[1] for s in decoder_states]),
                encoder_outputs,
            )

            # Update states
            decoder_states = [
                (new_h_states[i], new_c_states[i]) for i in range(self.num_decoder_lstm_layers)
            ]

            # Choose the most likely token (greedy choice)
            next_token = tf.argmax(softmax_probs, axis=-1, output_type=tf.int32)
            next_token = tf.expand_dims(next_token, axis=1)

            decoded_sequence = tf.concat([decoded_sequence, next_token], axis=1)

        return decoded_sequence

    def _beam_decode(self, encoder_outputs, max_decode_len):
        """Performs beam search decoding using the class's beam_width."""
        batch_size = tf.shape(encoder_outputs)[0]

        # Initialize beams, scores, and states
        beams = tf.ones((batch_size, self.beam_width, 1), dtype=tf.int32) * SOS_TOKEN

        initial_scores = tf.constant(
            [[0.0] + [-float("inf")] * (self.beam_width - 1)], shape=(1, self.beam_width)
        )
        beam_scores = tf.tile(initial_scores, [batch_size, 1])

        # Tile encoder outputs and initialize states for each beam
        flat_encoder_outputs = tf.reshape(
            tf.tile(tf.expand_dims(encoder_outputs, 1), [1, self.beam_width, 1, 1]),
            [-1, tf.shape(encoder_outputs)[1], tf.shape(encoder_outputs)[2]],
        )

        decoder_states = [
            (
                tf.zeros((batch_size * self.beam_width, self.lstm_units)),
                tf.zeros((batch_size * self.beam_width, self.lstm_units)),
            )
            for _ in range(self.num_decoder_lstm_layers)
        ]
        context_vector = tf.zeros((batch_size * self.beam_width, tf.shape(encoder_outputs)[-1]))

        for t in range(max_decode_len):
            last_tokens = tf.reshape(beams[:, :, -1], [-1])

            if tf.reduce_all(tf.equal(last_tokens, EOS_TOKEN)):
                break

            char_embedding = self.embedding(last_tokens)
            (
                softmax_probs,
                context_vector,
                new_h_states,
                new_c_states,
            ) = self._decoder_step(
                char_embedding,
                context_vector,
                tf.stack([s[0] for s in decoder_states]),
                tf.stack([s[1] for s in decoder_states]),
                flat_encoder_outputs,
            )
            decoder_states = [
                (new_h_states[i], new_c_states[i]) for i in range(self.num_decoder_lstm_layers)
            ]

            log_probs = tf.math.log(softmax_probs + 1e-10)
            total_scores = tf.reshape(beam_scores, [-1, 1]) + log_probs
            total_scores = tf.reshape(total_scores, [batch_size, -1])

            beam_scores, top_indices = tf.nn.top_k(total_scores, k=self.beam_width)

            beam_indices = top_indices // self.output_vocab_size
            token_indices = top_indices % self.output_vocab_size

            # --- Start of The Fix ---
            # Create an offset to correctly index into the flat (batch_size * beam_width) tensors
            batch_offset = tf.range(batch_size, dtype=tf.int32) * self.beam_width
            flat_beam_indices = beam_indices + tf.expand_dims(batch_offset, 1)

            # Update states and context by gathering from the winning beams
            updated_states = []
            for i in range(self.num_decoder_lstm_layers):
                h = tf.gather(decoder_states[i][0], flat_beam_indices)
                c = tf.gather(decoder_states[i][1], flat_beam_indices)
                # Reshape back to the flat format for the next loop iteration
                h = tf.reshape(h, [-1, self.lstm_units])
                c = tf.reshape(c, [-1, self.lstm_units])
                updated_states.append((h, c))
            decoder_states = updated_states

            context_vector = tf.gather(context_vector, flat_beam_indices)
            context_vector = tf.reshape(context_vector, [-1, tf.shape(encoder_outputs)[-1]])

            # Update the beams by gathering the previous beams and appending the new tokens
            new_tokens = tf.cast(
                tf.reshape(token_indices, [batch_size, self.beam_width, 1]), tf.int32
            )
            beams = tf.concat([tf.gather(beams, beam_indices, batch_dims=1), new_tokens], axis=2)
            # --- End of The Fix ---

        return beams[:, 0, :]

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Performs the forward pass of the Speller layer, generating character sequences.

        During training, it uses scheduled sampling to decide whether to use the
        ground truth previous character or the model's own prediction as input
        for the next step. During inference, it always uses the model's own
        previous prediction.

        Args:
            inputs (List[tf.Tensor]): A list containing two TensorFlow Tensors:
                                      - encoder_outputs (tf.Tensor): Output features from the Listener (encoder).
                                                                     Shape: (batch_size, U, encoder_feature_dim)
                                      - target_sequences (tf.Tensor): Ground truth target character sequences.
                                                                      During training, used for teacher forcing
                                                                      and scheduled sampling. During inference,
                                                                      the first token (e.g., <sos>) is used.
                                                                      Shape: (batch_size, max_target_len)
            training (bool): Whether the layer is in training mode. This flag
                             controls the scheduled sampling behavior.

        Returns:
            tf.Tensor: A tensor containing the predicted logits for each character
                       at each time step of the generated sequence.
                       Shape: (batch_size, max_target_len, output_vocab_size)
        """
        encoder_outputs, target_sequences = inputs
        batch_size = tf.shape(encoder_outputs)[0]
        max_target_len = tf.shape(target_sequences)[1]

        initial_hidden_states = tf.zeros(
            (self.num_decoder_lstm_layers, batch_size, self.lstm_units), dtype=tf.float32
        )
        initial_cell_states = tf.zeros(
            (self.num_decoder_lstm_layers, batch_size, self.lstm_units), dtype=tf.float32
        )

        encoder_feature_dim = encoder_outputs.shape[-1]
        initial_context = tf.zeros((batch_size, encoder_feature_dim), dtype=tf.float32)

        # This is a placeholder as there are no previous logits at time step 0.
        initial_logits_dummy = tf.zeros((batch_size, self.output_vocab_size), dtype=tf.float32)
        initial_scan_state = (
            initial_logits_dummy,
            initial_context,
            initial_hidden_states,
            initial_cell_states,
        )

        def scan_fn(state, current_time_step_idx):
            """
            The scan function that defines the computation for each time step.
            This function is passed to tf.scan and iteratively updates the decoder state.

            Args:
                state (Tuple[tf.Tensor, ...]): The current state of the decoder from the previous time step,
                                               containing: (prev_logits, prev_context, hidden_states, cell_states).
                current_time_step_idx (tf.Tensor): The current time step index (scalar).

            Returns:
                Tuple[tf.Tensor, ...]: The updated state for the next iteration of scan,
                                       containing: (current_logits, current_context, new_hidden_states, new_cell_states).
            """
            (
                prev_logits,
                prev_context_scan,
                decoder_lstm_hidden_states_scan,
                decoder_lstm_cell_states_scan,
            ) = state

            """
            Determine which character ID to use for embedding in the current step:
            - If in training mode:
            - If it's not the very first time step (current_time_step_idx > 0) AND
            - A random uniform number is less than self.sampling_probability (scheduled sampling):
                Use the argmax of the `prev_logits` (model's own prediction).
              - Otherwise (not first step and no scheduled sampling, or first step):
                Use the ground truth character from `target_sequences`.
            - If in inference mode (not training):
              - If it's the very first time step (current_time_step_idx == 0):
                Use the ground truth character from `target_sequences` (which should be the <sos> token).
              - Otherwise (not first step in inference):
                Use the argmax of the `prev_logits` (model's own previous output).
            """
            char_id_for_embedding = tf.cond(
                tf.cast(training, dtype=tf.bool),
                true_fn=lambda: tf.cond(
                    tf.logical_and(
                        tf.greater(current_time_step_idx, 0),
                        # commented out for testing dynamic sampling_probability
                        # tf.random.uniform([], minval=0.0, maxval=1.0) < self.sampling_probability,
                        tf.random.uniform([], minval=0.0, maxval=1.0)
                        < self._sampling_probability,  # Use the tf.Variable
                    ),
                    true_fn=lambda: tf.argmax(prev_logits, axis=-1, output_type=tf.int32),
                    false_fn=lambda: tf.cast(target_sequences[:, current_time_step_idx], tf.int32),
                ),
                false_fn=lambda: tf.cond(
                    tf.equal(current_time_step_idx, 0),
                    true_fn=lambda: tf.cast(target_sequences[:, current_time_step_idx], tf.int32),
                    false_fn=lambda: tf.argmax(prev_logits, axis=-1, output_type=tf.int32),
                ),
            )

            prev_char_embedding = self.embedding(char_id_for_embedding)

            """
            Returns the following
                logits,
                current_context_output,
                new_decoder_lstm_hidden_states,
                new_decoder_lstm_cell_states,
            """
            return self._decoder_step(
                prev_char_embedding,
                prev_context_scan,
                decoder_lstm_hidden_states_scan,
                decoder_lstm_cell_states_scan,
                encoder_outputs,
            )

        time_step_indices = tf.range(max_target_len)
        all_outputs_scan_tuple = tf.scan(
            fn=scan_fn,
            elems=time_step_indices,
            initializer=initial_scan_state,
        )

        logits_sequence = all_outputs_scan_tuple[0]
        final_outputs = tf.transpose(logits_sequence, perm=[1, 0, 2])
        return final_outputs

    def get_config(self):
        """
        Returns the serializable configuration of the layer.
        This allows the layer to be recreated from its configuration, which is
        important for saving and loading models.

        Returns:
            dict: A dictionary containing the layer's configuration.
        """
        config = super().get_config()
        config.update(
            {
                "lstm_units": self.lstm_units,
                "num_decoder_lstm_layers": self.num_decoder_lstm_layers,
                "attention_units": self.attention_units,
                "output_vocab_size": self.output_vocab_size,
                "embedding_dim": self.embedding_dim,
                "sampling_probability": self.sampling_probability,
            }
        )
        return config

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:  # type: ignore
        """
        Computes the output shape of the Speller layer given the input shapes.

        Args:
            input_shape (List[tf.TensorShape]): A list containing two TensorFlow TensorShapes:
                                                 - encoder_outputs_shape: Shape of the encoder's output features.
                                                                          Expected: (batch_size, U, encoder_feature_dim)
                                                 - target_sequences_shape: Shape of the target character sequences.
                                                                           Expected: (batch_size, max_target_len)

        Returns:
            tf.TensorShape: The shape of the output tensor, which is the logits
                            for each character at each time step.
                            Shape: (batch_size, max_target_len, output_vocab_size)
        """
        encoder_outputs_shape, target_sequences_shape = input_shape
        batch_size = encoder_outputs_shape[0]

        # The output sequence length is determined by the target sequence length
        output_sequence_length = target_sequences_shape[1]
        return tf.TensorShape([batch_size, output_sequence_length, self.output_vocab_size])
