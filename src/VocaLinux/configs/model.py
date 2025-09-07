"""This file contains the configuration for the LAS model."""

# Listener (Encoder) parameters
LISTENER_LSTM_UNITS = 256
NUM_PBLSTM_LAYERS = 3

# Speller (Decoder) parameters
SPELLER_LSTM_UNITS = 512
NUM_DECODER_LSTM_LAYERS = 2
ATTENTION_UNITS = 512
EMBEDDING_DIM = 256
SAMPLING_PROBABILITY = 0.1
BEAM_WIDTH = 32
