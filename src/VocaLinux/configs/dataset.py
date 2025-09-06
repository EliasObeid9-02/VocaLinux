"""
This file contains the configuration for the dataset processing.
"""

# Default parameters for augmentation, based on the "LD" policy from the SpecAugment paper.
AUGMENTATION_PARAMS = {"W": 80, "F": 27, "m_F": 2, "T": 100, "p": 1.0, "m_T": 2}

# Audio processing parameters
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 100
LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 7600.0
