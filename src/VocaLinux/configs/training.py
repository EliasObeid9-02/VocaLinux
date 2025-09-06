"""
This file contains the configuration for the training process.
"""

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
GRAD_CLIP_NORM = 2.5

# Dataset parameters
TRAIN_SPLIT = "train-clean-100"
VALID_SPLIT = "dev-clean"
TEST_SPLIT = "test-clean"
