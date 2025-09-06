"""
This file contains the configuration for the training process.
"""

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

# Dataset parameters
TRAIN_SPLIT = "train-clean-100"
VALID_SPLIT = "dev-clean"
TEST_SPLIT = "test-clean"

# Checkpoint and logging parameters
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
