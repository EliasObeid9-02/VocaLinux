"""This file contains the configuration for the training process."""

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 2.5
BATCH_PERIOD = 100  # For HistoryCallback

# Scheduled Sampling parameters
USE_SCHEDULED_SAMPLING = True
SCHEDULED_SAMPLING_START_PROB = 0.05
SCHEDULED_SAMPLING_END_PROB = 0.5
SCHEDULED_SAMPLING_RAMP_EPOCHS = 240

# Early Stopping parameters
EARLY_STOPPING_PATIENCE = 10

# Cyclical Learning Rate parameters
USE_CYCLICAL_LR = False  # Set to True to enable Cyclical LR
CYCLICAL_LR_MIN_LR = 1e-5
CYCLICAL_LR_MAX_LR = 1e-3
CYCLICAL_LR_DECAY = 1.0  # Factor by which max_lr is multiplied at the end of each cycle
CYCLICAL_LR_CYCLE_LENGTH = 10  # Number of epochs in a cycle
CYCLICAL_LR_MULT_FACTOR = 1.0  # Factor by which cycle_length is multiplied at the end of each cycle

# Dataset parameters
TRAIN_SPLIT = "train-clean-100"
VALID_SPLIT = "dev-clean"
TEST_SPLIT = "test-clean"

# Checkpoint and logging parameters
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
