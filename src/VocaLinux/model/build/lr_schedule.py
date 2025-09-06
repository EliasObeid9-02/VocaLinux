"""Custom learning rate schedules for TensorFlow models."""

import numpy as np
import tensorflow as tf


class WarmupHoldDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate scheduler that implements a warm-up, hold, and decay phase.
    This is similar to the schedule described in "Attention Is All You Need".

    The schedule is as follows:
    1. Warm-up: Linearly increases the learning rate from 0 to `peak_learning_rate` over `warmup_steps`.
    2. Hold: Keeps the learning rate at `peak_learning_rate` for a set number of steps.
    3. Decay: Smoothly decreases the learning rate using a cosine decay function until it reaches a minimum value.
    """

    def __init__(
        self,
        peak_learning_rate: float,
        warmup_steps: int,
        total_steps: int,
        hold_steps: int = 0,
        min_learning_rate: float = 0.0,
    ):
        """Initializes the learning rate scheduler.

        Args:
            peak_learning_rate (float): The maximum learning rate to reach after warm-up.
            warmup_steps (int): The number of steps for the warm-up phase.
            total_steps (int): The total number of training steps.
            hold_steps (int): The number of steps to hold the peak learning rate. Defaults to 0.
            min_learning_rate (float): The final minimum learning rate after decay. Defaults to 0.0.
        """
        super().__init__()
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold_steps = hold_steps
        self.min_learning_rate = min_learning_rate
        self.decay_steps = total_steps - warmup_steps - hold_steps

        if self.decay_steps < 0:
            raise ValueError(
                "total_steps must be greater than or equal to warmup_steps + hold_steps"
            )

    def __call__(self, step: int | tf.Tensor) -> tf.Tensor:
        """Calculates the learning rate for a given step.

        Args:
            step (tf.Tensor): The current training step (a scalar tensor).

        Returns:
            tf.Tensor: The learning rate for the current step.
        """
        step = tf.cast(step, dtype=tf.float32)

        # --- Warm-up Phase ---
        # Linearly increase the learning rate.
        warmup_lr = self.peak_learning_rate * (step / tf.cast(self.warmup_steps, tf.float32))

        # --- Decay Phase ---
        # Cosine decay from peak to minimum learning rate.
        # We start the decay *after* the warm-up and hold phases.
        step_after_warmup_hold = step - tf.cast(self.warmup_steps + self.hold_steps, tf.float32)

        # Cosine decay formula
        cosine_decay = 0.5 * (
            1 + tf.cos(np.pi * step_after_warmup_hold / tf.cast(self.decay_steps, tf.float32))
        )
        decay_lr = (
            self.peak_learning_rate - self.min_learning_rate
        ) * cosine_decay + self.min_learning_rate

        # --- Determine which phase we are in ---
        # Use tf.cond for conditional logic that works in graph mode.
        is_warmup = step < self.warmup_steps
        is_hold = (step >= self.warmup_steps) & (step < (self.warmup_steps + self.hold_steps))

        learning_rate = tf.case(
            [(is_warmup, lambda: warmup_lr), (is_hold, lambda: self.peak_learning_rate)],
            default=lambda: decay_lr,
        )

        return learning_rate

    def get_config(self) -> dict:
        """Returns a dictionary of the scheduler's configuration parameters.

        This allows the scheduler to be serialized and deserialized.

        Returns:
            dict: A dictionary containing the configuration of the scheduler.
        """
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "hold_steps": self.hold_steps,
            "min_learning_rate": self.min_learning_rate,
        }
