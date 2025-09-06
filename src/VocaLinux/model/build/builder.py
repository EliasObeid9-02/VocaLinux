"""This module provides utilities for building and rebuilding the Listen, Attend, and Spell (LAS) model.

It includes functions to create a new model from scratch and to recompile an existing model,
resetting its optimizer state.
"""

from typing import cast
import tensorflow as tf

from VocaLinux.configs import dataset as dataset_config
from VocaLinux.configs import training as training_config
from VocaLinux.model.build.losses import safe_sparse_categorical_crossentropy
from VocaLinux.model.build.metrics import CharacterErrorRate, WordErrorRate
from VocaLinux.model.las_model import LASModel


def _compile_model(model: LASModel) -> None:
    """Compiles the given LASModel with the specified optimizer, loss, and metrics.

    This is a private helper function used by both `create_model_from_scratch`
    and `rebuild_model` to ensure consistent compilation.

    Args:
        model (LASModel): The LASModel instance to compile.
    """

    _ = model(
        [
            tf.keras.Input(shape=dataset_config.INPUT_SHAPE, dtype=tf.float32, name="input"),
            tf.keras.Input(shape=dataset_config.OUTPUT_SHAPE, dtype=tf.int32, name="output"),
        ]
    )

    loss = safe_sparse_categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=training_config.LEARNING_RATE, clipnorm=training_config.GRAD_CLIP_NORM
    )

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            "accuracy",
            CharacterErrorRate(),
            WordErrorRate(),
        ],
    )


def create_model_from_scratch() -> LASModel:
    """Creates and compiles a new Listen, Attend, and Spell (LAS) model from scratch.

    The model is initialized with configurations from `VocaLinux.configs.model`
    and compiled with an Adam optimizer, a custom learning rate schedule,
    `safe_sparse_categorical_crossentropy` loss, and `CharacterErrorRate`
    and `WordErrorRate` metrics.

    Returns:
        LASModel: A newly created and compiled LAS model.
    """
    model = LASModel()
    model = cast(LASModel, model)  # Explicit cast for type checker
    _compile_model(model)
    return model


def rebuild_model(model: LASModel) -> LASModel:
    """Recompiles an existing Listen, Attend, and Spell (LAS) model.

    This method takes an already instantiated LAS model, recompiles it with
    the standard optimizer, loss, and metrics, effectively resetting the
    optimizer's state. This is useful for continuing training with a fresh
    optimizer state or applying new compilation settings.

    Args:
        model (LASModel): The existing LASModel instance to rebuild.

    Returns:
        LASModel: The recompiled LAS model.
    """
    _compile_model(model)
    return model
