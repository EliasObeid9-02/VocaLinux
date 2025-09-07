"""This module provides utilities for building, rebuilding, and loading the Listen, Attend, and Spell (LAS) model.

It includes functions to create a new model from scratch, to recompile an existing model,
resetting its optimizer state, and to load a previously saved model from a file.
"""

from typing import cast
import tensorflow as tf

from VocaLinux.configs import training as training_config, dataset as dataset_config
from VocaLinux.model.build.loss import safe_sparse_categorical_crossentropy
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
        learning_rate=training_config.LEARNING_RATE,
        clipnorm=training_config.GRAD_CLIP_NORM,
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


def load_model_from_file(filepath: str) -> LASModel:
    """Loads and compiles a Listen, Attend, and Spell (LAS) model from a saved file.

    The loaded model is recompiled with the standard optimizer, loss, and metrics,
    ensuring it's ready for further training or evaluation. Custom objects
    (loss functions, metrics, and learning rate schedules) are handled during loading.

    Args:
        filepath (str): The absolute path to the saved model file.

    Returns:
        LASModel: The loaded and recompiled LAS model.
    """
    custom_objects = {
        "LASModel": LASModel,
        "safe_sparse_categorical_crossentropy": safe_sparse_categorical_crossentropy,
        "CharacterErrorRate": CharacterErrorRate,
        "WordErrorRate": WordErrorRate,
    }
    loaded_model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
    loaded_model = cast(LASModel, loaded_model)
    _compile_model(loaded_model)
    return loaded_model
