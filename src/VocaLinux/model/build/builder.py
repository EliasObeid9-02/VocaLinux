"""This module provides utilities for building, rebuilding, and loading the Listen, Attend, and Spell (LAS) model.

It includes functions to create a new model from scratch, to recompile an existing model,
resetting its optimizer state, and to load a previously saved model from a file.
"""

from typing import cast, Optional
import tensorflow as tf

from VocaLinux.model.layers import PBLSTMLayer, Listener, LocationAwareAttention, Speller
from VocaLinux.configs import training as training_config, dataset as dataset_config
from VocaLinux.model.build.loss import safe_sparse_categorical_crossentropy
from VocaLinux.model.build.metrics import CharacterErrorRate, WordErrorRate
from VocaLinux.model.las_model import LASModel


def _compile_model(model: LASModel, strategy: Optional[tf.distribute.Strategy] = None) -> None:
    """Compiles the given LASModel with the specified optimizer, loss, and metrics.

    This is a private helper function used by both `create_model_from_scratch`
    and `rebuild_model` to ensure consistent compilation.

    Args:
        model (LASModel): The LASModel instance to compile.
        strategy (Optional[tf.distribute.Strategy]): The distribution strategy to use for compilation.
    """

    def compile_fn():
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

    if strategy:
        with strategy.scope():
            compile_fn()
    else:
        compile_fn()

    _ = model(
        [
            tf.keras.Input(shape=dataset_config.INPUT_SHAPE, dtype=tf.float32, name="input"),
            tf.keras.Input(shape=dataset_config.OUTPUT_SHAPE, dtype=tf.int32, name="output"),
        ]
    )


def create_model_from_scratch(strategy: Optional[tf.distribute.Strategy] = None) -> LASModel:
    """Creates and compiles a new Listen, Attend, and Spell (LAS) model from scratch.

    Args:
        strategy (Optional[tf.distribute.Strategy]): The distribution strategy to use for compilation.

    Returns:
        LASModel: A newly created and compiled LAS model.
    """

    def model_fn():
        return LASModel()

    if strategy:
        with strategy.scope():
            model = model_fn()
    else:
        model = model_fn()

    model = cast(LASModel, model)
    _compile_model(model, strategy)
    return model


def rebuild_model(model: LASModel, strategy: Optional[tf.distribute.Strategy] = None) -> LASModel:
    """Recompiles an existing Listen, Attend, and Spell (LAS) model.

    Args:
        model (LASModel): The existing LASModel instance to rebuild.
        strategy (Optional[tf.distribute.Strategy]): The distribution strategy to use for compilation.

    Returns:
        LASModel: The recompiled LAS model.
    """
    _compile_model(model, strategy)
    return model


def load_model_from_file(
    filepath: str, strategy: Optional[tf.distribute.Strategy] = None
) -> LASModel:
    """Loads and compiles a Listen, Attend, and Spell (LAS) model from a saved file.

    Args:
        filepath (str): The absolute path to the saved model file.
        strategy (Optional[tf.distribute.Strategy]): The distribution strategy to use for compilation.

    Returns:
        LASModel: The loaded and recompiled LAS model.
    """
    custom_objects = {
        "PBLSTMLayer": PBLSTMLayer,
        "Listener": Listener,
        "LocationAwareAttention": LocationAwareAttention,
        "Speller": Speller,
        "LASModel": LASModel,
        "CharacterErrorRate": CharacterErrorRate,
        "WordErrorRate": WordErrorRate,
    }

    def load_fn():
        return tf.keras.models.load_model(filepath, custom_objects=custom_objects)

    if strategy:
        with strategy.scope():
            loaded_model = load_fn()
    else:
        loaded_model = load_fn()

    loaded_model = cast(LASModel, loaded_model)
    _compile_model(loaded_model, strategy)
    return loaded_model
