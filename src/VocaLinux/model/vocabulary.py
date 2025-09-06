from typing import Union

import numpy as np
import tensorflow as tf

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
CHARS_LIST = list(" abcdefghijklmnopqrstuvwxyz'")

CHARS = SPECIAL_TOKENS + CHARS_LIST
CHAR_TO_ID = {char: i for i, char in enumerate(CHARS)}
ID_TO_CHAR = {i: char for i, char in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

PAD_TOKEN = CHAR_TO_ID["<pad>"]
UNK_TOKEN = CHAR_TO_ID["<unk>"]
SOS_TOKEN = CHAR_TO_ID["<sos>"]
EOS_TOKEN = CHAR_TO_ID["<eos>"]


def ids_to_text(id_list: Union[list, np.ndarray, tf.Tensor]) -> str:
    """
    Converts a list/array/tensor of integer IDs to a human-readable string,
    ignoring specified special tokens.

    Args:
        id_list: A list, numpy array, or TensorFlow tensor of integer IDs.

    Returns:
        str: The decoded text string.
    """
    if isinstance(id_list, tf.Tensor):
        id_list = id_list.numpy()  # Convert TensorFlow tensor to numpy array

    decoded_chars = []
    ignored_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    for idx in id_list:
        if idx == EOS_TOKEN:
            break
        if idx == UNK_TOKEN:
            decoded_chars.append("!")
            continue
        if idx != PAD_TOKEN and idx != SOS_TOKEN:
            char = ID_TO_CHAR.get(idx, "<unk>")
            decoded_chars.append(char)
    return "".join(decoded_chars)
