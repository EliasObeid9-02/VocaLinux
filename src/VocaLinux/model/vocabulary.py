"""Manages the vocabulary for the Listen, Attend, and Spell (LAS) model.

This module defines the set of characters and special tokens used by the model
and provides utilities for converting between character IDs and human-readable text.
"""

from typing import Dict, List, Union

import numpy as np
import tensorflow as tf


class Vocabulary:
    """Manages the mapping between characters/tokens and their integer IDs.

    Attributes:
        SPECIAL_TOKENS (List[str]): A list of special tokens used in the vocabulary.
        CHARS_LIST (List[str]): A list of standard characters.
        CHARS (List[str]): The complete list of all characters and special tokens.
        CHAR_TO_ID (Dict[str, int]): A mapping from character/token to its integer ID.
        ID_TO_CHAR (Dict[int, str]): A mapping from integer ID to its character/token.
        VOCAB_SIZE (int): The total number of unique characters/tokens in the vocabulary.
        PAD_TOKEN (int): The integer ID for the padding token.
        UNK_TOKEN (int): The integer ID for the unknown token.
        SOS_TOKEN (int): The integer ID for the start-of-sequence token.
        EOS_TOKEN (int): The integer ID for the end-of-sequence token.
    """

    def __init__(self) -> None:
        """Initializes the Vocabulary by defining special tokens, characters, and their mappings."""
        self.SPECIAL_TOKENS: List[str] = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.CHARS_LIST: List[str] = list(" abcdefghijklmnopqrstuvwxyz'")

        self.CHARS: List[str] = self.SPECIAL_TOKENS + self.CHARS_LIST
        self.CHAR_TO_ID: Dict[str, int] = {char: i for i, char in enumerate(self.CHARS)}
        self.ID_TO_CHAR: Dict[int, str] = {i: char for i, char in enumerate(self.CHARS)}
        self.VOCAB_SIZE: int = len(self.CHARS)

        self.PAD_TOKEN: int = self.CHAR_TO_ID["<pad>"]
        self.UNK_TOKEN: int = self.CHAR_TO_ID["<unk>"]
        self.SOS_TOKEN: int = self.CHAR_TO_ID["<sos>"]
        self.EOS_TOKEN: int = self.CHAR_TO_ID["<eos>"]

    def size(self) -> int:
        """Returns the total size of the vocabulary.

        Returns:
            int: The number of unique characters/tokens in the vocabulary.
        """
        return self.VOCAB_SIZE

    def ids_to_text(self, id_list: Union[List[int], np.ndarray, tf.Tensor]) -> str:
        """Converts a list/array/tensor of integer IDs to a human-readable string, ignoring specified special tokens.

        Args:
            id_list: A list, numpy array, or TensorFlow tensor of integer IDs.

        Returns:
            str: The decoded text string.
        """
        if isinstance(id_list, tf.Tensor):
            id_list = id_list.numpy()  # Convert TensorFlow tensor to numpy array

        decoded_chars: List[str] = []
        for idx in id_list:
            if idx == self.EOS_TOKEN:
                break
            if idx == self.UNK_TOKEN:
                decoded_chars.append("!")
                continue
            if idx != self.PAD_TOKEN and idx != self.SOS_TOKEN:
                char = self.ID_TO_CHAR.get(idx, "<unk>")
                decoded_chars.append(char)
        return "".join(decoded_chars)
