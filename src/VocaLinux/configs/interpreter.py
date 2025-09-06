"""
This module contains the configuration for the interpreter.
It includes the vocabulary and mappings used for translating spoken words into text.
"""

from typing import Dict, List

PHONETIC_ALPHABET: Dict[str, str] = {
    "adam": "a",
    "boy": "b",
    "charlie": "c",
    "david": "d",
    "edward": "e",
    "frank": "f",
    "george": "g",
    "henry": "h",
    "ida": "i",
    "john": "j",
    "king": "k",
    "lincoln": "l",
    "mary": "m",
    "nora": "n",
    "ocean": "o",
    "paul": "p",
    "queen": "q",
    "robert": "r",
    "sam": "s",
    "tom": "t",
    "union": "u",
    "victor": "v",
    "william": "w",
    "x-ray": "x",
    "young": "y",
    "zebra": "z",
}

SPECIAL_CHARS: Dict[str, str] = {
    "semicolon": ";",
    "slash": "/",
    "double quote": '"',
    "single quote": "'",
    "dash": "-",
    "hyphen": "-",
    "dot": ".",
    "double dot": "..",
    "space": " ",
    "pipe": "|",
    "greater than": ">",
    "underscore": "_",
}

DIGIT_WORDS: Dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

VOCABULARY: List[str] = (
    list(PHONETIC_ALPHABET.keys())
    + list(SPECIAL_CHARS.keys())
    + list(DIGIT_WORDS.keys())
)

MAPPINGS: Dict[str, str] = {
    **PHONETIC_ALPHABET,
    **SPECIAL_CHARS,
    **DIGIT_WORDS,
}
