from typing import List, Optional, Tuple

from interpreter.config import MAPPINGS


class Translator:
    """A class to translate words into their corresponding commands or characters."""

    def __init__(self) -> None:
        """Initializes the Translator with a set of mappings."""

        self.mappings = MAPPINGS
        self.max_keyword_len = (
            max(len(k.split()) for k in self.mappings.keys()) if self.mappings else 0
        )

    def _find_longest_keyword(self, words: List[str]) -> Optional[str]:
        """
        Finds the longest keyword in a list of words.

        Args:
            words: A list of words to search for keywords in.

        Returns:
            The longest keyword found, or None if no keyword is found.
        """

        for length in range(min(self.max_keyword_len, len(words)), 0, -1):
            phrase = " ".join(words[:length])
            if phrase in self.mappings:
                return phrase
        return None

    def _translate_one_step(self, words: List[str], escaped: bool) -> Tuple[bool, str, int]:
        """
        Translates a single keyword or word.

        Args:
            words: A list of words to translate.
            escaped: A boolean indicating whether the next word is escaped.

        Returns:
            A tuple containing:
                - escape_next: A boolean indicating whether the next word should be escaped.
                - translated: The translated word or keyword.
                - consumed: The number of words consumed from the input list.
        """

        if not words:
            return False, "", 0

        if escaped:
            keyword = self._find_longest_keyword(words)
            if keyword:
                return False, keyword, len(keyword.split())
            else:
                return False, words[0], 1

        if words[0] == "backslash":
            return True, "", 1

        keyword = self._find_longest_keyword(words)
        if keyword:
            return False, self.mappings[keyword], len(keyword.split())
        else:
            return False, words[0], 1

    def translate(self, words: List[str]) -> List[str]:
        """
        Translates a list of words into a list of commands and characters.

        Args:
            words: A list of words to translate.

        Returns:
            A list of translated words.
        """

        translated_words: List[str] = []
        i = 0
        escaped = False
        while i < len(words):
            escape_next, translated, consumed = self._translate_one_step(words[i:], escaped)
            if translated:
                translated_words.append(translated)
            escaped = escape_next
            i += consumed
        return translated_words
