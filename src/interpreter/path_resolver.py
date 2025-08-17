import os
from difflib import get_close_matches
from typing import List


class PathResolver:
    """A class to resolve paths in a list of words."""

    def resolve(self, words: List[str]) -> List[str]:
        """
        Resolves paths in a list of words.

        Args:
            words: A list of words to resolve paths in.

        Returns:
            A list of words with paths resolved.
        """

        resolved_words: List[str] = []
        for word in words:
            if "/" in word:
                resolved_words.append(self._resolve_path(word))
            else:
                resolved_words.append(word)
        return resolved_words

    def _resolve_path(self, path: str) -> str:
        """
        Resolves a single path.

        Args:
            path: The path to resolve.

        Returns:
            The resolved path.
        """

        if path.startswith("/"):
            return self._resolve_absolute_path(path)
        else:
            return self._resolve_relative_path(path)

    def _resolve_absolute_path(self, path: str) -> str:
        """
        Resolves an absolute path.

        Args:
            path: The absolute path to resolve.

        Returns:
            The resolved absolute path.
        """

        parts = path.split("/")
        resolved_path = "/"
        for part in parts[1:]:
            if not part:
                continue
            try:
                current_dir = os.path.join(resolved_path)
                entries = os.listdir(current_dir)
                matches = get_close_matches(part, entries, n=1, cutoff=0.5)
                if matches:
                    resolved_path = os.path.join(resolved_path, matches[0])
                else:
                    resolved_path = os.path.join(resolved_path, part)
            except FileNotFoundError:
                resolved_path = os.path.join(resolved_path, part)
        return resolved_path

    def _resolve_relative_path(self, path: str) -> str:
        """
        Resolves a relative path.

        Args:
            path: The relative path to resolve.

        Returns:
            The resolved relative path.
        """

        parts = path.split("/")
        resolved_path = "."
        for part in parts:
            if not part:
                continue
            try:
                current_dir = os.path.join(resolved_path)
                entries = os.listdir(current_dir)
                matches = get_close_matches(part, entries, n=1, cutoff=0.5)
                if matches:
                    resolved_path = os.path.join(resolved_path, matches[0])
                else:
                    resolved_path = os.path.join(resolved_path, part)
            except FileNotFoundError:
                resolved_path = os.path.join(resolved_path, part)
        return os.path.relpath(resolved_path, ".")
