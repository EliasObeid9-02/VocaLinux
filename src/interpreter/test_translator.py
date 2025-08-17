import unittest

from interpreter.translator import Translator


class TranslatorTest(unittest.TestCase):
    def setUp(self):
        """Set up a new Translator instance for each test."""

        self.translator = Translator()

    def test_empty_input(self):
        """Test that an empty list of words results in an empty list."""

        self.assertEqual(self.translator.translate([]), [])

    def test_single_word_translation(self):
        """Test translation of single-word keywords."""

        words = ["adam", "one", "slash", "space", "echo"]
        expected = ["a", "1", "/", " ", "echo"]
        self.assertEqual(self.translator.translate(words), expected)

    def test_multi_word_translation(self):
        """Test translation of multi-word keywords."""

        words = ["double", "quote", "hello", "double", "quote"]
        expected = ['"', "hello", '"']
        self.assertEqual(self.translator.translate(words), expected)

    def test_no_translation(self):
        """Test that words not in the mapping are passed through unchanged."""

        words = ["hello", "world"]
        expected = ["hello", "world"]
        self.assertEqual(self.translator.translate(words), expected)

    def test_simple_escape(self):
        """Test escaping a single word."""

        words = ["backslash", "slash"]
        expected = ["slash"]
        self.assertEqual(self.translator.translate(words), expected)

    def test_escape_multi_word_keyword(self):
        """Test escaping a multi-word keyword like 'double quote'."""

        words = ["echo", "space", "backslash", "double", "quote"]
        expected = ["echo", " ", "double quote"]
        self.assertEqual(self.translator.translate(words), expected)

    def test_trailing_backslash(self):
        """Test a sentence ending with a backslash."""

        words = ["echo", "space", "hello", "backslash"]
        expected = ["echo", " ", "hello"]
        self.assertEqual(self.translator.translate(words), expected)

    def test_mixed_translation(self):
        """Test a mix of translated, untranslated, and escaped words."""

        words = ["echo", "space", "adam", "backslash", "space", "world"]
        expected = ["echo", " ", "a", "space", "world"]
        self.assertEqual(self.translator.translate(words), expected)

    def test_complex_command(self):
        """Test a more complex sequence of words representing a command."""

        words = [
            "m",
            "k",
            "d",
            "i",
            "r",
            "space",
            "dash",
            "p",
            "space",
            "slash",
            "home",
            "slash",
            "user",
            "slash",
            "new",
            "underscore",
            "project",
        ]
        expected = [
            "m",
            "k",
            "d",
            "i",
            "r",
            " ",
            "-",
            "p",
            " ",
            "/",
            "home",
            "/",
            "user",
            "/",
            "new",
            "_",
            "project",
        ]
        self.assertEqual(self.translator.translate(words), expected)

    def test_consecutive_escapes(self):
        """Test that 'backslash backslash' results in a literal 'backslash'."""

        words = ["backslash", "backslash", "slash"]
        expected = ["backslash", "/"]
        self.assertEqual(self.translator.translate(words), expected)


if __name__ == "__main__":
    unittest.main()
