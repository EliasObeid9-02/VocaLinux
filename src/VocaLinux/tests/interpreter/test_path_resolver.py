import os
import unittest

from VocaLinux.interpreter.path_resolver import PathResolver


class PathResolverTest(unittest.TestCase):
    def setUp(self):
        """Set up a new PathResolver instance for each test."""

        self.resolver = PathResolver()
        self.original_cwd = os.getcwd()
        # Assuming the test is run from the project root directory
        self.project_root = "/home/elias/Coding/Projects/VocaLinux"
        os.chdir(self.project_root)

    def tearDown(self):
        """Restore the original working directory after each test."""

        os.chdir(self.original_cwd)

    def test_no_paths(self):
        """Test that a list of words without paths is returned unchanged."""

        words = ["echo", "hello", "world"]
        self.assertEqual(self.resolver.resolve(words), words)

    def test_simple_relative_path(self):
        """Test resolving a simple, correct relative path."""

        words = ["ls", "src/interpreter"]
        expected = ["ls", "src/interpreter"]
        self.assertEqual(self.resolver.resolve(words), expected)

    def test_relative_path_with_typo(self):
        """Test resolving a relative path with a typo in one component."""

        # Assuming 'interpret' is a typo for 'interpreter'
        words = ["cat", "src/interpret/translator.py"]
        expected = ["cat", "src/interpreter/translator.py"]
        self.assertEqual(self.resolver.resolve(words), expected)

    def test_absolute_path(self):
        """Test resolving a correct absolute path."""

        abs_path = os.path.join(self.project_root, "README.md")
        words = ["cat", abs_path]
        self.assertEqual(self.resolver.resolve(words), words)

    def test_absolute_path_with_typo(self):
        """Test resolving an absolute path with a typo."""

        # Assuming 'sorc' is a typo for 'src'
        bad_path = os.path.join(self.project_root, "sorc/interpret.py")
        good_path = os.path.join(self.project_root, "src/interpret.py")
        words = ["cat", bad_path]
        expected = ["cat", good_path]
        self.assertEqual(self.resolver.resolve(words), expected)

    def test_path_with_dot(self):
        """Test resolving a path starting with './'."""

        words = ["ls", "./src"]
        expected = ["ls", "src"]  # os.path.relpath will simplify ./src to src
        self.assertEqual(self.resolver.resolve(words), expected)

    def test_path_with_double_dot(self):
        """Test resolving a path with '../'."""

        # from inside src/interpreter, go up to src
        os.chdir(os.path.join(self.project_root, "src/interpreter"))
        words = ["ls", "../"]
        # The resolver will run os.listdir('.') in the CWD (src/interpreter)
        # then try to match '..' which isn't a dir entry, so it will just use '..'
        # The final os.path.relpath will resolve it correctly relative to the CWD.
        # Let's test the final output from the root perspective.
        # The resolver's _resolve_relative_path will build './../' and relpath will make it '..'
        expected = ["ls", ".."]
        self.assertEqual(self.resolver.resolve(words), expected)
        os.chdir(self.project_root)  # change back for other tests

    def test_nonexistent_path(self):
        """Test that a path with a non-matching component is not changed."""

        words = ["ls", "nonexistent/directory"]
        self.assertEqual(self.resolver.resolve(words), words)

    def test_mixed_words_and_paths(self):
        """Test a list containing a mix of paths and regular words."""

        words = ["cp", "./README.md", "doc/backup_readme.md"]
        expected = ["cp", "README.md", "doc/backup_readme.md"]
        self.assertEqual(self.resolver.resolve(words), expected)


if __name__ == "__main__":
    unittest.main()
