"""Keep runtime, distributable metadata and release notes aligned."""
from pathlib import Path
import tomllib
import unittest

from ophanim import __version__


class ReleaseMetadataTests(unittest.TestCase):
    def test_runtime_package_and_changelog_agree(self):
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["version"], __version__)
        self.assertIn(f"## [{__version__}] - ", (root / "CHANGELOG.md").read_text())


if __name__ == "__main__":
    unittest.main()
