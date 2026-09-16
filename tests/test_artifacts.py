"""Tests for immutable artifact byte storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ophanim.artifacts import ArtifactIntegrityError, FilesystemArtifactStore


class FilesystemArtifactStoreTests(unittest.TestCase):
    def test_put_is_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FilesystemArtifactStore(directory)

            first = store.put(b"same source bytes", suffix=".ionex")
            second = store.put(b"same source bytes", suffix=".ionex")

            self.assertEqual(first, second)
            self.assertTrue(first.storage_ref.endswith(".ionex"))
            self.assertEqual(store.read(first.storage_ref), b"same source bytes")
            files = [path for path in Path(directory).rglob("*") if path.is_file()]
            self.assertEqual(len(files), 1)

    def test_read_rejects_a_reference_outside_the_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FilesystemArtifactStore(directory)

            with self.assertRaisesRegex(ValueError, "escapes"):
                store.read("../../outside")

    def test_read_rejects_corrupted_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FilesystemArtifactStore(directory)
            artifact = store.put(b"original", suffix=".ionex")
            stored_path = Path(directory) / artifact.storage_ref
            stored_path.write_bytes(b"silently changed")

            with self.assertRaisesRegex(ArtifactIntegrityError, "checksum"):
                store.read(artifact.storage_ref)


if __name__ == "__main__":
    unittest.main()
