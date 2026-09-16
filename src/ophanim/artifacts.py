"""Immutable byte storage for source and model artifacts.

Database repositories own searchable metadata. This module owns the bytes those
records point at. The filesystem implementation is content-addressed so a retry
does not create a second copy of the same file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class ArtifactIntegrityError(OSError):
    """Stored bytes no longer match their content-addressed reference."""


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Result of durably storing an immutable byte payload."""

    checksum_sha256: str
    storage_ref: str
    byte_count: int


class ArtifactStore(Protocol):
    """Storage boundary for immutable artifact bytes."""

    def put(self, content: bytes, *, suffix: str = "") -> StoredArtifact: ...

    def read(self, storage_ref: str) -> bytes: ...


class FilesystemArtifactStore:
    """Content-addressed artifact storage rooted in one directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes, *, suffix: str = "") -> StoredArtifact:
        digest = sha256(content).hexdigest()
        safe_suffix = self._safe_suffix(suffix)
        relative_path = Path(digest[:2]) / f"{digest}{safe_suffix}"
        destination = self._root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not destination.exists():
            with NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{digest}.",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            try:
                temporary_path.replace(destination)
                self._sync_directory(destination.parent)
            finally:
                temporary_path.unlink(missing_ok=True)
        elif sha256(destination.read_bytes()).hexdigest() != digest:
            raise ArtifactIntegrityError(
                f"content-addressed artifact is corrupted: {relative_path}"
            )

        return StoredArtifact(
            checksum_sha256=digest,
            storage_ref=relative_path.as_posix(),
            byte_count=len(content),
        )

    def read(self, storage_ref: str) -> bytes:
        path = (self._root / storage_ref).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("artifact storage reference escapes its configured root")
        expected_checksum = self._checksum_from_ref(path, storage_ref)
        content = path.read_bytes()
        if sha256(content).hexdigest() != expected_checksum:
            raise ArtifactIntegrityError(
                f"stored artifact fails checksum verification: {storage_ref}"
            )
        return content

    def _checksum_from_ref(self, path: Path, storage_ref: str) -> str:
        try:
            relative_path = path.relative_to(self._root)
        except ValueError as error:
            raise ValueError(
                "artifact storage reference escapes its configured root"
            ) from error
        if len(relative_path.parts) != 2:
            raise ValueError(f"invalid artifact storage reference: {storage_ref!r}")

        directory, filename = relative_path.parts
        checksum = filename.split(".", maxsplit=1)[0]
        if (
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or directory != checksum[:2]
        ):
            raise ValueError(f"invalid artifact storage reference: {storage_ref!r}")
        return checksum

    @staticmethod
    def _safe_suffix(suffix: str) -> str:
        if not suffix:
            return ""
        candidate = suffix.lower()
        if (
            not candidate.startswith(".")
            or len(candidate) > 32
            or any(
                character not in ".abcdefghijklmnopqrstuvwxyz0123456789"
                for character in candidate
            )
        ):
            return ""
        return candidate

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
