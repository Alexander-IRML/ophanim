"""Source acquisition, validation, parsing, and atomic commit boundary."""

from __future__ import annotations

import gzip
import os
import selectors
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from uuid import uuid4

from ophanim.artifacts import ArtifactStore
from ophanim.domain import SourceArtifact, TECObservation
from ophanim.ionex import IONEXParser
from ophanim.repositories import UnitOfWork


UNIX_COMPRESS_MAGIC = b"\x1f\x9d"
GZIP_MAGIC = b"\x1f\x8b"
DEFAULT_MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024
DEFAULT_DECOMPRESSION_TIMEOUT_SECONDS = 30.0
_DECOMPRESS_COMMANDS = (
    ("/usr/bin/gzip", "-cd"),
    ("/bin/gzip", "-cd"),
    ("/usr/bin/uncompress", "-c"),
    ("/bin/uncompress", "-c"),
)
_STDERR_PREVIEW_BYTES = 8 * 1024


def decompress_ionex(
    content: bytes,
    *,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
) -> bytes:
    """Return bounded plain IONEX bytes from plain, gzip, or Unix ``.Z`` input.

    Historic CODE products use the legacy Unix ``compress`` format, which the
    Python standard library cannot decode.  For those files this boundary uses
    a fixed, trusted system executable and drains its output incrementally so a
    corrupt or adversarial stream cannot grow an unbounded in-memory result.
    Original compressed bytes remain the artifact of record.
    """

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    if (
        not isinstance(max_decompressed_bytes, int)
        or isinstance(max_decompressed_bytes, bool)
        or max_decompressed_bytes <= 0
    ):
        raise ValueError("max_decompressed_bytes must be a positive integer")

    if content.startswith(GZIP_MAGIC):
        try:
            with gzip.GzipFile(fileobj=BytesIO(content), mode="rb") as source:
                decompressed = source.read(max_decompressed_bytes + 1)
        except (gzip.BadGzipFile, EOFError, OSError) as error:
            raise ValueError("source is not a valid gzip stream") from error
        if len(decompressed) > max_decompressed_bytes:
            raise ValueError(
                "source exceeds maximum decompressed size of "
                f"{max_decompressed_bytes} bytes"
            )
        return decompressed

    if content.startswith(UNIX_COMPRESS_MAGIC):
        return _decompress_unix_compress(
            content,
            max_decompressed_bytes=max_decompressed_bytes,
            timeout_seconds=DEFAULT_DECOMPRESSION_TIMEOUT_SECONDS,
        )

    if len(content) > max_decompressed_bytes:
        raise ValueError(
            "source exceeds maximum decompressed size of "
            f"{max_decompressed_bytes} bytes"
        )
    return content


def _decompress_unix_compress(
    content: bytes,
    *,
    max_decompressed_bytes: int,
    timeout_seconds: float,
) -> bytes:
    command = next(
        (
            candidate
            for candidate in _DECOMPRESS_COMMANDS
            if Path(candidate[0]).is_file()
            and os.access(candidate[0], os.X_OK)
        ),
        None,
    )
    if command is None:
        raise ValueError(
            "Unix-compress IONEX support requires /usr/bin/gzip or "
            "/usr/bin/uncompress"
        )

    with tempfile.TemporaryFile() as compressed:
        compressed.write(content)
        compressed.seek(0)
        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdin=compressed,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
        except OSError as error:
            raise ValueError("could not start Unix-compress decoder") from error

        assert process.stdout is not None
        assert process.stderr is not None
        selector = selectors.DefaultSelector()
        output = bytearray()
        error_output = bytearray()
        try:
            for stream, name in (
                (process.stdout, "stdout"),
                (process.stderr, "stderr"),
            ):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)

            deadline = time.monotonic() + timeout_seconds
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    _terminate_process(process)
                    raise ValueError("Unix-compress decompression timed out")
                events = selector.select(remaining)
                if not events:
                    _terminate_process(process)
                    raise ValueError("Unix-compress decompression timed out")
                for key, _ in events:
                    try:
                        chunk = os.read(key.fd, 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stdout":
                        available = max_decompressed_bytes + 1 - len(output)
                        output.extend(chunk[:available])
                        if (
                            len(chunk) > available
                            or len(output) > max_decompressed_bytes
                        ):
                            _terminate_process(process)
                            raise ValueError(
                                "source exceeds maximum decompressed size of "
                                f"{max_decompressed_bytes} bytes"
                            )
                    elif len(error_output) < _STDERR_PREVIEW_BYTES:
                        available = _STDERR_PREVIEW_BYTES - len(error_output)
                        error_output.extend(chunk[:available])

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_process(process)
                raise ValueError("Unix-compress decompression timed out")
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                _terminate_process(process)
                raise ValueError(
                    "Unix-compress decompression timed out"
                ) from error
            if return_code != 0:
                detail = error_output.decode("utf-8", errors="replace").strip()
                suffix = f": {detail}" if detail else ""
                raise ValueError(
                    "source is not a valid Unix-compress stream" + suffix
                )
            return bytes(output)
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
            if process.poll() is None:
                _terminate_process(process)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    """Identifies one remote or local source product to ingest."""

    provider: str
    product: str
    source_uri: str
    revision: str | None = None
    revision_priority: int = 0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("provider", self.provider),
            ("product", self.product),
            ("source_uri", self.source_uri),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be None or non-empty")
        if (
            not isinstance(self.revision_priority, int)
            or isinstance(self.revision_priority, bool)
            or self.revision_priority < 0
        ):
            raise ValueError("revision_priority must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary returned only after an ingestion transaction succeeds."""

    artifact: SourceArtifact
    observation_count: int
    already_present: bool = False


@dataclass(frozen=True, slots=True)
class LoadedSource:
    """Bytes and a filename hint returned by a source loader."""

    content: bytes
    filename: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must not be empty")


class SourceLoader(Protocol):
    """Acquire source bytes without parsing or persisting them."""

    def load(self, source_uri: str) -> LoadedSource: ...


class DataIngester(Protocol):
    """Acquire and atomically persist one source artifact and its observations."""

    def ingest(self, request: IngestionRequest) -> IngestionResult: ...


class StandardSourceLoader:
    """Load local paths, ``file://`` URIs, and HTTP(S) sources.

    This is intentionally small. Authentication, provider discovery, retries,
    and conditional requests belong in provider-specific loaders later.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes

    def load(self, source_uri: str) -> LoadedSource:
        parsed = urlparse(source_uri)
        if parsed.scheme in {"http", "https"}:
            with urlopen(source_uri, timeout=self._timeout_seconds) as response:  # noqa: S310
                content = response.read(self._max_bytes + 1)
            filename = Path(unquote(parsed.path)).name or "download.ionex"
        elif parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("remote file URI hosts are not supported")
            path = Path(unquote(parsed.path))
            content = self._read_local(path)
            filename = path.name
        elif not parsed.scheme:
            path = Path(source_uri)
            content = self._read_local(path)
            filename = path.name
        else:
            raise ValueError(f"unsupported source URI scheme: {parsed.scheme!r}")

        if len(content) > self._max_bytes:
            raise ValueError(f"source exceeds maximum size of {self._max_bytes} bytes")
        return LoadedSource(content=content, filename=filename)

    def _read_local(self, path: Path) -> bytes:
        with path.open("rb") as source:
            return source.read(self._max_bytes + 1)


class IONEXDataIngester:
    """Concrete, idempotent IONEX ingestion service for the v0 pipeline."""

    def __init__(
        self,
        *,
        source_loader: SourceLoader,
        artifact_store: ArtifactStore,
        parser: IONEXParser,
        unit_of_work_factory: Callable[[], UnitOfWork],
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
        max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    ) -> None:
        if (
            not isinstance(parser.parser_version, str)
            or not parser.parser_version.strip()
        ):
            raise ValueError("parser_version must not be empty")
        if max_decompressed_bytes <= 0:
            raise ValueError("max_decompressed_bytes must be positive")
        self._source_loader = source_loader
        self._artifact_store = artifact_store
        self._parser = parser
        self._unit_of_work_factory = unit_of_work_factory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._max_decompressed_bytes = max_decompressed_bytes

    def ingest(self, request: IngestionRequest) -> IngestionResult:
        loaded = self._source_loader.load(request.source_uri)
        return self.ingest_loaded(request, loaded)

    def ingest_loaded(
        self,
        request: IngestionRequest,
        loaded: LoadedSource,
        *,
        observation_validator: Callable[
            [tuple[TECObservation, ...]], None
        ]
        | None = None,
    ) -> IngestionResult:
        """Persist already-acquired bytes while retaining their source URI.

        Provider-specific acquisition can validate and download a source once,
        then hand it across the same parsing and atomic commit boundary used by
        :meth:`ingest`.  ``request.source_uri`` remains the durable provenance;
        ``loaded.filename`` is only a storage suffix hint.
        """

        checksum_sha256 = sha256(loaded.content).hexdigest()

        with self._unit_of_work_factory() as unit_of_work:
            existing = unit_of_work.artifacts.find_acquisition(
                provider=request.provider,
                product=request.product,
                parser_version=self._parser.parser_version,
                revision_priority=request.revision_priority,
                revision=request.revision,
                source_uri=request.source_uri,
                checksum_sha256=checksum_sha256,
            )
            if existing is not None:
                return self._existing_result(
                    artifact=existing,
                    unit_of_work=unit_of_work,
                )

        acquisition_started_at = self._clock()
        if (
            acquisition_started_at.tzinfo is None
            or acquisition_started_at.utcoffset() is None
        ):
            raise ValueError("ingestion clock must return a timezone-aware datetime")
        provisional_artifact = SourceArtifact(
            artifact_id=self._id_factory(),
            provider=request.provider,
            product=request.product,
            parser_version=self._parser.parser_version,
            revision_priority=request.revision_priority,
            checksum_sha256=checksum_sha256,
            storage_ref="",
            ingested_at=acquisition_started_at,
            revision=request.revision,
            source_uri=request.source_uri,
        )
        parse_content = self._decompress_if_needed(loaded.content)
        observations = tuple(
            self._parser.parse(
                artifact=provisional_artifact,
                stream=BytesIO(parse_content),
            )
        )
        if not observations:
            raise ValueError("IONEX artifact contains no TEC observations")
        if any(
            observation.artifact_id != provisional_artifact.artifact_id
            for observation in observations
        ):
            raise ValueError("parser returned an observation for a different artifact")
        if observation_validator is not None:
            observation_validator(observations)

        stored = self._artifact_store.put(
            loaded.content,
            suffix=self._artifact_suffix(loaded.filename),
        )
        if stored.checksum_sha256 != checksum_sha256:
            raise RuntimeError("artifact store returned a mismatched checksum")

        with self._unit_of_work_factory() as unit_of_work:
            # Recheck inside the write transaction in case another worker won
            # the race after the first idempotency lookup.
            existing = unit_of_work.artifacts.find_acquisition(
                provider=request.provider,
                product=request.product,
                parser_version=self._parser.parser_version,
                revision_priority=request.revision_priority,
                revision=request.revision,
                source_uri=request.source_uri,
                checksum_sha256=checksum_sha256,
            )
            if existing is not None:
                return self._existing_result(
                    artifact=existing,
                    unit_of_work=unit_of_work,
                )
            # Stamp availability only after acquiring the final writer lock.
            # Any transaction that completed while parsing or artifact storage
            # was in progress therefore precedes this acquisition durably.
            ingested_at = self._clock()
            if ingested_at.tzinfo is None or ingested_at.utcoffset() is None:
                raise ValueError(
                    "ingestion clock must return a timezone-aware datetime"
                )
            if ingested_at < acquisition_started_at:
                raise ValueError("ingestion clock moved backwards during acquisition")
            artifact = replace(
                provisional_artifact,
                storage_ref=stored.storage_ref,
                ingested_at=ingested_at,
            )
            unit_of_work.artifacts.add(artifact)
            unit_of_work.tec_observations.add_many(observations)
            unit_of_work.commit()

        return IngestionResult(
            artifact=artifact,
            observation_count=len(observations),
        )

    def _existing_result(
        self,
        *,
        artifact: SourceArtifact,
        unit_of_work: UnitOfWork,
    ) -> IngestionResult:
        content = self._artifact_store.read(artifact.storage_ref)
        if sha256(content).hexdigest() != artifact.checksum_sha256:
            raise RuntimeError(
                "stored source bytes do not match acquisition metadata checksum"
            )
        count = len(unit_of_work.tec_observations.for_artifact(artifact.artifact_id))
        if count == 0:
            raise RuntimeError("stored acquisition has no parsed TEC observations")
        return IngestionResult(
            artifact=artifact,
            observation_count=count,
            already_present=True,
        )

    def _decompress_if_needed(self, content: bytes) -> bytes:
        return decompress_ionex(
            content,
            max_decompressed_bytes=self._max_decompressed_bytes,
        )

    @staticmethod
    def _artifact_suffix(filename: str) -> str:
        suffixes = Path(filename).suffixes
        return "".join(suffixes[-2:]) if suffixes else ".ionex"
