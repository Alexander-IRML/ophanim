"""Integration tests for idempotent source ingestion."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Event, Lock

from ophanim.artifacts import ArtifactIntegrityError, FilesystemArtifactStore
from ophanim.domain import TECObservation
from ophanim.ingestion import (
    IONEXDataIngester,
    IngestionRequest,
    LoadedSource,
    StandardSourceLoader,
    decompress_ionex,
)
from ophanim.ionex import IONEXV1Parser
from ophanim.sqlite import SQLiteUnitOfWork


UTC = timezone.utc
INGESTED_AT = datetime(2026, 1, 2, tzinfo=UTC)


def _record(payload: str, label: str) -> str:
    return f"{payload:<60}{label:<20}\n"


def _grid_payload(*values: float) -> str:
    return "  " + "".join(f"{value:6.1f}" for value in values)


def _data_record(*values: int) -> str:
    return "".join(f"{value:5d}" for value in values).ljust(80) + "\n"


def _unix_compress_literals(content: bytes) -> bytes:
    if len(content) > 255:
        raise ValueError("literal-only .Z fixture must stay on nine-bit codes")
    output = bytearray(b"\x1f\x9d\x09")
    accumulator = 0
    bit_count = 0
    for code in content:
        accumulator |= code << bit_count
        bit_count += 9
        while bit_count >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bit_count -= 8
    if bit_count:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _wrap_endpoint_ionex() -> str:
    return "".join(
        (
            _record(f"{1.0:8.1f}{'':12}I{'':39}", "IONEX VERSION / TYPE"),
            _record(f"{1:6d}", "# OF MAPS IN FILE"),
            _record(f"{2:6d}", "MAP DIMENSION"),
            _record(_grid_payload(450.0, 450.0, 0.0), "HGT1 / HGT2 / DHGT"),
            _record(_grid_payload(32.5, 32.5, 0.0), "LAT1 / LAT2 / DLAT"),
            _record(_grid_payload(0.0, 360.0, 120.0), "LON1 / LON2 / DLON"),
            _record(f"{-1:6d}", "EXPONENT"),
            _record("", "END OF HEADER"),
            _record(f"{1:6d}", "START OF TEC MAP"),
            _record(
                f"{2024:6d}{1:6d}{2:6d}{0:6d}{0:6d}{0:6d}",
                "EPOCH OF CURRENT MAP",
            ),
            _record(
                _grid_payload(32.5, 0.0, 360.0, 120.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            ),
            _data_record(10, 20, 30, 10),
            _record(f"{1:6d}", "END OF TEC MAP"),
            _record("", "END OF FILE"),
        )
    )


class _BarrierParser:
    parser_version = "barrier-parser/1"

    def __init__(self, barrier: Barrier) -> None:
        self._barrier = barrier

    def parse(self, *, artifact, stream):
        self._barrier.wait(timeout=5)
        yield TECObservation(
            artifact_id=artifact.artifact_id,
            observed_at=INGESTED_AT,
            latitude_degrees=30.0,
            longitude_degrees=-98.0,
            vtec_tecu=12.0,
        )


class _FailingParser:
    parser_version = "failing-parser/1"

    def parse(self, *, artifact, stream):
        raise ValueError("invalid IONEX")
        yield  # pragma: no cover


class _NeverCalledParser:
    parser_version = "never-called-parser/1"

    def parse(self, *, artifact, stream):
        raise AssertionError("oversized decompressed input reached the parser")
        yield  # pragma: no cover


class _NeverCalledLoader:
    def load(self, source_uri: str) -> LoadedSource:
        raise AssertionError("already-loaded ingestion called the source loader")


class _ValueParser:
    def __init__(self, *, parser_version: str, value: float) -> None:
        self.parser_version = parser_version
        self._value = value

    def parse(self, *, artifact, stream):
        yield TECObservation(
            artifact_id=artifact.artifact_id,
            observed_at=INGESTED_AT,
            latitude_degrees=30.0,
            longitude_degrees=-98.0,
            vtec_tecu=self._value,
        )


class _ExpectedContentParser:
    parser_version = "expected-content-parser/1"

    def __init__(self, expected: bytes) -> None:
        self._expected = expected

    def parse(self, *, artifact, stream):
        if stream.read() != self._expected:
            raise AssertionError("parser did not receive decompressed bytes")
        yield TECObservation(
            artifact_id=artifact.artifact_id,
            observed_at=INGESTED_AT,
            latitude_degrees=30.0,
            longitude_degrees=-98.0,
            vtec_tecu=12.0,
        )


class _PausingParser:
    parser_version = "pausing-parser/1"

    def __init__(self, parse_started: Event, continue_parse: Event) -> None:
        self._parse_started = parse_started
        self._continue_parse = continue_parse

    def parse(self, *, artifact, stream):
        self._parse_started.set()
        if not self._continue_parse.wait(timeout=5):
            raise TimeoutError("test did not release parser")
        yield TECObservation(
            artifact_id=artifact.artifact_id,
            observed_at=INGESTED_AT,
            latitude_degrees=30.0,
            longitude_degrees=-98.0,
            vtec_tecu=12.0,
        )


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.database = self.root / "ophanim.sqlite3"
        self.artifacts = self.root / "artifacts"
        self.source = self.root / "source.ionex"
        self.source.write_bytes(b"test source bytes")
        self.request = IngestionRequest(
            provider="test-provider",
            product="test-product",
            source_uri=str(self.source),
            revision="r1",
        )

    def test_concurrent_retry_returns_one_shared_acquisition(self) -> None:
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_BarrierParser(Barrier(2)),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(ingester.ingest, self.request) for _ in range(2)]
            results = [future.result(timeout=10) for future in futures]

        self.assertEqual(
            {result.artifact.artifact_id for result in results},
            {results[0].artifact.artifact_id},
        )
        self.assertEqual(sorted(result.already_present for result in results), [False, True])
        with SQLiteUnitOfWork(self.database) as unit_of_work:
            observations = unit_of_work.tec_observations.for_artifact(
                results[0].artifact.artifact_id
            )
            self.assertEqual(len(observations), 1)

    def test_wrap_endpoint_count_matches_persistence_and_retry(self) -> None:
        self.source.write_text(_wrap_endpoint_ionex(), encoding="ascii")
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=IONEXV1Parser(),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )

        first = ingester.ingest(self.request)
        retry = ingester.ingest(self.request)

        self.assertEqual(first.observation_count, 3)
        self.assertEqual(retry.observation_count, 3)
        self.assertTrue(retry.already_present)
        with SQLiteUnitOfWork(self.database) as unit_of_work:
            persisted = unit_of_work.tec_observations.for_artifact(
                first.artifact.artifact_id
            )
        self.assertEqual(len(persisted), 3)

    def test_availability_is_stamped_after_acquiring_final_write_lock(self) -> None:
        parse_started = Event()
        continue_parse = Event()
        final_write_attempted = Event()
        blocker_released = Event()
        factory_lock = Lock()
        factory_calls = 0
        clock_calls = 0

        def unit_of_work_factory():
            nonlocal factory_calls
            with factory_lock:
                factory_calls += 1
                if factory_calls == 2:
                    final_write_attempted.set()
            return SQLiteUnitOfWork(self.database)

        def clock() -> datetime:
            nonlocal clock_calls
            clock_calls += 1
            if clock_calls == 1:
                return INGESTED_AT
            if not blocker_released.is_set():
                raise AssertionError("availability was stamped before the write lock")
            return INGESTED_AT.replace(second=1)

        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_PausingParser(parse_started, continue_parse),
            unit_of_work_factory=unit_of_work_factory,
            clock=clock,
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ingester.ingest, self.request)
            self.assertTrue(parse_started.wait(timeout=5))
            with SQLiteUnitOfWork(self.database):
                continue_parse.set()
                self.assertTrue(final_write_attempted.wait(timeout=5))
                blocker_released.set()
            result = future.result(timeout=10)

        self.assertEqual(result.artifact.ingested_at, INGESTED_AT.replace(second=1))
        self.assertEqual(clock_calls, 2)

    def test_parse_failure_does_not_write_artifact_bytes_or_metadata(self) -> None:
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_FailingParser(),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )

        with self.assertRaisesRegex(ValueError, "invalid IONEX"):
            ingester.ingest(self.request)

        stored_files = [path for path in self.artifacts.rglob("*") if path.is_file()]
        self.assertEqual(stored_files, [])
        with SQLiteUnitOfWork(self.database) as unit_of_work:
            self.assertIsNone(
                unit_of_work.artifacts.find_acquisition(
                    provider=self.request.provider,
                    product=self.request.product,
                    parser_version=_FailingParser.parser_version,
                    revision_priority=self.request.revision_priority,
                    revision=self.request.revision,
                    source_uri=self.request.source_uri,
                    checksum_sha256=sha256(self.source.read_bytes()).hexdigest(),
                )
            )

    def test_ingest_loaded_preserves_remote_provenance_without_redownloading(self) -> None:
        source_uri = "https://example.test/maps/source.INX.gz"
        request = replace(self.request, source_uri=source_uri)
        ingester = IONEXDataIngester(
            source_loader=_NeverCalledLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_ValueParser(parser_version="test-parser/1", value=12.0),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )

        result = ingester.ingest_loaded(
            request,
            LoadedSource(content=b"already downloaded", filename="source.INX.gz"),
        )

        self.assertEqual(result.artifact.source_uri, source_uri)
        self.assertTrue(result.artifact.storage_ref.endswith(".inx.gz"))
        self.assertEqual(result.observation_count, 1)

    def test_decompressed_size_limit_rejects_high_ratio_gzip_before_parse(self) -> None:
        self.source.write_bytes(gzip.compress(b"x" * 65))
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_NeverCalledParser(),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
            max_decompressed_bytes=64,
        )

        with self.assertRaisesRegex(ValueError, "maximum decompressed size"):
            ingester.ingest(self.request)

        stored_files = [path for path in self.artifacts.rglob("*") if path.is_file()]
        self.assertEqual(stored_files, [])

    @unittest.skipUnless(
        Path("/usr/bin/gzip").is_file() or Path("/bin/gzip").is_file(),
        "system gzip is required for legacy .Z compatibility",
    )
    def test_unix_compress_source_is_bounded_and_decompressed_before_parse(self) -> None:
        plain = b"legacy IONEX bytes"
        compressed = _unix_compress_literals(plain)
        self.source = self.root / "CODG0020.06I.Z"
        self.source.write_bytes(compressed)
        request = replace(self.request, source_uri=str(self.source))
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_ExpectedContentParser(plain),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )

        result = ingester.ingest(request)

        self.assertEqual(result.observation_count, 1)
        self.assertTrue(result.artifact.storage_ref.endswith(".06i.z"))
        self.assertEqual(
            (self.artifacts / result.artifact.storage_ref).read_bytes(),
            compressed,
        )
        with self.assertRaisesRegex(ValueError, "maximum decompressed size"):
            decompress_ionex(
                _unix_compress_literals(b"x" * 65),
                max_decompressed_bytes=64,
            )

    @unittest.skipUnless(
        Path("/usr/bin/gzip").is_file() or Path("/bin/gzip").is_file(),
        "system gzip is required for legacy .Z compatibility",
    )
    def test_malformed_unix_compress_is_reported_as_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid Unix-compress"):
            decompress_ionex(
                b"\x1f\x9d\x90garbage",
                max_decompressed_bytes=1024,
            )

    def test_malformed_gzip_is_reported_as_invalid_input(self) -> None:
        self.source.write_bytes(b"\x1f\x8bnot-a-complete-gzip-stream")
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_NeverCalledParser(),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )

        with self.assertRaisesRegex(ValueError, "not a valid gzip stream"):
            ingester.ingest(self.request)

        stored_files = [path for path in self.artifacts.rglob("*") if path.is_file()]
        self.assertEqual(stored_files, [])

    def test_new_parser_version_reparses_same_acquisition_into_new_raw_set(self) -> None:
        def ingester(parser_version: str, value: float) -> IONEXDataIngester:
            return IONEXDataIngester(
                source_loader=StandardSourceLoader(),
                artifact_store=FilesystemArtifactStore(self.artifacts),
                parser=_ValueParser(
                    parser_version=parser_version,
                    value=value,
                ),
                unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
                clock=lambda: INGESTED_AT,
            )

        first = ingester("test-parser/1", 12.0).ingest(self.request)
        retry = ingester("test-parser/1", 999.0).ingest(self.request)
        reparsed = ingester("test-parser/2", 13.0).ingest(self.request)
        prioritized = ingester("test-parser/2", 14.0).ingest(
            replace(self.request, revision_priority=5)
        )

        self.assertFalse(first.already_present)
        self.assertTrue(retry.already_present)
        self.assertFalse(reparsed.already_present)
        self.assertFalse(prioritized.already_present)
        self.assertEqual(first.artifact, retry.artifact)
        self.assertNotEqual(first.artifact.artifact_id, reparsed.artifact.artifact_id)
        self.assertNotEqual(
            reparsed.artifact.artifact_id,
            prioritized.artifact.artifact_id,
        )
        self.assertEqual(first.artifact.checksum_sha256, reparsed.artifact.checksum_sha256)
        self.assertEqual(first.artifact.storage_ref, reparsed.artifact.storage_ref)
        self.assertEqual(first.artifact.parser_version, "test-parser/1")
        self.assertEqual(reparsed.artifact.parser_version, "test-parser/2")
        self.assertEqual(prioritized.artifact.revision_priority, 5)

        with SQLiteUnitOfWork(self.database) as unit_of_work:
            first_raw = unit_of_work.tec_observations.for_artifact(
                first.artifact.artifact_id
            )
            reparsed_raw = unit_of_work.tec_observations.for_artifact(
                reparsed.artifact.artifact_id
            )
            prioritized_raw = unit_of_work.tec_observations.for_artifact(
                prioritized.artifact.artifact_id
            )
        self.assertEqual(first_raw[0].vtec_tecu, 12.0)
        self.assertEqual(reparsed_raw[0].vtec_tecu, 13.0)
        self.assertEqual(prioritized_raw[0].vtec_tecu, 14.0)
        stored_files = [path for path in self.artifacts.rglob("*") if path.is_file()]
        self.assertEqual(len(stored_files), 1)

    def test_idempotent_retry_detects_corrupted_source_bytes(self) -> None:
        ingester = IONEXDataIngester(
            source_loader=StandardSourceLoader(),
            artifact_store=FilesystemArtifactStore(self.artifacts),
            parser=_ValueParser(parser_version="test-parser/1", value=12.0),
            unit_of_work_factory=partial(SQLiteUnitOfWork, self.database),
            clock=lambda: INGESTED_AT,
        )
        first = ingester.ingest(self.request)
        (self.artifacts / first.artifact.storage_ref).write_bytes(b"corrupted")

        with self.assertRaises(ArtifactIntegrityError):
            ingester.ingest(self.request)

    def test_revision_priority_must_be_a_non_negative_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            replace(self.request, revision_priority=-1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            replace(self.request, revision_priority=True)


if __name__ == "__main__":
    unittest.main()
