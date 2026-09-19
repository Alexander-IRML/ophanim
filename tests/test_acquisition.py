"""Model-independent native acquisition, durable retries, and indexed reads."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ophanim.artifacts import FilesystemArtifactStore
from ophanim.core.acquisition import acquire_recent, read_recent, _persist
from ophanim.gim import GIMNotFound
from ophanim.ingestion import IngestionRequest, LoadedSource
from ophanim.sensing import SensingError, read_desktop_sequence, read_ionex_sequence
from test_sensing import EPOCH, artifact, fixture


HAS_ARRAYS = all(find_spec(name) is not None for name in ("numpy", "xarray"))


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2024, 1, 5, 12, tzinfo=UTC)


class Source:
    def __init__(self, *, missing=False):
        self.requests = []
        self.missing = missing

    def fetch(self, request):
        self.requests.append(request)
        if self.missing:
            raise GIMNotFound("not published")
        day = request.product_date
        raw = fixture().replace(b"2024 1 2 0 0 0", f"{day.year} {day.month} {day.day} 0 0 0".encode().ljust(14))
        return SimpleNamespace(
            loaded_source=LoadedSource(raw, "fixture.inx"),
            ingestion_request=IngestionRequest("code", "gim", "https://www.aiub.unibe.ch/download/CODE/fixture", revision="rapid", revision_priority=10),
            resolved_uri="https://www.aiub.unibe.ch/download/CODE/fixture",
        )


@unittest.skipUnless(HAS_ARRAYS, "optional scientific dependencies unavailable")
class AcquisitionTests(unittest.TestCase):
    def test_downloaded_native_history_is_cached_without_models(self):
        with TemporaryDirectory() as directory, patch("ophanim.core.acquisition.datetime", FixedDatetime):
            provider = Source()
            progress = []
            acquired = acquire_recent(directory, days=3, source=provider, progress=progress.append)
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual(acquired.sizes["time"], 3)
            self.assertEqual(acquired.source_rms_tecu.values[0, 0, 0], 1)
            self.assertTrue((Path(directory) / "core" / "catalog.sqlite3").is_file())
            self.assertFalse((Path(directory) / "ophanim.sqlite3").exists())
            second = Source()
            again = acquire_recent(directory, days=3, source=second)
            self.assertEqual(second.requests, [])
            self.assertEqual(acquired.attrs["snapshot_id"], again.attrs["snapshot_id"])
            cached = read_recent(directory, days=3)
            self.assertEqual(cached.tec.values[0, 0, 0], 10)
            self.assertTrue(progress)

    def test_cancel_preserves_complete_download_and_resumes_missing_days(self):
        with TemporaryDirectory() as directory, patch("ophanim.core.acquisition.datetime", FixedDatetime):
            provider = Source()
            with self.assertRaisesRegex(InterruptedError, "remain cached"):
                acquire_recent(directory, days=3, source=provider, cancelled=lambda: len(provider.requests) >= 1)
            self.assertEqual(read_recent(directory).sizes["time"], 1)
            resumed = Source()
            acquire_recent(directory, days=3, source=resumed)
            self.assertEqual(len(resumed.requests), 2)

    def test_missing_source_does_not_poison_a_retry(self):
        with TemporaryDirectory() as directory, patch("ophanim.core.acquisition.datetime", FixedDatetime):
            with self.assertRaisesRegex(SensingError, "no native observations"):
                acquire_recent(directory, days=1, source=Source(missing=True))
            resumed = Source()
            self.assertEqual(acquire_recent(directory, days=1, source=resumed).sizes["time"], 1)
            self.assertEqual(len(resumed.requests), 1)

    def test_recent_success_recheck_is_rate_bounded_and_preserves_availability(self):
        with TemporaryDirectory() as directory, patch("ophanim.core.acquisition.datetime", FixedDatetime):
            before = acquire_recent(directory, days=3, source=Source())
            with sqlite3.connect(Path(directory) / "core" / "catalog.sqlite3") as connection:
                connection.execute("UPDATE product_checks SET last_checked=?", ("2024-01-04T11:00:00+00:00",))
            provider = Source()
            after = acquire_recent(directory, days=3, source=provider)
            self.assertEqual([r.product_date.day for r in provider.requests], [3, 4])
            self.assertEqual(before.available_at.values.tolist(), after.available_at.values.tolist())
            self.assertEqual(before.attrs["snapshot_id"], after.attrs["snapshot_id"])
            again = Source()
            acquire_recent(directory, days=3, source=again)
            self.assertEqual(again.requests, [])

    def test_cache_read_detects_modified_array_bytes(self):
        with TemporaryDirectory() as directory, patch("ophanim.core.acquisition.datetime", FixedDatetime):
            acquire_recent(directory, days=1, source=Source())
            cached = next((Path(directory) / "core" / "arrays").rglob("*.npz"))
            cached.write_bytes(b"corrupted fixture")
            with self.assertRaisesRegex(OSError, "checksum"):
                read_recent(directory)

    def test_revision_selection_prefers_priority_not_insertion_order(self):
        with TemporaryDirectory() as directory:
            raw = fixture()
            early = artifact(raw, identity="early")
            high_raw = raw.replace(b"  100", b"  300")
            high = replace(artifact(high_raw, identity="high", priority=20), ingested_at=EPOCH + timedelta(days=2))
            for content, source in ((high_raw, high), (raw, early)):
                dataset = read_ionex_sequence(content, artifact=source)
                _persist(directory, content, source, dataset, EPOCH.date())
            selected = read_recent(directory)
            self.assertEqual(selected.source_id.values.tolist(), ["high"])
            self.assertEqual(selected.tec.values[0, 0, 0], 30)
            self.assertEqual(len(selected.attrs["source_metadata"]["revision_history"]), 2)

    def test_identical_refresh_keeps_first_availability_without_array_orphans(self):
        with TemporaryDirectory() as directory:
            raw = fixture()
            original = artifact(raw, identity="unchanged")
            for source in (original, replace(original, ingested_at=original.ingested_at + timedelta(days=2))):
                dataset = read_ionex_sequence(raw, artifact=source)
                _persist(directory, raw, source, dataset, EPOCH.date())
            cached = read_recent(directory)
            self.assertEqual(str(cached.available_at.values[0]), "2024-01-03T00:00:00.000000")
            self.assertEqual(len(list((Path(directory) / "core" / "arrays").rglob("*.npz"))), 1)

    def test_empty_cache_and_invalid_bounds_do_not_start_downloads(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SensingError, "no cached"):
                read_recent(directory)
            provider = Source()
            for days in (0, 32, True, 2.5):
                with self.assertRaises(ValueError):
                    acquire_recent(directory, days=days, source=provider)
            with self.assertRaises(InterruptedError):
                acquire_recent(directory, cancelled=lambda: True, source=provider)
            self.assertEqual(provider.requests, [])
            self.assertFalse((Path(directory) / "core").exists())

    def test_desktop_time_index_excludes_irrelevant_unparseable_artifact(self):
        with TemporaryDirectory() as directory:
            store = FilesystemArtifactStore(Path(directory) / "artifacts")
            sources = []
            for identity, raw in (("inside", fixture()), ("outside", b"not ionex")):
                stored = store.put(raw, suffix=".inx")
                sources.append(replace(artifact(raw, identity=identity), storage_ref=stored.storage_ref))
            with sqlite3.connect(Path(directory) / "ophanim.sqlite3") as connection:
                connection.execute("CREATE TABLE source_artifacts (artifact_id TEXT, provider TEXT, product TEXT, parser_version TEXT, revision_priority INTEGER, checksum_sha256 TEXT, storage_ref TEXT, ingested_at TEXT, revision TEXT, source_uri TEXT)")
                connection.execute("CREATE TABLE tec_observations (artifact_id TEXT, observed_at TEXT)")
                connection.execute("CREATE INDEX time_idx ON tec_observations(artifact_id, observed_at)")
                for index, source in enumerate(sources):
                    connection.execute("INSERT INTO source_artifacts VALUES (?,?,?,?,?,?,?,?,?,?)", (
                        source.artifact_id, source.provider, source.product, source.parser_version,
                        source.revision_priority, source.checksum_sha256, source.storage_ref,
                        source.ingested_at.isoformat(), source.revision, source.source_uri,
                    ))
                    connection.execute("INSERT INTO tec_observations VALUES (?,?)", (
                        source.artifact_id, (EPOCH - timedelta(days=index * 5)).isoformat(timespec="microseconds"),
                    ))
            selected = read_desktop_sequence(directory, start=EPOCH, end=EPOCH)
            self.assertEqual(selected.source_id.values.tolist(), ["inside"])
            # Cached discovery can reuse the old source without touching its DB.
            self.assertEqual(read_recent(directory, days=3).sizes["time"], 1)


if __name__ == "__main__":
    unittest.main()
