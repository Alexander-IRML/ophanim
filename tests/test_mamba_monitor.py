"""Durable pipeline tests for the desktop Mamba monitor."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ophanim.domain import TECObservation
from ophanim.gim import GIMNotFound, GIMSourceUnavailable
from ophanim.mamba_monitor import (
    AcquiredReadoutDay,
    MambaMonitor,
    MambaMonitorUnavailable,
    MonitorRegion,
    RegionalReadout,
    _ReadoutDayUnavailable,
    _regional_readouts,
)


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _Provider:
    def __init__(self) -> None:
        self.requests: list[date] = []
        self.fail_on: set[date] = set()
        self.missing_on: set[date] = set()
        self.transient_failures: dict[date, int] = {}
        self.spike_on: set[date] = set()

    def acquire(self, product_date: date, region: MonitorRegion) -> AcquiredReadoutDay:
        self.requests.append(product_date)
        remaining = self.transient_failures.get(product_date, 0)
        if remaining:
            self.transient_failures[product_date] = remaining - 1
            raise GIMSourceUnavailable("synthetic transient archive outage")
        if product_date in self.missing_on:
            raise GIMNotFound("synthetic archive day is unavailable")
        if product_date in self.fail_on:
            raise GIMSourceUnavailable("synthetic archive outage")
        start = datetime.combine(product_date, datetime.min.time(), tzinfo=UTC)
        readouts = []
        for index in range(12):
            phase = (product_date.toordinal() * 12 + index) % 12
            baseline = 15.0 + phase * 0.15
            spike = 75.0 if product_date in self.spike_on and index == 6 else 0.0
            readouts.append(
                RegionalReadout(
                    observed_at=start + timedelta(hours=index * 2),
                    mean_vtec_tecu=baseline + 1.0 + spike,
                    median_vtec_tecu=baseline + spike * 0.9,
                    coverage_fraction=1.0,
                    cell_count=2,
                )
            )
        content = f"synthetic-final-{product_date.isoformat()}".encode("ascii")
        return AcquiredReadoutDay(
            product_date=product_date,
            source_uri=f"https://example.test/{product_date.isoformat()}.ionex",
            resolved_uri=f"https://example.test/{product_date.isoformat()}.ionex",
            filename=f"{product_date.isoformat()}.ionex",
            content=content,
            revision="final",
            revision_priority=20,
            observations=(),
            readouts=tuple(readouts),
        )


class _BulkArchive:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.artifacts = []

    def archive(self, *, artifact, observations, grid, include_core):
        self.artifacts.append(artifact)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("synthetic catalog outage")
        return SimpleNamespace(
            manifest=SimpleNamespace(
                grid_set_id=sha256(artifact.artifact_id.encode("utf-8")).hexdigest()
            )
        )


class MambaMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "monitor"
        self.clock = _Clock(datetime(2002, 1, 10, 12, tzinfo=UTC))
        self.provider = _Provider()
        self.monitor = MambaMonitor(
            data_directory=self.directory,
            clock=self.clock,
            readout_provider=self.provider,
            background=False,
            minimum_training_readouts=8,
            revision_overlap_days=3,
            acquisition_attempts=1,
        )
        self.addCleanup(self.monitor.close)

    def _initialize(self) -> dict:
        queued = self.monitor.initialize({"history_years": 1})
        self.assertEqual(queued["state"], "initializing")
        self.assertEqual(queued["active_job"]["status"], "queued")
        self.assertTrue(self.monitor.run_next_job())
        return self.monitor.status()

    def test_shared_compute_cancellation_is_interrupted_not_failed(self) -> None:
        from ophanim.core.jobs import WorkCancelled
        queued = self.monitor.initialize({"history_years": 1})
        job_id = queued["active_job"]["job_id"]
        with patch.object(self.monitor, "_run_initialize", side_effect=WorkCancelled("waiting cancelled")):
            self.assertTrue(self.monitor.run_next_job())
        with self.monitor._connection() as connection:
            job = connection.execute("SELECT status,retryable,error_message FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        self.assertEqual(job["status"], "interrupted")
        self.assertEqual(job["retryable"], 1)
        self.assertIsNone(job["error_message"])
        self.assertIsNone(self.monitor.status()["model"])

    def test_initialization_is_background_shaped_versioned_and_durable(self) -> None:
        self.assertEqual(self.monitor.status()["state"], "uninitialized")

        ready = self._initialize()

        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["model"]["algorithm"], "ophanim-mamba-selective-ssm/1")
        self.assertEqual(ready["model"]["training_readout_count"], 9 * 12)
        self.assertEqual(len(ready["model"]["artifact_checksum_sha256"]), 64)
        self.assertEqual(ready["scan"]["last_successful_cursor"], 9 * 12)
        self.assertFalse(ready["storage"]["derived_half_degree_history"])
        self.assertTrue((self.directory / "mamba-monitor.sqlite3").is_file())

        # Repeating the same initialization is an idempotent no-op.
        repeated = self.monitor.initialize({"history_years": 1})
        self.assertEqual(repeated["state"], "ready")
        self.assertIsNone(repeated["active_job"])

        self.assertTrue(self.monitor.close())
        reopened = MambaMonitor(
            data_directory=self.directory,
            clock=self.clock,
            readout_provider=self.provider,
            background=False,
            minimum_training_readouts=8,
        )
        self.addCleanup(reopened.close)
        self.assertEqual(
            reopened.status()["model"]["model_version_id"],
            ready["model"]["model_version_id"],
        )

    def test_scan_fetches_days_since_baseline_and_reports_candidate(self) -> None:
        ready = self._initialize()
        old_cursor = ready["scan"]["last_successful_cursor"]
        self.clock.value = datetime(2002, 1, 12, 12, tzinfo=UTC)
        self.provider.spike_on.add(date(2002, 1, 11))

        queued = self.monitor.scan()
        self.assertEqual(queued["state"], "scanning")
        self.assertTrue(self.monitor.run_next_job())
        finished = self.monitor.status()

        self.assertEqual(finished["state"], "ready")
        result = finished["scan"]["last_result"]
        self.assertEqual(result["readout_count"], 24)
        self.assertGreater(result["candidate_readout_count"], 0)
        self.assertEqual(
            result["candidate_event_count"], result["candidate_readout_count"]
        )
        self.assertEqual(result["requested_day_count"], 5)
        self.assertEqual(result["available_day_count"], 5)
        self.assertEqual(result["missing_day_count"], 0)
        self.assertTrue(result["candidates"])
        self.assertEqual(result["candidates"][0]["assessment"], "candidate")
        self.assertEqual(result["candidates"][0]["region_name"], "Central Texas")
        self.assertEqual(result["candidates"][0]["contributing_cell_count"], 2)
        self.assertNotIn("affected_cell_count", result["candidates"][0])
        self.assertGreater(finished["scan"]["last_successful_cursor"], old_cursor)

        # No new source bytes means a successful zero-readout scan, not a retry
        # of previously scored targets.
        self.monitor.scan()
        self.monitor.run_next_job()
        repeated = self.monitor.status()["scan"]["last_result"]
        self.assertEqual(repeated["readout_count"], 0)
        self.assertEqual(repeated["candidate_readout_count"], 0)
        self.assertEqual(repeated["requested_day_count"], 3)
        self.assertEqual(repeated["available_day_count"], 3)
        self.assertEqual(repeated["missing_day_count"], 0)

        # The exact-window comparison must remain available after a later
        # empty scan replaces status().scan.last_result.
        comparison = self.monitor.comparison_window(
            first_observed_at=datetime.fromisoformat(
                result["first_observed_at"].replace("Z", "+00:00")
            ),
            last_observed_at=datetime.fromisoformat(
                result["last_observed_at"].replace("Z", "+00:00")
            ),
            expected_region=MonitorRegion(),
            maximum_source_cursor=10**9,
            expected_pipeline_id=finished["pipeline_id"],
            expected_model_version_id=finished["model"]["model_version_id"],
        )
        self.assertEqual(comparison["readout_count"], result["readout_count"])
        self.assertEqual(
            comparison["candidate_readout_count"],
            result["candidate_readout_count"],
        )
        self.assertTrue(comparison["complete"])
        with self.assertRaisesRegex(MambaMonitorUnavailable, "identical region"):
            self.monitor.comparison_window(
                first_observed_at=datetime.fromisoformat(
                    result["first_observed_at"].replace("Z", "+00:00")
                ),
                last_observed_at=datetime.fromisoformat(
                    result["last_observed_at"].replace("Z", "+00:00")
                ),
                expected_region=MonitorRegion(
                    name="Elsewhere", south=10, west=10, north=20, east=20
                ),
                maximum_source_cursor=10**9,
                expected_pipeline_id=finished["pipeline_id"],
                expected_model_version_id=finished["model"]["model_version_id"],
            )

    def test_failed_scan_does_not_advance_success_cursor_and_can_retry(self) -> None:
        ready = self._initialize()
        original_cursor = ready["scan"]["last_successful_cursor"]
        self.clock.value = datetime(2002, 1, 11, 12, tzinfo=UTC)
        self.provider.fail_on.add(date(2002, 1, 10))

        self.monitor.scan()
        self.monitor.run_next_job()
        failed = self.monitor.status()

        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["scan"]["last_successful_cursor"], original_cursor)
        self.provider.fail_on.clear()
        self.monitor.scan()
        self.monitor.run_next_job()
        recovered = self.monitor.status()
        self.assertEqual(recovered["state"], "ready")
        self.assertGreater(recovered["scan"]["last_successful_cursor"], original_cursor)

    def test_scan_requires_an_initialized_model(self) -> None:
        with self.assertRaisesRegex(MambaMonitorUnavailable, "Initialize"):
            self.monitor.scan()

    def test_single_writer_lock_rejects_a_second_live_monitor(self) -> None:
        with self.assertRaisesRegex(MambaMonitorUnavailable, "already active"):
            MambaMonitor(
                data_directory=self.directory,
                clock=self.clock,
                readout_provider=self.provider,
                background=False,
                minimum_training_readouts=8,
            )

        self.assertTrue(self.monitor.close())
        reopened = MambaMonitor(
            data_directory=self.directory,
            clock=self.clock,
            readout_provider=self.provider,
            background=False,
            minimum_training_readouts=8,
        )
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.status()["state"], "uninitialized")

    def test_transient_source_failure_retries_the_day_automatically(self) -> None:
        provider = _Provider()
        provider.transient_failures[date(2002, 1, 1)] = 1
        monitor = MambaMonitor(
            data_directory=self.directory / "automatic-retry",
            clock=self.clock,
            readout_provider=provider,
            background=False,
            minimum_training_readouts=8,
            acquisition_attempts=3,
            retry_backoff_seconds=0,
        )
        self.addCleanup(monitor.close)

        monitor.initialize({"history_years": 1})
        self.assertTrue(monitor.run_next_job())
        self.assertEqual(monitor.status()["state"], "ready")
        self.assertEqual(provider.requests.count(date(2002, 1, 1)), 2)

    def test_initialization_rejects_a_long_missing_span(self) -> None:
        provider = _Provider()
        provider.missing_on.update({date(2002, 1, 4), date(2002, 1, 5)})
        monitor = MambaMonitor(
            data_directory=self.directory / "long-gap",
            clock=self.clock,
            readout_provider=provider,
            background=False,
            minimum_training_readouts=8,
            acquisition_attempts=1,
            maximum_baseline_gap_days=1,
        )
        self.addCleanup(monitor.close)

        monitor.initialize({"history_years": 1})
        self.assertTrue(monitor.run_next_job())
        status = monitor.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("missing span of 48 hours", status["error"])

    def test_version_one_monitor_database_migrates_in_place(self) -> None:
        legacy = self.directory / "legacy-schema"
        legacy.mkdir(parents=True)
        database = legacy / "mamba-monitor.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE monitor_schema (singleton INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO monitor_schema VALUES (1, 1)")
        connection.execute(
            "CREATE TABLE scan_runs (scan_run_id TEXT PRIMARY KEY)"
        )
        connection.commit()
        connection.close()

        monitor = MambaMonitor(
            data_directory=legacy,
            clock=self.clock,
            readout_provider=self.provider,
            background=False,
            minimum_training_readouts=8,
        )
        self.addCleanup(monitor.close)
        connection = sqlite3.connect(database)
        version = connection.execute(
            "SELECT schema_version FROM monitor_schema WHERE singleton = 1"
        ).fetchone()[0]
        scan_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(scan_runs)")
        }
        model_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(models)")
        }
        connection.close()
        self.assertEqual(version, 2)
        self.assertIn("job_id", scan_columns)
        self.assertTrue(
            {
                "regularized_sample_count",
                "missing_slot_count",
                "maximum_missing_gap_hours",
            }.issubset(model_columns)
        )

    def test_region_validation_rejects_inverted_or_wrapping_bounds(self) -> None:
        with self.assertRaises(ValueError):
            self.monitor.initialize(
                {
                    "history_years": 1,
                    "region": {
                        "name": "bad",
                        "south": 33,
                        "north": 30,
                        "west": -100,
                        "east": -96,
                    },
                }
            )

    def test_globally_complete_day_without_regional_cells_is_missing_not_fatal(self) -> None:
        product_date = date(2002, 1, 3)
        start = datetime.combine(product_date, datetime.min.time(), tzinfo=UTC)

        def observations():
            for epoch in range(6):
                for _ in range(4_856):
                    yield TECObservation(
                        artifact_id="synthetic-global-day",
                        observed_at=start + timedelta(hours=epoch * 2),
                        latitude_degrees=0.0,
                        longitude_degrees=0.0,
                        vtec_tecu=12.0,
                    )

        with self.assertRaisesRegex(
            _ReadoutDayUnavailable,
            "no usable regional readouts",
        ):
            _regional_readouts(
                observations(),
                product_date=product_date,
                region=MonitorRegion(),
                retain_observations=False,
            )

    def test_bulk_archive_retry_reuses_immutable_source_provenance(self) -> None:
        archive = _BulkArchive(failures=1)
        monitor = MambaMonitor(
            data_directory=self.directory / "bulk-retry",
            clock=self.clock,
            readout_provider=self.provider,
            bulk_archive=archive,
            background=False,
            minimum_training_readouts=8,
        )
        self.addCleanup(monitor.close)

        monitor.initialize({"history_years": 1})
        with redirect_stderr(StringIO()):
            self.assertTrue(monitor.run_next_job())
        self.assertEqual(monitor.status()["state"], "failed")
        # Retrying days later must retain the original historical window.
        self.clock.value = datetime(2002, 1, 12, 13, tzinfo=UTC)

        monitor.initialize({"history_years": 1})
        self.assertTrue(monitor.run_next_job())
        ready = monitor.status()
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["model"]["training_end"], "2002-01-09T22:00:00.000000Z")
        self.assertGreaterEqual(len(archive.artifacts), 2)
        self.assertEqual(
            archive.artifacts[0].ingested_at,
            archive.artifacts[1].ingested_at,
        )

    def test_second_region_does_not_republish_existing_native_grids(self) -> None:
        archive = _BulkArchive()
        monitor = MambaMonitor(
            data_directory=self.directory / "bulk-regions",
            clock=self.clock,
            readout_provider=self.provider,
            bulk_archive=archive,
            background=False,
            minimum_training_readouts=8,
        )
        self.addCleanup(monitor.close)

        monitor.initialize({"history_years": 1})
        self.assertTrue(monitor.run_next_job())
        first_archive_count = len(archive.artifacts)
        first_model_id = monitor.status()["model"]["model_version_id"]
        self.assertEqual(first_archive_count, 9)

        monitor.initialize(
            {
                "history_years": 1,
                "region": {
                    "name": "Central Texas alternate experiment",
                    "south": 29.0,
                    "west": -100.5,
                    "north": 32.5,
                    "east": -96.0,
                },
            }
        )
        self.assertTrue(monitor.run_next_job())
        second_status = monitor.status()
        self.assertEqual(second_status["state"], "ready")
        self.assertNotEqual(
            second_status["model"]["model_version_id"],
            first_model_id,
        )
        self.assertEqual(len(archive.artifacts), first_archive_count)

    def test_native_grid_snapshot_reuses_frozen_bulk_history(self) -> None:
        archive = _BulkArchive()
        monitor = MambaMonitor(
            data_directory=self.directory / "spatial-source",
            clock=self.clock,
            readout_provider=self.provider,
            bulk_archive=archive,
            background=False,
            minimum_training_readouts=8,
        )
        self.addCleanup(monitor.close)

        with self.assertRaisesRegex(MambaMonitorUnavailable, "Initialize"):
            monitor.native_grid_snapshot()
        monitor.initialize({"history_years": 1})
        self.assertTrue(monitor.run_next_job())

        snapshot = monitor.native_grid_snapshot()
        status = monitor.status()
        self.assertEqual(snapshot.pipeline_id, status["pipeline_id"])
        self.assertEqual(
            snapshot.model_version_id,
            status["model"]["model_version_id"],
        )
        self.assertEqual(len(snapshot.days), 9)
        self.assertEqual(snapshot.days[0].product_date, date(2002, 1, 1))
        self.assertEqual(snapshot.days[-1].product_date, date(2002, 1, 9))
        self.assertEqual(
            snapshot.source_cursor,
            max(item.source_day_id for item in snapshot.days),
        )
        self.assertTrue(
            all(len(item.grid_set_id) == 64 for item in snapshot.days)
        )


if __name__ == "__main__":
    unittest.main()
