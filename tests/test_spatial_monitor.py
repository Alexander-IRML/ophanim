"""Lifecycle tests for the native-grid ConvLSTM monitor.

The expensive tensor work is deliberately replaced with a recording backend.
These tests exercise the durable orchestration boundary: frozen source
snapshots, immutable model artifacts, scan cursors, retries, locking, and the
side-by-side Mamba comparison presented by the desktop application.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from ophanim.mamba_monitor import NativeGridDay, NativeGridSnapshot
from ophanim.spatial_monitor import (
    SPATIAL_ALGORITHM,
    SpatialMonitor,
    SpatialMonitorConflict,
    SpatialMonitorUnavailable,
    SpatialScanResult,
    SpatialTrainingResult,
    _GridFrame,
    _chronological_day_splits,
    _iter_native_frames,
    _iter_native_windows,
)


_NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _days(count: int, *, start: date = date(2025, 1, 1)) -> tuple[NativeGridDay, ...]:
    return tuple(
        NativeGridDay(
            source_day_id=index + 1,
            product_date=start + timedelta(days=index),
            grid_set_id=sha256(f"grid-{index}".encode()).hexdigest(),
            checksum_sha256=sha256(f"source-{index}".encode()).hexdigest(),
            revision_priority=20,
        )
        for index in range(count)
    )


def _snapshot(days: tuple[NativeGridDay, ...]) -> NativeGridSnapshot:
    return NativeGridSnapshot(
        pipeline_id="mamba-pipeline-test",
        model_version_id="mamba-model-test",
        sync_through_date=days[-1].product_date,
        source_cursor=max(day.source_day_id for day in days),
        days=days,
    )


class _FakeMambaMonitor:
    def __init__(self, snapshot: NativeGridSnapshot) -> None:
        self.snapshot = snapshot
        self.scan_calls = 0
        self.snapshot_calls = 0
        self.scan_state = "ready"
        self.error: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.region: dict[str, Any] | None = None

    def native_grid_snapshot(self) -> NativeGridSnapshot:
        self.snapshot_calls += 1
        return self.snapshot

    def scan(self) -> dict[str, Any]:
        self.scan_calls += 1
        return self.status()

    def status(self) -> dict[str, Any]:
        payload = {
            "state": self.scan_state,
            "error": self.error,
            "scan": {"last_result": self.last_result},
        }
        if self.region is not None:
            payload["region"] = self.region
        return payload

    def comparison_window(self, **_kwargs: Any) -> dict[str, Any]:
        result = self.last_result or {"readout_count": 0, "candidates": []}
        return {
            **result,
            "unscored_readout_count": 0,
            "complete": True,
        }


class _FakeArchive:
    pass


_DEFAULT_ARCHIVE = object()


class _NativeReadArchive:
    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results
        self.requests: list[str] = []

    def read_native(self, grid_set_id: str) -> Any:
        self.requests.append(grid_set_id)
        return self.results[grid_set_id]


class _FakeSpatialBackend:
    def __init__(self) -> None:
        self.available = True
        self.unavailable_reason: str | None = None
        self.train_calls: list[dict[str, Any]] = []
        self.scan_calls: list[dict[str, Any]] = []
        self.train_error: Exception | None = None
        self.scan_error: Exception | None = None
        self.training_window_count = 321
        self.scan_result = SpatialScanResult(
            first_observed_at=datetime(2025, 1, 21, 0, tzinfo=UTC),
            last_observed_at=datetime(2025, 1, 21, 4, tzinfo=UTC),
            map_count=3,
            usable_map_count=3,
            max_anomaly_score=7.25,
            candidates=(
                {
                    "observed_at": "2025-01-21T00:00:00Z",
                    "anomaly_score": 7.25,
                    "threshold": 3.5,
                    "q95_absolute_residual_tecu": 8.0,
                    "peak_absolute_residual_tecu": 14.0,
                    "peak_latitude": 30.0,
                    "peak_longitude": -98.0,
                    "affected_cell_count": 12,
                    "assessment": "candidate",
                },
                {
                    "observed_at": "2025-01-21T02:00:00Z",
                    "anomaly_score": 4.5,
                    "threshold": 3.5,
                    "q95_absolute_residual_tecu": 5.0,
                    "peak_absolute_residual_tecu": 9.0,
                    "peak_latitude": 31.0,
                    "peak_longitude": -97.5,
                    "affected_cell_count": 5,
                    "assessment": "candidate",
                },
            ),
        )

    def availability(self) -> tuple[bool, str | None]:
        return self.available, self.unavailable_reason

    def train(self, **kwargs: Any) -> SpatialTrainingResult:
        self.train_calls.append(kwargs)
        kwargs["progress"]("training", 7, 10, "Fitting causal windows")
        kwargs["checkpoint_path"].write_text("checkpoint", encoding="utf-8")
        if self.train_error is not None:
            raise self.train_error
        snapshot = kwargs["snapshot"]
        return SpatialTrainingResult(
            model_content=b'{"fake":"convlstm-model"}',
            training_data_hash=sha256(b"frozen training set").hexdigest(),
            training_start_at=datetime.combine(
                snapshot.days[0].product_date, datetime.min.time(), tzinfo=UTC
            ),
            training_end_at=datetime.combine(
                snapshot.days[-1].product_date, datetime.min.time(), tzinfo=UTC
            ),
            training_map_count=len(snapshot.days) * 12,
            training_window_count=self.training_window_count,
            validation_window_count=45,
            calibration_window_count=44,
            test_window_count=43,
            completed_epochs=int(kwargs["options"]["epochs"]),
            validation_mae_tecu=1.25,
            test_mae_tecu=1.5,
            residual_center_tecu=0.4,
            residual_scale_tecu=1.1,
            threshold=3.5,
        )

    def scan(self, **kwargs: Any) -> SpatialScanResult:
        self.scan_calls.append(kwargs)
        kwargs["progress"]("scoring", 3, 3, "Scored three native maps")
        if self.scan_error is not None:
            raise self.scan_error
        return self.scan_result


def _grid_frame(observed_at: datetime, *, source_day_id: int) -> _GridFrame:
    return _GridFrame(
        source_day_id=source_day_id,
        product_date=observed_at.date(),
        observed_at=observed_at,
        latitudes=(),
        longitudes=(),
        vtec_tecu=None,
        valid=None,
    )


def _regular_frames(days: tuple[NativeGridDay, ...]) -> tuple[_GridFrame, ...]:
    return tuple(
        _grid_frame(
            datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=2 * epoch),
            source_day_id=day.source_day_id,
        )
        for day in days
        for epoch in range(12)
    )


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy is required for native-grid reads")
class SpatialNativeFrameTests(unittest.TestCase):
    @staticmethod
    def _read_result(
        day: NativeGridDay,
        observed_at: tuple[datetime, ...],
        *,
        manifest_grid_set_id: str | None = None,
        source_checksum_sha256: str | None = None,
        revision_priority: int | None = None,
        manifest_shape: tuple[int, int, int] | None = None,
        latitudes: Any | None = None,
        longitudes: Any | None = None,
    ) -> Any:
        import numpy as np

        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        times = np.asarray(
            [int((value - epoch).total_seconds() * 1_000_000) for value in observed_at],
            dtype=np.int64,
        )
        values = np.empty((len(times), 71, 72), dtype=np.float32)
        for index in range(len(times)):
            values[index].fill(float(index))
        valid = np.ones(values.shape, dtype=np.bool_)
        canonical_latitudes = np.asarray(
            tuple(index * 2.5 - 87.5 for index in range(71)), dtype=np.float64
        )
        canonical_longitudes = np.asarray(
            tuple(index * 5.0 - 180.0 for index in range(72)), dtype=np.float64
        )
        return SimpleNamespace(
            manifest=SimpleNamespace(
                grid_set_id=(
                    day.grid_set_id
                    if manifest_grid_set_id is None
                    else manifest_grid_set_id
                ),
                source_checksum_sha256=(
                    day.checksum_sha256
                    if source_checksum_sha256 is None
                    else source_checksum_sha256
                ),
                revision_priority=(
                    day.revision_priority
                    if revision_priority is None
                    else revision_priority
                ),
                shape=(
                    values.shape if manifest_shape is None else manifest_shape
                ),
            ),
            times_utc_microseconds=times,
            latitudes_degrees=(
                canonical_latitudes if latitudes is None else latitudes
            ),
            longitudes_degrees=(
                canonical_longitudes if longitudes is None else longitudes
            ),
            vtec_tecu=values,
            valid=valid,
            quality_mask=np.zeros(values.shape, dtype=np.uint8),
        )

    def test_canonical_day_filters_repeated_next_day_midnight(self) -> None:
        day = _days(1)[0]
        start = datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
        observed = tuple(start + timedelta(hours=2 * index) for index in range(13))
        result = self._read_result(day, observed)
        archive = _NativeReadArchive({day.grid_set_id: result})

        frames = tuple(_iter_native_frames((day,), archive))

        self.assertEqual(len(frames), 12)
        self.assertEqual(frames[0].observed_at, start)
        self.assertEqual(frames[-1].observed_at, start + timedelta(hours=22))
        self.assertTrue(all(frame.product_date == day.product_date for frame in frames))
        self.assertEqual(float(frames[-1].vtec_tecu[0, 0]), 11.0)
        self.assertEqual(archive.requests, [day.grid_set_id])

    def test_noncanonical_axis_or_manifest_shape_is_rejected(self) -> None:
        import numpy as np

        day = _days(1)[0]
        start = datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
        observed = (start,)
        wrong_latitudes = np.asarray(
            tuple(index * 2.5 - 87.5 for index in range(71)), dtype=np.float64
        )
        wrong_latitudes[0] = -90.0
        cases = (
            (
                self._read_result(day, observed, latitudes=wrong_latitudes),
                "axes",
            ),
            (
                self._read_result(day, observed, manifest_shape=(1, 351, 720)),
                "manifest shape",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message):
                archive = _NativeReadArchive({day.grid_set_id: result})
                with self.assertRaisesRegex(SpatialMonitorUnavailable, message):
                    tuple(_iter_native_frames((day,), archive))

    def test_duplicate_epoch_and_duplicate_product_day_are_rejected(self) -> None:
        day = _days(1)[0]
        start = datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
        duplicate_epochs = (start, start + timedelta(hours=2), start + timedelta(hours=2))
        archive = _NativeReadArchive(
            {day.grid_set_id: self._read_result(day, duplicate_epochs)}
        )
        with self.assertRaisesRegex(SpatialMonitorUnavailable, "duplicate epochs"):
            tuple(_iter_native_frames((day,), archive))

        duplicate_day = NativeGridDay(
            source_day_id=day.source_day_id + 1,
            product_date=day.product_date,
            grid_set_id=sha256(b"duplicate-day-grid").hexdigest(),
            checksum_sha256=sha256(b"duplicate-day-source").hexdigest(),
            revision_priority=20,
        )
        second_result = self._read_result(duplicate_day, (start,))
        archive = _NativeReadArchive(
            {
                day.grid_set_id: self._read_result(day, (start,)),
                duplicate_day.grid_set_id: second_result,
            }
        )
        with self.assertRaisesRegex(SpatialMonitorUnavailable, "unique and chronological"):
            tuple(_iter_native_frames((day, duplicate_day), archive))

    def test_frozen_manifest_identity_checksum_and_revision_are_enforced(self) -> None:
        day = _days(1)[0]
        start = datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
        cases = (
            (
                self._read_result(
                    day,
                    (start,),
                    manifest_grid_set_id=sha256(b"wrong-grid").hexdigest(),
                ),
                "identity mismatch",
            ),
            (
                self._read_result(
                    day,
                    (start,),
                    source_checksum_sha256=sha256(b"wrong-source").hexdigest(),
                ),
                "checksum mismatch",
            ),
            (
                self._read_result(day, (start,), revision_priority=10),
                "revision provenance mismatch",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message):
                archive = _NativeReadArchive({day.grid_set_id: result})
                with self.assertRaisesRegex(SpatialMonitorUnavailable, message):
                    tuple(_iter_native_frames((day,), archive))


class SpatialNativeWindowTests(unittest.TestCase):
    def test_windows_require_exact_two_hour_cadence_and_twelve_history_frames(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        frames = tuple(
            _grid_frame(
                start + timedelta(hours=2 * index),
                source_day_id=1 + index // 12,
            )
            for index in range(36)
        )
        with patch(
            "ophanim.spatial_monitor._iter_native_frames",
            return_value=iter(frames),
        ):
            windows = tuple(_iter_native_windows((), _FakeArchive()))

        self.assertEqual(len(windows), 24)
        self.assertTrue(all(len(window.history) == 12 for window in windows))
        self.assertEqual(windows[0].history[0].observed_at, start)
        self.assertEqual(windows[0].target.observed_at, start + timedelta(hours=24))
        self.assertEqual(
            windows[-1].target.observed_at, start + timedelta(hours=70)
        )
        self.assertTrue(
            all(
                window.target.observed_at - window.history[-1].observed_at
                == timedelta(hours=2)
                for window in windows
            )
        )

        with patch(
            "ophanim.spatial_monitor._iter_native_frames",
            return_value=iter(frames),
        ):
            last_day = tuple(
                _iter_native_windows((), _FakeArchive(), target_day_ids={3})
            )
        self.assertEqual(len(last_day), 12)
        self.assertTrue(all(window.target.source_day_id == 3 for window in last_day))

    def test_gap_resets_history_instead_of_bridging_missing_epoch(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        before_gap = tuple(
            _grid_frame(start + timedelta(hours=2 * index), source_day_id=1)
            for index in range(12)
        )
        after_start = start + timedelta(hours=26)
        after_gap = tuple(
            _grid_frame(
                after_start + timedelta(hours=2 * index),
                source_day_id=2 + index // 11,
            )
            for index in range(13)
        )
        frames = before_gap + after_gap
        with patch(
            "ophanim.spatial_monitor._iter_native_frames",
            return_value=iter(frames),
        ):
            windows = tuple(_iter_native_windows((), _FakeArchive()))

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].history[0].observed_at, after_start)
        self.assertEqual(
            windows[0].target.observed_at, after_start + timedelta(hours=24)
        )
        self.assertNotIn(before_gap[-1], windows[0].history)

    def test_chronological_splits_never_create_cross_boundary_windows(self) -> None:
        days = _days(20)
        splits = _chronological_day_splits(days)
        self.assertEqual(tuple(map(len, splits)), (14, 2, 2, 2))

        def frame_source(
            selected_days: tuple[NativeGridDay, ...], _archive: Any
        ) -> Any:
            return iter(_regular_frames(tuple(selected_days)))

        split_windows = []
        with patch(
            "ophanim.spatial_monitor._iter_native_frames",
            side_effect=frame_source,
        ):
            for split in splits:
                windows = tuple(_iter_native_windows(split, _FakeArchive()))
                allowed = {day.source_day_id for day in split}
                self.assertTrue(
                    all(
                        {
                            frame.source_day_id
                            for frame in window.history + (window.target,)
                        }
                        <= allowed
                        for window in windows
                    )
                )
                split_windows.extend(windows)
        self.assertEqual(len(split_windows), 192)

        with patch(
            "ophanim.spatial_monitor._iter_native_frames",
            return_value=iter(_regular_frames(days)),
        ):
            unsplit = tuple(_iter_native_windows(days, _FakeArchive()))
        self.assertEqual(len(unsplit), 228)
        self.assertEqual(len(unsplit) - len(split_windows), 36)


class SpatialMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "spatial"
        self.clock = _Clock(datetime(2025, 1, 20, 12, tzinfo=UTC))
        self.mamba = _FakeMambaMonitor(_snapshot(_days(20)))
        self.backend = _FakeSpatialBackend()
        self.archive = _FakeArchive()
        self.monitor = self._new_monitor()
        self.addCleanup(self.monitor.close)

    def _new_monitor(
        self,
        *,
        directory: Path | None = None,
        archive: object | None = _DEFAULT_ARCHIVE,
        backend: _FakeSpatialBackend | None = None,
        minimum_training_windows: int = 1,
    ) -> SpatialMonitor:
        selected_archive = self.archive if archive is _DEFAULT_ARCHIVE else archive
        return SpatialMonitor(
            data_directory=directory or self.directory,
            mamba_monitor=self.mamba,  # type: ignore[arg-type]
            bulk_archive=selected_archive,  # type: ignore[arg-type]
            clock=self.clock,
            background=False,
            backend=backend or self.backend,
            minimum_training_windows=minimum_training_windows,
            mamba_poll_seconds=0,
        )

    def _initialize(self, **overrides: Any) -> dict[str, Any]:
        payload = {
            "history_years": 1,
            "architecture": "predictive-convlstm",
            "input_frame_count": 12,
            "forecast_horizon_hours": 2,
            "epochs": 3,
        }
        payload.update(overrides)
        queued = self.monitor.initialize(payload)
        self.assertEqual(queued["state"], "initializing")
        self.assertEqual(queued["active_job"]["status"], "queued")
        self.assertTrue(self.monitor.run_next_job())
        return self.monitor.status()

    def test_initialization_freezes_snapshot_and_publishes_durable_model(self) -> None:
        initial = self.monitor.status()
        self.assertTrue(initial["available"])
        self.assertEqual(initial["state"], "uninitialized")
        self.assertIsNone(initial["model"])

        queued = self.monitor.initialize(
            {
                "history_years": 1,
                "architecture": "predictive-convlstm",
                "input_frame_count": 12,
                "forecast_horizon_hours": 2,
                "epochs": 3,
            }
        )
        self.assertEqual(queued["state"], "initializing")
        self.assertEqual(queued["active_job"]["kind"], "initialize")
        self.assertEqual(queued["active_job"]["total_units"], 20)

        # Repeating an identical request while queued must not enqueue another
        # model build or mutate its frozen input snapshot.
        repeated = self.monitor.initialize(
            {
                "history_years": 1,
                "architecture": "predictive-convlstm",
                "input_frame_count": 12,
                "forecast_horizon_hours": 2,
                "epochs": 3,
            }
        )
        self.assertEqual(
            repeated["active_job"]["job_id"], queued["active_job"]["job_id"]
        )
        self.mamba.snapshot = _snapshot(_days(21))

        self.assertTrue(self.monitor.run_next_job())
        ready = self.monitor.status()
        self.assertEqual(ready["state"], "ready")
        self.assertIsNone(ready["active_job"])
        self.assertEqual(len(self.backend.train_calls), 1)
        trained_snapshot = self.backend.train_calls[0]["snapshot"]
        self.assertEqual(trained_snapshot.source_cursor, 20)
        self.assertEqual(len(trained_snapshot.days), 20)
        self.assertEqual(self.backend.train_calls[0]["options"], {"epochs": 3})

        model = ready["model"]
        self.assertEqual(model["algorithm"], SPATIAL_ALGORITHM)
        self.assertEqual(model["input_frame_count"], 12)
        self.assertEqual(model["forecast_horizon_hours"], 2)
        self.assertEqual(model["native_grid_shape"], [71, 72])
        self.assertEqual(model["training_window_count"], 321)
        self.assertEqual(model["validation_window_count"], 45)
        self.assertEqual(model["calibration_window_count"], 44)
        self.assertEqual(model["test_window_count"], 43)
        self.assertEqual(model["completed_epochs"], 3)
        self.assertEqual(model["threshold"], 3.5)
        self.assertEqual(len(model["artifact_checksum_sha256"]), 64)
        self.assertEqual(ready["scan"]["last_successful_cursor"], 20)
        self.assertEqual(ready["scan"]["latest_available_cursor"], 20)
        self.assertFalse(any(self.directory.glob("spatial-checkpoints/*.json")))

        # Closing and reopening preserves the selected pipeline and immutable
        # model metadata without invoking the trainer again.
        model_id = model["model_version_id"]
        self.assertTrue(self.monitor.close())
        reopened = self._new_monitor()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.status()["model"]["model_version_id"], model_id)

    def test_v1_migration_retires_unpinned_model_and_allows_retraining(self) -> None:
        ready = self._initialize()
        model_id = ready["model"]["model_version_id"]
        self.assertTrue(self.monitor.close())
        with closing(
            sqlite3.connect(self.directory / "spatial-monitor.sqlite3")
        ) as connection:
            connection.execute(
                "UPDATE jobs SET status = 'interrupted' WHERE pipeline_id = ?",
                (ready["pipeline_id"],),
            )
            connection.execute("ALTER TABLE models DROP COLUMN source_pipeline_id")
            connection.execute("ALTER TABLE models DROP COLUMN source_model_version_id")
            connection.execute(
                "UPDATE spatial_schema SET schema_version = 1 WHERE singleton = 1"
            )
            connection.commit()

        reopened = self._new_monitor()
        self.addCleanup(reopened.close)
        migrated = reopened.status()
        self.assertEqual(migrated["state"], "uninitialized")
        self.assertIsNone(migrated["model"])
        self.assertIn("retrain", migrated["error"])
        with closing(
            sqlite3.connect(self.directory / "spatial-monitor.sqlite3")
        ) as connection:
            preserved = connection.execute(
                "SELECT source_pipeline_id, source_model_version_id FROM models "
                "WHERE model_version_id = ?",
                (model_id,),
            ).fetchone()
            self.assertEqual(preserved, ("", ""))
            self.assertEqual(
                connection.execute(
                    "SELECT status FROM jobs WHERE pipeline_id = ?",
                    (ready["pipeline_id"],),
                ).fetchone()[0],
                "failed",
            )

        queued = reopened.initialize(
            {
                "history_years": 1,
                "architecture": "predictive-convlstm",
                "input_frame_count": 12,
                "forecast_horizon_hours": 2,
                "epochs": 3,
            }
        )
        self.assertEqual(queued["state"], "initializing")
        self.assertTrue(reopened.run_next_job())
        retrained = reopened.status()
        self.assertEqual(retrained["state"], "ready")
        self.assertEqual(retrained["model"]["model_version_id"], model_id)
        with closing(
            sqlite3.connect(self.directory / "spatial-monitor.sqlite3")
        ) as connection:
            provenance = connection.execute(
                "SELECT source_pipeline_id, source_model_version_id FROM models "
                "WHERE model_version_id = ?",
                (model_id,),
            ).fetchone()
            self.assertEqual(
                provenance,
                (
                    self.mamba.snapshot.pipeline_id,
                    self.mamba.snapshot.model_version_id,
                ),
            )

    def test_scan_advances_cursor_only_after_success_and_compares_mamba(self) -> None:
        ready = self._initialize()
        old_cursor = ready["scan"]["last_successful_cursor"]
        self.assertEqual(old_cursor, 20)
        self.mamba.snapshot = _snapshot(_days(22))
        self.mamba.last_result = {
            "readout_count": 3,
            "candidates": [
                {"observed_at": "2025-01-21T00:00:00Z"},
                {"observed_at": "2025-01-21T04:00:00Z"},
            ],
        }

        queued = self.monitor.scan()
        self.assertEqual(queued["state"], "scanning")
        self.assertEqual(queued["active_job"]["kind"], "scan")
        self.assertEqual(queued["scan"]["last_successful_cursor"], old_cursor)
        self.assertTrue(self.monitor.run_next_job())

        finished = self.monitor.status()
        self.assertEqual(finished["state"], "ready")
        self.assertEqual(self.mamba.scan_calls, 1)
        self.assertEqual(len(self.backend.scan_calls), 1)
        call = self.backend.scan_calls[0]
        self.assertEqual(call["after_source_cursor"], old_cursor)
        self.assertEqual(call["snapshot"].source_cursor, 22)
        self.assertEqual(call["model_content"], b'{"fake":"convlstm-model"}')
        self.assertEqual(finished["scan"]["last_successful_cursor"], 22)
        self.assertEqual(finished["scan"]["latest_available_cursor"], 22)

        result = finished["scan"]["last_result"]
        self.assertEqual(result["map_count"], 3)
        self.assertEqual(result["usable_map_count"], 3)
        self.assertEqual(result["candidate_map_count"], 2)
        self.assertEqual(result["cursor_start"], 20)
        self.assertEqual(result["cursor_end"], 22)
        self.assertEqual(len(result["candidates"]), 2)
        comparison = result["comparison"]
        self.assertEqual(comparison["both_candidate_count"], 1)
        self.assertEqual(comparison["spatial_only_count"], 1)
        self.assertEqual(comparison["mamba_only_count"], 1)
        self.assertEqual(comparison["compared_readout_count"], 3)
        self.assertAlmostEqual(comparison["agreement_fraction"], 1 / 3)

    def test_no_new_maps_is_a_successful_empty_scan(self) -> None:
        ready = self._initialize()
        cursor = ready["scan"]["last_successful_cursor"]
        self.backend.scan_result = SpatialScanResult(
            first_observed_at=None,
            last_observed_at=None,
            map_count=0,
            usable_map_count=0,
            max_anomaly_score=None,
            candidates=(),
        )

        self.monitor.scan()
        self.assertTrue(self.monitor.run_next_job())
        finished = self.monitor.status()

        self.assertEqual(finished["state"], "ready")
        self.assertEqual(finished["scan"]["last_successful_cursor"], cursor)
        result = finished["scan"]["last_result"]
        self.assertEqual(result["map_count"], 0)
        self.assertEqual(result["usable_map_count"], 0)
        self.assertEqual(result["candidate_map_count"], 0)
        self.assertIsNone(result["max_anomaly_score"])
        self.assertEqual(result["candidates"], [])

    def test_failed_scan_keeps_cursor_and_retry_scores_same_snapshot(self) -> None:
        ready = self._initialize()
        original_cursor = ready["scan"]["last_successful_cursor"]
        self.mamba.snapshot = _snapshot(_days(22))
        self.backend.scan_error = RuntimeError("synthetic tensor failure")

        self.monitor.scan()
        with redirect_stderr(StringIO()):
            self.assertTrue(self.monitor.run_next_job())
        failed = self.monitor.status()
        self.assertEqual(failed["state"], "failed")
        self.assertIn("synthetic tensor failure", failed["error"])
        self.assertEqual(failed["scan"]["last_successful_cursor"], original_cursor)
        self.assertIsNone(failed["scan"]["last_result"])
        with closing(
            sqlite3.connect(self.directory / "spatial-monitor.sqlite3")
        ) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT status FROM scan_runs ORDER BY rowid DESC LIMIT 1"
                ).fetchone()[0],
                "failed",
            )

        self.backend.scan_error = None
        self.monitor.scan()
        self.assertTrue(self.monitor.run_next_job())
        recovered = self.monitor.status()
        self.assertEqual(recovered["state"], "ready")
        self.assertEqual(recovered["scan"]["last_successful_cursor"], 22)
        self.assertEqual(self.backend.scan_calls[-1]["after_source_cursor"], original_cursor)

    def test_source_refresh_failure_keeps_cursor_and_never_calls_spatial_backend(self) -> None:
        ready = self._initialize()
        original_cursor = ready["scan"]["last_successful_cursor"]
        self.mamba.scan_state = "failed"
        self.mamba.error = "synthetic CODE archive failure"

        first = self.monitor.scan()
        repeated = self.monitor.scan()
        self.assertEqual(
            first["active_job"]["job_id"], repeated["active_job"]["job_id"]
        )
        self.assertTrue(self.monitor.run_next_job())
        failed = self.monitor.status()

        self.assertEqual(failed["state"], "failed")
        self.assertIn("synthetic CODE archive failure", failed["error"])
        self.assertEqual(failed["scan"]["last_successful_cursor"], original_cursor)
        self.assertEqual(self.backend.scan_calls, [])

    def test_failed_training_can_be_retried_without_publishing_a_model(self) -> None:
        self.backend.train_error = RuntimeError("synthetic optimizer failure")
        self.monitor.initialize({"history_years": 1})
        with redirect_stderr(StringIO()):
            self.assertTrue(self.monitor.run_next_job())
        failed = self.monitor.status()
        self.assertEqual(failed["state"], "failed")
        self.assertIsNone(failed["model"])
        self.assertIn("synthetic optimizer failure", failed["error"])

        self.backend.train_error = None
        retried = self.monitor.initialize({"history_years": 1})
        self.assertEqual(retried["state"], "initializing")
        self.assertTrue(self.monitor.run_next_job())
        self.assertEqual(self.monitor.status()["state"], "ready")

    def test_too_few_training_windows_never_publishes_model(self) -> None:
        self.assertTrue(self.monitor.close())
        backend = _FakeSpatialBackend()
        backend.training_window_count = 4
        monitor = self._new_monitor(
            directory=Path(self.temporary.name) / "minimum-window-check",
            backend=backend,
            minimum_training_windows=5,
        )
        self.addCleanup(monitor.close)
        monitor.initialize({"history_years": 1})
        self.assertTrue(monitor.run_next_job())
        status = monitor.status()
        self.assertEqual(status["state"], "failed")
        self.assertIsNone(status["model"])
        self.assertIn("at least 5 are required", status["error"])

    def test_active_job_conflicts_and_identical_request_is_idempotent(self) -> None:
        first = self.monitor.initialize({"history_years": 1, "epochs": 2})
        repeated = self.monitor.initialize({"history_years": 1, "epochs": 2})
        self.assertEqual(
            first["active_job"]["job_id"], repeated["active_job"]["job_id"]
        )
        with self.assertRaisesRegex(SpatialMonitorConflict, "already running"):
            self.monitor.initialize({"history_years": 1, "epochs": 3})

    def test_validation_rejects_non_causal_or_unsupported_configuration(self) -> None:
        invalid_payloads = (
            ({"history_years": 0}, "history_years"),
            ({"history_years": True}, "history_years"),
            ({"history_years": 31}, "history_years"),
            ({"architecture": "autoencoder"}, "architecture"),
            ({"input_frame_count": 11}, "input_frame_count"),
            ({"forecast_horizon_hours": 1}, "forecast_horizon_hours"),
            ({"epochs": 0}, "epochs"),
            ({"epochs": True}, "epochs"),
            ({"region": "Texas"}, "region"),
            ({"region": {"south": 33, "north": 29}}, "latitude"),
        )
        for payload, message in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, message):
                    self.monitor.initialize(payload)

    def test_spatial_region_must_match_active_mamba_region(self) -> None:
        self.mamba.region = {
            "name": "Central Texas",
            "south": 29.0,
            "west": -100.5,
            "north": 32.5,
            "east": -96.0,
        }
        with self.assertRaisesRegex(SpatialMonitorUnavailable, "same region"):
            self.monitor.initialize(
                {
                    "history_years": 1,
                    "region": {
                        "name": "Different",
                        "south": 30.0,
                        "west": -100.0,
                        "north": 31.0,
                        "east": -97.0,
                    },
                }
            )
        self.assertEqual(self.monitor.status()["state"], "uninitialized")

    def test_unavailable_archive_or_backend_is_reported_without_partial_job(self) -> None:
        self.assertTrue(self.monitor.close())
        unavailable = self._new_monitor(
            directory=Path(self.temporary.name) / "no-archive",
            archive=None,
        )
        self.addCleanup(unavailable.close)
        status = unavailable.status()
        self.assertFalse(status["available"])
        self.assertEqual(status["state"], "unavailable")
        self.assertIn("Zarr", status["reason"])
        with self.assertRaises(SpatialMonitorUnavailable):
            unavailable.initialize({"history_years": 1})

        backend = _FakeSpatialBackend()
        backend.available = False
        backend.unavailable_reason = "CPU PyTorch is unavailable"
        backend_unavailable = self._new_monitor(
            directory=Path(self.temporary.name) / "no-torch",
            backend=backend,
        )
        self.addCleanup(backend_unavailable.close)
        self.assertEqual(backend_unavailable.status()["reason"], backend.unavailable_reason)
        with self.assertRaisesRegex(SpatialMonitorUnavailable, "PyTorch"):
            backend_unavailable.initialize({"history_years": 1})

    def test_scan_requires_model_and_single_writer_lock_is_released_on_close(self) -> None:
        with self.assertRaisesRegex(SpatialMonitorUnavailable, "Train"):
            self.monitor.scan()
        with self.assertRaisesRegex(SpatialMonitorUnavailable, "already active"):
            self._new_monitor()

        self.assertTrue(self.monitor.close())
        reopened = self._new_monitor()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.status()["state"], "uninitialized")


if __name__ == "__main__":
    unittest.main()
