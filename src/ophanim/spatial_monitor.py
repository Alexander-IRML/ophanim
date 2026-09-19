"""Durable causal ConvLSTM training and native-grid anomaly scans.

The spatial monitor reuses the immutable native Zarr groups published by the
Mamba acquisition pipeline.  It never downloads a second historical copy and
never treats the interpolated 0.5-degree display layer as new evidence.

PyTorch and NumPy remain optional application dependencies.  This module is
safe to import without them; status then reports the spatial experiment as
unavailable while the rest of OPHANIM continues to work.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import math
import os
import queue
import secrets
import sqlite3
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from ophanim.artifacts import FilesystemArtifactStore
from ophanim.mamba_monitor import (
    MambaMonitor,
    MambaMonitorConflict,
    MambaMonitorUnavailable,
    MonitorRegion,
    NativeGridDay,
    NativeGridSnapshot,
)
from ophanim.tec_archive import BulkTECArchive


SPATIAL_MONITOR_SCHEMA_VERSION = 2
SPATIAL_ALGORITHM = "ophanim-convlstm-prior-day-residual/1"
DEFAULT_HISTORY_YEARS = 20
DEFAULT_EPOCHS = 2
DEFAULT_THRESHOLD_QUANTILE = 0.995
DEFAULT_MINIMUM_TRAINING_WINDOWS = 128
MAX_CANDIDATES_IN_STATUS = 50
_CADENCE = timedelta(hours=2)


class SpatialMonitorError(RuntimeError):
    """Base class for expected spatial-monitor lifecycle errors."""


class SpatialMonitorConflict(SpatialMonitorError):
    """The requested operation conflicts with an active spatial job."""


class SpatialMonitorUnavailable(SpatialMonitorError):
    """The spatial pipeline cannot run with the current prerequisites."""


class _SpatialInterrupted(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SpatialTrainingResult:
    model_content: bytes
    training_data_hash: str
    training_start_at: datetime
    training_end_at: datetime
    training_map_count: int
    training_window_count: int
    validation_window_count: int
    calibration_window_count: int
    test_window_count: int
    completed_epochs: int
    validation_mae_tecu: float
    test_mae_tecu: float
    residual_center_tecu: float
    residual_scale_tecu: float
    threshold: float


@dataclass(frozen=True, slots=True)
class SpatialScanResult:
    first_observed_at: datetime | None
    last_observed_at: datetime | None
    map_count: int
    usable_map_count: int
    max_anomaly_score: float | None
    candidates: tuple[dict[str, Any], ...]
    candidate_map_count: int | None = None


class SpatialBackend(Protocol):
    def availability(self) -> tuple[bool, str | None]: ...

    def train(
        self,
        *,
        snapshot: NativeGridSnapshot,
        archive: BulkTECArchive,
        region: MonitorRegion,
        options: Mapping[str, Any],
        checkpoint_path: Path,
        progress: Callable[[str, int, int, str], None],
        interrupted: Callable[[], bool],
    ) -> SpatialTrainingResult: ...

    def scan(
        self,
        *,
        model_content: bytes,
        model_metadata: Mapping[str, Any],
        snapshot: NativeGridSnapshot,
        archive: BulkTECArchive,
        after_source_cursor: int,
        region: MonitorRegion,
        progress: Callable[[str, int, int, str], None],
        interrupted: Callable[[], bool],
    ) -> SpatialScanResult: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS spatial_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id TEXT PRIMARY KEY,
    configuration_hash TEXT NOT NULL UNIQUE,
    history_years INTEGER NOT NULL,
    region_name TEXT NOT NULL,
    south REAL NOT NULL,
    west REAL NOT NULL,
    north REAL NOT NULL,
    east REAL NOT NULL,
    algorithm TEXT NOT NULL,
    requested_epochs INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('uninitialized', 'initializing', 'ready', 'scanning', 'failed')
    ),
    active_model_version_id TEXT,
    source_cursor INTEGER NOT NULL DEFAULT 0,
    scan_cursor INTEGER NOT NULL DEFAULT 0,
    last_successful_scan_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    kind TEXT NOT NULL CHECK (kind IN ('initialize', 'scan')),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'interrupted')
    ),
    phase TEXT NOT NULL,
    snapshot_json TEXT,
    completed_units INTEGER NOT NULL DEFAULT 0,
    total_units INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    error_message TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS spatial_jobs_one_active_idx
    ON jobs((1)) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS spatial_jobs_pipeline_created_idx
    ON jobs(pipeline_id, created_at DESC);

CREATE TABLE IF NOT EXISTS models (
    model_version_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    algorithm TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    artifact_checksum_sha256 TEXT NOT NULL,
    training_data_hash TEXT NOT NULL,
    training_start_at TEXT NOT NULL,
    training_end_at TEXT NOT NULL,
    training_map_count INTEGER NOT NULL,
    training_window_count INTEGER NOT NULL,
    validation_window_count INTEGER NOT NULL,
    calibration_window_count INTEGER NOT NULL,
    test_window_count INTEGER NOT NULL,
    completed_epochs INTEGER NOT NULL,
    validation_mae_tecu REAL NOT NULL,
    test_mae_tecu REAL NOT NULL,
    residual_center_tecu REAL NOT NULL,
    residual_scale_tecu REAL NOT NULL,
    threshold REAL NOT NULL,
    source_cursor INTEGER NOT NULL,
    source_pipeline_id TEXT NOT NULL,
    source_model_version_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    scan_run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    model_version_id TEXT NOT NULL REFERENCES models(model_version_id),
    from_cursor_exclusive INTEGER NOT NULL,
    to_cursor_inclusive INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    result_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    error_message TEXT
);
"""


class SpatialMonitor:
    """Own spatial baseline jobs, model artifacts, and successful scan cursor."""

    def __init__(
        self,
        *,
        data_directory: str | Path,
        mamba_monitor: MambaMonitor,
        bulk_archive: BulkTECArchive | None,
        clock: Callable[[], datetime] | None = None,
        background: bool = True,
        backend: SpatialBackend | None = None,
        minimum_training_windows: int = DEFAULT_MINIMUM_TRAINING_WINDOWS,
        mamba_poll_seconds: float = 1.0,
    ) -> None:
        if minimum_training_windows < 1:
            raise ValueError("minimum_training_windows must be positive")
        if not math.isfinite(mamba_poll_seconds) or mamba_poll_seconds < 0:
            raise ValueError("mamba_poll_seconds must be finite and non-negative")
        self._directory = Path(data_directory).expanduser().resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._database = self._directory / "spatial-monitor.sqlite3"
        self._model_store = FilesystemArtifactStore(
            self._directory / "spatial-model-artifacts"
        )
        self._checkpoint_directory = self._directory / "spatial-checkpoints"
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._mamba = mamba_monitor
        self._archive = bulk_archive
        self._clock = clock or (lambda: datetime.now(UTC))
        self._backend = backend or TorchSpatialBackend()
        self._minimum_training_windows = minimum_training_windows
        self._mamba_poll_seconds = float(mamba_poll_seconds)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._process_lock_file: Any | None = None
        self._process_lock_guard = threading.Lock()
        self._acquire_process_lock()
        try:
            self._initialize_store()
            resumable = self._recover_jobs()
        except BaseException:
            self._release_process_lock()
            raise
        if background:
            self._thread = threading.Thread(
                target=self._worker,
                name="ophanim-spatial-monitor",
                daemon=True,
            )
            self._thread.start()
            for job_id in resumable:
                self._queue.put(job_id)

    def status(self) -> dict[str, Any]:
        available, reason = self._availability()
        with self._connection() as connection:
            pipeline = self._active_pipeline(connection)
            if pipeline is None:
                return {
                    "ok": True,
                    "available": available,
                    "reason": reason,
                    "state": "uninitialized" if available else "unavailable",
                    "model": None,
                    "active_job": None,
                    "scan": _empty_scan_payload(),
                    "storage": _storage_payload(self._archive is not None),
                }
            active_job = connection.execute(
                """SELECT * FROM jobs WHERE status IN ('queued', 'running')
                   ORDER BY created_at DESC, rowid DESC LIMIT 1"""
            ).fetchone()
            model = None
            if pipeline["active_model_version_id"]:
                row = connection.execute(
                    "SELECT * FROM models WHERE model_version_id = ?",
                    (pipeline["active_model_version_id"],),
                ).fetchone()
                if row is not None:
                    model = _model_payload(row, pipeline)
            run = connection.execute(
                """SELECT * FROM scan_runs
                   WHERE pipeline_id = ? AND status = 'succeeded'
                   ORDER BY completed_at DESC, rowid DESC LIMIT 1""",
                (pipeline["pipeline_id"],),
            ).fetchone()
            state = pipeline["status"]
            if not available and active_job is None:
                state = "unavailable"
            elif active_job is not None:
                state = (
                    "initializing"
                    if active_job["kind"] == "initialize"
                    else "scanning"
                )
            return {
                "ok": True,
                "available": available,
                "reason": reason,
                "state": state,
                "pipeline_id": pipeline["pipeline_id"],
                "history_years": pipeline["history_years"],
                "region": _region_from_row(pipeline).to_dict(),
                "model": model,
                "active_job": None if active_job is None else _job_payload(active_job),
                "scan": {
                    "last_successful_at": pipeline["last_successful_scan_at"],
                    "last_successful_cursor": pipeline["scan_cursor"],
                    "latest_available_cursor": pipeline["source_cursor"],
                    "pending_frame_count": None,
                    "last_result": (
                        None
                        if run is None or run["result_json"] is None
                        else json.loads(run["result_json"])
                    ),
                },
                "error": pipeline["error_message"],
                "storage": _storage_payload(self._archive is not None),
            }

    def initialize(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_available()
        values = payload or {}
        if not isinstance(values, dict):
            raise ValueError("spatial initialization payload must be an object")
        years = values.get("history_years", DEFAULT_HISTORY_YEARS)
        if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 30:
            raise ValueError("history_years must be an integer between 1 and 30")
        architecture = values.get("architecture", "predictive-convlstm")
        if architecture != "predictive-convlstm":
            raise ValueError("architecture must be predictive-convlstm")
        frames = values.get("input_frame_count", 12)
        horizon = values.get("forecast_horizon_hours", 2)
        if frames != 12:
            raise ValueError("input_frame_count must be 12")
        if horizon != 2:
            raise ValueError("forecast_horizon_hours must be 2")
        epochs = values.get("epochs", DEFAULT_EPOCHS)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10:
            raise ValueError("epochs must be an integer between 1 and 10")
        region = MonitorRegion.from_payload(values.get("region"))
        mamba_status = self._mamba.status()
        active_region = mamba_status.get("region")
        if isinstance(active_region, dict):
            try:
                mamba_region = MonitorRegion.from_payload(active_region)
            except ValueError as error:
                raise SpatialMonitorUnavailable(
                    "The active Mamba region metadata is invalid"
                ) from error
            if (
                mamba_region.south,
                mamba_region.west,
                mamba_region.north,
                mamba_region.east,
            ) != (region.south, region.west, region.north, region.east):
                raise SpatialMonitorUnavailable(
                    "Train the spatial baseline with the same region bounds as Mamba"
                )
        try:
            snapshot = self._mamba.native_grid_snapshot()
        except MambaMonitorConflict as error:
            raise SpatialMonitorConflict(str(error)) from error
        except MambaMonitorUnavailable as error:
            raise SpatialMonitorUnavailable(str(error)) from error
        days = _history_suffix(snapshot.days, years)
        if not days:
            raise SpatialMonitorUnavailable("No native history falls in the requested window")
        snapshot = NativeGridSnapshot(
            pipeline_id=snapshot.pipeline_id,
            model_version_id=snapshot.model_version_id,
            sync_through_date=days[-1].product_date,
            source_cursor=max(day.source_day_id for day in days),
            days=days,
        )
        configuration = {
            "algorithm": SPATIAL_ALGORITHM,
            "architecture": architecture,
            "epochs": epochs,
            "forecast_horizon_hours": horizon,
            "history_years": years,
            "input_frame_count": frames,
            "region": region.to_dict(),
            "source_pipeline_id": snapshot.pipeline_id,
            "source_model_version_id": snapshot.model_version_id,
        }
        configuration_hash = _json_hash(configuration)
        pipeline_id = f"spatial-{configuration_hash[:24]}"
        now = _aware_utc(self._clock(), "spatial monitor clock")
        now_text = _datetime_text(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
            if active is not None:
                if active["kind"] == "initialize" and active["pipeline_id"] == pipeline_id:
                    connection.commit()
                    return self.status()
                connection.rollback()
                raise SpatialMonitorConflict("Another spatial operation is already running")
            existing = connection.execute(
                "SELECT * FROM pipelines WHERE pipeline_id = ?", (pipeline_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO pipelines (
                        pipeline_id, configuration_hash, history_years,
                        region_name, south, west, north, east, algorithm,
                        requested_epochs, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'uninitialized', ?, ?)""",
                    (
                        pipeline_id,
                        configuration_hash,
                        years,
                        region.name,
                        region.south,
                        region.west,
                        region.north,
                        region.east,
                        SPATIAL_ALGORITHM,
                        epochs,
                        now_text,
                        now_text,
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('active_pipeline_id', ?)",
                (pipeline_id,),
            )
            existing = connection.execute(
                "SELECT * FROM pipelines WHERE pipeline_id = ?", (pipeline_id,)
            ).fetchone()
            if existing["active_model_version_id"] and existing["status"] == "ready":
                connection.commit()
                return self.status()
            job_id = _job_id("initialize", pipeline_id, now)
            connection.execute(
                """INSERT INTO jobs (
                    job_id, pipeline_id, kind, status, phase, snapshot_json,
                    completed_units, total_units, message, created_at, updated_at
                ) VALUES (?, ?, 'initialize', 'queued', 'planning', ?, 0, ?, ?, ?, ?)""",
                (
                    job_id,
                    pipeline_id,
                    _snapshot_json(snapshot),
                    len(snapshot.days),
                    f"Preparing {len(snapshot.days):,} immutable native-grid days",
                    now_text,
                    now_text,
                ),
            )
            connection.execute(
                "UPDATE pipelines SET status = 'initializing', error_message = NULL, updated_at = ? WHERE pipeline_id = ?",
                (now_text, pipeline_id),
            )
            connection.commit()
        self._queue.put(job_id)
        return self.status()

    def scan(self) -> dict[str, Any]:
        self._require_available()
        now = _aware_utc(self._clock(), "spatial monitor clock")
        now_text = _datetime_text(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pipeline = self._active_pipeline(connection)
            if pipeline is None or not pipeline["active_model_version_id"]:
                connection.rollback()
                raise SpatialMonitorUnavailable(
                    "Train the spatial baseline before checking native maps"
                )
            active = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
            if active is not None:
                if active["kind"] == "scan":
                    connection.commit()
                    return self.status()
                connection.rollback()
                raise SpatialMonitorConflict("Spatial baseline training is still running")
            job_id = _job_id("scan", pipeline["pipeline_id"], now)
            connection.execute(
                """INSERT INTO jobs (
                    job_id, pipeline_id, kind, status, phase,
                    completed_units, total_units, message, created_at, updated_at
                ) VALUES (?, ?, 'scan', 'queued', 'planning', 0, 0, ?, ?, ?)""",
                (
                    job_id,
                    pipeline["pipeline_id"],
                    "Refreshing the shared Mamba source ledger",
                    now_text,
                    now_text,
                ),
            )
            connection.execute(
                "UPDATE pipelines SET status = 'scanning', error_message = NULL, updated_at = ? WHERE pipeline_id = ?",
                (now_text, pipeline["pipeline_id"]),
            )
            connection.commit()
        self._queue.put(job_id)
        return self.status()

    def run_next_job(self) -> bool:
        try:
            job_id = self._queue.get_nowait()
        except queue.Empty:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
                ).fetchone()
            if row is None:
                return False
            job_id = row["job_id"]
        if job_id is None:
            return False
        self._run_job(job_id)
        return True

    def close(self) -> bool:
        if self._closed:
            return self._thread is None or not self._thread.is_alive()
        self._closed = True
        self._stop.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        stopped = self._thread is None or not self._thread.is_alive()
        if stopped:
            self._release_process_lock()
        return stopped

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                job_id = self._queue.get()
                if job_id is None:
                    return
                self._run_job(job_id)
        finally:
            if self._closed:
                self._release_process_lock()

    def _run_job(self, job_id: str) -> None:
        now = _aware_utc(self._clock(), "spatial monitor clock")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None or job["status"] not in {"queued", "interrupted"}:
                connection.rollback()
                return
            connection.execute(
                """UPDATE jobs SET status = 'running', phase = 'starting',
                   started_at = COALESCE(started_at, ?), updated_at = ?,
                   completed_at = NULL, error_message = NULL, retryable = 0
                   WHERE job_id = ?""",
                (_datetime_text(now), _datetime_text(now), job_id),
            )
            connection.commit()
        try:
            if job["kind"] == "initialize":
                self._run_initialize(job_id)
            else:
                self._run_scan(job_id)
        except (_SpatialInterrupted, InterruptedError):
            self._mark_interrupted(job_id)
        except Exception as error:
            self._mark_failed(job_id, error)
            if not isinstance(error, SpatialMonitorError):
                traceback.print_exc()

    def _run_initialize(self, job_id: str) -> None:
        if self._archive is None:
            raise SpatialMonitorUnavailable("The native Zarr archive is unavailable")
        job, pipeline = self._job_and_pipeline(job_id)
        snapshot = _snapshot_from_json(job["snapshot_json"])
        region = _region_from_row(pipeline)
        checkpoint = self._checkpoint_directory / f"{pipeline['pipeline_id']}.json"

        def progress(phase: str, completed: int, total: int, message: str) -> None:
            self._assert_running()
            self._update_job(job_id, phase, completed, total, message)

        from ophanim.core.jobs import heavy_work
        with heavy_work(self._directory, self._stop.is_set):
            result = self._backend.train(
                snapshot=snapshot,
                archive=self._archive,
                region=region,
                options={"epochs": pipeline["requested_epochs"]},
                checkpoint_path=checkpoint,
                progress=progress,
                interrupted=self._stop.is_set,
            )
        _validate_training_result(result)
        if result.training_window_count < self._minimum_training_windows:
            raise SpatialMonitorUnavailable(
                f"Spatial history produced {result.training_window_count:,} training windows; "
                f"at least {self._minimum_training_windows:,} are required"
            )
        self._assert_running()
        stored = self._model_store.put(result.model_content, suffix=".json")
        if stored.checksum_sha256 != sha256(result.model_content).hexdigest():
            raise RuntimeError("spatial model artifact checksum mismatch")
        created = _aware_utc(self._clock(), "spatial monitor clock")
        model_identity = sha256(
            (
                stored.checksum_sha256
                + "|"
                + pipeline["configuration_hash"]
                + "|"
                + result.training_data_hash
            ).encode("ascii")
        ).hexdigest()
        model_version_id = f"spatial-model-{model_identity[:24]}"
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO models (
                    model_version_id, pipeline_id, algorithm, configuration_hash,
                    artifact_ref, artifact_checksum_sha256, training_data_hash,
                    training_start_at, training_end_at, training_map_count,
                    training_window_count, validation_window_count,
                    calibration_window_count, test_window_count, completed_epochs,
                    validation_mae_tecu, test_mae_tecu, residual_center_tecu,
                    residual_scale_tecu, threshold, source_cursor,
                    source_pipeline_id, source_model_version_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version_id) DO UPDATE SET
                    source_pipeline_id = excluded.source_pipeline_id,
                    source_model_version_id = excluded.source_model_version_id
                WHERE models.pipeline_id = excluded.pipeline_id
                  AND models.configuration_hash = excluded.configuration_hash
                  AND models.artifact_checksum_sha256 = excluded.artifact_checksum_sha256
                  AND models.training_data_hash = excluded.training_data_hash
                  AND (
                      models.source_pipeline_id = ''
                      OR models.source_model_version_id = ''
                  )""",
                (
                    model_version_id,
                    pipeline["pipeline_id"],
                    SPATIAL_ALGORITHM,
                    pipeline["configuration_hash"],
                    stored.storage_ref,
                    stored.checksum_sha256,
                    result.training_data_hash,
                    _datetime_text(result.training_start_at),
                    _datetime_text(result.training_end_at),
                    result.training_map_count,
                    result.training_window_count,
                    result.validation_window_count,
                    result.calibration_window_count,
                    result.test_window_count,
                    result.completed_epochs,
                    result.validation_mae_tecu,
                    result.test_mae_tecu,
                    result.residual_center_tecu,
                    result.residual_scale_tecu,
                    result.threshold,
                    snapshot.source_cursor,
                    snapshot.pipeline_id,
                    snapshot.model_version_id,
                    _datetime_text(created),
                ),
            )
            connection.execute(
                """UPDATE pipelines SET status = 'ready', active_model_version_id = ?,
                   source_cursor = ?, scan_cursor = ?, error_message = NULL, updated_at = ?
                   WHERE pipeline_id = ?""",
                (
                    model_version_id,
                    snapshot.source_cursor,
                    snapshot.source_cursor,
                    _datetime_text(created),
                    pipeline["pipeline_id"],
                ),
            )
            self._finish_job(
                connection,
                job_id,
                created,
                f"Spatial baseline fitted on {result.training_window_count:,} causal windows",
            )
            connection.commit()
        checkpoint.unlink(missing_ok=True)

    def _run_scan(self, job_id: str) -> None:
        if self._archive is None:
            raise SpatialMonitorUnavailable("The native Zarr archive is unavailable")
        job, pipeline = self._job_and_pipeline(job_id)
        self._refresh_mamba_sources(job_id)
        self._assert_running()
        snapshot = self._mamba.native_grid_snapshot()
        if snapshot.source_cursor < int(pipeline["scan_cursor"]):
            raise SpatialMonitorUnavailable(
                "The shared native-grid cursor moved backwards; refusing to skip provenance"
            )
        with self._connection() as connection:
            connection.execute(
                "UPDATE jobs SET snapshot_json = ?, total_units = ?, updated_at = ? WHERE job_id = ?",
                (
                    _snapshot_json(snapshot),
                    len(snapshot.days),
                    _datetime_text(_aware_utc(self._clock(), "spatial monitor clock")),
                    job_id,
                ),
            )
            model_row = connection.execute(
                "SELECT * FROM models WHERE model_version_id = ?",
                (pipeline["active_model_version_id"],),
            ).fetchone()
            connection.commit()
        if model_row is None:
            raise SpatialMonitorUnavailable("The active spatial model is missing")
        if model_row["pipeline_id"] != pipeline["pipeline_id"]:
            raise SpatialMonitorUnavailable("The active spatial model belongs to another pipeline")
        if snapshot.pipeline_id != model_row["source_pipeline_id"]:
            raise SpatialMonitorUnavailable(
                "The active Mamba source pipeline changed; retrain the spatial baseline"
            )
        if snapshot.model_version_id != model_row["source_model_version_id"]:
            raise SpatialMonitorUnavailable(
                "The active Mamba source model changed; retrain the spatial baseline"
            )
        model_content = self._model_store.read(model_row["artifact_ref"])
        if sha256(model_content).hexdigest() != model_row["artifact_checksum_sha256"]:
            raise SpatialMonitorUnavailable("The active spatial model artifact is corrupted")

        def progress(phase: str, completed: int, total: int, message: str) -> None:
            self._assert_running()
            self._update_job(job_id, phase, completed, total, message)

        created = _aware_utc(self._clock(), "spatial monitor clock")
        run_id = _run_id(pipeline["pipeline_id"], created)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO scan_runs (
                    scan_run_id, job_id, pipeline_id, model_version_id,
                    from_cursor_exclusive, to_cursor_inclusive, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)""",
                (
                    run_id,
                    job_id,
                    pipeline["pipeline_id"],
                    model_row["model_version_id"],
                    pipeline["scan_cursor"],
                    snapshot.source_cursor,
                    _datetime_text(created),
                ),
            )
            connection.commit()
        from ophanim.core.jobs import heavy_work
        with heavy_work(self._directory, self._stop.is_set):
            result = self._backend.scan(
                model_content=model_content,
                model_metadata=dict(model_row),
                snapshot=snapshot,
                archive=self._archive,
                after_source_cursor=pipeline["scan_cursor"],
                region=_region_from_row(pipeline),
                progress=progress,
                interrupted=self._stop.is_set,
            )
        _validate_scan_result(result)
        self._assert_running()
        completed = _aware_utc(self._clock(), "spatial monitor clock")
        if result.first_observed_at is None or result.last_observed_at is None:
            mamba_window: Mapping[str, Any] = {
                "readout_count": 0,
                "unscored_readout_count": 0,
                "candidates": [],
                "complete": True,
            }
        else:
            try:
                mamba_window = self._mamba.comparison_window(
                    first_observed_at=result.first_observed_at,
                    last_observed_at=result.last_observed_at,
                    expected_region=_region_from_row(pipeline),
                    maximum_source_cursor=snapshot.source_cursor,
                    expected_pipeline_id=model_row["source_pipeline_id"],
                    expected_model_version_id=model_row[
                        "source_model_version_id"
                    ],
                )
            except MambaMonitorUnavailable as error:
                raise SpatialMonitorUnavailable(str(error)) from error
        comparison = _compare_with_mamba(result, mamba_window)
        result_payload = _scan_payload(
            run_id,
            model_row,
            pipeline["scan_cursor"],
            snapshot.source_cursor,
            result,
            comparison,
            completed,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE scan_runs SET status = 'succeeded', result_json = ?,
                   completed_at = ? WHERE scan_run_id = ?""",
                (
                    _canonical_json_text(result_payload),
                    _datetime_text(completed),
                    run_id,
                ),
            )
            connection.execute(
                """UPDATE pipelines SET status = 'ready', source_cursor = ?,
                   scan_cursor = ?, last_successful_scan_at = ?, error_message = NULL,
                   updated_at = ? WHERE pipeline_id = ?""",
                (
                    snapshot.source_cursor,
                    snapshot.source_cursor,
                    _datetime_text(completed),
                    _datetime_text(completed),
                    pipeline["pipeline_id"],
                ),
            )
            self._finish_job(
                connection,
                job_id,
                completed,
                f"Checked {result.usable_map_count:,} spatial maps; {len(result.candidates):,} need review",
            )
            connection.commit()

    def _refresh_mamba_sources(self, job_id: str) -> None:
        self._update_job(
            job_id,
            "acquiring",
            0,
            0,
            "Checking the shared Mamba ledger for new complete UTC days",
        )
        try:
            status = self._mamba.scan()
        except MambaMonitorConflict as error:
            raise SpatialMonitorConflict(str(error)) from error
        except MambaMonitorUnavailable as error:
            raise SpatialMonitorUnavailable(str(error)) from error
        while status.get("state") in {"initializing", "scanning"}:
            self._assert_running()
            if self._stop.wait(self._mamba_poll_seconds):
                raise _SpatialInterrupted
            status = self._mamba.status()
        if status.get("state") == "failed":
            raise SpatialMonitorUnavailable(
                status.get("error") or "The shared Mamba source refresh failed"
            )
        if status.get("state") != "ready":
            raise SpatialMonitorUnavailable(
                "The shared Mamba source ledger did not become ready"
            )

    def _availability(self) -> tuple[bool, str | None]:
        if self._archive is None:
            return False, "Start the Docker PostgreSQL/Zarr data plane to train spatial maps"
        available, reason = self._backend.availability()
        return bool(available), reason

    def _require_available(self) -> None:
        available, reason = self._availability()
        if not available:
            raise SpatialMonitorUnavailable(reason or "The spatial backend is unavailable")

    def _assert_running(self) -> None:
        if self._stop.is_set():
            raise _SpatialInterrupted

    def _initialize_store(self) -> None:
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT schema_version FROM spatial_schema WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO spatial_schema (singleton, schema_version) VALUES (1, ?)",
                    (SPATIAL_MONITOR_SCHEMA_VERSION,),
                )
            elif row["schema_version"] == 1:
                columns = {
                    item["name"]
                    for item in connection.execute("PRAGMA table_info(models)").fetchall()
                }
                if "source_pipeline_id" not in columns:
                    connection.execute(
                        "ALTER TABLE models ADD COLUMN source_pipeline_id TEXT NOT NULL DEFAULT ''"
                    )
                if "source_model_version_id" not in columns:
                    connection.execute(
                        "ALTER TABLE models ADD COLUMN source_model_version_id TEXT NOT NULL DEFAULT ''"
                    )
                migration_message = (
                    "Spatial model provenance was upgraded; retrain the baseline "
                    "before scanning"
                )
                migrated_at = _datetime_text(
                    _aware_utc(self._clock(), "spatial monitor clock")
                )
                connection.execute(
                    """UPDATE jobs SET status = 'failed', phase = 'failed',
                       message = ?, error_message = ?, retryable = 0,
                       updated_at = ?, completed_at = ?
                       WHERE status IN ('queued', 'running', 'interrupted')
                         AND pipeline_id IN (
                             SELECT pipelines.pipeline_id
                             FROM pipelines
                             JOIN models
                               ON models.model_version_id = pipelines.active_model_version_id
                             WHERE models.source_pipeline_id = ''
                                OR models.source_model_version_id = ''
                         )""",
                    (
                        migration_message,
                        migration_message,
                        migrated_at,
                        migrated_at,
                    ),
                )
                connection.execute(
                    """UPDATE pipelines SET active_model_version_id = NULL,
                       status = 'uninitialized', error_message = ?, updated_at = ?
                       WHERE active_model_version_id IN (
                           SELECT model_version_id FROM models
                           WHERE source_pipeline_id = ''
                              OR source_model_version_id = ''
                       )""",
                    (migration_message, migrated_at),
                )
                connection.execute(
                    "UPDATE spatial_schema SET schema_version = ? WHERE singleton = 1",
                    (SPATIAL_MONITOR_SCHEMA_VERSION,),
                )
            elif row["schema_version"] != SPATIAL_MONITOR_SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported spatial monitor schema version {row['schema_version']}"
                )
            connection.commit()

    def _recover_jobs(self) -> tuple[str, ...]:
        now = _datetime_text(_aware_utc(self._clock(), "spatial monitor clock"))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE scan_runs SET status = 'failed', completed_at = ?,
                   error_message = 'Process stopped before the scan committed'
                   WHERE status = 'running'""",
                (now,),
            )
            connection.execute(
                """UPDATE jobs SET status = 'queued', phase = 'resuming',
                   message = 'Resuming from the durable spatial checkpoint',
                   updated_at = ?, retryable = 1
                   WHERE status IN ('running', 'interrupted')""",
                (now,),
            )
            rows = connection.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
            connection.commit()
        return tuple(row["job_id"] for row in rows)

    def _job_and_pipeline(self, job_id: str) -> tuple[sqlite3.Row, sqlite3.Row]:
        with self._connection() as connection:
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise RuntimeError("spatial job disappeared")
            pipeline = connection.execute(
                "SELECT * FROM pipelines WHERE pipeline_id = ?",
                (job["pipeline_id"],),
            ).fetchone()
        if pipeline is None:
            raise RuntimeError("spatial pipeline disappeared")
        return job, pipeline

    def _update_job(
        self,
        job_id: str,
        phase: str,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "spatial monitor clock"))
        with self._connection() as connection:
            connection.execute(
                """UPDATE jobs SET phase = ?, completed_units = ?, total_units = ?,
                   message = ?, updated_at = ? WHERE job_id = ?""",
                (phase, completed, total, message, now, job_id),
            )
            connection.commit()

    def _mark_interrupted(self, job_id: str) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "spatial monitor clock"))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE jobs SET status = 'interrupted', phase = 'interrupted',
                   message = 'Stopped safely; launch again to resume', retryable = 1,
                   updated_at = ?, completed_at = ? WHERE job_id = ?""",
                (now, now, job_id),
            )
            connection.execute(
                """UPDATE scan_runs SET status = 'failed', completed_at = ?,
                   error_message = 'Spatial scan was interrupted'
                   WHERE job_id = ? AND status = 'running'""",
                (now, job_id),
            )
            row = connection.execute(
                "SELECT pipeline_id FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE pipelines SET status = 'failed', error_message = ?, updated_at = ? WHERE pipeline_id = ?",
                    ("Spatial job was interrupted and can be resumed", now, row["pipeline_id"]),
                )
            connection.commit()

    def _mark_failed(self, job_id: str, error: Exception) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "spatial monitor clock"))
        message = str(error) or error.__class__.__name__
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT pipeline_id FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            connection.execute(
                """UPDATE jobs SET status = 'failed', phase = 'failed',
                   error_message = ?, message = ?, retryable = 1,
                   updated_at = ?, completed_at = ? WHERE job_id = ?""",
                (message, message, now, now, job_id),
            )
            connection.execute(
                """UPDATE scan_runs SET status = 'failed', completed_at = ?,
                   error_message = ? WHERE job_id = ? AND status = 'running'""",
                (now, message, job_id),
            )
            if row is not None:
                connection.execute(
                    "UPDATE pipelines SET status = 'failed', error_message = ?, updated_at = ? WHERE pipeline_id = ?",
                    (message, now, row["pipeline_id"]),
                )
            connection.commit()

    @staticmethod
    def _finish_job(
        connection: sqlite3.Connection,
        job_id: str,
        completed_at: datetime,
        message: str,
    ) -> None:
        stamp = _datetime_text(completed_at)
        connection.execute(
            """UPDATE jobs SET status = 'succeeded', phase = 'complete',
               completed_units = total_units, message = ?, updated_at = ?,
               completed_at = ?, error_message = NULL, retryable = 0
               WHERE job_id = ?""",
            (message, stamp, stamp, job_id),
        )

    def _active_pipeline(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = 'active_pipeline_id'"
        ).fetchone()
        if row is None:
            return None
        return connection.execute(
            "SELECT * FROM pipelines WHERE pipeline_id = ?", (row["value"],)
        ).fetchone()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def _acquire_process_lock(self) -> None:
        lock_file = (self._directory / "spatial-monitor.lock").open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise SpatialMonitorUnavailable(
                "The spatial monitor is already active in another OPHANIM process"
            ) from error
        self._process_lock_file = lock_file

    def _release_process_lock(self) -> None:
        with self._process_lock_guard:
            lock_file = self._process_lock_file
            if lock_file is None:
                return
            self._process_lock_file = None
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()


class TorchSpatialBackend:
    """Bounded-memory CPU training and inference over immutable daily grids."""

    def __init__(
        self,
        *,
        batch_size: int = 4,
        checkpoint_interval: int = 250,
        normalization_sample_cells: int = 1_000_000,
        calibration_sample_cells: int = 1_000_000,
    ) -> None:
        if batch_size < 1 or checkpoint_interval < 1:
            raise ValueError("batch_size and checkpoint_interval must be positive")
        self._batch_size = batch_size
        self._checkpoint_interval = checkpoint_interval
        self._normalization_sample_cells = normalization_sample_cells
        self._calibration_sample_cells = calibration_sample_cells

    def availability(self) -> tuple[bool, str | None]:
        if importlib.util.find_spec("numpy") is None:
            return False, "Install OPHANIM's spatial extra (NumPy and CPU PyTorch)"
        if importlib.util.find_spec("torch") is None:
            return False, "Install OPHANIM's spatial extra (CPU PyTorch)"
        return True, None

    def train(
        self,
        *,
        snapshot: NativeGridSnapshot,
        archive: BulkTECArchive,
        region: MonitorRegion,
        options: Mapping[str, Any],
        checkpoint_path: Path,
        progress: Callable[[str, int, int, str], None],
        interrupted: Callable[[], bool],
    ) -> SpatialTrainingResult:
        """Fit the causal model without materializing the multi-year cube."""

        available, reason = self.availability()
        if not available:
            raise SpatialMonitorUnavailable(reason or "The spatial backend is unavailable")
        import numpy as np

        from ophanim.convlstm_model import (
            ConvLSTMConfig,
            GridNormalization,
            create_model,
            create_optimizer,
            load_checkpoint,
            serialize_checkpoint,
            serialize_model,
            train_batch,
        )

        epochs = options.get("epochs", DEFAULT_EPOCHS)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10:
            raise ValueError("spatial training epochs must be an integer between 1 and 10")
        config = ConvLSTMConfig()
        splits = _chronological_day_splits(snapshot.days)
        training_days, validation_days, calibration_days, test_days = splits
        training_hash = _training_manifest_hash(snapshot, config.to_dict(), splits)

        progress("normalizing", 0, len(training_days), "Fitting training-only TEC normalization")
        sample_parts: list[Any] = []
        sample_count = 0
        estimated_frames = max(1, len(training_days) * 12)
        cells_per_frame = max(1, self._normalization_sample_cells // estimated_frames)
        training_map_count = 0
        training_start: datetime | None = None
        training_end: datetime | None = None
        for frame_index, frame in enumerate(_iter_native_frames(training_days, archive)):
            if interrupted():
                raise _SpatialInterrupted
            training_map_count += 1
            training_start = training_start or frame.observed_at
            training_end = frame.observed_at
            remaining = self._normalization_sample_cells - sample_count
            if remaining > 0:
                picked = _deterministic_cell_sample(
                    frame.vtec_tecu,
                    frame.valid,
                    min(cells_per_frame, remaining),
                    phase=frame_index,
                    np=np,
                )
                if picked.size:
                    sample_parts.append(picked)
                    sample_count += int(picked.size)
            if frame_index % 250 == 0:
                progress(
                    "normalizing",
                    min(len(training_days), frame_index // 12),
                    len(training_days),
                    f"Sampled {sample_count:,} valid training cells",
                )
        if training_start is None or training_end is None:
            raise SpatialMonitorUnavailable("The training split contains no native maps")
        if sample_count < 2:
            raise SpatialMonitorUnavailable("The training split contains too few valid TEC cells")
        normalization_values = np.concatenate(sample_parts)
        center = float(np.median(normalization_values))
        mad = float(np.median(np.abs(normalization_values - center)))
        normalization = GridNormalization(
            center_tecu=center,
            scale_tecu=max(0.1, 1.4826 * mad),
        )

        progress("planning", 0, 4, "Counting gap-free causal windows in each chronological split")
        counts = tuple(
            _count_windows(days, archive, interrupted=interrupted)
            for days in splits
        )
        training_count, validation_count, calibration_count, test_count = counts
        if training_count == 0:
            raise SpatialMonitorUnavailable(
                "The training history has no complete 12-frame two-hour windows"
            )
        if validation_count == 0 or calibration_count == 0 or test_count == 0:
            raise SpatialMonitorUnavailable(
                "Every chronological evaluation split must contain a complete causal window"
            )

        model = create_model(config)
        optimizer = create_optimizer(model, config)
        completed_epochs = 0
        next_window_index = 0
        if checkpoint_path.exists():
            try:
                restored = load_checkpoint(checkpoint_path.read_bytes())
            except Exception:
                checkpoint_path.unlink(missing_ok=True)
            else:
                if (
                    restored.config == config
                    and restored.normalization == normalization
                    and restored.training_data_sha256 == training_hash
                    and restored.completed_epochs <= epochs
                    and restored.next_window_index <= training_count
                ):
                    model = restored.model
                    optimizer = restored.optimizer
                    completed_epochs = restored.completed_epochs
                    next_window_index = restored.next_window_index
                else:
                    checkpoint_path.unlink(missing_ok=True)

        total_training_units = epochs * training_count
        for epoch_index in range(completed_epochs, epochs):
            skip = next_window_index if epoch_index == completed_epochs else 0
            batch_windows: list[_GridWindow] = []
            processed = skip
            last_checkpoint_cursor = skip
            for window_index, window in enumerate(
                _iter_native_windows(training_days, archive)
            ):
                if window_index < skip:
                    continue
                if interrupted():
                    self._save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        normalization=normalization,
                        training_hash=training_hash,
                        completed_epochs=epoch_index,
                        next_window_index=window_index - len(batch_windows),
                        serialize_checkpoint=serialize_checkpoint,
                    )
                    raise _SpatialInterrupted
                batch_windows.append(window)
                if len(batch_windows) < self._batch_size:
                    continue
                batch = _causal_batch(batch_windows, config, normalization)
                train_batch(model, optimizer, batch)
                processed = window_index + 1
                batch_windows.clear()
                if processed - last_checkpoint_cursor >= self._checkpoint_interval:
                    self._save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        normalization=normalization,
                        training_hash=training_hash,
                        completed_epochs=epoch_index,
                        next_window_index=processed,
                        serialize_checkpoint=serialize_checkpoint,
                    )
                    last_checkpoint_cursor = processed
                progress(
                    "training",
                    epoch_index * training_count + processed,
                    total_training_units,
                    f"Epoch {epoch_index + 1} of {epochs}: {processed:,}/{training_count:,} windows",
                )
            if batch_windows:
                batch = _causal_batch(batch_windows, config, normalization)
                train_batch(model, optimizer, batch)
                processed = training_count
            if processed != training_count:
                raise SpatialMonitorUnavailable(
                    "The immutable training source changed while it was being read"
                )
            completed_epochs = epoch_index + 1
            next_window_index = 0
            self._save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                normalization=normalization,
                training_hash=training_hash,
                completed_epochs=completed_epochs,
                next_window_index=0,
                serialize_checkpoint=serialize_checkpoint,
            )
            progress(
                "training",
                completed_epochs * training_count,
                total_training_units,
                f"Completed epoch {completed_epochs} of {epochs}",
            )

        validation_mae = _stream_mae(
            model,
            config,
            normalization,
            validation_days,
            archive,
            interrupted,
            progress,
            "validating",
            validation_count,
        )
        residual_center, residual_scale, threshold = _calibrate_residuals(
            model=model,
            config=config,
            normalization=normalization,
            days=calibration_days,
            archive=archive,
            region=region,
            maximum_cells=self._calibration_sample_cells,
            interrupted=interrupted,
            progress=progress,
            window_count=calibration_count,
        )
        test_mae = _stream_mae(
            model,
            config,
            normalization,
            test_days,
            archive,
            interrupted,
            progress,
            "testing",
            test_count,
        )
        content = serialize_model(
            model,
            config=config,
            normalization=normalization,
            training_data_sha256=training_hash,
            completed_epochs=completed_epochs,
        )
        return SpatialTrainingResult(
            model_content=content,
            training_data_hash=training_hash,
            training_start_at=training_start,
            training_end_at=training_end,
            training_map_count=training_map_count,
            training_window_count=training_count,
            validation_window_count=validation_count,
            calibration_window_count=calibration_count,
            test_window_count=test_count,
            completed_epochs=completed_epochs,
            validation_mae_tecu=validation_mae,
            test_mae_tecu=test_mae,
            residual_center_tecu=residual_center,
            residual_scale_tecu=residual_scale,
            threshold=threshold,
        )

    def scan(
        self,
        *,
        model_content: bytes,
        model_metadata: Mapping[str, Any],
        snapshot: NativeGridSnapshot,
        archive: BulkTECArchive,
        after_source_cursor: int,
        region: MonitorRegion,
        progress: Callable[[str, int, int, str], None],
        interrupted: Callable[[], bool],
    ) -> SpatialScanResult:
        """Score only source days published after the successful scan cursor."""

        available, reason = self.availability()
        if not available:
            raise SpatialMonitorUnavailable(reason or "The spatial backend is unavailable")
        import numpy as np

        from ophanim.convlstm_model import load_model, predict_next

        loaded = load_model(model_content)
        if loaded.training_data_sha256 != model_metadata.get("training_data_hash"):
            raise SpatialMonitorUnavailable("Spatial model provenance does not match its database row")
        changed = tuple(
            day for day in snapshot.days if day.source_day_id > after_source_cursor
        )
        if not changed:
            return SpatialScanResult(None, None, 0, 0, None, ())
        changed_dates = {day.product_date for day in changed}
        # A revised map is both a target and a causal input for the following
        # 24 hours.  Rescore that dependent day when it already exists.
        affected_dates = changed_dates | {
            value + timedelta(days=1) for value in changed_dates
        }
        target_days = tuple(
            day for day in snapshot.days if day.product_date in affected_dates
        )
        needed_dates = affected_dates | {
            value - timedelta(days=1) for value in affected_dates
        }
        scan_days = tuple(day for day in snapshot.days if day.product_date in needed_dates)
        training_end = _parse_datetime(str(model_metadata["training_end_at"]))
        threshold = float(model_metadata["threshold"])
        residual_center = float(model_metadata["residual_center_tecu"])
        residual_scale = float(model_metadata["residual_scale_tecu"])
        if not all(math.isfinite(value) for value in (threshold, residual_center, residual_scale)):
            raise SpatialMonitorUnavailable("Spatial model calibration metadata is invalid")
        if residual_scale <= 0.0:
            raise SpatialMonitorUnavailable("Spatial residual scale must be positive")
        region_mask: Any | None = None
        candidates: list[dict[str, Any]] = []
        first: datetime | None = None
        last: datetime | None = None
        map_count = 0
        usable_count = 0
        maximum_score: float | None = None
        windows = _iter_native_windows(
            scan_days,
            archive,
            target_day_ids={day.source_day_id for day in target_days},
        )
        for window in windows:
            if interrupted():
                raise _SpatialInterrupted
            if window.target.observed_at <= training_end:
                continue
            map_count += 1
            first = first or window.target.observed_at
            last = window.target.observed_at
            if region_mask is None:
                region_mask = _region_mask(window, region, np=np)
            prediction = predict_next(
                loaded.model,
                _window_history_values(window, np=np),
                _window_history_masks(window, np=np),
                config=loaded.config,
                normalization=loaded.normalization,
            )
            predicted = prediction.predicted_vtec_tecu[0, 0].numpy()
            valid = (
                prediction.valid_mask[0, 0].numpy()
                & window.target.valid
                & region_mask
            )
            if not bool(np.any(valid)):
                progress("scanning", map_count, len(target_days) * 12, "Skipping a map with no valid regional cells")
                continue
            usable_count += 1
            signed = window.target.vtec_tecu - predicted
            absolute = np.abs(signed)
            cell_scores = np.zeros_like(absolute, dtype=np.float32)
            cell_scores[valid] = np.maximum(
                0.0,
                (absolute[valid] - residual_center) / residual_scale,
            )
            score = float(np.quantile(cell_scores[valid], 0.95))
            maximum_score = score if maximum_score is None else max(maximum_score, score)
            if score >= threshold:
                peak_index = int(np.argmax(np.where(valid, cell_scores, -1.0)))
                latitude_index, longitude_index = np.unravel_index(
                    peak_index, cell_scores.shape
                )
                cluster = _peak_cluster_size(
                    (cell_scores >= threshold) & valid,
                    latitude_index,
                    longitude_index,
                    np=np,
                )
                candidates.append(
                    {
                        "observed_at": _datetime_text(window.target.observed_at),
                        "source_day_id": window.target.source_day_id,
                        "anomaly_score": score,
                        "threshold": threshold,
                        "peak": {
                            "latitude_degrees": float(window.latitudes[latitude_index]),
                            "longitude_degrees": float(window.longitudes[longitude_index]),
                            "residual_tecu": float(signed[latitude_index, longitude_index]),
                            "anomaly_score": float(cell_scores[latitude_index, longitude_index]),
                        },
                        "peak_residual_tecu": float(signed[latitude_index, longitude_index]),
                        "cluster_cell_count": cluster,
                        "contributing_cell_count": int(np.count_nonzero(valid)),
                        "assessment": "candidate",
                    }
                )
            progress(
                "scanning",
                map_count,
                len(target_days) * 12,
                f"Scored {usable_count:,} causal native maps",
            )
        candidates.sort(
            key=lambda item: (float(item["anomaly_score"]), item["observed_at"]),
            reverse=True,
        )
        candidate_count = len(candidates)
        return SpatialScanResult(
            first_observed_at=first,
            last_observed_at=last,
            map_count=map_count,
            usable_map_count=usable_count,
            max_anomaly_score=maximum_score,
            candidates=tuple(candidates[:MAX_CANDIDATES_IN_STATUS]),
            candidate_map_count=candidate_count,
        )

    @staticmethod
    def _save_checkpoint(
        path: Path,
        *,
        model: Any,
        optimizer: Any,
        config: Any,
        normalization: Any,
        training_hash: str,
        completed_epochs: int,
        next_window_index: int,
        serialize_checkpoint: Callable[..., bytes],
    ) -> None:
        content = serialize_checkpoint(
            model,
            optimizer,
            config=config,
            normalization=normalization,
            training_data_sha256=training_hash,
            completed_epochs=completed_epochs,
            next_window_index=next_window_index,
        )
        _atomic_write(path, content)


@dataclass(frozen=True, slots=True)
class _GridFrame:
    source_day_id: int
    product_date: date
    observed_at: datetime
    latitudes: Any
    longitudes: Any
    vtec_tecu: Any
    valid: Any


@dataclass(frozen=True, slots=True)
class _GridWindow:
    history: tuple[_GridFrame, ...]
    target: _GridFrame

    @property
    def latitudes(self) -> Any:
        return self.target.latitudes

    @property
    def longitudes(self) -> Any:
        return self.target.longitudes


def _iter_native_frames(
    days: Sequence[NativeGridDay],
    archive: BulkTECArchive,
) -> Iterator[_GridFrame]:
    """Yield verified, owned-day frames in strict timestamp order.

    CODE daily products commonly repeat the following day's 00:00 epoch.  The
    product-day filter below is therefore an integrity boundary, not merely an
    optimization: it prevents a duplicated midnight map from entering two
    chronological splits or causal windows.
    """

    import numpy as np

    expected_latitudes = np.asarray(
        tuple(index * 2.5 - 87.5 for index in range(71)), dtype=np.float64
    )
    expected_longitudes = np.asarray(
        tuple(index * 5.0 - 180.0 for index in range(72)), dtype=np.float64
    )
    previous_time: datetime | None = None
    previous_date: date | None = None
    for day in days:
        if previous_date is not None and day.product_date <= previous_date:
            raise SpatialMonitorUnavailable(
                "Native-grid snapshot days must be unique and chronological"
            )
        previous_date = day.product_date
        loaded = archive.read_native(day.grid_set_id)
        if loaded.manifest.grid_set_id != day.grid_set_id:
            raise SpatialMonitorUnavailable("Native-grid manifest identity mismatch")
        if loaded.manifest.source_checksum_sha256 != day.checksum_sha256:
            raise SpatialMonitorUnavailable("Native-grid source checksum mismatch")
        if loaded.manifest.revision_priority != day.revision_priority:
            raise SpatialMonitorUnavailable("Native-grid revision provenance mismatch")
        if tuple(loaded.manifest.shape[1:]) != (71, 72):
            raise SpatialMonitorUnavailable("Native-grid manifest shape is not 71 by 72")
        times = np.asarray(loaded.times_utc_microseconds)
        latitudes = np.asarray(loaded.latitudes_degrees, dtype=np.float64)
        longitudes = np.asarray(loaded.longitudes_degrees, dtype=np.float64)
        values = np.asarray(loaded.vtec_tecu, dtype=np.float32)
        valid = np.asarray(loaded.valid, dtype=np.bool_)
        if times.ndim != 1:
            raise SpatialMonitorUnavailable("Native-grid time axis must be one-dimensional")
        if values.shape != (len(times), 71, 72) or valid.shape != values.shape:
            raise SpatialMonitorUnavailable(
                "Spatial modeling requires the canonical 71 by 72 native CODE grid"
            )
        if (
            latitudes.shape != expected_latitudes.shape
            or longitudes.shape != expected_longitudes.shape
            or not np.array_equal(latitudes, expected_latitudes)
            or not np.array_equal(longitudes, expected_longitudes)
        ):
            raise SpatialMonitorUnavailable(
                "Native-grid axes do not match the canonical 2.5 by 5 degree CODE grid"
            )
        start = datetime.combine(day.product_date, datetime.min.time(), tzinfo=UTC)
        end = start + timedelta(days=1)
        ordered: list[tuple[datetime, int]] = []
        for index, raw_time in enumerate(times):
            try:
                observed_at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                    microseconds=int(raw_time)
                )
            except (OverflowError, TypeError, ValueError) as error:
                raise SpatialMonitorUnavailable("Native-grid timestamp is invalid") from error
            if start <= observed_at < end:
                ordered.append((observed_at, index))
        ordered.sort(key=lambda item: item[0])
        if len({item[0] for item in ordered}) != len(ordered):
            raise SpatialMonitorUnavailable(
                f"Native-grid day {day.product_date.isoformat()} contains duplicate epochs"
            )
        for observed_at, index in ordered:
            if previous_time is not None and observed_at <= previous_time:
                raise SpatialMonitorUnavailable(
                    "Native-grid epochs are duplicated or out of chronological order"
                )
            frame_values = values[index]
            frame_valid = valid[index]
            if not bool(np.all(np.isfinite(frame_values[frame_valid]))):
                raise SpatialMonitorUnavailable("A valid native TEC cell is non-finite")
            previous_time = observed_at
            yield _GridFrame(
                source_day_id=day.source_day_id,
                product_date=day.product_date,
                observed_at=observed_at,
                latitudes=latitudes,
                longitudes=longitudes,
                vtec_tecu=frame_values,
                valid=frame_valid,
            )


def _iter_native_windows(
    days: Sequence[NativeGridDay],
    archive: BulkTECArchive,
    *,
    target_day_ids: set[int] | None = None,
) -> Iterator[_GridWindow]:
    frames: deque[_GridFrame] = deque(maxlen=13)
    previous: datetime | None = None
    for frame in _iter_native_frames(days, archive):
        if previous is None or frame.observed_at - previous != _CADENCE:
            frames.clear()
        previous = frame.observed_at
        frames.append(frame)
        if len(frames) != 13:
            continue
        target = frames[-1]
        if target_day_ids is not None and target.source_day_id not in target_day_ids:
            continue
        yield _GridWindow(history=tuple(frames)[:-1], target=target)


def _count_windows(
    days: Sequence[NativeGridDay],
    archive: BulkTECArchive,
    *,
    interrupted: Callable[[], bool],
) -> int:
    count = 0
    for count, _window in enumerate(_iter_native_windows(days, archive), start=1):
        if count % 250 == 0 and interrupted():
            raise _SpatialInterrupted
    if interrupted():
        raise _SpatialInterrupted
    return count


def _chronological_day_splits(
    days: Sequence[NativeGridDay],
) -> tuple[
    tuple[NativeGridDay, ...],
    tuple[NativeGridDay, ...],
    tuple[NativeGridDay, ...],
    tuple[NativeGridDay, ...],
]:
    ordered = tuple(days)
    count = len(ordered)
    if count < 20:
        raise SpatialMonitorUnavailable(
            "At least 20 native-grid days are required for chronological train, validation, calibration, and test splits"
        )
    train_end = int(count * 0.70)
    validation_end = int(count * 0.80)
    calibration_end = int(count * 0.90)
    splits = (
        ordered[:train_end],
        ordered[train_end:validation_end],
        ordered[validation_end:calibration_end],
        ordered[calibration_end:],
    )
    if any(len(split) < 2 for split in splits):
        raise SpatialMonitorUnavailable("Chronological spatial splits are too small")
    return splits


def _training_manifest_hash(
    snapshot: NativeGridSnapshot,
    configuration: Mapping[str, Any],
    splits: Sequence[Sequence[NativeGridDay]],
) -> str:
    labels = ("training", "validation", "calibration", "test")
    return _json_hash(
        {
            "algorithm": SPATIAL_ALGORITHM,
            "configuration": dict(configuration),
            "source_pipeline_id": snapshot.pipeline_id,
            "source_model_version_id": snapshot.model_version_id,
            "splits": {
                label: [
                    {
                        "source_day_id": day.source_day_id,
                        "product_date": day.product_date.isoformat(),
                        "grid_set_id": day.grid_set_id,
                        "checksum_sha256": day.checksum_sha256,
                        "revision_priority": day.revision_priority,
                    }
                    for day in days
                ]
                for label, days in zip(labels, splits, strict=True)
            },
        }
    )


def _deterministic_cell_sample(
    values: Any,
    valid: Any,
    maximum: int,
    *,
    phase: int,
    np: Any,
) -> Any:
    if maximum <= 0:
        return np.empty((0,), dtype=np.float32)
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    indexes = np.flatnonzero(np.asarray(valid, dtype=np.bool_).reshape(-1))
    if indexes.size <= maximum:
        return flattened[indexes].copy()
    offset = phase % indexes.size
    positions = (offset + np.linspace(0, indexes.size - 1, maximum, dtype=np.int64)) % indexes.size
    return flattened[indexes[positions]].copy()


def _window_history_values(window: _GridWindow, *, np: Any) -> Any:
    return np.stack([frame.vtec_tecu for frame in window.history], axis=0)


def _window_history_masks(window: _GridWindow, *, np: Any) -> Any:
    return np.stack([frame.valid for frame in window.history], axis=0)


def _causal_batch(windows: Sequence[_GridWindow], config: Any, normalization: Any) -> Any:
    import numpy as np
    import torch

    from ophanim.convlstm_model import CausalGridBatch, build_causal_batch

    batches = []
    for window in windows:
        values = np.concatenate(
            (_window_history_values(window, np=np), window.target.vtec_tecu[None, :, :]),
            axis=0,
        )
        masks = np.concatenate(
            (_window_history_masks(window, np=np), window.target.valid[None, :, :]),
            axis=0,
        )
        times = tuple(frame.observed_at for frame in window.history) + (
            window.target.observed_at,
        )
        batches.append(
            build_causal_batch(
                values,
                masks,
                (config.history_frames,),
                config=config,
                normalization=normalization,
                observed_at=times,
            )
        )
    return CausalGridBatch(
        inputs=torch.cat([batch.inputs for batch in batches], dim=0),
        reference_vtec=torch.cat([batch.reference_vtec for batch in batches], dim=0),
        target_residual=torch.cat([batch.target_residual for batch in batches], dim=0),
        target_vtec=torch.cat([batch.target_vtec for batch in batches], dim=0),
        loss_mask=torch.cat([batch.loss_mask for batch in batches], dim=0),
        target_indices=tuple(range(len(batches))),
    )


def _prediction_error(
    model: Any,
    config: Any,
    normalization: Any,
    window: _GridWindow,
    *,
    np: Any,
) -> tuple[Any, Any]:
    from ophanim.convlstm_model import predict_next

    prediction = predict_next(
        model,
        _window_history_values(window, np=np),
        _window_history_masks(window, np=np),
        config=config,
        normalization=normalization,
    )
    predicted = prediction.predicted_vtec_tecu[0, 0].numpy()
    valid = prediction.valid_mask[0, 0].numpy() & window.target.valid
    return window.target.vtec_tecu - predicted, valid


def _stream_mae(
    model: Any,
    config: Any,
    normalization: Any,
    days: Sequence[NativeGridDay],
    archive: BulkTECArchive,
    interrupted: Callable[[], bool],
    progress: Callable[[str, int, int, str], None],
    phase: str,
    window_count: int,
) -> float:
    import numpy as np

    if window_count == 0:
        raise SpatialMonitorUnavailable(f"The {phase} split has no complete causal windows")
    absolute_sum = 0.0
    cell_count = 0
    for index, window in enumerate(_iter_native_windows(days, archive), start=1):
        if interrupted():
            raise _SpatialInterrupted
        error, valid = _prediction_error(
            model, config, normalization, window, np=np
        )
        if bool(np.any(valid)):
            absolute_sum += float(np.abs(error[valid]).sum(dtype=np.float64))
            cell_count += int(np.count_nonzero(valid))
        if index == 1 or index % 25 == 0 or index == window_count:
            progress(
                phase,
                index,
                window_count,
                f"Assessed {index:,} of {window_count:,} causal windows",
            )
    if cell_count == 0:
        raise SpatialMonitorUnavailable(f"The {phase} split has no valid target cells")
    return absolute_sum / cell_count


def _calibrate_residuals(
    *,
    model: Any,
    config: Any,
    normalization: Any,
    days: Sequence[NativeGridDay],
    archive: BulkTECArchive,
    region: MonitorRegion,
    maximum_cells: int,
    interrupted: Callable[[], bool],
    progress: Callable[[str, int, int, str], None],
    window_count: int,
) -> tuple[float, float, float]:
    import numpy as np

    if window_count == 0:
        raise SpatialMonitorUnavailable("The calibration split has no complete causal windows")
    sample_parts: list[Any] = []
    sample_count = 0
    per_window = max(1, maximum_cells // window_count)
    region_mask: Any | None = None
    for index, window in enumerate(_iter_native_windows(days, archive), start=1):
        if interrupted():
            raise _SpatialInterrupted
        error, valid = _prediction_error(
            model, config, normalization, window, np=np
        )
        if region_mask is None:
            region_mask = _region_mask(window, region, np=np)
        valid = valid & region_mask
        remaining = maximum_cells - sample_count
        if remaining > 0 and bool(np.any(valid)):
            selected = _deterministic_cell_sample(
                np.abs(error),
                valid,
                min(per_window, remaining),
                phase=index,
                np=np,
            )
            if selected.size:
                sample_parts.append(selected)
                sample_count += int(selected.size)
        if index == 1 or index % 25 == 0 or index == window_count:
            progress(
                "calibrating",
                index,
                window_count * 2,
                f"Collected residuals from {index:,} calibration windows",
            )
    if sample_count < 2:
        raise SpatialMonitorUnavailable("Calibration has too few valid regional residuals")
    residuals = np.concatenate(sample_parts)
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    scale = max(0.1, 1.4826 * mad)

    map_scores: list[float] = []
    for index, window in enumerate(_iter_native_windows(days, archive), start=1):
        if interrupted():
            raise _SpatialInterrupted
        error, valid = _prediction_error(
            model, config, normalization, window, np=np
        )
        if region_mask is None:
            region_mask = _region_mask(window, region, np=np)
        valid = valid & region_mask
        if bool(np.any(valid)):
            cell_scores = np.maximum(0.0, (np.abs(error[valid]) - center) / scale)
            map_scores.append(float(np.quantile(cell_scores, 0.95)))
        if index == 1 or index % 25 == 0 or index == window_count:
            progress(
                "calibrating",
                window_count + index,
                window_count * 2,
                f"Calibrated {index:,} of {window_count:,} residual maps",
            )
    if not map_scores:
        raise SpatialMonitorUnavailable("Calibration has no scorable regional maps")
    threshold = max(
        3.0,
        float(np.quantile(np.asarray(map_scores, dtype=np.float64), DEFAULT_THRESHOLD_QUANTILE)),
    )
    return center, scale, threshold


def _region_mask(window: _GridWindow, region: MonitorRegion, *, np: Any) -> Any:
    latitude_mask = (window.latitudes >= region.south) & (
        window.latitudes <= region.north
    )
    longitude_mask = (window.longitudes >= region.west) & (
        window.longitudes <= region.east
    )
    mask = latitude_mask[:, None] & longitude_mask[None, :]
    if not bool(np.any(mask)):
        raise SpatialMonitorUnavailable(
            "The selected region contains no native CODE grid cells"
        )
    return mask


def _peak_cluster_size(
    mask: Any,
    latitude_index: int,
    longitude_index: int,
    *,
    np: Any,
) -> int:
    values = np.asarray(mask, dtype=np.bool_)
    height, width = values.shape
    if not bool(values[latitude_index, longitude_index]):
        return 0
    pending = [(latitude_index, longitude_index)]
    visited = {(latitude_index, longitude_index)}
    while pending:
        latitude, longitude = pending.pop()
        neighbors = (
            (latitude - 1, longitude),
            (latitude + 1, longitude),
            (latitude, (longitude - 1) % width),
            (latitude, (longitude + 1) % width),
        )
        for candidate_latitude, candidate_longitude in neighbors:
            if not 0 <= candidate_latitude < height:
                continue
            candidate = (candidate_latitude, candidate_longitude)
            if candidate in visited or not bool(values[candidate]):
                continue
            visited.add(candidate)
            pending.append(candidate)
    return len(visited)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _history_suffix(
    days: Sequence[NativeGridDay], years: int
) -> tuple[NativeGridDay, ...]:
    ordered = tuple(days)
    if not ordered:
        return ()
    cutoff = _subtract_years(ordered[-1].product_date, years)
    return tuple(day for day in ordered if cutoff <= day.product_date <= ordered[-1].product_date)


def _snapshot_json(snapshot: NativeGridSnapshot) -> str:
    return _canonical_json_text(
        {
            "pipeline_id": snapshot.pipeline_id,
            "model_version_id": snapshot.model_version_id,
            "sync_through_date": snapshot.sync_through_date.isoformat(),
            "source_cursor": snapshot.source_cursor,
            "days": [
                {
                    "source_day_id": day.source_day_id,
                    "product_date": day.product_date.isoformat(),
                    "grid_set_id": day.grid_set_id,
                    "checksum_sha256": day.checksum_sha256,
                    "revision_priority": day.revision_priority,
                }
                for day in snapshot.days
            ],
        }
    )


def _snapshot_from_json(content: str | None) -> NativeGridSnapshot:
    if not content:
        raise SpatialMonitorUnavailable("Spatial job is missing its frozen source snapshot")
    try:
        payload = json.loads(content)
        if not isinstance(payload, dict) or set(payload) != {
            "pipeline_id",
            "model_version_id",
            "sync_through_date",
            "source_cursor",
            "days",
        }:
            raise ValueError
        raw_days = payload["days"]
        if not isinstance(raw_days, list):
            raise ValueError
        days = tuple(
            NativeGridDay(
                source_day_id=_positive_integer(item["source_day_id"]),
                product_date=date.fromisoformat(item["product_date"]),
                grid_set_id=_digest_text(item["grid_set_id"], "grid_set_id"),
                checksum_sha256=_digest_text(
                    item["checksum_sha256"], "checksum_sha256"
                ),
                revision_priority=_nonnegative_integer(item["revision_priority"]),
            )
            for item in raw_days
            if isinstance(item, dict)
            and set(item)
            == {
                "source_day_id",
                "product_date",
                "grid_set_id",
                "checksum_sha256",
                "revision_priority",
            }
        )
        if len(days) != len(raw_days) or not days:
            raise ValueError
        source_cursor = _positive_integer(payload["source_cursor"])
        snapshot = NativeGridSnapshot(
            pipeline_id=_nonempty_text(payload["pipeline_id"], "pipeline_id"),
            model_version_id=_nonempty_text(
                payload["model_version_id"], "model_version_id"
            ),
            sync_through_date=date.fromisoformat(payload["sync_through_date"]),
            source_cursor=source_cursor,
            days=days,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SpatialMonitorUnavailable("Spatial job source snapshot is invalid") from error
    if tuple(sorted(days, key=lambda item: item.product_date)) != days:
        raise SpatialMonitorUnavailable("Spatial job source snapshot is not chronological")
    if len({day.product_date for day in days}) != len(days):
        raise SpatialMonitorUnavailable("Spatial job source snapshot contains duplicate days")
    if source_cursor != max(day.source_day_id for day in days):
        raise SpatialMonitorUnavailable("Spatial job source cursor does not match its days")
    return snapshot


def _region_from_row(row: Mapping[str, Any]) -> MonitorRegion:
    return MonitorRegion(
        name=row["region_name"],
        south=float(row["south"]),
        west=float(row["west"]),
        north=float(row["north"]),
        east=float(row["east"]),
    )


def _model_payload(row: Mapping[str, Any], pipeline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_version_id": row["model_version_id"],
        "algorithm": row["algorithm"],
        "architecture": "predictive-convlstm",
        "trained_at": row["created_at"],
        "training_start": row["training_start_at"],
        "training_end": row["training_end_at"],
        "training_map_count": row["training_map_count"],
        "training_window_count": row["training_window_count"],
        "validation_window_count": row["validation_window_count"],
        "calibration_window_count": row["calibration_window_count"],
        "test_window_count": row["test_window_count"],
        "completed_epochs": row["completed_epochs"],
        "validation_mae_tecu": row["validation_mae_tecu"],
        "test_mae_tecu": row["test_mae_tecu"],
        "residual_center_tecu": row["residual_center_tecu"],
        "residual_scale_tecu": row["residual_scale_tecu"],
        "artifact_checksum_sha256": row["artifact_checksum_sha256"],
        "training_data_hash": row["training_data_hash"],
        "threshold": row["threshold"],
        "source_cursor": row["source_cursor"],
        "input_frame_count": 12,
        "forecast_horizon_hours": 2,
        "native_grid_shape": [71, 72],
        "native_grid_resolution_degrees": {
            "latitude": 2.5,
            "longitude": 5.0,
        },
        "region": _region_from_row(pipeline).to_dict(),
    }


def _job_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    total = int(row["total_units"])
    completed = int(row["completed_units"])
    return {
        "job_id": row["job_id"],
        "kind": row["kind"],
        "status": row["status"],
        "phase": row["phase"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "completed_units": completed,
        "total_units": total,
        "progress_fraction": 0.0 if total <= 0 else min(1.0, completed / total),
        "message": row["message"],
        "retryable": bool(row["retryable"]),
        "error": row["error_message"],
    }


def _empty_scan_payload() -> dict[str, Any]:
    return {
        "last_successful_at": None,
        "last_successful_cursor": None,
        "latest_available_cursor": None,
        "pending_frame_count": 0,
        "last_result": None,
    }


def _storage_payload(has_native_zarr: bool) -> dict[str, Any]:
    return {
        "native_zarr_enabled": has_native_zarr,
        "training_source": "shared immutable native CODE Zarr",
        "derived_half_degree_history": False,
        "model_artifacts": "content-addressed canonical JSON",
        "checkpoint_format": "pickle-free model and AdamW state",
    }


def _compare_with_mamba(
    result: SpatialScanResult,
    mamba_source: Mapping[str, Any],
) -> dict[str, Any]:
    # Accept the old status envelope for compatibility with injected test
    # monitors, but production passes Mamba's exact persisted time window.
    scan = mamba_source.get("scan") if isinstance(mamba_source, Mapping) else None
    enclosed = scan.get("last_result") if isinstance(scan, Mapping) else None
    mamba_result = enclosed if isinstance(enclosed, Mapping) else mamba_source
    raw_candidates = mamba_result.get("candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    spatial_times = {
        _canonical_observed_text(item.get("observed_at"))
        for item in result.candidates
        if isinstance(item, Mapping) and item.get("observed_at")
    }
    spatial_times.discard(None)
    mamba_times = {
        _canonical_observed_text(item.get("observed_at"))
        for item in raw_candidates
        if isinstance(item, Mapping) and item.get("observed_at")
    }
    mamba_times.discard(None)
    if result.first_observed_at is not None and result.last_observed_at is not None:
        first = result.first_observed_at
        last = result.last_observed_at
        mamba_times = {
            value
            for value in mamba_times
            if value is not None and first <= _parse_datetime(value) <= last
        }
    both = spatial_times & mamba_times
    spatial_only = spatial_times - mamba_times
    mamba_only = mamba_times - spatial_times
    union = spatial_times | mamba_times
    spatial_total = (
        len(result.candidates)
        if result.candidate_map_count is None
        else result.candidate_map_count
    )
    raw_mamba_total = mamba_result.get(
        "candidate_readout_count", len(mamba_times)
    )
    try:
        mamba_total = max(0, int(raw_mamba_total))
    except (TypeError, ValueError, OverflowError):
        mamba_total = len(mamba_times)
    details_complete = (
        spatial_total == len(spatial_times)
        and mamba_total == len(mamba_times)
        and bool(mamba_result.get("complete", True))
    )
    compared = mamba_result.get("readout_count", result.usable_map_count)
    try:
        compared_count = max(0, int(compared))
    except (TypeError, ValueError, OverflowError):
        compared_count = result.usable_map_count
    payload: dict[str, Any] = {
        "both_candidate_count": len(both),
        "spatial_only_count": len(spatial_only),
        "mamba_only_count": len(mamba_only),
        "compared_readout_count": compared_count,
        "candidate_set_overlap_fraction": (
            None if compared_count == 0 else (1.0 if not union else len(both) / len(union))
        ),
        "mamba_candidate_times": sorted(value for value in mamba_times if value),
        "complete": details_complete,
        "counts_are_lower_bounds": not details_complete,
        "unscored_mamba_readout_count": int(
            mamba_result.get("unscored_readout_count", 0) or 0
        ),
        "summary": "Candidate sets were compared over the exact persisted Mamba time window.",
    }
    if compared_count > 0 and details_complete:
        payload["agreement_fraction"] = payload["candidate_set_overlap_fraction"]
    elif not details_complete:
        payload["candidate_set_overlap_fraction"] = None
    return payload


def _scan_payload(
    run_id: str,
    model: Mapping[str, Any],
    from_cursor: int,
    to_cursor: int,
    result: SpatialScanResult,
    comparison: Mapping[str, Any],
    completed_at: datetime,
) -> dict[str, Any]:
    matched = set(comparison.get("mamba_candidate_times", ()))
    candidates: list[dict[str, Any]] = []
    for raw in result.candidates:
        candidate = dict(raw)
        observed = _canonical_observed_text(candidate.get("observed_at"))
        candidate["mamba_candidate"] = observed in matched
        candidate["mamba_assessment"] = (
            "Also flagged" if candidate["mamba_candidate"] else "Spatial only"
        )
        candidates.append(candidate)
    public_comparison = dict(comparison)
    public_comparison.pop("mamba_candidate_times", None)
    return {
        "scan_run_id": run_id,
        "model_version_id": model["model_version_id"],
        "first_observed_at": (
            None
            if result.first_observed_at is None
            else _datetime_text(result.first_observed_at)
        ),
        "last_observed_at": (
            None
            if result.last_observed_at is None
            else _datetime_text(result.last_observed_at)
        ),
        "map_count": result.map_count,
        "usable_map_count": result.usable_map_count,
        "candidate_map_count": (
            len(result.candidates)
            if result.candidate_map_count is None
            else result.candidate_map_count
        ),
        "max_anomaly_score": result.max_anomaly_score,
        "threshold": model["threshold"],
        "completed_at": _datetime_text(completed_at),
        "cursor_start": from_cursor,
        "cursor_end": to_cursor,
        "candidates": candidates,
        "comparison": public_comparison,
    }


def _canonical_observed_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _datetime_text(_parse_datetime(value))
    except (TypeError, ValueError):
        return None


def _validate_training_result(result: SpatialTrainingResult) -> None:
    if not isinstance(result, SpatialTrainingResult):
        raise TypeError("spatial backend returned an invalid training result")
    if not isinstance(result.model_content, bytes) or not result.model_content:
        raise SpatialMonitorUnavailable("Spatial backend returned an empty model artifact")
    _digest_text(result.training_data_hash, "training_data_hash")
    start = _aware_utc(result.training_start_at, "training_start_at")
    end = _aware_utc(result.training_end_at, "training_end_at")
    if end < start:
        raise SpatialMonitorUnavailable("Spatial training range is reversed")
    for name in (
        "training_map_count",
        "training_window_count",
        "validation_window_count",
        "calibration_window_count",
        "test_window_count",
        "completed_epochs",
    ):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SpatialMonitorUnavailable(f"Spatial {name} is invalid")
    if result.completed_epochs < 1:
        raise SpatialMonitorUnavailable("Spatial training completed no epochs")
    for name in (
        "validation_mae_tecu",
        "test_mae_tecu",
        "residual_center_tecu",
        "residual_scale_tecu",
        "threshold",
    ):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SpatialMonitorUnavailable(f"Spatial {name} is invalid")
    if result.validation_mae_tecu < 0 or result.test_mae_tecu < 0:
        raise SpatialMonitorUnavailable("Spatial evaluation MAE cannot be negative")
    if result.residual_center_tecu < 0 or result.residual_scale_tecu <= 0 or result.threshold <= 0:
        raise SpatialMonitorUnavailable("Spatial calibration values are invalid")


def _validate_scan_result(result: SpatialScanResult) -> None:
    if not isinstance(result, SpatialScanResult):
        raise TypeError("spatial backend returned an invalid scan result")
    for name in ("map_count", "usable_map_count"):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SpatialMonitorUnavailable(f"Spatial scan {name} is invalid")
    if result.usable_map_count > result.map_count:
        raise SpatialMonitorUnavailable("Spatial usable map count exceeds map count")
    if (result.first_observed_at is None) != (result.last_observed_at is None):
        raise SpatialMonitorUnavailable("Spatial scan time range is incomplete")
    if result.first_observed_at is not None and result.last_observed_at is not None:
        first = _aware_utc(result.first_observed_at, "scan first_observed_at")
        last = _aware_utc(result.last_observed_at, "scan last_observed_at")
        if last < first:
            raise SpatialMonitorUnavailable("Spatial scan time range is reversed")
    if result.max_anomaly_score is not None and (
        isinstance(result.max_anomaly_score, bool)
        or not isinstance(result.max_anomaly_score, (int, float))
        or not math.isfinite(float(result.max_anomaly_score))
        or result.max_anomaly_score < 0
    ):
        raise SpatialMonitorUnavailable("Spatial maximum anomaly score is invalid")
    if not isinstance(result.candidates, tuple) or len(result.candidates) > MAX_CANDIDATES_IN_STATUS:
        raise SpatialMonitorUnavailable("Spatial candidate details are invalid or unbounded")
    total = len(result.candidates) if result.candidate_map_count is None else result.candidate_map_count
    if isinstance(total, bool) or not isinstance(total, int) or total < len(result.candidates):
        raise SpatialMonitorUnavailable("Spatial candidate count is invalid")
    if total > result.usable_map_count:
        raise SpatialMonitorUnavailable("Spatial candidate count exceeds usable maps")
    try:
        _canonical_json_text(list(result.candidates))
    except (TypeError, ValueError) as error:
        raise SpatialMonitorUnavailable("Spatial candidates are not finite JSON data") from error
def _job_id(kind: str, pipeline_id: str, now: datetime) -> str:
    entropy = secrets.token_hex(8)
    digest = sha256(
        f"{kind}|{pipeline_id}|{_datetime_text(now)}|{entropy}".encode("utf-8")
    ).hexdigest()
    return f"spatial-{kind}-{digest[:24]}"


def _run_id(pipeline_id: str, now: datetime) -> str:
    return _job_id("run", pipeline_id, now)


def _canonical_json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_hash(payload: Any) -> str:
    return sha256(_canonical_json_text(payload).encode("utf-8")).hexdigest()


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _datetime_text(value: datetime) -> str:
    return _aware_utc(value, "datetime").isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("datetime must be text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _aware_utc(parsed, "datetime")


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _digest_text(value: Any, name: str) -> str:
    candidate = _nonempty_text(value, name)
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return candidate


def _positive_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("value must be a positive integer")
    return value


def _nonnegative_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("value must be a non-negative integer")
    return value


__all__ = [
    "DEFAULT_EPOCHS",
    "DEFAULT_HISTORY_YEARS",
    "SPATIAL_ALGORITHM",
    "SPATIAL_MONITOR_SCHEMA_VERSION",
    "SpatialBackend",
    "SpatialMonitor",
    "SpatialMonitorConflict",
    "SpatialMonitorError",
    "SpatialMonitorUnavailable",
    "SpatialScanResult",
    "SpatialTrainingResult",
    "TorchSpatialBackend",
]
