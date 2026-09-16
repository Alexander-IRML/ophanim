"""Resumable historical training and incremental Mamba anomaly monitoring.

The monitor is deliberately separate from the interactive forecast slice.  A
twenty-year backfill is a long-running data-engineering job, not an HTTP
request, and putting every historical TEC cell into the desktop SQLite file
would be wasteful.  This module therefore owns a compact readout ledger and
stores the original compressed IONEX artifacts as the reproducibility boundary.

The model consumes native-grid regional mean/median readouts at a two-hour
cadence.  The 0.5-degree display grid is not used as training evidence because
its extra cells are interpolated estimates rather than new measurements.
"""

from __future__ import annotations

import fcntl
import json
import math
import queue
import secrets
import sqlite3
import statistics
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

from ophanim.artifacts import FilesystemArtifactStore
from ophanim.domain import SourceArtifact, TECObservation
from ophanim.gim import (
    GIMAcquisitionError,
    GIMEdition,
    GIMFetchRequest,
    GIMNotFound,
    CodeGIMSource,
)
from ophanim.ionex import IONEXV1Parser
from ophanim.tec_archive import BulkTECArchive, standard_gim_grid_definition


MAMBA_MONITOR_SCHEMA_VERSION = 2
MAMBA_ALGORITHM = "ophanim-mamba-selective-ssm/1"
FEATURE_ALGORITHM = "regional-mean-median-2h/1"
DEFAULT_HISTORY_YEARS = 20
DEFAULT_THRESHOLD = 4.5
DEFAULT_REVISION_OVERLAP_DAYS = 30
DEFAULT_MINIMUM_TRAINING_READOUTS = 365 * 12
DEFAULT_ACQUISITION_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_MAXIMUM_BASELINE_GAP_DAYS = 31
MINIMUM_DAILY_READOUTS = 6
EARLIEST_CODE_DATE = date(2002, 1, 1)
MAX_CANDIDATES_IN_STATUS = 50


class MambaMonitorError(RuntimeError):
    """Base class for expected monitor lifecycle errors."""


class MambaMonitorConflict(MambaMonitorError):
    """The requested operation conflicts with a running operation."""


class MambaMonitorUnavailable(MambaMonitorError):
    """The monitor cannot perform the requested operation yet."""


class _MonitorInterrupted(Exception):
    pass


class _ReadoutDayUnavailable(Exception):
    pass


@dataclass(frozen=True, slots=True)
class MonitorRegion:
    """Versioned rectangular region used to derive the model sequence."""

    name: str = "Central Texas"
    south: float = 29.0
    west: float = -100.5
    north: float = 32.5
    east: float = -96.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("region name must not be empty")
        values = (self.south, self.west, self.north, self.east)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("region bounds must be finite numbers")
        if not -90.0 <= self.south < self.north <= 90.0:
            raise ValueError("region latitude bounds must satisfy -90 <= south < north <= 90")
        if not -180.0 <= self.west < self.east < 180.0:
            raise ValueError(
                "Mamba v1 region longitude bounds must satisfy "
                "-180 <= west < east < 180"
            )

    def contains(self, latitude: float, longitude: float) -> bool:
        return self.south <= latitude <= self.north and self.west <= longitude <= self.east

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "south": float(self.south),
            "west": float(self.west),
            "north": float(self.north),
            "east": float(self.east),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "MonitorRegion":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("region must be an object")
        name = payload.get("name", "Central Texas")
        if not isinstance(name, str):
            raise ValueError("region name must be text")
        try:
            return cls(
                name=name.strip(),
                south=_finite_float(payload.get("south", 29.0), "region south"),
                west=_finite_float(payload.get("west", -100.5), "region west"),
                north=_finite_float(payload.get("north", 32.5), "region north"),
                east=_finite_float(payload.get("east", -96.0), "region east"),
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError("region bounds must be numbers") from error


@dataclass(frozen=True, slots=True)
class AcquiredReadoutDay:
    """One source artifact and the compact readouts derived from it."""

    product_date: date
    source_uri: str
    resolved_uri: str
    filename: str
    content: bytes
    revision: str
    revision_priority: int
    observations: tuple[TECObservation, ...]
    readouts: tuple["RegionalReadout", ...]


@dataclass(frozen=True, slots=True)
class RegionalReadout:
    """Compact native-grid regional statistic retained by the monitor."""

    observed_at: datetime
    mean_vtec_tecu: float
    median_vtec_tecu: float
    coverage_fraction: float
    cell_count: int


@dataclass(frozen=True, slots=True)
class NativeGridDay:
    """One immutable native GIM selected for spatial model consumption."""

    source_day_id: int
    product_date: date
    grid_set_id: str
    checksum_sha256: str
    revision_priority: int


@dataclass(frozen=True, slots=True)
class NativeGridSnapshot:
    """Frozen native-grid source set shared with spatial baselines."""

    pipeline_id: str
    model_version_id: str
    sync_through_date: date
    source_cursor: int
    days: tuple[NativeGridDay, ...]


class ReadoutProvider(Protocol):
    def acquire(self, product_date: date, region: MonitorRegion) -> AcquiredReadoutDay: ...


class CodeRegionalReadoutProvider:
    """Stream one CODE final GIM into native regional two-hour readouts."""

    def __init__(
        self,
        *,
        source: CodeGIMSource | None = None,
        clock: Callable[[], datetime] | None = None,
        retain_observations: bool = False,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._source = source or CodeGIMSource(clock=self._clock)
        self._parser = IONEXV1Parser()
        self._retain_observations = retain_observations

    def acquire(self, product_date: date, region: MonitorRegion) -> AcquiredReadoutDay:
        fetched = self._source.fetch(
            GIMFetchRequest(edition=GIMEdition.FINAL, product_date=product_date)
        )
        # Imported lazily so the monitor and the normal ingestion path share the
        # exact same bounded gzip / UNIX-compress implementation.
        from ophanim.ingestion import decompress_ionex

        content = decompress_ionex(fetched.loaded_source.content)
        checksum = sha256(fetched.loaded_source.content).hexdigest()
        artifact = SourceArtifact(
            artifact_id=f"mamba-source-{checksum}",
            provider="code",
            product="gim",
            parser_version=self._parser.parser_version,
            revision_priority=fetched.ingestion_request.revision_priority,
            checksum_sha256=checksum,
            storage_ref="pending",
            ingested_at=_aware_utc(self._clock(), "Mamba source clock"),
            revision=fetched.ingestion_request.revision,
            source_uri=fetched.source_uri,
        )
        parsed = self._parser.parse(artifact=artifact, stream=BytesIO(content))
        observations, readouts = _regional_readouts(
            parsed,
            product_date=product_date,
            region=region,
            retain_observations=self._retain_observations,
        )
        return AcquiredReadoutDay(
            product_date=product_date,
            source_uri=fetched.source_uri,
            resolved_uri=fetched.resolved_uri,
            filename=fetched.loaded_source.filename,
            content=fetched.loaded_source.content,
            revision=fetched.ingestion_request.revision or "final",
            revision_priority=fetched.ingestion_request.revision_priority,
            observations=observations,
            readouts=readouts,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS monitor_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL
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
    feature_algorithm TEXT NOT NULL,
    model_algorithm TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('uninitialized', 'initializing', 'ready', 'scanning', 'failed')
    ),
    active_model_version_id TEXT,
    training_start_at TEXT,
    training_end_at TEXT,
    training_readout_count INTEGER,
    training_snapshot_cursor INTEGER NOT NULL DEFAULT 0,
    scan_cursor INTEGER NOT NULL DEFAULT 0,
    sync_through_date TEXT,
    last_successful_scan_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    kind TEXT NOT NULL CHECK (kind IN ('initialize', 'scan')),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'interrupted')
    ),
    phase TEXT NOT NULL,
    requested_start_date TEXT NOT NULL,
    requested_end_date TEXT NOT NULL,
    completed_units INTEGER NOT NULL DEFAULT 0,
    total_units INTEGER NOT NULL,
    message TEXT,
    error_message TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_idx
    ON jobs((1)) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS jobs_pipeline_created_idx
    ON jobs(pipeline_id, created_at DESC);

CREATE TABLE IF NOT EXISTS job_days (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    product_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('fetching', 'published', 'reused', 'missing', 'failed')
    ),
    attempt_count INTEGER NOT NULL,
    error_message TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, product_date)
);

CREATE TABLE IF NOT EXISTS source_days (
    source_day_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_date TEXT NOT NULL,
    edition TEXT NOT NULL,
    revision_priority INTEGER NOT NULL,
    source_uri TEXT NOT NULL,
    resolved_uri TEXT NOT NULL,
    filename TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    storage_ref TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    native_grid_set_id TEXT,
    UNIQUE (product_date, edition, checksum_sha256)
);

CREATE INDEX IF NOT EXISTS source_days_date_idx
    ON source_days(product_date, revision_priority DESC, source_day_id DESC);

CREATE TABLE IF NOT EXISTS readouts (
    readout_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    source_day_id INTEGER NOT NULL REFERENCES source_days(source_day_id),
    observed_at TEXT NOT NULL,
    mean_vtec_tecu REAL NOT NULL,
    median_vtec_tecu REAL NOT NULL,
    cell_count INTEGER NOT NULL,
    expected_cell_count INTEGER NOT NULL,
    coverage_fraction REAL NOT NULL,
    UNIQUE (pipeline_id, source_day_id, observed_at)
);

CREATE INDEX IF NOT EXISTS readouts_pipeline_time_idx
    ON readouts(pipeline_id, observed_at, readout_id);

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
    training_readout_count INTEGER NOT NULL,
    regularized_sample_count INTEGER NOT NULL,
    missing_slot_count INTEGER NOT NULL,
    maximum_missing_gap_hours REAL NOT NULL,
    threshold REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    scan_run_id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(job_id),
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id),
    model_version_id TEXT NOT NULL REFERENCES models(model_version_id),
    from_cursor_exclusive INTEGER NOT NULL,
    to_cursor_inclusive INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    first_observed_at TEXT,
    last_observed_at TEXT,
    readout_count INTEGER NOT NULL DEFAULT 0,
    candidate_event_count INTEGER NOT NULL DEFAULT 0,
    max_anomaly_score REAL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS scan_scores (
    scan_run_id TEXT NOT NULL REFERENCES scan_runs(scan_run_id) ON DELETE CASCADE,
    readout_id INTEGER NOT NULL REFERENCES readouts(readout_id),
    observed_at TEXT NOT NULL,
    predicted_mean_vtec_tecu REAL NOT NULL,
    observed_mean_vtec_tecu REAL NOT NULL,
    mean_residual_tecu REAL NOT NULL,
    mean_anomaly_score REAL NOT NULL,
    predicted_median_vtec_tecu REAL NOT NULL,
    observed_median_vtec_tecu REAL NOT NULL,
    median_residual_tecu REAL NOT NULL,
    median_anomaly_score REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    threshold REAL NOT NULL,
    is_candidate INTEGER NOT NULL CHECK (is_candidate IN (0, 1)),
    PRIMARY KEY (scan_run_id, readout_id)
);
"""


class MambaMonitor:
    """Own background initialization, incremental scans, and durable status."""

    def __init__(
        self,
        *,
        data_directory: str | Path,
        clock: Callable[[], datetime] | None = None,
        gim_source: CodeGIMSource | None = None,
        readout_provider: ReadoutProvider | None = None,
        bulk_archive: BulkTECArchive | None = None,
        background: bool = True,
        minimum_training_readouts: int = DEFAULT_MINIMUM_TRAINING_READOUTS,
        revision_overlap_days: int = DEFAULT_REVISION_OVERLAP_DAYS,
        acquisition_attempts: int = DEFAULT_ACQUISITION_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        maximum_baseline_gap_days: int = DEFAULT_MAXIMUM_BASELINE_GAP_DAYS,
    ) -> None:
        if minimum_training_readouts < 8:
            raise ValueError("minimum_training_readouts must be at least 8")
        if revision_overlap_days < 1:
            raise ValueError("revision_overlap_days must be positive")
        if (
            not isinstance(acquisition_attempts, int)
            or isinstance(acquisition_attempts, bool)
            or acquisition_attempts < 1
            or acquisition_attempts > 10
        ):
            raise ValueError("acquisition_attempts must be an integer in [1, 10]")
        if (
            isinstance(retry_backoff_seconds, bool)
            or not isinstance(retry_backoff_seconds, (int, float))
            or not math.isfinite(float(retry_backoff_seconds))
            or retry_backoff_seconds < 0.0
            or retry_backoff_seconds > 60.0
        ):
            raise ValueError("retry_backoff_seconds must be in [0, 60]")
        if (
            not isinstance(maximum_baseline_gap_days, int)
            or isinstance(maximum_baseline_gap_days, bool)
            or maximum_baseline_gap_days < 1
        ):
            raise ValueError("maximum_baseline_gap_days must be a positive integer")
        self._directory = Path(data_directory).expanduser().resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._process_lock_guard = threading.Lock()
        self._process_lock_file: Any | None = None
        self._acquire_process_lock()
        self._database = self._directory / "mamba-monitor.sqlite3"
        self._source_store = FilesystemArtifactStore(
            self._directory / "mamba-source-artifacts"
        )
        self._model_store = FilesystemArtifactStore(
            self._directory / "mamba-model-artifacts"
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._bulk_archive = bulk_archive
        self._minimum_training_readouts = minimum_training_readouts
        self._revision_overlap_days = revision_overlap_days
        self._acquisition_attempts = acquisition_attempts
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        self._maximum_baseline_gap_days = maximum_baseline_gap_days
        self._provider = readout_provider or CodeRegionalReadoutProvider(
            source=gim_source,
            clock=self._clock,
            retain_observations=bulk_archive is not None,
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        try:
            self._initialize_store()
            resumable = self._recover_jobs()
        except BaseException:
            self._release_process_lock()
            raise
        if background:
            self._thread = threading.Thread(
                target=self._worker,
                name="ophanim-mamba-monitor",
                daemon=True,
            )
            self._thread.start()
            for job_id in resumable:
                self._queue.put(job_id)

    def status(self) -> dict[str, Any]:
        """Return one bounded status document safe for frequent UI polling."""

        with self._connection() as connection:
            pipeline = self._active_pipeline(connection)
            if pipeline is None:
                return {
                    "ok": True,
                    "available": True,
                    "state": "uninitialized",
                    "model": None,
                    "active_job": None,
                    "scan": _empty_scan_payload(),
                    "storage": _storage_payload(self._bulk_archive is not None),
                }
            active_job = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
            model = None
            if pipeline["active_model_version_id"] is not None:
                row = connection.execute(
                    "SELECT * FROM models WHERE model_version_id = ?",
                    (pipeline["active_model_version_id"],),
                ).fetchone()
                if row is not None:
                    model = _model_payload(row)
            last_run = connection.execute(
                """
                SELECT * FROM scan_runs
                WHERE pipeline_id = ? AND status = 'succeeded'
                ORDER BY completed_at DESC, rowid DESC LIMIT 1
                """,
                (pipeline["pipeline_id"],),
            ).fetchone()
            last_result = (
                None
                if last_run is None
                else self._scan_result_payload(connection, last_run)
            )
            pending = connection.execute(
                """
                SELECT COUNT(*) AS count, MIN(observed_at) AS first,
                       MAX(observed_at) AS last, MAX(readout_id) AS latest
                FROM readouts
                WHERE pipeline_id = ? AND readout_id > ?
                """,
                (pipeline["pipeline_id"], pipeline["scan_cursor"]),
            ).fetchone()
            state = pipeline["status"]
            if active_job is not None:
                state = "initializing" if active_job["kind"] == "initialize" else "scanning"
            return {
                "ok": True,
                "available": True,
                "state": state,
                "pipeline_id": pipeline["pipeline_id"],
                "region": {
                    "name": pipeline["region_name"],
                    "south": pipeline["south"],
                    "west": pipeline["west"],
                    "north": pipeline["north"],
                    "east": pipeline["east"],
                },
                "history_years": pipeline["history_years"],
                "model": model,
                "active_job": None if active_job is None else _job_payload(active_job),
                "scan": {
                    "last_successful_at": pipeline["last_successful_scan_at"],
                    "last_successful_cursor": pipeline["scan_cursor"],
                    "latest_available_cursor": pending["latest"] or pipeline["scan_cursor"],
                    "pending_readout_count": pending["count"],
                    "pending_observed_start": pending["first"],
                    "pending_observed_end": pending["last"],
                    "last_result": last_result,
                },
                "error": pipeline["error_message"],
                "storage": _storage_payload(self._bulk_archive is not None),
            }

    def initialize(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Queue or resume historical acquisition and model fitting."""

        values = payload or {}
        years = values.get("history_years", DEFAULT_HISTORY_YEARS)
        if isinstance(years, bool) or not isinstance(years, int):
            raise ValueError("history_years must be an integer")
        if not 1 <= years <= 30:
            raise ValueError("history_years must be between 1 and 30")
        region = MonitorRegion.from_payload(values.get("region"))
        now = _aware_utc(self._clock(), "Mamba monitor clock")
        end = now.date() - timedelta(days=1)
        start = max(EARLIEST_CODE_DATE, _subtract_years(end, years) + timedelta(days=1))
        config = {
            "history_years": years,
            "region": region.to_dict(),
            "edition": "final",
            "cadence_hours": 2,
            "feature_algorithm": FEATURE_ALGORITHM,
            "model_algorithm": MAMBA_ALGORITHM,
            "threshold": DEFAULT_THRESHOLD,
        }
        configuration_hash = _json_hash(config)
        pipeline_id = f"mamba-{configuration_hash[:24]}"
        now_text = _datetime_text(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
            if active is not None:
                if active["pipeline_id"] != pipeline_id or active["kind"] != "initialize":
                    connection.rollback()
                    raise MambaMonitorConflict(
                        "Another Mamba operation is already running"
                    )
                connection.commit()
                return self.status()
            existing = connection.execute(
                "SELECT * FROM pipelines WHERE pipeline_id = ?", (pipeline_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO pipelines (
                        pipeline_id, configuration_hash, history_years,
                        region_name, south, west, north, east,
                        feature_algorithm, model_algorithm, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'uninitialized', ?, ?)
                    """,
                    (
                        pipeline_id,
                        configuration_hash,
                        years,
                        region.name,
                        region.south,
                        region.west,
                        region.north,
                        region.east,
                        FEATURE_ALGORITHM,
                        MAMBA_ALGORITHM,
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
            if existing["status"] == "ready" and existing["active_model_version_id"]:
                connection.commit()
                return self.status()
            original_range = connection.execute(
                """
                SELECT requested_start_date, requested_end_date
                FROM jobs
                WHERE pipeline_id = ? AND kind = 'initialize'
                ORDER BY created_at, rowid LIMIT 1
                """,
                (pipeline_id,),
            ).fetchone()
            if original_range is not None:
                # A retry is a continuation of the original experiment.  Freeze
                # its historical window instead of silently moving both ends as
                # wall-clock days pass while a long download is interrupted.
                start = date.fromisoformat(original_range["requested_start_date"])
                end = date.fromisoformat(original_range["requested_end_date"])
            job_id = _job_id("initialize", pipeline_id, now)
            total_days = (end - start).days + 1
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, pipeline_id, kind, status, phase,
                    requested_start_date, requested_end_date,
                    completed_units, total_units, message,
                    created_at, updated_at
                ) VALUES (?, ?, 'initialize', 'queued', 'planning', ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    pipeline_id,
                    start.isoformat(),
                    end.isoformat(),
                    total_days,
                    f"Preparing {total_days:,} complete UTC days",
                    now_text,
                    now_text,
                ),
            )
            connection.execute(
                """
                UPDATE pipelines
                SET status = 'initializing', error_message = NULL, updated_at = ?
                WHERE pipeline_id = ?
                """,
                (now_text, pipeline_id),
            )
            connection.commit()
        self._queue.put(job_id)
        return self.status()

    def scan(self) -> dict[str, Any]:
        """Queue a scan through the latest complete UTC day."""

        now = _aware_utc(self._clock(), "Mamba monitor clock")
        end = now.date() - timedelta(days=1)
        now_text = _datetime_text(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pipeline = self._active_pipeline(connection)
            if pipeline is None or pipeline["active_model_version_id"] is None:
                connection.rollback()
                raise MambaMonitorUnavailable(
                    "Initialize the Mamba baseline before checking new readouts"
                )
            active = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
            if active is not None:
                if active["kind"] != "scan":
                    connection.rollback()
                    raise MambaMonitorConflict(
                        "The historical Mamba initialization is still running"
                    )
                connection.commit()
                return self.status()
            synced = (
                date.fromisoformat(pipeline["sync_through_date"])
                if pipeline["sync_through_date"]
                else date.fromisoformat(pipeline["training_end_at"][:10])
            )
            start = max(EARLIEST_CODE_DATE, synced - timedelta(days=self._revision_overlap_days - 1))
            if start > end:
                start = end
            job_id = _job_id("scan", pipeline["pipeline_id"], now)
            total_days = (end - start).days + 1
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, pipeline_id, kind, status, phase,
                    requested_start_date, requested_end_date,
                    completed_units, total_units, message,
                    created_at, updated_at
                ) VALUES (?, ?, 'scan', 'queued', 'planning', ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    pipeline["pipeline_id"],
                    start.isoformat(),
                    end.isoformat(),
                    total_days,
                    "Snapshotting new complete daily maps",
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

    def native_grid_snapshot(self) -> NativeGridSnapshot:
        """Freeze the latest published native grid for every available UTC day.

        The spatial baseline deliberately reuses the Mamba acquisition ledger:
        downloading a second twenty-year copy would weaken provenance and waste
        both bandwidth and disk.  All returned Zarr groups are immutable, so a
        background model job can safely keep using this snapshot while a later
        Mamba scan publishes newer revisions.
        """

        if self._bulk_archive is None:
            raise MambaMonitorUnavailable(
                "Spatial modeling requires the Docker PostgreSQL/Zarr data plane"
            )
        with self._connection() as connection:
            pipeline = self._active_pipeline(connection)
            if pipeline is None or pipeline["active_model_version_id"] is None:
                raise MambaMonitorUnavailable(
                    "Initialize the Mamba history before training the spatial baseline"
                )
            active = connection.execute(
                "SELECT kind FROM jobs WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
            if active is not None and active["kind"] == "initialize":
                raise MambaMonitorUnavailable(
                    "Wait for historical acquisition to finish before training the spatial baseline"
                )
            sync_through = (
                date.fromisoformat(pipeline["sync_through_date"])
                if pipeline["sync_through_date"]
                else date.fromisoformat(pipeline["training_end_at"][:10])
            )
            rows = connection.execute(
                """
                SELECT candidate.* FROM source_days candidate
                WHERE candidate.edition = 'final'
                  AND candidate.native_grid_set_id IS NOT NULL
                  AND candidate.product_date <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM source_days newer
                      WHERE newer.product_date = candidate.product_date
                        AND newer.edition = candidate.edition
                        AND newer.native_grid_set_id IS NOT NULL
                        AND (
                            newer.revision_priority > candidate.revision_priority
                            OR (
                                newer.revision_priority = candidate.revision_priority
                                AND newer.source_day_id > candidate.source_day_id
                            )
                        )
                )
                ORDER BY candidate.product_date, candidate.source_day_id
                """,
                (sync_through.isoformat(),),
            ).fetchall()
        days = tuple(
            NativeGridDay(
                source_day_id=int(row["source_day_id"]),
                product_date=date.fromisoformat(row["product_date"]),
                grid_set_id=row["native_grid_set_id"],
                checksum_sha256=row["checksum_sha256"],
                revision_priority=int(row["revision_priority"]),
            )
            for row in rows
        )
        if not days:
            raise MambaMonitorUnavailable(
                "No archived native grids are available for spatial modeling"
            )
        return NativeGridSnapshot(
            pipeline_id=pipeline["pipeline_id"],
            model_version_id=pipeline["active_model_version_id"],
            sync_through_date=sync_through,
            source_cursor=max(item.source_day_id for item in days),
            days=days,
        )

    def comparison_window(
        self,
        *,
        first_observed_at: datetime,
        last_observed_at: datetime,
        expected_region: MonitorRegion,
        maximum_source_cursor: int,
        expected_pipeline_id: str,
        expected_model_version_id: str,
    ) -> dict[str, Any]:
        """Return persisted current-revision scores for one exact time window.

        Spatial scans may lag behind an independently run Mamba scan.  Reading
        only ``status().scan.last_result`` would then compare different time
        windows.  This query instead resolves the latest readout revision at
        every requested timestamp and its most recent successful score under
        the active Mamba model.
        """

        first = _aware_utc(first_observed_at, "comparison first_observed_at")
        last = _aware_utc(last_observed_at, "comparison last_observed_at")
        if last < first:
            raise ValueError("comparison time range must not be reversed")
        if not isinstance(expected_region, MonitorRegion):
            raise TypeError("expected_region must be a MonitorRegion")
        if (
            isinstance(maximum_source_cursor, bool)
            or not isinstance(maximum_source_cursor, int)
            or maximum_source_cursor < 1
        ):
            raise ValueError("maximum_source_cursor must be a positive integer")
        if not isinstance(expected_pipeline_id, str) or not expected_pipeline_id:
            raise ValueError("expected_pipeline_id must be non-empty text")
        if (
            not isinstance(expected_model_version_id, str)
            or not expected_model_version_id
        ):
            raise ValueError("expected_model_version_id must be non-empty text")
        with self._connection() as connection:
            pipeline = self._active_pipeline(connection)
            if pipeline is None or pipeline["active_model_version_id"] is None:
                raise MambaMonitorUnavailable(
                    "Initialize the Mamba baseline before comparing spatial candidates"
                )
            if (
                pipeline["pipeline_id"] != expected_pipeline_id
                or pipeline["active_model_version_id"]
                != expected_model_version_id
            ):
                raise MambaMonitorUnavailable(
                    "The active Mamba model changed; retrain the spatial baseline"
                )
            actual_region = _region_from_row(pipeline)
            if (
                actual_region.south,
                actual_region.west,
                actual_region.north,
                actual_region.east,
            ) != (
                expected_region.south,
                expected_region.west,
                expected_region.north,
                expected_region.east,
            ):
                raise MambaMonitorUnavailable(
                    "Spatial and Mamba comparisons require identical region bounds"
                )
            rows = connection.execute(
                """
                WITH latest AS (
                    SELECT readouts.observed_at,
                           MAX(readouts.readout_id) AS readout_id
                    FROM readouts
                    JOIN source_days USING (source_day_id)
                    WHERE readouts.pipeline_id = ?
                      AND source_days.source_day_id <= ?
                      AND readouts.observed_at >= ?
                      AND readouts.observed_at <= ?
                    GROUP BY readouts.observed_at
                ), ranked_scores AS (
                    SELECT scores.readout_id, scores.observed_at,
                           scores.anomaly_score, scores.threshold,
                           scores.is_candidate,
                           ROW_NUMBER() OVER (
                               PARTITION BY scores.readout_id
                               ORDER BY runs.completed_at DESC, runs.rowid DESC
                           ) AS score_rank
                    FROM scan_scores scores
                    JOIN scan_runs runs USING (scan_run_id)
                    WHERE runs.pipeline_id = ?
                      AND runs.model_version_id = ?
                      AND runs.status = 'succeeded'
                )
                SELECT readouts.observed_at, ranked_scores.anomaly_score,
                       ranked_scores.threshold, ranked_scores.is_candidate
                FROM latest
                JOIN readouts ON readouts.readout_id = latest.readout_id
                LEFT JOIN ranked_scores
                  ON ranked_scores.readout_id = latest.readout_id
                 AND ranked_scores.score_rank = 1
                ORDER BY readouts.observed_at
                """,
                (
                    pipeline["pipeline_id"],
                    maximum_source_cursor,
                    _datetime_text(first),
                    _datetime_text(last),
                    pipeline["pipeline_id"],
                    pipeline["active_model_version_id"],
                ),
            ).fetchall()
        scored = [row for row in rows if row["is_candidate"] is not None]
        candidates = [
            {
                "observed_at": row["observed_at"],
                "anomaly_score": row["anomaly_score"],
                "threshold": row["threshold"],
            }
            for row in scored
            if bool(row["is_candidate"])
        ]
        return {
            "pipeline_id": pipeline["pipeline_id"],
            "model_version_id": pipeline["active_model_version_id"],
            "maximum_source_cursor": maximum_source_cursor,
            "region": actual_region.to_dict(),
            "first_observed_at": _datetime_text(first),
            "last_observed_at": _datetime_text(last),
            "readout_count": len(scored),
            "unscored_readout_count": len(rows) - len(scored),
            "candidate_readout_count": len(candidates),
            "candidates": candidates,
            "complete": len(rows) == len(scored),
        }

    def run_next_job(self) -> bool:
        """Run one queued job synchronously (primarily useful to tests/CLI)."""

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
        """Cooperatively stop the worker; durable checkpoints remain resumable."""

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
        now = _aware_utc(self._clock(), "Mamba monitor clock")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None or job["status"] not in {"queued", "interrupted"}:
                connection.rollback()
                return
            connection.execute(
                """
                UPDATE jobs SET status = 'running', phase = 'discovering',
                    started_at = COALESCE(started_at, ?), updated_at = ?,
                    completed_at = NULL, error_message = NULL, retryable = 0
                WHERE job_id = ?
                """,
                (_datetime_text(now), _datetime_text(now), job_id),
            )
            connection.commit()
        try:
            if job["kind"] == "initialize":
                self._run_initialize(job_id)
            else:
                self._run_scan(job_id)
        except _MonitorInterrupted:
            self._mark_interrupted(job_id)
        except Exception as error:  # background boundary: persist, never kill server
            self._mark_failed(job_id, error)
            if not isinstance(error, (GIMAcquisitionError, MambaMonitorError)):
                traceback.print_exc()

    def _run_initialize(self, job_id: str) -> None:
        job, pipeline = self._job_and_pipeline(job_id)
        region = _region_from_row(pipeline)
        start = date.fromisoformat(job["requested_start_date"])
        end = date.fromisoformat(job["requested_end_date"])
        self._process_days(job_id, pipeline["pipeline_id"], region, start, end, refresh=False)
        self._assert_running()
        with self._connection() as connection:
            coverage = connection.execute(
                """
                SELECT COUNT(*) AS considered,
                       SUM(CASE WHEN status IN ('published', 'reused') THEN 1 ELSE 0 END)
                           AS available
                FROM job_days WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        available_days = int(coverage["available"] or 0)
        total_days = int(coverage["considered"] or 0)
        minimum_days = max(1, math.ceil(total_days * 0.70))
        if available_days < minimum_days:
            raise MambaMonitorUnavailable(
                f"Only {available_days:,} of {total_days:,} requested historical "
                f"days were available; at least {minimum_days:,} are required"
            )
        self._update_job(job_id, phase="training", message="Fitting the CPU selective state-space model")
        rows = self._latest_readout_rows(
            pipeline["pipeline_id"],
            observed_on_or_after=start,
            observed_on_or_before=end,
        )
        if len(rows) < self._minimum_training_readouts:
            raise MambaMonitorUnavailable(
                f"Historical acquisition produced {len(rows):,} usable readouts; "
                f"at least {self._minimum_training_readouts:,} are required"
            )
        from ophanim.mamba_model import TrainingSample, fit_model

        actual_samples = [
            TrainingSample(
                observed_at=_parse_datetime(row["observed_at"]),
                mean_vtec_tecu=row["mean_vtec_tecu"],
                median_vtec_tecu=row["median_vtec_tecu"],
                coverage_fraction=row["coverage_fraction"],
            )
            for row in rows
        ]
        maximum_missing_gap_hours = _maximum_missing_gap_hours(
            actual_samples,
            start=start,
            end=end,
        )
        maximum_allowed_gap_hours = self._maximum_baseline_gap_days * 24
        if maximum_missing_gap_hours > maximum_allowed_gap_hours:
            raise MambaMonitorUnavailable(
                "Historical readouts contain a missing span of "
                f"{maximum_missing_gap_hours:g} hours; the baseline limit is "
                f"{maximum_allowed_gap_hours:g} hours"
            )
        samples = _regularize_samples(actual_samples)
        model = fit_model(samples, threshold=DEFAULT_THRESHOLD)
        model_bytes = model.to_bytes()
        stored = self._model_store.put(model_bytes, suffix=".json")
        created_at = _aware_utc(self._clock(), "Mamba monitor clock")
        first = actual_samples[0].observed_at
        last = actual_samples[-1].observed_at
        snapshot_cursor = max(row["readout_id"] for row in rows)
        model_config_hash = _json_hash(
            {
                "pipeline_configuration_hash": pipeline["configuration_hash"],
                "algorithm": MAMBA_ALGORITHM,
                "threshold": DEFAULT_THRESHOLD,
            }
        )
        model_identity_hash = sha256(
            (
                stored.checksum_sha256
                + "\0"
                + model_config_hash
            ).encode("ascii")
        ).hexdigest()
        model_version_id = f"model-mamba-{model_identity_hash[:24]}"
        training_hash = _readout_manifest_hash(rows)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO models (
                    model_version_id, pipeline_id, algorithm,
                    configuration_hash, artifact_ref,
                    artifact_checksum_sha256, training_data_hash,
                    training_start_at, training_end_at,
                    training_readout_count, regularized_sample_count,
                    missing_slot_count, maximum_missing_gap_hours,
                    threshold, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_version_id,
                    pipeline["pipeline_id"],
                    MAMBA_ALGORITHM,
                    model_config_hash,
                    stored.storage_ref,
                    stored.checksum_sha256,
                    training_hash,
                    _datetime_text(first),
                    _datetime_text(last),
                    len(actual_samples),
                    len(samples),
                    len(samples) - len(actual_samples),
                    maximum_missing_gap_hours,
                    DEFAULT_THRESHOLD,
                    _datetime_text(created_at),
                ),
            )
            connection.execute(
                """
                UPDATE pipelines SET
                    status = 'ready', active_model_version_id = ?,
                    training_start_at = ?, training_end_at = ?,
                    training_readout_count = ?, training_snapshot_cursor = ?,
                    scan_cursor = ?, sync_through_date = ?, error_message = NULL,
                    updated_at = ?
                WHERE pipeline_id = ?
                """,
                (
                    model_version_id,
                    _datetime_text(first),
                    _datetime_text(last),
                    len(actual_samples),
                    snapshot_cursor,
                    snapshot_cursor,
                    end.isoformat(),
                    _datetime_text(created_at),
                    pipeline["pipeline_id"],
                ),
            )
            self._finish_job(connection, job_id, created_at, message="Historical baseline is ready")
            connection.commit()

    def _run_scan(self, job_id: str) -> None:
        job, pipeline = self._job_and_pipeline(job_id)
        region = _region_from_row(pipeline)
        start = date.fromisoformat(job["requested_start_date"])
        end = date.fromisoformat(job["requested_end_date"])
        self._process_days(job_id, pipeline["pipeline_id"], region, start, end, refresh=True)
        self._assert_running()
        self._update_job(job_id, phase="scoring", message="Scoring unseen regional readouts")
        with self._connection() as connection:
            pipeline = connection.execute(
                "SELECT * FROM pipelines WHERE pipeline_id = ?",
                (pipeline["pipeline_id"],),
            ).fetchone()
            snapshot_row = connection.execute(
                "SELECT COALESCE(MAX(readout_id), 0) AS cursor FROM readouts WHERE pipeline_id = ?",
                (pipeline["pipeline_id"],),
            ).fetchone()
            snapshot = snapshot_row["cursor"]
            target_rows = connection.execute(
                """
                SELECT r.* FROM readouts r
                JOIN (
                    SELECT observed_at, MAX(readout_id) AS latest_id
                    FROM readouts
                    WHERE pipeline_id = ? AND readout_id <= ?
                    GROUP BY observed_at
                ) latest ON latest.latest_id = r.readout_id
                WHERE r.readout_id > ?
                ORDER BY r.observed_at, r.readout_id
                """,
                (pipeline["pipeline_id"], snapshot, pipeline["scan_cursor"]),
            ).fetchall()
            model_row = connection.execute(
                "SELECT * FROM models WHERE model_version_id = ?",
                (pipeline["active_model_version_id"],),
            ).fetchone()
        from ophanim.mamba_model import MambaModel, TrainingSample

        model = MambaModel.from_bytes(self._model_store.read(model_row["artifact_ref"]))
        score_results: list[Any] = []
        scorable_rows = [
            row
            for row in target_rows
            if _parse_datetime(row["observed_at"]) > model.training_end_at
        ]
        if scorable_rows:
            last_target = _parse_datetime(scorable_rows[-1]["observed_at"])
            replay_rows = self._post_training_rows(
                pipeline["pipeline_id"],
                after=model.training_end_at,
                through=last_target,
                snapshot=snapshot,
            )
            replay_actual = [_training_sample(row, TrainingSample) for row in replay_rows]
            replay = _regularize_samples(
                replay_actual,
                anchor=model.anchor_sample,
            )
            replay_results = model.score_sequence(history=(), targets=replay)
            by_time = {result.observed_at: result for result in replay_results}
            score_results = [
                by_time[_parse_datetime(row["observed_at"])] for row in scorable_rows
            ]
        completed_at = _aware_utc(self._clock(), "Mamba monitor clock")
        scan_run_id = _job_id("run", pipeline["pipeline_id"], completed_at)
        max_score = max((float(result.anomaly_score) for result in score_results), default=None)
        candidate_count = sum(bool(result.is_candidate) for result in score_results)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO scan_runs (
                    scan_run_id, job_id, pipeline_id, model_version_id,
                    from_cursor_exclusive, to_cursor_inclusive, status,
                    first_observed_at, last_observed_at, readout_count,
                    candidate_event_count, max_anomaly_score,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'succeeded', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_run_id,
                    job_id,
                    pipeline["pipeline_id"],
                    pipeline["active_model_version_id"],
                    pipeline["scan_cursor"],
                    snapshot,
                    None if not scorable_rows else scorable_rows[0]["observed_at"],
                    None if not scorable_rows else scorable_rows[-1]["observed_at"],
                    len(scorable_rows),
                    candidate_count,
                    max_score,
                    _datetime_text(completed_at),
                    _datetime_text(completed_at),
                ),
            )
            for row, result in zip(scorable_rows, score_results, strict=True):
                connection.execute(
                    """
                    INSERT INTO scan_scores (
                        scan_run_id, readout_id, observed_at,
                        predicted_mean_vtec_tecu, observed_mean_vtec_tecu,
                        mean_residual_tecu, mean_anomaly_score,
                        predicted_median_vtec_tecu, observed_median_vtec_tecu,
                        median_residual_tecu, median_anomaly_score,
                        anomaly_score, threshold, is_candidate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_run_id,
                        row["readout_id"],
                        row["observed_at"],
                        result.predicted_mean_vtec_tecu,
                        result.observed_mean_vtec_tecu,
                        result.mean_residual_tecu,
                        result.mean_anomaly_score,
                        result.predicted_median_vtec_tecu,
                        result.observed_median_vtec_tecu,
                        result.median_residual_tecu,
                        result.median_anomaly_score,
                        result.anomaly_score,
                        result.threshold,
                        int(result.is_candidate),
                    ),
                )
            connection.execute(
                """
                UPDATE pipelines SET status = 'ready', scan_cursor = ?,
                    sync_through_date = ?, last_successful_scan_at = ?,
                    error_message = NULL, updated_at = ?
                WHERE pipeline_id = ?
                """,
                (
                    snapshot,
                    end.isoformat(),
                    _datetime_text(completed_at),
                    _datetime_text(completed_at),
                    pipeline["pipeline_id"],
                ),
            )
            self._finish_job(
                connection,
                job_id,
                completed_at,
                message=(
                    f"Checked {len(scorable_rows):,} new readouts; "
                    f"{candidate_count:,} need review"
                ),
            )
            connection.commit()

    def _process_days(
        self,
        job_id: str,
        pipeline_id: str,
        region: MonitorRegion,
        start: date,
        end: date,
        *,
        refresh: bool,
    ) -> None:
        total = (end - start).days + 1
        for index in range(total):
            self._assert_running()
            product_date = start + timedelta(days=index)
            self._update_job(
                job_id,
                phase="downloading",
                completed=index,
                message=f"Acquiring CODE final map for {product_date.isoformat()}",
            )
            for attempt in range(1, self._acquisition_attempts + 1):
                try:
                    reused = self._ensure_day(
                        pipeline_id,
                        product_date,
                        region,
                        refresh=refresh,
                    )
                except GIMNotFound as error:
                    self._record_job_day(job_id, product_date, "missing", str(error))
                    break
                except _ReadoutDayUnavailable as error:
                    self._record_job_day(job_id, product_date, "missing", str(error))
                    break
                except GIMAcquisitionError as error:
                    self._record_job_day(job_id, product_date, "failed", str(error))
                    if attempt >= self._acquisition_attempts:
                        raise
                    delay = self._retry_backoff_seconds * (2 ** (attempt - 1))
                    self._update_job(
                        job_id,
                        phase="downloading",
                        completed=index,
                        message=(
                            f"Retrying {product_date.isoformat()} after attempt "
                            f"{attempt} of {self._acquisition_attempts}"
                        ),
                    )
                    if self._stop.wait(delay):
                        raise _MonitorInterrupted
                    continue
                except Exception as error:
                    self._record_job_day(job_id, product_date, "failed", str(error))
                    raise
                self._record_job_day(
                    job_id,
                    product_date,
                    "reused" if reused else "published",
                    None,
                )
                break
        self._update_job(
            job_id,
            phase="archiving",
            completed=total,
            message="Daily checkpoints are complete",
        )

    def _ensure_day(
        self,
        pipeline_id: str,
        product_date: date,
        region: MonitorRegion,
        *,
        refresh: bool,
    ) -> bool:
        with self._connection() as connection:
            prior = connection.execute(
                """
                SELECT * FROM source_days
                WHERE product_date = ? AND edition = 'final'
                ORDER BY revision_priority DESC, source_day_id DESC LIMIT 1
                """,
                (product_date.isoformat(),),
            ).fetchone()
            if prior is not None:
                readout_count = connection.execute(
                    "SELECT COUNT(*) FROM readouts WHERE pipeline_id = ? AND source_day_id = ?",
                    (pipeline_id, prior["source_day_id"]),
                ).fetchone()[0]
                needs_native_archive = (
                    self._bulk_archive is not None
                    and prior["native_grid_set_id"] is None
                )
                if readout_count and not refresh and not needs_native_archive:
                    return True
        acquired = self._provider.acquire(product_date, region)
        self._assert_running()
        checksum = sha256(acquired.content).hexdigest()
        same_source = prior is not None and prior["checksum_sha256"] == checksum
        if same_source:
            needs_native_archive = (
                self._bulk_archive is not None
                and prior["native_grid_set_id"] is None
            )
            if readout_count and not needs_native_archive:
                return True

        # Establish immutable source provenance before attempting the optional
        # Zarr/PostgreSQL publication.  If publication is interrupted after the
        # Zarr group is written, the next attempt reconstructs the exact same
        # SourceArtifact instead of assigning a new ingestion timestamp and
        # conflicting with that already-written immutable group.
        if not same_source:
            stored = self._source_store.put(
                acquired.content,
                suffix="".join(Path(acquired.filename).suffixes[-2:]),
            )
            ingested_at = _aware_utc(self._clock(), "Mamba monitor clock")
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO source_days (
                        product_date, edition, revision_priority, source_uri,
                        resolved_uri, filename, checksum_sha256, storage_ref,
                        artifact_id, ingested_at, native_grid_set_id
                    ) VALUES (?, 'final', ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        product_date.isoformat(),
                        acquired.revision_priority,
                        acquired.source_uri,
                        acquired.resolved_uri,
                        acquired.filename,
                        checksum,
                        stored.storage_ref,
                        f"mamba-source-{checksum}",
                        _datetime_text(ingested_at),
                    ),
                )
                connection.commit()

        with self._connection() as connection:
            source_row = connection.execute(
                """
                SELECT * FROM source_days
                WHERE product_date = ? AND edition = 'final' AND checksum_sha256 = ?
                """,
                (product_date.isoformat(), checksum),
            ).fetchone()
        if source_row is None:  # Defensive: the insert/select pair is invariant.
            raise RuntimeError("Mamba source provenance was not persisted")

        artifact = SourceArtifact(
            artifact_id=source_row["artifact_id"],
            provider="code",
            product="gim",
            parser_version=IONEXV1Parser.parser_version,
            revision_priority=source_row["revision_priority"],
            checksum_sha256=source_row["checksum_sha256"],
            storage_ref=source_row["storage_ref"],
            ingested_at=_parse_datetime(source_row["ingested_at"]),
            revision="final",
            source_uri=source_row["source_uri"],
        )
        native_grid_set_id = source_row["native_grid_set_id"]
        if self._bulk_archive is not None and native_grid_set_id is None:
            archived = self._bulk_archive.archive(
                artifact=artifact,
                observations=acquired.observations,
                grid=standard_gim_grid_definition(),
                include_core=False,
            )
            native_grid_set_id = archived.manifest.grid_set_id
            with self._connection() as connection:
                connection.execute(
                    """
                    UPDATE source_days SET native_grid_set_id = ?
                    WHERE source_day_id = ? AND native_grid_set_id IS NULL
                    """,
                    (native_grid_set_id, source_row["source_day_id"]),
                )
                connection.commit()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expected = _expected_region_cell_count(region)
            for readout in acquired.readouts:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO readouts (
                        pipeline_id, source_day_id, observed_at,
                        mean_vtec_tecu, median_vtec_tecu, cell_count,
                        expected_cell_count, coverage_fraction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pipeline_id,
                        source_row["source_day_id"],
                        _datetime_text(readout.observed_at),
                        readout.mean_vtec_tecu,
                        readout.median_vtec_tecu,
                        readout.cell_count,
                        expected,
                        readout.coverage_fraction,
                    ),
                )
            connection.commit()
        return same_source

    def _latest_readout_rows(
        self,
        pipeline_id: str,
        *,
        observed_on_or_after: date,
        observed_on_or_before: date,
    ) -> list[sqlite3.Row]:
        start = datetime.combine(
            observed_on_or_after, datetime.min.time(), tzinfo=UTC
        )
        cutoff = datetime.combine(
            observed_on_or_before + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        with self._connection() as connection:
            return list(
                connection.execute(
                    """
                    SELECT r.* FROM readouts r
                    JOIN (
                        SELECT observed_at, MAX(readout_id) AS latest_id
                        FROM readouts
                        WHERE pipeline_id = ? AND observed_at >= ? AND observed_at < ?
                        GROUP BY observed_at
                    ) latest ON latest.latest_id = r.readout_id
                    ORDER BY r.observed_at, r.readout_id
                    """,
                    (
                        pipeline_id,
                        _datetime_text(start),
                        _datetime_text(cutoff),
                    ),
                ).fetchall()
            )

    def _history_rows(
        self, pipeline_id: str, *, before: datetime, limit: int
    ) -> list[sqlite3.Row]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT r.* FROM readouts r
                JOIN (
                    SELECT observed_at, MAX(readout_id) AS latest_id
                    FROM readouts
                    WHERE pipeline_id = ? AND observed_at < ?
                    GROUP BY observed_at
                ) latest ON latest.latest_id = r.readout_id
                ORDER BY r.observed_at DESC LIMIT ?
                """,
                (pipeline_id, _datetime_text(before), limit),
            ).fetchall()
        return list(reversed(rows))

    def _post_training_rows(
        self,
        pipeline_id: str,
        *,
        after: datetime,
        through: datetime,
        snapshot: int,
    ) -> list[sqlite3.Row]:
        with self._connection() as connection:
            return list(
                connection.execute(
                    """
                    SELECT r.* FROM readouts r
                    JOIN (
                        SELECT observed_at, MAX(readout_id) AS latest_id
                        FROM readouts
                        WHERE pipeline_id = ? AND readout_id <= ?
                          AND observed_at > ? AND observed_at <= ?
                        GROUP BY observed_at
                    ) latest ON latest.latest_id = r.readout_id
                    ORDER BY r.observed_at, r.readout_id
                    """,
                    (
                        pipeline_id,
                        snapshot,
                        _datetime_text(after),
                        _datetime_text(through),
                    ),
                ).fetchall()
            )

    def _scan_result_payload(
        self, connection: sqlite3.Connection, run: sqlite3.Row
    ) -> dict[str, Any]:
        pipeline = connection.execute(
            "SELECT region_name FROM pipelines WHERE pipeline_id = ?",
            (run["pipeline_id"],),
        ).fetchone()
        requested_day_count = 0
        available_day_count = 0
        if run["job_id"] is not None:
            job = connection.execute(
                "SELECT total_units FROM jobs WHERE job_id = ?",
                (run["job_id"],),
            ).fetchone()
            if job is not None:
                requested_day_count = int(job["total_units"])
            available = connection.execute(
                """
                SELECT COUNT(*) AS count FROM job_days
                WHERE job_id = ? AND status IN ('published', 'reused')
                """,
                (run["job_id"],),
            ).fetchone()
            available_day_count = int(available["count"])
        missing_day_count = max(0, requested_day_count - available_day_count)
        scores = connection.execute(
            """
            SELECT scan_scores.*, readouts.cell_count
            FROM scan_scores JOIN readouts USING (readout_id)
            WHERE scan_run_id = ? AND is_candidate = 1
            ORDER BY anomaly_score DESC, observed_at DESC
            LIMIT ?
            """,
            (run["scan_run_id"], MAX_CANDIDATES_IN_STATUS),
        ).fetchall()
        return {
            "scan_run_id": run["scan_run_id"],
            "model_version_id": run["model_version_id"],
            "first_observed_at": run["first_observed_at"],
            "last_observed_at": run["last_observed_at"],
            "readout_count": run["readout_count"],
            "usable_readout_count": run["readout_count"],
            "candidate_readout_count": run["candidate_event_count"],
            # Kept during the v2 API transition for older desktop clients.
            "candidate_event_count": run["candidate_event_count"],
            "requested_day_count": requested_day_count,
            "available_day_count": available_day_count,
            "missing_day_count": missing_day_count,
            "max_anomaly_score": run["max_anomaly_score"],
            "completed_at": run["completed_at"],
            "cursor_start": run["from_cursor_exclusive"],
            "cursor_end": run["to_cursor_inclusive"],
            "candidates": [
                {
                    "observed_at": row["observed_at"],
                    "anomaly_score": row["anomaly_score"],
                    "threshold": row["threshold"],
                    "mean_residual_tecu": row["mean_residual_tecu"],
                    "median_residual_tecu": row["median_residual_tecu"],
                    "residual_tecu": (
                        row["mean_residual_tecu"]
                        if abs(row["mean_residual_tecu"])
                        >= abs(row["median_residual_tecu"])
                        else row["median_residual_tecu"]
                    ),
                    "predicted_mean_vtec_tecu": row["predicted_mean_vtec_tecu"],
                    "observed_mean_vtec_tecu": row["observed_mean_vtec_tecu"],
                    "predicted_median_vtec_tecu": row["predicted_median_vtec_tecu"],
                    "observed_median_vtec_tecu": row["observed_median_vtec_tecu"],
                    "cell_count": row["cell_count"],
                    "contributing_cell_count": row["cell_count"],
                    "region_name": (
                        "Regional mean / median"
                        if pipeline is None
                        else pipeline["region_name"]
                    ),
                    "assessment": "candidate",
                }
                for row in scores
            ],
        }

    def _initialize_store(self) -> None:
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT schema_version FROM monitor_schema WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO monitor_schema (singleton, schema_version) VALUES (1, ?)",
                    (MAMBA_MONITOR_SCHEMA_VERSION,),
                )
            elif row["schema_version"] == 1:
                model_columns = {
                    item["name"]
                    for item in connection.execute("PRAGMA table_info(models)")
                }
                for name, declaration in (
                    ("regularized_sample_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("missing_slot_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("maximum_missing_gap_hours", "REAL NOT NULL DEFAULT 0"),
                ):
                    if name not in model_columns:
                        connection.execute(
                            f"ALTER TABLE models ADD COLUMN {name} {declaration}"
                        )
                scan_columns = {
                    item["name"]
                    for item in connection.execute("PRAGMA table_info(scan_runs)")
                }
                if "job_id" not in scan_columns:
                    connection.execute(
                        "ALTER TABLE scan_runs ADD COLUMN job_id TEXT REFERENCES jobs(job_id)"
                    )
                connection.execute(
                    "UPDATE monitor_schema SET schema_version = ? WHERE singleton = 1",
                    (MAMBA_MONITOR_SCHEMA_VERSION,),
                )
            elif row["schema_version"] != MAMBA_MONITOR_SCHEMA_VERSION:
                raise RuntimeError(
                    "unsupported Mamba monitor schema version "
                    f"{row['schema_version']}"
                )
            connection.commit()

    def _acquire_process_lock(self) -> None:
        lock_file = (self._directory / "mamba-monitor.lock").open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise MambaMonitorUnavailable(
                "The Mamba monitor is already active in another OPHANIM process"
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

    def _recover_jobs(self) -> tuple[str, ...]:
        now = _datetime_text(_aware_utc(self._clock(), "Mamba monitor clock"))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE jobs SET status = 'queued', phase = 'resuming',
                    message = 'Resuming from durable daily checkpoints',
                    updated_at = ?, retryable = 1
                WHERE status IN ('running', 'interrupted')
                """,
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
                raise RuntimeError("Mamba job disappeared")
            pipeline = connection.execute(
                "SELECT * FROM pipelines WHERE pipeline_id = ?",
                (job["pipeline_id"],),
            ).fetchone()
            if pipeline is None:
                raise RuntimeError("Mamba pipeline disappeared")
            return job, pipeline

    def _update_job(
        self,
        job_id: str,
        *,
        phase: str,
        message: str,
        completed: int | None = None,
    ) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "Mamba monitor clock"))
        with self._connection() as connection:
            if completed is None:
                connection.execute(
                    "UPDATE jobs SET phase = ?, message = ?, updated_at = ? WHERE job_id = ?",
                    (phase, message, now, job_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs SET phase = ?, message = ?, completed_units = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (phase, message, completed, now, job_id),
                )
            connection.commit()

    def _record_job_day(
        self, job_id: str, product_date: date, status: str, error: str | None
    ) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "Mamba monitor clock"))
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO job_days (
                    job_id, product_date, status, attempt_count,
                    error_message, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(job_id, product_date) DO UPDATE SET
                    status = excluded.status,
                    attempt_count = job_days.attempt_count + 1,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (job_id, product_date.isoformat(), status, error, now),
            )
            connection.commit()

    def _mark_interrupted(self, job_id: str) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "Mamba monitor clock"))
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'interrupted', phase = 'interrupted',
                    message = 'Stopped safely; launch again to resume', retryable = 1,
                    updated_at = ?, completed_at = ? WHERE job_id = ?
                """,
                (now, now, job_id),
            )
            connection.commit()

    def _mark_failed(self, job_id: str, error: Exception) -> None:
        now = _datetime_text(_aware_utc(self._clock(), "Mamba monitor clock"))
        message = str(error).strip() or error.__class__.__name__
        retryable = isinstance(error, (GIMAcquisitionError, MambaMonitorUnavailable))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT pipeline_id FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            connection.execute(
                """
                UPDATE jobs SET status = 'failed', phase = 'failed',
                    error_message = ?, message = ?, retryable = ?,
                    updated_at = ?, completed_at = ? WHERE job_id = ?
                """,
                (message, message, int(retryable), now, now, job_id),
            )
            connection.execute(
                """
                UPDATE pipelines SET status = 'failed', error_message = ?, updated_at = ?
                WHERE pipeline_id = ?
                """,
                (message, now, row["pipeline_id"]),
            )
            connection.commit()

    @staticmethod
    def _finish_job(
        connection: sqlite3.Connection,
        job_id: str,
        completed_at: datetime,
        *,
        message: str,
    ) -> None:
        timestamp = _datetime_text(completed_at)
        connection.execute(
            """
            UPDATE jobs SET status = 'succeeded', phase = 'complete',
                completed_units = total_units, message = ?, error_message = NULL,
                retryable = 0, updated_at = ?, completed_at = ?
            WHERE job_id = ?
            """,
            (message, timestamp, timestamp, job_id),
        )

    def _assert_running(self) -> None:
        if self._stop.is_set():
            raise _MonitorInterrupted

    def _active_pipeline(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        selected = connection.execute(
            "SELECT value FROM settings WHERE key = 'active_pipeline_id'"
        ).fetchone()
        if selected is None:
            return None
        return connection.execute(
            "SELECT * FROM pipelines WHERE pipeline_id = ?", (selected["value"],)
        ).fetchone()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _regional_readouts(
    observations: Iterable[TECObservation],
    *,
    product_date: date,
    region: MonitorRegion,
    retain_observations: bool,
) -> tuple[tuple[TECObservation, ...], tuple[RegionalReadout, ...]]:
    start = datetime.combine(product_date, datetime.min.time(), tzinfo=UTC)
    end = start + timedelta(days=1)
    by_epoch: dict[datetime, list[float]] = defaultdict(list)
    global_counts: dict[datetime, int] = defaultdict(int)
    retained: list[TECObservation] = []
    for observation in observations:
        if retain_observations:
            retained.append(observation)
        observed_at = observation.observed_at.astimezone(UTC)
        if not start <= observed_at < end:
            continue
        if observed_at.minute or observed_at.second or observed_at.microsecond:
            continue
        if observed_at.hour % 2:
            continue
        global_counts[observed_at] += 1
        if region.contains(observation.latitude_degrees, observation.longitude_degrees):
            by_epoch[observed_at].append(observation.vtec_tecu)
    if not global_counts:
        raise ValueError(f"CODE GIM for {product_date.isoformat()} has no owned 2-hour epochs")
    # The parser omits official 9999 missing cells.  Reject grossly incomplete
    # products while still allowing explicit source missingness to flow into the
    # regional coverage feature.
    minimum_global = min(global_counts.values())
    if minimum_global < int(5_112 * 0.95):
        raise ValueError(
            f"CODE GIM for {product_date.isoformat()} is incomplete: "
            f"only {minimum_global:,} native cells in an epoch"
        )
    expected = _expected_region_cell_count(region)
    if expected == 0:
        raise ValueError("region contains no native 5 by 2.5 degree GIM nodes")
    readouts = []
    for observed_at in sorted(by_epoch):
        values = by_epoch[observed_at]
        if not values:
            continue
        readouts.append(
            RegionalReadout(
                observed_at=observed_at,
                mean_vtec_tecu=sum(values) / len(values),
                median_vtec_tecu=statistics.median(values),
                coverage_fraction=min(1.0, len(values) / expected),
                cell_count=len(values),
            )
        )
    if not readouts:
        raise _ReadoutDayUnavailable(
            f"CODE GIM for {product_date.isoformat()} has no usable regional readouts"
        )
    if len(readouts) < MINIMUM_DAILY_READOUTS:
        raise _ReadoutDayUnavailable(
            f"CODE GIM for {product_date.isoformat()} has only "
            f"{len(readouts)} usable regional readouts; at least "
            f"{MINIMUM_DAILY_READOUTS} are required"
        )
    return tuple(retained), tuple(readouts)


def _expected_region_cell_count(region: MonitorRegion) -> int:
    return sum(
        1
        for latitude_index in range(71)
        for longitude_index in range(72)
        if region.contains(
            latitude_index * 2.5 - 87.5,
            longitude_index * 5.0 - 180.0,
        )
    )


def _training_sample(row: sqlite3.Row, sample_type: Any) -> Any:
    return sample_type(
        observed_at=_parse_datetime(row["observed_at"]),
        mean_vtec_tecu=row["mean_vtec_tecu"],
        median_vtec_tecu=row["median_vtec_tecu"],
        coverage_fraction=row["coverage_fraction"],
    )


def _regularize_samples(
    samples: Iterable[Any],
    *,
    anchor: Any | None = None,
) -> list[Any]:
    """Represent missing two-hour slots explicitly using coverage near zero.

    The value carried through a missing slot is never presented as an observed
    readout.  It exists only so the reference recurrence sees a regular clock;
    its near-zero coverage feature marks the placeholder for the model.
    """

    from ophanim.mamba_model import TrainingSample

    values = list(samples)
    if not values:
        return []
    result: list[TrainingSample] = []
    previous = anchor
    for sample in values:
        if previous is not None:
            cursor = previous.observed_at + timedelta(hours=2)
            if sample.observed_at < cursor:
                raise ValueError("regional readouts must be strictly chronological")
            while cursor < sample.observed_at:
                result.append(
                    TrainingSample(
                        observed_at=cursor,
                        mean_vtec_tecu=previous.mean_vtec_tecu,
                        median_vtec_tecu=previous.median_vtec_tecu,
                        coverage_fraction=1e-6,
                    )
                )
                cursor += timedelta(hours=2)
        result.append(sample)
        previous = sample
    return result


def _maximum_missing_gap_hours(
    samples: Iterable[Any],
    *,
    start: date,
    end: date,
) -> float:
    values = list(samples)
    if not values:
        return float("inf")
    requested_start = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    requested_end = datetime.combine(end, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=22
    )
    gaps = [
        max(0.0, (values[0].observed_at - requested_start).total_seconds() / 3600.0),
        max(0.0, (requested_end - values[-1].observed_at).total_seconds() / 3600.0),
    ]
    gaps.extend(
        max(
            0.0,
            (current.observed_at - previous.observed_at).total_seconds() / 3600.0
            - 2.0,
        )
        for previous, current in zip(values, values[1:])
    )
    return max(gaps)


def _region_from_row(row: sqlite3.Row) -> MonitorRegion:
    return MonitorRegion(
        name=row["region_name"],
        south=row["south"],
        west=row["west"],
        north=row["north"],
        east=row["east"],
    )


def _model_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "model_version_id": row["model_version_id"],
        "algorithm": row["algorithm"],
        "trained_at": row["created_at"],
        "training_start": row["training_start_at"],
        "training_end": row["training_end_at"],
        "training_readout_count": row["training_readout_count"],
        "regularized_sample_count": row["regularized_sample_count"],
        "missing_slot_count": row["missing_slot_count"],
        "maximum_missing_gap_hours": row["maximum_missing_gap_hours"],
        "artifact_checksum_sha256": row["artifact_checksum_sha256"],
        "training_data_hash": row["training_data_hash"],
        "threshold": row["threshold"],
    }


def _job_payload(row: sqlite3.Row) -> dict[str, Any]:
    total = row["total_units"]
    completed = row["completed_units"]
    return {
        "job_id": row["job_id"],
        "kind": row["kind"],
        "status": row["status"],
        "phase": row["phase"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "completed_units": completed,
        "total_units": total,
        "progress_fraction": 0.0 if not total else min(1.0, completed / total),
        "message": row["message"],
        "retryable": bool(row["retryable"]),
        "error": row["error_message"],
        "requested_start_date": row["requested_start_date"],
        "requested_end_date": row["requested_end_date"],
    }


def _empty_scan_payload() -> dict[str, Any]:
    return {
        "last_successful_at": None,
        "last_successful_cursor": None,
        "latest_available_cursor": None,
        "pending_readout_count": 0,
        "pending_observed_start": None,
        "pending_observed_end": None,
        "last_result": None,
    }


def _storage_payload(has_native_zarr: bool) -> dict[str, Any]:
    return {
        "source_artifacts": "content-addressed compressed IONEX",
        "feature_store": "compact SQLite regional readouts",
        "native_zarr_enabled": has_native_zarr,
        "derived_half_degree_history": False,
    }


def _readout_manifest_hash(rows: Iterable[sqlite3.Row]) -> str:
    digest = sha256()
    for row in rows:
        payload = [
            row["readout_id"],
            row["source_day_id"],
            row["observed_at"],
            row["mean_vtec_tecu"],
            row["median_vtec_tecu"],
            row["coverage_fraction"],
        ]
        digest.update(
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _job_id(kind: str, pipeline_id: str, now: datetime) -> str:
    entropy = secrets.token_hex(8)
    digest = sha256(
        f"{kind}|{pipeline_id}|{_datetime_text(now)}|{entropy}".encode("utf-8")
    ).hexdigest()
    return f"mamba-{kind}-{digest[:24]}"


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:  # February 29 into a non-leap year
        return value.replace(year=value.year - years, day=28)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_hash(payload: Any) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _datetime_text(value: datetime) -> str:
    return _aware_utc(value, "datetime").isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


__all__ = [
    "AcquiredReadoutDay",
    "CodeRegionalReadoutProvider",
    "DEFAULT_HISTORY_YEARS",
    "FEATURE_ALGORITHM",
    "MAMBA_ALGORITHM",
    "MambaMonitor",
    "MambaMonitorConflict",
    "MambaMonitorError",
    "MambaMonitorUnavailable",
    "MonitorRegion",
    "RegionalReadout",
    "ReadoutProvider",
]
