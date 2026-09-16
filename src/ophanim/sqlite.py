"""SQLite-backed durable state for the OPHANIM application.

The database is deliberately one transaction boundary even though callers see
small, purpose-specific repositories.  Domain records remain ordinary frozen
dataclasses; this module owns their relational representation.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
from types import TracebackType
from typing import Self

from ophanim.domain import (
    AnomalyDecision,
    DetectorVersion,
    DisturbanceAssessment,
    Forecast,
    ForecastMode,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    SourceArtifact,
    TargetMetric,
    TECObservation,
)


SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_artifacts (
    artifact_id TEXT PRIMARY KEY CHECK (length(trim(artifact_id)) > 0),
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    product TEXT NOT NULL CHECK (length(trim(product)) > 0),
    parser_version TEXT NOT NULL CHECK (length(trim(parser_version)) > 0),
    revision_priority INTEGER NOT NULL CHECK (revision_priority >= 0),
    checksum_sha256 TEXT NOT NULL CHECK (length(trim(checksum_sha256)) > 0),
    storage_ref TEXT NOT NULL CHECK (length(trim(storage_ref)) > 0),
    ingested_at TEXT NOT NULL,
    revision TEXT CHECK (revision IS NULL OR length(trim(revision)) > 0),
    source_uri TEXT CHECK (source_uri IS NULL OR length(trim(source_uri)) > 0)
);

CREATE INDEX IF NOT EXISTS source_artifacts_checksum_idx
    ON source_artifacts(checksum_sha256);

CREATE UNIQUE INDEX IF NOT EXISTS source_artifacts_acquisition_identity_idx
    ON source_artifacts(
        provider,
        product,
        parser_version,
        revision_priority,
        IFNULL(revision, ''),
        IFNULL(source_uri, ''),
        checksum_sha256
    );

CREATE TABLE IF NOT EXISTS processing_versions (
    processing_version_id TEXT PRIMARY KEY,
    code_revision TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL CHECK (length(trim(parser_version)) > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS region_versions (
    region_version_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    boundary_ref TEXT NOT NULL,
    boundary_checksum_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    artifact_ref TEXT,
    artifact_checksum_sha256 TEXT,
    training_data_version TEXT,
    CHECK (
        (artifact_ref IS NULL AND artifact_checksum_sha256 IS NULL)
        OR (artifact_ref IS NOT NULL AND artifact_checksum_sha256 IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS detector_versions (
    detector_version_id TEXT PRIMARY KEY,
    scoring_method TEXT NOT NULL,
    threshold REAL NOT NULL,
    configuration_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    calibration_data_version TEXT,
    calibration_residual_mean_tecu REAL,
    calibration_residual_standard_deviation_tecu REAL,
    calibration_sample_count INTEGER
        CHECK (calibration_sample_count IS NULL OR calibration_sample_count >= 0)
);

CREATE TABLE IF NOT EXISTS tec_observations (
    artifact_id TEXT NOT NULL REFERENCES source_artifacts(artifact_id),
    observed_at TEXT NOT NULL,
    latitude_degrees REAL NOT NULL,
    longitude_degrees REAL NOT NULL,
    vtec_tecu REAL NOT NULL,
    quality_flags_json TEXT NOT NULL,
    PRIMARY KEY (
        artifact_id,
        observed_at,
        latitude_degrees,
        longitude_degrees
    )
);

CREATE INDEX IF NOT EXISTS tec_observations_artifact_time_idx
    ON tec_observations(artifact_id, observed_at);

CREATE TABLE IF NOT EXISTS regional_observations (
    observation_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES source_artifacts(artifact_id),
    processing_version_id TEXT NOT NULL
        REFERENCES processing_versions(processing_version_id),
    region_version_id TEXT NOT NULL REFERENCES region_versions(region_version_id),
    observed_at TEXT NOT NULL,
    produced_at TEXT NOT NULL,
    mean_vtec_tecu REAL NOT NULL,
    median_vtec_tecu REAL NOT NULL,
    cell_count INTEGER NOT NULL CHECK (cell_count > 0),
    coverage_fraction REAL NOT NULL
        CHECK (coverage_fraction > 0.0 AND coverage_fraction <= 1.0),
    quality_flags_json TEXT NOT NULL,
    CHECK (produced_at >= observed_at),
    UNIQUE (
        artifact_id,
        processing_version_id,
        region_version_id,
        observed_at
    )
);

CREATE INDEX IF NOT EXISTS regional_observations_region_time_idx
    ON regional_observations(region_version_id, observed_at);

CREATE INDEX IF NOT EXISTS regional_observations_artifact_idx
    ON regional_observations(artifact_id, observed_at);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    origin_at TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    valid_at TEXT NOT NULL,
    source_provider TEXT NOT NULL CHECK (length(trim(source_provider)) > 0),
    source_product TEXT NOT NULL CHECK (length(trim(source_product)) > 0),
    region_version_id TEXT NOT NULL REFERENCES region_versions(region_version_id),
    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
    processing_version_id TEXT NOT NULL
        REFERENCES processing_versions(processing_version_id),
    target_metric TEXT NOT NULL
        CHECK (target_metric IN ('mean_vtec', 'median_vtec')),
    predicted_vtec_tecu REAL NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('live', 'hindcast')),
    input_manifest_hash TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'scored', 'insufficient_data', 'failed')),
    CHECK (valid_at > origin_at),
    CHECK (issued_at >= origin_at),
    CHECK (mode = 'hindcast' OR issued_at < valid_at),
    UNIQUE (
        origin_at,
        issued_at,
        valid_at,
        source_provider,
        source_product,
        region_version_id,
        model_version_id,
        processing_version_id,
        target_metric,
        mode,
        input_manifest_hash
    )
);

CREATE INDEX IF NOT EXISTS forecasts_pending_valid_at_idx
    ON forecasts(status, valid_at);

CREATE TABLE IF NOT EXISTS forecast_input_observations (
    forecast_id TEXT NOT NULL REFERENCES forecasts(forecast_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    observation_id TEXT NOT NULL REFERENCES regional_observations(observation_id),
    PRIMARY KEY (forecast_id, position),
    UNIQUE (forecast_id, observation_id)
);

CREATE TABLE IF NOT EXISTS anomaly_decisions (
    decision_id TEXT PRIMARY KEY,
    forecast_id TEXT NOT NULL REFERENCES forecasts(forecast_id),
    observation_id TEXT NOT NULL REFERENCES regional_observations(observation_id),
    detector_version_id TEXT NOT NULL REFERENCES detector_versions(detector_version_id),
    scored_at TEXT NOT NULL,
    residual_tecu REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    threshold REAL NOT NULL,
    is_forecast_anomaly INTEGER NOT NULL
        CHECK (is_forecast_anomaly IN (0, 1)),
    disturbance_assessment TEXT NOT NULL
        CHECK (disturbance_assessment IN ('normal', 'candidate', 'confirmed', 'unknown')),
    assessment_source TEXT NOT NULL,
    supersedes_decision_id TEXT REFERENCES anomaly_decisions(decision_id),
    CHECK (anomaly_score >= 0.0),
    CHECK (threshold > 0.0),
    CHECK (
        supersedes_decision_id IS NULL
        OR supersedes_decision_id != decision_id
    ),
    UNIQUE (forecast_id, observation_id, detector_version_id)
);

CREATE INDEX IF NOT EXISTS anomaly_decisions_forecast_scored_idx
    ON anomaly_decisions(forecast_id, scored_at);

CREATE UNIQUE INDEX IF NOT EXISTS anomaly_decisions_single_successor_idx
    ON anomaly_decisions(supersedes_decision_id)
    WHERE supersedes_decision_id IS NOT NULL;
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the current schema on an idle SQLite connection.

    Foreign-key enforcement is connection-local in SQLite, so it is enabled
    here as well as by :class:`SQLiteUnitOfWork`.
    """

    connection.execute("PRAGMA foreign_keys = ON")
    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if current_version not in {0, SCHEMA_VERSION}:
        raise RuntimeError(
            f"unsupported database schema version {current_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    connection.executescript(_SCHEMA)
    if current_version == 0:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _dump_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("persisted datetimes must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _load_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _dump_strings(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _load_strings(value: str) -> tuple[str, ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) for item in decoded
    ):
        raise ValueError("stored string tuple is malformed")
    return tuple(decoded)


def _input_manifest_hash(observation_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        observation_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _immutable_conflict(record_type: str, identity: str) -> None:
    raise sqlite3.IntegrityError(
        f"{record_type} {identity!r} already exists with different values"
    )


def _assert_exact(existing: object, incoming: object, record_type: str, identity: str) -> None:
    if existing != incoming:
        _immutable_conflict(record_type, identity)


def _observation_provenance(
    connection: sqlite3.Connection,
    observation_id: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT regional_observations.*,
               source_artifacts.provider AS source_provider,
               source_artifacts.product AS source_product,
               source_artifacts.parser_version AS artifact_parser_version,
               source_artifacts.ingested_at AS artifact_ingested_at
        FROM regional_observations
        JOIN source_artifacts USING (artifact_id)
        WHERE regional_observations.observation_id = ?
        """,
        (observation_id,),
    ).fetchone()


def _artifact_from_row(row: sqlite3.Row) -> SourceArtifact:
    return SourceArtifact(
        artifact_id=row["artifact_id"],
        provider=row["provider"],
        product=row["product"],
        parser_version=row["parser_version"],
        revision_priority=row["revision_priority"],
        checksum_sha256=row["checksum_sha256"],
        storage_ref=row["storage_ref"],
        ingested_at=_load_datetime(row["ingested_at"]),
        revision=row["revision"],
        source_uri=row["source_uri"],
    )


def _processing_version_from_row(row: sqlite3.Row) -> ProcessingVersion:
    return ProcessingVersion(
        processing_version_id=row["processing_version_id"],
        code_revision=row["code_revision"],
        configuration_hash=row["configuration_hash"],
        parser_version=row["parser_version"],
        created_at=_load_datetime(row["created_at"]),
    )


def _region_version_from_row(row: sqlite3.Row) -> RegionVersion:
    return RegionVersion(
        region_version_id=row["region_version_id"],
        name=row["name"],
        boundary_ref=row["boundary_ref"],
        boundary_checksum_sha256=row["boundary_checksum_sha256"],
        created_at=_load_datetime(row["created_at"]),
    )


def _tec_observation_from_row(row: sqlite3.Row) -> TECObservation:
    return TECObservation(
        artifact_id=row["artifact_id"],
        observed_at=_load_datetime(row["observed_at"]),
        latitude_degrees=row["latitude_degrees"],
        longitude_degrees=row["longitude_degrees"],
        vtec_tecu=row["vtec_tecu"],
        quality_flags=_load_strings(row["quality_flags_json"]),
    )


def _regional_observation_from_row(row: sqlite3.Row) -> RegionalObservation:
    return RegionalObservation(
        observation_id=row["observation_id"],
        artifact_id=row["artifact_id"],
        processing_version_id=row["processing_version_id"],
        region_version_id=row["region_version_id"],
        observed_at=_load_datetime(row["observed_at"]),
        produced_at=_load_datetime(row["produced_at"]),
        mean_vtec_tecu=row["mean_vtec_tecu"],
        median_vtec_tecu=row["median_vtec_tecu"],
        cell_count=row["cell_count"],
        coverage_fraction=row["coverage_fraction"],
        quality_flags=_load_strings(row["quality_flags_json"]),
    )


def _model_version_from_row(row: sqlite3.Row) -> ModelVersion:
    return ModelVersion(
        model_version_id=row["model_version_id"],
        model_type=row["model_type"],
        configuration_hash=row["configuration_hash"],
        created_at=_load_datetime(row["created_at"]),
        artifact_ref=row["artifact_ref"],
        artifact_checksum_sha256=row["artifact_checksum_sha256"],
        training_data_version=row["training_data_version"],
    )


def _detector_version_from_row(row: sqlite3.Row) -> DetectorVersion:
    return DetectorVersion(
        detector_version_id=row["detector_version_id"],
        scoring_method=row["scoring_method"],
        threshold=row["threshold"],
        configuration_hash=row["configuration_hash"],
        created_at=_load_datetime(row["created_at"]),
        calibration_data_version=row["calibration_data_version"],
        calibration_residual_mean_tecu=row["calibration_residual_mean_tecu"],
        calibration_residual_standard_deviation_tecu=row[
            "calibration_residual_standard_deviation_tecu"
        ],
        calibration_sample_count=row["calibration_sample_count"],
    )


class SQLiteSourceArtifactRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, artifact: SourceArtifact) -> None:
        if not isinstance(artifact.storage_ref, str) or not artifact.storage_ref.strip():
            raise ValueError("source artifact storage_ref must not be empty")
        existing = self.get(artifact.artifact_id)
        if existing is not None:
            _assert_exact(existing, artifact, "source artifact", artifact.artifact_id)
            return

        self._connection.execute(
            """
            INSERT INTO source_artifacts (
                artifact_id, provider, product, parser_version,
                revision_priority, checksum_sha256, storage_ref, ingested_at,
                revision, source_uri
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                artifact.provider,
                artifact.product,
                artifact.parser_version,
                artifact.revision_priority,
                artifact.checksum_sha256,
                artifact.storage_ref,
                _dump_datetime(artifact.ingested_at),
                artifact.revision,
                artifact.source_uri,
            ),
        )

    def get(self, artifact_id: str) -> SourceArtifact | None:
        row = self._connection.execute(
            "SELECT * FROM source_artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        return None if row is None else _artifact_from_row(row)

    def find_by_checksum(self, checksum_sha256: str) -> SourceArtifact | None:
        row = self._connection.execute(
            """
            SELECT * FROM source_artifacts WHERE checksum_sha256 = ?
            ORDER BY revision_priority DESC, ingested_at DESC, artifact_id DESC
            LIMIT 1
            """,
            (checksum_sha256,),
        ).fetchone()
        return None if row is None else _artifact_from_row(row)

    def find_acquisition(
        self,
        *,
        provider: str,
        product: str,
        parser_version: str,
        revision_priority: int,
        revision: str | None,
        source_uri: str | None,
        checksum_sha256: str,
    ) -> SourceArtifact | None:
        for field_name, value in (
            ("provider", provider),
            ("product", product),
            ("parser_version", parser_version),
            ("checksum_sha256", checksum_sha256),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if (
            not isinstance(revision_priority, int)
            or isinstance(revision_priority, bool)
            or revision_priority < 0
        ):
            raise ValueError("revision_priority must be a non-negative integer")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("revision must be None or non-empty")
        if source_uri is not None and (
            not isinstance(source_uri, str) or not source_uri.strip()
        ):
            raise ValueError("source_uri must be None or non-empty")
        row = self._connection.execute(
            """
            SELECT * FROM source_artifacts
            WHERE provider = ? AND product = ?
              AND parser_version = ? AND revision_priority = ?
              AND revision IS ? AND source_uri IS ? AND checksum_sha256 = ?
            ORDER BY ingested_at DESC, artifact_id DESC LIMIT 1
            """,
            (
                provider,
                product,
                parser_version,
                revision_priority,
                revision,
                source_uri,
                checksum_sha256,
            ),
        ).fetchone()
        return None if row is None else _artifact_from_row(row)


class SQLiteTECObservationRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add_many(self, observations: Iterable[TECObservation]) -> None:
        self._connection.execute("SAVEPOINT add_tec_observations")
        try:
            for observation in observations:
                self._validate(observation)
                existing = self._find(observation)
                if existing is not None:
                    _assert_exact(
                        existing,
                        observation,
                        "TEC observation",
                        (
                            f"{observation.artifact_id}/"
                            f"{observation.observed_at.isoformat()}/"
                            f"{observation.latitude_degrees}/"
                            f"{observation.longitude_degrees}"
                        ),
                    )
                    continue
                self._connection.execute(
                    """
                    INSERT INTO tec_observations (
                        artifact_id, observed_at, latitude_degrees,
                        longitude_degrees, vtec_tecu, quality_flags_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.artifact_id,
                        _dump_datetime(observation.observed_at),
                        observation.latitude_degrees,
                        observation.longitude_degrees,
                        observation.vtec_tecu,
                        _dump_strings(observation.quality_flags),
                    ),
                )
        except BaseException:
            self._connection.execute("ROLLBACK TO SAVEPOINT add_tec_observations")
            self._connection.execute("RELEASE SAVEPOINT add_tec_observations")
            raise
        else:
            self._connection.execute("RELEASE SAVEPOINT add_tec_observations")

    @staticmethod
    def _validate(observation: TECObservation) -> None:
        if not isinstance(observation.artifact_id, str) or not observation.artifact_id:
            raise ValueError("TEC observation artifact_id must not be empty")
        _dump_datetime(observation.observed_at)
        if not isfinite(observation.latitude_degrees) or not (
            -90.0 <= observation.latitude_degrees <= 90.0
        ):
            raise ValueError("TEC observation latitude must be in [-90, 90]")
        if not isfinite(observation.longitude_degrees) or not (
            -180.0 <= observation.longitude_degrees < 180.0
        ):
            raise ValueError("TEC observation longitude must be in [-180, 180)")
        if not isfinite(observation.vtec_tecu):
            raise ValueError("TEC observation VTEC must be finite")
        if any(
            not isinstance(flag, str) or not flag
            for flag in observation.quality_flags
        ):
            raise ValueError("TEC observation quality flags must not be empty")

    def _find(self, observation: TECObservation) -> TECObservation | None:
        row = self._connection.execute(
            """
            SELECT * FROM tec_observations
            WHERE artifact_id = ? AND observed_at = ?
              AND latitude_degrees = ? AND longitude_degrees = ?
            """,
            (
                observation.artifact_id,
                _dump_datetime(observation.observed_at),
                observation.latitude_degrees,
                observation.longitude_degrees,
            ),
        ).fetchone()
        return None if row is None else _tec_observation_from_row(row)

    def for_artifact(self, artifact_id: str) -> Sequence[TECObservation]:
        rows = self._connection.execute(
            """
            SELECT * FROM tec_observations
            WHERE artifact_id = ?
            ORDER BY observed_at, latitude_degrees, longitude_degrees
            """,
            (artifact_id,),
        ).fetchall()
        return tuple(_tec_observation_from_row(row) for row in rows)

    def for_artifact_at(
        self,
        artifact_id: str,
        observed_at: datetime,
    ) -> Sequence[TECObservation]:
        rows = self._connection.execute(
            """
            SELECT * FROM tec_observations
            WHERE artifact_id = ? AND observed_at = ?
            ORDER BY latitude_degrees, longitude_degrees
            """,
            (artifact_id, _dump_datetime(observed_at)),
        ).fetchall()
        return tuple(_tec_observation_from_row(row) for row in rows)


class SQLiteProcessingVersionRegistry:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def register(self, processing_version: ProcessingVersion) -> None:
        existing = self.get(processing_version.processing_version_id)
        if existing is not None:
            _assert_exact(
                existing,
                processing_version,
                "processing version",
                processing_version.processing_version_id,
            )
            return
        self._connection.execute(
            """
            INSERT INTO processing_versions (
                processing_version_id, code_revision, configuration_hash,
                parser_version, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                processing_version.processing_version_id,
                processing_version.code_revision,
                processing_version.configuration_hash,
                processing_version.parser_version,
                _dump_datetime(processing_version.created_at),
            ),
        )

    def get(self, processing_version_id: str) -> ProcessingVersion | None:
        row = self._connection.execute(
            """
            SELECT * FROM processing_versions WHERE processing_version_id = ?
            """,
            (processing_version_id,),
        ).fetchone()
        return None if row is None else _processing_version_from_row(row)


class SQLiteRegionRegistry:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def register(self, region_version: RegionVersion) -> None:
        existing = self.get(region_version.region_version_id)
        if existing is not None:
            _assert_exact(
                existing,
                region_version,
                "region version",
                region_version.region_version_id,
            )
            return
        self._connection.execute(
            """
            INSERT INTO region_versions (
                region_version_id, name, boundary_ref,
                boundary_checksum_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                region_version.region_version_id,
                region_version.name,
                region_version.boundary_ref,
                region_version.boundary_checksum_sha256,
                _dump_datetime(region_version.created_at),
            ),
        )

    def get(self, region_version_id: str) -> RegionVersion | None:
        row = self._connection.execute(
            "SELECT * FROM region_versions WHERE region_version_id = ?",
            (region_version_id,),
        ).fetchone()
        return None if row is None else _region_version_from_row(row)


class SQLiteRegionalObservationRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, observation: RegionalObservation) -> None:
        for field_name, value in (
            ("observation_id", observation.observation_id),
            ("artifact_id", observation.artifact_id),
            ("processing_version_id", observation.processing_version_id),
            ("region_version_id", observation.region_version_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"regional observation {field_name} must not be empty"
                )
        _dump_datetime(observation.observed_at)
        _dump_datetime(observation.produced_at)
        if observation.produced_at < observation.observed_at:
            raise ValueError(
                "regional observation cannot be produced before it is observed"
            )
        if not isfinite(observation.mean_vtec_tecu) or not isfinite(
            observation.median_vtec_tecu
        ):
            raise ValueError("regional observation TEC values must be finite")
        if (
            not isinstance(observation.cell_count, int)
            or isinstance(observation.cell_count, bool)
            or observation.cell_count <= 0
        ):
            raise ValueError("regional observation cell_count must be positive")
        if not 0.0 < observation.coverage_fraction <= 1.0:
            raise ValueError(
                "regional observation coverage_fraction must be in (0, 1]"
            )
        if any(
            not isinstance(flag, str) or not flag
            for flag in observation.quality_flags
        ):
            raise ValueError(
                "regional observation quality flags must not be empty"
            )
        provenance = self._connection.execute(
            """
            SELECT source_artifacts.parser_version AS artifact_parser_version,
                   source_artifacts.ingested_at AS artifact_ingested_at,
                   processing_versions.parser_version AS processing_parser_version,
                   processing_versions.created_at AS processing_created_at,
                   region_versions.created_at AS region_created_at
            FROM source_artifacts
            CROSS JOIN processing_versions
            CROSS JOIN region_versions
            WHERE source_artifacts.artifact_id = ?
              AND processing_versions.processing_version_id = ?
              AND region_versions.region_version_id = ?
            """,
            (
                observation.artifact_id,
                observation.processing_version_id,
                observation.region_version_id,
            ),
        ).fetchone()
        if provenance is None:
            raise sqlite3.IntegrityError(
                "regional observation requires existing source, processing, and "
                "region versions"
            )
        if (
            provenance["artifact_parser_version"]
            != provenance["processing_parser_version"]
        ):
            raise sqlite3.IntegrityError(
                "regional observation processing version does not declare "
                "the artifact parser version"
            )
        artifact_ingested_at = _load_datetime(provenance["artifact_ingested_at"])
        dependency_available_at = max(
            artifact_ingested_at,
            _load_datetime(provenance["processing_created_at"]),
            _load_datetime(provenance["region_created_at"]),
        )
        if observation.produced_at < dependency_available_at:
            raise ValueError(
                "regional observation cannot predate its source or version metadata"
            )
        existing = self.get(observation.observation_id)
        if existing is not None:
            _assert_exact(
                existing,
                observation,
                "regional observation",
                observation.observation_id,
            )
            return
        self._connection.execute(
            """
            INSERT INTO regional_observations (
                observation_id, artifact_id, processing_version_id,
                region_version_id, observed_at, produced_at, mean_vtec_tecu,
                median_vtec_tecu, cell_count, coverage_fraction,
                quality_flags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.observation_id,
                observation.artifact_id,
                observation.processing_version_id,
                observation.region_version_id,
                _dump_datetime(observation.observed_at),
                _dump_datetime(observation.produced_at),
                observation.mean_vtec_tecu,
                observation.median_vtec_tecu,
                observation.cell_count,
                observation.coverage_fraction,
                _dump_strings(observation.quality_flags),
            ),
        )

    def get(self, observation_id: str) -> RegionalObservation | None:
        row = self._connection.execute(
            "SELECT * FROM regional_observations WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()
        return None if row is None else _regional_observation_from_row(row)

    def find(
        self,
        *,
        source_provider: str,
        source_product: str,
        region_version_id: str,
        processing_version_id: str,
        observed_at: datetime,
        available_at_or_before: datetime | None = None,
    ) -> RegionalObservation | None:
        parameters: list[object] = [
            source_provider,
            source_product,
            region_version_id,
            processing_version_id,
            _dump_datetime(observed_at),
        ]
        availability_filter = ""
        if available_at_or_before is not None:
            availability_filter = (
                "AND source_artifacts.ingested_at <= ? "
                "AND regional_observations.produced_at <= ?"
            )
            available_at = _dump_datetime(available_at_or_before)
            parameters.extend((available_at, available_at))
        row = self._connection.execute(
            f"""
            SELECT regional_observations.*
            FROM regional_observations
            JOIN source_artifacts USING (artifact_id)
            WHERE source_artifacts.provider = ?
              AND source_artifacts.product = ?
              AND regional_observations.region_version_id = ?
              AND regional_observations.processing_version_id = ?
              AND regional_observations.observed_at = ?
              {availability_filter}
            ORDER BY source_artifacts.revision_priority DESC,
                     source_artifacts.ingested_at DESC,
                     regional_observations.artifact_id DESC,
                     regional_observations.processing_version_id DESC,
                     regional_observations.observation_id DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        return None if row is None else _regional_observation_from_row(row)

    def for_artifact(self, artifact_id: str) -> Sequence[RegionalObservation]:
        rows = self._connection.execute(
            """
            SELECT * FROM regional_observations
            WHERE artifact_id = ?
            ORDER BY observed_at, region_version_id, processing_version_id,
                     observation_id
            """,
            (artifact_id,),
        ).fetchall()
        return tuple(_regional_observation_from_row(row) for row in rows)

    def history(
        self,
        *,
        source_provider: str,
        source_product: str,
        region_version_id: str,
        processing_version_id: str,
        observed_at_or_before: datetime,
        available_at_or_before: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence[RegionalObservation]:
        if limit is not None and limit < 0:
            raise ValueError("history limit must not be negative")
        if limit == 0:
            return ()

        order_and_limit = "ORDER BY observed_at ASC, observation_id ASC"
        parameters: list[object] = [
            source_provider,
            source_product,
            region_version_id,
            processing_version_id,
            _dump_datetime(observed_at_or_before),
        ]
        availability_filter = ""
        if available_at_or_before is not None:
            availability_filter = (
                "AND source_artifacts.ingested_at <= ? "
                "AND regional_observations.produced_at <= ?"
            )
            available_at = _dump_datetime(available_at_or_before)
            parameters.extend((available_at, available_at))
        if limit is not None:
            order_and_limit = "ORDER BY observed_at DESC, observation_id DESC LIMIT ?"
            parameters.append(limit)

        rows = self._connection.execute(
            f"""
            WITH ranked AS (
                SELECT regional_observations.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY regional_observations.observed_at
                           ORDER BY source_artifacts.revision_priority DESC,
                                    source_artifacts.ingested_at DESC,
                                    regional_observations.artifact_id DESC,
                                    regional_observations.processing_version_id DESC,
                                    regional_observations.observation_id DESC
                       ) AS revision_rank
                FROM regional_observations
                JOIN source_artifacts USING (artifact_id)
                WHERE source_artifacts.provider = ?
                  AND source_artifacts.product = ?
                  AND regional_observations.region_version_id = ?
                  AND regional_observations.processing_version_id = ?
                  AND regional_observations.observed_at <= ?
                  {availability_filter}
            )
            SELECT * FROM ranked
            WHERE revision_rank = 1
            {order_and_limit}
            """,
            parameters,
        ).fetchall()
        observations = tuple(_regional_observation_from_row(row) for row in rows)
        if limit is not None:
            return tuple(reversed(observations))
        return observations


class SQLiteModelRegistry:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def register(self, model_version: ModelVersion) -> None:
        if (model_version.artifact_ref is None) != (
            model_version.artifact_checksum_sha256 is None
        ):
            raise ValueError(
                "model artifact_ref and artifact_checksum_sha256 must be set together"
            )
        existing = self.get(model_version.model_version_id)
        if existing is not None:
            _assert_exact(
                existing,
                model_version,
                "model version",
                model_version.model_version_id,
            )
            return
        self._connection.execute(
            """
            INSERT INTO model_versions (
                model_version_id, model_type, configuration_hash, created_at,
                artifact_ref, artifact_checksum_sha256, training_data_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_version.model_version_id,
                model_version.model_type,
                model_version.configuration_hash,
                _dump_datetime(model_version.created_at),
                model_version.artifact_ref,
                model_version.artifact_checksum_sha256,
                model_version.training_data_version,
            ),
        )

    def get(self, model_version_id: str) -> ModelVersion | None:
        row = self._connection.execute(
            "SELECT * FROM model_versions WHERE model_version_id = ?",
            (model_version_id,),
        ).fetchone()
        return None if row is None else _model_version_from_row(row)


class SQLiteDetectorRegistry:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def register(self, detector_version: DetectorVersion) -> None:
        existing = self.get(detector_version.detector_version_id)
        if existing is not None:
            _assert_exact(
                existing,
                detector_version,
                "detector version",
                detector_version.detector_version_id,
            )
            return
        self._connection.execute(
            """
            INSERT INTO detector_versions (
                detector_version_id, scoring_method, threshold,
                configuration_hash, created_at, calibration_data_version,
                calibration_residual_mean_tecu,
                calibration_residual_standard_deviation_tecu,
                calibration_sample_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detector_version.detector_version_id,
                detector_version.scoring_method,
                detector_version.threshold,
                detector_version.configuration_hash,
                _dump_datetime(detector_version.created_at),
                detector_version.calibration_data_version,
                detector_version.calibration_residual_mean_tecu,
                detector_version.calibration_residual_standard_deviation_tecu,
                detector_version.calibration_sample_count,
            ),
        )

    def get(self, detector_version_id: str) -> DetectorVersion | None:
        row = self._connection.execute(
            "SELECT * FROM detector_versions WHERE detector_version_id = ?",
            (detector_version_id,),
        ).fetchone()
        return None if row is None else _detector_version_from_row(row)


class SQLiteForecastRepository:
    _ALLOWED_TRANSITIONS = {
        ForecastStatus.PENDING: {
            ForecastStatus.SCORED,
            ForecastStatus.INSUFFICIENT_DATA,
            ForecastStatus.FAILED,
        },
        ForecastStatus.SCORED: set(),
        ForecastStatus.INSUFFICIENT_DATA: {ForecastStatus.SCORED},
        ForecastStatus.FAILED: set(),
    }

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, forecast: Forecast) -> None:
        existing = self.get(forecast.forecast_id)
        if existing is not None:
            _assert_exact(existing, forecast, "forecast", forecast.forecast_id)
            return

        self._validate_new_forecast(forecast)

        self._connection.execute("SAVEPOINT add_forecast")
        try:
            self._connection.execute(
                """
                INSERT INTO forecasts (
                    forecast_id, origin_at, issued_at, valid_at,
                    source_provider, source_product, region_version_id,
                    model_version_id, processing_version_id, target_metric,
                    predicted_vtec_tecu, mode, input_manifest_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    forecast.forecast_id,
                    _dump_datetime(forecast.origin_at),
                    _dump_datetime(forecast.issued_at),
                    _dump_datetime(forecast.valid_at),
                    forecast.source_provider,
                    forecast.source_product,
                    forecast.region_version_id,
                    forecast.model_version_id,
                    forecast.processing_version_id,
                    forecast.target_metric.value,
                    forecast.predicted_vtec_tecu,
                    forecast.mode.value,
                    _input_manifest_hash(forecast.input_observation_ids),
                    forecast.status.value,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO forecast_input_observations (
                    forecast_id, position, observation_id
                ) VALUES (?, ?, ?)
                """,
                (
                    (forecast.forecast_id, position, observation_id)
                    for position, observation_id in enumerate(
                        forecast.input_observation_ids
                    )
                ),
            )
        except BaseException:
            self._connection.execute("ROLLBACK TO SAVEPOINT add_forecast")
            self._connection.execute("RELEASE SAVEPOINT add_forecast")
            raise
        else:
            self._connection.execute("RELEASE SAVEPOINT add_forecast")

    def _validate_new_forecast(self, forecast: Forecast) -> None:
        for field_name, value in (
            ("forecast_id", forecast.forecast_id),
            ("source_provider", forecast.source_provider),
            ("source_product", forecast.source_product),
            ("region_version_id", forecast.region_version_id),
            ("model_version_id", forecast.model_version_id),
            ("processing_version_id", forecast.processing_version_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"forecast {field_name} must not be empty")
        for instant in (forecast.origin_at, forecast.issued_at, forecast.valid_at):
            _dump_datetime(instant)
        if forecast.valid_at <= forecast.origin_at:
            raise ValueError("forecast valid_at must be later than origin_at")
        if forecast.issued_at < forecast.origin_at:
            raise ValueError("forecast issued_at cannot precede origin_at")
        if (
            forecast.mode is ForecastMode.LIVE
            and forecast.issued_at >= forecast.valid_at
        ):
            raise ValueError("a live forecast must be issued before valid_at")
        if not isinstance(forecast.mode, ForecastMode):
            raise ValueError("forecast mode must be a ForecastMode")
        if not isinstance(forecast.target_metric, TargetMetric):
            raise ValueError("forecast target_metric must be a TargetMetric")
        if forecast.status is not ForecastStatus.PENDING:
            raise ValueError("a new forecast must begin pending")
        if not isfinite(forecast.predicted_vtec_tecu):
            raise ValueError("forecast prediction must be finite")
        if not forecast.input_observation_ids:
            raise ValueError("forecast must identify at least one input observation")
        if len(set(forecast.input_observation_ids)) != len(
            forecast.input_observation_ids
        ):
            raise ValueError("forecast input observation IDs must be unique")
        if any(
            not isinstance(observation_id, str) or not observation_id
            for observation_id in forecast.input_observation_ids
        ):
            raise ValueError("forecast input observation IDs must not be empty")

        versions = self._connection.execute(
            """
            SELECT region_versions.created_at AS region_created_at,
                   model_versions.created_at AS model_created_at,
                   processing_versions.created_at AS processing_created_at,
                   processing_versions.parser_version AS processing_parser_version
            FROM region_versions
            CROSS JOIN model_versions
            CROSS JOIN processing_versions
            WHERE region_versions.region_version_id = ?
              AND model_versions.model_version_id = ?
              AND processing_versions.processing_version_id = ?
            """,
            (
                forecast.region_version_id,
                forecast.model_version_id,
                forecast.processing_version_id,
            ),
        ).fetchone()
        if versions is None:
            raise sqlite3.IntegrityError(
                "forecast requires existing region, model, and processing versions"
            )
        for field_name in (
            "region_created_at",
            "model_created_at",
            "processing_created_at",
        ):
            if _load_datetime(versions[field_name]) > forecast.issued_at:
                raise sqlite3.IntegrityError(
                    f"forecast uses {field_name.removesuffix('_created_at')} "
                    "created after issuance"
                )

        expected_series = (forecast.source_provider, forecast.source_product)
        for observation_id in forecast.input_observation_ids:
            provenance = _observation_provenance(
                self._connection,
                observation_id,
            )
            if provenance is None:
                raise sqlite3.IntegrityError(
                    f"unknown forecast input observation {observation_id!r}"
                )
            actual_series = (
                provenance["source_provider"],
                provenance["source_product"],
            )
            if actual_series != expected_series:
                raise sqlite3.IntegrityError(
                    "forecast input observation source series does not match "
                    "the forecast"
                )
            if provenance["region_version_id"] != forecast.region_version_id:
                raise sqlite3.IntegrityError(
                    "forecast input observation region does not match the forecast"
                )
            if (
                provenance["processing_version_id"]
                != forecast.processing_version_id
            ):
                raise sqlite3.IntegrityError(
                    "forecast input observation processing version does not match "
                    "the forecast"
                )
            if (
                provenance["artifact_parser_version"]
                != versions["processing_parser_version"]
            ):
                raise sqlite3.IntegrityError(
                    "forecast input parser version does not match processing version"
                )

            observed_at = _load_datetime(provenance["observed_at"])
            produced_at = _load_datetime(provenance["produced_at"])
            ingested_at = _load_datetime(provenance["artifact_ingested_at"])
            if observed_at > forecast.origin_at:
                raise sqlite3.IntegrityError(
                    "forecast input observation occurs after forecast origin"
                )
            if forecast.mode is ForecastMode.LIVE and (
                produced_at > forecast.issued_at
                or ingested_at > forecast.issued_at
            ):
                raise sqlite3.IntegrityError(
                    "live forecast input was unavailable at issuance"
                )

    def get(self, forecast_id: str) -> Forecast | None:
        row = self._connection.execute(
            "SELECT * FROM forecasts WHERE forecast_id = ?", (forecast_id,)
        ).fetchone()
        if row is None:
            return None
        return self._from_row(row)

    def find_equivalent(self, forecast: Forecast) -> Forecast | None:
        row = self._connection.execute(
            """
            SELECT * FROM forecasts
            WHERE origin_at = ? AND issued_at = ? AND valid_at = ?
              AND source_provider = ? AND source_product = ?
              AND region_version_id = ? AND model_version_id = ?
              AND processing_version_id = ? AND target_metric = ?
              AND mode = ? AND input_manifest_hash = ?
            """,
            (
                _dump_datetime(forecast.origin_at),
                _dump_datetime(forecast.issued_at),
                _dump_datetime(forecast.valid_at),
                forecast.source_provider,
                forecast.source_product,
                forecast.region_version_id,
                forecast.model_version_id,
                forecast.processing_version_id,
                forecast.target_metric.value,
                forecast.mode.value,
                _input_manifest_hash(forecast.input_observation_ids),
            ),
        ).fetchone()
        return None if row is None else self._from_row(row)

    def _from_row(self, row: sqlite3.Row) -> Forecast:
        input_rows = self._connection.execute(
            """
            SELECT observation_id FROM forecast_input_observations
            WHERE forecast_id = ? ORDER BY position
            """,
            (row["forecast_id"],),
        ).fetchall()
        return Forecast(
            forecast_id=row["forecast_id"],
            origin_at=_load_datetime(row["origin_at"]),
            issued_at=_load_datetime(row["issued_at"]),
            valid_at=_load_datetime(row["valid_at"]),
            source_provider=row["source_provider"],
            source_product=row["source_product"],
            region_version_id=row["region_version_id"],
            model_version_id=row["model_version_id"],
            processing_version_id=row["processing_version_id"],
            target_metric=TargetMetric(row["target_metric"]),
            predicted_vtec_tecu=row["predicted_vtec_tecu"],
            mode=ForecastMode(row["mode"]),
            status=ForecastStatus(row["status"]),
            input_observation_ids=tuple(
                input_row["observation_id"] for input_row in input_rows
            ),
        )

    def list_pending(self, *, valid_at_or_before: datetime) -> Sequence[Forecast]:
        rows = self._connection.execute(
            """
            SELECT * FROM forecasts
            WHERE status = ? AND valid_at <= ?
            ORDER BY valid_at, issued_at, forecast_id
            """,
            (ForecastStatus.PENDING.value, _dump_datetime(valid_at_or_before)),
        ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def replace(self, forecast: Forecast) -> None:
        current = self.get(forecast.forecast_id)
        if current is None:
            raise KeyError(f"unknown forecast {forecast.forecast_id!r}")
        if current == forecast:
            return

        incoming_with_current_status = replace(forecast, status=current.status)
        if incoming_with_current_status != current:
            _immutable_conflict("forecast", forecast.forecast_id)

        allowed = self._ALLOWED_TRANSITIONS[current.status]
        if forecast.status not in allowed:
            raise ValueError(
                f"invalid forecast status transition: "
                f"{current.status.value} -> {forecast.status.value}"
            )

        cursor = self._connection.execute(
            """
            UPDATE forecasts SET status = ?
            WHERE forecast_id = ? AND status = ?
            """,
            (forecast.status.value, forecast.forecast_id, current.status.value),
        )
        if cursor.rowcount != 1:
            raise sqlite3.OperationalError(
                f"forecast {forecast.forecast_id!r} changed concurrently"
            )


def _decision_from_row(row: sqlite3.Row) -> AnomalyDecision:
    return AnomalyDecision(
        decision_id=row["decision_id"],
        forecast_id=row["forecast_id"],
        observation_id=row["observation_id"],
        detector_version_id=row["detector_version_id"],
        scored_at=_load_datetime(row["scored_at"]),
        residual_tecu=row["residual_tecu"],
        anomaly_score=row["anomaly_score"],
        threshold=row["threshold"],
        is_forecast_anomaly=bool(row["is_forecast_anomaly"]),
        disturbance_assessment=DisturbanceAssessment(
            row["disturbance_assessment"]
        ),
        assessment_source=row["assessment_source"],
        supersedes_decision_id=row["supersedes_decision_id"],
    )


class SQLiteDecisionRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, decision: AnomalyDecision) -> None:
        existing = self._get(decision.decision_id)
        if existing is not None:
            _assert_exact(existing, decision, "anomaly decision", decision.decision_id)
            return

        self._validate_new_decision(decision)

        self._connection.execute(
            """
            INSERT INTO anomaly_decisions (
                decision_id, forecast_id, observation_id, detector_version_id,
                scored_at, residual_tecu, anomaly_score, threshold,
                is_forecast_anomaly, disturbance_assessment,
                assessment_source, supersedes_decision_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.forecast_id,
                decision.observation_id,
                decision.detector_version_id,
                _dump_datetime(decision.scored_at),
                decision.residual_tecu,
                decision.anomaly_score,
                decision.threshold,
                int(decision.is_forecast_anomaly),
                decision.disturbance_assessment.value,
                decision.assessment_source,
                decision.supersedes_decision_id,
            ),
        )

    def _validate_new_decision(self, decision: AnomalyDecision) -> None:
        for field_name, value in (
            ("decision_id", decision.decision_id),
            ("forecast_id", decision.forecast_id),
            ("observation_id", decision.observation_id),
            ("detector_version_id", decision.detector_version_id),
            ("assessment_source", decision.assessment_source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"decision {field_name} must not be empty")
        _dump_datetime(decision.scored_at)
        numeric_values = (
            decision.residual_tecu,
            decision.anomaly_score,
            decision.threshold,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("decision numeric values must be finite")
        if decision.anomaly_score < 0.0:
            raise ValueError("decision anomaly_score must be non-negative")
        if decision.threshold <= 0.0:
            raise ValueError("decision threshold must be positive")
        if not isinstance(decision.is_forecast_anomaly, bool):
            raise ValueError("decision is_forecast_anomaly must be a bool")
        if not isinstance(
            decision.disturbance_assessment,
            DisturbanceAssessment,
        ):
            raise ValueError(
                "decision disturbance_assessment must be a DisturbanceAssessment"
            )
        if decision.is_forecast_anomaly != (
            decision.anomaly_score >= decision.threshold
        ):
            raise ValueError(
                "decision anomaly flag does not agree with score and threshold"
            )

        forecast = SQLiteForecastRepository(self._connection).get(
            decision.forecast_id
        )
        if forecast is None:
            raise sqlite3.IntegrityError(
                f"unknown forecast {decision.forecast_id!r}"
            )
        if forecast.status not in {ForecastStatus.PENDING, ForecastStatus.SCORED}:
            raise sqlite3.IntegrityError(
                "a decision cannot be added to a terminal unscored forecast"
            )
        provenance = _observation_provenance(
            self._connection,
            decision.observation_id,
        )
        if provenance is None:
            raise sqlite3.IntegrityError(
                f"unknown decision observation {decision.observation_id!r}"
            )
        if (
            provenance["source_provider"],
            provenance["source_product"],
        ) != (forecast.source_provider, forecast.source_product):
            raise sqlite3.IntegrityError(
                "decision observation source series does not match the forecast"
            )
        if provenance["region_version_id"] != forecast.region_version_id:
            raise sqlite3.IntegrityError(
                "decision observation region does not match the forecast"
            )
        if (
            provenance["processing_version_id"]
            != forecast.processing_version_id
        ):
            raise sqlite3.IntegrityError(
                "decision observation processing version does not match the forecast"
            )

        observed_at = _load_datetime(provenance["observed_at"])
        produced_at = _load_datetime(provenance["produced_at"])
        ingested_at = _load_datetime(provenance["artifact_ingested_at"])
        if observed_at != forecast.valid_at:
            raise sqlite3.IntegrityError(
                "decision observation time does not match forecast valid_at"
            )
        if decision.scored_at < max(forecast.issued_at, forecast.valid_at):
            raise ValueError(
                "decision cannot be scored before issuance or forecast valid_at"
            )
        if decision.scored_at < produced_at or decision.scored_at < ingested_at:
            raise ValueError(
                "decision cannot use an observation unavailable at scoring time"
            )

        detector = SQLiteDetectorRegistry(self._connection).get(
            decision.detector_version_id
        )
        if detector is None:
            raise sqlite3.IntegrityError(
                f"unknown detector version {decision.detector_version_id!r}"
            )
        if detector.created_at > decision.scored_at:
            raise sqlite3.IntegrityError(
                "decision uses a detector created after scoring time"
            )
        if decision.threshold != detector.threshold:
            raise sqlite3.IntegrityError(
                "decision threshold does not match its detector version"
            )

        actual_vtec_tecu = (
            provenance["mean_vtec_tecu"]
            if forecast.target_metric is TargetMetric.MEAN_VTEC
            else provenance["median_vtec_tecu"]
        )
        expected_residual = actual_vtec_tecu - forecast.predicted_vtec_tecu
        if not isclose(
            decision.residual_tecu,
            expected_residual,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise sqlite3.IntegrityError(
                "decision residual does not match its forecast and observation"
            )

        if decision.supersedes_decision_id is not None:
            superseded = self._get(decision.supersedes_decision_id)
            if superseded is None:
                raise sqlite3.IntegrityError(
                    f"unknown superseded decision "
                    f"{decision.supersedes_decision_id!r}"
                )
            if superseded.forecast_id != decision.forecast_id:
                raise sqlite3.IntegrityError(
                    "a decision may only supersede one for the same forecast"
                )
            if superseded.scored_at >= decision.scored_at:
                raise sqlite3.IntegrityError(
                    "a decision must be scored after the decision it supersedes"
                )

    def _get(self, decision_id: str) -> AnomalyDecision | None:
        row = self._connection.execute(
            "SELECT * FROM anomaly_decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        return None if row is None else _decision_from_row(row)

    def for_forecast(self, forecast_id: str) -> Sequence[AnomalyDecision]:
        rows = self._connection.execute(
            """
            SELECT * FROM anomaly_decisions
            WHERE forecast_id = ? ORDER BY scored_at, decision_id
            """,
            (forecast_id,),
        ).fetchall()
        return tuple(_decision_from_row(row) for row in rows)

    def search(
        self,
        *,
        valid_at_or_after: datetime,
        valid_before: datetime,
        region_version_id: str | None = None,
        is_forecast_anomaly: bool | None = None,
        disturbance_assessment: DisturbanceAssessment | None = None,
        include_superseded: bool = False,
    ) -> Sequence[AnomalyDecision]:
        if valid_before <= valid_at_or_after:
            raise ValueError("valid_before must be later than valid_at_or_after")

        filters = ["forecasts.valid_at >= ?", "forecasts.valid_at < ?"]
        parameters: list[object] = [
            _dump_datetime(valid_at_or_after),
            _dump_datetime(valid_before),
        ]
        if not include_superseded:
            filters.append(
                "NOT EXISTS ("
                "SELECT 1 FROM anomaly_decisions AS successor "
                "WHERE successor.supersedes_decision_id = "
                "anomaly_decisions.decision_id)"
            )
        if region_version_id is not None:
            filters.append("forecasts.region_version_id = ?")
            parameters.append(region_version_id)
        if is_forecast_anomaly is not None:
            filters.append("anomaly_decisions.is_forecast_anomaly = ?")
            parameters.append(int(is_forecast_anomaly))
        if disturbance_assessment is not None:
            filters.append("anomaly_decisions.disturbance_assessment = ?")
            parameters.append(disturbance_assessment.value)

        rows = self._connection.execute(
            f"""
            SELECT anomaly_decisions.*
            FROM anomaly_decisions
            JOIN forecasts USING (forecast_id)
            WHERE {' AND '.join(filters)}
            ORDER BY forecasts.valid_at, anomaly_decisions.scored_at,
                     anomaly_decisions.decision_id
            """,
            parameters,
        ).fetchall()
        return tuple(_decision_from_row(row) for row in rows)


class SQLiteUnitOfWork:
    """Explicit-commit unit of work over one SQLite database.

    Exiting without calling :meth:`commit`, or exiting because of an exception,
    rolls back the entire unit.  Repository attributes are available only
    inside the context manager.
    """

    artifacts: SQLiteSourceArtifactRepository
    tec_observations: SQLiteTECObservationRepository
    processing_versions: SQLiteProcessingVersionRegistry
    regions: SQLiteRegionRegistry
    regional_observations: SQLiteRegionalObservationRepository
    models: SQLiteModelRegistry
    detectors: SQLiteDetectorRegistry
    forecasts: SQLiteForecastRepository
    decisions: SQLiteDecisionRepository

    def __init__(
        self,
        database: str | Path,
        *,
        timeout_seconds: float = 5.0,
        uri: bool = False,
    ) -> None:
        self._database = str(database)
        self._timeout_seconds = timeout_seconds
        self._uri = uri
        self._connection: sqlite3.Connection | None = None
        self._finished = False

    def __enter__(self) -> Self:
        if self._connection is not None:
            raise RuntimeError("SQLiteUnitOfWork is already active")

        connection = sqlite3.connect(
            self._database,
            timeout=self._timeout_seconds,
            isolation_level="DEFERRED",
            uri=self._uri,
        )
        try:
            connection.row_factory = sqlite3.Row
            initialize_schema(connection)
            # Every unit of work may write. Taking the reserved writer lock up
            # front prevents the read-then-write upgrade race that otherwise turns
            # concurrent idempotent ingestion into an immediate SQLITE_BUSY error.
            connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._finished = False

        self.artifacts = SQLiteSourceArtifactRepository(connection)
        self.tec_observations = SQLiteTECObservationRepository(connection)
        self.processing_versions = SQLiteProcessingVersionRegistry(connection)
        self.regions = SQLiteRegionRegistry(connection)
        self.regional_observations = SQLiteRegionalObservationRepository(connection)
        self.models = SQLiteModelRegistry(connection)
        self.detectors = SQLiteDetectorRegistry(connection)
        self.forecasts = SQLiteForecastRepository(connection)
        self.decisions = SQLiteDecisionRepository(connection)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        connection = self._require_connection()
        try:
            if not self._finished:
                connection.rollback()
        finally:
            connection.close()
            self._connection = None
            self._finished = False
        return False

    def commit(self) -> None:
        connection = self._require_unfinished_connection()
        connection.commit()
        self._finished = True

    def rollback(self) -> None:
        connection = self._require_unfinished_connection()
        connection.rollback()
        self._finished = True

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("SQLiteUnitOfWork is not active")
        return self._connection

    def _require_unfinished_connection(self) -> sqlite3.Connection:
        connection = self._require_connection()
        if self._finished:
            raise RuntimeError("SQLiteUnitOfWork transaction is already complete")
        return connection


__all__ = [
    "SCHEMA_VERSION",
    "SQLiteDecisionRepository",
    "SQLiteDetectorRegistry",
    "SQLiteForecastRepository",
    "SQLiteModelRegistry",
    "SQLiteProcessingVersionRegistry",
    "SQLiteRegionRegistry",
    "SQLiteRegionalObservationRepository",
    "SQLiteSourceArtifactRepository",
    "SQLiteTECObservationRepository",
    "SQLiteUnitOfWork",
    "initialize_schema",
]
