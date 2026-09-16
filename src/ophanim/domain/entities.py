"""Immutable records passed between OPHANIM modules.

These are domain contracts, not database models. Persistence implementations
may map them to tables, files, or another durable representation later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ophanim.domain.enums import (
    DisturbanceAssessment,
    ForecastMode,
    ForecastStatus,
    TargetMetric,
)


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """One immutable acquisition parsed by one explicit parser revision.

    Multiple records may point at the same content-addressed ``storage_ref``.
    This is intentional: reprocessing identical source bytes with a new parser
    revision creates a new artifact identity and a separately versioned set of
    raw TEC observations without duplicating the source blob.
    """

    artifact_id: str
    provider: str
    product: str
    parser_version: str
    revision_priority: int
    checksum_sha256: str
    storage_ref: str
    ingested_at: datetime
    revision: str | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("provider", self.provider),
            ("product", self.product),
            ("checksum_sha256", self.checksum_sha256),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source artifact {field_name} must not be empty")
        if not isinstance(self.parser_version, str) or not self.parser_version.strip():
            raise ValueError("source artifact parser_version must not be empty")
        if len(self.checksum_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.checksum_sha256
        ):
            raise ValueError(
                "source artifact checksum_sha256 must be 64 lowercase hex characters"
            )
        if self.ingested_at.tzinfo is None or self.ingested_at.utcoffset() is None:
            raise ValueError("source artifact ingested_at must be timezone-aware")
        if (
            not isinstance(self.revision_priority, int)
            or isinstance(self.revision_priority, bool)
            or self.revision_priority < 0
        ):
            raise ValueError("source artifact revision_priority must be non-negative")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("source artifact revision must be None or non-empty")
        if self.source_uri is not None and (
            not isinstance(self.source_uri, str) or not self.source_uri.strip()
        ):
            raise ValueError("source artifact source_uri must be None or non-empty")


@dataclass(frozen=True, slots=True)
class ProcessingVersion:
    """The code, configuration, and raw parse dependency used for derivation."""

    processing_version_id: str
    code_revision: str
    configuration_hash: str
    parser_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.parser_version, str) or not self.parser_version.strip():
            raise ValueError("processing parser_version must not be empty")


@dataclass(frozen=True, slots=True)
class RegionVersion:
    """A versioned geographic definition, independent of aggregation choice."""

    region_version_id: str
    name: str
    boundary_ref: str
    boundary_checksum_sha256: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TECObservation:
    """One validated VTEC value from a source grid."""

    artifact_id: str
    observed_at: datetime
    latitude_degrees: float
    longitude_degrees: float
    vtec_tecu: float
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionalObservation:
    """Mean and median VTEC derived for one region and observation time."""

    observation_id: str
    artifact_id: str
    processing_version_id: str
    region_version_id: str
    observed_at: datetime
    produced_at: datetime
    mean_vtec_tecu: float
    median_vtec_tecu: float
    cell_count: int
    coverage_fraction: float
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Immutable metadata for one executable forecasting model."""

    model_version_id: str
    model_type: str
    configuration_hash: str
    created_at: datetime
    artifact_ref: str | None = None
    artifact_checksum_sha256: str | None = None
    training_data_version: str | None = None


@dataclass(frozen=True, slots=True)
class Forecast:
    """A regional forecast whose actual value may not exist yet."""

    forecast_id: str
    origin_at: datetime
    issued_at: datetime
    valid_at: datetime
    source_provider: str
    source_product: str
    region_version_id: str
    model_version_id: str
    processing_version_id: str
    target_metric: TargetMetric
    predicted_vtec_tecu: float
    mode: ForecastMode = ForecastMode.LIVE
    status: ForecastStatus = ForecastStatus.PENDING
    input_observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectorVersion:
    """Immutable scoring, calibration, and threshold configuration.

    Residual z-score detectors populate all three ``calibration_*`` statistic
    fields. Other detector types may leave them unset.
    """

    detector_version_id: str
    scoring_method: str
    threshold: float
    configuration_hash: str
    created_at: datetime
    calibration_data_version: str | None = None
    calibration_residual_mean_tecu: float | None = None
    calibration_residual_standard_deviation_tecu: float | None = None
    calibration_sample_count: int | None = None


@dataclass(frozen=True, slots=True)
class AnomalyDecision:
    """A scored forecast and its distinct disturbance assessment."""

    decision_id: str
    forecast_id: str
    observation_id: str
    detector_version_id: str
    scored_at: datetime
    residual_tecu: float
    anomaly_score: float
    threshold: float
    is_forecast_anomaly: bool
    disturbance_assessment: DisturbanceAssessment
    assessment_source: str
    supersedes_decision_id: str | None = None
