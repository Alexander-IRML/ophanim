"""Residual scoring and disturbance-assessment implementations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from math import fsum, isfinite, sqrt
from typing import Protocol

from ophanim.domain import (
    AnomalyDecision,
    DetectorVersion,
    DisturbanceAssessment,
    Forecast,
    ForecastMode,
    ForecastStatus,
    RegionalObservation,
    TargetMetric,
)


class DetectionError(ValueError):
    """Base class for inputs that cannot be scored safely."""


class CalibrationConfigurationError(DetectionError):
    """A detector version lacks valid, reproducible calibration parameters."""


class InsufficientObservationDataError(DetectionError):
    """An observation is present but not fit for anomaly scoring."""


class ForecastObservationMismatchError(DetectionError):
    """A forecast and observation refer to different targets or instants."""


class AnomalyDecisionEngine(Protocol):
    """Score one forecast against an observation without persisting state."""

    def assess(
        self,
        *,
        forecast: Forecast,
        observation: RegionalObservation,
        detector_version: DetectorVersion,
        scored_at: datetime,
    ) -> AnomalyDecision: ...


class DecisionIdFactory(Protocol):
    """Injectable decision identity policy for repository-specific IDs."""

    def __call__(
        self,
        *,
        forecast: Forecast,
        observation: RegionalObservation,
        detector_version: DetectorVersion,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ResidualCalibration:
    """Explicit residual location and sample scale for a detector version.

    ``from_residuals`` uses sample standard deviation (``n - 1``). Its output
    is intended to be copied into the corresponding immutable
    :class:`DetectorVersion`, where it becomes durable provenance.
    """

    residual_mean_tecu: float
    residual_standard_deviation_tecu: float
    sample_count: int

    def __post_init__(self) -> None:
        if not isfinite(self.residual_mean_tecu):
            raise CalibrationConfigurationError(
                "calibration residual mean must be finite"
            )
        if not isfinite(self.residual_standard_deviation_tecu) or (
            self.residual_standard_deviation_tecu <= 0.0
        ):
            raise CalibrationConfigurationError(
                "calibration residual standard deviation must be finite and positive"
            )
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count < 2
        ):
            raise CalibrationConfigurationError(
                "calibration requires at least two residuals"
            )

    @classmethod
    def from_residuals(cls, residuals_tecu: Iterable[float]) -> ResidualCalibration:
        values = tuple(residuals_tecu)
        if len(values) < 2:
            raise CalibrationConfigurationError(
                "calibration requires at least two residuals"
            )
        if not all(isfinite(value) for value in values):
            raise CalibrationConfigurationError(
                "calibration residuals must all be finite"
            )
        mean = fsum(values) / len(values)
        variance = fsum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        return cls(
            residual_mean_tecu=mean,
            residual_standard_deviation_tecu=sqrt(variance),
            sample_count=len(values),
        )


def _default_decision_id(
    *,
    forecast: Forecast,
    observation: RegionalObservation,
    detector_version: DetectorVersion,
) -> str:
    payload = "\x1f".join(
        (
            forecast.forecast_id,
            observation.observation_id,
            detector_version.detector_version_id,
        )
    ).encode("utf-8")
    return f"decision-{sha256(payload).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class ResidualZScoreDecisionEngine:
    """Two-sided residual z-score baseline for v0.

    A threshold crossing is a forecast anomaly. With adequate observation
    quality it becomes a disturbance ``CANDIDATE``, never ``CONFIRMED``. A
    calculable score with an uncertainty flag remains ``UNKNOWN`` so forecast
    difficulty can be studied independently from physical disturbances.
    """

    minimum_coverage_fraction: float = 0.5
    disqualifying_quality_flags: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"invalid", "insufficient_data", "unusable"}
        )
    )
    uncertain_quality_flags: frozenset[str] = field(
        default_factory=lambda: frozenset({"low_coverage"})
    )
    decision_id_factory: DecisionIdFactory = field(
        default=_default_decision_id,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_coverage_fraction <= 1.0:
            raise ValueError("minimum_coverage_fraction must be in (0, 1]")
        all_flags = self.disqualifying_quality_flags | self.uncertain_quality_flags
        if any(not isinstance(flag, str) or not flag for flag in all_flags):
            raise ValueError("quality flags must be non-empty strings")
        if self.disqualifying_quality_flags & self.uncertain_quality_flags:
            raise ValueError(
                "a quality flag cannot be both disqualifying and uncertain"
            )

    def assess(
        self,
        *,
        forecast: Forecast,
        observation: RegionalObservation,
        detector_version: DetectorVersion,
        scored_at: datetime,
    ) -> AnomalyDecision:
        _validate_pair(forecast, observation, scored_at)
        calibration = _calibration_from_version(detector_version, scored_at)

        if detector_version.scoring_method != "residual_zscore":
            raise CalibrationConfigurationError(
                "ResidualZScoreDecisionEngine requires scoring_method "
                "'residual_zscore'"
            )
        if not isfinite(detector_version.threshold) or detector_version.threshold <= 0:
            raise CalibrationConfigurationError(
                "detector threshold must be finite and positive"
            )
        if (
            not isinstance(observation.cell_count, int)
            or isinstance(observation.cell_count, bool)
            or observation.cell_count <= 0
        ):
            raise InsufficientObservationDataError(
                "observation contains no usable cells"
            )
        if not 0.0 <= observation.coverage_fraction <= 1.0:
            raise InsufficientObservationDataError(
                "observation coverage_fraction must be in [0, 1]"
            )
        if observation.coverage_fraction < self.minimum_coverage_fraction:
            raise InsufficientObservationDataError(
                "observation coverage is below the detector quality floor"
            )
        quality_flags = set(observation.quality_flags)
        bad_flags = quality_flags & self.disqualifying_quality_flags
        if bad_flags:
            raise InsufficientObservationDataError(
                "observation has disqualifying quality flags: "
                + ", ".join(sorted(bad_flags))
            )

        actual = _target_value(observation, forecast.target_metric)
        if not isfinite(actual):
            raise InsufficientObservationDataError(
                "observed target value must be finite"
            )
        if not isfinite(forecast.predicted_vtec_tecu):
            raise DetectionError("forecast prediction must be finite")

        residual = actual - forecast.predicted_vtec_tecu
        signed_z_score = (
            residual - calibration.residual_mean_tecu
        ) / calibration.residual_standard_deviation_tecu
        anomaly_score = abs(signed_z_score)
        is_forecast_anomaly = anomaly_score >= detector_version.threshold

        if quality_flags & self.uncertain_quality_flags:
            assessment = DisturbanceAssessment.UNKNOWN
        elif is_forecast_anomaly:
            assessment = DisturbanceAssessment.CANDIDATE
        else:
            assessment = DisturbanceAssessment.NORMAL

        decision_id = self.decision_id_factory(
            forecast=forecast,
            observation=observation,
            detector_version=detector_version,
        )
        if not decision_id:
            raise DetectionError("decision_id_factory returned an empty ID")
        return AnomalyDecision(
            decision_id=decision_id,
            forecast_id=forecast.forecast_id,
            observation_id=observation.observation_id,
            detector_version_id=detector_version.detector_version_id,
            scored_at=scored_at,
            residual_tecu=residual,
            anomaly_score=anomaly_score,
            threshold=detector_version.threshold,
            is_forecast_anomaly=is_forecast_anomaly,
            disturbance_assessment=assessment,
            assessment_source="residual_zscore",
        )


def _calibration_from_version(
    detector_version: DetectorVersion, scored_at: datetime
) -> ResidualCalibration:
    _require_aware(detector_version.created_at, "detector_version.created_at")
    if detector_version.created_at > scored_at:
        raise CalibrationConfigurationError(
            "detector version was created after scoring time"
        )
    if not detector_version.calibration_data_version:
        raise CalibrationConfigurationError(
            "residual z-score detector requires calibration_data_version"
        )
    values = (
        detector_version.calibration_residual_mean_tecu,
        detector_version.calibration_residual_standard_deviation_tecu,
        detector_version.calibration_sample_count,
    )
    if any(value is None for value in values):
        raise CalibrationConfigurationError(
            "residual z-score detector requires calibration mean, scale, and count"
        )
    return ResidualCalibration(
        residual_mean_tecu=detector_version.calibration_residual_mean_tecu,
        residual_standard_deviation_tecu=(
            detector_version.calibration_residual_standard_deviation_tecu
        ),
        sample_count=detector_version.calibration_sample_count,
    )


def _validate_pair(
    forecast: Forecast,
    observation: RegionalObservation,
    scored_at: datetime,
) -> None:
    _require_aware(forecast.issued_at, "forecast.issued_at")
    _require_aware(forecast.origin_at, "forecast.origin_at")
    _require_aware(forecast.valid_at, "forecast.valid_at")
    _require_aware(observation.observed_at, "observation.observed_at")
    _require_aware(observation.produced_at, "observation.produced_at")
    _require_aware(scored_at, "scored_at")
    if forecast.valid_at <= forecast.origin_at:
        raise DetectionError("forecast valid_at must be later than origin_at")
    if forecast.issued_at < forecast.origin_at:
        raise DetectionError("forecast issued_at cannot precede origin_at")
    if (
        forecast.mode is ForecastMode.LIVE
        and forecast.issued_at >= forecast.valid_at
    ):
        raise DetectionError("a live forecast must be issued before valid_at")
    if forecast.status not in {ForecastStatus.PENDING, ForecastStatus.SCORED}:
        raise DetectionError(
            "only pending or previously scored forecasts can be assessed"
        )
    if observation.observed_at != forecast.valid_at:
        raise ForecastObservationMismatchError(
            "observation time does not equal forecast valid_at"
        )
    if observation.region_version_id != forecast.region_version_id:
        raise ForecastObservationMismatchError(
            "observation and forecast regions do not match"
        )
    if observation.processing_version_id != forecast.processing_version_id:
        raise ForecastObservationMismatchError(
            "observation and forecast processing versions do not match"
        )
    if scored_at < forecast.issued_at or scored_at < observation.observed_at:
        raise DetectionError(
            "scored_at cannot precede issuance or the target observation"
        )
    if observation.produced_at > scored_at:
        raise DetectionError(
            "scored_at cannot precede regional observation production"
        )
    if any(
        not isinstance(flag, str) or not flag
        for flag in observation.quality_flags
    ):
        raise InsufficientObservationDataError(
            "observation quality flags must be non-empty strings"
        )


def _target_value(
    observation: RegionalObservation, target_metric: TargetMetric
) -> float:
    if target_metric is TargetMetric.MEAN_VTEC:
        return observation.mean_vtec_tecu
    if target_metric is TargetMetric.MEDIAN_VTEC:
        return observation.median_vtec_tecu
    raise DetectionError(f"unsupported target metric: {target_metric!r}")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DetectionError(f"{field_name} must be timezone-aware")
