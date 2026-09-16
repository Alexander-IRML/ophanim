"""Stateless regional forecasting and concrete v0 persistence model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from typing import Protocol

from ophanim.domain import (
    Forecast,
    ForecastMode,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionalObservation,
    TargetMetric,
)


class ForecastingError(ValueError):
    """Base class for inputs that cannot produce a defensible forecast."""


class InsufficientHistoryError(ForecastingError):
    """No sufficiently complete observation is available for forecasting."""


class HistoryLeakageError(ForecastingError):
    """Forecast history contains information unavailable at issuance time."""


class AmbiguousHistoryError(ForecastingError):
    """History has multiple candidate observations at the latest instant."""


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """The temporal, regional, and scalar target for a forecast.

    ``issued_at`` may be omitted at the application-workflow boundary.  A live
    issuance must omit it so the workflow can stamp the instant after it has
    acquired its transaction lock.  The pure model runner still requires an
    explicit value, as do controlled hindcast/replay calls.
    """

    origin_at: datetime
    valid_at: datetime
    source_provider: str
    source_product: str
    region_version_id: str
    target_metric: TargetMetric
    mode: ForecastMode = ForecastMode.LIVE
    issued_at: datetime | None = None


class Forecaster(Protocol):
    """Executable prediction logic for a registered model version."""

    def predict(
        self,
        *,
        history: Sequence[RegionalObservation],
        model_version: ModelVersion,
        target_metric: TargetMetric,
    ) -> float: ...


class ModelRunner(Protocol):
    """Construct and return a pending forecast from explicit inputs."""

    def run(
        self,
        *,
        request: ForecastRequest,
        history: Sequence[RegionalObservation],
        model_version: ModelVersion,
        processing_version: ProcessingVersion,
    ) -> Forecast: ...


class ForecastIdFactory(Protocol):
    """Injectable forecast identity policy for repository-specific IDs."""

    def __call__(
        self,
        *,
        request: ForecastRequest,
        source_observation: RegionalObservation,
        model_version: ModelVersion,
        processing_version: ProcessingVersion,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class PersistenceForecaster:
    """Predict that the latest regional value persists to ``valid_at``.

    This is intentionally a scalar baseline. It selects exactly one latest
    observation and refuses ambiguous same-instant revisions instead of using
    sequence order as an undocumented tie-breaker.
    """

    minimum_coverage_fraction: float = 0.5
    disqualifying_quality_flags: frozenset[str] = field(
        default_factory=lambda: frozenset({"invalid", "insufficient_data"})
    )
    supported_model_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"persistence", "naive_persistence"})
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_coverage_fraction <= 1.0:
            raise ValueError("minimum_coverage_fraction must be in (0, 1]")
        if not self.supported_model_types:
            raise ValueError("at least one model type must be supported")
        if any(not value for value in self.supported_model_types):
            raise ValueError("supported model types must be non-empty strings")
        if any(not value for value in self.disqualifying_quality_flags):
            raise ValueError("quality flags must be non-empty strings")

    def select_source_observation(
        self,
        *,
        history: Sequence[RegionalObservation],
        model_version: ModelVersion,
        target_metric: TargetMetric,
    ) -> RegionalObservation:
        _require_aware(model_version.created_at, "model_version.created_at")
        normalized_model_type = model_version.model_type.strip().casefold()
        normalized_supported_types = {
            value.strip().casefold() for value in self.supported_model_types
        }
        if normalized_model_type not in normalized_supported_types:
            raise ForecastingError(
                f"model_type={model_version.model_type!r} is not a persistence model"
            )
        if not isinstance(target_metric, TargetMetric):
            raise ForecastingError("target_metric must be a TargetMetric")
        if not history:
            raise InsufficientHistoryError("persistence forecasting needs history")

        observation_ids: set[str] = set()
        region_ids: set[str] = set()
        for observation in history:
            _validate_regional_observation(observation)
            if observation.observation_id in observation_ids:
                raise AmbiguousHistoryError(
                    f"duplicate history observation_id={observation.observation_id!r}"
                )
            observation_ids.add(observation.observation_id)
            region_ids.add(observation.region_version_id)
        if len(region_ids) != 1:
            raise AmbiguousHistoryError("forecast history spans multiple regions")

        latest_at = max(observation.observed_at for observation in history)
        latest = [
            observation
            for observation in history
            if observation.observed_at == latest_at
        ]
        if len(latest) != 1:
            raise AmbiguousHistoryError(
                "multiple observations exist at the latest history instant; "
                "select a source revision explicitly"
            )

        source = latest[0]
        if (
            not isinstance(source.cell_count, int)
            or isinstance(source.cell_count, bool)
            or source.cell_count <= 0
        ):
            raise InsufficientHistoryError("latest observation has no usable cells")
        if source.coverage_fraction < self.minimum_coverage_fraction:
            raise InsufficientHistoryError(
                "latest observation coverage is below the persistence quality floor"
            )
        bad_flags = (
            set(source.quality_flags) & self.disqualifying_quality_flags
        )
        if bad_flags:
            raise InsufficientHistoryError(
                "latest observation has disqualifying quality flags: "
                + ", ".join(sorted(bad_flags))
            )

        value = _target_value(source, target_metric)
        if not isfinite(value):
            raise InsufficientHistoryError("latest target value is not finite")
        return source

    def predict(
        self,
        *,
        history: Sequence[RegionalObservation],
        model_version: ModelVersion,
        target_metric: TargetMetric,
    ) -> float:
        source = self.select_source_observation(
            history=history,
            model_version=model_version,
            target_metric=target_metric,
        )
        return _target_value(source, target_metric)


class PersistenceModelRunner:
    """Validate issuance-time inputs and construct a pending forecast."""

    def __init__(
        self,
        *,
        forecaster: PersistenceForecaster | None = None,
        forecast_id_factory: ForecastIdFactory | None = None,
    ) -> None:
        self._forecaster = forecaster or PersistenceForecaster()
        self._forecast_id_factory = forecast_id_factory or _default_forecast_id

    def run(
        self,
        *,
        request: ForecastRequest,
        history: Sequence[RegionalObservation],
        model_version: ModelVersion,
        processing_version: ProcessingVersion,
    ) -> Forecast:
        _validate_request(request)
        issued_at = request.issued_at
        assert issued_at is not None
        _require_aware(model_version.created_at, "model_version.created_at")
        _require_aware(processing_version.created_at, "processing_version.created_at")
        if model_version.created_at > issued_at:
            raise HistoryLeakageError(
                "model version was created after forecast issuance"
            )
        if processing_version.created_at > issued_at:
            raise HistoryLeakageError(
                "processing version was created after forecast issuance"
            )

        for observation in history:
            _validate_regional_observation(observation)
            if observation.observed_at > request.origin_at:
                raise HistoryLeakageError(
                    "history includes an observation after the forecast origin"
                )
            if (
                request.mode is ForecastMode.LIVE
                and observation.produced_at > issued_at
            ):
                raise HistoryLeakageError(
                    "live history includes a regional observation produced after "
                    "forecast issuance"
                )
            if observation.region_version_id != request.region_version_id:
                raise ForecastingError(
                    "history region does not match the forecast request"
                )
            if (
                observation.processing_version_id
                != processing_version.processing_version_id
            ):
                raise ForecastingError(
                    "history processing version does not match the forecast"
                )

        source = self._forecaster.select_source_observation(
            history=history,
            model_version=model_version,
            target_metric=request.target_metric,
        )
        prediction = _target_value(source, request.target_metric)
        forecast_id = self._forecast_id_factory(
            request=request,
            source_observation=source,
            model_version=model_version,
            processing_version=processing_version,
        )
        if not forecast_id:
            raise ForecastingError("forecast_id_factory returned an empty ID")

        return Forecast(
            forecast_id=forecast_id,
            origin_at=request.origin_at,
            issued_at=issued_at,
            valid_at=request.valid_at,
            source_provider=request.source_provider,
            source_product=request.source_product,
            region_version_id=request.region_version_id,
            model_version_id=model_version.model_version_id,
            processing_version_id=processing_version.processing_version_id,
            target_metric=request.target_metric,
            predicted_vtec_tecu=prediction,
            mode=request.mode,
            status=ForecastStatus.PENDING,
            input_observation_ids=(source.observation_id,),
        )


def _validate_request(request: ForecastRequest) -> None:
    _require_aware(request.origin_at, "request.origin_at")
    if request.issued_at is None:
        raise ForecastingError("request.issued_at must be set before model execution")
    _require_aware(request.issued_at, "request.issued_at")
    _require_aware(request.valid_at, "request.valid_at")
    if request.valid_at <= request.origin_at:
        raise ForecastingError("valid_at must be later than origin_at")
    if request.issued_at < request.origin_at:
        raise ForecastingError("issued_at cannot precede origin_at")
    if not isinstance(request.mode, ForecastMode):
        raise ForecastingError("mode must be a ForecastMode")
    if request.mode is ForecastMode.LIVE and request.issued_at >= request.valid_at:
        raise ForecastingError("a live forecast must be issued before valid_at")
    if (
        not isinstance(request.source_provider, str)
        or not request.source_provider.strip()
    ):
        raise ForecastingError("source_provider must not be empty")
    if (
        not isinstance(request.source_product, str)
        or not request.source_product.strip()
    ):
        raise ForecastingError("source_product must not be empty")
    if not request.region_version_id:
        raise ForecastingError("region_version_id must not be empty")
    if not isinstance(request.target_metric, TargetMetric):
        raise ForecastingError("target_metric must be a TargetMetric")


def _validate_regional_observation(observation: RegionalObservation) -> None:
    _require_aware(observation.observed_at, "observation.observed_at")
    _require_aware(observation.produced_at, "observation.produced_at")
    if not observation.observation_id:
        raise ForecastingError("history observation_id must not be empty")
    if not observation.region_version_id:
        raise ForecastingError("history region_version_id must not be empty")
    if not 0.0 <= observation.coverage_fraction <= 1.0:
        raise ForecastingError("history coverage_fraction must be in [0, 1]")
    if any(not isinstance(flag, str) or not flag for flag in observation.quality_flags):
        raise ForecastingError("history quality flags must be non-empty strings")


def _target_value(
    observation: RegionalObservation, target_metric: TargetMetric
) -> float:
    if target_metric is TargetMetric.MEAN_VTEC:
        return observation.mean_vtec_tecu
    if target_metric is TargetMetric.MEDIAN_VTEC:
        return observation.median_vtec_tecu
    raise ForecastingError(f"unsupported target metric: {target_metric!r}")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ForecastingError(f"{field_name} must be timezone-aware")


def _default_forecast_id(
    *,
    request: ForecastRequest,
    source_observation: RegionalObservation,
    model_version: ModelVersion,
    processing_version: ProcessingVersion,
) -> str:
    if request.issued_at is None:
        raise ForecastingError("cannot identify a forecast before issuance")
    payload = "\x1f".join(
        (
            request.issued_at.astimezone(timezone.utc).isoformat(),
            request.origin_at.astimezone(timezone.utc).isoformat(),
            request.valid_at.astimezone(timezone.utc).isoformat(),
            request.source_provider,
            request.source_product,
            request.region_version_id,
            request.target_metric.value,
            request.mode.value,
            source_observation.observation_id,
            model_version.model_version_id,
            processing_version.processing_version_id,
        )
    ).encode("utf-8")
    return f"forecast-{sha256(payload).hexdigest()[:24]}"
