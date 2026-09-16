"""Application workflows composing processors with durable repositories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from ophanim.aggregation import RegionalAggregator
from ophanim.domain import (
    AnomalyDecision,
    DetectorVersion,
    Forecast,
    ForecastMode,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    TargetMetric,
)
from ophanim.forecasting import ForecastRequest, ForecastingError, ModelRunner
from ophanim.ingestion import DataIngester, IngestionRequest, IngestionResult
from ophanim.reconciliation import PendingForecastReconciler, ReconciliationSummary
from ophanim.repositories import UnitOfWork


@dataclass(frozen=True, slots=True)
class AggregationSummary:
    artifact_id: str
    epochs_seen: int
    observations_created: int
    observations_reused: int


class RegionalAggregationWorkflow:
    """Aggregate every map epoch in one ingested artifact, idempotently."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        aggregator: RegionalAggregator,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._aggregator = aggregator
        self._clock = clock or (lambda: datetime.now(UTC))

    def aggregate_artifact(
        self,
        *,
        artifact_id: str,
        region: RegionVersion,
        processing_version: ProcessingVersion,
    ) -> AggregationSummary:
        with self._unit_of_work_factory() as unit_of_work:
            # Capture production time only after the transaction lock is held.
            # If ingestion is concurrent, this timestamp therefore follows the
            # source commit rather than the time spent waiting for it.
            derivation_started_at = self._clock()
            if (
                derivation_started_at.tzinfo is None
                or derivation_started_at.utcoffset() is None
            ):
                raise ValueError(
                    "aggregation clock must return a timezone-aware datetime"
                )
            artifact = unit_of_work.artifacts.get(artifact_id)
            if artifact is None:
                raise LookupError(f"source artifact does not exist: {artifact_id}")
            if artifact.parser_version != processing_version.parser_version:
                raise ValueError(
                    "processing version requires parser "
                    f"{processing_version.parser_version!r}, but artifact was parsed "
                    f"with {artifact.parser_version!r}"
                )
            if derivation_started_at < artifact.ingested_at:
                raise ValueError(
                    "regional observation cannot be produced before source ingestion"
                )
            raw_observations = tuple(
                unit_of_work.tec_observations.for_artifact(artifact_id)
            )
            existing = tuple(
                unit_of_work.regional_observations.for_artifact(artifact_id)
            )

        by_epoch = defaultdict(list)
        for observation in raw_observations:
            by_epoch[observation.observed_at].append(observation)

        existing_keys = {
            (
                observation.processing_version_id,
                observation.region_version_id,
                observation.observed_at,
            )
            for observation in existing
        }
        created_observations: list[RegionalObservation] = []
        reused = 0
        for observed_at in sorted(by_epoch):
            key = (
                processing_version.processing_version_id,
                region.region_version_id,
                observed_at,
            )
            if key in existing_keys:
                reused += 1
                continue
            created_observations.append(
                self._aggregator.aggregate(
                    observations=by_epoch[observed_at],
                    region=region,
                    processing_version=processing_version,
                    produced_at=derivation_started_at,
                )
            )

        created = 0
        with self._unit_of_work_factory() as unit_of_work:
            published_at = self._clock()
            if published_at.tzinfo is None or published_at.utcoffset() is None:
                raise ValueError(
                    "aggregation clock must return a timezone-aware datetime"
                )
            if published_at < derivation_started_at:
                raise ValueError("aggregation clock moved backwards during derivation")
            unit_of_work.regions.register(region)
            unit_of_work.processing_versions.register(processing_version)
            # Another worker may have committed between the read phase and
            # this write transaction. Recheck under the writer lock so an
            # idempotent retry does not conflict solely because its local
            # ``produced_at`` differs.
            current_by_key = {
                (
                    observation.processing_version_id,
                    observation.region_version_id,
                    observation.observed_at,
                ): observation
                for observation in unit_of_work.regional_observations.for_artifact(
                    artifact_id
                )
            }
            for provisional_observation in created_observations:
                observation = replace(
                    provisional_observation,
                    produced_at=published_at,
                )
                key = (
                    observation.processing_version_id,
                    observation.region_version_id,
                    observation.observed_at,
                )
                current = current_by_key.get(key)
                if current is not None:
                    if replace(observation, produced_at=current.produced_at) != current:
                        raise ValueError(
                            "concurrent aggregation produced conflicting scientific "
                            f"output for observation {current.observation_id!r}"
                        )
                    reused += 1
                    continue
                unit_of_work.regional_observations.add(observation)
                current_by_key[key] = observation
                created += 1
            unit_of_work.commit()

        return AggregationSummary(
            artifact_id=artifact_id,
            epochs_seen=len(by_epoch),
            observations_created=created,
            observations_reused=reused,
        )


@dataclass(frozen=True, slots=True)
class ForecastIssueResult:
    forecast: Forecast
    already_present: bool


class ForecastConflictError(RuntimeError):
    """An issuance retry produced different scientific output."""


class ForecastIssuingWorkflow:
    """Load explicit history, run a model, and atomically store its forecast.

    SQLite units of work acquire a reserved writer lock on entry.  Live runs
    are timestamped only after that lock is held, then read their inputs and
    persist the forecast in the same transaction.  Consequently every input
    visible to a live run was durably published before its recorded run-start
    cutoff, even when an ingestion or aggregation was concurrent.
    """

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        model_runner: ModelRunner,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._model_runner = model_runner
        self._clock = clock or (lambda: datetime.now(UTC))

    def issue(
        self,
        *,
        request: ForecastRequest,
        model_version: ModelVersion,
        processing_version: ProcessingVersion,
        history_limit: int | None = None,
    ) -> ForecastIssueResult:
        if request.mode is ForecastMode.LIVE and request.issued_at is not None:
            raise ValueError(
                "live issuance timestamps are workflow-owned; "
                "set request.issued_at to None"
            )
        with self._unit_of_work_factory() as unit_of_work:
            effective_request = request
            if request.mode is ForecastMode.LIVE:
                issued_at = self._clock()
                if issued_at.tzinfo is None or issued_at.utcoffset() is None:
                    raise ValueError(
                        "forecast issuance clock must return a timezone-aware datetime"
                    )
                effective_request = replace(request, issued_at=issued_at)
            elif request.issued_at is None:
                issued_at = self._clock()
                if issued_at.tzinfo is None or issued_at.utcoffset() is None:
                    raise ValueError(
                        "forecast issuance clock must return a timezone-aware datetime"
                    )
                effective_request = replace(request, issued_at=issued_at)
            if effective_request.issued_at is None:
                raise RuntimeError("forecast issuance timestamp was not resolved")

            region = unit_of_work.regions.get(effective_request.region_version_id)
            if region is None:
                raise LookupError(
                    "region version does not exist: "
                    f"{effective_request.region_version_id}"
                )
            if region.created_at > effective_request.issued_at:
                raise ValueError("region version was created after forecast issuance")
            history = tuple(
                unit_of_work.regional_observations.history(
                    source_provider=effective_request.source_provider,
                    source_product=effective_request.source_product,
                    region_version_id=effective_request.region_version_id,
                    processing_version_id=processing_version.processing_version_id,
                    observed_at_or_before=effective_request.origin_at,
                    available_at_or_before=(
                        effective_request.issued_at
                        if effective_request.mode is ForecastMode.LIVE
                        else None
                    ),
                    limit=history_limit,
                )
            )
            forecast = self._model_runner.run(
                request=effective_request,
                history=history,
                model_version=model_version,
                processing_version=processing_version,
            )
            if effective_request.mode is ForecastMode.LIVE:
                completed_at = self._clock()
                if (
                    completed_at.tzinfo is None
                    or completed_at.utcoffset() is None
                ):
                    raise ValueError(
                        "forecast issuance clock must return a timezone-aware datetime"
                    )
                if completed_at < effective_request.issued_at:
                    raise ValueError("forecast issuance clock moved backwards")
                if completed_at >= effective_request.valid_at:
                    raise ForecastingError(
                        "live forecast computation did not finish before valid_at"
                    )
            unit_of_work.models.register(model_version)
            unit_of_work.processing_versions.register(processing_version)
            existing = unit_of_work.forecasts.find_equivalent(forecast)
            if existing is not None:
                comparable = replace(
                    forecast,
                    forecast_id=existing.forecast_id,
                    status=existing.status,
                )
                if comparable != existing:
                    raise ForecastConflictError(
                        "an equivalent forecast exists with different output"
                    )
                return ForecastIssueResult(
                    forecast=existing,
                    already_present=True,
                )
            unit_of_work.forecasts.add(forecast)
            unit_of_work.commit()

        return ForecastIssueResult(forecast=forecast, already_present=False)


@dataclass(frozen=True, slots=True)
class VerticalSliceRequest:
    ingestion: IngestionRequest
    region: RegionVersion
    processing_version: ProcessingVersion
    model_version: ModelVersion
    detector_version: DetectorVersion
    forecast_origin: datetime
    forecast_horizon: timedelta = timedelta(hours=2)
    target_metric: TargetMetric = TargetMetric.MEDIAN_VTEC
    mode: ForecastMode = ForecastMode.HINDCAST
    history_limit: int | None = None


@dataclass(frozen=True, slots=True)
class VerticalSliceResult:
    ingestion: IngestionResult
    aggregation: AggregationSummary
    forecast: ForecastIssueResult
    reconciliation: ReconciliationSummary
    decisions: tuple[AnomalyDecision, ...]


class VerticalSliceWorkflow:
    """Run one IONEX artifact through the complete experimental v0 path."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        ingester: DataIngester,
        aggregation: RegionalAggregationWorkflow,
        forecasting: ForecastIssuingWorkflow,
        reconciler: PendingForecastReconciler,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._ingester = ingester
        self._aggregation = aggregation
        self._forecasting = forecasting
        self._reconciler = reconciler

    def run(self, request: VerticalSliceRequest) -> VerticalSliceResult:
        reconciler_detector_id = self._reconciler.detector_version_id
        if reconciler_detector_id != request.detector_version.detector_version_id:
            raise ValueError(
                "vertical-slice detector does not match the reconciler configuration"
            )

        ingestion = self._ingester.ingest(request.ingestion)
        aggregation = self._aggregation.aggregate_artifact(
            artifact_id=ingestion.artifact.artifact_id,
            region=request.region,
            processing_version=request.processing_version,
        )

        with self._unit_of_work_factory() as unit_of_work:
            unit_of_work.detectors.register(request.detector_version)
            unit_of_work.commit()

        forecast_request = ForecastRequest(
            origin_at=request.forecast_origin,
            valid_at=request.forecast_origin + request.forecast_horizon,
            source_provider=request.ingestion.provider,
            source_product=request.ingestion.product,
            region_version_id=request.region.region_version_id,
            target_metric=request.target_metric,
            mode=request.mode,
        )
        forecast = self._forecasting.issue(
            request=forecast_request,
            model_version=request.model_version,
            processing_version=request.processing_version,
            history_limit=request.history_limit,
        )
        reconciliation = self._reconciler.reconcile(
            valid_at_or_before=forecast.forecast.valid_at,
        )

        with self._unit_of_work_factory() as unit_of_work:
            stored_forecast = unit_of_work.forecasts.get(
                forecast.forecast.forecast_id
            )
            if stored_forecast is None:
                raise RuntimeError("stored forecast disappeared during reconciliation")
            decisions = tuple(
                unit_of_work.decisions.for_forecast(forecast.forecast.forecast_id)
            )
        forecast = replace(forecast, forecast=stored_forecast)

        return VerticalSliceResult(
            ingestion=ingestion,
            aggregation=aggregation,
            forecast=forecast,
            reconciliation=reconciliation,
            decisions=decisions,
        )
