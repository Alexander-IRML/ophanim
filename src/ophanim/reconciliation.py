"""Pending-forecast resolution boundary.

Live observation arrivals, scheduled sweeps, and backfills all enter this same
path so scoring and persistence semantics do not diverge.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from typing import Protocol

from ophanim.detection import (
    AnomalyDecisionEngine,
    InsufficientObservationDataError,
)
from ophanim.domain import AnomalyDecision, ForecastStatus
from ophanim.repositories import UnitOfWork


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconciliationSummary:
    """Outcome counts for one reconciliation pass."""

    considered: int
    scored: int
    insufficient_data: int
    retryable_errors: int


class PendingForecastReconciler(Protocol):
    """Resolve eligible pending forecasts through the shared scoring path."""

    @property
    def detector_version_id(self) -> str: ...

    def reconcile(self, *, valid_at_or_before: datetime) -> ReconciliationSummary: ...

    def reconcile_forecast(
        self,
        *,
        forecast_id: str,
        valid_at_or_before: datetime,
    ) -> ReconciliationSummary: ...

    def reassess_scored(self, *, forecast_id: str) -> AnomalyDecision | None: ...

    def retry_insufficient(self, *, forecast_id: str) -> AnomalyDecision | None: ...


class DefaultPendingForecastReconciler:
    """Resolve each eligible pending forecast in its own transaction.

    A missing observation remains pending until ``observation_grace_period``
    elapses. Live arrivals, scheduled sweeps, and backfills should all call this
    same method.
    """

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        decision_engine: AnomalyDecisionEngine,
        detector_version_id: str,
        observation_grace_period_seconds: float = 6 * 60 * 60,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            isinstance(observation_grace_period_seconds, bool)
            or not isinstance(observation_grace_period_seconds, (int, float))
            or not isfinite(observation_grace_period_seconds)
            or observation_grace_period_seconds < 0
        ):
            raise ValueError(
                "observation_grace_period_seconds must be a finite "
                "non-negative number"
            )
        if (
            not isinstance(detector_version_id, str)
            or not detector_version_id.strip()
        ):
            raise ValueError("detector_version_id must not be empty")
        self._unit_of_work_factory = unit_of_work_factory
        self._decision_engine = decision_engine
        self._detector_version_id = detector_version_id
        self._observation_grace_period_seconds = observation_grace_period_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def detector_version_id(self) -> str:
        return self._detector_version_id

    def reconcile(self, *, valid_at_or_before: datetime) -> ReconciliationSummary:
        with self._unit_of_work_factory() as unit_of_work:
            pending = tuple(
                unit_of_work.forecasts.list_pending(
                    valid_at_or_before=valid_at_or_before,
                )
            )

        scored = 0
        insufficient_data = 0
        retryable_errors = 0
        for forecast in pending:
            try:
                outcome = self._reconcile_one(
                    forecast_id=forecast.forecast_id,
                )
            except Exception:
                _LOGGER.exception(
                    "forecast reconciliation failed",
                    extra={"forecast_id": forecast.forecast_id},
                )
                # Operational and configuration failures remain pending so a
                # corrected configuration or later sweep can retry them. A
                # future attempt log can make this durable without destroying
                # the recoverable state here.
                retryable_errors += 1
                continue

            if outcome is ForecastStatus.SCORED:
                scored += 1
            elif outcome is ForecastStatus.INSUFFICIENT_DATA:
                insufficient_data += 1

        return ReconciliationSummary(
            considered=len(pending),
            scored=scored,
            insufficient_data=insufficient_data,
            retryable_errors=retryable_errors,
        )

    def reconcile_forecast(
        self,
        *,
        forecast_id: str,
        valid_at_or_before: datetime,
    ) -> ReconciliationSummary:
        """Resolve one eligible pending forecast with this reconciler's detector.

        The cutoff preserves the eligibility rule used by the bulk sweep. An
        existing forecast that is not pending or is not yet valid is ignored;
        an unknown ID remains a caller error. Operational and configuration
        failures retain the pending state and are reported as retryable.
        """

        with self._unit_of_work_factory() as unit_of_work:
            forecast = unit_of_work.forecasts.get(forecast_id)
            if forecast is None:
                raise LookupError(f"forecast does not exist: {forecast_id}")
            if (
                forecast.status is not ForecastStatus.PENDING
                or forecast.valid_at > valid_at_or_before
            ):
                return ReconciliationSummary(
                    considered=0,
                    scored=0,
                    insufficient_data=0,
                    retryable_errors=0,
                )

        try:
            outcome = self._reconcile_one(forecast_id=forecast_id)
        except Exception:
            _LOGGER.exception(
                "forecast reconciliation failed",
                extra={"forecast_id": forecast_id},
            )
            return ReconciliationSummary(
                considered=1,
                scored=0,
                insufficient_data=0,
                retryable_errors=1,
            )

        return ReconciliationSummary(
            considered=1,
            scored=int(outcome is ForecastStatus.SCORED),
            insufficient_data=int(
                outcome is ForecastStatus.INSUFFICIENT_DATA
            ),
            retryable_errors=0,
        )

    def _reconcile_one(
        self,
        *,
        forecast_id: str,
    ) -> ForecastStatus | None:
        with self._unit_of_work_factory() as unit_of_work:
            # Stamp after acquiring the transaction lock. Any observation now
            # visible was committed before this scoring attempt began.
            reconciled_at = self._clock()
            forecast = unit_of_work.forecasts.get(forecast_id)
            if forecast is None:
                raise LookupError(f"forecast does not exist: {forecast_id}")
            if forecast.status is not ForecastStatus.PENDING:
                # Another sweep may have transitioned the forecast after this
                # one took its pending snapshot. It was considered here, but
                # no work performed by this sweep should be counted.
                return None

            observation = unit_of_work.regional_observations.find(
                source_provider=forecast.source_provider,
                source_product=forecast.source_product,
                region_version_id=forecast.region_version_id,
                processing_version_id=forecast.processing_version_id,
                observed_at=forecast.valid_at,
                available_at_or_before=reconciled_at,
            )
            if observation is None:
                elapsed_seconds = (reconciled_at - forecast.valid_at).total_seconds()
                if elapsed_seconds < self._observation_grace_period_seconds:
                    return ForecastStatus.PENDING
                unit_of_work.forecasts.replace(
                    replace(forecast, status=ForecastStatus.INSUFFICIENT_DATA)
                )
                unit_of_work.commit()
                return ForecastStatus.INSUFFICIENT_DATA

            detector_version = unit_of_work.detectors.get(self._detector_version_id)
            if detector_version is None:
                raise LookupError(
                    f"detector version does not exist: {self._detector_version_id}"
                )
            try:
                decision = self._decision_engine.assess(
                    forecast=forecast,
                    observation=observation,
                    detector_version=detector_version,
                    scored_at=reconciled_at,
                )
            except InsufficientObservationDataError:
                elapsed_seconds = (
                    reconciled_at - forecast.valid_at
                ).total_seconds()
                if elapsed_seconds < self._observation_grace_period_seconds:
                    return ForecastStatus.PENDING
                unit_of_work.forecasts.replace(
                    replace(forecast, status=ForecastStatus.INSUFFICIENT_DATA)
                )
                unit_of_work.commit()
                return ForecastStatus.INSUFFICIENT_DATA
            unit_of_work.decisions.add(decision)
            unit_of_work.forecasts.replace(
                replace(forecast, status=ForecastStatus.SCORED)
            )
            unit_of_work.commit()
            return ForecastStatus.SCORED

    def reassess_scored(self, *, forecast_id: str) -> AnomalyDecision | None:
        """Score a newer actual revision and supersede the prior detector result."""

        with self._unit_of_work_factory() as unit_of_work:
            scored_at = self._clock()
            forecast = unit_of_work.forecasts.get(forecast_id)
            if forecast is None:
                raise LookupError(f"forecast does not exist: {forecast_id}")
            if forecast.status is not ForecastStatus.SCORED:
                raise ValueError("only a scored forecast can be reassessed")

            observation = unit_of_work.regional_observations.find(
                source_provider=forecast.source_provider,
                source_product=forecast.source_product,
                region_version_id=forecast.region_version_id,
                processing_version_id=forecast.processing_version_id,
                observed_at=forecast.valid_at,
                available_at_or_before=scored_at,
            )
            if observation is None:
                raise LookupError("no actual observation exists for reassessment")
            detector_version = unit_of_work.detectors.get(self._detector_version_id)
            if detector_version is None:
                raise LookupError(
                    f"detector version does not exist: {self._detector_version_id}"
                )

            prior = tuple(
                decision
                for decision in unit_of_work.decisions.for_forecast(forecast_id)
                if decision.detector_version_id == self._detector_version_id
            )
            previous = prior[-1] if prior else None
            if previous is not None and previous.observation_id == observation.observation_id:
                return None

            decision = self._decision_engine.assess(
                forecast=forecast,
                observation=observation,
                detector_version=detector_version,
                scored_at=scored_at,
            )
            if previous is not None:
                decision = replace(
                    decision,
                    supersedes_decision_id=previous.decision_id,
                )
            unit_of_work.decisions.add(decision)
            unit_of_work.commit()
            return decision

    def retry_insufficient(self, *, forecast_id: str) -> AnomalyDecision | None:
        """Score an insufficient forecast when a usable actual arrives later.

        Returning ``None`` means that no usable actual is available at the retry
        time. The forecast remains ``INSUFFICIENT_DATA`` and can be retried again.
        Repeating a successful retry returns the already-current decision.
        """

        with self._unit_of_work_factory() as unit_of_work:
            scored_at = self._clock()
            forecast = unit_of_work.forecasts.get(forecast_id)
            if forecast is None:
                raise LookupError(f"forecast does not exist: {forecast_id}")
            if forecast.status is ForecastStatus.SCORED:
                decisions = tuple(
                    decision
                    for decision in unit_of_work.decisions.for_forecast(forecast_id)
                    if decision.detector_version_id == self._detector_version_id
                )
                if not decisions:
                    raise RuntimeError(
                        "scored forecast has no decision for the configured detector"
                    )
                return decisions[-1]
            if forecast.status is not ForecastStatus.INSUFFICIENT_DATA:
                raise ValueError(
                    "only an insufficient-data forecast can be retried"
                )

            observation = unit_of_work.regional_observations.find(
                source_provider=forecast.source_provider,
                source_product=forecast.source_product,
                region_version_id=forecast.region_version_id,
                processing_version_id=forecast.processing_version_id,
                observed_at=forecast.valid_at,
                available_at_or_before=scored_at,
            )
            if observation is None:
                return None
            detector_version = unit_of_work.detectors.get(self._detector_version_id)
            if detector_version is None:
                raise LookupError(
                    f"detector version does not exist: {self._detector_version_id}"
                )

            try:
                decision = self._decision_engine.assess(
                    forecast=replace(forecast, status=ForecastStatus.PENDING),
                    observation=observation,
                    detector_version=detector_version,
                    scored_at=scored_at,
                )
            except InsufficientObservationDataError:
                return None

            # Both writes share one transaction. The temporary SCORED state is
            # therefore never externally visible without its decision.
            unit_of_work.forecasts.replace(
                replace(forecast, status=ForecastStatus.SCORED)
            )
            unit_of_work.decisions.add(decision)
            unit_of_work.commit()
            return decision
