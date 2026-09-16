"""Composition root for the dependency-free SQLite v0 application."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from ophanim.aggregation import (
    AggregationPolicy,
    BoundsRegionSelector,
    GeographicBounds,
    MeanMedianRegionalAggregator,
)
from ophanim.artifacts import FilesystemArtifactStore
from ophanim.detection import ResidualZScoreDecisionEngine
from ophanim.forecasting import PersistenceModelRunner
from ophanim.ingestion import IONEXDataIngester, StandardSourceLoader
from ophanim.ionex import IONEXV1Parser
from ophanim.reconciliation import DefaultPendingForecastReconciler
from ophanim.sqlite import SQLiteUnitOfWork
from ophanim.workflows import (
    ForecastIssuingWorkflow,
    RegionalAggregationWorkflow,
    VerticalSliceWorkflow,
)


@dataclass(frozen=True, slots=True)
class OphanimApplication:
    """Ready-to-call application workflows sharing one SQLite database."""

    unit_of_work_factory: Callable[[], SQLiteUnitOfWork]
    ingester: IONEXDataIngester
    aggregation: RegionalAggregationWorkflow
    forecasting: ForecastIssuingWorkflow
    reconciler: DefaultPendingForecastReconciler
    vertical_slice: VerticalSliceWorkflow


def create_sqlite_application(
    *,
    database: str | Path,
    artifact_directory: str | Path,
    region_bounds: Mapping[str, GeographicBounds],
    aggregation_policy: AggregationPolicy,
    detector_version_id: str,
    clock: Callable[[], datetime] | None = None,
    observation_grace_period_seconds: float = 6 * 60 * 60,
) -> OphanimApplication:
    """Wire concrete v0 components without introducing a service container."""

    database_path = Path(database).expanduser()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    unit_of_work_factory = partial(SQLiteUnitOfWork, database_path)
    ingester = IONEXDataIngester(
        source_loader=StandardSourceLoader(),
        artifact_store=FilesystemArtifactStore(artifact_directory),
        parser=IONEXV1Parser(),
        unit_of_work_factory=unit_of_work_factory,
        clock=clock,
    )
    aggregator = MeanMedianRegionalAggregator(
        selection_strategy=BoundsRegionSelector(region_bounds),
        policy=aggregation_policy,
    )
    aggregation = RegionalAggregationWorkflow(
        unit_of_work_factory=unit_of_work_factory,
        aggregator=aggregator,
        clock=clock,
    )
    forecasting = ForecastIssuingWorkflow(
        unit_of_work_factory=unit_of_work_factory,
        model_runner=PersistenceModelRunner(),
        clock=clock,
    )
    reconciler = DefaultPendingForecastReconciler(
        unit_of_work_factory=unit_of_work_factory,
        decision_engine=ResidualZScoreDecisionEngine(),
        detector_version_id=detector_version_id,
        observation_grace_period_seconds=observation_grace_period_seconds,
        clock=clock,
    )
    vertical_slice = VerticalSliceWorkflow(
        unit_of_work_factory=unit_of_work_factory,
        ingester=ingester,
        aggregation=aggregation,
        forecasting=forecasting,
        reconciler=reconciler,
    )
    return OphanimApplication(
        unit_of_work_factory=unit_of_work_factory,
        ingester=ingester,
        aggregation=aggregation,
        forecasting=forecasting,
        reconciler=reconciler,
        vertical_slice=vertical_slice,
    )


__all__ = ["OphanimApplication", "create_sqlite_application"]
