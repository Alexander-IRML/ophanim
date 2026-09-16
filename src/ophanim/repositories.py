"""Contracts for all durable state owned by OPHANIM.

The initial implementation may back every contract with one relational
database. Splitting interfaces here does not imply separate deployments or
physical databases.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from types import TracebackType
from typing import Protocol, Self

from ophanim.domain import (
    AnomalyDecision,
    DetectorVersion,
    DisturbanceAssessment,
    Forecast,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    SourceArtifact,
    TECObservation,
)


class SourceArtifactRepository(Protocol):
    def add(self, artifact: SourceArtifact) -> None: ...

    def get(self, artifact_id: str) -> SourceArtifact | None: ...

    def find_by_checksum(self, checksum_sha256: str) -> SourceArtifact | None: ...

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
    ) -> SourceArtifact | None: ...


class TECObservationRepository(Protocol):
    def add_many(self, observations: Iterable[TECObservation]) -> None: ...

    def for_artifact(self, artifact_id: str) -> Sequence[TECObservation]: ...

    def for_artifact_at(
        self,
        artifact_id: str,
        observed_at: datetime,
    ) -> Sequence[TECObservation]: ...


class RegionalObservationRepository(Protocol):
    def add(self, observation: RegionalObservation) -> None: ...

    def get(self, observation_id: str) -> RegionalObservation | None: ...

    def find(
        self,
        *,
        source_provider: str,
        source_product: str,
        region_version_id: str,
        processing_version_id: str,
        observed_at: datetime,
        available_at_or_before: datetime | None = None,
    ) -> RegionalObservation | None: ...

    def for_artifact(self, artifact_id: str) -> Sequence[RegionalObservation]: ...

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
        """Return canonical observations through the cutoff, chronologically.

        When ``limit`` is supplied, return the most recent ``limit`` records
        while retaining chronological order in the result.
        """

        ...


class ProcessingVersionRegistry(Protocol):
    def register(self, processing_version: ProcessingVersion) -> None: ...

    def get(self, processing_version_id: str) -> ProcessingVersion | None: ...


class RegionRegistry(Protocol):
    def register(self, region_version: RegionVersion) -> None: ...

    def get(self, region_version_id: str) -> RegionVersion | None: ...


class ModelRegistry(Protocol):
    def register(self, model_version: ModelVersion) -> None: ...

    def get(self, model_version_id: str) -> ModelVersion | None: ...


class DetectorRegistry(Protocol):
    def register(self, detector_version: DetectorVersion) -> None: ...

    def get(self, detector_version_id: str) -> DetectorVersion | None: ...


class ForecastRepository(Protocol):
    def add(self, forecast: Forecast) -> None: ...

    def get(self, forecast_id: str) -> Forecast | None: ...

    def find_equivalent(self, forecast: Forecast) -> Forecast | None:
        """Find a forecast with the same issuance identity, if one exists."""

        ...

    def list_pending(self, *, valid_at_or_before: datetime) -> Sequence[Forecast]: ...

    def replace(self, forecast: Forecast) -> None: ...


class DecisionRepository(Protocol):
    def add(self, decision: AnomalyDecision) -> None: ...

    def for_forecast(self, forecast_id: str) -> Sequence[AnomalyDecision]: ...

    def search(
        self,
        *,
        valid_at_or_after: datetime,
        valid_before: datetime,
        region_version_id: str | None = None,
        is_forecast_anomaly: bool | None = None,
        disturbance_assessment: DisturbanceAssessment | None = None,
        include_superseded: bool = False,
    ) -> Sequence[AnomalyDecision]: ...


class UnitOfWork(Protocol):
    """Transaction boundary shared by ingestion, forecasting, and scoring."""

    artifacts: SourceArtifactRepository
    tec_observations: TECObservationRepository
    processing_versions: ProcessingVersionRegistry
    regions: RegionRegistry
    regional_observations: RegionalObservationRepository
    models: ModelRegistry
    detectors: DetectorRegistry
    forecasts: ForecastRepository
    decisions: DecisionRepository

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...
