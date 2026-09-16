"""Shared domain records for the OPHANIM pipeline."""

from ophanim.domain.entities import (
    AnomalyDecision,
    DetectorVersion,
    Forecast,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    SourceArtifact,
    TECObservation,
)
from ophanim.domain.enums import (
    DisturbanceAssessment,
    ForecastMode,
    ForecastStatus,
    TargetMetric,
)

__all__ = [
    "AnomalyDecision",
    "DetectorVersion",
    "DisturbanceAssessment",
    "Forecast",
    "ForecastMode",
    "ForecastStatus",
    "ModelVersion",
    "ProcessingVersion",
    "RegionVersion",
    "RegionalObservation",
    "SourceArtifact",
    "TECObservation",
    "TargetMetric",
]
