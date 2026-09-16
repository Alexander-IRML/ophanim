"""Lifecycle and classification values shared across module boundaries."""

from enum import Enum


class TargetMetric(str, Enum):
    """The regional value a forecast targets."""

    MEAN_VTEC = "mean_vtec"
    MEDIAN_VTEC = "median_vtec"


class ForecastStatus(str, Enum):
    """Lifecycle state of a forecast awaiting or completing reconciliation."""

    PENDING = "pending"
    SCORED = "scored"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


class ForecastMode(str, Enum):
    """Whether source availability was enforced at forecast execution time."""

    LIVE = "live"
    HINDCAST = "hindcast"


class DisturbanceAssessment(str, Enum):
    """Current assessment of a physical ionospheric disturbance."""

    NORMAL = "normal"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
