"""Reproducible experiment and future agent-training contracts, without a trainer.

Time intervals are half-open UTC intervals. Artifact references identify frozen
partitions, not a mutable query. These structural checks prevent obvious leakage;
they cannot certify that an external dataset or model obeys its declared bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Protocol, runtime_checkable

from ophanim.core.models import ArtifactReference, ModelAdapter, ModelCapability


def _utc(value, field):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field} must be an aware UTC timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be an aware UTC timestamp")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    role: str
    start_at: datetime
    end_at: datetime
    artifacts: tuple[ArtifactReference, ...]

    def __post_init__(self):
        if self.role not in {"train", "validation", "test"}:
            raise ValueError("split role must be train, validation or test")
        object.__setattr__(self, "start_at", _utc(self.start_at, "split start"))
        object.__setattr__(self, "end_at", _utc(self.end_at, "split end"))
        if self.start_at >= self.end_at:
            raise ValueError("split must have a non-empty half-open time interval")
        if not isinstance(self.artifacts, tuple) or not self.artifacts or not all(
                isinstance(item, ArtifactReference) for item in self.artifacts):
            raise ValueError("split artifacts must be a non-empty tuple of frozen references")
        if len({item.reference for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("split contains duplicate artifact references")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    model_id: str
    seed: int
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    metrics: tuple[str, ...] = ("mae_tecu",)
    model_artifact: ArtifactReference | None = None

    def __post_init__(self):
        for name in ("experiment_id", "model_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        splits = (self.train, self.validation, self.test)
        for role, split in zip(("train", "validation", "test"), splits):
            if not isinstance(split, DatasetSplit) or split.role != role:
                raise ValueError(f"{role} requires a matching DatasetSplit")
        if self.train.end_at > self.validation.start_at or self.validation.end_at > self.test.start_at:
            raise ValueError("train/validation/test must be chronological and non-overlapping")
        references = [item.reference for split in splits for item in split.artifacts]
        if len(set(references)) != len(references):
            raise ValueError("train/validation/test cannot reuse the same partition artifact")
        hashes = [item.sha256 for split in splits for item in split.artifacts]
        if len(set(hashes)) != len(hashes):
            raise ValueError("train/validation/test cannot reuse identical partition content")
        if not isinstance(self.metrics, tuple) or not self.metrics or not all(
                isinstance(item, str) and item.strip() for item in self.metrics):
            raise ValueError("metrics must be a non-empty tuple of names")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metric names must be unique")
        if self.model_artifact is not None and not isinstance(self.model_artifact, ArtifactReference):
            raise ValueError("model_artifact must be an ArtifactReference")


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float
    unit: str
    split: str = "test"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.unit, str):
            raise ValueError("metric name and unit must be explicit")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("metric split must be declared")


@dataclass(frozen=True, slots=True)
class AgentTransition:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool = False


@runtime_checkable
class AgentEnvironment(Protocol):
    """Future reproducible environment seam; no environment is configured."""

    def reset(self, *, seed: int) -> Any: ...
    def step(self, action: Any) -> AgentTransition: ...
    def close(self) -> None: ...


@runtime_checkable
class AgentTrainer(Protocol):
    """A future trainer returns a real checkpoint reference, not a placeholder."""

    def availability(self) -> ModelCapability: ...
    def train(self, environment: AgentEnvironment, spec: ExperimentSpec) -> ArtifactReference: ...


def agent_training_capability() -> ModelCapability:
    return ModelCapability(False, "No agent environment or trainer is configured; contracts only.",
                           "future experimental training, not implemented")


__all__ = ["ArtifactReference", "ModelAdapter", "DatasetSplit", "ExperimentSpec", "MetricResult",
           "AgentTransition", "AgentEnvironment", "AgentTrainer", "agent_training_capability"]
