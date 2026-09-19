"""Dependency-free contracts for interchangeable scientific model adapters.

Implementations may wrap existing regional/spatial baselines or novel models.
This interface does not imply a model has been trained or calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .state import DynamicState


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Immutable identity for an existing dataset, checkpoint, or report.

    The reference is descriptive and is never automatically opened or fetched.
    Readers must verify the checksum using their own allowed storage backend.
    """

    reference: str
    sha256: str
    kind: str = "dataset"

    def __post_init__(self):
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("artifact reference must be non-empty")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("artifact kind must be non-empty")


@dataclass(frozen=True, slots=True)
class ModelCapability:
    available: bool
    reason: str | None = None
    output_semantics: str = "experimental inference, not a causal diagnosis"


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """A canonical scientific state with checkpoint and input provenance.

    Opaque latent tensors are not valid predictions at this boundary. Their
    producer must explicitly decode and describe them through a state adapter.
    """

    model_id: str
    model_version: str
    input_artifacts: tuple[ArtifactReference, ...]
    output: DynamicState
    model_artifact: ArtifactReference | None = None
    output_semantics: str = "experimental inference, not a causal diagnosis"

    def __post_init__(self):
        from .state import DynamicState
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.model_id, self.model_version, self.output_semantics)):
            raise ValueError("model identity, version and output semantics must be explicit")
        if not isinstance(self.input_artifacts, tuple) or not self.input_artifacts or not all(
                isinstance(item, ArtifactReference) for item in self.input_artifacts):
            raise ValueError("model prediction requires frozen input artifact references")
        if self.model_artifact is not None and not isinstance(self.model_artifact, ArtifactReference):
            raise ValueError("model checkpoint must be an ArtifactReference")
        if not isinstance(self.output, DynamicState):
            raise ValueError("model output must be a validated canonical DynamicState, not an opaque tensor")
        self.output.validate()


@runtime_checkable
class ModelAdapter(Protocol):
    """Inference seam shared by core workflows and independent experiments.

    Training deliberately is not required for all adapters: a deterministic
    baseline may have no checkpoint. A learned adapter must fail explicitly if
    its required checkpoint, calibration, or source support is unavailable.
    """

    model_id: str
    model_version: str

    def availability(self) -> ModelCapability: ...

    def predict(self, dataset: Any, *, input_artifacts: tuple[ArtifactReference, ...],
                model_artifact: ArtifactReference | None = None) -> ModelPrediction: ...


class EngineeredModelAdapter:
    """Concrete deterministic baseline implementing the same inference boundary.

    This runs existing scientific estimators, not a learned transition model.
    Its configuration and returned state remain inspectable and reusable by a
    later benchmark runner without importing experiments or artistic modules.
    """

    model_id = "engineered-tec-dynamics"
    model_version = "1"

    def __init__(self, config=None):
        self.config = config

    def availability(self) -> ModelCapability:
        from importlib.util import find_spec
        missing = [name for name in ("numpy", "xarray", "scipy", "skimage", "pyproj") if find_spec(name) is None]
        return ModelCapability(not missing, "Missing optional science dependencies: " + ", ".join(missing) if missing else None,
                               "Deterministic engineered dynamic state; apparent motion is not plasma velocity")

    def predict(self, dataset: Any, *, input_artifacts: tuple[ArtifactReference, ...],
                model_artifact: ArtifactReference | None = None) -> ModelPrediction:
        if model_artifact is not None:
            raise ValueError("engineered baseline has no learned checkpoint")
        capability = self.availability()
        if not capability.available:
            raise RuntimeError(capability.reason)
        from ophanim.dynamics import AnalysisConfig, analyze_dataset
        from .state import EngineeredStateAdapter
        config = self.config
        if config is None:
            config = AnalysisConfig()
        elif isinstance(config, dict):
            config = AnalysisConfig.from_dict(config)
        if not isinstance(config, AnalysisConfig):
            raise ValueError("engineered model config must be AnalysisConfig or its serialized mapping")
        result = analyze_dataset(dataset, config)
        state = EngineeredStateAdapter().from_result(result)
        return ModelPrediction(self.model_id, self.model_version, input_artifacts, state,
                               output_semantics=capability.output_semantics)
