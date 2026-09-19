"""Versioned, producer-independent dense scientific state.

A learned model must supply an explicit decoded, spatially indexed mapping. A
latent vector is not a temperature, velocity, density field, or observation.
Importing these contracts does not import numerical or artistic dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .models import ArtifactReference

STATE_SCHEMA = "ophanim.scientific-state/1"
TIMELINE_SCHEMA = "ophanim-event-timeline/1"
SOURCE_KINDS = {"native", "synthetic", "learned"}
SEMANTIC_CLASSES = {"measured", "derived", "inferred", "synthetic"}
CONFIDENCE_FIELDS = ("observation_confidence", "anomaly_confidence", "flow_confidence",
                     "wave_confidence", "event_confidence")
REQUIRED_FIELDS = ("signal", "anomaly", "normalized_anomaly", "observation_support", *CONFIDENCE_FIELDS)
ENGINEERED_FIELDS = {
    "signal": "tec", "background": "tec_background", "anomaly": "dtec",
    "normalized_anomaly": "dtec_z", "relative_anomaly": "dtec_relative",
    "temporal_derivative": "dtec_dt", "gradient_east": "dtec_dx",
    "gradient_north": "dtec_dy", "gradient_magnitude": "grad_mag",
    "laplacian": "laplacian", "structure_orientation": "structure_orientation",
    "structure_coherence": "structure_coherence", "feature_velocity_east": "flow_u",
    "feature_velocity_north": "flow_v", "feature_speed": "flow_speed",
    "flow_divergence": "flow_divergence", "flow_vorticity": "flow_vorticity",
    "flow_strain": "flow_strain", "advection_residual": "advection_residual",
    "advection_residual_confidence": "advection_residual_confidence",
    "source_uncertainty": "source_uncertainty", "source_uncertainty_known": "source_uncertainty_known",
    "measurement_reliability": "measurement_reliability", "growth_ambiguity": "growth_ambiguity",
    "flow_measurement_reliability": "flow_measurement_reliability",
    "flow_uncertainty_known": "flow_uncertainty_known", "flow_observability": "flow_observability",
    "flow_normal_confidence": "flow_normal_confidence", "flow_normal_velocity": "flow_normal_velocity",
    "warp_confidence": "warp_confidence", "forward_backward_confidence": "forward_backward_confidence",
    "observation_support": "observation_support", "observation_confidence": "obs_confidence",
    "anomaly_confidence": "anomaly_confidence", "flow_confidence": "flow_confidence",
    "flow_interpretation_confidence": "flow_interpretation_confidence",
    "wave_confidence": "wave_confidence", "event_confidence": "event_confidence",
}


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """Explicit semantics for one decoded model field; no implicit unit conversion."""

    source: str
    unit: str
    semantic_class: str
    description: str

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.source, self.unit, self.description)):
            raise ValueError("field source, unit and description must be explicit nonempty strings")
        if self.semantic_class not in SEMANTIC_CLASSES:
            raise ValueError("invalid scientific semantic class; artistic fields are not state")


@dataclass(frozen=True, slots=True)
class DynamicState:
    """Canonical fields plus independent event/timeline and producer provenance.

    Arrays belong to this result, not to its producer. Publication verifies their
    immutable hash; the frozen wrapper alone is not an array immutability claim.
    """

    dataset: Any
    event: Mapping[str, Any]
    timeline: Mapping[str, Any] | None
    provenance: Mapping[str, Any]

    def validate(self) -> "DynamicState":
        validate_state(self)
        return self


def validate_state(state: DynamicState) -> None:
    """Reject missing units, invalid confidence, invented observations and bad axes."""
    import numpy as np
    import xarray as xr

    if not isinstance(state, DynamicState) or not isinstance(state.dataset, xr.Dataset):
        raise TypeError("state requires a DynamicState containing an xarray.Dataset")
    data = state.dataset
    if data.attrs.get("state_schema") != STATE_SCHEMA:
        raise ValueError("unsupported scientific state schema")
    source_kind = data.attrs.get("source_kind")
    if source_kind not in SOURCE_KINDS:
        raise ValueError("state source_kind must explicitly be native, synthetic or learned; attach verified source metadata before adapting")
    for axis in ("time", "y", "x"):
        if axis not in data.coords or data[axis].dims != (axis,) or data.sizes[axis] < 1:
            raise ValueError("state requires one-dimensional time/y/x coordinates")
        values = np.asarray(data[axis].values)
        if axis == "time":
            if values.dtype.kind != "M" or np.any(np.isnat(values)):
                raise ValueError("state time must contain valid UTC datetime64 coordinates")
            if data.time.attrs.get("timezone") != "UTC":
                raise ValueError("state time timezone must explicitly be UTC")
            increments = np.diff(values) > np.timedelta64(0, "ns")
        else:
            if data[axis].attrs.get("units") != "m" or not np.all(np.isfinite(values)):
                raise ValueError("state spatial axes must be finite metric coordinates in m")
            increments = np.diff(values) > 0
        if not np.all(increments):
            raise ValueError("state coordinates must be strictly increasing")
    missing = set(REQUIRED_FIELDS) - set(data.data_vars)
    if missing:
        raise ValueError(f"state is missing required fields: {sorted(missing)}")
    for name, field in data.data_vars.items():
        if not set(field.dims) <= {"time", "y", "x"}:
            raise ValueError(f"{name} has noncanonical dimensions")
        if not isinstance(field.attrs.get("units"), str) or not field.attrs["units"].strip():
            raise ValueError(f"{name} requires units")
        semantic = field.attrs.get("semantic_class")
        if semantic not in SEMANTIC_CLASSES or not field.attrs.get("semantic"):
            raise ValueError(f"{name} requires scientific semantics")
        if source_kind in {"synthetic", "learned"} and semantic == "measured":
            raise ValueError(f"{source_kind} output {name} cannot be labeled measured")
        if np.asarray(field.values).dtype.kind not in "fibu":
            raise ValueError(f"{name} must contain numerical values")
        if np.any(np.isinf(field.values)):
            raise ValueError(f"{name} contains infinity; unknown state must be NaN")
    for name in ("signal", "anomaly", "normalized_anomaly", "observation_support"):
        if data[name].dims != ("time", "y", "x"):
            raise ValueError(f"{name} requires ordered dimensions (time,y,x)")
    for name in ("observation_confidence", "anomaly_confidence", "flow_confidence"):
        if data[name].dims != ("time", "y", "x"):
            raise ValueError(f"{name} requires dense (time,y,x) support")
    if "flow_interpretation_confidence" in data:
        field = data.flow_interpretation_confidence
        if field.dims != ("time", "y", "x") or field.attrs["semantic_class"] != "inferred":
            raise ValueError("flow_interpretation_confidence requires dense (time,y,x) inferred eligibility")
    for name in ("wave_confidence", "event_confidence"):
        if data[name].dims not in ((), ("time",)):
            raise ValueError(f"{name} must be regional scalar or exact-time support")
    if data.signal.attrs["units"] != data.anomaly.attrs["units"]:
        raise ValueError("signal and anomaly must use matching declared units")
    for name in ("normalized_anomaly", "observation_support", *CONFIDENCE_FIELDS):
        if data[name].attrs["units"] != "1":
            raise ValueError(f"{name} must be dimensionless")
    bounded_fields = ("observation_support", *CONFIDENCE_FIELDS, "flow_interpretation_confidence", "measurement_reliability", "growth_ambiguity",
                      "source_uncertainty_known", "flow_measurement_reliability", "flow_uncertainty_known",
                      "advection_residual_confidence", "flow_observability", "flow_normal_confidence",
                      "warp_confidence", "forward_backward_confidence")
    for name in bounded_fields:
        if name not in data:
            continue
        if data[name].attrs["units"] != "1":
            raise ValueError(f"{name} must be dimensionless")
        finite = np.asarray(data[name].values)
        finite = finite[np.isfinite(finite)]
        if np.any((finite < 0) | (finite > 1)):
            raise ValueError(f"{name} must be between zero and one, or NaN when unavailable")
    for name in ("feature_velocity_east", "feature_velocity_north", "feature_speed", "flow_normal_velocity"):
        if name in data and data[name].attrs["units"] != "m/s":
            raise ValueError(f"{name} must be an explicitly decoded apparent velocity in m/s")
    if "source_uncertainty" in data:
        if data.source_uncertainty.attrs["units"] != data.signal.attrs["units"]:
            raise ValueError("source uncertainty must preserve the signal's declared units")
        if np.any(np.asarray(data.source_uncertainty.values) < 0):
            raise ValueError("source uncertainty cannot be negative")
    if ("feature_velocity_east" in data) != ("feature_velocity_north" in data):
        raise ValueError("feature velocity requires both east and north components")
    fixed_units = {"structure_orientation": "radian", "structure_coherence": "1",
                   "flow_divergence": "1/s", "flow_vorticity": "1/s", "flow_strain": "1/s"}
    for name, unit in fixed_units.items():
        if name in data and data[name].attrs["units"] != unit:
            raise ValueError(f"{name} must declare canonical units {unit}")
    for name in ("feature_velocity_east", "feature_velocity_north", "feature_speed", "structure_orientation",
                 "structure_coherence", "flow_divergence", "flow_vorticity", "flow_strain"):
        if name in data and data[name].dims != ("time", "y", "x"):
            raise ValueError(f"{name} must be a dense exact-epoch field")
    if not isinstance(state.event, Mapping) or not isinstance(state.provenance, Mapping):
        raise ValueError("state event and provenance must be mappings")
    if not state.provenance.get("producer"):
        raise ValueError("state requires an explicit producer")
    if state.timeline is not None:
        validate_timeline(state.timeline, data.time.values)


def validate_timeline(timeline, times=None):
    """Validate exact-time event association; no implicit nearest-time extrapolation."""
    import numpy as np
    from ophanim.dynamics.regularize import utc64

    if not isinstance(timeline, Mapping) or timeline.get("schema_version") != TIMELINE_SCHEMA:
        raise ValueError("unsupported event timeline schema")
    frames = timeline.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("event timeline requires nonempty frames")
    previous = None
    for frame in frames:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("event"), Mapping):
            raise ValueError("timeline frame requires a scientific event mapping")
        timestamp = utc64(frame.get("time"))
        if np.isnat(timestamp) or (previous is not None and timestamp <= previous):
            raise ValueError("timeline times must be strictly increasing")
        if times is not None and timestamp not in times:
            raise ValueError("timeline timestamp must match an exact scientific epoch")
        if utc64(frame["event"].get("time")) != timestamp:
            raise ValueError("timeline event must describe its exact timestamp")
        if frame.get("status") not in {"analyzed", "partial", "unavailable"}:
            raise ValueError("timeline frame requires explicit inference status")
        if not isinstance(frame.get("analysis_window"), Mapping):
            raise ValueError("timeline frame requires its analysis windows")
        previous = timestamp


class EngineeredStateAdapter:
    """Convert the current engineered TEC pipeline to a validated generic state."""

    def from_result(self, result, *, timeline=None) -> DynamicState:
        import xarray as xr
        from copy import deepcopy

        original = result.dataset
        source_kind = original.attrs.get("source_kind", "unknown")
        fields = {}
        for canonical, source in ENGINEERED_FIELDS.items():
            if source not in original:
                continue
            field = original[source].copy(deep=True)
            semantic = field.attrs.get("semantic_class", "derived")
            if source_kind == "synthetic" and semantic == "measured":
                semantic = "synthetic"
            field.attrs.update(units=field.attrs.get("units", "1"), semantic_class=semantic,
                               semantic=field.attrs.get("semantic", f"Engineered {canonical} from a declared source"),
                               source_field=source)
            fields[canonical] = field
        data = xr.Dataset(fields, attrs=deepcopy(original.attrs))
        for name in ("lat", "lon"):
            if name in original.coords:
                data = data.assign_coords({name: original.coords[name].copy(deep=True)})
        data.attrs.update(state_schema=STATE_SCHEMA, source_kind=source_kind)
        data.time.attrs.update(timezone="UTC")
        for axis in ("x", "y"):
            data[axis].attrs.update(units="m")
        return DynamicState(data, deepcopy(result.event), deepcopy(timeline),
                            {"producer": "engineered-tec-dynamics", "source_kind": source_kind,
                             "mapping": dict(ENGINEERED_FIELDS)}).validate()


class LearnedStateAdapter:
    """Validate an explicitly decoded learned output; does not train or decode.

    Callers supply the decoder's scientifically justified field mapping, units,
    semantics and real model/input references. No uncertainty or observation
    support is invented when the model omits it.
    """

    def adapt(self, decoded, *, fields: Mapping[str, FieldMapping],
              model_artifact: ArtifactReference, input_artifacts: tuple[ArtifactReference, ...],
              model_id: str, event: Mapping[str, Any], timeline=None) -> DynamicState:
        import xarray as xr
        from copy import deepcopy
        from dataclasses import asdict

        if not isinstance(decoded, xr.Dataset):
            raise TypeError("learned output must be an explicitly decoded xarray.Dataset")
        if not isinstance(model_artifact, ArtifactReference) or not input_artifacts or not all(
                isinstance(item, ArtifactReference) for item in input_artifacts):
            raise ValueError("learned state requires real checkpoint and input artifact references")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("learned state requires a model identity")
        if not isinstance(fields, Mapping) or set(fields) - set(ENGINEERED_FIELDS):
            raise ValueError("learned state needs explicit canonical field mappings")
        if not set(REQUIRED_FIELDS) <= set(fields):
            raise ValueError("learned state mapping is missing required fields")
        variables = {}
        for canonical, spec in fields.items():
            if not isinstance(spec, FieldMapping) or spec.source not in decoded:
                raise ValueError(f"{canonical} needs a valid explicit decoded field mapping")
            if spec.semantic_class in {"measured", "synthetic"}:
                raise ValueError("learned outputs must be derived or inferred, never measured")
            source = decoded[spec.source]
            if source.attrs.get("units") != spec.unit:
                raise ValueError(f"decoded {spec.source} units disagree with explicit mapping")
            field = source.copy(deep=True)
            field.attrs.update(units=spec.unit, semantic_class=spec.semantic_class,
                               semantic=spec.description, source_field=spec.source)
            variables[canonical] = field
        data = xr.Dataset(variables, attrs=deepcopy(decoded.attrs))
        for name in ("lat", "lon"):
            if name in decoded.coords:
                data = data.assign_coords({name: decoded.coords[name].copy(deep=True)})
        data.attrs.update(state_schema=STATE_SCHEMA, source_kind="learned")
        provenance = {"producer": model_id, "model_artifact": asdict(model_artifact),
                      "input_artifacts": [asdict(item) for item in input_artifacts],
                      "field_mapping": {name: asdict(spec) for name, spec in fields.items()},
                      "semantics": "Explicit decoded model inference; not a physical measurement or latent-state reconstruction"}
        return DynamicState(data, deepcopy(event), deepcopy(timeline), provenance).validate()


def to_mapping_dataset(state: DynamicState):
    """Compatibility aliases for consumers; aliases do not change field units.

    ``dtec``/``tec`` are legacy consumer names, not a claim that arbitrary model
    output measures TEC. New consumers should use the canonical signal names.
    """
    state.validate()
    result = state.dataset.rename({name: ENGINEERED_FIELDS[name]
                                   for name in state.dataset.data_vars if name in ENGINEERED_FIELDS})
    result.attrs = dict(result.attrs, canonical_state_schema=STATE_SCHEMA,
                        signal_units=state.dataset.signal.attrs["units"],
                        compatibility_semantics="Legacy field aliases preserve producer units and semantics; names do not imply measured TEC")
    return result


def read_state(science_directory) -> DynamicState:
    """Load and validate a verified engineered state without importing art code."""
    import json
    from pathlib import Path
    import xarray as xr
    from ophanim.dynamics.schemas import AnalysisResult
    from .artifacts import verify_run

    directory = Path(science_directory).resolve()
    manifest = verify_run(directory, kind="science")
    if "state_contract.json" not in manifest["files"]:
        raise ValueError("legacy scientific run has no versioned state contract; reanalysis required")
    contract = json.loads((directory / "state_contract.json").read_text())
    if contract.get("schema_version") != STATE_SCHEMA or contract.get("dataset") != "dynamic.zarr":
        raise ValueError("unsupported published state contract")
    event = json.loads((directory / "event.json").read_text())
    timeline = json.loads((directory / "events.json").read_text()) if contract.get("timeline") == "events.json" else None
    with xr.open_zarr(directory / "dynamic.zarr", consolidated=True) as opened:
        dataset = opened.load()
    result = AnalysisResult(dataset, event, dataset.attrs.get("capabilities", {}))
    state = EngineeredStateAdapter().from_result(result, timeline=timeline)
    if contract["source_kind"] != state.dataset.attrs["source_kind"]:
        raise ValueError("published state source kind does not match its dataset")
    return state


__all__ = ["STATE_SCHEMA", "TIMELINE_SCHEMA", "DynamicState", "FieldMapping", "EngineeredStateAdapter",
           "LearnedStateAdapter", "validate_state", "validate_timeline", "to_mapping_dataset", "read_state"]
