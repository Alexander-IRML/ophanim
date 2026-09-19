"""Editable, reproducible mathematical scenarios downstream of observed candidates.

An observed candidate may suggest an amplitude and location, but does not recover
fine-scale structure or physical causes. Every parameter records that distinction.
No acquisition, scientific observation mutation, rendering, or training occurs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import math
from typing import Any, Mapping


SCENARIO_SCHEMA = "ophanim-scenario/1"
MAX_SCENARIO_CELLS = 200_000
CADENCE_MINUTES = 5.0
KINDS = ("wave_packet", "moving_front", "translating_gaussian", "growing_gaussian",
         "quiet", "localized_depletion")
SEMANTICS = ("Hypothetical mathematical TEC field, not a reconstruction of the observed event "
             "and not a physical ionosphere simulation. Fine-scale structure and evolution are chosen.")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    kind: str = "wave_packet"
    amplitude_tecu: float = 3.0
    width_km: float = 150.0
    speed_m_s: float = 100.0
    bearing_deg: float = 90.0
    period_minutes: float = 60.0
    duration_minutes: float = 120.0
    extent_km: float = 1000.0
    spacing_km: float = 25.0
    seed: int = 42
    center_latitude: float = 30.0
    center_longitude: float = -98.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"scenario kind must be one of {KINDS}")
        ranges = {"amplitude_tecu": (0, 50), "width_km": (5, 1000),
                  "speed_m_s": (0, 500), "bearing_deg": (0, 360),
                  "period_minutes": (10, 720), "duration_minutes": (10, 720),
                  "extent_km": (50, 2000), "spacing_km": (2, 100),
                  "center_latitude": (-90, 90), "center_longitude": (-180, 180)}
        for name, (minimum, maximum) in ranges.items():
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not minimum <= value <= maximum):
                raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.kind == "localized_depletion" and self.amplitude_tecu > 20:
            raise ValueError("depletion amplitude cannot exceed the chosen 20 TECU background")
        if self.width_km > self.extent_km:
            raise ValueError("width_km cannot exceed the scenario extent_km")
        nx, nt = self.dimensions
        if nx < 3:
            raise ValueError("scenario requires at least three grid points per spatial axis")
        if nx * nx * nt > MAX_SCENARIO_CELLS:
            raise ValueError(f"scenario exceeds {MAX_SCENARIO_CELLS:,} cells; increase spacing or reduce extent/duration")

    @property
    def dimensions(self) -> tuple[int, int]:
        return (int(math.floor(self.extent_km / self.spacing_km)) + 1,
                int(math.floor(self.duration_minutes / CADENCE_MINUTES)) + 1)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "ScenarioConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("scenario parameters must be an object")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown scenario parameters: {sorted(unknown)}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _candidate_center(candidate):
    center = candidate.get("centroid")
    if isinstance(center, Mapping):
        latitude = center.get("latitude", center.get("latitude_degrees", center.get("lat")))
        longitude = center.get("longitude", center.get("longitude_degrees", center.get("lon")))
    elif isinstance(center, (list, tuple)) and len(center) == 2:
        latitude, longitude = center
    else:
        return None
    if (_finite(latitude) and _finite(longitude)
            and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        return float(latitude), float(longitude)
    raise ValueError("candidate centroid must contain finite geographic coordinates")


def scenario_from_candidate(candidate: Mapping[str, Any] | None,
                            overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON recipe with per-parameter provenance and a stable parent ID.

    Passing None creates an independent experiment. Only source location and a
    bounded amplitude proxy are inherited. Period, velocity, width, duration,
    spacing, and structure are not recovered from a coarse candidate.
    """
    parameters = ScenarioConfig().to_dict()
    provenance = {name: {"status": "chosen", "reason": "Editable experiment default; not measured."}
                  for name in parameters}
    reference = None
    if candidate is not None:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate must be an object or None")
        identifier = candidate.get("candidate_id")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 200:
            raise ValueError("candidate requires a stable candidate_id")
        reference = {"candidate_id": identifier}
        for name in ("scan_id", "scan_run_id", "snapshot_id", "start_time", "end_time", "peak_time"):
            value = candidate.get(name)
            if isinstance(value, str) and len(value) <= 200:
                reference[name] = value
        deviation = candidate.get("peak_deviation_tecu")
        if _finite(deviation):
            limit = 20.0 if deviation < 0 else 50.0
            parameters["amplitude_tecu"] = min(abs(float(deviation)), limit)
            provenance["amplitude_tecu"] = {
                "status": "estimated", "source_field": "peak_deviation_tecu",
                "source_value": deviation,
                "reason": "Magnitude of baseline-relative native-grid deviation; bounded proxy, not measured fine-scale amplitude."}
            if deviation < 0:
                parameters["kind"] = "localized_depletion"
                provenance["kind"] = {"status": "chosen", "reason": "A hypothetical depletion suggested by the deviation sign, not a diagnosis."}
        center = _candidate_center(candidate)
        if center is not None:
            for name, value in zip(("center_latitude", "center_longitude"), center):
                parameters[name] = value
                provenance[name] = {"status": "estimated", "source_field": "centroid",
                                    "reason": "Native candidate centroid used as the experiment location."}
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("scenario overrides must be an object")
        parameters.update(overrides)
        for name in overrides:
            provenance[name] = {"status": "chosen", "reason": "Explicit experiment setting, not an observation."}
    config = ScenarioConfig.from_mapping(parameters)
    return {"schema": SCENARIO_SCHEMA, "parameters": config.to_dict(),
            "provenance": provenance, "candidate_reference": reference,
            "semantics": SEMANTICS,
            "fixed_parameters": {"background_tecu": 20.0, "cadence_minutes": CADENCE_MINUTES,
                                 "growth_rate_per_s": 0.00003,
                                 "status": "chosen"}}


def make_scenario(recipe: Mapping[str, Any]):
    """Generate a bounded xarray field; observations and source archives are untouched."""
    if not isinstance(recipe, Mapping) or recipe.get("schema") != SCENARIO_SCHEMA:
        raise ValueError("unsupported scenario recipe")
    allowed = {"schema", "parameters", "provenance", "candidate_reference", "semantics", "fixed_parameters"}
    if set(recipe) - allowed:
        raise ValueError("unknown scenario recipe sections")
    config = ScenarioConfig.from_mapping(recipe.get("parameters"))
    provenance = recipe.get("provenance")
    if (not isinstance(provenance, Mapping) or set(provenance) != set(config.to_dict())
            or any(not isinstance(value, Mapping) or value.get("status") not in {"observed", "estimated", "chosen"}
                   for value in provenance.values())):
        raise ValueError("scenario requires provenance for every parameter")
    expected_fixed = {"background_tecu": 20.0, "cadence_minutes": CADENCE_MINUTES,
                      "growth_rate_per_s": 0.00003, "status": "chosen"}
    if recipe.get("fixed_parameters", expected_fixed) != expected_fixed:
        raise ValueError("fixed scenario parameters cannot be changed under this schema")
    if recipe.get("semantics", SEMANTICS) != SEMANTICS:
        raise ValueError("scenario semantics must remain explicitly hypothetical")
    from ophanim.core.artifacts import json_value, software_identity
    from .synthetic import make_synthetic_dataset

    start = "2024-05-10T00:00:00Z"
    reference = recipe.get("candidate_reference")
    if reference is not None and not isinstance(reference, Mapping):
        raise ValueError("candidate_reference must be an object or None")
    if reference and reference.get("peak_time"):
        try:
            moment = datetime.fromisoformat(reference["peak_time"].replace("Z", "+00:00"))
            if moment.tzinfo is None or moment.utcoffset() is None:
                raise ValueError("naive timestamp")
            start = moment.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("candidate peak_time must be a timezone-aware timestamp") from error
    nx, nt = config.dimensions
    theta = math.radians(config.bearing_deg)
    # Period controls the chosen phase speed in a wave fixture; speed controls its
    # envelope translation. They are different editable mathematical quantities.
    data = make_synthetic_dataset(
        kind="translating_gaussian" if config.kind == "localized_depletion" else config.kind,
        nx=nx, ny=nx, nt=nt, spacing_m=config.spacing_km * 1000,
        cadence_s=CADENCE_MINUTES * 60,
        amplitude_tecu=-config.amplitude_tecu if config.kind == "localized_depletion" else config.amplitude_tecu,
        background_tecu=20.0,
        velocity_u_m_s=(config.speed_m_s if config.kind == "moving_front"
                        else config.speed_m_s * math.sin(theta)),
        velocity_v_m_s=config.speed_m_s * math.cos(theta), sigma_m=config.width_km * 1000,
        wavelength_m=max(config.spacing_km * 4, config.width_km * 2) * 1000,
        period_s=config.period_minutes * 60, bearing_deg=config.bearing_deg,
        seed=config.seed, start=start)
    meta = dict(data.attrs["source_metadata"])
    meta.update(projection_center_lat=config.center_latitude, projection_center_lon=config.center_longitude)
    data.attrs.update(source_metadata=meta, scenario=json_value(recipe),
                      source_kind="synthetic", scientific_scope=SEMANTICS,
                      generator_software=software_identity(extra_sections=("experiments",)))
    return data


__all__ = ["ScenarioConfig", "scenario_from_candidate", "make_scenario", "SCENARIO_SCHEMA", "KINDS"]
