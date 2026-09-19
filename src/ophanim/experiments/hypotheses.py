"""Evidence-linked hypotheses whose guesses never become scientific detections."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import math
from typing import Any, Mapping

from ophanim.core.artifacts import canonical_json
from ophanim.core.volumes import MAX_FRAME_CELLS, MAX_VOLUME_CELLS, MAX_VOLUME_FRAMES

HYPOTHESIS_SCHEMA = "ophanim-hypothesis/1"
HYPOTHESIS_KINDS = ("wave_packet", "front", "sheared_jet")
HYPOTHESIS_SEMANTICS = ("Speculative local 3-D hypothesis inspired by evidence; assumed motion and altitude "
                        "may be wrong. No generated field or chosen parameter is an observed diagnosis.")
MAX_SIMULATION_STEPS = 360
MAX_SIMULATION_WORK_CELLS = 50_000_000


@dataclass(frozen=True, slots=True)
class HypothesisConfig:
    kind: str = "wave_packet"
    amplitude: float = 1.0
    width_km: float = 150.0
    speed_m_s: float = 150.0
    bearing_deg: float = 60.0
    period_minutes: float = 35.0
    shear_per_s: float = 0.0002
    diffusivity_m2_s: float = 20_000.0
    duration_minutes: float = 90.0
    frame_count: int = 12
    extent_km: float = 800.0
    spacing_km: float = 20.0
    altitude_min_km: float = 180.0
    altitude_max_km: float = 480.0
    vertical_spacing_km: float = 12.5
    seed: int = 42
    center_latitude: float = 30.0
    center_longitude: float = -98.0

    def __post_init__(self):
        if self.kind not in HYPOTHESIS_KINDS:
            raise ValueError(f"hypothesis kind must be one of {HYPOTHESIS_KINDS}")
        ranges = {"amplitude": (0.1, 4), "width_km": (20, 500), "speed_m_s": (0, 600),
                  "bearing_deg": (0, 360), "period_minutes": (5, 240), "shear_per_s": (0, 0.003),
                  "diffusivity_m2_s": (0, 2_000_000), "duration_minutes": (5, 240),
                  "extent_km": (100, 1600), "spacing_km": (5, 100), "altitude_min_km": (80, 700),
                  "altitude_max_km": (100, 1000), "vertical_spacing_km": (5, 100),
                  "center_latitude": (-90, 90), "center_longitude": (-180, 180)}
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}")
        for name, low, high in (("seed", 0, 2**32 - 1), ("frame_count", 3, MAX_VOLUME_FRAMES)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} through {high}")
        if self.width_km > self.extent_km or self.altitude_max_km <= self.altitude_min_km:
            raise ValueError("width must fit the horizontal extent and altitude bounds must increase")
        nt, nz, ny, nx = self.dimensions
        if min(nx, ny, nz) < 5:
            raise ValueError("hypothesis needs at least five points on each spatial axis")
        cells = nz * ny * nx
        if cells > MAX_FRAME_CELLS or nt * cells > MAX_VOLUME_CELLS:
            raise ValueError("hypothesis exceeds the bounded volume cell budget; increase spacing")
        steps = self.substeps_per_frame * (self.frame_count - 1)
        if steps > MAX_SIMULATION_STEPS or steps * cells > MAX_SIMULATION_WORK_CELLS:
            raise ValueError("hypothesis exceeds the numerical step/work budget; reduce duration, shear or resolution")

    @property
    def dimensions(self):
        n = int(math.floor(self.extent_km / self.spacing_km)) + 1
        z = int(math.floor((self.altitude_max_km - self.altitude_min_km) / self.vertical_spacing_km)) + 1
        return self.frame_count, z, n, n

    @property
    def substeps_per_frame(self):
        # A conservative displacement bound, not a claim of physical stability.
        velocity_bound = 1.4 * self.speed_m_s + self.shear_per_s * self.extent_km * 500 + 30
        step = min(120.0, 0.8 * min(self.spacing_km, self.vertical_spacing_km) * 1000 / velocity_bound)
        return max(1, math.ceil(self.duration_minutes * 60 / (self.frame_count - 1) / step))

    @classmethod
    def from_mapping(cls, values=None):
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("hypothesis parameters must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown hypothesis parameters: {sorted(unknown)}")
        return cls(**dict(values))

    def to_dict(self):
        return asdict(self)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def hypothesis_from_evidence(candidate=None, event=None, overrides=None):
    """Choose one revisable hypothesis; scientific abstention never disables imagination.

    ``event`` is frozen as evidence. Its supported wave parameters can seed a
    hypothesis, but motion defaults remain chosen, never inferred from missing
    optical flow. Alternative kinds are available through alternatives_from_evidence.
    """
    parameters = HypothesisConfig().to_dict()
    provenance = {key: {"status": "chosen", "reason": "Explicit imagined-model assumption, not measured."}
                  for key in parameters}
    provenance["kind"]["reason"] = "Illustrative fallback; no mechanism established. Alternative scenarios are not diagnoses."
    reference = None
    evidence = {"candidate": None, "scientific_event": None}
    if candidate is not None:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("candidate_id"), str) or not candidate["candidate_id"].strip():
            raise ValueError("candidate must contain a stable candidate_id")
        reference = {key: candidate[key] for key in ("candidate_id", "scan_id", "snapshot_id", "peak_time")
                     if isinstance(candidate.get(key), str)}
        center = candidate.get("centroid")
        if center is not None:
            if not isinstance(center, Mapping) or not all(_finite(center.get(key)) for key in ("latitude", "longitude")):
                raise ValueError("candidate centroid requires finite latitude/longitude")
            for parameter, key in (("center_latitude", "latitude"), ("center_longitude", "longitude")):
                parameters[parameter] = center[key]
                provenance[parameter] = {"status": "estimated", "source_field": f"candidate.centroid.{key}",
                                         "reason": "Coarse candidate center, not a measured 3-D location."}
        deviation = candidate.get("peak_deviation_tecu")
        if _finite(deviation):
            parameters["amplitude"] = max(0.4, min(2.0, abs(deviation) / 3))
            provenance["amplitude"] = {"status": "chosen", "source_field": "candidate.peak_deviation_tecu",
                                       "source_value": deviation,
                                       "reason": "Chosen dimensionless material-strength mapping; not electron density or TEC."}
        evidence["candidate"] = {key: deepcopy(candidate[key]) for key in
                                  ("candidate_id", "title", "peak_deviation_tecu", "centroid", "bounds", "peak_time", "native_cell_count",
                                   "hypotheses", "evidence_strength", "source_kind", "interpretation")
                                  if key in candidate}
        # Only explicit, unambiguous categories propose a shape. Generic TEC
        # disturbance/flow/variability is not evidence for an imagined jet.
        labels = {"wave_like": "wave_packet", "traveling_wave": "wave_packet",
                  "front_like": "front", "moving_front": "front", "sheared_jet": "sheared_jet"}
        categories = {labels[item.get("category", item.get("label"))]
                      for item in candidate.get("hypotheses", [])
                      if isinstance(item, Mapping) and item.get("category", item.get("label")) in labels}
        if len(categories) == 1:
            parameters["kind"] = categories.pop()
            provenance["kind"] = {"status": "chosen", "source_field": "candidate.hypotheses",
                                  "reason": "Illustrative shape proposed from an explicit candidate category; not a physical diagnosis."}
    if event is not None:
        if not isinstance(event, Mapping):
            raise ValueError("scientific event must be an object or None")
        evidence["scientific_event"] = {key: deepcopy(event[key]) for key in
                                         ("time", "wave", "channel_status", "event_scores", "event_confidence", "display_class") if key in event}
        scores = event.get("event_scores", {})
        if isinstance(scores, Mapping) and _finite(scores.get("front")) and scores["front"] >= .5:
            others = [value for name, value in scores.items() if name != "front" and _finite(value)]
            if scores["front"] - max(others, default=0) >= .15:
                parameters["kind"] = "front"
                provenance["kind"] = {"status": "chosen", "source_field": "scientific_event.event_scores.front",
                                      "reason": "A relatively strong front-like image score proposes an imagined curtain; heuristic, not a diagnosis."}
        wave = event.get("wave", {})
        if isinstance(wave, Mapping) and wave.get("status") == "estimated" and _finite(wave.get("confidence")) and wave["confidence"] >= .7:
            parameters["kind"] = "wave_packet"
            provenance["kind"] = {"status": "chosen", "source_field": "scientific_event.wave",
                                  "reason": "Supported periodic image structure proposes a wave-like volume, not a confirmed physical wave mechanism."}
            for parameter, key, low, high in (("period_minutes", "period_min", 5, 240), ("bearing_deg", "bearing_deg", 0, 360)):
                if _finite(wave.get(key)) and low <= wave[key] <= high:
                    parameters[parameter] = wave[key]
                    provenance[parameter] = {"status": "estimated", "source_field": f"scientific_event.wave.{key}",
                                             "reason": "Supported image-wave estimate used as a revisable hypothesis, not proof of a mechanism."}
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("hypothesis overrides must be an object")
        parameters.update(overrides)
        for name in overrides:
            provenance[name] = {"status": "chosen", "reason": "Explicit experiment override, not an observation."}
    config = HypothesisConfig.from_mapping(parameters)
    start = reference.get("peak_time") if reference else None
    if start is None:
        start = event.get("time") if isinstance(event, Mapping) else None
    start = start or "2024-05-10T00:00:00Z"
    try:
        stamp = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("naive timestamp")
    except (ValueError, AttributeError) as error:
        raise ValueError("hypothesis start time requires an aware ISO timestamp") from error
    return {"schema": HYPOTHESIS_SCHEMA, "parameters": config.to_dict(), "provenance": provenance,
            "candidate_reference": reference, "evidence": evidence,
            "evidence_sha256": sha256(canonical_json(evidence)).hexdigest(),
            "start_time": stamp.astimezone(timezone.utc).isoformat(),
            "semantics": HYPOTHESIS_SEMANTICS,
            "ifm": {"status": "deferred", "target_version": "0.3", "used": False}}


def alternatives_from_evidence(candidate=None, event=None, overrides=None):
    """Three hypotheses, not three diagnoses or probability-ranked explanations."""
    return [hypothesis_from_evidence(candidate, event, {**(overrides or {}), "kind": kind})
            for kind in HYPOTHESIS_KINDS]


def validate_hypothesis(recipe):
    if not isinstance(recipe, Mapping) or recipe.get("schema") != HYPOTHESIS_SCHEMA:
        raise ValueError("unsupported hypothesis recipe")
    expected = {"schema", "parameters", "provenance", "candidate_reference", "evidence", "evidence_sha256", "start_time", "semantics", "ifm"}
    if set(recipe) != expected or recipe["semantics"] != HYPOTHESIS_SEMANTICS:
        raise ValueError("hypothesis requires its exact versioned sections and hypothetical semantics")
    config = HypothesisConfig.from_mapping(recipe["parameters"])
    provenance = recipe["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != set(config.to_dict()):
        raise ValueError("hypothesis requires provenance for every parameter")
    if any(not isinstance(value, Mapping) or value.get("status") not in {"estimated", "chosen"}
           or not isinstance(value.get("reason"), str) for value in provenance.values()):
        raise ValueError("hypothesis parameters must remain explicitly estimated or chosen, never observed")
    if sha256(canonical_json(recipe["evidence"])).hexdigest() != recipe["evidence_sha256"]:
        raise ValueError("frozen hypothesis evidence checksum mismatch")
    if recipe["ifm"] != {"status": "deferred", "target_version": "0.3", "used": False}:
        raise ValueError("IFM is explicitly deferred and unused in this hypothesis schema")
    try:
        stamp = datetime.fromisoformat(recipe["start_time"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("naive timestamp")
    except (ValueError, AttributeError) as error:
        raise ValueError("hypothesis start_time requires an aware ISO timestamp") from error
    return config


__all__ = ["HypothesisConfig", "hypothesis_from_evidence", "alternatives_from_evidence", "validate_hypothesis"]
