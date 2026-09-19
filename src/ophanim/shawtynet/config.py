"""Validated artistic settings, separate from scientific processing settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return float(value)


@dataclass(frozen=True, slots=True)
class VisualStyleConfig:
    """All heights, colors, opacity and texture choices are artistic settings.

    ``quiet_opacity_floor`` intentionally preserves a faint supported veil when
    anomaly magnitude is zero; it is not evidence for an ionospheric event.
    """

    name: str = "ghost_atmospheric"
    seed: int = 0
    render_spacing_km: float = 10.0
    max_render_cells: int = 1_000_000
    max_render_voxels: int = 2_000_000
    shell_altitude_km: float = 300.0
    vertical_exaggeration_km: float = 4.0
    base_opacity: float = 0.4
    quiet_opacity_floor: float = 0.06
    boundary_feather_km: float = 30.0
    emission_gain: float = 2.5
    emission_white_point: float = 2.0
    quiet_emission: float = 0.12
    detail_layer_offset_km: float = 0.2
    detail_emission_gain: float = 2.0
    flow_texture_strength: float = 0.7
    lic_streamline_steps: int = 25
    lic_step_cells: float = 0.75
    lic_contrast: float = 1.8
    min_flow_speed_m_s: float = 0.01
    fiber_sharpness: float = 2.5
    kinematic_timescale_seconds: float = 900.0
    kinematic_modulation: float = 0.3
    front_width_km: float = 60.0
    front_ribbon_strength: float = 0.8
    front_background_strength: float = 0.2
    front_emission_strength: float = 0.9
    front_opacity_gain: float = 0.45
    front_shading_strength: float = 0.35
    wave_strength: float = 0.35
    front_fold_strength: float = 0.45
    localized_strength: float = 0.3
    breakup_strength: float = 0.5
    procedural_displacement_km: float = 0.15
    positive_color: tuple[float, float, float] = (0.70, 0.80, 1.0)
    negative_color: tuple[float, float, float] = (0.95, 0.60, 0.48)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("style name must be nonempty")
        for name, low, high in (
            ("seed", 0, 2**32 - 1), ("max_render_cells", 4, 100_000_000),
            ("max_render_voxels", 4, 100_000_000),
            ("lic_streamline_steps", 1, 1000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, low, high in (
            ("render_spacing_km", 0.001, 1000), ("shell_altitude_km", 1, 10_000),
            ("vertical_exaggeration_km", 0, 500), ("base_opacity", 0, 1),
            ("quiet_opacity_floor", 0, 1), ("emission_gain", 0, 1000),
            ("emission_white_point", 0.01, 100), ("quiet_emission", 0, 1),
            ("detail_layer_offset_km", 0.001, 10), ("detail_emission_gain", 0, 5),
            ("boundary_feather_km", 0, 1000),
            ("flow_texture_strength", 0, 1), ("lic_step_cells", 0.01, 1),
            ("lic_contrast", 0.01, 20), ("min_flow_speed_m_s", 0, 100_000),
            ("fiber_sharpness", 1, 12), ("kinematic_timescale_seconds", 1, 86400),
            ("kinematic_modulation", 0, 1), ("front_width_km", 1, 2000),
            ("front_ribbon_strength", 0, 1),
            ("front_background_strength", 0, 1), ("front_emission_strength", 0, 3),
            ("front_opacity_gain", 0, 0.9), ("front_shading_strength", 0, 0.8),
            ("wave_strength", 0, 1), ("front_fold_strength", 0, 1),
            ("localized_strength", 0, 1), ("breakup_strength", 0, 1),
            ("procedural_displacement_km", 0, 50),
        ):
            _number(getattr(self, name), name, low, high)
        if self.quiet_opacity_floor > self.base_opacity:
            raise ValueError("quiet_opacity_floor must not exceed base_opacity")
        if self.vertical_exaggeration_km * 3 + self.procedural_displacement_km >= self.shell_altitude_km:
            raise ValueError("artistic displacement must keep the visualization shell above ground")
        for name in ("positive_color", "negative_color"):
            color = getattr(self, name)
            if not isinstance(color, tuple) or len(color) != 3:
                raise ValueError(f"{name} must contain three RGB components")
            for value in color:
                _number(value, name, 0, 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> VisualStyleConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("visual style must be a mapping")
        values = dict(payload)
        # Accept the brief's human-friendly nested YAML as well as resolved flat
        # settings; unknown keys fail so misspelled controls cannot be ignored.
        groups = {
            "sheet": {"vertical_exaggeration": "vertical_exaggeration_km", "base_opacity": "base_opacity", "quiet_opacity_floor": "quiet_opacity_floor", "boundary_feather_km": "boundary_feather_km"},
            "emission": {"gain": "emission_gain", "white_point": "emission_white_point", "quiet": "quiet_emission"},
            "flow": {"lic_strength": "flow_texture_strength", "streamline_length": "lic_streamline_steps", "contrast": "lic_contrast"},
            "wave": {"displacement_strength": "wave_strength"},
            "uncertainty": {"breakup_strength": "breakup_strength"},
        }
        for group, mapping in groups.items():
            if group not in values:
                continue
            nested = values.pop(group)
            if not isinstance(nested, Mapping) or set(nested) - set(mapping):
                raise ValueError(f"invalid {group} style settings")
            for key, value in nested.items():
                target = mapping[key]
                if target in values:
                    raise ValueError(f"duplicate style setting: {target}")
                values[target] = value
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown style settings: {', '.join(sorted(unknown))}")
        for key in ("positive_color", "negative_color"):
            if key in values and isinstance(values[key], list):
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def from_file(cls, path: str | Path) -> VisualStyleConfig:
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError("YAML styles require PyYAML; JSON styles are also supported") from error
            payload = yaml.safe_load(text)
        return cls.from_mapping(payload)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Manual photograph alignment in local east/north/up coordinates."""

    observer_latitude: float
    observer_longitude: float
    observer_altitude_km: float = 0.0
    heading_deg: float = 0.0
    pitch_deg: float = 45.0
    roll_deg: float = 0.0
    horizontal_fov_deg: float = 70.0

    def __post_init__(self) -> None:
        for name, low, high in (
            ("observer_latitude", -90, 90), ("observer_longitude", -180, 180),
            ("observer_altitude_km", -1, 1000), ("heading_deg", -360, 360),
            ("pitch_deg", -90, 90), ("roll_deg", -180, 180),
            ("horizontal_fov_deg", 1, 179),
        ):
            _number(getattr(self, name), name, low, high)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
