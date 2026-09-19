"""Validated, serializable scientific configuration and result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Literal


class AnalysisError(ValueError):
    """Input or configuration cannot support a well-defined analysis."""


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise AnalysisError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class FlowConfig:
    method: Literal["tvl1", "ilk"] = "tvl1"
    enabled: bool = True
    attachment: float = 15.0
    tightness: float = 0.3
    num_warp: int = 5
    num_iter: int = 10
    ilk_radius: int = 7
    warp_error_scale: float = 0.3
    forward_backward_scale_pixels: float = 1.0
    growth_log_gain_scale: float = 0.10
    minimum_pair_coverage: float = 0.8
    maximum_gap_factor: float = 1.5
    minimum_confidence: float = 0.4
    mask_erosion_cells: int = 2
    texture_smoothing_cells: float = 1.5
    texture_tensor_cells: float = 2.0
    noise_reliability_penalty: float = 2.0
    photometric_window_cells: float = 6.0
    photometric_explained_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.method not in ("tvl1", "ilk"):
            raise AnalysisError("flow method must be tvl1 or ilk")
        for name in ("attachment", "tightness", "warp_error_scale", "forward_backward_scale_pixels", "growth_log_gain_scale", "maximum_gap_factor", "texture_smoothing_cells", "texture_tensor_cells", "noise_reliability_penalty", "photometric_window_cells"):
            _positive(name, getattr(self, name))
        if min(self.num_warp, self.num_iter, self.ilk_radius) < 1 or self.mask_erosion_cells < 0:
            raise AnalysisError("flow iterations/radius must be positive; erosion nonnegative")
        if not 0 < self.minimum_pair_coverage <= 1 or not 0 <= self.minimum_confidence <= 1:
            raise AnalysisError("flow coverage/confidence must be in [0,1]")
        if not 0 < self.photometric_explained_threshold < 1:
            raise AnalysisError("photometric_explained_threshold must be within (0,1)")


@dataclass(frozen=True, slots=True)
class WaveConfig:
    enabled: bool = True
    min_period_minutes: float = 30.0
    max_period_minutes: float = 180.0
    min_wavelength_km: float = 150.0
    max_wavelength_km: float = 2000.0
    minimum_samples_per_period: float = 6.0
    minimum_cycles: float = 3.0
    minimum_native_cells: float = 4.0
    minimum_spatial_cycles: float = 2.0
    minimum_coverage: float = 0.9
    minimum_concentration: float = 0.2
    minimum_amplitude_tecu: float = 0.01
    maximum_gap_factor: float = 1.5
    minimum_temporal_occupancy: float = 0.65
    occupancy_amplitude_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in ("min_period_minutes", "max_period_minutes", "min_wavelength_km", "max_wavelength_km", "minimum_samples_per_period", "minimum_cycles", "minimum_native_cells", "minimum_spatial_cycles", "minimum_amplitude_tecu", "maximum_gap_factor"):
            _positive(name, getattr(self, name))
        if self.min_period_minutes >= self.max_period_minutes or self.min_wavelength_km >= self.max_wavelength_km:
            raise AnalysisError("wave search minima must be smaller than maxima")
        if self.minimum_samples_per_period < 2 or self.minimum_native_cells < 2:
            raise AnalysisError("wave admission cannot be below formal sampling limits")
        if not 0 < self.minimum_coverage <= 1 or not 0 <= self.minimum_concentration <= 1:
            raise AnalysisError("wave coverage/concentration must be in [0,1]")
        if not 0 < self.minimum_temporal_occupancy <= 1 or not 0 < self.occupancy_amplitude_fraction < 1:
            raise AnalysisError("wave occupancy settings must be within (0,1]")


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    analysis_mode: Literal["retrospective", "causal"] = "retrospective"
    as_of: str | None = None
    target_time: str | None = None
    science_spacing_km: float = 25.0
    projection_center_lat: float | None = None
    projection_center_lon: float | None = None
    max_projection_radius_km: float = 2500.0
    max_projection_scale_error: float = 0.08
    max_grid_cells: int = 1_000_000
    max_cube_cells: int = 2_000_000
    cadence_seconds: float | None = None
    maximum_interpolation_gap_seconds: float = 1800.0
    temporal_interpolation_weight: float = 0.5
    spatial_interpolation_weight: float = 0.85
    smooth_sigma_km: float = 18.75
    smoothing_minimum_support: float = 0.8
    structure_sigma_km: float = 50.0
    short_window_minutes: float = 90.0
    baseline_method: Literal["savgol", "rolling_median", "constant"] = "savgol"
    baseline_window_minutes: float = 110.0
    baseline_polyorder: int = 2
    constant_background_tecu: float | None = None
    wave_window_hours: float = 4.0
    robust_scale_floor_tecu: float = 0.05
    anomaly_scale_tecu: float = 1.0
    temporal_scale_tecu_per_minute: float = 0.05
    gradient_scale_tecu_per_km: float = 0.01
    flow_speed_scale_m_s: float = 100.0
    event_focus_bounds: dict[str, float] | None = None
    flow: FlowConfig = field(default_factory=FlowConfig)
    wave: WaveConfig = field(default_factory=WaveConfig)

    def __post_init__(self) -> None:
        if self.analysis_mode not in ("retrospective", "causal"):
            raise AnalysisError("analysis_mode must be retrospective or causal")
        if self.analysis_mode == "causal" and not self.as_of:
            raise AnalysisError("causal analysis requires an explicit as_of cutoff")
        for name in ("as_of", "target_time"):
            value = getattr(self, name)
            if value is not None:
                try:
                    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except (ValueError, AttributeError) as exc:
                    raise AnalysisError(f"{name} must be an ISO timestamp with timezone") from exc
                if stamp.tzinfo is None:
                    raise AnalysisError(f"{name} must include an explicit timezone")
        if self.baseline_method not in ("savgol", "rolling_median", "constant"):
            raise AnalysisError("unknown baseline method")
        if self.baseline_method == "constant" and self.constant_background_tecu is None:
            raise AnalysisError("constant baseline requires constant_background_tecu")
        if self.constant_background_tecu is not None and not isfinite(self.constant_background_tecu):
            raise AnalysisError("constant background must be finite")
        for name in ("science_spacing_km", "max_projection_radius_km", "max_projection_scale_error", "structure_sigma_km", "short_window_minutes", "baseline_window_minutes", "wave_window_hours", "robust_scale_floor_tecu", "anomaly_scale_tecu", "temporal_scale_tecu_per_minute", "gradient_scale_tecu_per_km", "flow_speed_scale_m_s"):
            _positive(name, getattr(self, name))
        if self.cadence_seconds is not None:
            _positive("cadence_seconds", self.cadence_seconds)
        if not isfinite(self.smooth_sigma_km) or self.smooth_sigma_km < 0 or self.maximum_interpolation_gap_seconds < 0:
            raise AnalysisError("smoothing and interpolation limits must be nonnegative")
        if self.baseline_polyorder < 0 or self.max_grid_cells < 9:
            raise AnalysisError("invalid polynomial order or grid cell limit")
        if isinstance(self.max_cube_cells, bool) or not isinstance(self.max_cube_cells, int) or self.max_cube_cells < 9:
            raise AnalysisError("max_cube_cells must be an integer of at least nine")
        for name in ("temporal_interpolation_weight", "spatial_interpolation_weight", "smoothing_minimum_support"):
            if not 0 <= getattr(self, name) <= 1:
                raise AnalysisError(f"{name} must be in [0,1]")
        if not isinstance(self.flow, FlowConfig) or not isinstance(self.wave, WaveConfig):
            raise AnalysisError("flow/wave settings must be typed FlowConfig/WaveConfig")
        if self.event_focus_bounds is not None:
            bounds = self.event_focus_bounds
            if not isinstance(bounds, dict) or set(bounds) != {"south", "north", "west", "east"}:
                raise AnalysisError("event_focus_bounds requires south/north/west/east")
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in bounds.values())
                    or not -90 <= bounds["south"] <= bounds["north"] <= 90
                    or not -180 <= bounds["west"] <= 180 or not -180 <= bounds["east"] <= 180):
                raise AnalysisError("invalid event_focus_bounds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AnalysisConfig:
        fields = dict(value)
        if isinstance(fields.get("flow"), dict):
            fields["flow"] = FlowConfig(**fields["flow"])
        if isinstance(fields.get("wave"), dict):
            fields["wave"] = WaveConfig(**fields["wave"])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    dataset: Any
    event: dict[str, Any]
    capabilities: dict[str, Any]
