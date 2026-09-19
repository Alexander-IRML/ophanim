"""Admission policy based on original support, never interpolated grid density."""

from __future__ import annotations

import numpy as np

from .coords import source_metadata
from .regularize import utc64
from .schemas import AnalysisConfig


def assess_capabilities(data, config: AnalysisConfig) -> dict:
    meta = source_metadata(data)
    original = data.attrs.get("source_observation_times", [str(t) for t in data.time.values])
    stamps = np.asarray([utc64(t) for t in original])
    if config.target_time:
        target = utc64(config.target_time)
    else:
        target = data.time.values[-1]
    lower = target - np.timedelta64(int(config.wave_window_hours * 3600 * 1e9), "ns")
    stamps = stamps[(stamps >= lower) & (stamps <= target)]
    deltas = np.diff(stamps).astype("timedelta64[ns]").astype(float) / 1e9
    cadence = float(np.median(deltas)) if len(deltas) else None
    duration = float((stamps[-1] - stamps[0]) / np.timedelta64(1, "s")) if len(stamps) > 1 else 0.0
    analyzed = data.time.values[(data.time.values >= lower) & (data.time.values <= target)]
    analysis_deltas = np.diff(analyzed).astype("timedelta64[ns]").astype(float) / 1e9
    analysis_cadence = float(np.median(analysis_deltas)) if len(analysis_deltas) else None
    analysis_duration = float((analyzed[-1] - analyzed[0]) / np.timedelta64(1, "s")) if len(analyzed) > 1 else 0.0
    admission_cadence = max(cadence or float("inf"), analysis_cadence or float("inf"))
    native_dx, native_dy = meta.get("native_dx_m"), meta.get("native_dy_m")
    known = all(v is not None and np.isfinite(v) and v > 0 for v in (native_dx, native_dy))
    min_period = max(config.wave.min_period_minutes * 60, admission_cadence * config.wave.minimum_samples_per_period)
    max_period = min(config.wave.max_period_minutes * 60, min(duration, analysis_duration) / config.wave.minimum_cycles)
    wave_status, reasons = "supported", []
    if not known:
        wave_status = "unsupported_resolution"
        reasons.append("original native spatial support is unknown")
    if cadence is None or analysis_cadence is None or max_period < min_period or min(len(stamps), len(analyzed)) < config.wave.minimum_samples_per_period * config.wave.minimum_cycles:
        wave_status = "insufficient_samples"
        reasons.append("both original observations and the analyzed time grid must cover the requested period with the configured samples/cycles")
    if len(deltas) and deltas.max() > cadence * config.wave.maximum_gap_factor:
        wave_status = "invalid_support"
        reasons.append("source timestamps contain a gap beyond the wave admission limit")
    if len(analysis_deltas) and not np.allclose(analysis_deltas, analysis_cadence, rtol=1e-8, atol=1e-6):
        wave_status = "invalid_support"
        reasons.append("FFT analysis requires a regular analyzed time grid")
    extent_x = float(data.x.values[-1] - data.x.values[0])
    extent_y = float(data.y.values[-1] - data.y.values[0])
    native_count_x = extent_x / native_dx + 1 if known else None
    native_count_y = extent_y / native_dy + 1 if known else None
    flow_status = "supported"
    if len(original) < 2:
        flow_status = "insufficient_samples"
    elif not known or min(native_count_x, native_count_y) < 3:
        flow_status = "unsupported_resolution"
    return {
        "policy": "conservative engineering admission gates, not a physical resolution guarantee",
        "source_kind": meta.get("source_kind", "unknown"),
        "native_dx_m": float(native_dx) if known else None,
        "native_dy_m": float(native_dy) if known else None,
        "effective_resolution_m": meta.get("effective_resolution_m"),
        "effective_resolution_known": meta.get("effective_resolution_m") is not None,
        "source_observation_count": len(original),
        "wave_observation_count": len(stamps),
        "native_cadence_seconds": cadence,
        "analysis_cadence_seconds": analysis_cadence,
        "wave_admission_cadence_seconds": admission_cadence if np.isfinite(admission_cadence) else None,
        "wave_analysis_sample_count": len(analyzed),
        "maximum_observation_gap_seconds": float(deltas.max()) if len(deltas) else None,
        "wave_observed_duration_seconds": duration,
        "admissible_period_seconds": [float(min_period), float(max_period)] if min_period <= max_period else None,
        "native_samples_across_x": float(native_count_x) if known else None,
        "native_samples_across_y": float(native_count_y) if known else None,
        "analysis_grid_dx_m": float(np.median(np.diff(data.x.values))),
        "analysis_grid_dy_m": float(np.median(np.diff(data.y.values))),
        "interpolation_adds_resolution": False,
        "wave": {"status": wave_status, "reasons": reasons},
        "flow": {"status": flow_status},
    }
