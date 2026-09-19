"""One-way scientific-state to artistic-field mapping.

The returned render grid is disposable derived artwork. Input arrays, metadata,
and science archives are never mutated. No visual quantity is a measurement.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from ophanim.shawtynet.config import VisualStyleConfig
from ophanim.shawtynet.lic import line_integral_convolution

RENDER_INPUT_FIELDS = frozenset({
    "dtec", "dtec_z", "observation_support", "observed_mask", "tec",
    "anomaly_confidence", "flow_confidence", "flow_interpretation_confidence", "flow_u", "flow_v",
    "structure_coherence", "structure_orientation", "grad_mag",
    "flow_divergence", "flow_vorticity", "flow_strain",
    "advection_residual", "lat", "lon",
})


def _jsonable(value: Any) -> Any:
    """Canonical metadata supports NumPy scalars and missing estimates."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def dataset_digest(dataset) -> str:
    """Hash scientific content and metadata, independent of on-disk chunking."""
    import numpy as np

    digest = sha256(canonical_json(dict(dataset.attrs)).encode())
    for name in sorted(dataset.variables):
        variable = dataset[name]
        digest.update(canonical_json({"name": name, "dims": variable.dims, "attrs": variable.attrs}).encode())
        values = np.asarray(variable.values)
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        if values.dtype.kind in "fc":
            values = values.astype("<c16" if values.dtype.kind == "c" else "<f8", copy=True)
            values[np.isnan(values)] = np.nan
            digest.update(values.tobytes(order="C"))
        elif values.dtype.kind in "mM":
            digest.update(str(values.dtype).encode())
            digest.update(values.astype("<i8").tobytes(order="C"))
        elif values.dtype.kind in "biu":
            digest.update(values.astype("<i8").tobytes(order="C"))
        else:
            digest.update(canonical_json(values.tolist()).encode())
    return digest.hexdigest()


def _score(event: Mapping[str, Any], key: str, default: float = 0) -> float:
    scores = event.get("event_scores", event.get("scores", {}))
    value = scores.get(key, scores.get(f"{key}_score", default)) if isinstance(scores, Mapping) else default
    if isinstance(value, (int, float)) and math.isfinite(value):
        return min(1.0, max(0.0, float(value)))
    return default


def _validate_input(dynamic) -> None:
    import numpy as np

    if any(name not in dynamic.dims for name in ("time", "y", "x")):
        raise ValueError("dynamic state requires time, y and x dimensions")
    if dynamic.sizes["time"] < 1 or min(dynamic.sizes["x"], dynamic.sizes["y"]) < 2:
        raise ValueError("dynamic state needs at least one frame and a 2×2 metric grid")
    for name in ("x", "y"):
        coordinate = np.asarray(dynamic[name].values, dtype=float)
        if not np.all(np.isfinite(coordinate)) or not np.all(np.diff(coordinate) > 0):
            raise ValueError(f"{name} coordinates must be finite and strictly increasing")
        units = dynamic[name].attrs.get("units", "m")
        if units not in ("m", "metre", "meter", "metres", "meters"):
            raise ValueError(f"{name} coordinates must be metres")
    for name in ("dtec", "dtec_z"):
        if name not in dynamic or set(dynamic[name].dims) != {"time", "y", "x"}:
            raise ValueError(f"dynamic state requires {name}(time,y,x)")
    if "observation_support" not in dynamic and "observed_mask" not in dynamic:
        raise ValueError("dynamic state requires observation support or an observed mask")
    times = np.asarray(dynamic.time.values)
    if times.dtype.kind != "M" or np.any(np.isnat(times)):
        raise ValueError("dynamic time must contain valid UTC datetime64 coordinates")
    if len(times) > 1 and np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("dynamic time must be strictly increasing")


def _render_grid(dynamic, style: VisualStyleConfig):
    import numpy as np

    spacing = style.render_spacing_km * 1000
    axes = {}
    for name in ("x", "y"):
        source = np.asarray(dynamic[name].values, dtype=float)
        count = max(2, math.ceil((source[-1] - source[0]) / spacing) + 1)
        axes[name] = np.linspace(source[0], source[-1], count)
    if len(axes["x"]) * len(axes["y"]) > style.max_render_cells:
        raise ValueError("render grid exceeds max_render_cells; increase render_spacing_km")
    if len(axes["x"]) * len(axes["y"]) * dynamic.sizes["time"] > style.max_render_voxels:
        raise ValueError("visual sequence exceeds max_render_voxels; select one frame or increase render_spacing_km")
    # Boolean masks use a float support field while interpolating, keeping
    # invalid data holes visible instead of nearest-neighbor filling them.
    # Resample only consumers' controls, not dozens of unused scientific
    # derivatives/confidence intermediates. Inference remains on its own grid.
    working = dynamic[[name for name in dynamic.data_vars if name in RENDER_INPUT_FIELDS]].copy(deep=False)
    # Band orientation is axial (modulo pi), never interpolate wrapped angles.
    if "structure_orientation" in working:
        working["_orientation_cos"] = np.cos(2 * working.structure_orientation)
        working["_orientation_sin"] = np.sin(2 * working.structure_orientation)
        working = working.drop_vars("structure_orientation")
    for name in working.data_vars:
        if working[name].dtype.kind == "b":
            working[name] = working[name].astype(float)
    if "lon" in working.coords:
        longitude = np.asarray(working.lon.values, dtype=float)
        longitude = np.degrees(np.unwrap(np.unwrap(np.radians(longitude), axis=-1), axis=0))
        working = working.assign_coords(lon=(working.lon.dims, longitude))
    result = working.interp(x=axes["x"], y=axes["y"], method="linear")
    if "_orientation_cos" in result:
        result["structure_orientation"] = (0.5 * np.arctan2(result._orientation_sin, result._orientation_cos)) % np.pi
        result = result.drop_vars(["_orientation_cos", "_orientation_sin"])
    if "lon" in result.coords:
        result = result.assign_coords(lon=(result.lon.dims, ((result.lon.values + 180) % 360) - 180))
    return result


def _field(dataset, name: str, *, default: float = 0):
    import numpy as np

    shape = (dataset.sizes["time"], dataset.sizes["y"], dataset.sizes["x"])
    if name not in dataset:
        return np.full(shape, default, dtype=float)
    array = dataset[name]
    if set(array.dims) == {"y", "x"}:
        values = np.broadcast_to(array.transpose("y", "x").values, shape)
    elif set(array.dims) == {"time", "y", "x"}:
        values = array.transpose("time", "y", "x").values
    else:
        raise ValueError(f"{name} must have dimensions (time,y,x) or (y,x)")
    return np.asarray(values, dtype=float)


def _finite_field(dataset, name: str, default: float = 0):
    import numpy as np
    return np.nan_to_num(_field(dataset, name, default=default), nan=default, posinf=default, neginf=default)


def _wave_field(dataset, event: Mapping[str, Any]):
    import numpy as np

    shape = (dataset.sizes["time"], dataset.sizes["y"], dataset.sizes["x"])
    result = np.zeros(shape, dtype=float)
    wave = event.get("wave", {})
    if not isinstance(wave, Mapping) or wave.get("status") not in ("ok", "supported", "estimated", "valid"):
        return result, "unavailable"
    required = ("k_x_cycles_per_m", "k_y_cycles_per_m", "frequency_hz", "phase_rad")
    if any(not isinstance(wave.get(key), (int, float)) or not math.isfinite(wave[key]) for key in required):
        return result, "missing_phase_reference"
    if not wave.get("reference_time"):
        return result, "missing_phase_reference"
    try:
        reference = np.datetime64(str(wave["reference_time"]).removesuffix("Z"), "ns")
    except ValueError:
        return result, "invalid_phase_reference"
    if np.isnat(reference):
        return result, "invalid_phase_reference"
    confidence = wave.get("confidence", 0)
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        return result, "invalid_confidence"
    gain = np.clip(confidence, 0, 1) * _score(event, "wave", 1)
    xx, yy = np.meshgrid(dataset.x.values, dataset.y.values)
    spatial = wave["k_x_cycles_per_m"] * (xx - wave.get("reference_x_m", 0)) + wave["k_y_cycles_per_m"] * (yy - wave.get("reference_y_m", 0))
    for index, instant in enumerate(dataset.time.values):
        elapsed = float((instant - reference) / np.timedelta64(1, "s"))
        # Frequency is SIGNED according to the science FFT contract.
        phase = 2 * np.pi * (spatial + wave["frequency_hz"] * elapsed) + wave["phase_rad"]
        result[index] = gain * np.cos(phase)
    return result, "supported"


def _frame_events(dataset, event, timeline):
    """Resolve exact-time interpretations, never a nearest/future event.

    A timeless explicit event argument remains supported for small standalone
    artistic fixtures. Published runs supply time-bound event.json/timelines.
    """
    import numpy as np

    by_time = {}
    if timeline is not None:
        from ophanim.core.state import validate_timeline
        validate_timeline(timeline)
        for frame in timeline.get("frames", []):
            key = np.datetime64(str(frame["time"]).removesuffix("Z"), "ns")
            if np.isnat(key) or key in by_time:
                raise ValueError("event timeline has invalid or duplicate times")
            by_time[key] = frame
    records, statuses = [], []
    for instant in dataset.time.values.astype("datetime64[ns]"):
        frame = by_time.get(instant)
        if frame is not None:
            status = str(frame.get("status", "analyzed"))
            supported = status not in {"unsupported", "unavailable", "failed", "not_analyzed", "no_usable_data"}
            record = frame.get("event", {}) if supported else {}
            if not isinstance(record, Mapping):
                raise ValueError("timeline frame event must be a mapping")
        elif timeline is None and (not event.get("time") or instant == np.datetime64(str(event["time"]).removesuffix("Z"), "ns")):
            record, status = event, "explicit_static_event" if not event.get("time") else "analyzed"
        else:
            record, status = {}, "not_analyzed"
        records.append(record)
        statuses.append({"time": np.datetime_as_string(instant, unit="s") + "Z", "event_status": status})
    return records, statuses


def map_visual_fields(dynamic, event: Mapping[str, Any], style: VisualStyleConfig, *, event_timeline=None):
    """Generate deterministic artistic fields without modifying science input.

    ``dynamic`` uses UTC time and local x/y in metres. Input velocity describes
    apparent TEC-feature motion; the output LIC is seeded artistic texture.
    Unsupported inference channels contribute no corresponding visual detail.
    """
    import numpy as np
    import xarray as xr
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

    from ophanim.core.state import DynamicState, to_mapping_dataset
    producer_provenance = None
    if isinstance(dynamic, DynamicState):
        dynamic.validate()
        producer_provenance = _jsonable(dynamic.provenance)
        event_timeline = event_timeline if event_timeline is not None else dynamic.timeline
        event = dynamic.event
        dynamic = to_mapping_dataset(dynamic)
    if not isinstance(style, VisualStyleConfig) or not isinstance(event, Mapping):
        raise TypeError("mapping requires a VisualStyleConfig and event mapping")
    _validate_input(dynamic)
    source_digest = dataset_digest(dynamic)
    rendered = _render_grid(dynamic, style)
    frame_events, time_status = _frame_events(rendered, event, event_timeline)
    anomaly = _finite_field(rendered, "dtec_z")
    physical_anomaly = _field(rendered, "dtec")
    support_name = "observation_support" if "observation_support" in rendered else "observed_mask"
    support = np.clip(_finite_field(rendered, support_name), 0, 1)
    measured = _field(rendered, "tec") if "tec" in rendered else physical_anomaly
    support = np.where(np.isfinite(measured), support, 0)
    amplitude = np.tanh(np.abs(anomaly) / 2)
    anomaly_support = np.clip(_finite_field(rendered, "anomaly_confidence", default=1), 0, 1)
    amplitude *= anomaly_support
    density = support * (0.12 + 0.88 * amplitude)
    base_opacity = support * (style.quiet_opacity_floor + (style.base_opacity - style.quiet_opacity_floor) * amplitude)
    motion_channel = "flow_interpretation_confidence" if "flow_interpretation_confidence" in rendered else "unavailable"
    # Raw image-fit confidence is retained for research, but cannot authorize
    # interpreted motion by itself. Missing legacy admission is abstention,
    # not an invitation to restore directional fibers from a good-looking fit.
    flow_confidence = (np.minimum(np.clip(_finite_field(rendered, motion_channel), 0, 1),
                                 np.clip(_finite_field(rendered, "flow_confidence"), 0, 1))
                       if motion_channel != "unavailable" else np.zeros_like(support))
    legacy_motion_policy = ("not_applicable: interpretation-admissibility channel supplied" if motion_channel != "unavailable"
                            else "directional detail withheld: missing interpretation-admissibility channel; reanalysis required")
    u, v = _field(rendered, "flow_u", default=float("nan")), _field(rendered, "flow_v", default=float("nan"))
    flow_valid = (support > 0) & np.isfinite(u) & np.isfinite(v) & (flow_confidence > 0)
    flow_strength = support * flow_confidence * style.flow_texture_strength
    flow_strength = np.where(flow_valid & (np.hypot(u, v) > style.min_flow_speed_m_s), flow_strength, 0)
    texture = np.zeros_like(anomaly)
    dx, dy = float(np.diff(rendered.x.values)[0]), float(np.diff(rendered.y.values)[0])
    boundary_feather = np.ones_like(support)
    if style.boundary_feather_km > 0:
        for index in range(rendered.sizes["time"]):
            padded = np.pad(support[index] > 0, 1, constant_values=False)
            distance = distance_transform_edt(padded, sampling=(dy, dx))[1:-1, 1:-1]
            fraction = np.clip((distance - min(dx, dy)) / (style.boundary_feather_km * 1000), 0, 1)
            boundary_feather[index] = fraction**2 * (3 - 2 * fraction)
    density *= boundary_feather
    base_opacity *= boundary_feather
    flow_strength *= boundary_feather
    noise_rng = np.random.default_rng(style.seed)
    anchored_noise = noise_rng.random(anomaly.shape[1:])
    fiber_noise = anchored_noise.copy()
    anchored_detail = gaussian_filter(noise_rng.uniform(-1, 1, anomaly.shape[1:]), sigma=1.5, mode="reflect")
    anchored_detail /= max(1e-8, float(np.max(np.abs(anchored_detail))))
    detail_noise = anchored_detail.copy()
    noise = np.zeros_like(anomaly)
    pixel_y, pixel_x = np.indices(anomaly.shape[1:], dtype=float)
    for index in range(rendered.sizes["time"]):
        if index:
            elapsed = float((rendered.time.values[index] - rendered.time.values[index - 1]) / np.timedelta64(1, "s"))
            # Advect the artistic noise so animation does not flicker from
            # independent random seeds. Newly exposed edges get seeded noise.
            local_u = np.where(flow_valid[index], u[index], 0)
            local_v = np.where(flow_valid[index], v[index], 0)
            coordinates = [pixel_y - local_v * elapsed / dy, pixel_x - local_u * elapsed / dx]
            fiber_noise = map_coordinates(fiber_noise, coordinates, order=1, mode="constant", cval=np.nan)
            detail_noise = map_coordinates(detail_noise, coordinates, order=1, mode="constant", cval=np.nan)
            # A fixed spatial reservoir avoids frame-index-dependent births.
            fiber_noise = np.where(np.isfinite(fiber_noise), fiber_noise, anchored_noise)
            detail_noise = np.where(np.isfinite(detail_noise), detail_noise, anchored_detail)
        noise[index] = detail_noise
        texture[index] = line_integral_convolution(
            u[index], v[index], flow_valid[index], dx_m=dx, dy_m=dy,
            seed=(style.seed + index) % 2**32, steps=style.lic_streamline_steps,
            step_cells=style.lic_step_cells, contrast=style.lic_contrast,
            minimum_speed=style.min_flow_speed_m_s,
            seed_noise=fiber_noise,
        )
    # Nonlinear contrast creates sparse connected fibers rather than a mostly
    # uniform gray LIC overlay. Its position remains constrained by support.
    texture = np.where(flow_valid, np.clip((texture - 0.5) * 2.5 + 0.5, 0, 1) ** style.fiber_sharpness, 0)
    kinematics = np.tanh(style.kinematic_timescale_seconds * (
        np.abs(_finite_field(rendered, "flow_strain"))
        + 0.5 * np.abs(_finite_field(rendered, "flow_vorticity"))
        + np.maximum(0, -_finite_field(rendered, "flow_divergence"))))
    flow_strength *= (0.25 + 0.75 * amplitude) * (1 + style.kinematic_modulation * kinematics)
    flow_strength = np.clip(flow_strength, 0, 1)
    wave = np.zeros_like(anomaly)
    for index, frame_event in enumerate(frame_events):
        field, status = _wave_field(rendered.isel(time=slice(index, index + 1)), frame_event)
        wave[index] = field[0]
        time_status[index]["wave_status"] = status
        time_status[index]["flow_status"] = "supported" if np.any(flow_valid[index]) else "unsupported_or_low_confidence"
    wave_statuses = {record["wave_status"] for record in time_status}
    wave_status = next(iter(wave_statuses)) if len(wave_statuses) == 1 else "per_frame"
    wave *= support
    coherence = np.clip(_finite_field(rendered, "structure_coherence"), 0, 1)
    confidence = []
    for frame_event in frame_events:
        value = frame_event.get("event_confidence", 1 if frame_event else 0)
        confidence.append(float(np.clip(value, 0, 1)) if isinstance(value, (int, float)) and math.isfinite(value) else 0)
    event_confidence = np.asarray(confidence)[:, None, None]
    scores = {name: np.asarray([_score(item, name) for item in frame_events])[:, None, None]
              for name in ("front", "localized", "disturbed", "uncertain")}
    # A front folds across its observed band normal, not across an arbitrary
    # world axis. A cross-normal finite difference forms a localized ridge;
    # the axial orientation also guides fine ribbon texture along that ridge.
    gradient_y, gradient_x = np.gradient(anomaly, dy, dx, axis=(1, 2))
    fallback_tangent = (np.arctan2(gradient_y, gradient_x) + np.pi / 2) % np.pi
    orientation = _field(rendered, "structure_orientation", default=float("nan"))
    orientation = np.where(np.isfinite(orientation), orientation, fallback_tangent)
    front_ridge, ribbon_texture = np.zeros_like(anomaly), np.zeros_like(anomaly)
    normal_distance = style.front_width_km * 1000 / 2
    for index in range(rendered.sizes["time"]):
        if scores["front"][index, 0, 0] <= 0:
            continue
        nx, ny = -np.sin(orientation[index]), np.cos(orientation[index])
        above = map_coordinates(anomaly[index], [pixel_y + ny * normal_distance / dy, pixel_x + nx * normal_distance / dx], order=1, mode="nearest")
        below = map_coordinates(anomaly[index], [pixel_y - ny * normal_distance / dy, pixel_x - nx * normal_distance / dx], order=1, mode="nearest")
        front_ridge[index] = np.tanh(np.abs(above - below) / 2) * anomaly_support[index]
        tangent_valid = (support[index] > 0) & (coherence[index] > 0.1) & (front_ridge[index] > 0.01)
        ribbon_texture[index] = line_integral_convolution(np.cos(orientation[index]), np.sin(orientation[index]), tangent_valid,
            dx_m=dx, dy_m=dy, seed=style.seed, steps=style.lic_streamline_steps,
            step_cells=style.lic_step_cells, contrast=style.lic_contrast, seed_noise=anchored_noise)
    front_fold = scores["front"] * event_confidence * coherence * front_ridge
    front_gate = scores["front"] * event_confidence * coherence
    # Anomaly amplitude can vanish at the strongest front gradient. Illuminate
    # that observed ridge, not the two uniform anomaly plateaus. The broad
    # measured veil stays present, but is artistically subordinated to the fold.
    front_background = 1 - (1 - style.front_background_strength) * front_gate
    signed_fold = front_fold * np.tanh(anomaly)
    fold_shading = 1 + style.front_shading_strength * np.tanh(anomaly)
    ribbon_strength = support * boundary_feather * style.front_ribbon_strength * front_fold
    localized = scores["localized"] * event_confidence * amplitude**2
    residual = np.abs(_finite_field(rendered, "advection_residual"))
    nonzero = residual[(support > 0) & (residual > 0)]
    residual_scale = float(np.median(nonzero)) if nonzero.size else 1.0
    residual_support = np.tanh(residual / max(residual_scale, 1e-12))
    breakup_driver = np.clip(0.35 * (1 - support) + 0.50 * scores["disturbed"] * event_confidence + 0.15 * residual_support, 0, 1)
    breakup = support * style.breakup_strength * breakup_driver * (noise + 1) / 2
    height = support * (
        style.vertical_exaggeration_km * (
            0.4 * np.tanh(anomaly) + style.wave_strength * wave
            + style.front_fold_strength * signed_fold + style.localized_strength * localized
        ) + style.procedural_displacement_km * amplitude * noise
    ) * 1000
    # Base opacity remains tied to data support and anomaly. Uncertain motion
    # only weakens fibers; it cannot erase the supported base structure.
    emission = support * boundary_feather * style.emission_gain * (
        amplitude * front_background * np.clip(1 + 0.12 * wave + 0.15 * localized, 0, None)
        + style.front_emission_strength * front_fold * fold_shading)
    base_opacity = (base_opacity * front_background + support * boundary_feather * style.front_opacity_gain * front_fold)
    base_opacity *= np.clip(1 + 0.18 * style.wave_strength * wave, 0, 1.2)
    base_opacity = np.clip(base_opacity, 0, 0.95)
    # Polarity comes from the physical anomaly, not a centered robust z-score.
    hue_driver = np.sign(np.nan_to_num(physical_anomaly)) * amplitude
    display_emission = (emission + style.quiet_emission) / (emission + style.quiet_emission + style.emission_white_point)
    coordinates = {"time": rendered.time, "x": rendered.x, "y": rendered.y}
    for name in ("lat", "lon"):
        if name in rendered.coords:
            coordinates[name] = rendered.coords[name]
        elif name in rendered:
            coordinates[name] = rendered[name]
    output = xr.Dataset(coords=coordinates)
    variables = {
        "base_density": (density, "1", "artistic density from anomaly magnitude and observation support"),
        "emission": (emission, "1", "artistic emission from robust anomaly magnitude"),
        "display_emission": (display_emission, "1", "bounded artistic radiance shoulder; E/(E+white_point), including declared quiet illumination"),
        "hue_driver": (hue_driver, "1", "signed anomaly palette driver, not temperature"),
        "base_opacity": (base_opacity, "1", "artistic opacity; quiet floor deliberately retains a faint supported veil"),
        "height_displacement": (height, "m", "artistic vertical exaggeration; not measured height"),
        "flow_texture": (texture, "1", "seeded LIC artistic fibers following declared apparent signal-feature motion, not plasma velocity"),
        "flow_texture_strength": (flow_strength, "1", "artistic fiber strength gated by motion confidence"),
        "ribbon_texture": (ribbon_texture, "1", "artistic LIC along observed band tangents, not a velocity estimate"),
        "ribbon_strength": (ribbon_strength, "1", "front-supported artistic ribbon strength gated by event confidence"),
        "kinematic_modulation": (kinematics * flow_confidence, "1", "artistic detail modulation from apparent-feature strain, curl and convergence; not plasma physics"),
        "procedural_detail": (noise, "1", "seeded temporally advected artistic detail; stationary unsupported motion freezes rather than flickers"),
        "wave_modulation": (wave, "1", "artistic phase-coherent wave modulation gated by scientific wave confidence"),
        "breakup": (breakup, "1", "artistic fraying from support, advection residual and disturbed score"),
        "front_fold": (support * front_fold, "1", "artistic fold from front score and observed structural coherence"),
        "front_fold_displacement": (support * signed_fold, "1", "signed artistic cross-front fold from observed ridge; not physical uplift"),
        "localized_emphasis": (support * localized, "1", "artistic emphasis on compact anomaly support"),
        "render_support": (support, "1", "support resampled onto artistic render grid; not measurement accuracy"),
        "boundary_feather": (boundary_feather, "1", "artistic edge taper at render-region and missing-support boundaries; not observational confidence"),
    }
    for name, (values, units, semantic) in variables.items():
        output[name] = (("time", "y", "x"), np.asarray(values, dtype=np.float32))
        output[name].attrs.update(units=units, semantic=semantic, category="ARTISTIC")
    output.x.attrs.update(units="m", semantic="artistic render-grid eastward coordinate")
    output.y.attrs.update(units="m", semantic="artistic render-grid northward coordinate")
    event_digest = sha256(canonical_json(event).encode()).hexdigest()
    provenance = {
        "schema_version": "ophanim-visual/1", "mapping_version": "conditioned-veil/2",
        "input_content_sha256": source_digest, "event_sha256": event_digest,
        "science_run_id": dynamic.attrs.get("run_id", dynamic.attrs.get("science_run_id")),
        "source_provenance": _jsonable(dict(dynamic.attrs)), "style": asdict(style),
        "producer_provenance": producer_provenance,
        "signal_units": dynamic.attrs.get("signal_units", dynamic["dtec"].attrs.get("units", "unspecified legacy signal units")),
        "seed": style.seed, "wave_visual_status": wave_status,
        "motion_confidence_channel": motion_channel, "legacy_motion_policy": legacy_motion_policy,
        "event_timeline_sha256": sha256(canonical_json(event_timeline).encode()).hexdigest() if event_timeline else None,
        "time_status": time_status, "temporal_detail": "one seeded advected field with fixed spatial inflow reservoir; no per-frame reseeding",
        "semantics": "All fields are artistic transformations; shell altitude, sub-grid fibers and displacement are not observations.",
    }
    output.attrs.update(
        schema_version="ophanim-visual/1", category="ARTISTIC",
        visual_id=sha256(canonical_json(provenance).encode()).hexdigest(),
        input_content_sha256=source_digest, shell_altitude_km=style.shell_altitude_km,
        render_spacing_km=style.render_spacing_km,
        visual_provenance=canonical_json(provenance),
        wave_visual_status=wave_status,
        motion_confidence_channel=motion_channel, legacy_motion_policy=legacy_motion_policy,
        time_status=canonical_json(time_status),
    )
    return output
