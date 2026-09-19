"""Inspectable event heuristics, deliberately not calibrated probabilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label

from .regularize import iso_time, utc64


def _finite_mean(values, default=0.0):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else default


def _morphology(mask, x, y, weights=None):
    """Physical-area compactness and weighted second-moment elongation."""
    cells = int(mask.sum())
    if not cells:
        return {"cell_count": 0, "compactness": 0.0, "elongation": 1.0,
                "area_km2": 0.0, "touches_boundary": False}
    dx, dy = float(np.median(np.diff(x))), float(np.median(np.diff(y)))
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = (np.count_nonzero(np.diff(padded, axis=1)) * dy
                 + np.count_nonzero(np.diff(padded, axis=0)) * dx)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx[mask], yy[mask]))
    mass = np.ones(cells) if weights is None else np.maximum(np.asarray(weights)[mask], 1e-30)
    center = np.average(points, axis=0, weights=mass)
    centered = points - center
    covariance = (centered * mass[:, None]).T @ centered / mass.sum()
    covariance += np.diag([dx * dx / 12, dy * dy / 12])
    eigenvalues = np.linalg.eigvalsh(covariance)
    area = cells * dx * dy
    return {"cell_count": cells, "area_km2": area / 1e6,
            "compactness": float(np.clip(4 * np.pi * area / max(perimeter**2, 1e-30), 0, 1)),
            "elongation": float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-30))),
            "touches_boundary": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())}


def _event_region(frame, config):
    a = np.abs(frame.dtec.values)
    available = np.isfinite(a)
    permitted = available.copy()
    bounds = config.event_focus_bounds
    if bounds is not None:
        if "lat" not in frame or "lon" not in frame:
            raise ValueError("event_focus_bounds requires geographic coordinates")
        latitude, longitude = np.asarray(frame.lat), np.asarray(frame.lon)
        longitude = (longitude + 180) % 360 - 180
        allowed_lon = ((longitude >= bounds["west"]) & (longitude <= bounds["east"])) if bounds["west"] <= bounds["east"] else ((longitude >= bounds["west"]) | (longitude <= bounds["east"]))
        geographic = ((latitude >= bounds["south"]) & (latitude <= bounds["north"]) & allowed_lon)
        if not geographic.any():
            raise ValueError("event_focus_bounds contains no scientific grid cells")
        permitted &= geographic
    peak = float(np.max(a[permitted])) if permitted.any() else 0.0
    threshold = max(config.anomaly_scale_tecu, 0.25 * peak)
    objects, count = label(permitted & (a >= threshold))
    object_mask = np.zeros_like(available)
    if count:
        strength = np.bincount(objects.ravel(), weights=np.nan_to_num(a).ravel(), minlength=count + 1)
        strength[0] = -1
        object_mask = objects == int(np.argmax(strength))
    if bounds is not None:
        focus = permitted
        method = "explicit_candidate_bounds"
    elif object_mask.any():
        focus = binary_dilation(object_mask, iterations=1) & available
        method = "dominant_connected_anomaly_with_one_cell_context"
    else:
        focus = available
        method = "whole_region_no_resolved_anomaly_object"
    return focus, object_mask, {"selection": method, "bounds": bounds,
                                "focus_cell_count": int(focus.sum()), "analysis_cell_count": int(available.sum()),
                                "component_count": int(count), "anomaly_threshold_tecu": threshold}


def score_events(data, wave, config, baseline_status, flow_records):
    target = utc64(config.target_time) if config.target_time else data.time.values[-1]
    eligible = np.flatnonzero(data.time.values <= target)
    index = int(eligible[-1]) if len(eligible) else 0
    frame = data.isel(time=index)
    focus, object_mask, region = _event_region(frame, config)
    def mean(values, default=0.0):
        array = np.asarray(values)
        return _finite_mean(array[focus], default) if focus.any() else default
    coverage = _finite_mean(frame.observation_support)
    anomaly_support = mean(frame.anomaly_confidence)
    measurement = mean(frame.measurement_reliability, float("nan")) if "measurement_reliability" in frame else float("nan")
    measurement_factor = measurement if np.isfinite(measurement) else 1.0
    evidence_support = anomaly_support * measurement_factor
    a = np.abs(frame.dtec.values)
    finite = np.isfinite(a)
    anomaly = np.tanh(mean(a) / config.anomaly_scale_tecu)
    temporal = np.tanh(mean(np.abs(frame.dtec_dt.values)) * 60 / config.temporal_scale_tecu_per_minute)
    gradient_values = np.asarray(frame.grad_mag)
    peak_gradient = float(np.nanmax(gradient_values[focus])) if focus.any() and np.isfinite(gradient_values[focus]).any() else 0.0
    gradient_mask = focus & np.isfinite(gradient_values) & (gradient_values >= max(peak_gradient * 0.25, 1e-12))
    gradient = np.tanh(_finite_mean(gradient_values[gradient_mask]) * 1000 / config.gradient_scale_tecu_per_km)
    coherence = _finite_mean(np.asarray(frame.structure_coherence)[gradient_mask])
    speed = np.tanh(mean(frame.flow_speed) / config.flow_speed_scale_m_s)
    raw_flow_confidence = mean(frame.flow_confidence)
    flow_confidence = mean(frame.flow_interpretation_confidence) if "flow_interpretation_confidence" in frame else 0.0
    residual_values = np.abs(frame.advection_residual.values)
    residual_available = bool(np.isfinite(residual_values[focus]).any())
    residual = np.tanh(mean(residual_values) * 60 / config.temporal_scale_tecu_per_minute)
    growth_ambiguity = mean(frame.growth_ambiguity) if "growth_ambiguity" in frame else 0.0
    wave_score = float(wave.get("confidence", 0)) if wave.get("status") == "estimated" else 0.0
    morphology = _morphology(object_mask, data.x.values, data.y.values)
    front_morphology = _morphology(gradient_mask, data.x.values, data.y.values, gradient_values)
    support_fraction = float(object_mask.sum() / max(1, finite.sum()))
    compactness = np.sqrt(morphology["compactness"] / max(morphology["elongation"], 1))
    localized = evidence_support * anomaly * compactness * np.sqrt(1 - support_fraction) * (1 - wave_score)
    if morphology["touches_boundary"]:
        localized *= 0.5  # An edge-clipped feature does not establish compactness.
    quiet = evidence_support * (1 - anomaly) * (1 - temporal) * (1 - gradient) * (1 - wave_score)
    flow = flow_confidence * speed * (1 - residual) * (1 - 0.5 * wave_score) if residual_available else 0.0
    raw_residual_values = np.abs(frame.advection_residual_raw.values) if "advection_residual_raw" in frame else np.full_like(a, np.nan)
    raw_residual = np.tanh(mean(raw_residual_values) * 60 / config.temporal_scale_tecu_per_minute)
    raw_flow_score = raw_flow_confidence * speed * (1 - raw_residual) * (1 - 0.5 * wave_score) if np.isfinite(raw_residual_values[focus]).any() else 0.0
    elongation = np.clip((front_morphology["elongation"] - 1) / 3, 0, 1)
    front = evidence_support * gradient * coherence * elongation * (1 - 0.7 * wave_score)
    disturbed = evidence_support * np.clip((temporal + gradient * (1 - coherence) + residual + temporal * growth_ambiguity) / 3, 0, 1)
    # Missing anomalies cannot be mislabeled quiet. Unsupported optional wave
    # analysis does not, by itself, erase good observed macrostructure.
    interpretability = max(quiet, flow, wave_score, front, localized, disturbed)
    uncertain = max(1 - evidence_support, min(1, anomaly) * (1 - interpretability))
    scores = dict(quiet=quiet, flow=flow, wave=wave_score, front=front, localized=localized, disturbed=disturbed, uncertain=uncertain)
    scores = {name: float(np.clip(value, 0, 1)) for name, value in scores.items()}
    label = max(scores, key=scores.get)
    aliases = {"wave": "wave_like", "flow": "flow_dominant", "front": "front_like", "localized": "localized_anomaly"}
    supported_scores = sorted((quiet, flow, wave_score, front, localized, disturbed), reverse=True)
    margin = float((supported_scores[0] - supported_scores[1]) / max(supported_scores[0], 1e-12))
    matching_flow = [record for record in flow_records if record.get("time") is not None and utc64(record["time"]) == data.time.values[index]]
    return {
        "schema_version": "1.0", "time": iso_time(data.time.values[index]), "analysis_mode": config.analysis_mode,
        "as_of": config.as_of, "wave": wave, "event_scores": scores, "display_class": aliases.get(label, label),
        "event_confidence": float(np.clip(evidence_support * (1 - uncertain) * (0.5 + 0.5 * margin), 0, 1)),
        "region": region, "morphology": {"anomaly_object": morphology, "gradient_structure": front_morphology},
        "research_motion": {"raw_mean_flow_confidence": raw_flow_confidence, "raw_flow_score": float(raw_flow_score),
                            "semantics": "raw feature-fit evidence retained for research; not eligible downstream motion interpretation or validated accuracy"},
        "score_semantics": "independent uncalibrated heuristic scores; not probabilities and not required to sum to one",
        "channel_status": {"observations": "estimated" if coverage else "invalid_support", "baseline": baseline_status, "flow": matching_flow[-1] if matching_flow else {"status": "insufficient_samples"}, "wave": wave.get("status", "unavailable")},
        "metrics": {"observation_support": coverage, "anomaly_support": anomaly_support, "measurement_reliability": float(measurement) if np.isfinite(measurement) else None, "measurement_reliability_semantics": "unknown remains null; heuristic RMS sensitivity, not probability", "anomaly_strength": float(anomaly), "regional_mean_absolute_anomaly_tecu": _finite_mean(a), "focused_mean_absolute_anomaly_tecu": mean(a), "temporal_change": float(temporal), "gradient_strength": float(gradient), "structure_coherence": coherence, "mean_flow_confidence": flow_confidence, "residual_strength": float(residual) if residual_available else None, "residual_interpretable": residual_available, "growth_ambiguity": growth_ambiguity, "interpretation_margin": margin},
        "limitations": ["Candidate-conditioned morphology is a heuristic description, not a physical cause", "Display labels depend on the declared focus and available context", "Scores and component margins are not calibrated probabilities"],
        "flow_pairs": flow_records,
    }
