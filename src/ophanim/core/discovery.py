"""Conservative, reusable native-grid event discovery, without model training.

Candidates are descriptive regions for investigation, not causal diagnoses.
Scores are robust deviations, NOT calibrated anomaly probabilities. Native
support, missing observations, temporal gaps and actual source uncertainty are
kept explicit. The same UTC at a fixed cell approximates the same solar time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import warnings

from .mapping import native_map


@dataclass(frozen=True)
class DiscoveryConfig:
    baseline_days: int = 7
    minimum_prior_days: int = 3
    scan_hours: float = 24.0
    minimum_deviation_tecu: float = 5.0
    minimum_scale_tecu: float = 2.0
    score_threshold: float = 3.0
    minimum_cells: int = 2
    minimum_epochs: int = 2
    max_candidates: int = 3
    maximum_cube_cells: int = 3_000_000

    def __post_init__(self):
        for name in ("baseline_days", "minimum_prior_days", "minimum_cells", "minimum_epochs", "max_candidates", "maximum_cube_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_prior_days > self.baseline_days or self.baseline_days > 30:
            raise ValueError("baseline requires minimum_prior_days <= baseline_days <= 30")
        if self.minimum_epochs < 2 or self.minimum_cells < 2 or self.max_candidates > 3:
            raise ValueError("discovery requires >=2 native cells, >=2 epochs, and <=3 candidates")
        for name in ("scan_hours", "minimum_deviation_tecu", "minimum_scale_tecu", "score_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.scan_hours > 72:
            raise ValueError("scan_hours cannot exceed 72")


def _stamp(value):
    import numpy as np
    return np.datetime_as_string(value, unit="s") + "Z"


def _identity(dataset, settings):
    import numpy as np

    digest = sha256()
    for name in ("time", "lat", "lon", "tec", "observed_mask", "source_rms_tecu", "available_at", "source_id"):
        if name not in dataset:
            continue
        values = np.asarray(dataset[name].values)
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        if values.dtype.kind in "OU":
            digest.update(json.dumps(values.tolist(), ensure_ascii=True).encode())
        else:
            if values.dtype.kind == "f":
                values = values.copy()
                values[~np.isfinite(values)] = np.nan
            digest.update(np.ascontiguousarray(values).tobytes())
    source_hash = digest.hexdigest()
    code_hash = sha256(Path(__file__).read_bytes() + Path(__file__).with_name("mapping.py").read_bytes()).hexdigest()
    payload = {"schema": "ophanim-discovery/1", "input_sha256": source_hash,
               "snapshot_id": dataset.attrs.get("snapshot_id"), "settings": settings, "code_sha256": code_hash}
    payload["scan_id"] = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]
    return payload


def _components(mask, *, wrap):
    """Four-connected original cells; dateline edges join only global grids."""
    import numpy as np
    from scipy.ndimage import label

    labels, count = label(mask)
    parents = list(range(count + 1))

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    if wrap:
        for row in range(mask.shape[0]):
            left, right = int(labels[row, 0]), int(labels[row, -1])
            if left and right:
                parents[find(right)] = find(left)
    groups = {}
    for y, x in zip(*np.nonzero(labels)):
        groups.setdefault(find(int(labels[y, x])), set()).add((int(y), int(x)))
    return list(groups.values())


def _neighbors(cells, shape, wrap):
    expanded = set(cells)
    for y, x in cells:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if wrap:
                nx %= shape[1]
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                expanded.add((ny, nx))
    return expanded


def _bounds(footprint, latitudes, longitudes):
    import numpy as np

    lat = [float(latitudes[y]) for y, _ in footprint]
    lon = sorted({float(longitudes[x]) for _, x in footprint})
    # Omit the largest longitude gap to find the shortest containing arc.
    if len(lon) > 1:
        gaps = np.diff(lon + [lon[0] + 360])
        cut = int(np.argmax(gaps))
        west, east = lon[(cut + 1) % len(lon)], lon[cut]
    else:
        west = east = lon[0]
    return {"south": min(lat), "north": max(lat), "west": west, "east": east,
            "crosses_dateline": west > east}


def survey_dataset(dataset, config=None):
    """Find up to three spatially distinct persistent native deviation regions.

    A baseline uses ONLY earlier distinct days at exactly matching UTC epochs
    (no temporal interpolation), computed per original grid cell. Daily cycle
    matching is not seasonal or solar-cycle correction. A short history is
    intentionally allowed to produce no candidates or insufficient support.
    """
    import numpy as np

    config = (DiscoveryConfig(**config) if isinstance(config, dict) else config) or DiscoveryConfig()
    if not isinstance(config, DiscoveryConfig):
        raise TypeError("config must be DiscoveryConfig or a mapping")
    if "tec" not in dataset or dataset.tec.dims != ("time", "lat", "lon"):
        raise ValueError("discovery requires native TEC dimensions (time, lat, lon)")
    if dataset.tec.size > config.maximum_cube_cells:
        raise ValueError("discovery input exceeds its native cube memory budget")
    if dataset.attrs.get("source_kind", "native") not in {"native", "synthetic"}:
        raise ValueError("discovery requires native source cells, not interpolated estimates")
    dataset = dataset.sortby("time")
    times = np.asarray(dataset.time.values, dtype="datetime64[us]")
    if np.any(np.isnat(times)) or len(np.unique(times)) != len(times):
        raise ValueError("source epochs must be finite and unique; resolve revisions before discovery")
    latitudes, longitudes = (np.asarray(dataset[name].values, dtype=float) for name in ("lat", "lon"))
    if (np.any(~np.isfinite(latitudes)) or np.any(~np.isfinite(longitudes))
            or np.any(np.diff(latitudes) <= 0) or np.any(np.diff(longitudes) <= 0)
            or np.any(np.abs(latitudes) > 90) or np.any(longitudes < -180) or np.any(longitudes >= 180)):
        raise ValueError("native coordinates must be unique increasing geographic axes; longitude in [-180,180)")
    metadata = dataset.attrs.get("source_metadata", {})
    for coordinate, key in ((latitudes, "native_lat_spacing_deg"), (longitudes, "native_lon_spacing_deg")):
        spacing = metadata.get(key)
        if spacing is not None and len(coordinate) > 1 and np.min(np.diff(coordinate)) < spacing * (1 - 1e-6):
            raise ValueError("grid is finer than original native source support; interpolated estimates cannot become candidates")
    values = np.asarray(dataset.tec.values, dtype=float).copy()
    if "observed_mask" in dataset:
        values[~np.asarray(dataset.observed_mask.values, dtype=bool)] = np.nan
    values[~np.isfinite(values)] = np.nan
    settings = asdict(config)
    identity = _identity(dataset, settings)
    intervals = np.diff(times).astype("timedelta64[s]").astype(float)
    cadence = float(np.median(intervals)) if len(intervals) else None
    result = {
        "schema_version": "ophanim-discovery/1", "scan_id": identity["scan_id"], "identity": identity,
        "status": "no_usable_data", "candidates": [],
        "source": {"start": _stamp(times[0]) if len(times) else None, "end": _stamp(times[-1]) if len(times) else None,
                   "cadence_seconds": cadence, "cadence_is_regular": bool(len(intervals) and np.allclose(intervals, cadence)),
                   "source_kind": dataset.attrs.get("source_kind", "native"), "epoch_count": len(times),
                   "native_lat_spacing_deg": metadata.get("native_lat_spacing_deg", float(np.median(np.diff(latitudes))) if len(latitudes) > 1 else None),
                   "native_lon_spacing_deg": metadata.get("native_lon_spacing_deg", float(np.median(np.diff(longitudes))) if len(longitudes) > 1 else None),
                   "snapshot_id": dataset.attrs.get("snapshot_id"), "source_ids": metadata.get("source_ids", []),
                   "effective_resolution_status": metadata.get("effective_resolution_status", "unknown")},
        "baseline": {"method": "past-days same-UTC per-native-cell median / MAD", "settings": settings,
                     "minimum_prior_days": config.minimum_prior_days, "score_semantics": "robust deviation; not probability",
                     "supported_cell_epochs": 0, "scanned_cell_epochs": 0},
        "map": native_map(dataset, values=values[-1] if len(times) else None),
        "warnings": list(dataset.attrs.get("acquisition_warnings", [])) + [
            "Experimental candidate screening, not calibrated detection or physical diagnosis.",
            "Short same-time history does not control season, solar cycle, or all ordinary variability.",
            "Native GIM nodes are estimates in a coarse product, not independent instrument measurements.",
        ],
    }
    if not len(times) or not np.isfinite(values).any():
        return result
    target_indices = np.flatnonzero(times > times[-1] - np.timedelta64(round(config.scan_hours * 3600), "s"))
    baselines = np.full((len(target_indices), *values.shape[1:]), np.nan)
    scales = np.full_like(baselines, np.nan)
    counts = np.zeros_like(baselines, dtype=np.int16)
    rms = (np.asarray(dataset.source_rms_tecu.values, dtype=float) if "source_rms_tecu" in dataset else np.full_like(values, np.nan))
    if not np.isfinite(rms[target_indices]).any():
        result["warnings"].append("Source RMS is unavailable; the explicit TECU scale floor is used, not invented certainty.")
    for local, index in enumerate(target_indices):
        age = (times[index] - times).astype("timedelta64[us]").astype(np.int64)
        day_us = 86_400_000_000
        prior = np.flatnonzero((age >= day_us) & (age <= config.baseline_days * day_us) & (age % day_us == 0))
        if not len(prior):
            continue
        history = values[prior]
        count = np.isfinite(history).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmedian(history, axis=0)
            spread = 1.4826 * np.nanmedian(np.abs(history - center), axis=0)
            historic_rms = np.nanmedian(np.where(rms[prior] >= 0, rms[prior], np.nan), axis=0)
        supported = count >= config.minimum_prior_days
        baselines[local] = np.where(supported, center, np.nan)
        counts[local] = count
        # Source RMS estimates are not confidence or known-independent errors;
        # max(current,historical) is an explicit conservative scale floor.
        source_scale = np.fmax(np.where(rms[index] >= 0, rms[index], np.nan), historic_rms)
        scales[local] = np.fmax(np.fmax(spread, source_scale), config.minimum_scale_tecu)
    target_values = values[target_indices]
    if not np.isfinite(target_values).any():
        result["warnings"].append("The latest screening window has no usable native observations; older history cannot substitute for current support.")
        return result
    deviations = target_values - baselines
    scores = deviations / scales
    result["baseline"]["supported_cell_epochs"] = int(np.isfinite(deviations).sum())
    result["baseline"]["scanned_cell_epochs"] = int(np.isfinite(target_values).sum())
    result["baseline"]["supported_fraction"] = float(np.isfinite(deviations).sum() / max(1, np.isfinite(target_values).sum()))
    result["baseline"]["prior_days_range"] = [int(counts.min()), int(counts.max())]
    supported_epochs = np.isfinite(deviations).sum(axis=(1, 2)) >= config.minimum_cells
    result["baseline"]["supported_epoch_count"] = int(supported_epochs.sum())
    longest = run = 0
    for local, supported in enumerate(supported_epochs):
        continuous = (not local or (times[target_indices[local]] - times[target_indices[local - 1]]).astype("timedelta64[s]").astype(float) <= 1.5 * (cadence or 0))
        run = (run + 1 if continuous else 1) if supported else 0
        longest = max(longest, run)
    if longest < config.minimum_epochs:
        result["status"] = "insufficient_baseline"
        result["warnings"].append(f"Need at least {config.minimum_prior_days} earlier distinct days at matching UTC times per native cell, supporting >= {config.minimum_cells} cells across >= {config.minimum_epochs} successive observed epochs.")
        return result
    result["status"] = "complete"
    if result["baseline"]["supported_fraction"] < 0.8:
        result["warnings"].append("Some observed cells lack enough same-time history and were excluded from candidate screening.")
    result["deviation_map"] = native_map(dataset, values=deviations[-1], field="deviation_tecu")
    spacing = float(np.median(np.diff(longitudes))) if len(longitudes) > 1 else 0
    wrap = bool(len(longitudes) > 2 and np.allclose(np.diff(longitudes), spacing) and abs(longitudes[-1] - longitudes[0] + spacing - 360) < 1e-5)
    tracks = []
    previous = []
    def track_root(track):
        while track.get("parent") is not None:
            track = track["parent"]
        return track

    for local, index in enumerate(target_indices):
        if local and (times[index] - times[target_indices[local - 1]]).astype("timedelta64[s]").astype(float) > 1.5 * (cadence or 0):
            previous = []
        current = []
        for polarity, sign in (("enhancement", 1), ("depletion", -1)):
            mask = (sign * deviations[local] >= config.minimum_deviation_tecu) & (sign * scores[local] >= config.score_threshold)
            for cells in _components(mask, wrap=wrap):
                if len(cells) < config.minimum_cells:
                    continue
                neighboring = _neighbors(cells, values.shape[1:], wrap)
                matches = [track_root(track) for old_cells, track in previous if track["polarity"] == polarity and neighboring & old_cells]
                if matches:
                    track = matches[0]
                    for other in matches[1:]:
                        if other is track:
                            continue
                        track["epochs"].update(other["epochs"])
                        track["cells"].update(other["cells"])
                        other["merged"] = True
                        other["parent"] = track
                        for epoch, old in other["members"].items():
                            track["members"].setdefault(epoch, set()).update(old)
                else:
                    track = {"polarity": polarity, "epochs": set(), "cells": set(), "members": {}, "merged": False}
                    tracks.append(track)
                track["epochs"].add(local)
                track["cells"].update(cells)
                track["members"].setdefault(local, set()).update(cells)
                current.append((cells, track))
        previous = current
    candidates = []
    for track in tracks:
        if track["merged"] or len(track["epochs"]) < config.minimum_epochs:
            continue
        epochs, cells = sorted(track["epochs"]), sorted(track["cells"])
        peak_local, peak_cell = max(((epoch, cell) for epoch, members in track["members"].items() for cell in members),
                                   key=lambda pair: abs(scores[pair[0]][pair[1]]))
        ys, xs = np.asarray(cells).T
        peak_score = float(abs(scores[peak_local][peak_cell]))
        peak_deviation = float(deviations[peak_local][peak_cell])
        start_time, end_time, peak_time = (_stamp(times[target_indices[v]]) for v in (epochs[0], epochs[-1], peak_local))
        duration = float((times[target_indices[epochs[-1]]] - times[target_indices[epochs[0]]]).astype("timedelta64[s]").astype(float) / 3600)
        footprint = [{"latitude": float(latitudes[y]), "longitude": float(longitudes[x])} for y, x in cells]
        candidate_id = sha256(json.dumps([identity["scan_id"], track["polarity"], start_time, end_time, footprint], sort_keys=True).encode()).hexdigest()[:24]
        series = []
        for local in range(len(target_indices)):
            observed, baseline = target_values[local, ys, xs], baselines[local, ys, xs]
            supported = np.isfinite(observed) & np.isfinite(baseline)
            series.append({"time": _stamp(times[target_indices[local]]),
                           "observed_tecu": float(observed[supported].mean()) if supported.any() else None,
                           "baseline_tecu": float(baseline[supported].mean()) if supported.any() else None,
                           "deviation_tecu": float((observed[supported] - baseline[supported]).mean()) if supported.any() else None,
                           "supported_native_cells": int(supported.sum())})
        circular = np.mean(np.exp(1j * np.radians(longitudes[xs])))
        centroid_lon = math.degrees(math.atan2(float(circular.imag), float(circular.real)))
        if centroid_lon >= 180:
            centroid_lon -= 360
        source_rms = rms[target_indices[peak_local]][peak_cell]
        candidates.append({
            "candidate_id": candidate_id, "title": f"Persistent TEC {track['polarity']}", "polarity": track["polarity"],
            "start_time": start_time, "end_time": end_time, "peak_time": peak_time,
            "bounds": _bounds(cells, latitudes, longitudes),
            "centroid": {"latitude": float(latitudes[ys].mean()), "longitude": centroid_lon},
            "peak_deviation_tecu": peak_deviation, "peak_score": peak_score,
            "native_cell_count": len(cells), "observed_epoch_count": len(epochs), "duration_hours": duration,
            "source_rms_tecu_at_peak": float(source_rms) if np.isfinite(source_rms) and source_rms >= 0 else None,
            "evidence_strength": "moderate" if len(epochs) >= 3 and len(cells) >= 3 else "limited",
            "evidence_semantics": "heuristic persistence/support, not probability or independent validation",
            "hypotheses": [
                {"label": f"Persistent regional {track['polarity']}",
                 "evidence": f"{len(cells)} native cells across {len(epochs)} epochs differ from a past-days same-UTC reference; peak {peak_deviation:+.1f} TECU.",
                 "limitations": "Descriptive pattern only. Neighboring gridded cells are correlated; small structures and physical cause are unresolved."},
                {"label": "Ordinary variability or imperfect short baseline",
                 "evidence": "The reference matches UTC and fixed location, but contains only a small number of prior days.",
                 "limitations": "Seasonal, solar-cycle, geomagnetic and day-to-day differences are not separately identified."},
                {"label": "Source-product or coverage effects",
                 "evidence": "Native source support, missing epochs and source RMS are retained in this screening.",
                 "limitations": "RMS and persistence cannot exclude common-mode estimation errors. Physical cause remains unknown without independent evidence."},
            ],
            "time_series": series, "footprint": footprint,
            "interpretation": "candidate for investigation; no causal diagnosis",
        })
    candidates.sort(key=lambda item: (-item["peak_score"] * math.sqrt(min(item["native_cell_count"], 25)) * math.sqrt(item["observed_epoch_count"]), item["candidate_id"]))
    selected = []
    for candidate in candidates:
        cells = {(p["latitude"], p["longitude"]) for p in candidate["footprint"]}
        duplicate = False
        for other in selected:
            existing = {(p["latitude"], p["longitude"]) for p in other["footprint"]}
            if len(cells & existing) / min(len(cells), len(existing)) > 0.25:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) == config.max_candidates:
            break
    result["candidates"] = selected
    # This catches accidental ndarray/scalar/NaN leakage at the public boundary.
    json.dumps(result, allow_nan=False)
    return result
