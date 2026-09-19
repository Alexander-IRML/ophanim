"""Observation eligibility, revision selection and explicit time interpolation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr

from .schemas import AnalysisConfig, AnalysisError


def utc64(value) -> np.datetime64:
    """Normalize an ISO timestamp to timezone-naive datetime64 representing UTC."""
    if isinstance(value, str):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(stamp, "ns")
    return np.datetime64(value, "ns")


def iso_time(value) -> str:
    return np.datetime_as_string(utc64(value), unit="s") + "Z"


def eligible_sequence(raw: xr.Dataset, config: AnalysisConfig) -> xr.Dataset:
    """Freeze the eligible source revisions before using values or timestamps.

    Causal mode requires actual availability metadata. Observation time alone
    cannot establish when a revised historical product was available.
    """
    if "tec" not in raw or "time" not in raw.coords or "observed_mask" not in raw:
        raise AnalysisError("input needs tec, observed_mask and time coordinates")
    data = raw.copy(deep=True)
    if data.tec.dims not in (("time", "lat", "lon"), ("time", "y", "x")):
        raise AnalysisError("tec dimensions must be (time,lat,lon) or (time,y,x)")
    if data.observed_mask.dims != data.tec.dims:
        raise AnalysisError("observed_mask dimensions must match tec")
    times = np.asarray([utc64(t) for t in data.time.values])
    if np.isnat(times).any() or not len(times):
        raise AnalysisError("empty or invalid observation timestamps")
    data = data.assign_coords(time=times)
    available = None
    if "available_at" in data:
        if data.available_at.dims != ("time",):
            raise AnalysisError("available_at must be one timestamp per frame")
        available = np.asarray([utc64(t) for t in data.available_at.values])
        if np.isnat(available).any():
            raise AnalysisError("available_at contains unknown timestamps")
        data["available_at"] = ("time", available)
    if config.analysis_mode == "causal" and available is None:
        raise AnalysisError("causal analysis requires actual available_at metadata")
    eligible = np.ones(len(times), dtype=bool)
    if config.as_of:
        cutoff = utc64(config.as_of)
        eligible &= times <= cutoff
        if available is not None:
            eligible &= available <= cutoff
    if config.target_time:
        target = utc64(config.target_time)
        # A retrospective window may use its explicit future observations;
        # causal targets, however, are themselves information cutoffs.
        if config.analysis_mode == "causal":
            eligible &= times <= target
            eligible &= available <= target
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise AnalysisError("no observations eligible at the information cutoff")
    # A revision is selected by its actual availability, never by input ordering.
    selected = []
    for stamp in np.unique(times[indices]):
        group = indices[times[indices] == stamp]
        if len(group) > 1:
            if available is None:
                if not all(np.array_equal(data.tec.values[group[0]], data.tec.values[j], equal_nan=True) and np.array_equal(data.observed_mask.values[group[0]], data.observed_mask.values[j]) for j in group[1:]):
                    raise AnalysisError("conflicting duplicate timestamps require available_at revision metadata")
                chosen = group[0]
            else:
                latest = group[available[group] == available[group].max()]
                if len(latest) > 1 and not all(np.array_equal(data.tec.values[latest[0]], data.tec.values[j], equal_nan=True) for j in latest[1:]):
                    raise AnalysisError("conflicting revisions have identical availability")
                chosen = latest[-1]
        else:
            chosen = group[0]
        selected.append(chosen)
    data = data.isel(time=selected)
    valid = data.observed_mask.astype(bool) & np.isfinite(data.tec)
    data["tec"] = data.tec.where(valid)
    data["observed_mask"] = valid
    data.attrs["source_observation_times"] = [iso_time(t) for t in data.time.values[np.any(valid.values.reshape(len(selected), -1), axis=1)]]
    data.attrs["analysis_mode"] = config.analysis_mode
    data.attrs["as_of"] = config.as_of or ""
    data.attrs["duplicate_revision_count"] = int(len(indices) - len(selected))
    return data


def regularize_time(data: xr.Dataset, config: AnalysisConfig) -> xr.Dataset:
    """Use regular time coordinates, retaining support and interpolation masks.

    Retrospective interpolation is bounded linear interpolation. Causal mode
    only carries a past observation forward and labels it interpolated; it never
    uses a later frame to fill an earlier timestamp. Temporal decimation is
    rejected: selecting fewer source frames without an explicit anti-alias
    filter can reverse apparent wave propagation. This is not a low-pass API.
    """
    old = data.time.values.astype("datetime64[ns]")
    if len(old) < 2:
        result = data.copy(deep=True)
        result["temporal_interpolated"] = xr.zeros_like(result.observed_mask, dtype=bool)
        result.attrs["cadence_seconds"] = None
        return result
    seconds = (old - old[0]).astype("timedelta64[ns]").astype(float) / 1e9
    source_cadence = float(np.median(np.diff(seconds)))
    cadence = config.cadence_seconds or source_cadence
    if not np.isfinite(cadence) or cadence <= 0:
        raise AnalysisError("invalid time cadence")
    if cadence > source_cadence * (1 + 1e-9):
        raise AnalysisError("temporal downsampling is unsupported without an explicit anti-alias filter; retain the source cadence or supply a separately filtered source product")
    new_seconds = np.arange(int(np.floor(seconds[-1] / cadence + 1e-9)) + 1) * cadence
    new_times = old[0] + np.rint(new_seconds * 1e9).astype("timedelta64[ns]")
    if len(new_times) * data.sizes["x"] * data.sizes["y"] > config.max_cube_cells:
        raise AnalysisError("regularized cube exceeds max_cube_cells; select a smaller window or coarser science grid")
    result = data.drop_dims("time").assign_coords(time=new_times)
    shape = (len(new_times), data.sizes["y"], data.sizes["x"])
    observed = np.zeros(shape, bool)
    interpolated = np.zeros(shape, bool)
    numeric = {name: np.full(shape, np.nan, dtype=float) for name, var in data.data_vars.items() if var.dims == ("time", "y", "x") and var.dtype.kind not in "b"}
    spatial = np.zeros(shape, bool)
    availability = np.full(len(new_times), np.datetime64("NaT", "ns"))
    for i, t in enumerate(new_seconds):
        pos = int(np.searchsorted(seconds, t))
        exact = pos < len(seconds) and abs(seconds[pos] - t) < 1e-6
        if exact:
            observed[i] = data.observed_mask.values[pos]
            for name in numeric:
                numeric[name][i] = data[name].values[pos]
            if "spatial_interpolated" in data:
                spatial[i] = data.spatial_interpolated.values[pos]
            if "available_at" in data:
                availability[i] = data.available_at.values[pos]
            continue
        if pos == 0:
            continue
        left = pos - 1
        if config.analysis_mode == "causal":
            if t - seconds[left] > config.maximum_interpolation_gap_seconds:
                continue
            support = data.observed_mask.values[left]
            for name in numeric:
                numeric[name][i] = np.where(support, data[name].values[left], np.nan)
            if "spatial_interpolated" in data:
                spatial[i] = data.spatial_interpolated.values[left]
            if "available_at" in data:
                availability[i] = data.available_at.values[left]
        else:
            if pos >= len(seconds) or seconds[pos] - seconds[left] > config.maximum_interpolation_gap_seconds:
                continue
            support = data.observed_mask.values[left] & data.observed_mask.values[pos]
            fraction = (t - seconds[left]) / (seconds[pos] - seconds[left])
            for name in numeric:
                numeric[name][i] = np.where(support, (1 - fraction) * data[name].values[left] + fraction * data[name].values[pos], np.nan)
            if "spatial_interpolated" in data:
                spatial[i] = data.spatial_interpolated.values[left] | data.spatial_interpolated.values[pos]
            if "available_at" in data:
                availability[i] = max(data.available_at.values[left], data.available_at.values[pos])
        interpolated[i] = support
    for name, values in numeric.items():
        result[name] = (("time", "y", "x"), values, data[name].attrs)
    result["observed_mask"] = (("time", "y", "x"), observed)
    result["temporal_interpolated"] = (("time", "y", "x"), interpolated)
    result["spatial_interpolated"] = (("time", "y", "x"), spatial)
    if "available_at" in data:
        result["available_at"] = ("time", availability)
    result.attrs.update(data.attrs)
    result.attrs["cadence_seconds"] = float(cadence)
    result.attrs["regularization_method"] = "past_hold" if config.analysis_mode == "causal" else "bounded_linear"
    result.attrs["temporal_downsampling_policy"] = "reject_without_explicit_antialias_filter"
    return result
