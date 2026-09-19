"""Known-truth TEC image fields and explicit source-sampling degradation.

These are mathematical fixtures, not simulations of complete ionospheric
physics. They are always labeled synthetic and carry their generating truth.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ophanim.dynamics.regularize import utc64
from ophanim.dynamics.schemas import AnalysisError

KINDS = ("quiet", "translating_gaussian", "growing_gaussian", "translating_growing_gaussian", "plane_wave", "wave_packet", "moving_front", "crossing_waves", "disturbed", "missing_data")


def make_synthetic_dataset(kind="plane_wave", *, nx=64, ny=64, nt=49, spacing_m=25_000.0, cadence_s=300.0, amplitude_tecu=3.0, background_tecu=20.0, velocity_u_m_s=100.0, velocity_v_m_s=0.0, sigma_m=150_000.0, growth_rate_per_s=0.00003, wavelength_m=500_000.0, period_s=3600.0, bearing_deg=90.0, phase_rad=0.0, noise_std_tecu=0.0, missing_fraction=0.0, seed=42, start="2024-05-10T00:00:00Z"):
    """Return a labelled (time,y,x metre) fixture with known physical-unit truth.

    For the legacy moving_front recipe only, velocity_u_m_s is the signed
    normal speed along bearing_deg, not an eastward component; velocity_v_m_s
    does not affect this translationally invariant front. This compatibility
    convention is recorded separately from the resolved actual u/v truth.
    """
    aliases = {"translation": "translating_gaussian", "growth": "growing_gaussian", "translation_growth": "translating_growing_gaussian", "wave": "plane_wave", "front": "moving_front", "noise": "disturbed", "missing": "missing_data"}
    kind = aliases.get(kind, kind)
    if kind not in KINDS:
        raise AnalysisError(f"unknown synthetic kind {kind!r}; choose from {KINDS}")
    if min(nx, ny) < 3 or nt < 1 or min(spacing_m, cadence_s, sigma_m, wavelength_m, period_s) <= 0:
        raise AnalysisError("invalid synthetic dimensions or physical scales")
    if not 0 <= missing_fraction < 1 or noise_std_tecu < 0:
        raise AnalysisError("invalid synthetic missingness or noise")
    x, y = (np.arange(nx) - (nx - 1) / 2) * spacing_m, (np.arange(ny) - (ny - 1) / 2) * spacing_m
    seconds = np.arange(nt) * cadence_s
    centered_t = seconds - seconds[-1] / 2
    xx, yy = np.meshgrid(x, y)
    theta = np.deg2rad(bearing_deg)
    kx, ky = np.sin(theta) / wavelength_m, np.cos(theta) / wavelength_m
    phase = 2 * np.pi * (kx * xx[None] + ky * yy[None] - seconds[:, None, None] / period_s) + phase_rad
    gaussian_x = xx[None] - velocity_u_m_s * centered_t[:, None, None]
    gaussian_y = yy[None] - velocity_v_m_s * centered_t[:, None, None]
    translating = np.exp(-(gaussian_x**2 + gaussian_y**2) / (2 * sigma_m**2))
    stationary = np.exp(-(xx**2 + yy**2) / (2 * sigma_m**2))[None]
    gain = np.exp(growth_rate_per_s * centered_t[:, None, None])
    rng = np.random.default_rng(seed)
    if kind == "quiet":
        anomaly = np.zeros((nt, ny, nx))
    elif kind == "translating_gaussian":
        anomaly = amplitude_tecu * translating
    elif kind == "growing_gaussian":
        anomaly = amplitude_tecu * np.broadcast_to(stationary * gain, (nt, ny, nx))
    elif kind == "translating_growing_gaussian":
        anomaly = amplitude_tecu * translating * gain
    elif kind == "wave_packet":
        anomaly = amplitude_tecu * np.cos(phase) * translating
    elif kind == "moving_front":
        position = xx[None] * np.sin(theta) + yy[None] * np.cos(theta) - velocity_u_m_s * centered_t[:, None, None]
        anomaly = amplitude_tecu * np.tanh(position / (sigma_m / 3))
    elif kind == "crossing_waves":
        other = 2 * np.pi * (-ky * xx[None] + kx * yy[None] - seconds[:, None, None] / period_s)
        anomaly = amplitude_tecu * (np.cos(phase) + 0.9 * np.cos(other))
    elif kind == "disturbed":
        from scipy.ndimage import gaussian_filter
        anomaly = amplitude_tecu * gaussian_filter(rng.normal(size=(nt, ny, nx)), sigma=(0.5, 1, 1))
    else:
        anomaly = amplitude_tecu * np.cos(phase)
    tec = background_tecu + anomaly + rng.normal(0, noise_std_tecu, anomaly.shape)
    if kind == "missing_data" and missing_fraction == 0:
        missing_fraction = 0.25
    observed = rng.random(tec.shape) >= missing_fraction
    tec[~observed] = np.nan
    times = utc64(start) + np.rint(seconds * 1e9).astype("timedelta64[ns]")
    truth = dict(kind=kind, amplitude_tecu=amplitude_tecu, background_tecu=background_tecu, velocity_u_m_s=velocity_u_m_s, velocity_v_m_s=velocity_v_m_s, sigma_m=sigma_m, growth_rate_per_s=growth_rate_per_s, wavelength_m=wavelength_m, period_s=period_s, bearing_deg=bearing_deg, k_x_cycles_per_m=float(kx), k_y_cycles_per_m=float(ky), frequency_hz=-1 / period_s, phase_rad=phase_rad, seed=seed, noise_std_tecu=noise_std_tecu, missing_fraction=missing_fraction)
    if kind == "moving_front":
        truth.update(velocity_u_m_s=float(velocity_u_m_s * np.sin(theta)),
                     velocity_v_m_s=float(velocity_u_m_s * np.cos(theta)),
                     front_normal_speed_m_s=float(velocity_u_m_s), front_normal_bearing_deg=float(bearing_deg % 360),
                     motion_bearing_deg=float((bearing_deg + (180 if velocity_u_m_s < 0 else 0)) % 360),
                     generator_velocity_arguments={"velocity_u_m_s": velocity_u_m_s, "velocity_v_m_s": velocity_v_m_s},
                     velocity_semantics="resolved normal translation; legacy velocity_u argument supplies signed normal speed and velocity_v argument is unused; tangential front motion is unidentifiable")
    return xr.Dataset(
        data_vars={"tec": (("time", "y", "x"), tec, {"units": "TECU", "semantic_class": "synthetic"}), "observed_mask": (("time", "y", "x"), observed), "source_rms_tecu": (("time", "y", "x"), np.full_like(tec, noise_std_tecu)), "available_at": ("time", times)},
        coords={"time": times, "y": y, "x": x},
        attrs={"source_kind": "synthetic", "source_metadata": {"source_kind": "synthetic", "native_dx_m": spacing_m, "native_dy_m": spacing_m, "effective_resolution_m": spacing_m, "source_ids": [f"synthetic:{kind}:seed={seed}"], "projection_center_lat": 30.0, "projection_center_lon": -98.0}, "synthetic_truth": truth, "scientific_scope": "mathematical test field, not an ionospheric physics simulator"},
    )


def degrade_dataset(data, *, spatial_factor=1, temporal_stride=1, noise_std_tecu=0.0, missing_fraction=0.0, missing_frames=(), seed=42):
    """Emulate coarser native cells/cadence BEFORE analysis, retaining provenance.

    Block means model finite source support, not point-sampling an interpolated
    grid. Returned metadata describes this NEW coarse synthetic source. No
    upsampling is performed and interpolated readouts never count as evidence.

    Unknown source RMS remains unknown. With unknown error covariance, a block
    mean retains the mean of its input RMS proxies, not an unjustified sqrt(N)
    reduction. Added-noise RMS is added linearly by the L2 triangle inequality,
    making a conservative proxy bound without claiming independent errors.
    """
    if spatial_factor < 1 or temporal_stride < 1 or not isinstance(spatial_factor, int) or not isinstance(temporal_stride, int):
        raise AnalysisError("degradation factors must be positive integers")
    if not 0 <= missing_fraction < 1 or noise_std_tecu < 0:
        raise AnalysisError("invalid degradation noise or missingness")
    result = data.copy(deep=True)
    original = dict(data.attrs.get("source_metadata", {}))
    rms_name = next((name for name in ("source_rms_tecu", "source_rms") if name in result), None)
    inherited_rms = result[rms_name].copy(deep=True) if rms_name else xr.full_like(result.tec, np.nan)
    if inherited_rms.dims != result.tec.dims:
        raise AnalysisError("source RMS must match the synthetic TEC dimensions")
    inherited_rms = inherited_rms.where(np.isfinite(inherited_rms) & (inherited_rms >= 0))
    if spatial_factor > 1:
        values = result.tec.coarsen(x=spatial_factor, y=spatial_factor, boundary="trim").mean(skipna=False)
        masks = result.observed_mask.astype(float).coarsen(x=spatial_factor, y=spatial_factor, boundary="trim").min() == 1
        inherited_rms = inherited_rms.coarsen(x=spatial_factor, y=spatial_factor, boundary="trim").mean(skipna=False)
        fields = {"tec": values, "observed_mask": masks}
        if "available_at" in result:
            fields["available_at"] = result.available_at
        result = xr.Dataset(fields, attrs=dict(data.attrs))
    result = result.isel(time=slice(None, None, temporal_stride)).copy(deep=True)
    inherited_rms = inherited_rms.isel(time=slice(None, None, temporal_stride))
    if min(result.sizes["x"], result.sizes["y"]) < 3:
        raise AnalysisError("degradation leaves fewer than three spatial points")
    rng = np.random.default_rng(seed)
    values = result.tec.values + rng.normal(0, noise_std_tecu, result.tec.shape)
    mask = result.observed_mask.values & (rng.random(result.tec.shape) >= missing_fraction)
    for frame in missing_frames:
        if not 0 <= frame < len(mask):
            raise AnalysisError("missing frame index outside degraded sequence")
        mask[frame] = False
    values[~mask] = np.nan
    result["tec"] = (("time", "y", "x"), values, {"units": "TECU", "semantic_class": "synthetic"})
    result["observed_mask"] = (("time", "y", "x"), mask)
    propagated_rms = np.where(mask, inherited_rms.values + noise_std_tecu, np.nan)
    result["source_rms_tecu"] = (("time", "y", "x"), propagated_rms,
                                {"units": "TECU", "semantic_class": "synthetic",
                                 "uncertainty_model": "covariance_agnostic_RMS_proxy_upper_bound",
                                 "semantic": "block-mean inherited RMS plus added-noise RMS; no independence or sqrt(N) reduction assumed; unknown parent error remains NaN"})
    meta = dict(original)
    meta.update(native_dx_m=float(np.median(np.diff(result.x))), native_dy_m=float(np.median(np.diff(result.y))), effective_resolution_m=float(max(np.median(np.diff(result.x)), np.median(np.diff(result.y)))), source_kind="synthetic", source_ids=[f"degraded:{spatial_factor}:{temporal_stride}:{seed}"], parent_source_metadata=original)
    result.attrs.update(source_kind="synthetic", source_metadata=meta, degradation={"spatial_factor": spatial_factor, "temporal_stride": temporal_stride, "noise_std_tecu": noise_std_tecu, "missing_fraction": missing_fraction, "missing_frames": list(missing_frames), "seed": seed,
                        "uncertainty_model": "covariance_agnostic_RMS_proxy_upper_bound",
                        "uncertainty_assumptions": "input RMS proxies bound each input error in L2; block averaging uses triangle inequality; added errors need not be independent",
                        "temporal_sampling": "coarse synthetic-source point sampling; not an anti-alias filter"})
    return result
