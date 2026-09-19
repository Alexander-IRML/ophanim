"""Mask-aware smoothing, replaceable baselines and physical anomaly fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, minimum_filter1d
from scipy.signal import savgol_coeffs, savgol_filter

from .schemas import AnalysisConfig, AnalysisError


def masked_gaussian(values, sigma, minimum_support=0.8):
    """Normalized convolution with an explicit retained support fraction.

    This function never promotes a missing center into a source observation.
    Neighboring output is valid only above the supplied support threshold.
    """
    valid = np.isfinite(values)
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="constant", cval=0)
    support = gaussian_filter(valid.astype(float), sigma=sigma, mode="constant", cval=0)
    # The computational boundary itself is not a data hole: renormalize by
    # the fraction of the kernel falling within the supplied finite domain.
    domain = gaussian_filter(np.ones_like(values, dtype=float), sigma=sigma, mode="constant", cval=0)
    fraction = np.divide(support, domain, out=np.zeros_like(support), where=domain > 0)
    output = np.divide(numerator, support, out=np.full_like(numerator, np.nan), where=support > 0)
    output[~valid | (fraction < minimum_support)] = np.nan
    return output, fraction


class BaselineEstimator(Protocol):
    """A deterministic TECU baseline; no learned-monitor dependency."""

    def estimate(self, tec: xr.DataArray) -> xr.DataArray: ...


def baseline_window_samples(tec, window_minutes, polyorder=0):
    if tec.sizes["time"] < 2:
        raise AnalysisError("baseline needs more than one observation")
    seconds = np.diff(tec.time.values).astype("timedelta64[ns]").astype(float) / 1e9
    cadence = float(np.median(seconds))
    window = int(np.floor(window_minutes * 60 / cadence))
    if window % 2 == 0:
        window -= 1
    if window < max(3, polyorder + 2):
        raise AnalysisError("baseline window has too few samples for a nontrivial fitted background")
    if window > tec.sizes["time"]:
        raise AnalysisError("baseline window is larger than the available sequence")
    return window


@dataclass(frozen=True)
class SavitzkyGolayBaseline:
    window_minutes: float = 110.0
    polyorder: int = 2
    causal: bool = False

    def estimate(self, tec):
        window = baseline_window_samples(tec, self.window_minutes, self.polyorder)
        values = np.asarray(tec.values, float)
        valid = np.isfinite(values)
        if self.causal:
            result = np.full_like(values, np.nan)
            windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
            coefficients = savgol_coeffs(window, self.polyorder, pos=window - 1, use="dot")
            result[window - 1:] = np.tensordot(windows, coefficients, axes=([-1], [0]))
        else:
            result = savgol_filter(np.where(valid, values, 0), window, self.polyorder, axis=0, mode="interp")
            complete = minimum_filter1d(valid.astype(np.uint8), size=window, axis=0, mode="nearest").astype(bool)
            complete[:window // 2] &= np.all(valid[:window], axis=0)
            complete[-window // 2:] &= np.all(valid[-window:], axis=0)
            result[~complete] = np.nan
        return xr.DataArray(result, dims=tec.dims, coords=tec.coords, attrs={"units": "TECU", "semantic_class": "derived", "method": "causal_savgol" if self.causal else "centered_savgol", "window_samples": window})


@dataclass(frozen=True)
class RollingMedianBaseline:
    window_minutes: float = 110.0
    causal: bool = False

    def estimate(self, tec):
        window = baseline_window_samples(tec, self.window_minutes)
        result = tec.rolling(time=window, center=not self.causal, min_periods=window).median()
        result.attrs.update(units="TECU", semantic_class="derived", method="causal_rolling_median" if self.causal else "centered_rolling_median", window_samples=window)
        return result


def preprocess(data: xr.Dataset, config: AnalysisConfig):
    result = data.copy(deep=True)
    dims = ("time", "y", "x")
    valid = np.isfinite(data.tec.values)
    observed = data.observed_mask.values.astype(bool)
    temporal = data.temporal_interpolated.values.astype(bool)
    spatial = data.spatial_interpolated.values.astype(bool)
    support = np.where(observed, 1.0, np.where(temporal, config.temporal_interpolation_weight, 0.0))
    support *= np.where(spatial, config.spatial_interpolation_weight, 1.0)
    support[~valid] = 0
    for name in ("observation_support", "obs_confidence"):
        result[name] = (dims, support, {"units": "1", "semantic_class": "derived", "semantic": "heuristic source coverage/interpolation support, NOT provider accuracy or calibrated probability"})
    uncertainty_name = next((name for name in ("source_rms_tecu", "source_rms") if name in data), None)
    uncertainty = np.asarray(data[uncertainty_name].values, float).copy() if uncertainty_name else np.full_like(data.tec.values, np.nan, dtype=float)
    uncertainty[(uncertainty < 0) | ~valid] = np.nan
    synthetic = data.attrs.get("source_kind") == "synthetic"
    result["source_uncertainty"] = (dims, uncertainty, {"units": "TECU", "semantic_class": "synthetic" if synthetic else "derived" if spatial.any() or temporal.any() else "measured", "semantic": "supplied RMS proxy; NaN means unknown, interpolated RMS is not propagated measurement error"})
    result["source_uncertainty_known"] = (dims, np.isfinite(uncertainty), {"units": "1", "semantic_class": "derived", "semantic": "whether an RMS proxy was supplied; unknown is not zero uncertainty"})
    dx, dy = float(np.median(np.diff(data.x))), float(np.median(np.diff(data.y)))
    sigma = (0, config.smooth_sigma_km * 1000 / dy, config.smooth_sigma_km * 1000 / dx)
    smooth, fraction = masked_gaussian(data.tec.values, sigma, config.smoothing_minimum_support)
    result["tec_smooth"] = (dims, smooth, {"units": "TECU", "semantic_class": "derived", "semantic": "mask-aware Gaussian-smoothed source product", "sigma_m": config.smooth_sigma_km * 1000})
    result["smoothing_support"] = (dims, fraction, {"units": "1", "semantic_class": "derived"})
    status = {"status": "estimated", "method": config.baseline_method}
    try:
        if config.baseline_method == "constant":
            baseline = xr.full_like(result.tec_smooth, config.constant_background_tecu).where(result.tec_smooth.notnull())
            baseline.attrs.update(units="TECU", semantic_class="derived", method="explicit_constant_reference")
        else:
            estimator = SavitzkyGolayBaseline(config.baseline_window_minutes, config.baseline_polyorder, config.analysis_mode == "causal") if config.baseline_method == "savgol" else RollingMedianBaseline(config.baseline_window_minutes, config.analysis_mode == "causal")
            baseline = estimator.estimate(result.tec_smooth)
        result["tec_background"] = baseline
    except AnalysisError as exc:
        result["tec_background"] = xr.full_like(result.tec_smooth, np.nan)
        status = {"status": "insufficient_samples", "method": config.baseline_method, "reason": str(exc)}
    anomaly = smooth - result.tec_background.values
    result["dtec"] = (dims, anomaly, {"units": "TECU", "semantic_class": "derived", "semantic": "tec_smooth minus estimated temporal background; NOT forecast-model residual"})
    background = result.tec_background.values
    relative = np.divide(anomaly, background, out=np.full_like(anomaly, np.nan), where=np.abs(background) > config.robust_scale_floor_tecu)
    result["dtec_relative"] = (dims, relative, {"units": "1", "semantic_class": "derived", "semantic": "dtec/background; undefined near zero background"})
    normalized = np.full_like(anomaly, np.nan)
    scales = np.full(len(anomaly), np.nan)
    centers = np.full(len(anomaly), np.nan)
    if config.analysis_mode == "causal":
        for index in range(len(anomaly)):
            pool = anomaly[:index + 1]
            finite = pool[np.isfinite(pool)]
            if finite.size:
                centers[index] = np.median(finite)
                scales[index] = max(config.robust_scale_floor_tecu, 1.4826 * np.median(np.abs(finite - centers[index])))
                normalized[index] = (anomaly[index] - centers[index]) / scales[index]
    else:
        finite = anomaly[np.isfinite(anomaly)]
        if finite.size:
            centers[:] = np.median(finite)
            scales[:] = max(config.robust_scale_floor_tecu, 1.4826 * np.median(np.abs(finite - centers[0])))
            normalized = (anomaly - centers[0]) / scales[0]
    result["dtec_z"] = (dims, normalized, {"units": "1", "semantic_class": "derived", "semantic": "window-wide median/MAD normalization; causal prefixes in causal mode, not independently normalized frames"})
    result["anomaly_center_tecu"] = ("time", centers, {"units": "TECU", "semantic_class": "derived"})
    result["anomaly_scale_tecu"] = ("time", scales, {"units": "TECU", "semantic_class": "derived"})
    result["anomaly_confidence"] = (dims, np.where(np.isfinite(anomaly), support * fraction, 0), {"units": "1", "semantic_class": "derived", "semantic": "heuristic valid background/anomaly support, not calibrated uncertainty"})
    amplitude = np.maximum(np.abs(anomaly), config.anomaly_scale_tecu)
    reliability = np.divide(amplitude**2, amplitude**2 + uncertainty**2,
                            out=np.full_like(amplitude, np.nan), where=np.isfinite(uncertainty) & np.isfinite(amplitude))
    result["measurement_reliability"] = (dims, reliability, {"units": "1", "semantic_class": "inferred", "semantic": "amplitude^2/(amplitude^2+source_RMS^2), with an explicit anomaly-scale floor; heuristic sensitivity, not propagated error or probability; NaN means RMS unknown"})
    result.tec.attrs.update(units="TECU", semantic_class="synthetic" if synthetic else "measured", semantic="mathematical synthetic TEC fixture" if synthetic else "source gridded TEC product, not an independent sensor at each interpolated cell")
    result.observed_mask.attrs.update(semantic_class="synthetic" if synthetic else "measured", semantic="synthetic fixture availability mask" if synthetic else "support from an original source frame; spatial interpolation is separately labeled")
    result["missing_mask"] = (dims, ~valid, {"semantic_class": "derived"})
    return result, status
