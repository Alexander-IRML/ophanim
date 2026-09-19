"""Apparent TEC-feature flow, with fit, observability and growth kept separate.

Optical flow is a brightness-pattern inference, not a plasma velocity sensor.
Even excellent warp/round-trip agreement cannot prove physical advection.
"""

from __future__ import annotations

import time
import logging

import numpy as np
from scipy.ndimage import binary_erosion, convolve, distance_transform_edt, gaussian_filter, map_coordinates
from skimage.registration import optical_flow_ilk, optical_flow_tvl1

from .coords import flow_pixels_to_enu
from .derivatives import physical_derivative
from .regularize import utc64
from .schemas import FlowConfig


INTERPRETATION_POLICY = ("Conservative presentation eligibility requires an estimated pair-level fit, "
                         "both aligned source RMS proxies, and the existing local confidence gate; "
                         "this heuristic is not verified motion accuracy or propagated uncertainty")


def _fill_for_solver(values, mask):
    """Numerical inpainting only; filled locations never become valid outputs."""
    if mask.all():
        return values
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def _solve(reference, moving, config):
    if config.method == "tvl1":
        return optical_flow_tvl1(reference, moving, attachment=config.attachment, tightness=config.tightness, num_warp=config.num_warp, num_iter=config.num_iter, prefilter=True)
    return optical_flow_ilk(reference, moving, radius=config.ilk_radius, num_warp=config.num_warp, gaussian=True, prefilter=True)


def _photometric_alternative(previous, current, common, config):
    """Competing stationary affine-brightness explanation, measured locally.

    A(x,t+1)=gain(x)*A(x,t)+offset(x) can explain genuine growth without
    translation. An excellent optical warp cannot distinguish these cases.
    The score asks what fraction of observed change this stationary model
    explains, not whether its change exceeds an arbitrary per-frame gain.
    """
    sigma = config.photometric_window_cells
    weights = gaussian_filter(common.astype(float), sigma, mode="constant")
    def average(values):
        numerator = gaussian_filter(np.where(common, values, 0), sigma, mode="constant")
        return np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 1e-12)
    p, c = average(previous), average(current)
    vp = np.maximum(average(previous**2) - p**2, 0)
    vc = np.maximum(average(current**2) - c**2, 0)
    covariance = average(previous * current) - p * c
    gain = np.divide(covariance, vp, out=np.ones_like(vp), where=vp > 1e-14)
    residual = np.maximum(vc - np.divide(covariance**2, vp, out=np.zeros_like(vp), where=vp > 1e-14), 0)
    change_energy = average((current - previous)**2)
    explained = np.clip(1 - np.divide(residual, change_energy, out=np.ones_like(residual), where=change_energy > 1e-14), 0, 1)
    explained[(vp < 1e-14) | (weights < config.minimum_pair_coverage)] = 0
    threshold = config.photometric_explained_threshold
    ambiguity = np.clip((explained - threshold) / (1 - threshold), 0, 1)
    return ambiguity, explained, np.sqrt(residual), gain


def estimate_feature_flow(previous, current, dx_m, dy_m, dt_s, *, config=None, previous_support=None, current_support=None, lambda1=None, lambda2=None, previous_uncertainty=None, current_uncertainty=None):
    """Estimate forward apparent velocity evaluated on the CURRENT frame grid.

    Library flow B=current→previous is a pullback: previous(x+B)≈current(x).
    The returned forward velocity is −B*spacing/dt at current-grid locations.
    This destination alignment permits causal backward temporal derivatives.

    All normalization uses a single pair scale; independent per-frame scaling
    would erase growth. Mask inpainting is solver-only and eroded from output.
    """
    config = config or FlowConfig()
    previous, current = np.asarray(previous, float), np.asarray(current, float)
    if previous.ndim != 2 or previous.shape != current.shape or min(current.shape) < 3:
        raise ValueError("flow requires equally shaped two-dimensional frames of at least 3×3")
    if any(not np.isfinite(value) or value <= 0 for value in (dx_m, dy_m, dt_s)):
        raise ValueError("flow requires positive finite spacing and time interval")
    prev_valid, curr_valid = np.isfinite(previous), np.isfinite(current)
    shape = current.shape
    nan_fields = {name: np.full(shape, np.nan) for name in ("flow_u", "flow_v", "flow_speed", "flow_bearing", "warp_error", "forward_backward_error", "flow_normal_velocity")}
    zero_fields = {name: np.zeros(shape) for name in ("flow_confidence", "flow_interpretation_confidence", "warp_confidence", "forward_backward_confidence", "flow_texture_confidence", "flow_observability", "flow_normal_confidence", "growth_ambiguity", "flow_fit_quality", "flow_noise_reliability", "stationary_photometric_explained", "stationary_photometric_error", "flow_measurement_reliability", "flow_uncertainty_known")}
    fields = {**nan_fields, **zero_fields}
    withheld = {"interpretation_status": "withheld", "interpretation_eligible_fraction": 0.0,
                "interpretation_policy": INTERPRETATION_POLICY}
    if min(prev_valid.mean(), curr_valid.mean()) < config.minimum_pair_coverage:
        return fields, {"status": "invalid_support", "reason": "insufficient finite pair coverage", "uncertainty_status": "not_assessed", **withheld}
    pool = np.concatenate((previous[prev_valid], current[curr_valid]))
    scale = float(np.percentile(pool, 95) - np.percentile(pool, 5))
    if scale < 1e-10:
        return fields, {"status": "low_confidence", "reason": "featureless field", "normalization_scale_tecu": scale, "uncertainty_status": "not_assessed", **withheld}
    center = float(np.median(pool))
    p = np.asarray((_fill_for_solver(previous, prev_valid) - center) / scale, dtype=np.float32)
    c = np.asarray((_fill_for_solver(current, curr_valid) - center) / scale, dtype=np.float32)
    forward = _solve(p, c, config)
    backward = _solve(c, p, config)
    rows, cols = np.indices(shape, dtype=float)
    sampling = np.array([rows + backward[0], cols + backward[1]])
    inside = (sampling[0] >= 0) & (sampling[0] <= shape[0] - 1) & (sampling[1] >= 0) & (sampling[1] <= shape[1] - 1)
    prev_at_current = map_coordinates(prev_valid.astype(float), sampling, order=1, mode="constant", cval=0) >= 1 - 1e-6
    support = curr_valid & prev_at_current & inside
    if config.mask_erosion_cells:
        support = binary_erosion(support, iterations=config.mask_erosion_cells, border_value=0)
    warped = map_coordinates(p, sampling, order=1, mode="constant", cval=np.nan)
    warp_error = np.abs(c - warped)
    fb = np.hypot(backward[0] + map_coordinates(forward[0], sampling, order=1, mode="constant", cval=np.nan), backward[1] + map_coordinates(forward[1], sampling, order=1, mode="constant", cval=np.nan))
    warp_conf = np.exp(-0.5 * (warp_error / config.warp_error_scale)**2)
    fb_conf = np.exp(-0.5 * (fb / config.forward_backward_scale_pixels)**2)
    # Noise must not turn an aperture-limited stripe into "2-D texture". The
    # robust Laplacian estimator uses only original finite stencils. Its noise
    # floor is subtracted from a smoothed gradient tensor, and a separate SNR
    # term demotes random texture even when flow overfits its warp residual.
    common = prev_valid & curr_valid
    noise_support = binary_erosion(common, iterations=1, border_value=0)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    noise_estimates = []
    for image in (p, c):
        highpass = convolve(image, laplacian_kernel, mode="reflect")[noise_support]
        noise_estimates.append(float(np.median(np.abs(highpass - np.median(highpass))) / (0.67448975 * np.sqrt(20))) if highpass.size else float("inf"))
    noise_sigma = max(noise_estimates)
    signal_variance = max(float(np.var(p[common]) + np.var(c[common])) / 2 - noise_sigma**2, 0)
    noise_reliability = float(np.exp(-config.noise_reliability_penalty * noise_sigma / max(np.sqrt(signal_variance), 1e-12)))
    smooth = gaussian_filter(c, config.texture_smoothing_cells)
    gy, gx = np.gradient(smooth)
    jxx, jyy, jxy = (gaussian_filter(v, config.texture_tensor_cells) for v in (gx * gx, gy * gy, gx * gy))
    radius = int(np.ceil(4 * config.texture_smoothing_cells))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / config.texture_smoothing_cells)**2)
    kernel /= kernel.sum()
    derivative_kernel = np.convolve(kernel, [-0.5, 0, 0.5])
    gradient_noise_variance = noise_sigma**2 * float(np.sum(kernel**2) * np.sum(derivative_kernel**2))
    jxx, jyy = jxx - gradient_noise_variance, jyy - gradient_noise_variance
    gap = np.sqrt((jxx - jyy)**2 + 4 * jxy**2)
    lambda1 = np.maximum(0, (jxx + jyy + gap) / 2)
    lambda2 = np.maximum(0, (jxx + jyy - gap) / 2)
    finite_lambda = np.asarray(lambda1)[np.isfinite(lambda1)]
    reference = float(np.percentile(finite_lambda, 75)) if finite_lambda.size else 0.0
    strength = np.clip(np.asarray(lambda1) / max(reference, 1e-30), 0, 1)
    observability = np.clip(10 * np.asarray(lambda2) / np.maximum(np.asarray(lambda1), 1e-30), 0, 1)
    texture = strength * observability
    original_texture_support = binary_erosion(common, iterations=int(np.ceil(3 * (config.texture_smoothing_cells + config.texture_tensor_cells))), border_value=1)
    texture *= original_texture_support
    strength *= original_texture_support
    coverage = np.ones(shape)
    if previous_support is not None:
        coverage *= np.nan_to_num(map_coordinates(previous_support, sampling, order=1, mode="constant", cval=0))
    if current_support is not None:
        coverage = np.minimum(coverage, current_support)
    # A separate global affine brightness comparison detects a key ambiguity:
    # stationary growth can satisfy a radial optical-flow solution exactly.
    pa, ca = previous[common], current[common]
    pa, ca = pa - pa.mean(), ca - ca.mean()
    covariance = float(np.dot(pa, ca))
    gain = covariance / max(float(np.dot(pa, pa)), 1e-30)
    correlation = covariance / max(float(np.linalg.norm(pa) * np.linalg.norm(ca)), 1e-30)
    log_gain = abs(float(np.log(max(gain, 1e-12))))
    growth = float(np.clip((correlation - 0.9) / 0.1, 0, 1) * (1 - np.exp(-0.5 * (log_gain / config.growth_log_gain_scale)**2)))
    local_ambiguity, local_explained, photometric_error, _ = _photometric_alternative(p, c, common, config)
    # An exact/near-exact stationary model also protects boundary patches whose
    # local convolution lacks enough support. It does not set velocity to zero.
    offset = float(np.mean(current[common]) - gain * np.mean(previous[common]))
    stationary_error = float(np.mean((current[common] - gain * previous[common] - offset)**2))
    change_energy = float(np.mean((current[common] - previous[common])**2))
    explained = float(np.clip(1 - stationary_error / change_energy, 0, 1)) if change_energy > scale**2 * 1e-14 else 0.0
    threshold = config.photometric_explained_threshold
    stationary_ambiguity = float(np.clip((explained - threshold) / (1 - threshold), 0, 1))
    ambiguity = np.maximum(local_ambiguity, max(growth, stationary_ambiguity))
    uncertainty_known = np.zeros(shape, bool)
    measurement_reliability = np.ones(shape)
    if previous_uncertainty is not None and current_uncertainty is not None:
        prior_rms, current_rms = np.asarray(previous_uncertainty, float), np.asarray(current_uncertainty, float)
        if prior_rms.shape != shape or current_rms.shape != shape:
            raise ValueError("source uncertainty must match each flow frame")
        prior_rms = map_coordinates(np.where(prior_rms >= 0, prior_rms, np.nan), sampling, order=1, mode="constant", cval=np.nan)
        uncertainty_known = np.isfinite(prior_rms) & np.isfinite(current_rms) & (current_rms >= 0)
        rms = np.maximum(prior_rms, current_rms)
        # Conservative source-error proxy without assuming independent cells or
        # independent epochs. Unknown RMS remains explicitly unknown, not zero.
        signal_scale = max(scale * np.sqrt(max(signal_variance, 0)), scale * 1e-6)
        measurement_reliability[uncertainty_known] = signal_scale**2 / (signal_scale**2 + rms[uncertainty_known]**2)
    fit = np.sqrt(warp_conf * fb_conf)
    full_conf = coverage * np.cbrt(warp_conf * fb_conf * texture) * (1 - ambiguity) * noise_reliability * measurement_reliability
    normal_conf = coverage * np.cbrt(warp_conf * fb_conf * strength) * (1 - ambiguity) * noise_reliability * measurement_reliability
    u, v = flow_pixels_to_enu(-backward[0], -backward[1], dx_m, dy_m, dt_s)
    gy, gx = np.gradient(np.where(curr_valid, current, np.nan), dy_m, dx_m)
    grad = np.hypot(gx, gy)
    normal_velocity = np.divide(u * gx + v * gy, grad, out=np.full(shape, np.nan), where=grad > 1e-12)
    fields.update(flow_u=u, flow_v=v, flow_speed=np.hypot(u, v), flow_bearing=np.degrees(np.arctan2(u, v)) % 360, warp_error=warp_error, forward_backward_error=fb, flow_confidence=full_conf, warp_confidence=warp_conf, forward_backward_confidence=fb_conf, flow_texture_confidence=texture, flow_observability=observability, flow_normal_confidence=normal_conf, flow_normal_velocity=normal_velocity, growth_ambiguity=ambiguity, flow_fit_quality=fit, flow_noise_reliability=np.full(shape, noise_reliability), stationary_photometric_explained=local_explained, stationary_photometric_error=photometric_error, flow_measurement_reliability=measurement_reliability, flow_uncertainty_known=uncertainty_known.astype(float))
    for name in nan_fields:
        fields[name] = np.where(support, fields[name], np.nan)
    for name in zero_fields:
        fields[name] = np.where(support, np.nan_to_num(fields[name]), 0)
    accepted = fields["flow_confidence"][support]
    confidence = float(np.mean(accepted)) if accepted.size else 0.0
    vector_observable = support.any() and float(np.mean(observability[support])) >= 0.1
    known_fraction = float(uncertainty_known[support].mean()) if support.any() else 0.0
    uncertainty_status = "available" if known_fraction >= 1 - 1e-12 else "partial" if known_fraction > 0 else "unavailable"
    uncertainty_limitation = ("Source RMS is available; its heuristic sensitivity does not propagate source covariance or guarantee velocity accuracy"
                              if uncertainty_status == "available" else
                              "Source RMS is missing for some or all supported vectors; confidence describes feature consistency, not verified motion accuracy. Smooth correlated source errors can pass these gates")
    status = "estimated" if confidence >= config.minimum_confidence else "low_confidence"
    eligible = support & uncertainty_known & (fields["flow_confidence"] >= config.minimum_confidence) & (fields["flow_confidence"] > 0) & (status == "estimated")
    fields["flow_interpretation_confidence"] = np.where(eligible, fields["flow_confidence"], 0.0)
    reasons = []
    if status != "estimated":
        reasons.append("pair-level confidence gate not met")
    if uncertainty_status != "available":
        reasons.append("source RMS proxy missing for some or all aligned supported pixels")
    if not eligible.any():
        reasons.append("no vectors satisfy all presentation eligibility requirements")
    interpretation = {"interpretation_status": "eligible" if eligible.any() else "withheld",
                      "interpretation_eligible_fraction": float(eligible.sum() / support.sum()) if support.any() else 0.0,
                      "interpretation_policy": INTERPRETATION_POLICY, "interpretation_reasons": reasons}
    return fields, {"status": status, "mean_confidence": confidence, "growth_ambiguity": float(np.mean(ambiguity[support])) if support.any() else 0.0, "stationary_photometric_explained": explained, "stationary_photometric_error_tecu": float(np.sqrt(stationary_error)), "stationary_affine_gain": gain, "stationary_pattern_correlation": correlation, "normalization_scale_tecu": scale, "normalization_center_tecu": center, "estimated_noise_tecu": noise_sigma * scale, "noise_reliability": noise_reliability, "source_uncertainty_known_fraction": known_fraction, "uncertainty_status": uncertainty_status, "uncertainty_limitation": uncertainty_limitation, "confidence_scope": "heuristic TEC-feature consistency with optional source-RMS sensitivity, not verified motion accuracy", "measurement_reliability": float(measurement_reliability[support & uncertainty_known].mean()) if (support & uncertainty_known).any() else None, "vector_observability": "two_dimensional" if vector_observable else "normal_only_or_ambiguous", "limitations": "Brightness and translation can be competing explanations; confidence and RMS sensitivity are heuristic, not probabilities or propagated error bars", **interpretation}


def flow_kinematics(u, v, x, y):
    """Analytic-coordinate descriptors; NaN stencils remain unavailable."""
    ux, uy = physical_derivative(u, x, -1), physical_derivative(u, y, -2)
    vx, vy = physical_derivative(v, x, -1), physical_derivative(v, y, -2)
    return {"flow_divergence": ux + vy, "flow_vorticity": vx - uy,
            "flow_strain": np.sqrt((ux - vy)**2 + (uy + vx)**2)}


def advection_residual(temporal, gradient_x, gradient_y, u, v):
    """Numerical material derivative; callers must separately gate motion trust."""
    return np.asarray(temporal) + np.asarray(u) * gradient_x + np.asarray(v) * gradient_y


def add_flow(data, config, capabilities, *, frame_times=None):
    """Analyze the short target window, preserving absent-channel NaNs."""
    start = time.monotonic()
    shape = data.dtec.shape
    physical = ("flow_u", "flow_v", "flow_speed", "flow_bearing", "warp_error", "forward_backward_error", "flow_normal_velocity")
    scores = ("flow_confidence", "flow_interpretation_confidence", "warp_confidence", "forward_backward_confidence", "flow_texture_confidence", "flow_observability", "flow_normal_confidence", "growth_ambiguity", "flow_fit_quality", "flow_noise_reliability", "stationary_photometric_explained", "stationary_photometric_error", "flow_measurement_reliability", "flow_uncertainty_known")
    fields = {name: np.full(shape, np.nan) for name in physical}
    fields.update({name: np.zeros(shape) for name in scores})
    records = []
    target = utc64(config.target_time) if config.target_time else data.time.values[-1]
    lower = target - np.timedelta64(round(config.short_window_minutes * 60 * 1e9), "ns")
    dx, dy = np.median(np.diff(data.x.values)), np.median(np.diff(data.y.values))
    requested = None if frame_times is None else {utc64(value) for value in frame_times}
    for index in range(1, data.sizes["time"]):
        timestamp = data.time.values[index]
        if (requested is None and (timestamp < lower or timestamp > target)) or (requested is not None and timestamp not in requested):
            continue
        record = {"time": str(timestamp), "status": "disabled" if not config.flow.enabled else capabilities["flow"]["status"]}
        if config.flow.enabled and capabilities["flow"]["status"] == "supported":
            dt_s = float((timestamp - data.time.values[index - 1]) / np.timedelta64(1, "s"))
            original_pair = data.observed_mask.values[index - 1].any() and data.observed_mask.values[index].any()
            cadence = capabilities.get("native_cadence_seconds") or dt_s
            if not original_pair or dt_s > cadence * config.flow.maximum_gap_factor:
                record.update(status="invalid_support", reason="motion requires two supported source frames, not interpolated observations")
            else:
                uncertainty = data.source_uncertainty.values if "source_uncertainty" in data else None
                pair, summary = estimate_feature_flow(data.dtec.values[index - 1], data.dtec.values[index], dx, dy, dt_s, config=config.flow, previous_support=data.anomaly_confidence.values[index - 1], current_support=data.anomaly_confidence.values[index], lambda1=data.structure_lambda1.values[index], lambda2=data.structure_lambda2.values[index], previous_uncertainty=None if uncertainty is None else uncertainty[index - 1], current_uncertainty=None if uncertainty is None else uncertainty[index])
                record.update(summary)
                for name in fields:
                    fields[name][index] = pair[name]
        records.append(record)
    result = data.copy()
    for name, values in fields.items():
        units = "1" if name in scores or name == "warp_error" else "pixel" if name == "forward_backward_error" else "degrees" if name == "flow_bearing" else "m/s"
        semantic = "apparent TEC-feature velocity; not plasma velocity" if name.startswith("flow_") and name in physical else "heuristic fit/observability descriptor, not a calibrated probability"
        if name == "flow_bearing":
            semantic = "apparent TEC-feature bearing clockwise from north"
        elif name == "flow_interpretation_confidence":
            semantic = INTERPRETATION_POLICY
        elif name == "flow_measurement_reliability":
            semantic = "heuristic source-RMS sensitivity multiplier; neutral one where RMS is unknown, distinguished by flow_uncertainty_known; not a probability or propagated uncertainty"
        elif name == "flow_uncertainty_known":
            semantic = "both aligned source-frame RMS values are known and nonnegative; zero does not mean zero uncertainty"
        elif name == "stationary_photometric_error":
            semantic = "local stationary affine-brightness model RMS error divided by the shared pair normalization scale"
        result[name] = (("time", "y", "x"), values, {"units": units, "semantic_class": "inferred" if name.startswith("flow_") else "derived", "semantic": semantic, "time_alignment": "pair ending at timestamp, current-frame coordinates"})
    u, v = fields["flow_u"], fields["flow_v"]
    interpretable = fields["flow_interpretation_confidence"] > 0
    kinematics = flow_kinematics(np.where(interpretable, u, np.nan), np.where(interpretable, v, np.nan), data.x.values, data.y.values)
    for name, value in kinematics.items():
        result[name] = (("time", "y", "x"), value, {"units": "1/s", "semantic_class": "inferred", "semantic": "apparent TEC-feature-flow descriptor, not a plasma-fluid property"})
    residual = advection_residual(data.dtec_dt.values, data.dtec_dx.values, data.dtec_dy.values, u, v)
    result["advection_residual_raw"] = (("time", "y", "x"), residual, {"units": "TECU/s", "semantic_class": "derived", "semantic": "ungated numerical dA/dt+u*grad(A); may erase real growth when motion is ambiguous; diagnostics only"})
    result["advection_residual"] = (("time", "y", "x"), np.where(interpretable, residual, np.nan), {"units": "TECU/s", "semantic_class": "derived", "semantic": "dA/dt+u*grad(A), available only with flow_interpretation_confidence eligibility; low residual does not prove advection"})
    result["advection_residual_confidence"] = (("time", "y", "x"), np.where(np.isfinite(residual), fields["flow_interpretation_confidence"], 0), {"units": "1", "semantic_class": "inferred", "semantic": "conservative presentation eligibility of the residual; heuristic, not probability or verified motion accuracy"})
    logging.getLogger(__name__).info("Apparent-feature flow runtime %.3fs", time.monotonic() - start)
    return result, records
