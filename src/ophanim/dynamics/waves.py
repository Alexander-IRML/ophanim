"""Signed 3-D spectral evidence with original-support admission gates.

The Fourier convention is exp(-2πi[kx*x + ky*y + f*t]). A wave's
phase velocity is -f*k/|k|²; conjugate peaks describe the SAME velocity.
This is periodic image-structure evidence, not proof of a physical mechanism.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fftfreq, fftn
from scipy.signal import detrend, savgol_coeffs
from scipy.optimize import minimize

from .regularize import iso_time, utc64


def _unavailable(status, reason, **extra):
    return {"status": status, "reason": reason, "confidence": 0.0, "direction_confidence": 0.0, "orientation_confidence": 0.0, **extra}


def baseline_transfer(config, cadence, period_s):
    """Linear SavGol detrending response at a candidate period.

    Rolling median is nonlinear: its response is explicitly unknown and must
    be measured with injected signals, not silently described as unity.
    """
    if config.baseline_method == "constant":
        return {"amplitude_gain": 1.0, "phase_shift_rad": 0.0, "method": "constant_reference"}
    if config.baseline_method != "savgol":
        return {"amplitude_gain": None, "phase_shift_rad": None, "method": "nonlinear_response_requires_injection_test"}
    window = int(np.floor(config.baseline_window_minutes * 60 / cadence))
    if window % 2 == 0:
        window -= 1
    if window <= config.baseline_polyorder:
        return {"amplitude_gain": None, "phase_shift_rad": None, "method": "unsupported_window"}
    anchor = window - 1 if config.analysis_mode == "causal" else window // 2
    coefficients = savgol_coeffs(window, config.baseline_polyorder, pos=anchor, use="dot")
    response = 1 - np.sum(coefficients * np.exp(-2j * np.pi * cadence / period_s * (np.arange(window) - anchor)))
    return {"amplitude_gain": float(abs(response)), "phase_shift_rad": float(np.angle(response)), "method": "interior_linear_frequency_response", "limitations": "finite-window edges and missing support are not described by this analytic response"}


def temporal_wave_evidence(signal, phase_grid, times_seconds, period_s, amplitude_fraction):
    """Separate temporal phase coherence from actual carrier occupancy.

    Spatial demodulation supplies a complex amplitude at every original time.
    Counting active intervals (rather than the full timestamp span) prevents a
    brief wave burst from borrowing quiet history to satisfy a cycle gate.
    """
    ny, nx = signal.shape[1:]
    spatial = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    trace = 2 * np.sum(signal * np.exp(-1j * phase_grid) * spatial[None], axis=(1, 2)) / max(float(spatial.sum()), 1e-30)
    amplitude = np.abs(trace)
    reference = float(np.percentile(amplitude, 95))
    active = amplitude >= max(reference * amplitude_fraction, 1e-12)
    dt = np.diff(times_seconds)
    active_seconds = float(np.sum(dt * (active[:-1].astype(float) + active[1:].astype(float)) / 2))
    duration = float(times_seconds[-1] - times_seconds[0])
    phase_coherence = float(abs(np.sum(trace[active])) / max(float(np.sum(amplitude[active])), 1e-30)) if active.any() else 0.0
    return {"temporal_occupancy": active_seconds / duration if duration else 0.0,
            "effective_cycles": active_seconds / period_s, "active_duration_seconds": active_seconds,
            "temporal_phase_coherence": phase_coherence,
            "amplitude_threshold_tecu": reference * amplitude_fraction,
            "envelope_amplitude_tecu": amplitude.tolist(), "active_epoch_mask": active.tolist()}


def estimate_wave(data, config, capabilities):
    if not config.wave.enabled:
        return _unavailable("disabled", "wave analysis disabled")
    if capabilities["wave"]["status"] != "supported":
        return _unavailable(capabilities["wave"]["status"], "; ".join(capabilities["wave"]["reasons"]))
    target = utc64(config.target_time) if config.target_time else data.time.values[-1]
    lower = target - np.timedelta64(round(config.wave_window_hours * 3600 * 1e9), "ns")
    selected = data.sel(time=slice(lower, target))
    a = np.asarray(selected.dtec.values, float)
    if min(a.shape) < 3:
        return _unavailable("insufficient_samples", "wave cube has fewer than three points on an axis")
    valid = np.isfinite(a)
    coverage = float(valid.mean())
    observed = np.asarray(selected.observed_mask.values, bool)
    if coverage < config.wave.minimum_coverage or float(observed.mean()) < config.wave.minimum_coverage:
        return _unavailable("invalid_support", "wave cube lacks the configured original-frame/finite-data coverage", coverage=coverage)
    # A structured missing frame produces spectral artifacts even when overall
    # cell coverage appears high. Never replace it with fabricated observations.
    if np.any(np.mean(valid, axis=(1, 2)) < config.wave.minimum_coverage):
        return _unavailable("invalid_support", "at least one wave frame lacks sufficient support", coverage=coverage)
    nt, ny, nx = a.shape
    cadence = float(np.median(np.diff(selected.time.values).astype("timedelta64[ns]").astype(float) / 1e9))
    actual_steps = np.diff(selected.time.values).astype("timedelta64[ns]").astype(float) / 1e9
    if not np.allclose(actual_steps, cadence, rtol=1e-8, atol=1e-6):
        return _unavailable("invalid_support", "FFT analysis requires regularly spaced analyzed timestamps")
    supplied_periods = capabilities["admissible_period_seconds"]
    duration = float((selected.time.values[-1] - selected.time.values[0]) / np.timedelta64(1, "s"))
    periods = [max(supplied_periods[0], cadence * config.wave.minimum_samples_per_period),
               min(supplied_periods[1], duration / config.wave.minimum_cycles)]
    if periods[0] > periods[1]:
        return _unavailable("insufficient_samples", "the actual analyzed time grid cannot meet the period samples/cycles gate")
    dx, dy = np.median(np.diff(selected.x.values)), np.median(np.diff(selected.y.values))
    filled = np.where(valid, a, np.nanmean(a, axis=0, keepdims=True))
    if not np.isfinite(filled).all():
        return _unavailable("invalid_support", "permanently missing locations in wave cube")
    signal = detrend(filled, axis=0, type="linear")
    signal -= signal.mean()
    window = np.hanning(nt)[:, None, None] * np.hanning(ny)[None, :, None] * np.hanning(nx)[None, None, :]
    coefficients = fftn(signal * window)
    power = np.abs(coefficients)**2
    frequency, ky, kx = fftfreq(nt, cadence), fftfreq(ny, dy), fftfreq(nx, dx)
    ff, yy, xx = np.meshgrid(frequency, ky, kx, indexing="ij")
    k = np.hypot(xx, yy)
    wavelength = np.divide(1, k, out=np.full_like(k, np.inf), where=k > 0)
    dir_x = np.divide(xx, k, out=np.zeros_like(xx), where=k > 0)
    dir_y = np.divide(yy, k, out=np.zeros_like(yy), where=k > 0)
    native_dx, native_dy = capabilities["native_dx_m"], capabilities["native_dy_m"]
    direction_scale = np.hypot(dir_x * native_dx, dir_y * native_dy)
    minimum_wavelength = config.wave.minimum_native_cells * direction_scale
    effective = capabilities.get("effective_resolution_m")
    if effective is not None:
        minimum_wavelength = np.maximum(minimum_wavelength, 2 * float(effective))
    extent_x = float(selected.x.values[-1] - selected.x.values[0])
    extent_y = float(selected.y.values[-1] - selected.y.values[0])
    spatial_cycles = np.abs(xx) * extent_x + np.abs(yy) * extent_y
    # A bin can straddle the spatial-support boundary. Admit one neighboring
    # bin for local frequency refinement, then enforce the exact support gate
    # on the resulting continuous estimate below.
    # FFT bin centers use N*dt, whereas available history spans (N-1)*dt.
    # Seed bins whose refinement intervals intersect the valid physical band;
    # otherwise a valid three-cycle wave can be excluded before it is fitted.
    refinement_radius_bins = 0.65
    frequency_bounds = (-1 / periods[0], -1 / periods[1])
    frequency_radius = refinement_radius_bins / (nt * cadence)
    temporal_seed = (ff + frequency_radius >= frequency_bounds[0]) & (ff - frequency_radius <= frequency_bounds[1])
    admissible = (ff < 0) & (k > 0) & temporal_seed & (wavelength >= config.wave.min_wavelength_km * 1000) & (wavelength <= config.wave.max_wavelength_km * 1000) & (wavelength >= minimum_wavelength) & (spatial_cycles >= max(0, config.wave.minimum_spatial_cycles - 1))
    order_x, order_y = np.argsort(kx), np.argsort(ky)
    if len(order_x) > 128:
        order_x = order_x[np.linspace(0, len(order_x) - 1, 128, dtype=int)]
    if len(order_y) > 128:
        order_y = order_y[np.linspace(0, len(order_y) - 1, 128, dtype=int)]
    spatial_power = power[frequency < 0].sum(axis=0)[np.ix_(order_y, order_x)]
    summary = {"frequency_hz": frequency.tolist(), "power": power.sum(axis=(1, 2)).tolist(),
               "spatial_power": spatial_power.tolist(), "kx_rad_per_m": (2 * np.pi * kx[order_x]).tolist(),
               "ky_rad_per_m": (2 * np.pi * ky[order_y]).tolist(),
               "spatial_power_semantics": "sum over negative temporal frequencies; sorted spatial axes, at most 128 samples per axis; includes excluded-band competitors",
               "selected_peak": None}
    if not admissible.any():
        return _unavailable("unsupported_resolution", "no spectral bins meet source-direction spacing, spatial extent and actual-time gates", spectrum_summary=summary)
    candidates = np.where(admissible, power, 0)
    peak = np.unravel_index(np.argmax(candidates), candidates.shape)
    peak_power = float(candidates[peak])
    if peak_power < 1e-20:
        return _unavailable("low_confidence", "no nonzero periodic signal", spectrum_summary=summary)
    neighborhood = np.zeros(a.shape, bool)
    for it in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for ix in (-1, 0, 1):
                neighborhood[(peak[0] + it) % nt, (peak[1] + iy) % ny, (peak[2] + ix) % nx] = True
    # Concentration is relative to all non-DC negative-frequency energy, not
    # only the chosen search band (which could hide stronger excluded peaks).
    eligible_power = (ff < 0) & (k > 0)
    concentration = float(np.sum(power[neighborhood & eligible_power]) / max(np.sum(power[eligible_power]), 1e-30))
    competitor = float(np.max(np.where(admissible & ~neighborhood, power, 0)))
    uniqueness = float(np.clip(1 - competitor / peak_power, 0, 1))
    pkx, pky, pf = float(xx[peak]), float(yy[peak]), float(ff[peak])
    # Fit amplitude and phase in ORIGINAL reference coordinates. Unlike a
    # magnitude-only spectrum this retains reproducible signed phase metadata.
    tx = selected.x.values - selected.x.values[0]
    ty = selected.y.values - selected.y.values[0]
    tt = (selected.time.values - selected.time.values[0]).astype("timedelta64[ns]").astype(float) / 1e9
    # Deterministic bounded local sinusoid fitting avoids claiming that an FFT
    # bin center is an exact wavelength. This refines an existing peak; it does
    # not improve independent spatial/temporal sampling or resolve two peaks.
    grid_t, grid_y, grid_x = np.meshgrid(tt / (nt * cadence), ty / (ny * dy), tx / (nx * dx), indexing="ij")
    selection = np.flatnonzero((window > 1e-3).ravel() & valid.ravel())
    if len(selection) > 12_000:
        selection = selection[np.linspace(0, len(selection) - 1, 12_000, dtype=int)]
    sample_coordinates = np.column_stack((grid_t.ravel()[selection], grid_y.ravel()[selection], grid_x.ravel()[selection]))
    sample_values = filled.ravel()[selection]
    sample_weights = np.sqrt(window.ravel()[selection])
    def objective(bins):
        angles = 2 * np.pi * (sample_coordinates @ bins)
        design = np.column_stack((np.cos(angles), np.sin(angles), np.ones(len(angles)), sample_coordinates[:, 0]))
        weighted_design = design * sample_weights[:, None]
        beta = np.linalg.lstsq(weighted_design, sample_values * sample_weights, rcond=None)[0]
        error = (design @ beta - sample_values) * sample_weights
        return float(np.dot(error, error) / max(float(np.sum(sample_weights**2)), 1e-30))
    bin_peak = np.array([pf * nt * cadence, pky * ny * dy, pkx * nx * dx])
    bounds = [(float(v - refinement_radius_bins), float(v + refinement_radius_bins)) for v in bin_peak]
    # Do not force a genuinely unsupported signal onto an admission boundary.
    # Fit the neighboring peak, then reject its continuous estimate below if
    # the actual/native sampling cannot support it (e.g. 84 min in a 4 h cube).
    initial = np.clip(bin_peak, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    refined = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60, "ftol": 1e-12})
    successful_fit = bool(refined.success and np.isfinite(refined.fun) and np.isfinite(refined.x).all())
    boundary = np.isclose(refined.x, np.array(bounds)[:, 0], rtol=0, atol=1e-5) | np.isclose(refined.x, np.array(bounds)[:, 1], rtol=0, atol=1e-5)
    spectral_fit = {"status": "converged" if successful_fit else "failed", "method": "bounded_single_sinusoid_L-BFGS-B",
                    "message": str(refined.message), "seed_bins": bin_peak.tolist(), "initial_bins": initial.tolist(),
                    "search_bounds_bins": [list(pair) for pair in bounds],
                    "fitted_bins": [float(value) if np.isfinite(value) else None for value in refined.x],
                    "objective_tecu_squared": float(refined.fun) if np.isfinite(refined.fun) else None,
                    "at_search_boundary": {name: bool(flag) for name, flag in zip(("frequency", "ky", "kx"), boundary)},
                    "admissible_period_seconds": periods,
                    "admission_applied_after_fit": True,
                    "semantics": "fit convergence and boundaries are diagnostics, not uncertainty bounds or added sampling resolution"}
    summary["spectral_fit"] = spectral_fit
    if successful_fit:
        pf, pky, pkx = (float(refined.x[0] / (nt * cadence)), float(refined.x[1] / (ny * dy)), float(refined.x[2] / (nx * dx)))
    wave_number = float(np.hypot(pkx, pky))
    if wave_number <= 0 or pf >= 0:
        return _unavailable("low_confidence", "local spectral fit did not retain a propagating candidate", spectrum_summary=summary, spectral_fit=spectral_fit)
    fitted_wavelength, fitted_period = 1 / wave_number, 1 / abs(pf)
    summary["selected_peak"] = {"kx_rad_per_m": 2 * np.pi * pkx, "ky_rad_per_m": 2 * np.pi * pky,
                                "frequency_hz": pf, "wavelength_km": fitted_wavelength / 1000,
                                "period_min": fitted_period / 60}
    fitted_native_scale = np.hypot(pkx / wave_number * native_dx, pky / wave_number * native_dy)
    fitted_minimum = max(config.wave.min_wavelength_km * 1000, config.wave.minimum_native_cells * fitted_native_scale, 2 * float(effective or 0))
    if fitted_wavelength < fitted_minimum * (1 - 1e-6) or fitted_wavelength > config.wave.max_wavelength_km * 1000 or abs(pkx) * extent_x + abs(pky) * extent_y < config.wave.minimum_spatial_cycles * (1 - 1e-6):
        return _unavailable("unsupported_resolution", "refined wave does not meet original directional spatial support/extent", spectrum_summary=summary, spectral_fit=spectral_fit)
    if fitted_period < periods[0] * (1 - 1e-6) or fitted_period > periods[1] * (1 + 1e-6):
        return _unavailable("insufficient_samples", "refined wave does not meet actual observation cadence/cycle admission", spectrum_summary=summary, spectral_fit=spectral_fit)
    phase_grid = 2 * np.pi * (pf * tt[:, None, None] + pky * ty[None, :, None] + pkx * tx[None, None, :])
    cosine, sine = np.cos(phase_grid), np.sin(phase_grid)
    weight = window * valid
    normal = np.array([[np.sum(weight * cosine * cosine), np.sum(weight * cosine * sine)], [np.sum(weight * cosine * sine), np.sum(weight * sine * sine)]])
    rhs = np.array([np.sum(weight * signal * cosine), np.sum(weight * signal * sine)])
    fit = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    amplitude = float(np.hypot(*fit))
    phase = float(np.arctan2(-fit[1], fit[0]))
    period_s = 1 / abs(pf)
    temporal_evidence = temporal_wave_evidence(signal, phase_grid, tt, period_s,
                                               config.wave.occupancy_amplitude_fraction)
    persistence = temporal_evidence["temporal_occupancy"] * temporal_evidence["temporal_phase_coherence"]
    signal_power = float(np.mean(signal**2))
    snr_score = float(np.clip(amplitude**2 / max(2 * signal_power, 1e-30), 0, 1))
    support_score = float(selected.observation_support.mean())
    orientation_consistency = None
    if "structure_orientation" in selected and "structure_coherence" in selected:
        orientation = np.asarray(selected.structure_orientation.values)
        coherence = np.asarray(selected.structure_coherence.values)
        expected_tangent = (np.arctan2(pky, pkx) + np.pi / 2) % np.pi
        structure_weights = weight * np.clip(coherence, 0, 1)
        if "grad_mag" in selected:
            structure_weights *= np.nan_to_num(selected.grad_mag.values)
        supported_orientation = np.isfinite(orientation) & np.isfinite(structure_weights) & (structure_weights > 0)
        if supported_orientation.any():
            alignment = (1 + np.cos(2 * (orientation - expected_tangent))) / 2
            orientation_consistency = float(np.average(alignment[supported_orientation], weights=structure_weights[supported_orientation]))
    measurement_reliability = None
    uncertainty_coverage = 0.0
    if "source_uncertainty" in selected:
        uncertainty = np.asarray(selected.source_uncertainty.values)
        known = np.isfinite(uncertainty) & (uncertainty >= 0) & (weight > 0)
        uncertainty_coverage = float(known.sum() / max(1, (weight > 0).sum()))
        if known.any():
            rms_squared = float(np.average(uncertainty[known]**2, weights=weight[known]))
            measurement_reliability = float(amplitude**2 / max(amplitude**2 + rms_squared, 1e-30))
    structure_factor = orientation_consistency if orientation_consistency is not None else 1.0
    uncertainty_factor = measurement_reliability if measurement_reliability is not None else 1.0
    orientation_confidence = support_score * concentration * persistence * structure_factor * uncertainty_factor
    direction_confidence = orientation_confidence * uniqueness
    confidence = direction_confidence * np.sqrt(snr_score)
    velocity_x, velocity_y = -pf * pkx / wave_number**2, -pf * pky / wave_number**2
    response = baseline_transfer(config, cadence, period_s)
    status = "estimated"
    reasons = []
    if not successful_fit:
        status = "low_confidence"
        reasons.append("bounded spectral refinement did not converge; retained bin candidate is not an accepted fit")
    if amplitude < config.wave.minimum_amplitude_tecu or concentration < config.wave.minimum_concentration or confidence < 0.1:
        status = "low_confidence"
        reasons.append("weak amplitude, concentration, or combined evidence")
    if response["amplitude_gain"] is not None and response["amplitude_gain"] < 0.2:
        status = "low_confidence"
        confidence *= response["amplitude_gain"] / 0.2
        reasons.append("baseline suppresses the candidate period")
    if (temporal_evidence["temporal_occupancy"] < config.wave.minimum_temporal_occupancy
            or temporal_evidence["effective_cycles"] < config.wave.minimum_cycles * (1 - 1e-6)):
        status = "insufficient_persistence"
        reasons.append("active carrier duration, not only timestamp span, must satisfy occupancy and cycle gates")
    mean_u = np.asarray(selected.flow_u.values)
    mean_v = np.asarray(selected.flow_v.values)
    flow_weights = np.asarray(selected.flow_interpretation_confidence.values) if "flow_interpretation_confidence" in selected else np.zeros_like(mean_u)
    flow_ok = np.isfinite(mean_u) & np.isfinite(mean_v) & (flow_weights >= config.flow.minimum_confidence) & (flow_weights > 0)
    flow_agreement = None
    if flow_ok.any():
        u = float(np.average(mean_u[flow_ok], weights=flow_weights[flow_ok]))
        v = float(np.average(mean_v[flow_ok], weights=flow_weights[flow_ok]))
        if np.hypot(u, v) > 1e-8:
            flow_agreement = float((u * velocity_x + v * velocity_y) / (np.hypot(u, v) * np.hypot(velocity_x, velocity_y)))
    direction_consistency = (float(np.clip((1 + flow_agreement) / 2, 0, 1)) if flow_agreement is not None else None)
    if direction_consistency is not None:
        direction_confidence *= direction_consistency
        confidence *= direction_consistency
        if flow_agreement < 0:
            if status == "estimated":
                status = "conflicting_evidence"
            reasons.append("supported apparent motion opposes signed spectral propagation; direction is not flipped")
    if confidence < 0.1 and status == "estimated":
        status = "low_confidence"
    candidate_evidence_score = float(np.clip(confidence, 0, 1))
    if status in {"insufficient_persistence", "conflicting_evidence"}:
        confidence = 0.0
    return {
        "status": status, "reasons": reasons, "semantic": "inferred propagating periodic TEC-image structure, not physical mechanism identification",
        "k_x_cycles_per_m": pkx, "k_y_cycles_per_m": pky, "frequency_hz": pf,
        "phase_rad": phase, "phase_convention": "cos(2*pi*(kx*(x-xref)+ky*(y-yref)+f*(t-tref))+phase)",
        "reference_x_m": float(selected.x.values[0]), "reference_y_m": float(selected.y.values[0]), "reference_time": iso_time(selected.time.values[0]),
        "wavelength_km": 1 / wave_number / 1000, "period_min": period_s / 60,
        "phase_speed_m_s": abs(pf) / wave_number, "bearing_deg": float(np.degrees(np.arctan2(velocity_x, velocity_y)) % 360),
        "amplitude_tecu": amplitude, "amplitude": amplitude,
        "spectral_concentration": concentration, "confidence": float(np.clip(confidence, 0, 1)),
        "candidate_evidence_score": candidate_evidence_score,
        "orientation_confidence": orientation_confidence, "direction_confidence": direction_confidence,
        "coverage": coverage, "temporal_persistence": persistence, "peak_uniqueness": uniqueness,
        "temporal_evidence": temporal_evidence, "temporal_support": temporal_evidence, "structural_orientation_consistency": orientation_consistency,
        "measurement_reliability": measurement_reliability, "source_uncertainty_known_fraction": uncertainty_coverage,
        "directional_consistency": direction_consistency,
        "flow_direction_agreement": flow_agreement, "flow_agreement_role": "independent consistency gate; disagreement lowers direction confidence and never flips the signed FFT direction",
        "baseline_transfer": response, "spectrum_summary": summary, "spectral_fit": spectral_fit,
        "spectral_resolution": {"frequency_hz": 1 / (nt * cadence), "k_x_cycles_per_m": float(1 / (nx * dx)), "k_y_cycles_per_m": float(1 / (ny * dy))},
        "confidence_components": {"observation_support": support_score, "spectral_concentration": concentration,
                                  "temporal_occupancy": temporal_evidence["temporal_occupancy"],
                                  "temporal_phase_coherence": temporal_evidence["temporal_phase_coherence"],
                                  "structural_orientation_consistency": orientation_consistency,
                                  "directional_consistency": direction_consistency, "peak_uniqueness": uniqueness,
                                  "sinusoid_energy_fraction": snr_score, "measurement_reliability": measurement_reliability},
        "limitations": ["confidence and RMS sensitivity are heuristic, not calibrated probabilities or propagated uncertainty", "unknown RMS or unavailable cross-channel evidence is explicit, not proof of accuracy", "locally refined single dominant spectral peak; crossing/standing waves reduce uniqueness", "amplitude belongs to detrended/smoothed field, not an automatically corrected source amplitude"],
    }
