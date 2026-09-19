"""Physical-coordinate finite differences and local orientation descriptors."""

from __future__ import annotations

import numpy as np

from .preprocessing import masked_gaussian


def physical_derivative(values, coordinates, axis, *, backward=False):
    """Differentiate in coordinate units, leaving invalid stencils as NaN."""
    values = np.asarray(values, float)
    coords = np.asarray(coordinates, float)
    output = np.full_like(values, np.nan)
    if len(coords) < 2:
        return output
    if backward:
        dest = [slice(None)] * values.ndim
        dest[axis] = slice(1, None)
        delta_shape = [1] * values.ndim
        delta_shape[axis] = len(coords) - 1
        output[tuple(dest)] = np.diff(values, axis=axis) / np.diff(coords).reshape(delta_shape)
    else:
        output = np.gradient(values, coords, axis=axis, edge_order=2 if len(coords) >= 3 else 1)
    output[~np.isfinite(values)] = np.nan
    return output


def add_derivatives(data, config):
    result = data.copy()
    a = data.dtec.values
    seconds = (data.time.values - data.time.values[0]).astype("timedelta64[ns]").astype(float) / 1e9
    gx = physical_derivative(a, data.x.values, 2)
    gy = physical_derivative(a, data.y.values, 1)
    fields = {
        "dtec_dt": (physical_derivative(a, seconds, 0, backward=config.analysis_mode == "causal"), "TECU/s", "temporal derivative; change is not necessarily translation"),
        "dtec_dx": (gx, "TECU/m", "eastward projected anomaly derivative"),
        "dtec_dy": (gy, "TECU/m", "northward projected anomaly derivative"),
        "grad_mag": (np.hypot(gx, gy), "TECU/m", "anomaly spatial gradient magnitude"),
        "laplacian": (physical_derivative(gx, data.x.values, 2) + physical_derivative(gy, data.y.values, 1), "TECU/m^2", "anomaly Laplacian"),
    }
    dx, dy = np.median(np.diff(data.x.values)), np.median(np.diff(data.y.values))
    sigma = (0, config.structure_sigma_km * 1000 / dy, config.structure_sigma_km * 1000 / dx)
    jxx, _ = masked_gaussian(gx * gx, sigma, config.smoothing_minimum_support)
    jyy, _ = masked_gaussian(gy * gy, sigma, config.smoothing_minimum_support)
    jxy, _ = masked_gaussian(gx * gy, sigma, config.smoothing_minimum_support)
    gap = np.sqrt(np.maximum(0, (jxx - jyy) ** 2 + 4 * jxy**2))
    trace = jxx + jyy
    lambda1, lambda2 = (trace + gap) / 2, np.maximum(0, (trace - gap) / 2)
    coherence = np.divide(gap, trace, out=np.zeros_like(gap), where=trace > 1e-30)
    coherence[~np.isfinite(trace)] = np.nan
    orientation = (0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2) % np.pi
    orientation[trace <= 1e-30] = np.nan
    fields.update({
        "structure_lambda1": (lambda1, "TECU^2/m^2", "larger gradient structure-tensor eigenvalue"),
        "structure_lambda2": (lambda2, "TECU^2/m^2", "smaller gradient structure-tensor eigenvalue; full-vector observability support"),
        "structure_coherence": (coherence, "1", "band orientation coherence; NOT full-vector motion confidence"),
        "structure_orientation": (orientation, "radian", "band tangent counterclockwise from east, modulo pi; perpendicular to gradient"),
    })
    for name, (values, units, semantic) in fields.items():
        result[name] = (("time", "y", "x"), values, {"units": units, "semantic": semantic, "semantic_class": "derived"})
    return result
