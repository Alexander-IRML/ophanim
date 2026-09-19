"""Seeded local imagination: numerical advection/diffusion of folded 3-D material.

This is a forced passive-scalar model, not ionospheric electrodynamics. The
material satisfies a discretized rho_t + v.grad(rho) = kappa*laplacian(rho)
plus relaxation toward moving, folded analytic sources. Velocity, forcing,
altitude and scales are explicitly hypothesized. It is not mass-conserving
electron transport. No measured state or detection is modified.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from ophanim.core.volumes import ImaginedVolume, VOLUME_SCHEMA, VOLUME_SEMANTICS
from .hypotheses import validate_hypothesis

VOLUME_MODEL_VERSION = "imagined-advection-diffusion/1"


def simulate_hypothesis(recipe, cancelled=None):
    """Return a bounded, deterministic time-varying volume; no remote services."""
    recipe = deepcopy(recipe)
    config = validate_hypothesis(recipe)

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError("Imagined volume simulation cancelled")

    check()
    import numpy as np
    import xarray as xr
    from scipy.ndimage import gaussian_filter, map_coordinates
    from ophanim.dynamics.regularize import utc64

    nt, nz, ny, nx = config.dimensions
    dx, dz = config.spacing_km * 1000, config.vertical_spacing_km * 1000
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dx
    z = config.altitude_min_km * 1000 + np.arange(nz) * dz
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    iz, iy, ix = np.indices((nz, ny, nx), dtype=np.float32)
    theta = np.deg2rad(config.bearing_deg)
    sn, cs = np.sin(theta), np.cos(theta)
    along, cross = sn * xx + cs * yy, cs * xx - sn * yy
    radius = config.extent_km * 500
    zcenter, zspan = (z[-1] + z[0]) / 2, z[-1] - z[0]
    Z, U, V = (zz - zcenter) / zspan, along / radius, cross / radius
    width = config.width_km * 1000
    rng = np.random.default_rng(config.seed)
    phases = rng.uniform(-np.pi, np.pi, 8)
    noise = gaussian_filter(rng.normal(size=(nz, ny, nx)).astype(np.float32), sigma=1.2)
    noise /= max(float(np.std(noise)), 1e-6)
    period = config.period_minutes * 60
    times = np.linspace(0, config.duration_minutes * 60, nt)
    output = np.empty((nt, nz, ny, nx), dtype=np.float32)
    thickness = max(1.1 * dz, zspan * .045)
    horizontal_taper = np.exp(-.6 * (U**6 + V**6))
    vertical_taper = np.sin(np.pi * np.clip((zz - z[0]) / zspan, 0, 1)) ** .55

    def forcing(seconds):
        phase = 2 * np.pi * seconds / period
        if config.kind == "wave_packet":
            # Three folded ribbons; narrow coherent bundles run through each.
            height = zcenter + .18 * zspan * np.sin(2.7 * U - .45 * phase + phases[0])
            height += .12 * zspan * np.sin(4.1 * V + .25 * phase + phases[1])
            surface = sum(np.exp(-.5 * ((zz - height - offset * zspan) / thickness)**2)
                          for offset in (-.22, 0, .22))
            strands = .18 + .82 * (.5 + .5 * np.cos(9 * V + 2.1 * np.sin(3 * U + phases[2]) + 5 * Z - .2 * phase))**5
            envelope = np.exp(-.35 * (along / max(width * 2, radius * .6))**2)
            pulse = .6 + .4 * np.cos(3.2 * U - phase + phases[3])**2
            field = surface * strands * envelope * pulse
        elif config.kind == "front":
            # Corrugated vertical curtains, separated in actual horizontal depth.
            wall = .18 * radius * np.sin(3.5 * V + 4 * Z + .35 * phase + phases[0])
            wall += .10 * radius * np.sin(8 * Z - .3 * phase + phases[1])
            narrow = max(1.2 * dx, width * .16)
            field = sum(np.exp(-.5 * ((along - wall - offset * radius) / narrow)**2)
                        for offset in (-.42, 0, .42))
            field *= .3 + .7 * (.5 + .5 * np.cos(7 * V - 5 * Z + .3 * phase + phases[2]))**3
        else:
            # Several vertically twisting jet cores plus a helical skirt.
            field = np.zeros_like(xx)
            orbit = max(width * .48, dx * 2)
            for strand in range(4):
                angle = strand * np.pi / 2 + 6 * Z + .45 * phase + phases[0]
                cx, cy = orbit * np.cos(angle), orbit * np.sin(angle)
                cx += width * .45 * np.sin(4 * Z + phases[1])
                cy += width * .30 * np.sin(7 * Z + .2 * phase + phases[2])
                core = max(1.15 * dx, width * .13)
                field += np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * core**2))
            field *= .75 + .25 * np.cos(9 * Z - phase + phases[3])**2
        modulation = np.clip(1 + .13 * noise, .5, 1.5)
        return (config.amplitude * field * modulation * horizontal_taper * vertical_taper).astype(np.float32)

    material = forcing(0)
    output[0] = material
    substeps = config.substeps_per_frame
    dt = float(times[1]) / substeps
    # Relaxation supplies sustained fine structures despite coarse numerical
    # diffusion. It is part of the chosen model, not hidden render geometry.
    relax = 1 - np.exp(-dt / max(period * .18, 120))
    diffusion_sigma = (np.sqrt(2 * config.diffusivity_m2_s * dt) / dz,
                       np.sqrt(2 * config.diffusivity_m2_s * dt) / dx,
                       np.sqrt(2 * config.diffusivity_m2_s * dt) / dx)
    for frame in range(1, nt):
        for step in range(substeps):
            check()
            seconds = float(times[frame - 1] + (step + 1) * dt)
            phase = 2 * np.pi * seconds / period
            shear = config.shear_per_s * cross
            swirl = config.speed_m_s * .18
            u = sn * (config.speed_m_s + shear) - swirl * yy / radius * np.cos(3 * Z + phase * .1)
            v = cs * (config.speed_m_s + shear) + swirl * xx / radius * np.cos(3 * Z + phase * .1)
            w = (20 + config.speed_m_s * .1) * np.sin(3 * U + 2 * V + phases[4] - phase * .2) * vertical_taper
            back = np.asarray((iz - w * dt / dz, iy - v * dt / dx, ix - u * dt / dx), dtype=np.float32)
            material = map_coordinates(material, back, order=1, mode="constant", cval=0, prefilter=False)
            if config.diffusivity_m2_s:
                material = gaussian_filter(material, sigma=diffusion_sigma, mode="constant", cval=0)
            material = ((1 - relax) * material + relax * forcing(seconds)).astype(np.float32)
        output[frame] = material
    check()
    data = xr.Dataset({"density": (("time", "z", "y", "x"), output,
                                  {"units": "1", "semantic_class": "synthetic",
                                   "semantic": "dimensionless imagined material, not electron number density"})},
                      coords={"time": utc64(recipe["start_time"]) + np.rint(times * 1e9).astype("timedelta64[ns]"),
                              "x": x, "y": y, "z": z},
                      attrs={"volume_schema": VOLUME_SCHEMA, "source_kind": "imagined", "semantics": VOLUME_SEMANTICS,
                             "hypothesis_kind": config.kind, "seed": config.seed,
                             "center_latitude": config.center_latitude, "center_longitude": config.center_longitude,
                             "model_version": VOLUME_MODEL_VERSION, "numerical_steps": substeps * (nt - 1),
                             "numerical_step_seconds": dt, "boundary_condition": "zero material outside chosen domain",
                             "density_normalization": "fixed generation amplitude; no independent per-frame normalization",
                             "equation": "rho_t + v.grad(rho) = diffusivity*laplacian(rho) + (forcing-rho)/relaxation_time",
                             "equation_scope": "forced nonconservative visual scalar, not plasma continuity or measured dynamics"})
    data.time.attrs["timezone"] = "UTC"
    data.x.attrs.update(units="m", semantic="chosen local eastward distance")
    data.y.attrs.update(units="m", semantic="chosen local northward distance")
    data.z.attrs.update(units="m", semantic="chosen altitude above reference surface")
    provenance = {"producer": VOLUME_MODEL_VERSION, "evidence_sha256": recipe["evidence_sha256"],
                  "candidate_reference": deepcopy(recipe["candidate_reference"]),
                  "parameter_provenance": deepcopy(recipe["provenance"]),
                  "motion_policy": "chosen simulation motion remains allowed when scientific flow is withheld; never feeds detection",
                  "ifm": deepcopy(recipe["ifm"])}
    return ImaginedVolume(data, deepcopy(recipe), provenance).validate()


def publish_hypothesis_run(recipe, output_root, *, cancelled=None):
    """Atomically publish recipe and evolved volume under a content-derived identity."""
    from ophanim.core.artifacts import canonical_json, software_identity, _publish, _write_dataset, write_json, dataset_sha256

    recipe = deepcopy(recipe)
    validate_hypothesis(recipe)
    if cancelled is not None and cancelled():
        raise InterruptedError("Hypothesis publication cancelled")
    identity = {"recipe": deepcopy(recipe), "model_version": VOLUME_MODEL_VERSION,
                "software": software_identity(extra_sections=("experiments",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage):
        volume = simulate_hypothesis(recipe, cancelled=cancelled)
        if cancelled is not None and cancelled():
            raise InterruptedError("Hypothesis publication cancelled before writing")
        _write_dataset(volume.dataset, stage / "volume.zarr")
        write_json(stage / "hypothesis.json", volume.recipe)
        write_json(stage / "volume_contract.json", {"schema_version": VOLUME_SCHEMA, "dataset": "volume.zarr",
                   "dimensions": dict(volume.dataset.sizes), "density_units": "1", "semantics": VOLUME_SEMANTICS,
                   "dataset_sha256": dataset_sha256(volume.dataset), "provenance": volume.provenance})
        if cancelled is not None and cancelled():
            raise InterruptedError("Hypothesis publication cancelled before completion")

    return _publish(Path(output_root), run_id, "hypothesis", identity, build)


__all__ = ["simulate_hypothesis", "publish_hypothesis_run", "VOLUME_MODEL_VERSION"]
