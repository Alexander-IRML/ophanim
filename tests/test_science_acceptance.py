"""Independent numerical acceptance: admission, filtering and heuristic trust.

The confidence checks are stratified sensitivity regressions, not claims of
probability calibration or validation on physical ionospheric events.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.util import find_spec
import unittest

AVAILABLE = all(find_spec(name) for name in ("numpy", "scipy", "xarray", "pyproj", "skimage"))


def confidence_sweep():
    """Known translations across noise, speed, missingness and observability."""
    import numpy as np
    from scipy.ndimage import gaussian_filter, shift
    from scipy.stats import spearmanr
    from ophanim.dynamics.flow import estimate_feature_flow
    from ophanim.dynamics.schemas import FlowConfig

    flow_config = FlowConfig()
    rng = np.random.default_rng(42)
    yy, xx = np.indices((48, 48), dtype=float)
    textured = gaussian_filter(rng.normal(size=xx.shape), 2)
    textured = (textured - textured.mean()) / textured.std()
    textures = {
        "textured": textured,
        "gaussian": np.exp(-((xx - 24)**2 + (yy - 24)**2) / (2 * 7**2)),
        "stripe": np.cos(xx * 0.3), "flat": np.ones_like(xx),
    }
    cases = [("textured", speed, noise, 0.0) for speed in (0.25, 1.0, 3.0, 7.0) for noise in (0.0, 0.1, 0.5, 1.0)]
    cases += [("gaussian", 1.0, noise, 0.0) for noise in (0.0, 0.05, 0.1, 0.2, 0.5)]
    cases += [(name, 1.0, noise, 0.0) for name in ("stripe", "flat") for noise in (0.0, 0.1, 0.5)]
    cases += [("textured", 1.0, noise, missing) for missing in (0.05, 0.15) for noise in (0.0, 0.5)]
    interior = np.zeros(xx.shape, dtype=bool)
    interior[8:-8, 8:-8] = True
    rows = []
    for texture, speed, noise, missing in cases:
        rng = np.random.default_rng(2)
        original = textures[texture]
        previous = original + rng.normal(0, noise, xx.shape)
        current = shift(original, (speed / 2, speed), order=3, mode="nearest") + rng.normal(0, noise, xx.shape)
        previous[rng.random(xx.shape) < missing] = np.nan
        current[rng.random(xx.shape) < missing] = np.nan
        fields, summary = estimate_feature_flow(previous, current, 10_000, 10_000, 300, config=flow_config)
        eligible = interior & np.isfinite(fields["flow_u"]) & np.isfinite(fields["flow_v"])
        error = np.hypot(fields["flow_u"] - speed * 10_000 / 300, fields["flow_v"] - speed * 5_000 / 300)
        truth_speed = float(np.hypot(speed * 10_000 / 300, speed * 5_000 / 300))
        endpoint_error = float(error[eligible].mean()) if eligible.any() else None
        admitted_pixels = eligible & (fields["flow_confidence"] >= flow_config.minimum_confidence)
        inaccurate_pixels = admitted_pixels & (error > .5 * truth_speed)
        rows.append({
            "texture": texture, "shift_x_pixels": speed, "noise_std": noise,
            "missing_fraction": missing,
            "ground_truth_speed_m_s": truth_speed,
            "global_mean_confidence": float(summary.get("mean_confidence", 0.0)),
            "interior_mean_confidence": float(fields["flow_confidence"][interior].mean()),
            "supported_mean_confidence": float(fields["flow_confidence"][eligible].mean()) if eligible.any() else None,
            "valid_interior_fraction": float(eligible.sum() / interior.sum()),
            "mean_endpoint_error_m_s": endpoint_error,
            "relative_endpoint_error": endpoint_error / truth_speed if endpoint_error is not None else None,
            "locally_admitted_pixel_count": int(admitted_pixels.sum()),
            "locally_admitted_inaccurate_pixel_count": int(inaccurate_pixels.sum()),
            "locally_admitted_inaccurate_fraction": float(inaccurate_pixels.sum() / admitted_pixels.sum()) if admitted_pixels.any() else None,
            "status": summary["status"],
        })
    usable = [row for row in rows if row["texture"] in ("textured", "gaussian") and row["mean_endpoint_error_m_s"] is not None and row["valid_interior_fraction"] > 0.5]
    ranked = sorted(usable, key=lambda row: row["interior_mean_confidence"])
    size = max(1, len(ranked) // 3)
    low, high = ranked[:size], ranked[-size:]
    rho = spearmanr([row["interior_mean_confidence"] for row in usable], [row["mean_endpoint_error_m_s"] for row in usable]).statistic
    estimated = [row for row in rows if row["status"] == "estimated"]
    # Report accepted-but-inaccurate cases explicitly. A pooled rank statistic
    # is not a statement that every output over the heuristic gate is accurate.
    inaccurate = [row for row in estimated if row["relative_endpoint_error"] is not None and row["relative_endpoint_error"] > 0.5]
    by_texture = {}
    for texture in textures:
        selected = [row for row in rows if row["texture"] == texture]
        identifiable = [row for row in usable if row["texture"] == texture]
        correlation = float(spearmanr([row["interior_mean_confidence"] for row in identifiable], [row["mean_endpoint_error_m_s"] for row in identifiable]).statistic) if len(identifiable) > 1 else None
        by_texture[texture] = {
            "case_count": len(selected),
            "estimated_case_count": sum(row["status"] == "estimated" for row in selected),
            "estimated_cases_above_half_true_speed_error_count": sum(row["texture"] == texture for row in inaccurate),
            "identifiable_confidence_error_spearman": correlation,
            "unit_shift_complete_support_noise_series": [row for row in selected if row["shift_x_pixels"] == 1.0 and row["missing_fraction"] == 0.0],
        }
    return {
        "schema": "ophanim-confidence-sensitivity/1",
        "semantics": "Synthetic sensitivity/error ranking; confidence remains an uncalibrated heuristic, not a probability.",
        "case_count": len(rows), "cases": rows,
        "identifiable_case_spearman_confidence_vs_error": float(rho),
        "low_confidence_tertile_mean_error_m_s": float(np.mean([row["mean_endpoint_error_m_s"] for row in low])),
        "high_confidence_tertile_mean_error_m_s": float(np.mean([row["mean_endpoint_error_m_s"] for row in high])),
        "estimated_case_count": len(estimated),
        "estimated_cases_above_half_true_speed_error_count": len(inaccurate),
        "estimated_cases_above_half_true_speed_error": inaccurate,
        "by_texture": by_texture,
        "experiment": {"texture_seed": 42, "noise_mask_seed": 2, "shape": [48, 48], "interior_margin_cells": 8, "dx_m": 10_000, "dy_m": 10_000, "dt_s": 300, "noise_model": "independent additive Gaussian noise per frame", "flow_config": asdict(flow_config)},
        "limitations": ["Known mathematical translations, not real TEC ground truth", "Aperture-ambiguous stripes and noise-only fields must abstain", "Pooled correlation alone can hide noise-induced false confidence", "Large inter-frame displacements can exceed the solver capture range despite estimated status", "Independent Gaussian noise tests do not validate spatially or temporally correlated source errors", "The half-true-speed error diagnostic is a reporting threshold, not a calibrated acceptance guarantee"],
    }


def wave_sensitivity_sweep():
    """Report recovery and evidence loss for the brief's wave perturbations.

    Constant-amplitude, infinite-plane-wave assumptions deliberately fail in
    some fixtures. Their fitted amplitude is reported, not asserted to equal
    the generating carrier amplitude inside a finite envelope.
    """
    import numpy as np
    from ophanim.dynamics import AnalysisConfig, FlowConfig, analyze_dataset
    from ophanim.experiments import make_synthetic_dataset

    config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                            smooth_sigma_km=0, flow=FlowConfig(enabled=False))
    rows = []
    cases = ("clean", "weaker_second_wave", "changing_amplitude",
             "finite_envelope_broad", "finite_envelope_narrow",
             "noise_1", "noise_5", "noise_20")
    for case in cases:
        noise = float(case.removeprefix("noise_")) if case.startswith("noise_") else 0.0
        raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49, noise_std_tecu=noise)
        perturbation = {"noise_std_tecu": noise}
        if case == "weaker_second_wave":
            other = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49, bearing_deg=0)
            raw["tec"] = raw.tec + 0.35 * (other.tec - 20)
            perturbation.update(second_wave_amplitude_ratio=0.35, second_wave_bearing_deg=0)
        elif case == "changing_amplitude":
            raw.tec.values[:] = 20 + (raw.tec.values - 20) * np.linspace(0.25, 1.75, 49)[:, None, None]
            perturbation.update(amplitude_gain_start=0.25, amplitude_gain_end=1.75)
        elif case.startswith("finite_envelope"):
            sigma_m = 150_000 if case.endswith("narrow") else 450_000
            raw = make_synthetic_dataset("wave_packet", nx=48, ny=48, nt=49,
                                         sigma_m=sigma_m, velocity_u_m_s=0)
            perturbation.update(stationary_envelope_sigma_m=sigma_m)
        wave = analyze_dataset(raw, config).event["wave"]
        keys = ("confidence", "wavelength_km", "period_min", "phase_speed_m_s",
                "bearing_deg", "amplitude_tecu", "peak_uniqueness",
                "temporal_persistence", "spectral_concentration")
        row = {"case": case, "status": wave["status"], "perturbation": perturbation,
               **{key: wave.get(key) for key in keys}}
        row["wavelength_error_km"] = abs(wave["wavelength_km"] - 500) if "wavelength_km" in wave else None
        row["period_error_min"] = abs(wave["period_min"] - 60) if "period_min" in wave else None
        row["bearing_error_deg"] = abs((wave["bearing_deg"] - 90 + 180) % 360 - 180) if "bearing_deg" in wave else None
        rows.append(row)
    return {
        "semantics": "Synthetic wave sensitivity, not a calibrated event detector or real-TEC validation.",
        "case_count": len(rows), "cases": rows, "analysis_config": config.to_dict(),
        "carrier_truth": {"wavelength_km": 500, "period_min": 60,
                          "phase_speed_m_s": 500_000 / 3600, "bearing_deg": 90,
                          "amplitude_tecu": 3, "seed": 42},
        "limitations": ["Fitted finite-packet amplitude is not its carrier peak amplitude",
                        "Changing amplitude and overlapping waves violate a single stationary-wave model",
                        "Reduced confidence is evidence sensitivity, not an error guarantee"],
    }


@unittest.skipUnless(AVAILABLE, "optional scientific dependencies unavailable")
class IndependentScienceAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = confidence_sweep()
        cls.wave_report = wave_sensitivity_sweep()

    def _case(self, texture="textured", speed=1.0, noise=0.0, missing=0.0):
        return next(row for row in self.report["cases"] if (row["texture"], row["shift_x_pixels"], row["noise_std"], row["missing_fraction"]) == (texture, speed, noise, missing))

    def test_confidence_ranks_known_translation_error(self):
        report = self.report
        self.assertLess(report["identifiable_case_spearman_confidence_vs_error"], -0.5)
        self.assertLess(report["high_confidence_tertile_mean_error_m_s"], report["low_confidence_tertile_mean_error_m_s"])
        self.assertLess(self._case()["mean_endpoint_error_m_s"], 4)

    def test_noise_reduces_confidence_when_error_increases(self):
        for texture, noisy_level in (("textured", 1.0), ("gaussian", 0.5)):
            with self.subTest(texture=texture):
                clean = self._case(texture=texture)
                noisy = self._case(texture=texture, noise=noisy_level)
                self.assertGreater(noisy["mean_endpoint_error_m_s"], 5 * clean["mean_endpoint_error_m_s"])
                self.assertLess(noisy["interior_mean_confidence"], 0.8 * clean["interior_mean_confidence"])

    def test_noise_cannot_create_confident_full_vector_observability(self):
        for texture in ("flat", "stripe"):
            for noise in (0.0, 0.1, 0.5):
                with self.subTest(texture=texture, noise=noise):
                    case = self._case(texture=texture, noise=noise)
                    self.assertLess(case["interior_mean_confidence"], 0.2)
                    self.assertEqual(case["status"], "low_confidence")

    def test_intermediate_gaussian_noise_cannot_increase_confidence(self):
        confidence = [self._case(texture="gaussian", noise=noise)["interior_mean_confidence"] for noise in (0.0, 0.05, 0.1, 0.2, 0.5)]
        self.assertTrue(all(noisy < cleaner for cleaner, noisy in zip(confidence, confidence[1:])))

    def test_report_exposes_inaccurate_estimates_instead_of_hiding_in_pooled_rank(self):
        expected = [row for row in self.report["cases"] if row["status"] == "estimated" and row["relative_endpoint_error"] is not None and row["relative_endpoint_error"] > 0.5]
        self.assertEqual(self.report["estimated_cases_above_half_true_speed_error_count"], len(expected))
        self.assertEqual(self.report["estimated_cases_above_half_true_speed_error"], expected)
        # This is a regression on the declared finite fixture family only,
        # not a guaranteed error bound for future fields or observed TEC.
        self.assertEqual(expected, [])
        self.assertEqual(self._case(texture="gaussian")["status"], "estimated")

    def test_missingness_reduces_reported_spatial_support(self):
        complete, moderate, severe = (self._case(missing=value) for value in (0, 0.05, 0.15))
        self.assertLess(moderate["valid_interior_fraction"], complete["valid_interior_fraction"])
        self.assertLess(severe["valid_interior_fraction"], moderate["valid_interior_fraction"])
        self.assertLess(moderate["interior_mean_confidence"], complete["interior_mean_confidence"])
        self.assertLessEqual(severe["interior_mean_confidence"], moderate["interior_mean_confidence"])

    def test_wave_parameter_recovery_with_finite_envelope_gain_change_and_weaker_wave(self):
        for row in self.wave_report["cases"]:
            if row["case"].startswith("noise_"):
                continue
            with self.subTest(case=row["case"]):
                self.assertEqual(row["status"], "estimated")
                self.assertLess(row["wavelength_error_km"], 20)
                self.assertLess(row["period_error_min"], 2)
                self.assertLess(row["bearing_error_deg"], 3)

    def test_wave_evidence_degrades_and_errors_are_recorded_for_perturbed_fields(self):
        rows = {row["case"]: row for row in self.wave_report["cases"]}
        clean = rows["clean"]
        for case, row in rows.items():
            with self.subTest(case=case):
                self.assertGreaterEqual(row["confidence"], 0)
                self.assertLessEqual(row["confidence"], 1)
                self.assertIn("wavelength_error_km", row)
                self.assertIn("period_error_min", row)
                if case != "clean":
                    self.assertLess(row["confidence"], clean["confidence"])
        self.assertLess(rows["finite_envelope_narrow"]["confidence"], rows["finite_envelope_broad"]["confidence"])
        self.assertLess(rows["weaker_second_wave"]["peak_uniqueness"], clean["peak_uniqueness"])

    def test_full_preprocessing_savgol_amplitude_and_phase_response(self):
        import numpy as np
        from ophanim.dynamics import AnalysisConfig, FlowConfig, WaveConfig, analyze_dataset
        from ophanim.dynamics.waves import baseline_transfer
        from ophanim.experiments import make_synthetic_dataset

        for mode in ("retrospective", "causal"):
            for period in (45 * 60, 60 * 60, 120 * 60, 180 * 60):
                with self.subTest(mode=mode, period=period):
                    raw = make_synthetic_dataset("plane_wave", nx=8, ny=8, nt=121, cadence_s=300, period_s=period)
                    config = AnalysisConfig(
                        analysis_mode=mode, as_of="2024-05-10T10:00:00Z" if mode == "causal" else None,
                        baseline_method="savgol", baseline_window_minutes=110,
                        smooth_sigma_km=0, flow=FlowConfig(enabled=False), wave=WaveConfig(enabled=False),
                    )
                    result = analyze_dataset(raw, config).dataset
                    times = np.arange(121)[24:-24] * 300
                    spatial = 2 * np.pi * raw.x.values[4] / 500_000
                    phase = spatial - 2 * np.pi * times / period
                    design = np.column_stack((np.cos(phase), np.sin(phase), np.ones_like(phase)))
                    fitted = np.linalg.lstsq(design, result.dtec.values[24:-24, 4, 4], rcond=None)[0]
                    measured_gain = float(np.hypot(fitted[0], fitted[1]) / 3)
                    measured_phase = float(np.arctan2(-fitted[1], fitted[0]))
                    expected = baseline_transfer(config, 300, period)
                    self.assertAlmostEqual(measured_gain, expected["amplitude_gain"], delta=1e-8)
                    phase_error = float(abs(np.angle(np.exp(1j * (measured_phase - expected["phase_shift_rad"])))))
                    self.assertLess(phase_error, 1e-8)

    def test_finer_numerical_grid_cannot_change_source_cadence_or_resolution(self):
        import numpy as np
        import xarray as xr
        from ophanim.dynamics import AnalysisConfig, FlowConfig, analyze_dataset

        shape = (9, 5, 5)
        raw = xr.Dataset(
            {"tec": (("time", "lat", "lon"), np.full(shape, 20.0)), "observed_mask": (("time", "lat", "lon"), np.ones(shape, bool))},
            coords={"time": np.datetime64("2024-01-01", "ns") + np.arange(9) * np.timedelta64(1, "h"), "lat": np.arange(5) * 2.5 + 30, "lon": np.arange(5) * 5 - 110},
            attrs={"source_kind": "native", "source_metadata": {"source_kind": "native", "native_lat_spacing_deg": 2.5, "native_lon_spacing_deg": 5.0}},
        )
        base = AnalysisConfig(science_spacing_km=100, flow=FlowConfig(enabled=False))
        coarse = analyze_dataset(raw, base)
        fine = analyze_dataset(raw, replace(base, science_spacing_km=25))
        for key in ("native_dx_m", "native_dy_m", "native_cadence_seconds", "source_observation_count", "admissible_period_seconds", "wave"):
            self.assertEqual(coarse.capabilities[key], fine.capabilities[key])
        self.assertGreater(fine.dataset.sizes["x"], coarse.dataset.sizes["x"])
        self.assertEqual(coarse.event["wave"]["status"], "insufficient_samples")
        self.assertEqual(fine.event["display_class"], "uncertain")

    def test_future_values_and_late_revisions_cannot_change_causal_pipeline(self):
        import numpy as np
        import xarray as xr
        from ophanim.dynamics import AnalysisConfig, FlowConfig, WaveConfig, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset

        raw = make_synthetic_dataset("plane_wave", nx=8, ny=8, nt=25)
        config = AnalysisConfig(analysis_mode="causal", as_of="2024-05-10T01:00:00Z", baseline_window_minutes=45, smooth_sigma_km=0, flow=FlowConfig(enabled=False), wave=WaveConfig(enabled=False))
        baseline = analyze_dataset(raw, config)
        changed = raw.copy(deep=True)
        changed.tec.values[13:] = 1e9
        revision = raw.isel(time=[4]).copy(deep=True)
        revision.tec.values[:] = -1e9
        revision["available_at"] = ("time", np.array(["2024-05-11"], dtype="datetime64[ns]"))
        combined = xr.concat([changed, revision], dim="time")
        replay = analyze_dataset(combined, config)
        xr.testing.assert_identical(baseline.dataset, replay.dataset)
        self.assertEqual(baseline.event, replay.event)


if __name__ == "__main__":
    unittest.main()
