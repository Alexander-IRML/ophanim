"""Scientific counterexamples and stronger invariants, not probability claims."""

from dataclasses import asdict, replace
from importlib.util import find_spec
import unittest

AVAILABLE = all(find_spec(name) for name in ("numpy", "scipy", "xarray", "skimage", "pyproj"))


def reliability_sensitivity_sweep():
    """Bounded counterexamples, including failures of heuristic confidence.

    Seeds differ from the original independent-noise sweep. The correlated
    errors are deliberately reported even when inaccurate vectors pass the
    confidence gate; they are not used to retune that gate.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter, shift
    from ophanim.dynamics import AnalysisConfig, FlowConfig, analyze_dataset
    from ophanim.dynamics.flow import estimate_feature_flow
    from ophanim.experiments import make_synthetic_dataset

    config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                            smooth_sigma_km=0, flow=FlowConfig(enabled=False))
    yy, xx = np.indices((64, 64))
    previous = np.exp(-((xx - 30)**2 + (yy - 30)**2) / (2 * 8**2))
    growth = []
    for gain in (1.005, 1.01, 1.02, 1.05, 1.1, 1.5):
        fields, summary = estimate_feature_flow(previous, gain * previous, 10000, 10000, 300)
        eligible = (previous > .2) & (previous < .9) & np.isfinite(fields["flow_speed"])
        growth.append({"gain": gain, "true_velocity_m_s": 0,
                       "status": summary["status"], "valid_cell_count": int(eligible.sum()),
                       "mean_local_confidence": float(fields["flow_confidence"][eligible].mean()),
                       "maximum_local_confidence": float(fields["flow_confidence"][eligible].max()),
                       "median_raw_inferred_speed_m_s": float(np.median(fields["flow_speed"][eligible])),
                       "stationary_photometric_explained": summary["stationary_photometric_explained"]})
    bursts = []
    for active_frames in (49, 25, 13):
        raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49)
        if active_frames < 49:
            raw.tec.values[:-active_frames] = 20
        wave = analyze_dataset(raw, config).event["wave"]
        bursts.append({"active_input_frames": active_frames,
                       **{key: wave[key] for key in ("status", "confidence", "candidate_evidence_score", "period_min", "temporal_support", "confidence_components")}})
    translated = np.exp(-((xx - 31)**2 + (yy - 30)**2) / (2 * 8**2))
    rms_rows = []
    for rms in (None, .001, .1, 1, 100):
        options = {} if rms is None else {"previous_uncertainty": np.full_like(previous, rms),
                                         "current_uncertainty": np.full_like(previous, rms)}
        _, summary = estimate_feature_flow(previous, translated, 10000, 10000, 300, **options)
        rms_rows.append({"source_rms_tecu": rms, "status": summary["status"],
                         "confidence": summary["mean_confidence"],
                         "uncertainty_status": summary["uncertainty_status"],
                         "uncertainty_limitation": summary["uncertainty_limitation"],
                         "measurement_reliability": summary["measurement_reliability"],
                         "source_uncertainty_known_fraction": summary["source_uncertainty_known_fraction"]})
    correlated = []
    for seed in (101, 307, 991):
        rng = np.random.default_rng(seed)
        source = gaussian_filter(rng.normal(size=(48, 48)), 2)
        source /= source.std()
        translated = shift(source, (.5, 1), order=3, mode="nearest")
        for temporal_correlation in (0, .95):
            error = gaussian_filter(rng.normal(size=source.shape), 3)
            error = error / error.std() * 2
            innovation = gaussian_filter(rng.normal(size=source.shape), 3)
            innovation = innovation / innovation.std() * 2
            current = translated + temporal_correlation * error + np.sqrt(1 - temporal_correlation**2) * innovation
            for known_rms in (False, True):
                options = {} if not known_rms else {"previous_uncertainty": np.full_like(source, 2),
                                                    "current_uncertainty": np.full_like(source, 2)}
                fields, summary = estimate_feature_flow(source + error, current, 10000, 10000, 300, **options)
                eligible = np.zeros(source.shape, bool)
                eligible[8:-8, 8:-8] = True
                eligible &= np.isfinite(fields["flow_u"]) & np.isfinite(fields["flow_v"])
                endpoint = np.hypot(fields["flow_u"] - 10000 / 300, fields["flow_v"] - 5000 / 300)
                epe = float(endpoint[eligible].mean()) if eligible.any() else None
                admitted_pixels = eligible & (fields["flow_confidence"] >= FlowConfig().minimum_confidence)
                inaccurate_pixels = admitted_pixels & (endpoint > .5 * np.hypot(10000 / 300, 5000 / 300))
                interpretation_pixels = eligible & (fields["flow_interpretation_confidence"] > 0)
                correlated.append({"seed": seed, "temporal_error_correlation_coefficient": temporal_correlation,
                                   "provided_source_rms_tecu": 2 if known_rms else None,
                                   "status": summary["status"], "confidence": summary["mean_confidence"],
                                   "uncertainty_status": summary["uncertainty_status"],
                                   "uncertainty_limitation": summary["uncertainty_limitation"],
                                   "valid_interior_fraction": float(eligible.sum() / 32**2),
                                   "locally_admitted_pixel_count": int(admitted_pixels.sum()),
                                   "locally_admitted_inaccurate_pixel_count": int(inaccurate_pixels.sum()),
                                   "locally_admitted_inaccurate_fraction": float(inaccurate_pixels.sum() / admitted_pixels.sum()) if admitted_pixels.any() else None,
                                   "interpretation_eligible_pixel_count": int(interpretation_pixels.sum()),
                                   "interpretation_eligible_inaccurate_pixel_count": int((interpretation_pixels & (endpoint > .5 * np.hypot(10000 / 300, 5000 / 300))).sum()),
                                   "interpretation_status": summary["interpretation_status"],
                                   "interpretation_policy": summary["interpretation_policy"],
                                   "mean_endpoint_error_m_s": epe,
                                   "relative_endpoint_error": None if epe is None else epe / np.hypot(10000 / 300, 5000 / 300)})
    inaccurate = [row for row in correlated if row["status"] == "estimated"
                  and row["relative_endpoint_error"] is not None and row["relative_endpoint_error"] > .5]
    rng = np.random.default_rng(867)
    control_previous = gaussian_filter(rng.normal(size=(48, 48)), 2)
    control_previous /= control_previous.std()
    control_current = shift(control_previous, (.5, 1), order=3, mode="nearest")
    control, control_summary = estimate_feature_flow(control_previous, control_current, 10000, 10000, 300,
        previous_uncertainty=np.full_like(control_previous, .05), current_uncertainty=np.full_like(control_previous, .05))
    control_mask = np.zeros(control_previous.shape, bool)
    control_mask[8:-8, 8:-8] = True
    control_mask &= control["flow_interpretation_confidence"] > 0
    control_error = np.hypot(control["flow_u"] - 10000 / 300, control["flow_v"] - 5000 / 300)
    positive_control = {"seed": 867, "provided_rms_proxy_tecu": .05,
                        "status": control_summary["status"], "interpretation_status": control_summary["interpretation_status"],
                        "interpretation_eligible_pixel_count": int(control_mask.sum()),
                        "mean_eligible_endpoint_error_m_s": float(control_error[control_mask].mean()) if control_mask.any() else None,
                        "semantics": "clean translating texture with a declared RMS proxy; eligibility is not a general accuracy guarantee"}
    return {"schema": "ophanim-reliability-counterexamples/1",
            "semantics": "Synthetic sensitivity and explicit failure reporting, not confidence calibration or real-TEC accuracy validation.",
            "gradual_stationary_growth": growth, "wave_burst_occupancy": bursts,
            "source_rms_sensitivity": rms_rows, "held_out_correlated_errors": correlated,
            "known_rms_translation_control": positive_control,
            "correlated_error_admitted_inaccurate_cases": inaccurate,
            "correlated_error_admitted_inaccurate_count": len(inaccurate),
            "correlated_error_admitted_inaccurate_unknown_rms_count": sum(row["provided_source_rms_tecu"] is None for row in inaccurate),
            "correlated_error_admitted_inaccurate_known_rms_count": sum(row["provided_source_rms_tecu"] is not None for row in inaccurate),
            "experiment": {"dx_m": 10000, "dy_m": 10000, "dt_s": 300,
                           "growth_mask": "0.2 < previous Gaussian amplitude < 0.9, finite solver speed",
                           "correlated_noise_seeds": [101, 307, 991],
                           "noise_model": "spatial Gaussian filtering sigma=3 cells, normalized error standard deviation=2 TECU; temporally correlated additive errors",
                           "source_signal_standard_deviation_tecu": 1,
                           "correlated_field_shape": [48, 48], "interior_margin_cells": 8,
                           "flow_config": asdict(FlowConfig()), "wave_analysis_config": config.to_dict()},
            "limitations": ["Unknown RMS does not become measured zero error",
                            "Spatially smooth correlated errors can look like resolvable moving signal and pass heuristic motion gates",
                            "RMS sensitivity can reject these declared fixtures but is not a general accuracy guarantee",
                            "Pair-level abstention does not guarantee that every individual pixel has low confidence; local admitted-error counts are reported separately",
                            "Conservative presentation eligibility is separate from unchanged raw fit confidence and requires pair admission, known aligned RMS proxies, and the existing local gate",
                            "No confidence thresholds were fitted to these held-out seeds",
                            "Solver vectors are retained when trust is withheld; no fabricated zero velocity"]}


@unittest.skipUnless(AVAILABLE, "optional scientific dependencies unavailable")
class DynamicsReliabilityTests(unittest.TestCase):
    def gaussian(self, *, gain=1.0, shift=0.0):
        import numpy as np
        y, x = np.indices((64, 64))
        previous = np.exp(-((x - 30)**2 + (y - 30)**2) / (2 * 8**2))
        current = gain * np.exp(-((x - 30 - shift)**2 + (y - 30)**2) / (2 * 8**2))
        return previous, current

    def config(self, **options):
        from ophanim.dynamics import AnalysisConfig, FlowConfig
        settings = dict(baseline_method="constant", constant_background_tecu=20,
                        smooth_sigma_km=0, flow=FlowConfig(enabled=False))
        settings.update(options)
        return AnalysisConfig(**settings)

    def test_gradual_stationary_growth_cannot_have_confident_local_vectors(self):
        import numpy as np
        from ophanim.dynamics.flow import estimate_feature_flow
        for gain in (1.005, 1.01, 1.02, 1.05, 1.1, 1.5):
            with self.subTest(gain=gain):
                previous, current = self.gaussian(gain=gain)
                fields, report = estimate_feature_flow(previous, current, 10000, 10000, 300)
                blob = (previous > .2) & (previous < .9) & np.isfinite(fields["flow_speed"])
                self.assertLess(float(fields["flow_confidence"][blob].max()), .01)
                self.assertGreater(report["stationary_photometric_explained"], .99)
                self.assertEqual(report["status"], "low_confidence")
                # The solver's inference is retained for audit, not changed to
                # a fabricated measured zero to satisfy a ground-truth test.
                self.assertGreater(float(np.median(fields["flow_speed"][blob])), .01)

    def test_translation_survives_but_mixed_growth_exposes_more_ambiguity(self):
        import numpy as np
        from ophanim.dynamics.flow import estimate_feature_flow
        previous, moving = self.gaussian(shift=1)
        clean, report = estimate_feature_flow(previous, moving, 10000, 10000, 300)
        _, growing = self.gaussian(shift=1, gain=1.1)
        mixed, mixed_report = estimate_feature_flow(previous, growing, 10000, 10000, 300)
        mask = (moving > .2) & (moving < .9) & np.isfinite(clean["flow_u"])
        self.assertAlmostEqual(float(np.median(clean["flow_u"][mask])), 10000 / 300, delta=3)
        self.assertGreater(float(np.mean(clean["flow_confidence"][mask])), .4)
        self.assertLess(report["stationary_photometric_explained"], .05)
        self.assertGreater(mixed_report["growth_ambiguity"], report["growth_ambiguity"])
        self.assertLess(float(np.mean(mixed["flow_confidence"][mask])), float(np.mean(clean["flow_confidence"][mask])))

    def test_growth_ambiguity_is_local_when_another_region_translates(self):
        import numpy as np
        from scipy.ndimage import gaussian_filter, shift
        from ophanim.dynamics.flow import estimate_feature_flow
        y, x = np.indices((96, 96))
        blob = np.exp(-((x - 24)**2 + (y - 48)**2) / (2 * 7**2))
        texture = gaussian_filter(np.random.default_rng(17).normal(size=(96, 96)), 2)
        texture *= np.exp(-((x - 74)**2 + (y - 48)**2) / (2 * 9**2))
        previous = blob + texture
        current = 1.05 * blob + shift(texture, (0, 1), mode="nearest")
        fields, _ = estimate_feature_flow(previous, current, 10000, 10000, 300)
        stationary = (blob > .3) & (x < 28)
        self.assertGreater(float(np.median(fields["growth_ambiguity"][stationary])), .9)
        self.assertLess(float(np.max(fields["flow_confidence"][stationary])), .2)

    def test_source_rms_changes_motion_trust_not_solver_displacement(self):
        import numpy as np
        from ophanim.dynamics.flow import estimate_feature_flow
        previous, current = self.gaussian(shift=1)
        estimates = []
        for rms in (.001, 100):
            estimates.append(estimate_feature_flow(previous, current, 10000, 10000, 300,
                previous_uncertainty=np.full_like(previous, rms), current_uncertainty=np.full_like(previous, rms)))
        low, high = estimates
        np.testing.assert_allclose(low[0]["flow_u"], high[0]["flow_u"], equal_nan=True)
        self.assertLess(high[1]["mean_confidence"], low[1]["mean_confidence"] * .001)
        self.assertEqual(high[1]["source_uncertainty_known_fraction"], 1)

    def test_unknown_rms_strong_fit_is_retained_but_ineligible_for_interpretation(self):
        import numpy as np
        from scipy.ndimage import gaussian_filter, shift
        from ophanim.dynamics.flow import estimate_feature_flow
        from ophanim.dynamics import FlowConfig
        previous = gaussian_filter(np.random.default_rng(867).normal(size=(48, 48)), 2)
        previous /= previous.std()
        current = shift(previous, (.5, 1), order=3, mode="nearest")
        fields, summary = estimate_feature_flow(previous, current, 10000, 10000, 300)
        self.assertEqual(summary["status"], "estimated")
        self.assertGreater(float(fields["flow_confidence"].max()), .4)
        self.assertGreater(float(np.nanmedian(fields["flow_speed"])), 0)
        self.assertFalse(fields["flow_interpretation_confidence"].any())
        self.assertEqual(summary["interpretation_status"], "withheld")
        self.assertEqual(summary["interpretation_eligible_fraction"], 0)
        zero_gate, _ = estimate_feature_flow(previous, current, 10000, 10000, 300, config=FlowConfig(minimum_confidence=0))
        self.assertFalse(zero_gate["flow_interpretation_confidence"].any())

    def test_pipeline_unknown_rms_withholds_public_motion_derivatives(self):
        import numpy as np
        from ophanim.dynamics import FlowConfig, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        raw = make_synthetic_dataset("translating_gaussian", nx=48, ny=48, nt=9).drop_vars("source_rms_tecu")
        result = analyze_dataset(raw, self.config(flow=FlowConfig()))
        self.assertTrue(np.isfinite(result.dataset.flow_u).any())
        self.assertTrue(np.isfinite(result.dataset.advection_residual_raw).any())
        for name in ("advection_residual", "flow_divergence", "flow_vorticity", "flow_strain"):
            self.assertTrue(np.isnan(result.dataset[name]).all())
        self.assertEqual(result.event["event_scores"]["flow"], 0)
        self.assertIn("raw_flow_score", result.event["research_motion"])

    def test_untrusted_motion_cannot_hide_growth_in_interpretable_residual(self):
        import numpy as np
        from ophanim.dynamics import FlowConfig, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        raw = make_synthetic_dataset("growing_gaussian", nx=32, ny=32, nt=9)
        result = analyze_dataset(raw, self.config(flow=FlowConfig(enabled=True)))
        self.assertTrue(np.isfinite(result.dataset.advection_residual_raw).any())
        self.assertTrue(np.isnan(result.dataset.advection_residual).all())
        self.assertGreater(float(result.dataset.dtec_dt.max()), 0)
        self.assertFalse(result.event["metrics"]["residual_interpretable"])
        self.assertGreater(result.event["metrics"]["growth_ambiguity"], .99)

    def test_analytic_rotation_expansion_and_shear_kinematics(self):
        import numpy as np
        from ophanim.dynamics.flow import flow_kinematics
        x, y = np.arange(9) * 2000.0, np.arange(8) * 3000.0
        xx, yy = np.meshgrid(x, y)
        for a, b, c, d in ((0, -.002, .002, 0), (.003, 0, 0, .003), (.001, .004, -.002, -.003)):
            with self.subTest(coefficients=(a, b, c, d)):
                result = flow_kinematics(a * xx + b * yy, c * xx + d * yy, x, y)
                np.testing.assert_allclose(result["flow_divergence"], a + d, atol=1e-14)
                np.testing.assert_allclose(result["flow_vorticity"], c - b, atol=1e-14)
                np.testing.assert_allclose(result["flow_strain"], np.hypot(a - d, b + c), atol=1e-14)

    def test_analytic_translation_and_growth_residual_have_physical_units(self):
        import numpy as np
        from ophanim.dynamics.flow import advection_residual
        x, y = np.meshgrid(np.linspace(-100000, 100000, 25), np.linspace(-80000, 80000, 21))
        sigma, u, v, rate = 40000, 200, -100, .0001
        a = 3 * np.exp(-(x*x + y*y) / (2 * sigma**2))
        gx, gy = -x / sigma**2 * a, -y / sigma**2 * a
        for growth in (0, rate):
            temporal = -u * gx - v * gy + growth * a
            np.testing.assert_allclose(advection_residual(temporal, gx, gy, u, v), growth * a, atol=1e-15)

    def test_wave_burst_cannot_borrow_quiet_history_to_claim_three_cycles(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        for frames in (25, 13):
            raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49)
            raw.tec.values[:-frames] = 20
            wave = analyze_dataset(raw, self.config()).event["wave"]
            self.assertEqual(wave["status"], "insufficient_persistence")
            self.assertLess(wave["temporal_support"]["effective_cycles"], 3)
            self.assertEqual(wave["confidence"], 0)
            self.assertGreater(wave["temporal_support"]["temporal_phase_coherence"], .8)

    def test_clean_wave_components_and_spatial_spectrum_are_auditable(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        wave = analyze_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49), self.config()).event["wave"]
        self.assertEqual(wave["status"], "estimated")
        self.assertGreater(wave["confidence"], .5)
        self.assertGreater(wave["temporal_support"]["effective_cycles"], 3.9)
        self.assertGreater(wave["structural_orientation_consistency"], .99)
        spectrum = wave["spectrum_summary"]
        self.assertEqual(len(spectrum["spatial_power"]), len(spectrum["ky_rad_per_m"]))
        self.assertEqual(len(spectrum["spatial_power"][0]), len(spectrum["kx_rad_per_m"]))
        self.assertLessEqual(len(spectrum["spatial_power"]), 128)
        self.assertAlmostEqual(spectrum["selected_peak"]["wavelength_km"], 500, delta=20)

    def test_conflicting_flow_lowers_wave_direction_trust_without_flipping_it(self):
        import numpy as np
        from ophanim.dynamics import analyze_dataset
        from ophanim.dynamics.waves import estimate_wave
        from ophanim.experiments import make_synthetic_dataset
        config = self.config()
        result = analyze_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49), config)
        result.dataset.flow_u.values[:] = -100
        result.dataset.flow_v.values[:] = 0
        result.dataset.flow_confidence.values[:] = 1
        result.dataset.flow_interpretation_confidence.values[:] = 1
        wave = estimate_wave(result.dataset, config, result.capabilities)
        self.assertEqual(wave["status"], "conflicting_evidence")
        self.assertEqual(wave["confidence"], 0)
        self.assertLess(wave["flow_direction_agreement"], -.99)
        self.assertLess(abs((wave["bearing_deg"] - 90 + 180) % 360 - 180), 3)
        self.assertTrue(np.isfinite(wave["frequency_hz"]))

    def test_inconsistent_structure_and_large_rms_reduce_wave_trust(self):
        import numpy as np
        from ophanim.dynamics import analyze_dataset
        from ophanim.dynamics.waves import estimate_wave
        from ophanim.experiments import make_synthetic_dataset
        config = self.config()
        raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49)
        result = analyze_dataset(raw, config)
        reference = result.event["wave"]["confidence"]
        result.dataset.structure_orientation.values[:] += np.pi / 2
        contradictory = estimate_wave(result.dataset, config, result.capabilities)
        self.assertLess(contradictory["confidence"], reference * .01)
        raw.source_rms_tecu.values[:] = 100
        uncertain = analyze_dataset(raw, config)
        self.assertLess(uncertain.event["wave"]["confidence"], reference * .01)
        self.assertEqual(uncertain.event["display_class"], "uncertain")

    def test_unknown_rms_stays_unknown_and_synthetic_data_stays_synthetic(self):
        import numpy as np
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=3)
        result = analyze_dataset(raw, self.config()).dataset
        for name in ("tec", "observed_mask", "source_uncertainty"):
            self.assertEqual(result[name].attrs["semantic_class"], "synthetic")
        unknown = analyze_dataset(raw.drop_vars("source_rms_tecu"), self.config())
        self.assertTrue(np.isnan(unknown.dataset.measurement_reliability).all())
        self.assertFalse(unknown.dataset.source_uncertainty_known.any())
        self.assertIsNone(unknown.event["metrics"]["measurement_reliability"])

    def test_compact_events_do_not_disappear_under_quiet_background_padding(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        for size in (48, 80):
            raw = make_synthetic_dataset("growing_gaussian", nx=size, ny=size, nt=5)
            event = analyze_dataset(raw, self.config()).event
            self.assertEqual(event["display_class"], "localized_anomaly")
            self.assertGreater(event["event_scores"]["localized"], event["event_scores"]["quiet"])
            self.assertEqual(event["region"]["selection"], "dominant_connected_anomaly_with_one_cell_context")

    def test_moving_front_requires_elongated_gradient_geometry(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        event = analyze_dataset(make_synthetic_dataset("moving_front", nx=48, ny=48, nt=25), self.config()).event
        self.assertEqual(event["display_class"], "front_like")
        self.assertGreater(event["morphology"]["gradient_structure"]["elongation"], 4)
        self.assertGreater(event["event_scores"]["front"], .5)

    def test_explicit_candidate_focus_is_validated_and_recorded(self):
        from ophanim.dynamics import AnalysisConfig, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        bounds = {"south": 29, "north": 31, "west": -99, "east": -97}
        event = analyze_dataset(make_synthetic_dataset("growing_gaussian", nx=48, ny=48, nt=5), self.config(event_focus_bounds=bounds)).event
        self.assertEqual(event["region"]["selection"], "explicit_candidate_bounds")
        self.assertEqual(event["region"]["bounds"], bounds)
        for bad in ({"north": 1}, {"south": 91, "north": 92, "west": 0, "east": 1}):
            with self.assertRaises(ValueError):
                AnalysisConfig(event_focus_bounds=bad)

    def test_explicit_flow_frames_are_sparse_and_keep_adjacent_native_pairs(self):
        import numpy as np
        from ophanim.dynamics import FlowConfig, analyze_dataset
        from ophanim.dynamics.flow import add_flow
        from ophanim.experiments import make_synthetic_dataset
        raw = make_synthetic_dataset("translating_gaussian", nx=32, ny=32, nt=13)
        config = self.config(flow=FlowConfig(enabled=False))
        result = analyze_dataset(raw, config)
        selected = result.dataset.time.values[[2, 10]]
        fields, records = add_flow(result.dataset, replace(config, flow=FlowConfig(enabled=True), short_window_minutes=5), result.capabilities, frame_times=selected)
        self.assertEqual(len(records), 2)
        self.assertTrue(np.isfinite(fields.flow_u.isel(time=2)).any())
        self.assertTrue(np.isfinite(fields.flow_u.isel(time=10)).any())
        self.assertFalse(np.isfinite(fields.flow_u.isel(time=9)).any())


@unittest.skipUnless(AVAILABLE, "optional scientific dependencies unavailable")
class HeldOutReliabilityAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = reliability_sensitivity_sweep()

    def test_report_preserves_correlated_error_failures_and_unknown_rms(self):
        import json
        json.dumps(self.report, allow_nan=False)
        cases = self.report["held_out_correlated_errors"]
        self.assertEqual(len(cases), 12)
        expected = [row for row in cases if row["status"] == "estimated"
                    and row["relative_endpoint_error"] is not None and row["relative_endpoint_error"] > .5]
        self.assertEqual(self.report["correlated_error_admitted_inaccurate_cases"], expected)
        self.assertEqual(self.report["correlated_error_admitted_inaccurate_count"], len(expected))
        for row in cases:
            self.assertGreaterEqual(row["locally_admitted_pixel_count"], row["locally_admitted_inaccurate_pixel_count"])
            self.assertGreaterEqual(row["locally_admitted_inaccurate_pixel_count"], 0)
        self.assertEqual(self.report["source_rms_sensitivity"][0]["measurement_reliability"], None)
        self.assertEqual(self.report["source_rms_sensitivity"][0]["uncertainty_status"], "unavailable")

    def test_known_source_error_reduces_trust_without_rewriting_inference(self):
        cases = self.report["held_out_correlated_errors"]
        for unknown, known in zip(cases[::2], cases[1::2]):
            with self.subTest(seed=known["seed"], correlation=known["temporal_error_correlation_coefficient"]):
                self.assertEqual(unknown["mean_endpoint_error_m_s"], known["mean_endpoint_error_m_s"])
                self.assertLess(known["confidence"], unknown["confidence"])
                self.assertEqual(known["status"], "low_confidence")
                self.assertEqual(known["uncertainty_status"], "available")
                self.assertEqual(unknown["uncertainty_status"], "unavailable")
                self.assertIn("not verified motion accuracy", unknown["uncertainty_limitation"])
                self.assertEqual(known["interpretation_eligible_pixel_count"], 0)
                self.assertEqual(unknown["interpretation_eligible_pixel_count"], 0)
                self.assertEqual(known["interpretation_status"], "withheld")
        confidence = [row["confidence"] for row in self.report["source_rms_sensitivity"][1:]]
        self.assertTrue(all(low > high for low, high in zip(confidence, confidence[1:])))

    def test_clean_known_rms_translation_retains_eligible_vectors(self):
        control = self.report["known_rms_translation_control"]
        self.assertEqual(control["interpretation_status"], "eligible")
        self.assertEqual(control["status"], "estimated")
        self.assertGreater(control["interpretation_eligible_pixel_count"], 900)
        self.assertLess(control["mean_eligible_endpoint_error_m_s"], 4)


if __name__ == "__main__":
    unittest.main()
