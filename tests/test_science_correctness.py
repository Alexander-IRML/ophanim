"""Sampling, peak-search and synthetic-provenance correctness regressions."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AVAILABLE = all(find_spec(name) for name in ("numpy", "scipy", "xarray", "skimage", "pyproj"))


@unittest.skipUnless(AVAILABLE, "optional scientific dependencies unavailable")
class SamplingAndFitCorrectnessTests(unittest.TestCase):
    def config(self, **changes):
        from ophanim.dynamics import AnalysisConfig, FlowConfig
        return AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                              smooth_sigma_km=0, flow=FlowConfig(enabled=False), **changes)

    def test_unsafe_temporal_decimation_abstains_without_mutating_source(self):
        import xarray as xr
        from ophanim.dynamics import AnalysisError, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=289, period_s=72 * 60)
        original = raw.copy(deep=True)
        native = analyze_dataset(raw, self.config(wave_window_hours=24)).event["wave"]
        self.assertEqual(native["status"], "estimated")
        self.assertAlmostEqual(native["period_min"], 72, delta=.02)
        self.assertAlmostEqual(native["bearing_deg"], 90, delta=.02)
        with self.assertRaisesRegex(AnalysisError, "downsampling.*anti-alias"):
            analyze_dataset(raw, self.config(wave_window_hours=24, cadence_seconds=2700))
        xr.testing.assert_identical(raw, original)

    def test_capabilities_use_analyzed_cadence_as_well_as_native_history(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.dynamics.capabilities import assess_capabilities
        from ophanim.experiments import make_synthetic_dataset
        config = self.config(wave_window_hours=24)
        result = analyze_dataset(make_synthetic_dataset("plane_wave", nx=32, ny=32, nt=289), config)
        coarse = result.dataset.isel(time=slice(None, None, 9))
        capability = assess_capabilities(coarse, config)
        self.assertEqual(capability["native_cadence_seconds"], 300)
        self.assertEqual(capability["analysis_cadence_seconds"], 2700)
        self.assertEqual(capability["wave_admission_cadence_seconds"], 2700)
        self.assertEqual(capability["wave"]["status"], "insufficient_samples")
        self.assertIsNone(capability["admissible_period_seconds"])

    def test_wave_defensively_checks_its_actual_cube_even_with_stale_capabilities(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.dynamics.waves import estimate_wave
        from ophanim.experiments import make_synthetic_dataset
        config = self.config(wave_window_hours=24)
        result = analyze_dataset(make_synthetic_dataset("plane_wave", nx=32, ny=32, nt=289), config)
        wave = estimate_wave(result.dataset.isel(time=slice(None, None, 9)), config, result.capabilities)
        self.assertEqual(wave["status"], "insufficient_samples")
        self.assertEqual(wave["confidence"], 0)

    def test_three_cycle_boundary_periods_refine_instead_of_sticking_to_wrong_bin(self):
        import json
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        for period in (74, 76, 78, 80):
            with self.subTest(period=period):
                raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49, period_s=period * 60)
                wave = analyze_dataset(raw, self.config()).event["wave"]
                self.assertEqual(wave["status"], "estimated")
                self.assertAlmostEqual(wave["period_min"], period, delta=.02)
                self.assertAlmostEqual(wave["bearing_deg"], 90, delta=.02)
                self.assertGreater(wave["confidence"], .5)
                fit = wave["spectral_fit"]
                self.assertEqual(fit["status"], "converged")
                self.assertTrue(fit["admission_applied_after_fit"])
                self.assertAlmostEqual(fit["seed_bins"][0], -3)
                self.assertEqual(fit["admissible_period_seconds"], [1800, 4800])
                json.dumps(fit, allow_nan=False)

    def test_unsupported_period_is_not_forced_onto_admission_boundary(self):
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        wave = analyze_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49, period_s=84 * 60), self.config()).event["wave"]
        self.assertEqual(wave["status"], "insufficient_samples")
        self.assertEqual(wave["confidence"], 0)
        self.assertAlmostEqual(wave["spectrum_summary"]["selected_peak"]["period_min"], 84, delta=.02)

    def test_failed_refinement_is_exposed_not_called_estimated(self):
        import numpy as np
        from ophanim.dynamics import analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        def fail(objective, initial, **options):
            return SimpleNamespace(success=False, x=np.asarray(initial), fun=objective(initial), message="test convergence failure")
        with patch("ophanim.dynamics.waves.minimize", side_effect=fail):
            wave = analyze_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49), self.config()).event["wave"]
        self.assertNotEqual(wave["status"], "estimated")
        self.assertEqual(wave["spectral_fit"]["status"], "failed")
        self.assertIn("convergence failure", wave["spectral_fit"]["message"])


@unittest.skipUnless(AVAILABLE, "optional scientific dependencies unavailable")
class SyntheticProvenanceCorrectnessTests(unittest.TestCase):
    def test_identity_degradation_preserves_existing_rms_and_input(self):
        import numpy as np
        import xarray as xr
        from ophanim.experiments import degrade_dataset, make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5, noise_std_tecu=3)
        original = source.copy(deep=True)
        degraded = degrade_dataset(source)
        np.testing.assert_allclose(degraded.source_rms_tecu, 3)
        np.testing.assert_allclose(degraded.tec, source.tec)
        xr.testing.assert_identical(source, original)

    def test_block_average_retains_conservative_rms_without_independence_claim(self):
        import numpy as np
        from ophanim.experiments import degrade_dataset, make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5, noise_std_tecu=3)
        degraded = degrade_dataset(source, spatial_factor=2, temporal_stride=2)
        np.testing.assert_allclose(degraded.source_rms_tecu, 3)
        self.assertEqual(degraded.source_rms_tecu.shape, (3, 6, 6))
        self.assertIn("covariance_agnostic", degraded.source_rms_tecu.attrs["uncertainty_model"])

    def test_added_noise_preserves_parent_error_without_assuming_independence(self):
        import numpy as np
        from ophanim.experiments import degrade_dataset, make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5, noise_std_tecu=3)
        degraded = degrade_dataset(source, noise_std_tecu=4)
        np.testing.assert_allclose(degraded.source_rms_tecu, 7)
        self.assertIn("need not be independent", degraded.attrs["degradation"]["uncertainty_assumptions"])

    def test_unknown_parent_rms_remains_unknown_even_when_added_noise_is_known(self):
        import numpy as np
        from ophanim.experiments import degrade_dataset, make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5).drop_vars("source_rms_tecu")
        for factor in (1, 2):
            degraded = degrade_dataset(source, spatial_factor=factor, noise_std_tecu=1)
            self.assertTrue(np.isnan(degraded.source_rms_tecu).all())

    def test_partial_unknown_and_missing_cells_are_not_false_zero_error(self):
        import numpy as np
        from ophanim.experiments import degrade_dataset, make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5, noise_std_tecu=3)
        source.source_rms_tecu.values[0, 0, 0] = np.nan
        degraded = degrade_dataset(source, spatial_factor=2, missing_frames=(1,))
        self.assertTrue(np.isnan(degraded.source_rms_tecu.values[0, 0, 0]))
        self.assertTrue(np.isnan(degraded.source_rms_tecu.values[1]).all())
        self.assertTrue(np.isnan(degraded.tec.values[1]).all())
        self.assertFalse(degraded.observed_mask.values[1].any())

    def test_metric_grid_preserves_interpolation_masks_and_support_discount(self):
        import numpy as np
        import xarray as xr
        from ophanim.dynamics import AnalysisConfig, FlowConfig, analyze_dataset
        from ophanim.dynamics.coords import project_dataset
        from ophanim.experiments import make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5)
        source["spatial_interpolated"] = xr.ones_like(source.observed_mask, dtype=bool)
        source = source.isel(x=slice(None, None, -1))
        config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20, flow=FlowConfig(enabled=False))
        projected = project_dataset(source, config)
        self.assertTrue(projected.spatial_interpolated.values.all())
        result = analyze_dataset(source, config)
        np.testing.assert_allclose(result.dataset.observation_support, config.spatial_interpolation_weight)

    def test_metric_interpolation_flag_shape_is_validated(self):
        from ophanim.dynamics import AnalysisConfig, AnalysisError
        from ophanim.dynamics.coords import project_dataset
        from ophanim.experiments import make_synthetic_dataset
        source = make_synthetic_dataset("quiet", nx=12, ny=12, nt=5)
        source["spatial_interpolated"] = source.observed_mask.isel(time=0)
        with self.assertRaisesRegex(AnalysisError, "spatial_interpolated"):
            project_dataset(source, AnalysisConfig())

    def test_front_truth_resolves_normal_speed_to_east_and_north_components(self):
        import numpy as np
        from ophanim.experiments import make_synthetic_dataset
        for bearing in (0, 45, 90, 225):
            for speed in (-100, 100):
                with self.subTest(bearing=bearing, speed=speed):
                    data = make_synthetic_dataset("moving_front", nx=12, ny=12, nt=5, bearing_deg=bearing,
                                                  velocity_u_m_s=speed, velocity_v_m_s=333)
                    truth = data.attrs["synthetic_truth"]
                    theta = np.deg2rad(bearing)
                    self.assertAlmostEqual(truth["velocity_u_m_s"], speed * np.sin(theta))
                    self.assertAlmostEqual(truth["velocity_v_m_s"], speed * np.cos(theta))
                    x, y = np.meshgrid(data.x.values, data.y.values)
                    times = (data.time.values - data.time.values[0]) / np.timedelta64(1, "s")
                    times = times - times[-1] / 2
                    normal_speed = truth["velocity_u_m_s"] * np.sin(theta) + truth["velocity_v_m_s"] * np.cos(theta)
                    expected = 20 + 3 * np.tanh((x[None] * np.sin(theta) + y[None] * np.cos(theta) - normal_speed * times[:, None, None]) / (150000 / 3))
                    np.testing.assert_allclose(data.tec, expected)
                    self.assertEqual(truth["generator_velocity_arguments"]["velocity_v_m_s"], 333)

    def test_existing_northward_front_scenario_keeps_its_motion_and_correct_truth(self):
        from ophanim.experiments.scenarios import make_scenario, scenario_from_candidate
        recipe = scenario_from_candidate(None, {"kind": "moving_front", "bearing_deg": 0, "speed_m_s": 100,
                                              "extent_km": 200, "spacing_km": 25, "width_km": 50, "duration_minutes": 20})
        data = make_scenario(recipe)
        truth = data.attrs["synthetic_truth"]
        self.assertAlmostEqual(truth["velocity_u_m_s"], 0)
        self.assertAlmostEqual(truth["velocity_v_m_s"], 100)


if __name__ == "__main__":
    unittest.main()
