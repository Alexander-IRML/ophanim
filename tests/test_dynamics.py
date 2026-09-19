"""Numerical acceptance tests for source-aware, independently reusable dynamics."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import unittest

from ophanim.dynamics import AnalysisConfig, AnalysisError, FlowConfig, WaveConfig

SCIENCE_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "xarray", "scipy", "pyproj", "skimage"))
if SCIENCE_AVAILABLE:
    import numpy as np
    import xarray as xr
    from ophanim.dynamics import analyze_dataset
    from ophanim.dynamics.coords import flow_pixels_to_enu, project_dataset
    from ophanim.dynamics.derivatives import add_derivatives, physical_derivative
    from ophanim.dynamics.flow import estimate_feature_flow
    from ophanim.dynamics.preprocessing import SavitzkyGolayBaseline, masked_gaussian
    from ophanim.dynamics.regularize import eligible_sequence, regularize_time
    from ophanim.dynamics.waves import baseline_transfer
    from ophanim.experiments import degrade_dataset, make_synthetic_dataset


class DynamicsSchemaTests(unittest.TestCase):
    def test_typed_configuration_roundtrip(self):
        config = AnalysisConfig(flow=FlowConfig(method="ilk"), wave=WaveConfig(min_period_minutes=40))
        self.assertEqual(config, AnalysisConfig.from_dict(config.to_dict()))
        with self.assertRaises(Exception):
            config.science_spacing_km = 1

    def test_invalid_configuration_fails(self):
        for values in ({"science_spacing_km": -1}, {"analysis_mode": "causal"}, {"baseline_method": "constant"}, {"as_of": "2024-01-01"}, {"flow": {}}):
            with self.subTest(values=values), self.assertRaises(AnalysisError):
                AnalysisConfig(**values)


@unittest.skipUnless(SCIENCE_AVAILABLE, "install ophanim[shawtynet] for scientific numerical tests")
class DynamicsNumericalTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(baseline_method="constant", constant_background_tecu=20, smooth_sigma_km=0, flow=FlowConfig(enabled=False))
        values.update(overrides)
        return AnalysisConfig(**values)

    def test_physical_derivatives_exact_linear_field(self):
        t, y, x = np.arange(5) * 7.0, np.arange(6) * 3000.0, np.arange(7) * 2000.0
        a = 4 * t[:, None, None] + 3 * y[None, :, None] + 2 * x[None, None, :]
        for axis, coords, truth in ((0, t, 4), (1, y, 3), (2, x, 2)):
            np.testing.assert_allclose(physical_derivative(a, coords, axis), truth, atol=1e-10)

    def test_invalid_derivative_stencil_stays_invalid(self):
        a = np.ones((3, 7, 7))
        a[:, 3, 3] = np.nan
        gx = physical_derivative(a, np.arange(7), 2)
        self.assertTrue(np.isnan(gx[:, 3, 2:5]).all())

    def test_masked_smoothing_does_not_promote_holes(self):
        a = np.ones((1, 15, 15)) * 7
        a[0, 7, 7] = np.nan
        smooth, support = masked_gaussian(a, (0, 1, 1))
        self.assertTrue(np.isnan(smooth[0, 7, 7]))
        np.testing.assert_allclose(smooth[np.isfinite(smooth)], 7)
        self.assertLess(support[0, 7, 7], 1)

    def test_structure_tangent_semantics(self):
        raw = make_synthetic_dataset("quiet", nx=32, ny=32, nt=3, spacing_m=1000)
        x, y = np.meshgrid(raw.x.values, raw.y.values)
        for angle, truth in ((0, np.pi / 2), (np.pi / 2, 0), (np.pi / 4, 3 * np.pi / 4)):
            field = np.cos((np.cos(angle) * x + np.sin(angle) * y) / 5000)
            raw["dtec"] = (("time", "y", "x"), np.broadcast_to(field, raw.tec.shape))
            result = add_derivatives(raw, self.config(structure_sigma_km=2))
            measured = result.structure_orientation.values[:, 5:-5, 5:-5]
            error = np.abs(np.angle(np.exp(2j * (measured - truth)))) / 2
            self.assertLess(float(np.nanmedian(error)), 0.01)
            self.assertGreater(float(result.structure_coherence.mean()), 0.99)

    def test_projection_is_metric_and_original_resolution_survives(self):
        lat, lon = np.linspace(29, 33, 5), np.linspace(-101, -95, 7)
        raw = xr.Dataset({"tec": (("time", "lat", "lon"), np.ones((2, 5, 7)) * 20), "observed_mask": (("time", "lat", "lon"), np.ones((2, 5, 7), bool))}, coords={"time": np.array(["2024-01-01", "2024-01-01T01"], dtype="datetime64[ns]"), "lat": lat, "lon": lon}, attrs={"source_kind": "native"})
        projected = project_dataset(raw, self.config(science_spacing_km=25))
        self.assertEqual(projected.tec.dims, ("time", "y", "x"))
        np.testing.assert_allclose(np.diff(projected.x), 25000)
        self.assertGreater(projected.attrs["source_metadata"]["native_dx_m"], 90000)
        self.assertTrue(projected.spatial_interpolated.values.any())
        self.assertLess(projected.attrs["projection_max_scale_error"], 0.08)

    def test_large_projection_is_rejected(self):
        raw = xr.Dataset({"tec": (("time", "lat", "lon"), np.ones((1, 3, 3))), "observed_mask": (("time", "lat", "lon"), np.ones((1, 3, 3), bool))}, coords={"time": np.array(["2024-01-01"], dtype="datetime64[ns]"), "lat": [-60, 0, 60], "lon": [-120, 0, 120]}, attrs={"source_kind": "native"})
        with self.assertRaises(AnalysisError):
            project_dataset(raw, self.config())

    def test_time_interpolation_is_labeled_and_not_observed(self):
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=4, cadence_s=300).isel(time=[0, 1, 3])
        config = self.config(cadence_seconds=300, maximum_interpolation_gap_seconds=600)
        result = regularize_time(project_dataset(eligible_sequence(raw, config), config), config)
        self.assertTrue(result.temporal_interpolated.values[2].all())
        self.assertFalse(result.observed_mask.values[2].any())
        self.assertTrue(np.isfinite(result.tec.values[2]).all())

    def test_unknown_source_uncertainty_is_not_full_confidence(self):
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=3).drop_vars("source_rms_tecu")
        result = analyze_dataset(raw, self.config()).dataset
        self.assertTrue(np.isnan(result.source_uncertainty).all())
        self.assertTrue((result.observation_support == 1).all())

    def test_revision_selection_respects_availability(self):
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=3)
        revision = raw.isel(time=[1]).copy(deep=True)
        revision["tec"] = revision.tec + 100
        revision["available_at"] = ("time", np.array(["2024-05-11"], dtype="datetime64[ns]"))
        combined = xr.concat([raw, revision], dim="time")
        config = self.config(analysis_mode="causal", as_of="2024-05-10T00:10:00Z")
        selected = eligible_sequence(combined, config)
        self.assertEqual(selected.sizes["time"], 3)
        self.assertEqual(float(selected.tec.isel(time=1).mean()), 20)
        later = eligible_sequence(combined, self.config(as_of="2024-05-12T00:00:00Z"))
        self.assertEqual(float(later.tec.isel(time=1).mean()), 120)

    def test_causal_needs_actual_availability(self):
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=3).drop_vars("available_at")
        with self.assertRaises(AnalysisError):
            analyze_dataset(raw, self.config(analysis_mode="causal", as_of="2024-05-10T01:00:00Z"))

    def test_future_append_cannot_change_causal_result(self):
        raw = make_synthetic_dataset("plane_wave", nx=16, ny=16, nt=31)
        config = self.config(analysis_mode="causal", as_of="2024-05-10T01:40:00Z", baseline_method="savgol", baseline_window_minutes=45, flow=FlowConfig(enabled=True, num_iter=3))
        before = analyze_dataset(raw.isel(time=slice(0, 21)), config)
        after = analyze_dataset(raw, config)
        xr.testing.assert_identical(before.dataset, after.dataset)
        self.assertEqual(before.event, after.event)

    def test_baseline_invalid_window_preserves_macro_not_fake_quiet(self):
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=8, cadence_s=3600)
        result = analyze_dataset(raw, AnalysisConfig(flow=FlowConfig(enabled=False)))
        self.assertTrue(np.isfinite(result.dataset.tec_smooth).all())
        self.assertTrue(np.isnan(result.dataset.dtec).all())
        self.assertEqual(result.event["channel_status"]["baseline"]["status"], "insufficient_samples")
        self.assertEqual(result.event["display_class"], "uncertain")
        self.assertEqual(result.event["event_scores"]["quiet"], 0)

    def test_savgol_preserves_known_constant_background(self):
        raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=31)
        baseline = SavitzkyGolayBaseline(45, 2).estimate(raw.tec)
        np.testing.assert_allclose(baseline, 20, atol=1e-10)

    def test_injected_wave_matches_detrending_transfer(self):
        raw = make_synthetic_dataset("plane_wave", nx=8, ny=8, nt=121, cadence_s=300)
        config = self.config(baseline_method="savgol", baseline_window_minutes=110)
        baseline = SavitzkyGolayBaseline(110, 2).estimate(raw.tec)
        anomaly = raw.tec - baseline
        reference = raw.tec - 20
        response = baseline_transfer(config, 300, 3600)
        measured_gain = float(np.std(anomaly.values[15:-15]) / np.std(reference.values[15:-15]))
        self.assertAlmostEqual(measured_gain, response["amplitude_gain"], delta=0.015)

    def test_quiet_field_is_quiet_and_flow_unobservable(self):
        result = analyze_dataset(make_synthetic_dataset("quiet", nx=16, ny=16, nt=25), self.config(flow=FlowConfig(enabled=True)))
        self.assertEqual(result.event["display_class"], "quiet")
        self.assertGreater(result.event["event_scores"]["quiet"], 0.99)
        self.assertEqual(float(result.dataset.flow_confidence.max()), 0)

    def test_pipeline_is_deterministic_and_does_not_mutate_input(self):
        raw = make_synthetic_dataset("plane_wave", nx=32, ny=32, nt=25)
        original = raw.copy(deep=True)
        first = analyze_dataset(raw, self.config())
        second = analyze_dataset(raw, self.config())
        xr.testing.assert_identical(raw, original)
        xr.testing.assert_identical(first.dataset, second.dataset)
        self.assertEqual(first.event, second.event)
        json.dumps(first.event, allow_nan=False)
        json.dumps(first.capabilities, allow_nan=False)

    def test_synthetic_generators_and_degradation(self):
        from ophanim.experiments.synthetic import KINDS
        for kind in KINDS:
            with self.subTest(kind=kind):
                data = make_synthetic_dataset(kind, nx=16, ny=16, nt=5)
                self.assertEqual(data.tec.shape, (5, 16, 16))
                self.assertEqual(data.attrs["source_kind"], "synthetic")
        raw = make_synthetic_dataset(nx=16, ny=16, nt=9)
        degraded = degrade_dataset(raw, spatial_factor=2, temporal_stride=2, missing_frames=(1,))
        self.assertEqual(degraded.tec.shape, (5, 8, 8))
        self.assertEqual(degraded.attrs["source_metadata"]["native_dx_m"], 50000)
        self.assertFalse(degraded.observed_mask.values[1].any())


@unittest.skipUnless(SCIENCE_AVAILABLE, "scientific extras unavailable")
class DynamicsFlowTests(unittest.TestCase):
    def gaussian_pair(self, dx=1.0, dy=0.0, gain=1.0):
        y, x = np.indices((64, 64))
        previous = np.exp(-((x - 30)**2 + (y - 30)**2) / (2 * 8**2))
        current = gain * np.exp(-((x - 30 - dx)**2 + (y - 30 - dy)**2) / (2 * 8**2))
        return previous, current

    def test_pixel_conversion_all_signs(self):
        for row in (-1, 0, 1):
            for col in (-1, 0, 1):
                u, v = flow_pixels_to_enu(row, col, 2000, 3000, 100)
                self.assertEqual(u, col * 20)
                self.assertEqual(v, row * 30)

    def test_known_translation_directions_and_speeds(self):
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (.25, 0), (3, 0), (6, -3)]
        for dx, dy in shifts:
            with self.subTest(dx=dx, dy=dy):
                previous, current = self.gaussian_pair(dx, dy)
                fields, summary = estimate_feature_flow(previous, current, 10000, 10000, 300)
                support = (current > .2) & (current < .9) & np.isfinite(fields["flow_u"])
                u, v = np.median(fields["flow_u"][support]), np.median(fields["flow_v"][support])
                error = np.hypot(u - dx * 10000 / 300, v - dy * 10000 / 300)
                self.assertLess(error, max(3, np.hypot(dx, dy) * 10000 / 300 * .15))
                bearing = np.degrees(np.arctan2(u, v))
                truth = np.degrees(np.arctan2(dx, dy))
                self.assertLess(abs((bearing - truth + 180) % 360 - 180), 5)
                self.assertGreater(summary["mean_confidence"], .05)

    def test_stationary_growth_is_not_confident_translation(self):
        previous, current = self.gaussian_pair(0, 0, gain=1.5)
        fields, summary = estimate_feature_flow(previous, current, 10000, 10000, 300)
        self.assertGreater(summary["growth_ambiguity"], .95)
        self.assertEqual(summary["status"], "low_confidence")
        self.assertLess(float(fields["flow_confidence"].max()), .01)

    def test_front_aperture_does_not_claim_full_vector(self):
        _, x = np.indices((48, 48))
        previous, current = np.tanh((x - 24) / 4), np.tanh((x - 25) / 4)
        fields, summary = estimate_feature_flow(previous, current, 10000, 10000, 300)
        self.assertEqual(summary["vector_observability"], "normal_only_or_ambiguous")
        self.assertLess(float(fields["flow_confidence"].max()), .001)
        self.assertGreater(float(fields["flow_normal_confidence"].max()), .2)

    def test_missing_support_is_not_inpainted_confidence(self):
        previous, current = self.gaussian_pair()
        previous[25:30, 25:30] = np.nan
        current[25:30, 25:30] = np.nan
        fields, _ = estimate_feature_flow(previous, current, 10000, 10000, 300)
        self.assertTrue((fields["flow_confidence"][25:30, 25:30] == 0).all())
        self.assertTrue(np.isnan(fields["flow_u"][25:30, 25:30]).all())

    def test_ilk_comparison_backend(self):
        previous, current = self.gaussian_pair()
        fields, _ = estimate_feature_flow(previous, current, 10000, 10000, 300, config=FlowConfig(method="ilk"))
        support = (current > .2) & (current < .9)
        self.assertAlmostEqual(float(np.nanmedian(fields["flow_u"][support])), 10000 / 300, delta=5)


@unittest.skipUnless(SCIENCE_AVAILABLE, "scientific extras unavailable")
class DynamicsWaveTests(unittest.TestCase):
    def config(self, **kwargs):
        return AnalysisConfig(baseline_method="constant", constant_background_tecu=20, smooth_sigma_km=0, flow=FlowConfig(enabled=False), **kwargs)

    def test_off_bin_plane_wave_parameters_and_signed_directions(self):
        for bearing in (0, 45, 90, 180, 270):
            with self.subTest(bearing=bearing):
                raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49, bearing_deg=bearing)
                wave = analyze_dataset(raw, self.config()).event["wave"]
                self.assertEqual(wave["status"], "estimated")
                self.assertAlmostEqual(wave["wavelength_km"], 500, delta=20)
                self.assertAlmostEqual(wave["period_min"], 60, delta=2)
                self.assertAlmostEqual(wave["phase_speed_m_s"], 500000 / 3600, delta=10)
                self.assertLess(abs((wave["bearing_deg"] - bearing + 180) % 360 - 180), 3)
                self.assertAlmostEqual(wave["amplitude_tecu"], 3, delta=.3)
                self.assertGreater(wave["confidence"], .5)

    def test_retained_phase_reconstructs_analyzed_wave(self):
        raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49, phase_rad=.7)
        result = analyze_dataset(raw, self.config())
        wave = result.event["wave"]
        xx, yy = np.meshgrid(result.dataset.x.values - wave["reference_x_m"], result.dataset.y.values - wave["reference_y_m"])
        seconds = (result.dataset.time.values - np.datetime64(wave["reference_time"].removesuffix("Z"))).astype("timedelta64[ns]").astype(float) / 1e9
        reconstruction = wave["amplitude_tecu"] * np.cos(2 * np.pi * (wave["k_x_cycles_per_m"] * xx[None] + wave["k_y_cycles_per_m"] * yy[None] + wave["frequency_hz"] * seconds[:, None, None]) + wave["phase_rad"])
        self.assertLess(float(np.sqrt(np.mean((reconstruction - result.dataset.dtec.values)**2))), .1)

    def test_coarse_native_support_is_not_repaired_by_fine_grid(self):
        raw = make_synthetic_dataset("plane_wave", nx=48, ny=48, nt=49)
        meta = dict(raw.attrs["source_metadata"])
        meta.update(source_kind="derived", native_dx_m=500000, native_dy_m=278000, effective_resolution_m=None)
        raw.attrs.update(source_kind="derived", source_metadata=meta)
        result = analyze_dataset(raw, self.config())
        self.assertEqual(result.event["wave"]["status"], "unsupported_resolution")
        self.assertEqual(result.event["wave"]["confidence"], 0)

    def test_two_hour_sampling_cannot_measure_hour_wave(self):
        raw = make_synthetic_dataset("plane_wave", nx=32, ny=32, nt=49)
        coarse = degrade_dataset(raw, temporal_stride=24)
        result = analyze_dataset(coarse, self.config(cadence_seconds=300, maximum_interpolation_gap_seconds=7200))
        self.assertEqual(result.event["wave"]["status"], "insufficient_samples")
        self.assertEqual(result.capabilities["wave_observation_count"], 3)
        self.assertEqual(result.capabilities["native_cadence_seconds"], 7200)

    def test_missing_frame_declines_wave_estimate(self):
        raw = degrade_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48), missing_frames=(20,))
        wave = analyze_dataset(raw, self.config()).event["wave"]
        self.assertEqual(wave["status"], "invalid_support")

    def test_noise_and_crossing_reduce_confidence(self):
        clean = analyze_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48), self.config()).event["wave"]
        noisy = analyze_dataset(make_synthetic_dataset("plane_wave", nx=48, ny=48, noise_std_tecu=5), self.config()).event["wave"]
        crossing = analyze_dataset(make_synthetic_dataset("crossing_waves", nx=48, ny=48), self.config()).event["wave"]
        self.assertGreater(clean["confidence"], noisy["confidence"])
        self.assertGreater(clean["confidence"], crossing["confidence"])
        self.assertLess(crossing["peak_uniqueness"], .5)


if __name__ == "__main__":
    unittest.main()
