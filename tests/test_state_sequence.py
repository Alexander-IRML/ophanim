"""Exact-epoch inference timelines, bounded work and causal prefix isolation."""
from dataclasses import replace
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

from ophanim.dynamics.sequence import SequenceBudget, normalize_frame_times

SCIENCE = all(importlib.util.find_spec(name) is not None
              for name in ("numpy", "xarray", "scipy", "skimage", "pyproj", "zarr"))


class SequenceBudgetTests(unittest.TestCase):
    def test_resource_limits_are_hard_bounded(self):
        for values in ({"max_frames": 25}, {"max_work_cells": 24_000_001}, {"max_cube_cells": True}):
            with self.assertRaises(ValueError):
                SequenceBudget(**values)


@unittest.skipUnless(SCIENCE, "requires optional science environment")
class StateSequenceTests(unittest.TestCase):
    def setUp(self):
        from ophanim.dynamics import AnalysisConfig, FlowConfig
        from ophanim.experiments import make_synthetic_dataset
        self.raw = make_synthetic_dataset("plane_wave", nx=25, ny=25, nt=49,
                                          wavelength_m=200_000, period_s=1800)
        self.config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                                    smooth_sigma_km=0, flow=FlowConfig(enabled=False))
        self.times = ["2024-05-10T00:00:00Z", "2024-05-10T02:00:00Z", "2024-05-10T04:00:00Z"]

    def test_exact_frame_contract_rejects_duplicates_naive_and_invented_times(self):
        from ophanim.dynamics.sequence import analyze_sequence
        for times in ([self.times[0], self.times[0]], ["2024-05-10T00:00:00"], ["NaT"]):
            with self.assertRaises(ValueError):
                normalize_frame_times(times)
        with self.assertRaisesRegex(ValueError, "exact eligible source epoch"):
            analyze_sequence(self.raw, self.config, ["2024-05-10T00:02:00Z"])

    def test_independent_wave_windows_withhold_early_fit_and_recover_later(self):
        from ophanim.dynamics.sequence import analyze_sequence
        from ophanim.dynamics.pipeline import prepare_dataset
        with patch("ophanim.dynamics.pipeline.prepare_dataset", wraps=prepare_dataset) as prepare:
            result = analyze_sequence(self.raw, self.config, self.times)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(result.timeline["frames"][0]["event"]["wave"]["status"], "insufficient_samples")
        self.assertEqual(result.timeline["frames"][-1]["event"]["wave"]["status"], "estimated")
        self.assertEqual(result.timeline["scientific_target_time"], self.times[-1])
        self.assertEqual(result.event["time"], self.times[-1])
        self.assertTrue(result.dataset.wave_confidence.isel(time=1).isnull())
        self.assertEqual(int(result.dataset.inference_scheduled.sum()), 3)

    def test_flow_is_populated_at_requested_epochs_not_frozen_short_window(self):
        import numpy as np
        from ophanim.dynamics import FlowConfig
        from ophanim.dynamics.sequence import analyze_sequence
        from ophanim.experiments import make_synthetic_dataset
        raw = make_synthetic_dataset("translating_gaussian", nx=20, ny=20, nt=25)
        config = replace(self.config, short_window_minutes=10,
                         flow=FlowConfig(enabled=True, num_iter=2, num_warp=1))
        times = ["2024-05-10T00:05:00Z", "2024-05-10T01:00:00Z", "2024-05-10T02:00:00Z"]
        result = analyze_sequence(raw, config, times)
        self.assertTrue(np.isfinite(result.dataset.flow_u.isel(time=1)).any())
        self.assertTrue(np.isfinite(result.dataset.flow_u.isel(time=12)).any())
        self.assertTrue(np.isnan(result.dataset.flow_u.isel(time=2)).all())
        for frame in result.timeline["frames"]:
            records = frame["event"]["flow_pairs"]
            self.assertEqual(len(records), 1)

    def test_requested_frames_and_scientific_target_are_separate(self):
        from ophanim.dynamics.sequence import analyze_sequence
        result = analyze_sequence(self.raw, self.config, self.times[:2])
        self.assertEqual(result.event["time"], self.times[-1])
        self.assertEqual([frame["time"] for frame in result.timeline["frames"]], self.times)
        with self.assertRaisesRegex(ValueError, "1 through 2"):
            analyze_sequence(self.raw, self.config, self.times[:2], budget=SequenceBudget(max_frames=2))

    def test_work_budget_rejects_before_flow_or_wave_fits(self):
        from ophanim.dynamics.sequence import analyze_sequence
        with patch("ophanim.dynamics.flow.add_flow") as flow:
            with self.assertRaisesRegex(ValueError, "max_work_cells"):
                analyze_sequence(self.raw, self.config, self.times, budget=SequenceBudget(max_work_cells=10))
            flow.assert_not_called()

    def test_causal_frames_match_independent_prefixes_and_ignore_future_append(self):
        import xarray as xr
        from ophanim.dynamics import analyze_dataset
        from ophanim.dynamics.sequence import analyze_sequence
        config = replace(self.config, analysis_mode="causal", as_of=self.times[1], target_time=self.times[1])
        requested = ["2024-05-10T01:00:00Z", self.times[1]]
        before = analyze_sequence(self.raw.isel(time=slice(0, 25)), config, requested)
        future = self.raw.copy(deep=True)
        future.tec.values[25:] += 500
        after = analyze_sequence(future, config, requested)
        xr.testing.assert_identical(before.dataset, after.dataset)
        self.assertEqual(before.timeline, after.timeline)
        for frame in before.timeline["frames"]:
            independent = analyze_dataset(self.raw, replace(config, target_time=frame["time"]))
            self.assertEqual(frame["event"], independent.event)

    def test_publication_round_trip_records_timeline_and_separate_id(self):
        from ophanim.core.runs import analyze_run
        from ophanim.core.state import read_state
        raw = self.raw.isel(x=slice(0, 8), y=slice(0, 8), time=slice(0, 7))
        times = ["2024-05-10T00:00:00Z", "2024-05-10T00:15:00Z", "2024-05-10T00:30:00Z"]
        with tempfile.TemporaryDirectory() as directory, patch("ophanim.science_reports.science_diagnostics"):
            single = analyze_run(raw, self.config, directory)
            sequence = analyze_run(raw, self.config, directory, frame_times=times)
            self.assertNotEqual(single, sequence)
            self.assertEqual(sequence, analyze_run(raw, self.config, directory, frame_times=times))
            timeline = json.loads((sequence / "events.json").read_text())
            self.assertEqual([frame["time"] for frame in timeline["frames"]], times)
            self.assertEqual(read_state(sequence).timeline, timeline)

    def test_cancellation_never_publishes_a_partial_state(self):
        from pathlib import Path
        from ophanim.core.runs import analyze_run
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                analyze_run(self.raw, self.config, directory, frame_times=self.times, cancelled=lambda: True)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
