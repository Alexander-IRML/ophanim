"""Native candidate screening is useful without Mamba, training, or art."""

from importlib.util import find_spec
import json
import unittest

from ophanim.core.discovery import DiscoveryConfig, survey_dataset


HAS_ARRAYS = all(find_spec(name) is not None for name in ("numpy", "xarray", "scipy"))


def native_history(*, days=8, global_grid=False):
    import numpy as np
    import xarray as xr

    times = np.datetime64("2024-01-01T00:00:00") + np.arange(days * 4) * np.timedelta64(6, "h")
    latitude = np.arange(5) * 2.5 + 20
    longitude = np.arange(6) * (60 if global_grid else 5) - (180 if global_grid else 110)
    values = np.broadcast_to(20 + np.sin(np.arange(days * 4) % 4 * np.pi / 2)[:, None, None] * 3, (days * 4, 5, 6)).copy()
    return xr.Dataset({
        "tec": (("time", "lat", "lon"), values),
        "observed_mask": (("time", "lat", "lon"), np.ones_like(values, dtype=bool)),
        "source_rms_tecu": (("time", "lat", "lon"), np.ones_like(values)),
    }, coords={"time": times, "lat": latitude, "lon": longitude}, attrs={
        "source_kind": "native", "snapshot_id": "fixture-v1",
        "source_metadata": {"native_lat_spacing_deg": 2.5, "native_lon_spacing_deg": 60 if global_grid else 5},
    })


@unittest.skipUnless(HAS_ARRAYS, "optional scientific dependencies unavailable")
class DiscoveryTests(unittest.TestCase):
    def test_quiet_daily_cycle_is_not_an_anomaly(self):
        result = survey_dataset(native_history())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["baseline"]["supported_fraction"], 1)
        json.dumps(result, allow_nan=False)

    def test_two_regions_are_grouped_across_time_and_ranked(self):
        dataset = native_history()
        dataset.tec.values[-3:, 0, 0:2] += 12
        dataset.tec.values[-2:, 4, 4:6] -= 14
        result = survey_dataset(dataset)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual({c["polarity"] for c in result["candidates"]}, {"enhancement", "depletion"})
        first = next(c for c in result["candidates"] if c["polarity"] == "enhancement")
        self.assertEqual(first["native_cell_count"], 2)
        self.assertEqual(first["observed_epoch_count"], 3)
        self.assertEqual(first["duration_hours"], 12)
        self.assertEqual(first["peak_deviation_tecu"], 12)
        self.assertEqual(first["peak_score"], 6)
        self.assertEqual(len(first["candidate_id"]), 24)
        self.assertEqual(result, survey_dataset(dataset))
        changed = survey_dataset(dataset, {"score_threshold": 4})
        self.assertNotEqual(result["scan_id"], changed["scan_id"])

    def test_single_cells_or_single_epochs_are_not_events(self):
        dataset = native_history()
        dataset.tec.values[-3:, 0, 0] += 40
        dataset.tec.values[-1, 4, 4:6] -= 40
        self.assertEqual(survey_dataset(dataset)["candidates"], [])

    def test_missing_history_never_borrows_future_or_other_utc_times(self):
        dataset = native_history(days=3)
        result = survey_dataset(dataset)
        self.assertEqual(result["status"], "insufficient_baseline")
        dataset = native_history()
        dataset.observed_mask.values[:-4] = False
        self.assertEqual(survey_dataset(dataset)["status"], "insufficient_baseline")

    def test_missing_support_is_null_not_zero(self):
        dataset = native_history()
        dataset.observed_mask.values[-1, 0, 0] = False
        result = survey_dataset(dataset)
        self.assertIsNone(result["map"]["values"][0][0])
        self.assertIsNone(result["deviation_map"]["values"][0][0])
        dataset.observed_mask.values[:] = False
        self.assertEqual(survey_dataset(dataset)["status"], "no_usable_data")
        dataset = native_history()
        dataset.observed_mask.values[-4:] = False
        self.assertEqual(survey_dataset(dataset)["status"], "no_usable_data")

    def test_only_one_baseline_supported_epoch_is_insufficient(self):
        dataset = native_history()
        dataset.observed_mask.values[:-4] = False
        dataset.observed_mask.values[3:-4:4] = True
        result = survey_dataset(dataset)
        self.assertEqual(result["baseline"]["supported_epoch_count"], 1)
        self.assertEqual(result["status"], "insufficient_baseline")

    def test_dateline_cells_join_only_on_global_native_grid(self):
        dataset = native_history(global_grid=True)
        dataset.tec.values[-2:, 2, [0, -1]] += 12
        result = survey_dataset(dataset)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertTrue(result["candidates"][0]["bounds"]["crosses_dateline"])
        self.assertEqual(result["candidates"][0]["native_cell_count"], 2)
        dataset = native_history()
        dataset.tec.values[-2:, 2, [0, -1]] += 12
        self.assertEqual(survey_dataset(dataset)["candidates"], [])

    def test_actual_time_gap_breaks_a_track(self):
        import numpy as np
        dataset = native_history()
        dataset.tec.values[-4, 0, :2] += 12
        dataset.tec.values[-1, 0, :2] += 12
        dataset = dataset.isel(time=np.r_[0:29, 31])
        self.assertEqual(survey_dataset(dataset)["candidates"], [])

    def test_source_rms_is_a_scale_floor_not_probability(self):
        dataset = native_history()
        dataset.tec.values[-2:, 0, :2] += 12
        dataset.source_rms_tecu.values[-2:, 0, :2] = 10
        self.assertEqual(survey_dataset(dataset)["candidates"], [])

    def test_identity_changes_with_input_not_only_source_labels(self):
        dataset = native_history()
        before = survey_dataset(dataset)
        dataset.tec.values[-1, 0, 0] += 1
        self.assertNotEqual(before["scan_id"], survey_dataset(dataset)["scan_id"])

    def test_rejects_interpolated_axes_duplicates_and_resource_overruns(self):
        import numpy as np
        dataset = native_history()
        dataset.attrs["source_kind"] = "derived"
        with self.assertRaisesRegex(ValueError, "native"):
            survey_dataset(dataset)
        dataset = native_history()
        dataset.attrs["source_metadata"]["native_lon_spacing_deg"] = 10
        with self.assertRaisesRegex(ValueError, "finer"):
            survey_dataset(dataset)
        dataset = native_history()
        with self.assertRaisesRegex(ValueError, "memory budget"):
            survey_dataset(dataset, {"maximum_cube_cells": 3})
        dataset = dataset.isel(time=np.r_[0, 0, 1:32])
        with self.assertRaisesRegex(ValueError, "unique"):
            survey_dataset(dataset)

    def test_no_more_than_three_spatially_distinct_results(self):
        dataset = native_history()
        for row, col in ((0, 0), (0, 4), (4, 0), (4, 4)):
            dataset.tec.values[-2:, row, col:col + 2] += 12
        self.assertEqual(len(survey_dataset(dataset)["candidates"]), 3)

    def test_configuration_cannot_disable_minimum_evidence(self):
        for settings in ({"minimum_epochs": 1}, {"minimum_cells": 1}, {"max_candidates": 4},
                         {"scan_hours": float("nan")}, {"baseline_days": True}):
            with self.assertRaises(ValueError):
                DiscoveryConfig(**settings)


if __name__ == "__main__":
    unittest.main()
