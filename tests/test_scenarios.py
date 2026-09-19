"""Observed-candidate to explicitly hypothetical scenario regression tests."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
import unittest

from ophanim.experiments.scenarios import ScenarioConfig, make_scenario, scenario_from_candidate

ARRAYS_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "xarray", "scipy", "pyproj"))


class ScenarioRecipeTests(unittest.TestCase):
    def candidate(self):
        return {"candidate_id": "native-event-1", "peak_deviation_tecu": -4.5,
                "centroid": {"latitude": 42.5, "longitude": -70.0},
                "peak_time": "2026-09-12T18:00:00Z"}

    def test_defaults_are_bounded_and_immutable(self):
        config = ScenarioConfig()
        nx, nt = config.dimensions
        self.assertLessEqual(nx * nx * nt, 200_000)
        with self.assertRaises(FrozenInstanceError):
            config.seed = 2

    def test_candidate_recipe_preserves_input_and_honest_provenance(self):
        candidate = self.candidate()
        original = deepcopy(candidate)
        recipe = scenario_from_candidate(candidate)
        self.assertEqual(candidate, original)
        self.assertEqual(recipe["parameters"]["kind"], "localized_depletion")
        self.assertEqual(recipe["parameters"]["amplitude_tecu"], 4.5)
        self.assertEqual(recipe["parameters"]["center_latitude"], 42.5)
        self.assertEqual(recipe["provenance"]["amplitude_tecu"]["status"], "estimated")
        self.assertEqual(recipe["provenance"]["period_minutes"]["status"], "chosen")
        self.assertEqual(recipe["candidate_reference"]["candidate_id"], "native-event-1")
        self.assertIn("not a reconstruction", recipe["semantics"])

    def test_override_becomes_chosen_and_free_experiment_needs_no_candidate(self):
        recipe = scenario_from_candidate(self.candidate(), {"amplitude_tecu": 2})
        self.assertEqual(recipe["provenance"]["amplitude_tecu"]["status"], "chosen")
        independent = scenario_from_candidate(None, {"kind": "quiet"})
        self.assertIsNone(independent["candidate_reference"])

    def test_unknown_nonfinite_and_bool_parameters_rejected(self):
        for payload in ({"amplitude": 4}, {"amplitude_tecu": float("nan")}, {"speed_m_s": True},
                        {"seed": True}, {"seed": -1}, {"bearing_deg": 400}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ScenarioConfig.from_mapping(payload)

    def test_resource_budget_rejected_before_optional_imports(self):
        with self.assertRaisesRegex(ValueError, "200,000"):
            ScenarioConfig(extent_km=2000, spacing_km=2, duration_minutes=720)

    def test_bad_candidate_and_depletion_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            scenario_from_candidate({"peak_deviation_tecu": 4})
        with self.assertRaisesRegex(ValueError, "depletion"):
            ScenarioConfig(kind="localized_depletion", amplitude_tecu=30)

    def test_polar_candidate_is_not_silently_moved_to_default_location(self):
        candidate = self.candidate() | {"centroid": {"latitude": 82.5, "longitude": 10}}
        self.assertEqual(scenario_from_candidate(candidate)["parameters"]["center_latitude"], 82.5)
        with self.assertRaisesRegex(ValueError, "centroid"):
            scenario_from_candidate(candidate | {"centroid": {"latitude": 100, "longitude": 10}})


@unittest.skipUnless(ARRAYS_AVAILABLE, "requires optional numerical dependencies")
class ScenarioGenerationTests(unittest.TestCase):
    def recipe(self, **parameters):
        return scenario_from_candidate(None, {"extent_km": 200, "spacing_km": 25,
                                              "width_km": 50, "duration_minutes": 20, **parameters})

    def test_generation_is_reproducible_and_remains_synthetic(self):
        import xarray as xr
        recipe = self.recipe()
        original = deepcopy(recipe)
        first, second = make_scenario(recipe), make_scenario(recipe)
        xr.testing.assert_identical(first, second)
        self.assertEqual(recipe, original)
        self.assertEqual(first.attrs["source_kind"], "synthetic")
        self.assertIn("not a reconstruction", first.attrs["scientific_scope"])
        self.assertEqual(first.attrs["scenario"]["parameters"]["seed"], 42)

    def test_chosen_location_reaches_core_projection(self):
        from ophanim.dynamics.coords import project_dataset
        from ophanim.dynamics import AnalysisConfig
        raw = make_scenario(self.recipe(center_latitude=42.5, center_longitude=-70))
        projected = project_dataset(raw, AnalysisConfig())
        self.assertAlmostEqual(float(projected.lat[4, 4]), 42.5, places=5)
        self.assertAlmostEqual(float(projected.lon[4, 4]), -70, places=5)

    def test_northward_front_is_not_accidentally_stationary(self):
        import numpy as np
        data = make_scenario(self.recipe(kind="moving_front", bearing_deg=0, speed_m_s=100))
        self.assertFalse(np.allclose(data.tec[0], data.tec[-1]))

    def test_depletion_is_below_background_and_not_negative_tec(self):
        data = make_scenario(self.recipe(kind="localized_depletion", amplitude_tecu=4))
        self.assertLess(float(data.tec.min()), 20)
        self.assertGreaterEqual(float(data.tec.min()), 0)
        self.assertLessEqual(float(data.tec.max()), 20)

    def test_bad_recipe_provenance_or_fixed_semantics_rejected(self):
        for key, value in (("provenance", {}), ("fixed_parameters", {"background_tecu": 100}),
                           ("semantics", "An observed reconstruction")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_scenario(self.recipe() | {key: value})


if __name__ == "__main__":
    unittest.main()
