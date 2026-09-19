"""Bounded imaginative evolution is not fabricated scientific evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import tempfile
import unittest

from ophanim.experiments.hypotheses import (HypothesisConfig, hypothesis_from_evidence,
                                           alternatives_from_evidence, validate_hypothesis)

NUMERICAL = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "xarray"))
ZARR = importlib.util.find_spec("zarr") is not None


def tiny_recipe(**overrides):
    return hypothesis_from_evidence(overrides={"extent_km": 200, "spacing_km": 25,
                "width_km": 60, "altitude_min_km": 200, "altitude_max_km": 320,
                "vertical_spacing_km": 20, "frame_count": 4, "duration_minutes": 6, **overrides})


class HypothesisRecipeTests(unittest.TestCase):
    def candidate(self):
        return {"candidate_id": "candidate-42", "peak_deviation_tecu": 5,
                "centroid": {"latitude": 42, "longitude": -70}, "peak_time": "2026-09-12T12:00:00Z",
                "hypotheses": [{"label": "Persistent regional enhancement"}],
                "evidence_strength": "limited", "source_kind": "native"}

    def test_defaults_are_frozen_bounded_and_no_ifm(self):
        config = HypothesisConfig()
        self.assertEqual(config.dimensions, (12, 25, 41, 41))
        with self.assertRaises(FrozenInstanceError):
            config.seed = 4
        recipe = hypothesis_from_evidence()
        self.assertFalse(recipe["ifm"]["used"])
        self.assertIn("no mechanism established", recipe["provenance"]["kind"]["reason"])
        validate_hypothesis(recipe)

    def test_withheld_science_allows_chosen_motion_without_evidence_mutation(self):
        candidate = self.candidate()
        event = {"time": candidate["peak_time"], "wave": {"status": "insufficient_samples", "confidence": 0},
                 "channel_status": {"flow": {"status": "unobservable"}}, "event_scores": {"uncertain": 1}}
        original = deepcopy((candidate, event))
        recipe = hypothesis_from_evidence(candidate, event)
        self.assertEqual((candidate, event), original)
        self.assertEqual(recipe["provenance"]["speed_m_s"]["status"], "chosen")
        self.assertEqual(recipe["provenance"]["center_latitude"]["status"], "estimated")
        self.assertGreater(recipe["parameters"]["speed_m_s"], 0)
        self.assertEqual(recipe["evidence"]["scientific_event"]["channel_status"]["flow"]["status"], "unobservable")
        self.assertEqual(recipe["evidence"]["candidate"]["hypotheses"], candidate["hypotheses"])

    def test_unambiguous_shape_proposal_and_explicit_override(self):
        event = {"event_scores": {"front": .8, "wave": .1, "uncertain": .2}}
        recipe = hypothesis_from_evidence(event=event)
        self.assertEqual(recipe["parameters"]["kind"], "front")
        self.assertEqual(recipe["provenance"]["kind"]["status"], "chosen")
        recipe = hypothesis_from_evidence(event=event, overrides={"kind": "sheared_jet"})
        self.assertEqual(recipe["parameters"]["kind"], "sheared_jet")
        ambiguous = self.candidate() | {"hypotheses": [{"category": "front_like"}, {"category": "wave_like"}]}
        self.assertIn("fallback", hypothesis_from_evidence(ambiguous)["provenance"]["kind"]["reason"])

    def test_supported_wave_seeds_estimates_but_not_measured_motion(self):
        event = {"wave": {"status": "estimated", "confidence": .85, "period_min": 50, "bearing_deg": 110}}
        recipe = hypothesis_from_evidence(event=event)
        self.assertEqual(recipe["parameters"]["period_minutes"], 50)
        self.assertEqual(recipe["provenance"]["period_minutes"]["status"], "estimated")
        self.assertEqual(recipe["provenance"]["speed_m_s"]["status"], "chosen")

    def test_alternatives_are_explicitly_different_reproducible_recipes(self):
        recipes = alternatives_from_evidence(self.candidate())
        self.assertEqual([r["parameters"]["kind"] for r in recipes], ["wave_packet", "front", "sheared_jet"])
        self.assertEqual(len({r["evidence_sha256"] for r in recipes}), 1)

    def test_strict_parameters_and_budgets(self):
        for values in ({"foo": 1}, {"speed_m_s": True}, {"seed": -1}, {"frame_count": 25},
                       {"width_km": float("nan")}, {"altitude_min_km": 500, "altitude_max_km": 200},
                       {"spacing_km": 5, "extent_km": 1600},
                       {"speed_m_s": 600, "duration_minutes": 240, "shear_per_s": .003}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                HypothesisConfig.from_mapping(values)

    def test_tampered_evidence_or_measured_parameter_is_rejected(self):
        recipe = hypothesis_from_evidence(self.candidate())
        recipe["evidence"]["candidate"]["peak_deviation_tecu"] = 100
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_hypothesis(recipe)
        recipe = hypothesis_from_evidence()
        recipe["provenance"]["speed_m_s"]["status"] = "observed"
        with self.assertRaisesRegex(ValueError, "never observed"):
            validate_hypothesis(recipe)


@unittest.skipUnless(NUMERICAL, "requires optional numerical dependencies")
class VolumeEvolutionTests(unittest.TestCase):
    def test_deterministic_genuinely_3d_and_time_varying(self):
        import numpy as np
        import xarray as xr
        from ophanim.experiments.volume_model import simulate_hypothesis
        for kind in ("wave_packet", "front", "sheared_jet"):
            with self.subTest(kind=kind):
                recipe = tiny_recipe(kind=kind)
                original = deepcopy(recipe)
                first = simulate_hypothesis(recipe)
                second = simulate_hypothesis(recipe)
                xr.testing.assert_identical(first.dataset, second.dataset)
                self.assertEqual(recipe, original)
                self.assertEqual(first.dataset.density.dims, ("time", "z", "y", "x"))
                values = first.dataset.density.values
                self.assertGreater(float(np.mean(np.abs(values[-1] - values[0]))), .001)
                for axis in (1, 2, 3):
                    self.assertGreater(float(np.mean(np.std(values, axis=axis))), .005)
                self.assertGreater(float(np.corrcoef(values[0].ravel(), values[1].ravel())[0, 1]), .7)
                self.assertGreaterEqual(float(values.min()), 0)
                self.assertEqual(first.dataset.attrs["source_kind"], "imagined")
                self.assertIn("not electron density", first.dataset.attrs["semantics"])

    def test_seed_changes_coherent_geometry(self):
        import numpy as np
        from ophanim.experiments.volume_model import simulate_hypothesis
        first = simulate_hypothesis(tiny_recipe(seed=1)).dataset.density.values
        second = simulate_hypothesis(tiny_recipe(seed=2)).dataset.density.values
        self.assertFalse(np.allclose(first, second))

    def test_cancellation_checked_before_and_during_integration(self):
        from ophanim.experiments.volume_model import simulate_hypothesis
        with self.assertRaises(InterruptedError):
            simulate_hypothesis(tiny_recipe(), cancelled=lambda: True)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(InterruptedError):
            simulate_hypothesis(tiny_recipe(), cancelled=cancel)
        self.assertEqual(calls, 3)

    def test_volume_contract_rejects_measured_or_wrong_units(self):
        from ophanim.experiments.volume_model import simulate_hypothesis
        volume = simulate_hypothesis(tiny_recipe())
        volume.dataset.density.attrs["units"] = "electrons/m^3"
        with self.assertRaisesRegex(ValueError, "dimensionless"):
            volume.validate()

    @unittest.skipUnless(ZARR, "requires optional Zarr")
    def test_immutable_publication_and_verified_read(self):
        import xarray as xr
        from ophanim.core.artifacts import verify_run
        from ophanim.core.volumes import read_volume
        from ophanim.experiments.volume_model import publish_hypothesis_run, simulate_hypothesis
        with tempfile.TemporaryDirectory() as temporary:
            recipe = tiny_recipe()
            run = publish_hypothesis_run(recipe, temporary)
            self.assertEqual(publish_hypothesis_run(recipe, temporary), run)
            self.assertEqual(verify_run(run)["kind"], "hypothesis")
            volume = read_volume(run)
            xr.testing.assert_identical(volume.dataset, simulate_hypothesis(recipe).dataset)
            with self.assertRaises(InterruptedError):
                publish_hypothesis_run(tiny_recipe(seed=4), temporary, cancelled=lambda: True)
            self.assertEqual([p for p in Path(temporary).iterdir() if p.is_dir()], [run])


if __name__ == "__main__":
    unittest.main()
