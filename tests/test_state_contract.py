"""Versioned scientific-state adapters do not invent measurements or semantics."""
from dataclasses import replace
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

from ophanim.core.models import ArtifactReference, ModelPrediction, ModelAdapter, EngineeredModelAdapter
from ophanim.core.state import (STATE_SCHEMA, REQUIRED_FIELDS, DynamicState, FieldMapping,
    EngineeredStateAdapter, LearnedStateAdapter, to_mapping_dataset, read_state)

SCIENCE = all(importlib.util.find_spec(name) is not None
              for name in ("numpy", "xarray", "scipy", "skimage", "pyproj", "zarr"))


class StateSchemaTests(unittest.TestCase):
    def test_mapping_requires_scientific_semantics_and_units(self):
        for changes in ({"unit": ""}, {"semantic_class": "artistic"}, {"description": ""}):
            with self.assertRaises(ValueError):
                FieldMapping(**({"source": "x", "unit": "1", "semantic_class": "inferred",
                                 "description": "decoded signal"} | changes))

    def test_model_prediction_rejects_unstructured_latent_output(self):
        artifact = ArtifactReference("dataset", "a" * 64)
        with self.assertRaisesRegex(ValueError, "canonical DynamicState"):
            ModelPrediction("model", "1", (artifact,), [1, 2, 3])


@unittest.skipUnless(SCIENCE, "requires optional science environment")
class StateContractTests(unittest.TestCase):
    def setUp(self):
        from ophanim.dynamics import AnalysisConfig, FlowConfig, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        self.raw = make_synthetic_dataset("quiet", nx=8, ny=8, nt=7)
        self.config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                                    flow=FlowConfig(enabled=False), smooth_sigma_km=0)
        self.result = analyze_dataset(self.raw, self.config)
        self.state = EngineeredStateAdapter().from_result(self.result)

    def test_engineered_adapter_has_units_confidence_and_no_false_observations(self):
        import xarray as xr
        self.assertEqual(self.state.dataset.attrs["state_schema"], STATE_SCHEMA)
        self.assertTrue(set(REQUIRED_FIELDS) <= set(self.state.dataset))
        self.assertEqual(self.state.dataset.signal.attrs["units"], "TECU")
        self.assertEqual(self.state.dataset.signal.attrs["semantic_class"], "synthetic")
        self.assertEqual(self.state.dataset.time.attrs["timezone"], "UTC")
        for name in ("source_uncertainty", "source_uncertainty_known", "measurement_reliability",
                     "growth_ambiguity", "advection_residual_confidence"):
            self.assertIn(name, self.state.dataset)
        original = self.result.dataset.copy(deep=True)
        compatible = to_mapping_dataset(self.state)
        self.assertIn("dtec", compatible)
        self.assertEqual(compatible.tec.attrs["semantic_class"], "synthetic")
        xr.testing.assert_identical(original, self.result.dataset)

    def test_validator_rejects_missing_units_bad_confidence_and_semantics(self):
        for field, attribute, value in (("signal", "units", ""),
                                         ("flow_confidence", "units", "m/s"),
                                         ("signal", "semantic_class", "measured")):
            data = self.state.dataset.copy(deep=True)
            data[field].attrs[attribute] = value
            with self.assertRaises(ValueError):
                replace(self.state, dataset=data).validate()
        data = self.state.dataset.copy(deep=True)
        data.flow_confidence.values[:] = 1.01
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            replace(self.state, dataset=data).validate()

    def test_validator_rejects_unversioned_or_nonmetric_state(self):
        for target, key, value in (("dataset", "state_schema", "future/99"),
                                    ("x", "units", "degrees"), ("time", "timezone", "local")):
            data = self.state.dataset.copy(deep=True)
            (data.attrs if target == "dataset" else data[target].attrs)[key] = value
            with self.assertRaises(ValueError):
                replace(self.state, dataset=data).validate()

    def test_missing_source_kind_requires_explicit_provenance(self):
        data = self.result.dataset.copy(deep=True)
        data.attrs.pop("source_kind", None)
        with self.assertRaisesRegex(ValueError, "attach verified source metadata"):
            EngineeredStateAdapter().from_result(replace(self.result, dataset=data))

    def test_motion_presentation_eligibility_is_preserved_separately_from_fit(self):
        import numpy as np
        data = self.result.dataset.copy(deep=True)
        data.flow_confidence.values[:] = .8
        data["flow_interpretation_confidence"] = (("time", "y", "x"),
            np.full(data.tec.shape, .2), {"units": "1", "semantic_class": "inferred",
                "semantic": "Conservative motion presentation eligibility; not calibrated accuracy"})
        state = EngineeredStateAdapter().from_result(replace(self.result, dataset=data))
        handoff = to_mapping_dataset(state)
        np.testing.assert_array_equal(handoff.flow_confidence, .8)
        np.testing.assert_array_equal(handoff.flow_interpretation_confidence, .2)
        self.assertEqual(handoff.flow_interpretation_confidence.attrs["semantic_class"], "inferred")
        for invalid in (-.01, 1.01):
            changed = state.dataset.copy(deep=True)
            changed.flow_interpretation_confidence.values[:] = invalid
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                replace(state, dataset=changed).validate()
        changed = state.dataset.copy(deep=True)
        changed.flow_interpretation_confidence.attrs["semantic_class"] = "measured"
        with self.assertRaises(ValueError):
            replace(state, dataset=changed).validate()

    def learned(self, **overrides):
        decoded = self.state.dataset.copy(deep=True)
        # A producer explicitly decodes a dimensionless research signal. It is
        # not silently named/measured as TEC by the canonical state boundary.
        decoded.signal.attrs["units"] = "research_unit"
        decoded.anomaly.attrs["units"] = "research_unit"
        fields = {name: FieldMapping(name, decoded[name].attrs["units"], "inferred",
                                    "Explicit decoded research output; not a measurement")
                  for name in REQUIRED_FIELDS}
        arguments = dict(fields=fields, model_artifact=ArtifactReference("checkpoint", "b" * 64, "model"),
                         input_artifacts=(ArtifactReference("dataset", "a" * 64),), model_id="test-decoder",
                         event=self.state.event)
        arguments.update(overrides)
        return LearnedStateAdapter().adapt(decoded, **arguments)

    def test_explicit_learned_adapter_is_generic_and_provenance_linked(self):
        state = self.learned()
        self.assertEqual(state.dataset.attrs["source_kind"], "learned")
        self.assertEqual(state.dataset.signal.attrs["units"], "research_unit")
        self.assertEqual(to_mapping_dataset(state).tec.attrs["units"], "research_unit")
        self.assertEqual(state.provenance["model_artifact"]["sha256"], "b" * 64)
        prediction = ModelPrediction("test-decoder", "1", (ArtifactReference("dataset", "a" * 64),),
                                     state, ArtifactReference("checkpoint", "b" * 64, "model"))
        self.assertIs(prediction.output, state)

    def test_concrete_engineered_model_implements_shared_protocol(self):
        adapter = EngineeredModelAdapter(self.config)
        self.assertIsInstance(adapter, ModelAdapter)
        self.assertTrue(adapter.availability().available)
        prediction = adapter.predict(self.raw, input_artifacts=(ArtifactReference("fixture", "a" * 64),))
        self.assertEqual(prediction.output.dataset.attrs["state_schema"], STATE_SCHEMA)
        self.assertIsNone(prediction.model_artifact)
        with self.assertRaisesRegex(ValueError, "no learned checkpoint"):
            adapter.predict(self.raw, input_artifacts=(ArtifactReference("fixture", "a" * 64),),
                            model_artifact=ArtifactReference("pretend-model", "b" * 64))

    def test_learned_adapter_requires_real_refs_and_explicit_mapping(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.learned(model_artifact=None)
        with self.assertRaisesRegex(ValueError, "required fields"):
            self.learned(fields={})
        fields = {name: FieldMapping(name, "research_unit" if name in {"signal", "anomaly"} else "1",
                                    "inferred", "Explicit decoded research output") for name in REQUIRED_FIELDS}
        with self.assertRaisesRegex(ValueError, "never measured"):
            self.learned(fields=fields | {"signal": FieldMapping("signal", "research_unit", "measured", "wrong")})
        with self.assertRaisesRegex(ValueError, "units disagree"):
            self.learned(fields=fields | {"signal": FieldMapping("signal", "K", "inferred", "wrong")})

    def test_published_synthetic_packet_never_claims_gnss_measurements(self):
        from ophanim.core.runs import analyze_run
        with tempfile.TemporaryDirectory() as directory, patch("ophanim.science_reports.science_diagnostics"):
            run = analyze_run(self.raw, self.config, directory)
            packet = json.loads((run / "science_packet.json").read_text())
            self.assertEqual(packet["semantics"]["measured"], [])
            self.assertTrue(packet["semantics"]["synthetic"])
            self.assertEqual(packet["fields"]["tec"]["semantic_class"], "synthetic")
            self.assertEqual(read_state(run).dataset.attrs["source_kind"], "synthetic")
            contract = json.loads((run / "state_contract.json").read_text())
            self.assertEqual(contract["schema_version"], STATE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
