"""Persistence, optional imports and science-to-art boundary regression tests."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ophanim.science_runs import (canonical_json, file_sha256, json_value, verify_run,
                                  _publish, write_json)

SCIENCE_AVAILABLE = all(importlib.util.find_spec(name) is not None
                        for name in ("numpy", "xarray", "scipy", "skimage", "pyproj", "zarr", "PIL"))


class RunManifestTests(unittest.TestCase):
    def test_strict_json_represents_unknown_not_nan(self):
        self.assertEqual(canonical_json({"x": float("nan")}), b'{"x":null}')

    def test_atomic_complete_stage_is_idempotent_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = lambda stage: write_json(stage / "evidence.json", {"value": 3})
            run = _publish(root, "example", "test", {"input": "abc"}, build)
            self.assertEqual(verify_run(run, kind="test")["status"], "complete")
            again = _publish(root, "example", "test", {"input": "abc"},
                             lambda stage: self.fail("must reuse completed run"))
            self.assertEqual(run, again)
            (run / "evidence.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_run(run)

    def test_failure_never_publishes_complete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            def fail(stage):
                write_json(stage / "partial.json", {"partial": True})
                raise RuntimeError("deliberate interruption")
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                _publish(Path(temporary), "example", "test", {}, fail)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_identity_collision_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "same", "test", {"seed": 1},
                     lambda stage: write_json(stage / "a.json", {}))
            with self.assertRaisesRegex(ValueError, "identity"):
                _publish(root, "same", "test", {"seed": 2}, lambda _: None)

    def test_manifest_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _publish(Path(temporary), "example", "test", {},
                           lambda stage: write_json(stage / "a.json", {}))
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["files"] = {"../private": "0" * 64}
            write_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_run(run)

    def test_cli_help_without_running_science(self):
        result = subprocess.run([sys.executable, "-m", "ophanim.shawtynet_cli", "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("visualize", result.stdout)


@unittest.skipUnless(SCIENCE_AVAILABLE, "requires optional ophanim[shawtynet]")
class ScientificPipelineTests(unittest.TestCase):
    def test_round_trip_and_style_rerun_do_not_change_science(self):
        import xarray as xr
        from ophanim.dynamics import AnalysisConfig, FlowConfig
        from ophanim.experiments import make_synthetic_dataset
        from ophanim.science_runs import analyze_run, visualize_run
        from ophanim.shawtynet import VisualStyleConfig

        raw = make_synthetic_dataset(kind="quiet", nx=12, ny=12, nt=9)
        original = raw.copy(deep=True)
        config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                                flow=FlowConfig(enabled=False))
        style = VisualStyleConfig(render_spacing_km=25, lic_streamline_steps=3)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("ophanim.science_reports.science_diagnostics"), patch("ophanim.science_reports.visual_diagnostics"):
                run = analyze_run(raw, config, temporary)
                repeated = analyze_run(raw, config, temporary)
                self.assertEqual(run, repeated)
                manifest_digest = file_sha256(run / "manifest.json")
                visual = visualize_run(run, style)
                alternative = visualize_run(run, replace(style, emission_gain=3.0))
                self.assertNotEqual(visual, alternative)
                self.assertEqual(file_sha256(run / "manifest.json"), manifest_digest)
                verify_run(run, kind="science")
                verify_run(visual, kind="visual")
                with xr.open_zarr(run / "dynamic.zarr") as stored:
                    self.assertIn("dtec", stored)
                    self.assertEqual(stored.tec.attrs["units"], "TECU")
                with xr.open_zarr(visual / "visual.zarr") as stored_visual:
                    self.assertEqual(stored_visual.sizes["time"], 1)
                sequence = visualize_run(run, style, sequence=True)
                with xr.open_zarr(sequence / "visual.zarr") as stored_sequence:
                    self.assertEqual(stored_sequence.sizes["time"], raw.sizes["time"])
                packet = json.loads((run / "science_packet.json").read_text())
                self.assertEqual(packet["semantics"]["artistic"], [])
                self.assertIn("capabilities", packet)
        xr.testing.assert_identical(raw, original)

    def test_total_cube_budgets_fail_before_large_allocations(self):
        from ophanim.dynamics import AnalysisConfig, AnalysisError, FlowConfig, analyze_dataset
        from ophanim.experiments import make_synthetic_dataset
        from ophanim.shawtynet import VisualStyleConfig, map_visual_fields
        raw = make_synthetic_dataset(kind="quiet", nx=12, ny=12, nt=9)
        config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                                flow=FlowConfig(enabled=False))
        with self.assertRaisesRegex(AnalysisError, "max_cube_cells"):
            analyze_dataset(raw, replace(config, max_cube_cells=100))
        result = analyze_dataset(raw, config)
        with self.assertRaisesRegex(ValueError, "max_render_voxels"):
            map_visual_fields(result.dataset, result.event,
                              VisualStyleConfig(max_render_voxels=100))

    def test_science_modules_have_no_art_imports(self):
        import ast
        import ophanim
        root = Path(ophanim.__file__).parent
        for directory in (root / "dynamics", root / "experiments", root / "sensing"):
            for path in directory.rglob("*.py"):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith("ophanim.shawtynet"), str(path))
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(item.name.startswith("ophanim.shawtynet") for item in node.names), str(path))


if __name__ == "__main__":
    unittest.main()
