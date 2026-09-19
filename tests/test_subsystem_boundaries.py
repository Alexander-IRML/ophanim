"""Executable dependency direction and future experimentation-contract checks."""

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import ophanim
from ophanim.core.artifacts import software_identity
from ophanim.core.models import ArtifactReference
from ophanim.experiments.contracts import (DatasetSplit, ExperimentSpec, MetricResult,
                                           agent_training_capability)


class DependencyBoundaryTests(unittest.TestCase):
    def test_subsystem_dependency_directions(self):
        root = Path(ophanim.__file__).parent
        forbidden = {
            "core": ("ophanim.experiments", "ophanim.shawtynet", "ophanim.science_runs",
                     "ophanim.desktop", "ophanim.workspace"),
            "experiments": ("ophanim.shawtynet", "ophanim.science_runs", "ophanim.desktop"),
            "shawtynet": ("ophanim.experiments", "ophanim.desktop", "ophanim.workspace"),
        }
        for section, prefixes in forbidden.items():
            for path in (root / section).rglob("*.py"):
                package = "ophanim." + ".".join(path.relative_to(root).parts[:-1])
                tree = ast.parse(path.read_text())
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(item.name for item in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if node.level:
                            module = importlib.util.resolve_name("." * node.level + module, package)
                        imports.append(module)
                        imports.extend(module + "." + item.name for item in node.names)
                    elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                        if name in {"__import__", "import_module"} and isinstance(node.args[0].value, str):
                            imports.append(node.args[0].value)
                for module in imports:
                    self.assertFalse(module.startswith(prefixes), f"{path}: forbidden dependency {module}")

    def test_imports_without_optional_dependencies_or_art_loading(self):
        source = Path(ophanim.__file__).parent.parent
        environment = {**os.environ, "PYTHONPATH": str(source)}
        program = ("import sys; import ophanim.core; import ophanim.experiments; "
                   "import ophanim.science_runs; "
                   "assert not any(m.startswith(('ophanim.shawtynet', 'numpy', 'scipy', 'torch', 'xarray')) "
                   "for m in sys.modules)")
        result = subprocess.run([sys.executable, "-S", "-c", program], capture_output=True,
                                text=True, env=environment, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_scientific_api_points_to_core(self):
        from ophanim.core.runs import analyze_run, _write_dataset
        from ophanim import science_runs
        self.assertIs(science_runs.analyze_run, analyze_run)
        self.assertIs(science_runs._write_dataset, _write_dataset)

    def test_core_hash_ignores_changes_to_experiment_code(self):
        read = Path.read_bytes
        baseline = software_identity()["code_sha256"]
        experiment = software_identity(extra_sections=("experiments",))["code_sha256"]

        def changed(path):
            content = read(path)
            return content + b"\n# hypothetical experiment change" if "experiments" in path.parts else content

        with patch.object(Path, "read_bytes", changed):
            self.assertEqual(software_identity()["code_sha256"], baseline)
            self.assertNotEqual(software_identity(extra_sections=("experiments",))["code_sha256"], experiment)


class ExperimentContractTests(unittest.TestCase):
    def split(self, role, start, end, reference=None):
        origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return DatasetSplit(role, origin + timedelta(days=start), origin + timedelta(days=end),
                            (ArtifactReference(reference or role, sha256((reference or role).encode()).hexdigest()),))

    def spec(self, **changes):
        values = dict(experiment_id="benchmark-1", model_id="novel-baseline", seed=42,
                      train=self.split("train", 0, 10), validation=self.split("validation", 10, 12),
                      test=self.split("test", 12, 15))
        return ExperimentSpec(**(values | changes))

    def test_chronological_split_contract_is_immutable(self):
        spec = self.spec()
        with self.assertRaises(FrozenInstanceError):
            spec.seed = 5
        self.assertEqual(spec.train.end_at, spec.validation.start_at)

    def test_overlapping_or_reordered_splits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            self.spec(validation=self.split("validation", 9, 12))
        with self.assertRaisesRegex(ValueError, "chronological"):
            self.spec(test=self.split("test", 1, 2))

    def test_duplicate_partition_references_cannot_leak_across_splits(self):
        with self.assertRaisesRegex(ValueError, "reuse"):
            self.spec(test=self.split("test", 12, 15, reference="train"))

    def test_same_content_under_new_reference_cannot_leak_across_splits(self):
        test = self.split("test", 12, 15)
        duplicate = ArtifactReference("test-renamed", self.split("train", 0, 10).artifacts[0].sha256)
        with self.assertRaisesRegex(ValueError, "identical partition content"):
            self.spec(test=DatasetSplit("test", test.start_at, test.end_at, (duplicate,)))

    def test_naive_timestamps_and_bad_artifacts_rejected(self):
        with self.assertRaisesRegex(ValueError, "aware"):
            DatasetSplit("train", datetime(2020, 1, 1), datetime(2020, 1, 2),
                         (ArtifactReference("input", "a" * 64),))
        with self.assertRaisesRegex(ValueError, "sha256"):
            ArtifactReference("input", "unknown")

    def test_agent_training_is_explicitly_not_configured(self):
        capability = agent_training_capability()
        self.assertFalse(capability.available)
        self.assertIn("contracts only", capability.reason)

    def test_metrics_reject_nonfinite_and_unknown_splits(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            MetricResult("mae", float("nan"), "TECU")
        with self.assertRaisesRegex(ValueError, "split"):
            MetricResult("mae", 2.0, "TECU", "all")


if __name__ == "__main__":
    unittest.main()
