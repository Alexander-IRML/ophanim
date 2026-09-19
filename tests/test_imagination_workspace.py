"""Workspace contracts for evidence-linked imagination, without remote models."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ophanim.workspace import Workspace, WorkspaceUnavailable

NUMERICAL = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "xarray", "zarr", "PIL", "skimage"))


def saved_candidate(workspace, identity="a" * 24):
    candidate = {"candidate_id": identity, "title": "Persistent regional enhancement",
                 "peak_deviation_tecu": 4.5, "centroid": {"latitude": 30, "longitude": -98},
                 "peak_time": "2026-09-12T12:00:00Z", "native_cell_count": 3,
                 "hypotheses": [{"label": "Ordinary variability or imperfect short baseline"}],
                 "evidence_strength": "limited"}
    workspace._save("scan", {"status": "complete", "candidates": [candidate],
                             "snapshot_id": "b" * 24, "source": {"source_kind": "native"}})
    return workspace.candidate(identity)


def tiny_parameters(**changes):
    return {"kind": "wave_packet", "extent_km": 200, "spacing_km": 25,
            "width_km": 60, "altitude_min_km": 200, "altitude_max_km": 320,
            "vertical_spacing_km": 20, "frame_count": 4, "duration_minutes": 6, **changes}


class ImaginationValidationTests(unittest.TestCase):
    def test_capabilities_are_local_and_ifm_is_only_planned(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                capability = workspace.status()["capabilities"]
                self.assertTrue(capability["imagined_3d"])
                self.assertEqual(capability["ifm"], {"status": "planned", "milestone": "v0.3"})
                self.assertFalse(capability["remote_simulation"]["automatic_uploads"])
                self.assertEqual(capability["remote_simulation"]["status"], "not_configured")
                request = workspace._validate("imagine", {})
                self.assertEqual(request["style"], "luminous")
                self.assertIsNone(request["camera"])
                self.assertIsNone(workspace.status()["job"])
            finally:
                workspace.close()

    def test_missing_dependencies_disable_imagination_before_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: False)
            try:
                self.assertFalse(workspace.status()["capabilities"]["imagined_3d"])
                with self.assertRaises(WorkspaceUnavailable):
                    workspace.submit("imagine", {})
                self.assertIsNone(workspace.status()["job"])
            finally:
                workspace.close()

    def test_only_same_candidate_observation_project_can_supply_science(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                candidate = saved_candidate(workspace)
                base = {"candidate_id": candidate["candidate_id"], "kind": "observations",
                        "science_directory": str(workspace.root / "science" / ("c" * 24))}
                valid = workspace._save("project", base)
                request = {"candidate_id": candidate["candidate_id"], "source_project_id": valid["project_id"]}
                self.assertEqual(workspace._validate("imagine", request)["source_project_id"], valid["project_id"])
                for changes in ({"kind": "hypothetical"}, {"kind": "imagined_3d"},
                                {"candidate_id": "f" * 24}, {"science_directory": None}):
                    invalid = workspace._save("project", base | changes)
                    with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "same candidate"):
                        workspace._validate("imagine", request | {"source_project_id": invalid["project_id"]})
                with self.assertRaisesRegex(ValueError, "same candidate"):
                    workspace._validate("imagine", {"source_project_id": valid["project_id"]})
            finally:
                workspace.close()

    def test_orbit_controls_allowed_but_ground_camera_and_remote_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                camera = {"heading_deg": 50, "pitch_deg": 30, "roll_deg": 5, "horizontal_fov_deg": 60}
                self.assertEqual(workspace._validate("imagine", {"camera": camera})["camera"], camera)
                project = workspace._save("project", {"kind": "imagined_3d"})
                for kind, request in (("imagine", {"camera": {"observer_latitude": 30}}),
                                      ("style", {"project_id": project["project_id"], "camera": {"observer_altitude_km": .2}}),
                                      ("imagine", {"url": "https://example.org/model"}),
                                      ("imagine", {"volume_directory": "/etc"}),
                                      ("imagine", {"candidate_id": "../../private"}),
                                      ("imagine", {"source_project_id": "https://example.org"}),
                                      ("imagine", {"parameters": {"backend": "remote"}}),
                                      ("imagine", {"parameters": {"frame_count": True}})):
                    with self.subTest(kind=kind, request=request), self.assertRaises(ValueError):
                        workspace._validate(kind, request)
                self.assertIsNone(workspace.status()["job"])
            finally:
                workspace.close()

    def test_imagined_only_project_never_advertises_or_serves_science_report(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                project = workspace._save("project", {"kind": "imagined_3d", "volume_directory": "volume", "visual_directory": "visual"})
                public = workspace._public_project(project)
                self.assertNotIn("science_report_url", public)
                self.assertNotIn("volume_directory", public)
                with self.assertRaisesRegex(ValueError, "no scientific reconstruction"):
                    workspace.asset(project["project_id"] + "/science/report.html")
                for asset in (project["project_id"] + "/../volume.zarr", project["project_id"] + "/volume.zarr", "https://example.org/recipe.json"):
                    with self.assertRaises(ValueError):
                        workspace.asset(asset)
            finally:
                workspace.close()

    def test_source_science_path_cannot_escape_workspace_before_any_read(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                candidate = saved_candidate(workspace)
                source = workspace._save("project", {"kind": "observations", "candidate_id": candidate["candidate_id"],
                    "science_directory": str(Path(directory).parent / "outside-this-workspace")})
                request = workspace._validate("imagine", {"candidate_id": candidate["candidate_id"],
                                                          "source_project_id": source["project_id"]})
                with patch("ophanim.core.artifacts.verify_run") as verify:
                    with self.assertRaisesRegex(ValueError, "outside this workspace"):
                        workspace._imagine(request, lambda *args: None)
                    verify.assert_not_called()
                self.assertEqual(workspace.status()["scenarios"], [])
            finally:
                workspace.close()


@unittest.skipUnless(NUMERICAL, "requires optional science and rendering dependencies")
class ImaginationWorkspacePipelineTests(unittest.TestCase):
    def test_real_tiny_imagination_exports_recipe_volume_preview_and_history(self):
        from ophanim.core.artifacts import verify_run, dataset_sha256
        from ophanim.core.volumes import read_volume
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                candidate = saved_candidate(workspace)
                before = deepcopy(candidate)
                request = workspace._validate("imagine", {"candidate_id": candidate["candidate_id"], "parameters": tiny_parameters()})
                with patch("ophanim.core.runs.analyze_run") as scientific:
                    workspace._run("imagine", request, lambda *args: None)
                    scientific.assert_not_called()
                project = workspace.status()["projects"][0]
                private = workspace.record(project["project_id"], "project")
                volume = read_volume(private["volume_directory"])
                checksum = dataset_sha256(volume.dataset)
                self.assertEqual(project["kind"], "imagined_3d")
                self.assertEqual(project["candidate_id"], candidate["candidate_id"])
                self.assertEqual(workspace.candidate(candidate["candidate_id"]), before)
                self.assertNotIn("science_report_url", project)
                self.assertEqual(project["ifm"]["used"], False)
                self.assertEqual(project["provenance"]["speed_m_s"]["status"], "chosen")
                self.assertEqual(verify_run(private["visual_directory"])["kind"], "visual")
                for key in ("preview_url", "visual_report_url", "export_url", "recipe_url"):
                    content, mime = workspace.asset(project[key].removeprefix("/workspace-assets/"))
                    self.assertGreater(len(content), 10)
                    if key == "recipe_url":
                        recipe = json.loads(content)
                        self.assertEqual(recipe["schema"], "ophanim-hypothesis/1")
                        self.assertEqual(recipe["evidence_sha256"], project["evidence_sha256"])
                        self.assertEqual(recipe["evidence"]["candidate"]["source_kind"], "native")
                    if key == "export_url":
                        with zipfile.ZipFile(io.BytesIO(content)) as archive:
                            self.assertIn("scenario-recipe.json", archive.namelist())
                            self.assertIn("project.json", archive.namelist())
                            self.assertIn("render_package/scene_metadata.json", archive.namelist())
                            # Actual browser bundle must remain replayable under
                            # the standalone renderer's strict file inventory.
                            from ophanim.shawtynet.volume import _verify_volume_package
                            with tempfile.TemporaryDirectory() as exported:
                                archive.extractall(exported)
                                _verify_volume_package(Path(exported) / "render_package")
                style = workspace._validate("style", {"project_id": project["project_id"], "style": "quiet",
                                                      "camera": {"heading_deg": 70, "pitch_deg": 40}})
                workspace._run("style", style, lambda *args: None)
                child = workspace.status()["projects"][0]
                self.assertNotEqual(child["project_id"], project["project_id"])
                self.assertEqual(child["parent_project_id"], project["project_id"])
                self.assertEqual(child["scenario_id"], project["scenario_id"])
                self.assertEqual(child["active_view"], "preview")
                self.assertEqual(checksum, dataset_sha256(read_volume(private["volume_directory"]).dataset))
                old_id, child_id = project["project_id"], child["project_id"]
            finally:
                workspace.close()
            restored = Workspace(directory, availability=lambda: True)
            try:
                self.assertEqual(restored.status()["projects"][0]["project_id"], child_id)
                self.assertEqual(restored.record(old_id, "project")["style"], "luminous")
            finally:
                restored.close()

    def test_supported_parent_event_is_frozen_but_no_new_science_is_invented(self):
        from ophanim.core.artifacts import _publish, write_json
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                candidate = saved_candidate(workspace)
                event = {"time": candidate["peak_time"], "wave": {"status": "unavailable", "confidence": 0},
                         "channel_status": {"flow": {"status": "unobservable"}},
                         "event_scores": {"front": .8, "uncertain": .1}}
                science = _publish(workspace.root / "science", "d" * 24, "science", {"fixture": True},
                                   lambda stage: write_json(stage / "event.json", event))
                source = workspace._save("project", {"kind": "observations", "candidate_id": candidate["candidate_id"],
                                                       "science_directory": str(science)})
                request = workspace._validate("imagine", {"candidate_id": candidate["candidate_id"],
                    "source_project_id": source["project_id"], "parameters": {k: v for k, v in tiny_parameters().items() if k != "kind"}})
                captured = {}
                def capture(volume, request, progress, metadata):
                    captured.update(metadata)
                with patch.object(workspace, "_volume_project", side_effect=capture):
                    workspace._imagine(request, lambda *args: None)
                self.assertEqual(captured["parameters"]["kind"], "front")
                self.assertEqual(captured["source_project_id"], source["project_id"])
                self.assertEqual(captured["science_directory"], str(science))
                recipe = workspace.record(captured["scenario_id"], "scenario")
                self.assertEqual(recipe["evidence"]["scientific_event"]["channel_status"]["flow"]["status"], "unobservable")
                self.assertEqual(recipe["provenance"]["speed_m_s"]["status"], "chosen")
                self.assertEqual(json.loads((science / "event.json").read_text()), event)
            finally:
                workspace.close()

    def test_render_and_animation_dispatch_to_volume_not_science_path(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda: True)
            try:
                original = workspace._save("project", {"kind": "imagined_3d", "style": "luminous",
                    "volume_directory": str(workspace.root / "volume"), "visual_directory": str(workspace.root / "visual"),
                    "composite_directory": str(workspace.root / "old-photo"), "active_view": "composite"})
                with patch("ophanim.shawtynet.volume.render_volume_run", return_value=workspace.root / "render") as render, \
                     patch("ophanim.shawtynet.runs.render_run") as science_render:
                    workspace._run("render", {"project_id": original["project_id"]}, lambda *args: None)
                    render.assert_called_once()
                    science_render.assert_not_called()
                rendered = workspace.status()["projects"][0]
                self.assertEqual(rendered["active_view"], "render")
                with patch("ophanim.shawtynet.volume.animation_volume_run", return_value=workspace.root / "animation") as animate, \
                     patch("ophanim.shawtynet.studio.animation_run") as science_animation, \
                     patch.object(workspace, "_visual_settings", side_effect=AssertionError("imagined state cannot use science mapper")):
                    workspace._run("animation", {"project_id": rendered["project_id"], "frame_count": 3}, lambda *args: None)
                    animate.assert_called_once()
                    self.assertEqual(animate.call_args.kwargs["frame_count"], 3)
                    science_animation.assert_not_called()
                animated = workspace.status()["projects"][0]
                self.assertEqual(animated["active_view"], "animation")
                self.assertIn("composite_url", animated)
                self.assertEqual(workspace.record(original["project_id"], "project")["active_view"], "composite")
            finally:
                workspace.close()


if __name__ == "__main__":
    unittest.main()
