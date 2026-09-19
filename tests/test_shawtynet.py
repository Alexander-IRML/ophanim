"""Scientific/art firewall, deterministic visual mapping and portable export."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ophanim.shawtynet import CameraConfig, VisualStyleConfig


_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "xarray", "scipy", "PIL"))


class StyleConfigTests(unittest.TestCase):
    def test_nested_style_matches_resolved_configuration(self):
        style = VisualStyleConfig.from_mapping({"name": "test", "flow": {"lic_strength": 0.3, "streamline_length": 10}, "sheet": {"vertical_exaggeration": 2.0}})
        self.assertEqual(style.flow_texture_strength, 0.3)
        self.assertEqual(style.lic_streamline_steps, 10)
        self.assertEqual(style.vertical_exaggeration_km, 2.0)
        self.assertEqual(VisualStyleConfig.from_mapping(style.to_dict()), style)
        with self.assertRaises(ValueError):
            VisualStyleConfig.from_mapping({"emission": {"gian": 1}})
        with self.assertRaises(ValueError):
            VisualStyleConfig(quiet_opacity_floor=0.8)
        with self.assertRaises(ValueError):
            CameraConfig(95, 0)


@unittest.skipUnless(_AVAILABLE, "ShawtyNet optional dependencies unavailable")
class VisualMappingTests(unittest.TestCase):
    @staticmethod
    def dataset():
        import numpy as np
        import xarray as xr
        x, y = np.linspace(-100_000, 100_000, 21), np.linspace(-80_000, 80_000, 17)
        xx, yy = np.meshgrid(x, y)
        field = np.exp(-(xx**2 + yy**2) / (2 * 40_000**2))
        shape = (2, len(y), len(x))
        dataset = xr.Dataset(coords={"time": np.array(["2025-01-01T00:00", "2025-01-01T00:10"], dtype="datetime64[ns]"), "x": x, "y": y, "lat": (("y", "x"), 40 + yy / 111_000), "lon": (("y", "x"), -100 + xx / 85_000)})
        values = {"dtec": np.broadcast_to(field, shape), "dtec_z": np.broadcast_to(4 * field, shape), "observation_support": np.ones(shape), "flow_u": np.full(shape, 100.0), "flow_v": np.zeros(shape), "flow_confidence": np.ones(shape), "flow_interpretation_confidence": np.ones(shape), "structure_coherence": np.ones(shape), "advection_residual": np.zeros(shape)}
        for name, array in values.items():
            dataset[name] = (("time", "y", "x"), array.copy())
        dataset.x.attrs["units"] = "m"
        dataset.y.attrs["units"] = "m"
        dataset.attrs.update(run_id="science-fixture", category="DERIVED")
        return dataset

    def test_determinism_and_input_firewall(self):
        import numpy as np
        import xarray as xr
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        original = data.copy(deep=True)
        style = VisualStyleConfig(lic_streamline_steps=6)
        first = map_visual_fields(data, {}, style)
        second = map_visual_fields(data, {}, style)
        xr.testing.assert_identical(data, original)
        xr.testing.assert_identical(first, second)
        self.assertEqual(first.attrs["category"], "ARTISTIC")
        changed = map_visual_fields(data, {}, replace(style, seed=8))
        self.assertFalse(np.array_equal(first.flow_texture.values, changed.flow_texture.values))
        self.assertEqual(first.attrs["input_content_sha256"], changed.attrs["input_content_sha256"])
        self.assertNotEqual(first.attrs["visual_id"], changed.attrs["visual_id"])

    def test_uncertain_motion_only_removes_fibers(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        style = VisualStyleConfig(lic_streamline_steps=3)
        confident = map_visual_fields(data, {}, style)
        data["flow_confidence"][:] = 0
        uncertain = map_visual_fields(data, {}, style)
        np.testing.assert_array_equal(confident.base_opacity, uncertain.base_opacity)
        np.testing.assert_array_equal(confident.emission, uncertain.emission)
        self.assertEqual(float(uncertain.flow_texture_strength.max()), 0)
        self.assertEqual(float(uncertain.flow_texture.max()), 0)

    def test_interpretation_gate_overrides_high_raw_motion_fit_without_erasing_base(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        style = VisualStyleConfig(lic_streamline_steps=3)
        legacy = map_visual_fields(data.drop_vars("flow_interpretation_confidence"), {}, style)
        self.assertIn("reanalysis required", legacy.attrs["legacy_motion_policy"])
        self.assertEqual(float(legacy.flow_texture_strength.max()), 0)
        self.assertEqual(float(legacy.flow_texture.max()), 0)
        admitted = map_visual_fields(data, {}, style)
        data.flow_interpretation_confidence[:] = 0
        withheld = map_visual_fields(data, {}, style)
        self.assertEqual(float(data.flow_confidence.min()), 1)
        self.assertGreater(float(admitted.flow_texture_strength.max()), 0)
        self.assertEqual(float(withheld.flow_texture_strength.max()), 0)
        self.assertEqual(float(withheld.flow_texture.max()), 0)
        self.assertEqual(float(withheld.kinematic_modulation.max()), 0)
        np.testing.assert_array_equal(admitted.base_opacity, withheld.base_opacity)
        np.testing.assert_array_equal(admitted.emission, withheld.emission)
        np.testing.assert_array_equal(withheld.procedural_detail[0], withheld.procedural_detail[-1])
        self.assertEqual(withheld.attrs["motion_confidence_channel"], "flow_interpretation_confidence")
        self.assertIn("not_applicable", withheld.attrs["legacy_motion_policy"])

    def test_quiet_veil_and_missing_support(self):
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        data["dtec"][:] = 0
        data["dtec_z"][:] = 0
        data["flow_u"][:] = 0
        visual = map_visual_fields(data, {"event_scores": {"quiet": 1}}, VisualStyleConfig(lic_streamline_steps=2))
        self.assertGreater(float(visual.base_opacity.max()), 0)
        self.assertEqual(float(visual.flow_texture_strength.max()), 0)
        self.assertEqual(float(visual.emission.max()), 0)
        data["observation_support"][:] = 0
        missing = map_visual_fields(data, {}, VisualStyleConfig(lic_streamline_steps=2))
        self.assertEqual(float(missing.base_opacity.max()), 0)

    def test_missing_anomaly_does_not_hide_supported_tec(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        data["tec"] = data.dtec.copy()
        data["dtec"][:] = np.nan
        data["dtec_z"][:] = np.nan
        visual = map_visual_fields(data, {}, VisualStyleConfig(lic_streamline_steps=2))
        self.assertGreater(float(visual.base_opacity.max()), 0)
        self.assertEqual(float(visual.emission.max()), 0)

    def test_wave_uses_signed_frequency_and_refuses_missing_phase(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        wave = {"status": "ok", "confidence": 1.0, "k_x_cycles_per_m": 1 / 100_000, "k_y_cycles_per_m": 0.0, "frequency_hz": -1 / 2400, "phase_rad": 0.0, "reference_time": "2025-01-01T00:00:00Z", "reference_x_m": 0.0, "reference_y_m": 0.0}
        visual = map_visual_fields(data, {"wave": wave, "event_scores": {"wave": 1}}, VisualStyleConfig(lic_streamline_steps=2))
        expected = np.cos(2 * np.pi * (data.x.values / 100_000 - 600 / 2400))
        np.testing.assert_allclose(visual.wave_modulation.isel(time=1, y=0), expected, atol=1e-6)
        wave.pop("phase_rad")
        withheld = map_visual_fields(data, {"wave": wave}, VisualStyleConfig(lic_streamline_steps=2))
        self.assertEqual(float(withheld.wave_modulation.max()), 0)
        self.assertEqual(withheld.attrs["wave_visual_status"], "missing_phase_reference")

    def test_event_specific_visual_fields(self):
        from ophanim.shawtynet import map_visual_fields
        data, style = self.dataset(), VisualStyleConfig(lic_streamline_steps=2)
        quiet = map_visual_fields(data, {}, style)
        front = map_visual_fields(data, {"event_scores": {"front": 1}}, style)
        local = map_visual_fields(data, {"event_scores": {"localized": 1}}, style)
        disturbed = map_visual_fields(data, {"event_scores": {"disturbed": 1}}, style)
        self.assertGreater(float(front.front_fold.max()), float(quiet.front_fold.max()))
        self.assertGreater(float(local.localized_emphasis.max()), float(quiet.localized_emphasis.max()))
        self.assertGreater(float(disturbed.breakup.max()), float(quiet.breakup.max()))

    def test_render_resampling_is_separate_and_bounded(self):
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        original_shape = dict(data.sizes)
        visual = map_visual_fields(data, {}, VisualStyleConfig(render_spacing_km=5, lic_streamline_steps=2))
        self.assertGreater(visual.sizes["x"], data.sizes["x"])
        self.assertEqual(dict(data.sizes), original_shape)
        with self.assertRaises(ValueError):
            map_visual_fields(data, {}, VisualStyleConfig(render_spacing_km=0.1, max_render_cells=100))

    def test_artistic_boundary_taper_preserves_scientific_support(self):
        from ophanim.shawtynet import map_visual_fields
        visual = map_visual_fields(self.dataset(), {}, VisualStyleConfig(lic_streamline_steps=2))
        self.assertEqual(float(visual.base_opacity.isel(x=0).max()), 0)
        self.assertEqual(float(visual.base_opacity.isel(y=-1).max()), 0)
        self.assertEqual(float(visual.render_support.min()), 1)
        self.assertGreater(float(visual.base_opacity.max()), 0)

    def test_antimeridian_render_coordinates_do_not_cross_zero(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields
        data = self.dataset()
        longitudes = np.broadcast_to(np.linspace(179, 181, data.sizes["x"]), (data.sizes["y"], data.sizes["x"]))
        data = data.assign_coords(lon=(("y", "x"), (longitudes + 180) % 360 - 180))
        visual = map_visual_fields(data, {}, VisualStyleConfig(render_spacing_km=5, lic_streamline_steps=2))
        self.assertGreater(float(abs(visual.lon).min()), 178)

    def test_lic_direction_and_missing_hole(self):
        import numpy as np
        from ophanim.shawtynet.lic import line_integral_convolution
        shape = (48, 64)
        valid = np.ones(shape, dtype=bool)
        valid[:, 31:33] = False
        texture = line_integral_convolution(np.ones(shape), np.zeros(shape), valid, dx_m=1, dy_m=1, seed=4, steps=14)
        self.assertTrue(np.all(texture[:, 31:33] == 0))
        # Horizontal fibers have stronger neighbor correlation horizontally.
        horizontal = np.mean(np.abs(np.diff(texture[5:-5, 4:28], axis=1)))
        vertical = np.mean(np.abs(np.diff(texture[5:-5, 4:28], axis=0)))
        self.assertLess(horizontal, vertical)


@unittest.skipUnless(_AVAILABLE, "ShawtyNet optional dependencies unavailable")
class RenderAndCompositeTests(unittest.TestCase):
    def test_exr_header_inspection_checks_real_multipart_channels_and_bounds(self):
        import struct
        from ophanim.shawtynet.render import _exr_pass_channels

        def part(channels):
            channel_list = b"".join(name.encode()+b"\0"+struct.pack("<iB3xii", 2, 0, 1, 1) for name in channels)+b"\0"
            return b"channels\0chlist\0"+struct.pack("<I", len(channel_list))+channel_list+b"\0"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"passes.exr"
            names = ["ViewLayer.Combined.R", "ViewLayer.Emission.R", "ViewLayer.Depth.Z",
                     "ViewLayer.ArtisticBaseCoverage.X", "ViewLayer.ArtisticDetailCoverage.X"]
            path.write_bytes(struct.pack("<II", 20000630, 2|0x1000)+b"".join(part([name]) for name in names)+b"\0")
            self.assertEqual(_exr_pass_channels(path), sorted(names))
            path.write_bytes(struct.pack("<II", 20000630, 2)+part(names))
            self.assertEqual(_exr_pass_channels(path), sorted(names))
            for invalid in (b"not EXR", struct.pack("<II", 20000630, 2)+b"a\0b\0"+struct.pack("<I", 2**31),
                            struct.pack("<II", 20000630, 2)+b"a"*256):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    _exr_pass_channels(path)

    def test_shell_preserves_earth_curvature_and_observer_frame(self):
        import numpy as np
        from ophanim.shawtynet.render import geodetic_to_ecef, ecef_to_observer_enu, shell_mesh
        camera = CameraConfig(0, 0)
        overhead = ecef_to_observer_enu(geodetic_to_ecef(0, 0, 300), camera)
        np.testing.assert_allclose(overhead, [0, 0, 300], atol=1e-9)
        lat, lon = np.meshgrid(np.array([-1, 0, 1]), np.array([-2, 0, 2]), indexing="ij")
        mesh = shell_mesh(lat, lon, np.zeros((3, 3)), altitude_km=300, camera=camera)
        self.assertEqual(mesh["faces"].shape, (8, 3))
        self.assertEqual(mesh["uv"].shape, (9, 2))
        self.assertLess(mesh["vertices"][0, 2], mesh["vertices"][4, 2])
        self.assertLess(mesh["vertices"][3, 0], 0)
        self.assertGreater(mesh["vertices"][5, 0], 0)

    def test_export_contains_precise_values_and_detects_corruption(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields, export_render_package
        from ophanim.shawtynet.render import _verify_package
        visual = map_visual_fields(VisualMappingTests.dataset(), {}, VisualStyleConfig(lic_streamline_steps=2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "render"
            manifest = export_render_package(visual, root, frame_index=-1)
            self.assertEqual(_verify_package(root)["render_package_id"], manifest["render_package_id"])
            np.testing.assert_array_equal(np.load(root / "textures/emission.npy", allow_pickle=False), visual.emission.isel(time=-1))
            self.assertEqual(manifest["frame_index"], 1)
            self.assertTrue((root / "blender_scene.py").is_file())
            self.assertEqual(manifest["camera"]["pitch_deg"], 90)
            with self.assertRaises(FileExistsError):
                export_render_package(visual, root)
            (root / "textures/emission.png").write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                _verify_package(root)

    def test_windows_blender_receives_converted_paths(self):
        from types import SimpleNamespace
        from ophanim.shawtynet.render import _blender_argument_path
        with patch("ophanim.shawtynet.render.sys.platform", "linux"), patch("ophanim.shawtynet.render.shutil.which", return_value="/usr/bin/wslpath"), patch("ophanim.shawtynet.render.subprocess.run", return_value=SimpleNamespace(stdout="C:\\test\\render.png\n")) as run:
            value = _blender_argument_path(Path("/mnt/c/test/render.png"), "/mnt/c/Blender/blender.exe")
            self.assertEqual(value, "C:\\test\\render.png")
            self.assertEqual(run.call_args.args[0][:2], ["/usr/bin/wslpath", "-w"])

    def test_composite_respects_foreground_clouds_horizon(self):
        import numpy as np
        from ophanim.shawtynet.composite import composite_arrays
        photo = np.full((20, 30, 3), 0.2)
        render = np.ones((20, 30, 4))
        mask = np.zeros((20, 30))
        mask[:5, :5] = 1
        cloud = np.zeros((20, 30))
        cloud[:5, 10:15] = 0.5
        result = composite_arrays(photo, render, foreground_mask=mask, cloud_mask=cloud, horizon_y=0.75, horizon_fade=0.8)
        np.testing.assert_allclose(result[:5, :5], photo[:5, :5], atol=1e-6)
        np.testing.assert_allclose(result[15:], photo[15:], atol=1e-6)
        self.assertLess(float(result[1, 12, 0]), float(result[1, 20, 0]))
        self.assertGreater(float(result[1, 20, 0]), float(result[14, 20, 0]))

    def test_horizon_fade_is_continuous_and_manual_controls_validated(self):
        import numpy as np
        from ophanim.shawtynet.composite import composite_arrays
        photo = np.full((1000, 2, 3), 0.2)
        render = np.ones((1000, 2, 4))
        result = composite_arrays(photo, render, horizon_y=.5, horizon_fade=.6)
        self.assertLess(float(np.max(np.abs(result[499]-result[500]))), .001)
        self.assertGreater(float(result[300, 0, 0]), float(result[450, 0, 0]))
        for grade in (None, "red", (True, 1, 1), (float("nan"), 1, 1)):
            with self.assertRaises(ValueError):
                composite_arrays(photo, render, grade_rgb=grade)

    def test_photo_file_workflow_receipt_and_dimension_rejection(self):
        import numpy as np
        from PIL import Image
        from ophanim.shawtynet import composite_photograph
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo, render = root / "photo.png", root / "render.png"
            Image.fromarray(np.full((10, 12, 3), 30, dtype=np.uint8)).save(photo)
            Image.fromarray(np.full((10, 12, 4), 255, dtype=np.uint8)).save(render)
            output = composite_photograph(photo, render, root / "composite.png")
            self.assertTrue(output.is_file())
            receipt = json.loads((root / "composite.composite.json").read_text())
            self.assertEqual(receipt["category"], "ARTISTIC")
            self.assertEqual(len(receipt["inputs"]["photograph"]["sha256"]), 64)
            Image.fromarray(np.zeros((2, 3, 4), dtype=np.uint8)).save(root / "small.png")
            with self.assertRaises(ValueError):
                composite_photograph(photo, root / "small.png", root / "bad.png")


if __name__ == "__main__":
    unittest.main()
