"""Bounded uploads, coherent animation publication and manual photo overlays."""

from importlib.util import find_spec
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ophanim.core.artifacts import _publish, file_sha256, verify_run, write_json
from ophanim.shawtynet import VisualStyleConfig
from ophanim.shawtynet.studio import animation_run, studio_composite_run, upload_photo, upload_mask, MAX_UPLOAD_BYTES


HAS_IMAGES = find_spec("PIL") is not None
HAS_ARRAYS = HAS_IMAGES and all(find_spec(name) is not None for name in ("numpy", "xarray", "scipy"))


def image_bytes(*, size=(40, 30), mode="RGB", format="PNG", **options):
    from PIL import Image
    buffer = BytesIO()
    Image.new(mode, size, "red").save(buffer, format=format, **options)
    return buffer.getvalue()


def dynamic_fixture(count=9):
    import numpy as np
    import xarray as xr
    x = y = np.linspace(-20_000, 20_000, 5)
    xx, yy = np.meshgrid(x, y)
    shape = (count, len(y), len(x))
    times = np.datetime64("2024-01-01T00:00:00") + np.arange(count) * np.timedelta64(5, "m")
    dataset = xr.Dataset(coords={"time": times, "x": x, "y": y,
        "lat": (("y", "x"), 30 + yy / 111000), "lon": (("y", "x"), -100 + xx / 95000)})
    for name, value in {"dtec": 1, "dtec_z": 2, "observation_support": 1,
                        "flow_confidence": 1, "flow_interpretation_confidence": 1, "flow_u": 10, "flow_v": 0}.items():
        dataset[name] = (("time", "y", "x"), np.full(shape, value, dtype=float))
    dataset.attrs["run_id"] = "fixture"
    return dataset


def science_fixture(root):
    def build(stage):
        (stage / "dynamic.zarr").mkdir()
        write_json(stage / "dynamic.zarr" / "fixture.json", {"arrays": "opened by a test adapter"})
        write_json(stage / "event.json", {"time": "2024-01-01T00:20:00Z"})
        write_json(stage / "events.json", {"schema_version": "ophanim-event-timeline/1", "frames": [
            {"time": f"2024-01-01T00:{minute:02d}:00Z", "status": "analyzed", "event": {"time": f"2024-01-01T00:{minute:02d}:00Z", "event_scores": {"flow": 1}, "event_confidence": 1}, "analysis_window": {"semantics": "synthetic test fixture"}}
            for minute in range(0, 45, 5)]})
    return _publish(Path(root), "fixture", "science", {"source": "fixture"}, build)


@unittest.skipUnless(HAS_IMAGES, "optional Pillow unavailable")
class PhotographTests(unittest.TestCase):
    def test_mask_upload_is_grayscale_bounded_and_requires_photo_coordinates(self):
        from PIL import Image
        buffer = BytesIO()
        mask = Image.new("RGBA", (40, 30), (255, 255, 255, 0))
        mask.putpixel((2, 3), (255, 255, 255, 255))
        mask.save(buffer, format="PNG")
        with TemporaryDirectory() as directory:
            result = upload_mask(directory, buffer.getvalue(), expected_size=(40, 30))
            self.assertIn("mask_id", result)
            verify_run(result["directory"], kind="mask")
            with Image.open(Path(result["directory"])/"mask.png") as loaded:
                self.assertEqual(loaded.mode, "L")
                self.assertEqual(loaded.getpixel((0, 0)), 0)
                self.assertEqual(loaded.getpixel((2, 3)), 255)
                self.assertEqual(loaded.info, {})
            with self.assertRaisesRegex(ValueError, "dimensions"):
                upload_mask(directory, buffer.getvalue(), expected_size=(30, 40))
            with self.assertRaises(ValueError):
                upload_mask(directory, b"x"*(MAX_UPLOAD_BYTES+1))

    def test_normalized_metadata_free_upload_is_immutable_and_reused(self):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        metadata = PngInfo()
        metadata.add_text("private", "do not retain GPS location")
        content = image_bytes(pnginfo=metadata)
        with TemporaryDirectory() as temporary:
            first = upload_photo(temporary, content)
            second = upload_photo(temporary, content)
            self.assertEqual(first, second)
            manifest = verify_run(first["directory"], kind="photograph")
            self.assertIn("photo.png", manifest["files"])
            self.assertEqual((first["width"], first["height"]), (40, 30))
            with Image.open(Path(first["directory"]) / "photo.png") as image:
                self.assertEqual(image.info, {})
                self.assertEqual(image.mode, "RGB")

    def test_jpeg_orientation_is_applied_then_exif_is_removed(self):
        from PIL import Image
        exif = Image.Exif()
        exif[274] = 6
        content = image_bytes(size=(40, 30), format="JPEG", exif=exif)
        with TemporaryDirectory() as temporary:
            result = upload_photo(temporary, content)
            self.assertEqual((result["width"], result["height"]), (30, 40))
            with Image.open(Path(result["directory"]) / "photo.png") as image:
                self.assertFalse(image.getexif())

    def test_rejects_paths_unknown_formats_large_bodies_and_large_dimensions(self):
        with TemporaryDirectory() as temporary:
            for content in ("/etc/passwd", b"not a photo", b"x" * (MAX_UPLOAD_BYTES + 1), image_bytes(format="GIF")):
                with self.assertRaises(ValueError):
                    upload_photo(temporary, content)
            with patch("ophanim.shawtynet.studio.MAX_PHOTO_PIXELS", 100):
                with self.assertRaisesRegex(ValueError, "megapixels"):
                    upload_photo(temporary, image_bytes())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_transparency_is_flattened_and_animated_png_is_rejected(self):
        from PIL import Image
        buffer = BytesIO()
        Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(buffer, format="PNG")
        with TemporaryDirectory() as temporary:
            result = upload_photo(temporary, buffer.getvalue())
            with Image.open(Path(result["directory"]) / "photo.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            animated = BytesIO()
            Image.new("RGB", (4, 4), "red").save(animated, format="PNG", save_all=True,
                append_images=[Image.new("RGB", (4, 4), "blue")], duration=100)
            with self.assertRaisesRegex(ValueError, "single-frame"):
                upload_photo(temporary, animated.getvalue())


@unittest.skipUnless(HAS_ARRAYS, "optional scientific/image dependencies unavailable")
class AnimationTests(unittest.TestCase):
    def test_legacy_event_cannot_be_extrapolated_into_an_animation(self):
        with TemporaryDirectory() as directory:
            def build(stage):
                write_json(stage/"event.json", {"time": "2024-01-01T00:20:00Z"})
            science = _publish(Path(directory), "legacy", "science", {}, build)
            with self.assertRaisesRegex(ValueError, "Reanalyze"):
                animation_run(science)

    def test_portable_animation_maps_whole_selected_sequence_once(self):
        from ophanim.shawtynet import map_visual_fields
        data = dynamic_fixture()
        with TemporaryDirectory() as temporary:
            science = science_fixture(temporary)
            original = file_sha256(science / "manifest.json")
            with patch("xarray.open_zarr", side_effect=lambda *args, **kw: data.copy(deep=True)), patch("ophanim.shawtynet.map_visual_fields", wraps=map_visual_fields) as mapping:
                animation = animation_run(science, VisualStyleConfig(lic_streamline_steps=2), frame_count=3)
                repeated = animation_run(science, VisualStyleConfig(lic_streamline_steps=2), frame_count=3)
            self.assertEqual(animation, repeated)
            self.assertEqual(mapping.call_count, 1)
            self.assertEqual(mapping.call_args.args[0].sizes["time"], 3)
            manifest = verify_run(animation, kind="animation")
            self.assertEqual(manifest["identity"]["selected_frame_indices"], [0, 4, 8])
            self.assertIn("sequence/sequence.json", manifest["files"])
            self.assertFalse((animation / "animation.gif").exists())
            packet = json.loads((animation / "animation_packet.json").read_text())
            self.assertFalse(packet["rendered"])
            self.assertEqual(file_sha256(science / "manifest.json"), original)

    def test_rendered_animation_records_build_settings_and_actual_frames(self):
        from PIL import Image
        data = dynamic_fixture()
        calls = []
        def render(package, output, **options):
            calls.append(options)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", options["resolution"], (len(calls) * 50, 30, 80, 140)).save(output)
            return output
        with TemporaryDirectory() as temporary:
            science = science_fixture(temporary)
            with patch("xarray.open_zarr", side_effect=lambda *args, **kw: data.copy(deep=True)), patch("ophanim.shawtynet.studio.shutil.which", return_value="/fixture/blender"), patch("ophanim.shawtynet.studio.subprocess.run", return_value=SimpleNamespace(stdout="Blender 5.1\nbuild hash: abc\n")), patch("ophanim.shawtynet.render_blender", side_effect=render):
                animation = animation_run(science, VisualStyleConfig(lic_streamline_steps=2), frame_count=3, blender_executable="blender")
            manifest = verify_run(animation, kind="animation")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(c["samples"] == 8 and c["threads"] == 2 and c["resolution"] == (640, 426) for c in calls))
            self.assertIn("animation.gif", manifest["files"])
            self.assertEqual(len(list((animation / "frames").glob("*.png"))), 3)
            self.assertEqual(len(manifest["identity"]["blender"]["build_sha256"]), 64)
            with Image.open(animation / "animation.gif") as image:
                self.assertEqual(image.n_frames, 3)

    def test_cancel_between_frames_never_publishes_partial_animation(self):
        from PIL import Image
        data = dynamic_fixture()
        rendered = []
        def render(package, output, **options):
            rendered.append(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (10, 10), "red").save(output)
            return output
        with TemporaryDirectory() as temporary:
            science = science_fixture(temporary)
            with patch("xarray.open_zarr", side_effect=lambda *args, **kw: data.copy(deep=True)), patch("ophanim.shawtynet.studio.shutil.which", return_value="/fixture/blender"), patch("ophanim.shawtynet.studio.subprocess.run", return_value=SimpleNamespace(stdout="Blender fixture")), patch("ophanim.shawtynet.render_blender", side_effect=render):
                with self.assertRaises(InterruptedError):
                    animation_run(science, VisualStyleConfig(lic_streamline_steps=2), frame_count=3, blender_executable="blender", cancelled=lambda: bool(rendered))
            self.assertEqual(len(rendered), 1)
            self.assertEqual(list((science / "animations").iterdir()), [])

    def test_frame_limits_and_voxel_budget_are_enforced(self):
        for count in (True, 2, 13, 3.5):
            with self.assertRaises(ValueError):
                animation_run("unused", frame_count=count)
        with TemporaryDirectory() as temporary:
            science = science_fixture(temporary)
            with patch("xarray.open_zarr", side_effect=lambda *args, **kw: dynamic_fixture()):
                with self.assertRaisesRegex(ValueError, "budget"):
                    animation_run(science, VisualStyleConfig(max_render_voxels=8), frame_count=3)
            with patch("xarray.open_zarr", side_effect=lambda *args, **kw: dynamic_fixture(count=2)):
                with self.assertRaisesRegex(ValueError, "three scientific"):
                    animation_run(science)


@unittest.skipUnless(HAS_ARRAYS, "optional scientific/image dependencies unavailable")
class StudioCompositeTests(unittest.TestCase):
    def test_manual_masks_share_photo_crop_and_grade_is_recorded(self):
        import numpy as np
        from PIL import Image
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = upload_photo(root/"photos", image_bytes(size=(60, 30)))
            mask = Image.new("L", (60, 30), 0)
            mask.paste(255, (0, 0, 30, 30))
            buffer = BytesIO()
            mask.save(buffer, format="PNG")
            foreground = upload_mask(root/"masks", buffer.getvalue())
            def build(stage):
                Image.new("RGBA", (20, 30), (30, 60, 240, 255)).save(stage/"render.png")
            render = _publish(root/"renders", "fixture", "render", {}, build)
            composite = studio_composite_run(render, photo["directory"], opacity=.8,
                horizon_y=.8, horizon_fade=.9, saturation=.6, exposure=.7, grade_rgb=(.8, 1.1, 1),
                foreground_mask_directory=foreground["directory"])
            with Image.open(composite/"composite.png") as opened:
                pixels = np.asarray(opened)
                np.testing.assert_array_equal(pixels[2, :5], np.tile((255, 0, 0), (5, 1)))
                self.assertFalse(np.all(pixels[2, -5:] == (255, 0, 0)))
            packet = json.loads((composite/"composite_packet.json").read_text())
            self.assertEqual(packet["horizon_y"], .8)
            self.assertEqual(packet["grade_rgb"], [.8, 1.1, 1])
            self.assertEqual(packet["masks"]["foreground_mask"]["mask_id"], foreground["mask_id"])
            wrong = upload_mask(root/"masks", image_bytes(size=(20, 30)))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                studio_composite_run(render, photo["directory"], foreground_mask_directory=wrong["directory"])

    def test_center_crop_and_opacity_have_explicit_immutable_lineage(self):
        from PIL import Image
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = upload_photo(root / "photos", image_bytes(size=(60, 30)))
            def build(stage):
                Image.new("RGBA", (20, 30), (30, 60, 240, 255)).save(stage / "render.png")
            render = _publish(root / "renders", "example", "render", {"source": "fixture"}, build)
            original = file_sha256(Path(photo["directory"]) / "photo.png")
            composite = studio_composite_run(render, photo["directory"], opacity=0.3)
            repeated = studio_composite_run(render, photo["directory"], opacity=0.3)
            self.assertEqual(composite, repeated)
            manifest = verify_run(composite, kind="composite")
            self.assertEqual(manifest["identity"]["opacity"], 0.3)
            self.assertIn("center crop", manifest["identity"]["photo_fit"])
            self.assertEqual(file_sha256(Path(photo["directory"]) / "photo.png"), original)
            with Image.open(composite / "composite.png") as image:
                self.assertEqual(image.size, (20, 30))
                self.assertEqual(image.getpixel((10, 25)), (255, 0, 0))
                self.assertNotEqual(image.getpixel((10, 0)), (255, 0, 0))
            with self.assertRaises(ValueError):
                studio_composite_run(render, photo["directory"], opacity=float("nan"))


if __name__ == "__main__":
    unittest.main()
