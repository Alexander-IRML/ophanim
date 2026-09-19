"""Real 3-D geometry/preview regressions, deliberately not physical validation."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "xarray", "zarr", "PIL", "skimage"))
if _AVAILABLE:
    import numpy as np

from ophanim.core.artifacts import canonical_json, file_sha256, verify_run, write_json
from ophanim.experiments.hypotheses import hypothesis_from_evidence
from ophanim.experiments.volume_model import simulate_hypothesis, publish_hypothesis_run
from ophanim.shawtynet.config import CameraConfig
from ophanim.shawtynet.volume import (
    _verify_volume_package, export_volume_package, volume_preview,
    visualize_volume_run, animation_volume_run,
)
from ophanim.shawtynet.volume_blender_scene import validate_package, build_scene


def small_recipe(kind="wave_packet", **options):
    return hypothesis_from_evidence(overrides={"kind": kind, "extent_km": 400,
        "spacing_km": 25, "vertical_spacing_km": 25, "duration_minutes": 12,
        "frame_count": 3, "width_km": 100, **options})


@unittest.skipUnless(_AVAILABLE, "Imagined-volume rendering optional dependencies unavailable")
class VolumeRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.volume = simulate_hypothesis(small_recipe())

    def test_preview_is_camera_dependent_three_dimensional_projection(self):
        dataset = self.volume.dataset
        front = volume_preview(dataset, frame_index=1, resolution=(180, 120))
        side = volume_preview(dataset, frame_index=1, resolution=(180, 120),
            camera=CameraConfig(30, -98, heading_deg=55, pitch_deg=-32))
        self.assertEqual(front.shape, (120, 180, 4))
        # A camera orbit must change the projected structure, not merely its
        # color gain. Compare occupied silhouettes rather than an arbitrary
        # average RGB threshold that depends on the current artistic palette.
        first_mask = front[..., :3].max(axis=-1) > 60
        second_mask = side[..., :3].max(axis=-1) > 60
        overlap = np.count_nonzero(first_mask & second_mask) / np.count_nonzero(first_mask | second_mask)
        self.assertLess(overlap, .85)
        np.testing.assert_array_equal(front, volume_preview(dataset, frame_index=1, resolution=(180, 120)))
        self.assertLess(float(np.mean(np.all(front[..., :3] > 250, axis=-1))), .001)
        self.assertGreater(float(np.std(front[..., :3])), 20)
        self.assertGreater(float(np.mean(front[..., :3].max(axis=-1) > 60)), .04)

    def test_no_implicit_ground_camera_or_scientific_volume(self):
        with self.assertRaisesRegex(ValueError, "ground camera"):
            volume_preview(self.volume.dataset, camera=CameraConfig(31, -98), resolution=(64, 64))
        measured = self.volume.dataset.copy()
        measured.attrs["source_kind"] = "native"
        with self.assertRaisesRegex(ValueError, "imagined volumes only"):
            volume_preview(measured, resolution=(64, 64))

    def test_numerical_depth_changes_image_not_just_column_projection(self):
        dataset = self.volume.dataset
        reverse = dataset.copy(deep=True)
        reverse["density"].values[:] = reverse.density.values[:, ::-1]
        np.testing.assert_allclose(reverse.density.sum("z"), dataset.density.sum("z"), atol=1e-5)
        first = volume_preview(dataset, resolution=(160, 100))
        second = volume_preview(reverse, resolution=(160, 100))
        self.assertGreater(float(np.mean(abs(first.astype(float) - second))), 1)

    def test_package_contains_real_nonplanar_surfaces_and_threads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            metadata = export_volume_package(self.volume.dataset, root, preview=False)
            self.assertEqual(_verify_volume_package(root)["render_package_id"], metadata["render_package_id"])
            self.assertGreater(metadata["total_mesh_faces"], 100)
            self.assertGreater(metadata["thread_count"], 10)
            with np.load(root / metadata["meshes"][0]["file"]) as arrays:
                self.assertGreater(float(np.ptp(arrays["vertices"][:, 2])), 100)
                self.assertGreater(np.linalg.matrix_rank(arrays["vertices"] - arrays["vertices"].mean(axis=0)), 2)
            with np.load(root / "threads.npz") as arrays:
                self.assertGreater(float(np.ptp(arrays["points"][:, 2])), 100)
            self.assertIn("not plasma trajectories", metadata["semantic"])

    def test_repeatability_and_temporal_coherence(self):
        data = self.volume.dataset
        same = data.copy(deep=True)
        same["density"].values[:] = data.density.values[0]
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            export_volume_package(same, first, frame_index=0, preview=False, seed=123)
            export_volume_package(same, second, frame_index=1, preview=False, seed=123)
            self.assertEqual(file_sha256(first / "threads.npz"), file_sha256(second / "threads.npz"))
            self.assertEqual(file_sha256(first / "isosurface_0.npz"), file_sha256(second / "isosurface_0.npz"))
        frames = [volume_preview(data, frame_index=i, resolution=(120, 80), density_scale=float(data.density.max())) for i in range(3)]
        difference = float(np.mean(abs(frames[1].astype(float) - frames[0])))
        self.assertGreater(difference, .05)
        self.assertLess(difference, 40)

    def test_zero_volume_exports_honestly_empty_geometry(self):
        empty = self.volume.dataset.copy(deep=True)
        empty.density.values[:] = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "empty"
            metadata = export_volume_package(empty, root, preview=False)
            self.assertEqual(metadata["meshes"], [])
            self.assertEqual(metadata["thread_count"], 0)
            _verify_volume_package(root)

    def test_package_tamper_and_arbitrary_code_rejected(self):
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            metadata = export_volume_package(self.volume.dataset, root, preview=False)
            (root / "volume_blender_scene.py").write_text("raise RuntimeError('untrusted')")
            with self.assertRaisesRegex(ValueError, "checksum"):
                _verify_volume_package(root)
            metadata["files"]["volume_blender_scene.py"] = file_sha256(root / "volume_blender_scene.py")
            metadata.pop("render_package_id")
            metadata["render_package_id"] = sha256(canonical_json(metadata)).hexdigest()
            write_json(root / "scene_metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "trusted renderer"):
                _verify_volume_package(root)

    def test_standalone_rejects_extra_import_files_and_unsafe_mesh_paths(self):
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            metadata = export_volume_package(self.volume.dataset, root, preview=False)
            (root / "numpy.py").write_text("raise RuntimeError('untrusted import')")
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_package(root)
            (root / "numpy.py").unlink()
            metadata["meshes"][0]["file"] = "../outside.npz"
            metadata.pop("render_package_id")
            metadata["render_package_id"] = sha256(canonical_json(metadata)).hexdigest()
            write_json(root / "scene_metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "isosurface references"):
                validate_package(root)

    def test_unsigned_offsets_cannot_wrap_into_unbounded_spline_counts(self):
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            metadata = export_volume_package(self.volume.dataset, root, preview=False)
            for offsets in (np.array([0, 8, 3, 10], dtype=np.uint64), np.array([], dtype=np.uint64)):
                np.savez_compressed(root / "threads.npz", points=np.zeros((10, 3)),
                                    weights=np.ones(10), offsets=offsets)
                metadata["files"]["threads.npz"] = file_sha256(root / "threads.npz")
                metadata.pop("render_package_id", None)
                metadata["render_package_id"] = sha256(canonical_json(metadata)).hexdigest()
                write_json(root / "scene_metadata.json", metadata)
                with self.assertRaisesRegex(ValueError, "thread geometry"):
                    validate_package(root)

    def test_standalone_rejects_numpy_header_bomb_before_allocation(self):
        from hashlib import sha256
        import io
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            metadata = export_volume_package(self.volume.dataset, root, preview=False)
            with np.load(root / "threads.npz") as original:
                arrays = {name: original[name] for name in original.files}
            with zipfile.ZipFile(root / "threads.npz", "w") as archive:
                for name, value in arrays.items():
                    stream = io.BytesIO()
                    if name == "points":
                        np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": (2**50, 3)})
                    else:
                        np.save(stream, value, allow_pickle=False)
                    archive.writestr(name + ".npy", stream.getvalue())
            metadata["files"]["threads.npz"] = file_sha256(root / "threads.npz")
            metadata.pop("render_package_id")
            metadata["render_package_id"] = sha256(canonical_json(metadata)).hexdigest()
            write_json(root / "scene_metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "header"):
                validate_package(root)

    def test_standalone_checks_render_budget_without_importing_blender(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            build_scene("unused", samples=100_000)
        with self.assertRaisesRegex(ValueError, "megapixels"):
            build_scene("unused", width=2400, height=2400)

    def test_mesh_budget_is_aggregate_not_per_surface(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "package"
            metadata = export_volume_package(self.volume.dataset, root, preview=False)
            total = metadata["total_mesh_faces"]
            self.assertGreater(total, max(mesh["faces"] for mesh in metadata["meshes"]))
            with patch("ophanim.shawtynet.volume_blender_scene.MAX_MESH_FACES", total - 1):
                with self.assertRaisesRegex(ValueError, "aggregate face budget"):
                    validate_package(root)

    def test_publication_animation_lineage_and_fixed_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = publish_hypothesis_run(small_recipe(), temporary)
            visual = visualize_volume_run(source)
            self.assertEqual(verify_run(visual)["kind"], "visual")
            self.assertTrue((visual / "render_package" / "preview.png").exists())
            self.assertTrue((visual / "visual_diagnostics" / "report.html").exists())
            animation = animation_volume_run(visual, frame_count=3)
            manifest = verify_run(animation, kind="animation")
            packet = json.loads((animation / "animation_packet.json").read_text())
            self.assertFalse(packet["rendered"])
            self.assertEqual(packet["frame_count"], 3)
            sequence = json.loads((animation / "sequence" / "sequence.json").read_text())
            packages = [_verify_volume_package(animation / "sequence" / frame["path"]) for frame in sequence["frames"]]
            self.assertEqual(len({package["density_scale"] for package in packages}), 1)
            self.assertEqual(manifest["identity"]["visual_run_id"], verify_run(visual)["run_id"])

    def test_cancelled_export_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = publish_hypothesis_run(small_recipe(), temporary)
            with self.assertRaises(InterruptedError):
                visualize_volume_run(source, cancelled=lambda: True)
            self.assertFalse((source / "visuals").exists())


if __name__ == "__main__":
    unittest.main()
