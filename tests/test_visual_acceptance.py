"""Six-state science-field → material/export acceptance, not an aesthetic proof.

``visual_acceptance_cases`` is also a deterministic gallery input factory.
Fields are analytical synthetic evidence, not outputs of a learned model or
claims that a detector recovered their known ground truth. Actual rendered
images still require human art-direction review after these invariant tests.
"""
from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ophanim.shawtynet import VisualStyleConfig

AVAILABLE = all(find_spec(name) is not None for name in ("numpy", "xarray", "scipy", "PIL"))


def visual_acceptance_cases(*, cells=65, count=3, seed=17):
    """Yield six bounded, metrically explicit, known-state art inputs.

    Dataset coordinates use metres and five-minute UTC epochs. Gallery
    cameras should be identical across cases so differences are attributable
    to scientific state, not lighting/pose. Returned state is never mutated.
    """
    import numpy as np
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    x = y = np.linspace(-600_000, 600_000, cells)
    xx, yy = np.meshgrid(x, y)
    times = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(count) * np.timedelta64(5, "m")
    elapsed = np.arange(count) * 300.0
    shape = (count, cells, cells)
    width, speed = 200_000.0, 100.0
    moving = np.stack([6 * np.exp(-((xx - speed*t)**2 + yy**2) / (2*width**2)) for t in elapsed])
    phase = 2*np.pi*(xx[None] / 320_000 - elapsed[:, None, None]/3600)
    wave_field = 5*np.cos(phase)*np.exp(-(xx**2+yy**2)/(2*450_000**2))
    front_field = np.broadcast_to(5*np.tanh(xx/55_000), shape).copy()
    disturbance = gaussian_filter(np.random.default_rng(seed).normal(size=(cells, cells)), 2)
    disturbance /= np.std(disturbance)
    disturbance = np.broadcast_to(2*disturbance*np.exp(-(xx**2+yy**2)/(2*350_000**2)), shape).copy()
    for name, signal in (("quiet", np.zeros(shape)), ("translation", moving),
                         ("wave", wave_field), ("front", front_field),
                         ("uncertain", moving.copy()), ("disturbed", disturbance)):
        dataset = xr.Dataset(coords={"time": times, "x": x, "y": y,
            "lat": (("y", "x"), 35+yy/111_000), "lon": (("y", "x"), -105+xx/91_000)})
        confidence = 1.0 if name == "translation" else 0.0
        u = np.full(shape, speed if name in ("translation", "uncertain") else 0.0)
        v = np.zeros(shape)
        if name == "disturbed":
            u = np.broadcast_to(-yy/4000, shape).copy()
            v = np.broadcast_to(xx/4000, shape).copy()
            confidence = 0.4
        gy, gx = np.gradient(signal, y, x, axis=(1, 2))
        tangent = (np.arctan2(gy, gx)+np.pi/2) % np.pi
        coherence = np.where(np.hypot(gx, gy) > 1e-10, 0.95 if name != "disturbed" else 0.2, 0)
        fields = {"tec": (12+signal, "TECU"), "dtec": (signal, "TECU"), "dtec_z": (signal/2, "1"),
            "observation_support": (np.ones(shape), "1"), "anomaly_confidence": (np.ones(shape), "1"),
            "flow_u": (u, "m/s"), "flow_v": (v, "m/s"), "flow_confidence": (np.full(shape, confidence), "1"),
            "flow_interpretation_confidence": (np.full(shape, confidence), "1"),
            "structure_orientation": (tangent, "radian"), "structure_coherence": (coherence, "1"),
            "grad_mag": (np.hypot(gx, gy), "TECU/m"),
            "flow_divergence": (np.gradient(u, x, axis=2)+np.gradient(v, y, axis=1), "1/s"),
            "flow_vorticity": (np.gradient(v, x, axis=2)-np.gradient(u, y, axis=1), "1/s"),
            "flow_strain": (np.abs(np.gradient(u, x, axis=2)-np.gradient(v, y, axis=1)), "1/s"),
            "advection_residual": (np.full(shape, 0.05 if name == "disturbed" else 0.0), "TECU/s")}
        for field, (values, units) in fields.items():
            dataset[field] = (("time", "y", "x"), values, {"units": units, "semantic_class": "synthetic"})
        dataset.x.attrs["units"] = dataset.y.attrs["units"] = "m"
        dataset.attrs.update(source_kind="synthetic", fixture=name, seed=seed,
            evidence_semantics="Analytical known-state visual acceptance fixture; not recovered measurements")
        event = {"event_scores": {"flow" if name == "translation" else name: 1.0}, "event_confidence": 1.0}
        if name == "wave":
            event["wave"] = {"status": "estimated", "confidence": 1.0, "k_x_cycles_per_m": 1/320_000,
                "k_y_cycles_per_m": 0, "frequency_hz": -1/3600, "phase_rad": 0,
                "reference_time": "2025-01-01T00:00:00Z", "reference_x_m": 0, "reference_y_m": 0}
        timeline = {"schema_version": "ophanim-event-timeline/1", "frames": []}
        for time in times:
            timestamp = np.datetime_as_string(time, unit="s")+"Z"
            timeline["frames"].append({"time": timestamp, "status": "analyzed", "event": {**event, "time": timestamp},
                "analysis_window": {"start": timestamp, "end": timestamp,
                    "semantics": "Known analytical synthetic state at this epoch; no inference/recovery claim"}})
        yield name, dataset, event, timeline


@unittest.skipUnless(AVAILABLE, "optional visual dependencies unavailable")
class VisualAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ophanim.shawtynet import map_visual_fields
        cls.style = VisualStyleConfig(render_spacing_km=18.75, lic_streamline_steps=16, seed=17)
        cls.cases = {name: (data, event, timeline) for name, data, event, timeline in visual_acceptance_cases()}
        cls.visuals = {name: map_visual_fields(data, event, cls.style, event_timeline=timeline)
                       for name, (data, event, timeline) in cls.cases.items()}

    def test_six_states_export_distinct_non_saturated_materials(self):
        import numpy as np
        from PIL import Image
        from ophanim.shawtynet import export_render_package
        from ophanim.shawtynet.render import material_preview
        previews = []
        with TemporaryDirectory() as directory:
            for name, visual in self.visuals.items():
                manifest = export_render_package(visual, Path(directory)/name, frame_index=1)
                self.assertIn("preview.png", manifest["files"])
                with Image.open(Path(directory)/name/"preview.png") as image:
                    self.assertEqual(image.mode, "RGBA")
                preview = material_preview(visual, frame_index=1)
                self.assertTrue(np.all(np.isfinite(preview)))
                self.assertLess(float(np.mean(np.all(preview[..., :3] > 0.98, axis=-1))), 0.001)
                self.assertGreater(float(preview[..., 3].max()), 0)
                previews.append(preview)
            # Pairwise image distance catches identical-looking placeholder
            # fields even if named scalar control channels differ.
            for index, first in enumerate(previews):
                for second in previews[index+1:]:
                    self.assertGreater(float(np.mean(np.abs(first-second))), 0.001)

    def test_translation_has_aligned_fibers_and_uncertainty_removes_only_motion(self):
        import numpy as np
        translating, uncertain = self.visuals["translation"], self.visuals["uncertain"]
        texture = translating.flow_texture.values[1, 12:-12, 12:-12]
        north_change = np.mean(np.abs(np.diff(texture, axis=0)))
        east_change = np.mean(np.abs(np.diff(texture, axis=1)))
        self.assertGreater(north_change, 1.5*east_change)
        self.assertGreater(float(translating.flow_texture_strength.max()), 0.2)
        self.assertEqual(float(uncertain.flow_texture_strength.max()), 0)
        np.testing.assert_array_equal(translating.base_opacity, uncertain.base_opacity)
        np.testing.assert_array_equal(translating.emission, uncertain.emission)

    def test_front_is_an_elongated_normal_localized_fold(self):
        import numpy as np
        front = self.visuals["front"]
        ridge = front.front_fold.values[1]
        weights = ridge**2
        xx, yy = np.meshgrid(front.x, front.y)
        xvar, yvar = np.sum(weights*xx**2), np.sum(weights*yy**2)
        self.assertGreater(yvar, 5*xvar)
        self.assertGreater(float(front.ribbon_strength.max()), 0.1)
        self.assertEqual(float(front.flow_texture_strength.max()), 0)

    def test_front_material_lights_the_ridge_not_uniform_plateaus(self):
        import numpy as np
        from ophanim.shawtynet.render import material_preview
        front = self.visuals["front"]
        rgba = material_preview(front, frame_index=1)
        luminance = np.mean(rgba[..., :3], axis=-1) * rgba[..., 3]
        xx = front.x.values
        middle = slice(16, -16)
        ridge = luminance[middle][:, np.abs(xx) < 55_000].mean()
        plateaus = luminance[middle][:, (np.abs(xx) > 180_000) & (np.abs(xx) < 350_000)].mean()
        self.assertGreater(ridge, 2.5*plateaus)
        self.assertLess(float(front.front_fold_displacement.min()), -.1)
        self.assertGreater(float(front.front_fold_displacement.max()), .1)

    def test_wave_has_coherent_phase_and_disturbed_case_has_fraying(self):
        import numpy as np
        wave, disturbed, quiet = (self.visuals[name] for name in ("wave", "disturbed", "quiet"))
        expected = np.cos(2*np.pi*(wave.x.values/320_000-300/3600))
        np.testing.assert_allclose(wave.wave_modulation.values[1, 20], expected, atol=1e-5)
        self.assertGreater(float(wave.height_displacement.std()), 200)
        self.assertGreater(float(disturbed.breakup.mean()), float(quiet.breakup.mean())+0.05)
        self.assertEqual(float(quiet.height_displacement.max()), 0)

    def test_stationary_artistic_noise_does_not_flicker(self):
        import numpy as np
        disturbed = self.cases["disturbed"][0].copy(deep=True)
        disturbed.flow_confidence[:] = 0
        disturbed.flow_interpretation_confidence[:] = 0
        from ophanim.shawtynet import map_visual_fields
        visual = map_visual_fields(disturbed, self.cases["disturbed"][1], self.style)
        np.testing.assert_array_equal(visual.procedural_detail[0], visual.procedural_detail[-1])
        np.testing.assert_array_equal(visual.height_displacement[0], visual.height_displacement[-1])
        np.testing.assert_array_equal(visual.breakup[0], visual.breakup[-1])

    def test_exact_timeline_withholds_out_of_window_interpretation(self):
        from ophanim.shawtynet import map_visual_fields
        data, event, timeline = self.cases["wave"]
        partial = {**timeline, "frames": timeline["frames"][1:2]}
        visual = map_visual_fields(data, event, self.style, event_timeline=partial)
        self.assertEqual(float(abs(visual.wave_modulation[0]).max()), 0)
        self.assertGreater(float(abs(visual.wave_modulation[1]).max()), 0.9)
        self.assertEqual(float(abs(visual.wave_modulation[2]).max()), 0)
        self.assertEqual(json.loads(visual.attrs["time_status"])[0]["event_status"], "not_analyzed")

    def test_orientation_changes_front_ribbon_not_its_scientific_input(self):
        import numpy as np
        import xarray as xr
        from ophanim.shawtynet import map_visual_fields
        data, event, timeline = self.cases["front"]
        original = data.copy(deep=True)
        wrong = data.copy(deep=True)
        wrong.structure_orientation[:] = 0
        visual = map_visual_fields(wrong, event, self.style, event_timeline=timeline)
        self.assertLess(float(visual.front_fold.max()), float(self.visuals["front"].front_fold.max())*0.5)
        xr.testing.assert_identical(data, original)

    def test_kinematics_modulate_only_confident_detail(self):
        import numpy as np
        from ophanim.shawtynet import map_visual_fields
        original, event, timeline = self.cases["translation"]
        converging = original.copy(deep=True)
        converging.flow_divergence[:] = -0.003
        converging.flow_strain[:] = 0.003
        enhanced = map_visual_fields(converging, event, self.style, event_timeline=timeline)
        plain = self.visuals["translation"]
        self.assertGreater(float(enhanced.flow_texture_strength.mean()), float(plain.flow_texture_strength.mean())*1.15)
        np.testing.assert_array_equal(enhanced.base_opacity, plain.base_opacity)
        converging.flow_confidence[:] = 0
        converging.flow_interpretation_confidence[:] = 0
        withheld = map_visual_fields(converging, event, self.style, event_timeline=timeline)
        self.assertEqual(float(withheld.kinematic_modulation.max()), 0)
        self.assertEqual(float(withheld.flow_texture_strength.max()), 0)

    def test_axial_orientation_interpolation_does_not_turn_wrap_into_cross_flow(self):
        import numpy as np
        from ophanim.shawtynet.mapping import _render_grid
        data = self.cases["front"][0].copy(deep=True)
        data.structure_orientation[:, :, ::2] = 0.01
        data.structure_orientation[:, :, 1::2] = np.pi-0.01
        rendered = _render_grid(data, replace(self.style, render_spacing_km=9.375))
        angle = rendered.structure_orientation.values
        self.assertLess(float(np.max(np.minimum(angle, np.pi-angle))), 0.011)


if __name__ == "__main__":
    unittest.main()
