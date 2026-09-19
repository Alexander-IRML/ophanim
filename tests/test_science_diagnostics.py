"""Real bounded diagnostic rendering, including spatial spectra and timelines."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCIENCE = all(importlib.util.find_spec(name) is not None
              for name in ("numpy", "xarray", "scipy", "skimage", "pyproj", "matplotlib", "PIL"))


@unittest.skipUnless(SCIENCE, "requires optional science plotting dependencies")
class ScienceDiagnosticsTests(unittest.TestCase):
    def test_real_spatial_spectrum_and_exact_time_report_artifacts(self):
        from PIL import Image
        from ophanim.core.artifacts import write_json
        from ophanim.dynamics import AnalysisConfig, FlowConfig
        from ophanim.dynamics.sequence import analyze_sequence
        from ophanim.experiments import make_synthetic_dataset
        from ophanim.science_reports import science_diagnostics

        raw = make_synthetic_dataset("plane_wave", nx=25, ny=25, nt=49,
                                     wavelength_m=200_000, period_s=1800)
        config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                                smooth_sigma_km=0, flow=FlowConfig(enabled=False))
        times = ["2024-05-10T00:00:00Z", "2024-05-10T02:00:00Z", "2024-05-10T04:00:00Z"]
        result = analyze_sequence(raw, config, times)
        self.assertEqual(result.event["wave"]["status"], "estimated")
        self.assertIn("spatial_power", result.event["wave"]["spectrum_summary"])
        packet = {"config": config.to_dict(), "event_timeline": "events.json", "source_kind": "synthetic"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "events.json", result.timeline)
            output = root / "diagnostics"
            science_diagnostics(result.dataset, result.event, packet, output)
            report = (output / "report.html").read_text()
            for filename in ("wave_spatial_spectrum.png", "wave_temporal_support.png",
                             "wave_confidence_components.png", "event_timeline.png"):
                self.assertIn(filename, report)
                with Image.open(output / filename) as rendered:
                    rendered.load()
                    self.assertGreater(rendered.width, 100)
                    self.assertGreater(rendered.height, 100)
                    self.assertTrue(any(high > low for low, high in rendered.convert("RGB").getextrema()))
            self.assertIn("synthetic", report)


if __name__ == "__main__":
    unittest.main()
