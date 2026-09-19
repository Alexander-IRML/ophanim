"""Structural frontend contracts; interactive behavior is checked in Chrome."""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest


WEB = Path(__file__).resolve().parents[1] / "src" / "ophanim" / "web"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.elements = {}
        self.stack = []
        self.errors = []
        self.scenario_kinds = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        identity = attrs.get("id")
        if identity:
            self.ids.append(identity)
            self.elements[identity] = {"tag": tag, "attributes": attrs, "ancestors": tuple(item[1] for item in self.stack)}
        if tag == "option" and any(identity == "work-kind" for _, identity in self.stack):
            self.scenario_kinds.append(attrs["value"])
        if tag not in VOID:
            self.stack.append((tag, identity))

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1][0] != tag:
            self.errors.append((tag, self.stack[-1] if self.stack else None))
        else:
            self.stack.pop()


class WorkspaceUIContractTests(unittest.TestCase):
    def setUp(self):
        self.document = Document()
        self.document.feed((WEB / "index.html").read_text())

    def test_markup_has_unique_ids_and_balanced_elements(self):
        self.assertEqual(self.document.errors, [])
        self.assertEqual(self.document.stack, [])
        self.assertEqual([key for key, count in Counter(self.document.ids).items() if count > 1], [])

    def test_discover_is_default_and_legacy_panels_are_advanced_only(self):
        elements = self.document.elements
        for view in ("discover", "event", "studio", "experiments", "advanced"):
            tab = elements[f"tab-{view}"]["attributes"]
            panel = elements[f"work-{view}"]["attributes"]
            self.assertEqual(tab["aria-controls"], panel["id"])
            self.assertEqual(panel["aria-labelledby"], tab["id"])
            self.assertEqual(tab["aria-selected"], str(view == "discover").lower())
            self.assertEqual("hidden" in panel, view != "discover")
        for identity in ("source-panel", "regional-panel", "mamba-panel", "spatial-panel", "forecast-panel", "results", "shawtynet-panel"):
            self.assertIn("work-advanced", elements[identity]["ancestors"])

    def test_frontend_static_element_references_exist(self):
        ids = set(self.document.ids)
        script = (WEB / "workspace.js").read_text()
        for identity in re.findall(r"\$\(['\"]([a-z][a-z0-9-]*)['\"]\)", script):
            self.assertIn(f"work-{identity}", ids)
        # Existing source/model tools retain their expected DOM identifiers.
        for name in ("app.js", "shawtynet.js"):
            for identity in re.findall(r'querySelector\("#([a-z][a-z0-9-]*)"\)', (WEB / name).read_text()):
                self.assertIn(identity, ids)

    def test_default_scenario_is_valid_and_within_the_backend_budget(self):
        from ophanim.experiments.scenarios import KINDS, ScenarioConfig

        fields = {"kind": "kind", "amplitude_tecu": "amplitude", "width_km": "width", "speed_m_s": "speed",
                  "bearing_deg": "bearing", "period_minutes": "period", "duration_minutes": "duration",
                  "extent_km": "extent", "spacing_km": "spacing", "seed": "seed", "center_latitude": "latitude",
                  "center_longitude": "longitude"}
        self.assertTrue(set(self.document.scenario_kinds).issubset(KINDS))
        parameters = {key: float(self.document.elements[f"work-{identity}"]["attributes"]["value"])
                      for key, identity in fields.items() if key != "kind"}
        parameters["seed"] = int(parameters["seed"])
        parameters["kind"] = self.document.scenario_kinds[0]
        config = ScenarioConfig.from_mapping(parameters)
        nx, nt = config.dimensions
        self.assertLessEqual(nx * nx * nt, 200_000)

    def test_full_manual_photo_and_camera_controls_are_studio_only(self):
        for identity in ("camera-latitude", "camera-longitude", "camera-altitude", "camera-roll",
                         "photo-horizon", "photo-fade", "photo-saturation", "photo-exposure",
                         "photo-red", "photo-green", "photo-blue", "foreground-mask", "cloud-mask"):
            self.assertIn("work-studio", self.document.elements[f"work-{identity}"]["ancestors"])
        self.assertNotIn("required", self.document.elements["work-camera-latitude"]["attributes"])
        self.assertIn("scene center", self.document.elements["work-camera-latitude"]["attributes"]["placeholder"])

    def test_imagination_is_primary_and_ifm_remains_deferred(self):
        elements = self.document.elements
        self.assertIn("disabled", elements["work-2d-fields"]["attributes"])
        self.assertIn("hidden", elements["work-2d-fields"]["attributes"])
        self.assertNotIn("hidden", elements["work-3d-fields"]["attributes"])
        self.assertIn("checked", elements["work-camera-auto"]["attributes"])
        text = (WEB / "index.html").read_text()
        self.assertIn("IFM is reserved for v0.3", text)
        self.assertIn("remote simulation is not configured", text)


if __name__ == "__main__":
    unittest.main()
