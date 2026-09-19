"""Reproduce the curated manual-camera demonstration, not a physical event.

Run with the scientific extra installed. The photograph's acquisition time is
not matched to the synthetic TEC. This is explicitly an artistic demonstration
of the implemented end-to-end workflow, including a manually traced skyline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    from PIL import Image, ImageDraw, ImageFilter
    from ophanim.science_runs import composite_run, render_run, visualize_run
    from ophanim.shawtynet import CameraConfig, VisualStyleConfig
    from ophanim.shawtynet_cli import analyze_request, load_mapping

    repository = Path(__file__).resolve().parents[1]
    examples = repository / "examples" / "shawtynet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--output", type=Path, default=repository / "var" / "shawtynet-photo-demo")
    parser.add_argument("--samples", type=int, default=32)
    arguments = parser.parse_args()
    root = arguments.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = load_mapping(examples / "photo-synthetic.yaml")
    science = analyze_request(request, base=examples, output_root=root / "science")
    camera = CameraConfig(**load_mapping(examples / "photo-camera.json"))
    style = VisualStyleConfig.from_file(examples / "photo-style.yaml")
    visual = visualize_run(science, style, camera=camera)
    render = render_run(visual, blender_executable=arguments.blender,
                        samples=arguments.samples, resolution=(1200, 802))
    trace = load_mapping(examples / "photo-foreground.json")
    width, height = trace["image_width"], trace["image_height"]
    mask = Image.new("L", (width, height), 0)
    polygon = [tuple(point) for point in trace["skyline_pixels"]]
    ImageDraw.Draw(mask).polygon(polygon + [(width - 1, height - 1), (0, height - 1)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(0.65))
    mask_path = root / "manual-foreground-mask.png"
    if not mask_path.exists():
        mask.save(mask_path)
    else:
        with Image.open(mask_path) as previous:
            if previous.convert("L").tobytes() != mask.tobytes():
                raise ValueError("Existing foreground mask differs; select a new output directory")
    composite = composite_run(render, examples / "sunset-skies-nps.jpg", foreground_mask=mask_path,
                              horizon_y=0.455, horizon_fade=0.75, saturation=0.8, exposure=1.0)
    print(json.dumps({"science": str(science), "visual": str(visual), "render": str(render),
                      "composite": str(composite), "photo_credit": str(examples / "PHOTO-CREDIT.md"),
                      "disclosure": "ARTISTIC synthetic demonstration; not an observed ionospheric event or matched photograph time"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
