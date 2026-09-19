"""Generate a reproducible, explicitly imagined 3-D visual acceptance specimen."""
import argparse
import json
from pathlib import Path

from ophanim.experiments.hypotheses import hypothesis_from_evidence
from ophanim.experiments.volume_model import publish_hypothesis_run
from ophanim.shawtynet.volume import visualize_volume_run, render_volume_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("wave_packet", "front", "sheared_jet"), default="wave_packet")
    parser.add_argument("--output", type=Path, default=Path("var/imagined-volume-acceptance"))
    parser.add_argument("--blender")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--style", choices=("luminous", "ghost", "quiet"), default="luminous")
    args = parser.parse_args()
    recipe = hypothesis_from_evidence(overrides={"kind": args.kind})
    source = publish_hypothesis_run(recipe, args.output)
    visual = visualize_volume_run(source, style=args.style)
    result = {"source": str(source), "visual": str(visual), "preview": str(visual / "render_package" / "preview.png")}
    print(json.dumps(result), flush=True)
    if args.blender:
        rendered = render_volume_run(visual, blender_executable=args.blender, resolution=(900, 600), samples=args.samples)
        result["render"] = str(rendered / "render.png")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
