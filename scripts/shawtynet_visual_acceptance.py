"""Reproducible six-state material/Blender gallery for human acceptance review.

Uses analytical known-state fixtures, not detector-recovery results. Run from a
source checkout with the science extra. No network access or source acquisition.
Every case uses the same camera, style, resolution and seed. This intentionally
tests the science-to-art boundary separately from upstream scientific accuracy.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys


def main():
    import numpy as np
    from PIL import Image, ImageDraw
    from ophanim.core.artifacts import file_sha256, software_identity, write_json
    from ophanim.shawtynet import CameraConfig, VisualStyleConfig, map_visual_fields, export_render_package, render_blender

    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "tests"))
    from test_visual_acceptance import visual_acceptance_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=repository/"var"/"shawtynet-visual-acceptance")
    parser.add_argument("--blender", help="Optional Blender executable; omission exports material previews only")
    parser.add_argument("--passes", action="store_true", help="Also verify multilayer emission/depth/coverage export")
    arguments = parser.parse_args()
    root = arguments.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A source revision produces a new gallery, never overwrites old evidence.
    from hashlib import sha256
    identity = {"software":software_identity(extra_sections=("shawtynet",)),
                "script_sha256":file_sha256(Path(__file__)),
                "fixtures_sha256":file_sha256(repository/"tests"/"test_visual_acceptance.py"),
                "blender":arguments.blender,"passes":arguments.passes}
    run = root/sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    run.mkdir(exist_ok=True)
    style = VisualStyleConfig(render_spacing_km=18.75,lic_streamline_steps=16,seed=17)
    camera = CameraConfig(observer_latitude=35,observer_longitude=-105,pitch_deg=65,horizontal_fov_deg=90)
    size = (640,426)
    contact = Image.new("RGB", (size[0]*3,(size[1]+35)*2), (8,12,20))
    draw = ImageDraw.Draw(contact)
    cases = []
    for index,(name,data,event,timeline) in enumerate(visual_acceptance_cases()):
        visual = map_visual_fields(data,event,style,event_timeline=timeline)
        destination = run/name
        if destination.exists():
            manifest = json.loads((destination/"scene_metadata.json").read_text())
            if manifest.get("visual_id") != visual.attrs["visual_id"] or any(
                    file_sha256(destination/name) != digest for name,digest in manifest["files"].items()):
                raise ValueError("Existing gallery package has changed; select a new output directory")
        else:
            export_render_package(visual,destination,camera,frame_index=1)
        source = destination/"preview.png"
        receipt = None
        if arguments.blender:
            source = destination/"render.png"
            if not source.exists():
                receipt = render_blender(destination,source,blender_executable=arguments.blender,
                                         resolution=size,samples=12,threads=2,
                                         passes=arguments.passes and name=="translation")
        with Image.open(source) as opened:
            rgba = opened.convert("RGBA")
            values = np.asarray(rgba,dtype=float)/255
            matte = Image.new("RGBA",rgba.size,(8,12,20,255))
            image = Image.alpha_composite(matte,rgba).convert("RGB").resize(size)
        left,top = index%3*size[0],index//3*(size[1]+35)
        contact.paste(image,(left,top+35))
        draw.text((left+12,top+10),name.upper(),fill=(230,235,245))
        metrics = {"white_clipped_fraction":float(np.mean(np.all(values[...,:3]>.98,axis=-1)&(values[...,3]>.05))),
                   "alpha_max":float(values[...,3].max()),"rgb_standard_deviation":float(values[...,:3].std())}
        cases.append({"name":name,"image":str(source.relative_to(run)),"metrics":metrics,"render_receipt":receipt})
        print(json.dumps({"case":name,**metrics}),flush=True)
    contact.save(run/"gallery.png")
    metadata = {**identity,"cases":cases,"style":style.to_dict(),"camera":camera.to_dict(),
                "semantics":"ARTISTIC acceptance on analytical synthetic state; no claim of detector recovery or measured visible ionosphere"}
    write_json(run/"acceptance.json",metadata)
    cards = "".join(f'<figure><img src="{html.escape(case["image"])}"><figcaption>{case["name"]}</figcaption></figure>' for case in cases)
    (run/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>ShawtyNet six-state acceptance</title>'
        '<style>body{background:#080c14;color:#eef;font:16px system-ui;margin:2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}figure{margin:0}img{width:100%}figcaption{padding:1rem}</style>'
        '<h1>Six-state visual acceptance</h1><p>Analytical synthetic state → artistic mapping → identical camera. '
        'Not detector recovery or real observation validation. See acceptance.json for exact settings and metrics.</p>'
        '<div class="grid">'+cards+'</div>',encoding="utf-8")
    print(json.dumps({"gallery":str(run/"gallery.png"),"report":str(run/"index.html")}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
