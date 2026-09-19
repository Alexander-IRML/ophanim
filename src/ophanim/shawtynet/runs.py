"""Downstream visual, rendering and photo-compositing run publication."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from ophanim.core.artifacts import (canonical_json, dataset_sha256, file_sha256,
    json_value, software_identity, verify_run, write_json, _publish, _write_dataset)

def visualize_run(science_directory: str | Path, style: Any = None, *,
                  camera: Any = None, frame_index: int | None = None,
                  sequence: bool = False) -> Path:
    """Create an independent art recipe; single-frame previews bound memory.

    Full animation fields require explicit ``sequence=True``. The complete
    scientific time series remains immutable and available in either case.
    """
    import xarray as xr
    from ophanim.science_reports import visual_diagnostics
    from ophanim.shawtynet import VisualStyleConfig, export_render_package, map_visual_fields

    science = Path(science_directory).resolve()
    parent = verify_run(science, kind="science")
    style = style or VisualStyleConfig()
    identity = {"science_run_id": parent["run_id"],
                "science_manifest_sha256": file_sha256(science / "manifest.json"),
                "style": json_value(style), "camera": json_value(camera),
                "frame_index": frame_index, "sequence": sequence,
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage: Path) -> None:
        event = json.loads((science / "event.json").read_text())
        timeline = json.loads((science / "events.json").read_text()) if "events.json" in parent["files"] else None
        with xr.open_zarr(science / "dynamic.zarr", consolidated=True) as opened:
            selected_frame = frame_index
            if selected_frame is None:
                import numpy as np
                from ophanim.dynamics.regularize import utc64
                selected_frame = int(np.argmin(np.abs(opened.time.values - utc64(event["time"]))))
            count = opened.sizes["time"]
            if isinstance(selected_frame, bool) or not isinstance(selected_frame, int) or not -count <= selected_frame < count:
                raise ValueError("frame_index is outside the scientific sequence")
            selected_frame %= count
            dynamic = (opened if sequence else opened.isel(time=slice(selected_frame, selected_frame + 1))).load()
        export_frame = selected_frame if sequence else 0
        visual = map_visual_fields(dynamic, event, style, event_timeline=timeline)
        visual.attrs["science_run_id"] = parent["run_id"]
        _write_dataset(visual, stage / "visual.zarr")
        write_json(stage / "style.json", json_value(style))
        write_json(stage / "visual_packet.json", {
            "schema": "ophanim-visual-state/1", **identity,
            "visual_sha256": dataset_sha256(visual), "metadata": visual.attrs,
            "semantics": "All visual fields are artistic mappings of the referenced science run",
        })
        export_render_package(visual, stage / "render_package", camera, frame_index=export_frame)
        visual_diagnostics(visual, stage / "visual_diagnostics", frame_index=export_frame)

    return _publish(science / "visuals", run_id, "visual", identity, build)


def render_run(visual_directory: str | Path, *, blender_executable: str = "blender",
               samples: int = 32, resolution: tuple[int, int] = (1280, 720), threads: int = 2,
               passes: bool = False) -> Path:
    """Render an immutable exported package, preserving previous render settings."""
    from ophanim.shawtynet import render_blender

    visual = Path(visual_directory).resolve()
    manifest = verify_run(visual, kind="visual")
    executable = shutil.which(blender_executable)
    if executable is None:
        raise ValueError("Blender executable not found; pass --blender /path/to/blender")
    version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                             timeout=30, check=True).stdout.strip()
    identity = {"visual_run_id": manifest["run_id"],
                "visual_manifest_sha256": file_sha256(visual / "manifest.json"),
                "samples": samples, "resolution": resolution, "threads": threads, "passes": passes,
                "blender_executable": blender_executable,
                "blender_version": version.splitlines()[0] if version else "unknown",
                "blender_build_sha256": sha256(version.encode()).hexdigest(),
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage: Path) -> None:
        render_blender(visual / "render_package", stage / "render.png",
                       blender_executable=blender_executable, samples=samples, resolution=resolution, threads=threads, passes=passes)
        write_json(stage / "render_packet.json", identity)

    return _publish(visual / "renders", run_id, "render", identity, build)


def composite_run(render_directory: str | Path, photograph: str | Path, **options: Any) -> Path:
    """Manual-camera photo composite with input/mask checksums and a new identity."""
    from ophanim.shawtynet import composite_photograph

    render = Path(render_directory).resolve()
    parent = verify_run(render, kind="render")
    photo = Path(photograph).expanduser().resolve()
    identity = {"render_run_id": parent["run_id"], "photo_sha256": file_sha256(photo),
                "render_manifest_sha256": file_sha256(render / "manifest.json"),
                "options": json_value(options), "software": software_identity(extra_sections=("shawtynet",))}
    for name in ("foreground_mask", "cloud_mask"):
        if options.get(name) is not None:
            identity[name + "_sha256"] = file_sha256(Path(options[name]))
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage: Path) -> None:
        composite_photograph(photo, render / "render.png", stage / "composite.png", **options)
        write_json(stage / "composite_packet.json", identity)

    return _publish(render / "composites", run_id, "composite", identity, build)
