"""Bounded local studio operations with explicit artistic lineage.

Browser uploads are bytes, never server filesystem paths. Animation consumes a
verified scientific run, and photographs never feed back into its measurements.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import warnings

from ophanim.core.artifacts import (
    canonical_json, file_sha256, json_value, software_identity, verify_run,
    write_json, _publish,
)
from ophanim.shawtynet.config import CameraConfig, VisualStyleConfig


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PHOTO_PIXELS = 12_000_000
MAX_ANIMATION_VOXELS = 500_000
MAX_INPUT_VOXELS = 2_000_000
ANIMATION_RESOLUTION = (640, 426)


def upload_photo(root, content):
    """Validate JPEG/PNG bytes, orient them, strip metadata, and publish PNG.

    Only the decoded RGB pixels are retained. EXIF/GPS, filenames, source paths,
    profiles and arbitrary text metadata are not copied to the published file.
    """
    return _upload_image(root, content)


def upload_mask(root, content, *, expected_size=None):
    """Publish a grayscale occlusion mask: white occludes, black reveals.

    Masks must be drawn in the original oriented photo's pixel coordinates.
    They undergo exactly the same center-crop as that photo during compositing.
    Transparent mask pixels reveal the art. Metadata is stripped.
    """
    return _upload_image(root, content, mask=True, expected_size=expected_size)


def _upload_image(root, content, *, mask=False, expected_size=None):
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not isinstance(content, bytes) or not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Photograph must contain PNG or JPEG bytes, at most 8 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content), formats=("PNG", "JPEG")) as opened:
                if opened.format not in {"PNG", "JPEG"} or getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Only a single-frame PNG or JPEG photograph is supported")
                if opened.width * opened.height > MAX_PHOTO_PIXELS:
                    raise ValueError("Photograph cannot exceed 12 megapixels")
                opened.verify()
            with Image.open(BytesIO(content), formats=("PNG", "JPEG")) as opened:
                oriented = ImageOps.exif_transpose(opened)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    flattened = Image.new("RGBA", rgba.size, (0, 0, 0, 255) if mask else (255, 255, 255, 255))
                    oriented = Image.alpha_composite(flattened, rgba)
                if expected_size is not None and tuple(oriented.size) != tuple(expected_size):
                    raise ValueError("Mask dimensions must match the original oriented photograph")
                mode = "L" if mask else "RGB"
                oriented = oriented.convert(mode)
                clean = Image.frombytes(mode, oriented.size, oriented.tobytes())
        buffer = BytesIO()
        clean.save(buffer, format="PNG")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Photograph could not be decoded safely as PNG or JPEG") from error
    normalized = buffer.getvalue()
    kind = "mask" if mask else "photograph"
    filename = "mask.png" if mask else "photo.png"
    identity = {"schema": f"ophanim-studio-{kind}/1", "pixel_png_sha256": sha256(normalized).hexdigest(),
                "width": clean.width, "height": clean.height, "normalization": "EXIF orientation applied; RGB PNG; transparency flattened on white; metadata stripped"}
    if mask:
        identity["normalization"] = "EXIF orientation applied; grayscale PNG; transparent pixels black; metadata stripped; white occludes and black reveals"
    photo_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage):
        (stage / filename).write_bytes(normalized)
        write_json(stage / ("mask_packet.json" if mask else "photo_packet.json"), identity)

    directory = _publish(Path(root), photo_id, kind, identity, build)
    return {"mask_id" if mask else "photo_id": photo_id, "directory": str(directory), "width": clean.width, "height": clean.height}


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError("Studio animation cancelled; no partial animation was published")


def _notify(callback, message):
    if callback is not None:
        callback(message)


def _animation_style(style, width_m, height_m, count):
    """Resolve a documented low-resolution preview within both memory limits."""
    style = style or VisualStyleConfig()
    if isinstance(style, dict):
        style = VisualStyleConfig.from_mapping(style)
    if not isinstance(style, VisualStyleConfig):
        raise TypeError("style must be VisualStyleConfig or a mapping")
    budget = min(MAX_ANIMATION_VOXELS, style.max_render_voxels)
    max_cells = min(style.max_render_cells, budget // count)
    if max_cells < 4:
        raise ValueError("Animation voxel budget cannot fit three or more 2×2 frames")
    spacing = max(style.render_spacing_km, 10.0)
    while (max(2, math.ceil(width_m / (spacing * 1000)) + 1)
           * max(2, math.ceil(height_m / (spacing * 1000)) + 1) > max_cells):
        spacing *= 1.25
        if spacing > 1000:
            raise ValueError("Animation region exceeds bounded preview resolution")
    return replace(style, render_spacing_km=spacing, max_render_voxels=budget,
                   max_render_cells=max_cells, lic_streamline_steps=min(style.lic_streamline_steps, 12))


def animation_run(science_directory, style=None, camera=None, *, frame_count=6,
                  blender_executable=None, cancelled=lambda: False, progress=None):
    """Export 3..12 coherent frames and optionally render a tiny artistic GIF.

    ``blender_executable=None`` deliberately exports a portable sequence only.
    Rendered output uses 640×426, eight samples, and two Blender threads. GIF
    playback uses a uniform 300ms/frame: it is not the physical event timescale.
    The selected actual UTC epochs and rendering settings are recorded.
    """
    import numpy as np
    import xarray as xr
    from PIL import Image
    from ophanim.shawtynet import export_render_sequence, map_visual_fields, render_blender

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or not 3 <= frame_count <= 12:
        raise ValueError("frame_count must be an integer from 3 through 12")
    if camera is not None and not isinstance(camera, CameraConfig):
        if not isinstance(camera, dict):
            raise TypeError("camera must be CameraConfig or a mapping")
        camera = CameraConfig(**camera)
    _cancel(cancelled)
    science = Path(science_directory).expanduser().resolve()
    parent = verify_run(science, kind="science")
    if "events.json" not in parent["files"]:
        raise ValueError("Animation requires a per-frame scientific event timeline. Reanalyze this run for animation; a single event interpretation cannot be extrapolated.")
    timeline = json.loads((science / "events.json").read_text(encoding="utf-8"))
    if timeline.get("schema_version") != "ophanim-event-timeline/1":
        raise ValueError("Unsupported event timeline; reanalyze this run")
    with xr.open_zarr(science / "dynamic.zarr", consolidated=True) as opened:
        count = opened.sizes.get("time", 0)
        if count < 3:
            raise ValueError("Animation requires at least three scientific epochs")
        source_times = opened.time.values.astype("datetime64[ns]")
        requested_times = [np.datetime64(str(frame["time"]).removesuffix("Z"), "ns") for frame in timeline.get("frames", [])]
        if len(set(requested_times)) != len(requested_times) or any(np.isnat(instant) for instant in requested_times):
            raise ValueError("Timeline contains invalid or duplicate epochs")
        available = sorted(int(np.flatnonzero(source_times == instant)[0]) for instant in requested_times if np.any(source_times == instant))
        if len(available) < 3:
            raise ValueError("Animation requires at least three exact-time scientific interpretations; reanalyze with more frame times")
        indices = [available[index] for index in np.linspace(0, len(available) - 1, min(len(available), frame_count), dtype=int)]
        if len(indices) * opened.sizes.get("x", 0) * opened.sizes.get("y", 0) > MAX_INPUT_VOXELS:
            raise ValueError("Selected scientific frames exceed the animation input memory budget")
        width = float(opened.x.values[-1] - opened.x.values[0])
        height = float(opened.y.values[-1] - opened.y.values[0])
        resolved_style = _animation_style(style, width, height, len(indices))
        times = [np.datetime_as_string(t, unit="s") + "Z" for t in opened.time.values[indices]]
    blender = None
    executable = None
    if blender_executable is not None:
        if not isinstance(blender_executable, str) or not blender_executable.strip():
            raise ValueError("Blender executable must be a nonempty string")
        executable = shutil.which(blender_executable)
        if executable is None:
            raise ValueError("Blender executable not found; sequence export is available without Blender")
        version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                 timeout=30, check=True).stdout.strip()
        blender = {"executable": str(executable), "version": version.splitlines()[0] if version else "unknown",
                   "build_sha256": sha256(version.encode()).hexdigest(), "resolution": ANIMATION_RESOLUTION,
                   "samples": 8, "threads": 2}
    identity = {"schema": "ophanim-studio-animation/1", "science_run_id": parent["run_id"],
                "science_manifest_sha256": file_sha256(science / "manifest.json"),
                "selected_frame_indices": indices, "selected_times": times,
                "event_timeline_sha256": file_sha256(science / "events.json"),
                "requested_style": json_value(style or VisualStyleConfig()),
                "resolved_style": json_value(resolved_style), "camera": json_value(camera),
                "blender": blender, "gif_frame_duration_ms": 300,
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]
    _cancel(cancelled)

    def build(stage):
        _cancel(cancelled)
        _notify(progress, f"Mapping {len(indices)} coherent low-resolution artistic frames…")
        event = json.loads((science / "event.json").read_text(encoding="utf-8"))
        with xr.open_zarr(science / "dynamic.zarr", consolidated=True) as opened:
            from ophanim.shawtynet.mapping import RENDER_INPUT_FIELDS
            dynamic = opened[[name for name in opened.data_vars if name in RENDER_INPUT_FIELDS]].isel(time=indices).load()
        # One mapping call advects the same seeded fiber field through the
        # entire selected sequence. Mapping each frame separately would flicker.
        visual = map_visual_fields(dynamic, event, resolved_style, event_timeline=timeline)
        _cancel(cancelled)
        _notify(progress, "Exporting independently replayable artistic frame packages…")
        sequence = export_render_sequence(visual, stage / "sequence", camera, cancelled=cancelled)
        rendered = []
        if executable is not None:
            for index, frame in enumerate(sequence["frames"]):
                _cancel(cancelled)
                _notify(progress, f"Rendering animation frame {index + 1}/{len(sequence['frames'])}…")
                output = stage / "frames" / f"frame_{index:06d}.png"
                render_blender(stage / "sequence" / frame["path"], output, blender_executable=executable,
                               resolution=ANIMATION_RESOLUTION, samples=8, threads=2)
                rendered.append(output)
            _cancel(cancelled)
            frames = []
            for output in rendered:
                with Image.open(output) as image:
                    rgba = image.convert("RGBA")
                    # GIF previews use a declared dark matte. Original PNGs
                    # retain their full alpha for independent compositing.
                    matte = Image.new("RGBA", rgba.size, (8, 12, 20, 255))
                    frames.append(Image.alpha_composite(matte, rgba).convert("RGB"))
            frames[0].save(stage / "animation.gif", save_all=True, append_images=frames[1:],
                           duration=300, loop=0, disposal=2)
        _cancel(cancelled)
        write_json(stage / "animation_packet.json", {
            **identity, "category": "ARTISTIC", "rendered": bool(rendered), "frame_count": len(indices),
            "visual_id": visual.attrs.get("visual_id"), "preview_matte_rgb": [8, 12, 20],
            "time_status": json.loads(visual.attrs["time_status"]),
            "semantics": "Artistic low-resolution animation; shell geometry, detail and uniform playback speed are not measurements.",
        })

    return _publish(science / "animations", run_id, "animation", identity, build)


def studio_composite_run(render_directory, photo_directory, *, opacity=0.65,
                         horizon_y=0.5, horizon_fade=0.6, saturation=0.85,
                         exposure=1.0, grade_rgb=(1.0, 1.0, 1.0),
                         foreground_mask_directory=None, cloud_mask_directory=None):
    """Center-crop an uploaded photograph to render size and blend artistically.

    Camera alignment, horizon, grade and foreground/cloud masks are manual.
    Uploaded masks use original photo coordinates, before the identical crop.
    This operation does not infer depth, segment objects, or solve camera pose.
    """
    import numpy as np
    from PIL import Image, ImageOps
    from ophanim.shawtynet.composite import composite_arrays

    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("opacity must be finite and between zero and one")
    render, photo = (Path(path).expanduser().resolve() for path in (render_directory, photo_directory))
    parent = verify_run(render, kind="render")
    photograph = verify_run(photo, kind="photograph")
    if "render.png" not in parent["files"] or "photo.png" not in photograph["files"]:
        raise ValueError("Verified render.png and photo.png are required")
    with Image.open(render / "render.png") as opened:
        if "A" not in opened.getbands():
            raise ValueError("Studio render requires an alpha channel")
        if opened.width * opened.height > MAX_PHOTO_PIXELS:
            raise ValueError("Studio composite cannot exceed 12 megapixels")
        resolution = opened.size
    with Image.open(photo / "photo.png") as opened:
        photo_size = opened.size
    masks = {}
    mask_identity = {}
    for name, directory in (("foreground_mask", foreground_mask_directory), ("cloud_mask", cloud_mask_directory)):
        if directory is None:
            continue
        path = Path(directory).expanduser().resolve()
        mask_manifest = verify_run(path, kind="mask")
        if "mask.png" not in mask_manifest["files"]:
            raise ValueError("Verified mask.png is required")
        with Image.open(path / "mask.png") as opened:
            if opened.size != photo_size:
                raise ValueError("Mask dimensions must match the original oriented photograph")
        masks[name] = path / "mask.png"
        mask_identity[name] = {"mask_id": mask_manifest["run_id"], "manifest_sha256": file_sha256(path / "manifest.json")}
    # Validate all manual controls before opening/copying a large image or
    # publishing a staging directory. The common compositor is authoritative.
    composite_options = {"horizon_y": horizon_y, "horizon_fade": horizon_fade,
                         "saturation": saturation, "exposure": exposure, "grade_rgb": grade_rgb}
    composite_arrays(np.zeros((1, 1, 3)), np.zeros((1, 1, 4)), **composite_options)
    identity = {"schema": "ophanim-studio-composite/1", "render_run_id": parent["run_id"],
                "render_manifest_sha256": file_sha256(render / "manifest.json"),
                "photograph_id": photograph["run_id"], "photo_manifest_sha256": file_sha256(photo / "manifest.json"),
                "opacity": float(opacity), "resolution": resolution,
                "photo_fit": "center crop with ImageOps.fit / Lanczos; no automatic camera alignment",
                **composite_options, "masks": mask_identity,
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]

    def build(stage):
        with Image.open(photo / "photo.png") as opened:
            if opened.width * opened.height > MAX_PHOTO_PIXELS:
                raise ValueError("Studio photograph cannot exceed 12 megapixels")
            fitted = ImageOps.fit(opened.convert("RGB"), resolution, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            background = np.asarray(fitted, dtype=float) / 255
        with Image.open(render / "render.png") as opened:
            foreground = np.asarray(opened.convert("RGBA"), dtype=float) / 255
        foreground[..., 3] *= opacity
        fitted_masks = {}
        for name, path in masks.items():
            with Image.open(path) as opened:
                fitted_masks[name] = np.asarray(ImageOps.fit(opened.convert("L"), resolution,
                    method=Image.Resampling.BILINEAR, centering=(0.5, 0.5)), dtype=float) / 255
        composite = composite_arrays(background, foreground, **composite_options, **fitted_masks)
        Image.fromarray(np.rint(composite * 255).astype(np.uint8)).save(stage / "composite.png")
        write_json(stage / "composite_packet.json", {
            **identity, "category": "ARTISTIC", "output_sha256": file_sha256(stage / "composite.png"),
            "semantics": "Manual artistic overlay on a center-cropped photograph; not visible TEC, physical reconstruction, or camera-calibrated evidence.",
        })

    return _publish(render / "composites", run_id, "composite", identity, build)
