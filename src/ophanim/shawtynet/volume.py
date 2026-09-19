"""Numerical imagined volumes → bounded, portable, genuinely 3-D artwork.

This boundary accepts only the imagined-volume contract. It never turns a TEC
column into recovered altitude, nor labels its density ridges as plasma flow.
The inexpensive preview integrates the actual 3-D density along perspective
rays; Blender renders nested density isosurfaces and density-following threads.
"""
from __future__ import annotations

from hashlib import sha256
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from ophanim.core.artifacts import (
    _publish, canonical_json, file_sha256, json_value, software_identity,
    verify_run, write_json,
)
from ophanim.shawtynet.config import CameraConfig
from ophanim.shawtynet.volume_blender_scene import (
    SCHEMA, MAX_FRAME_VOXELS, MAX_MESH_FACES, MAX_PACKAGE_BYTES, STYLES,
    validate_package, validate_arrays,
)

MAX_SEQUENCE_VOXELS = 12_000_000


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError("Imagined-volume rendering cancelled; no partial run was published")


def _coordinates(dataset):
    """Renderer resource bounds supplement (not replace) the core contract."""
    import numpy as np
    from ophanim.core.volumes import VOLUME_SCHEMA
    if dataset.attrs.get("volume_schema") != VOLUME_SCHEMA or dataset.attrs.get("source_kind") != "imagined":
        raise ValueError("Volume renderer accepts explicitly imagined volumes only, never scientific measurements")
    if "density" not in dataset or dataset.density.dims != ("time", "z", "y", "x"):
        raise ValueError("Imagined volume requires density(time,z,y,x)")
    shape = dataset.density.shape
    if min(shape[1:]) < 4 or math.prod(shape[1:]) > MAX_FRAME_VOXELS or math.prod(shape) > MAX_SEQUENCE_VOXELS:
        raise ValueError("Imagined-volume render exceeds the bounded numerical grid")
    coordinates = []
    for name in ("x", "y", "z"):
        axis = np.asarray(dataset[name].values, dtype=float)
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError("Imagined volume coordinates must be finite and increasing")
        if dataset[name].attrs.get("units") not in ("m", "metres", "meters"):
            raise ValueError("Imagined volume coordinates must explicitly use metres")
        coordinates.append(axis / 1000)
    return coordinates


def _georeference(dataset):
    attrs = dataset.attrs
    geo = attrs.get("georeference", {})
    if isinstance(geo, str):
        geo = json.loads(geo)
    latitude = attrs.get("center_latitude", geo.get("center_latitude", 0))
    longitude = attrs.get("center_longitude", geo.get("center_longitude", 0))
    return float(latitude), float(longitude)


def _camera(dataset, camera, resolution=(600, 400)):
    """Artistic orbit, never a claimed ground observer or calibrated photograph."""
    import numpy as np
    axes = _coordinates(dataset)
    latitude, longitude = _georeference(dataset)
    if isinstance(camera, dict):
        camera = CameraConfig(**camera)
    camera = camera or CameraConfig(latitude, longitude, heading_deg=-35, pitch_deg=-18, horizontal_fov_deg=55)
    if not isinstance(camera, CameraConfig):
        raise TypeError("camera must be a CameraConfig or mapping")
    if (abs(camera.observer_latitude - latitude) > 1e-7
            or abs(((camera.observer_longitude - longitude + 180) % 360) - 180) > 1e-7
            or camera.observer_altitude_km != 0):
        raise ValueError("3-D imagination uses an auto-framed orbit, not a ground camera; observer coordinates must match the volume center and observer altitude must be zero")
    heading, pitch, roll = (math.radians(v) for v in (camera.heading_deg, camera.pitch_deg, camera.roll_deg))
    forward = np.array([math.sin(heading) * math.cos(pitch), math.cos(heading) * math.cos(pitch), math.sin(pitch)])
    right = np.array([math.cos(heading), -math.sin(heading), 0.0])
    up = np.cross(right, forward)
    right, up = right * math.cos(roll) + up * math.sin(roll), up * math.cos(roll) - right * math.sin(roll)
    # One occupied bounding box for the entire numerical sequence: attractive
    # framing without tracking/recentering each frame or hiding temporal drift.
    density = np.asarray(dataset.density.values)
    occupied = np.any(density > max(float(density.max()) * .085, 1e-12), axis=0)
    limits = []
    for axis_index, axis in enumerate(axes):
        active = np.flatnonzero(np.any(occupied, axis=tuple(i for i in range(3) if i != 2 - axis_index)))
        limits.append((float(axis[max(0, active[0] - 1)]), float(axis[min(len(axis) - 1, active[-1] + 1)])) if len(active) else (float(axis[0]), float(axis[-1])))
    center = np.mean(limits, axis=1)
    half = np.diff(limits, axis=1).ravel() / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    tan_x = math.tan(math.radians(camera.horizontal_fov_deg) / 2)
    tan_y = tan_x * resolution[1] / resolution[0]
    distance = max(float(np.max(np.abs(corners @ right) / tan_x - corners @ forward)),
                   float(np.max(np.abs(corners @ up) / tan_y - corners @ forward))) * 1.04
    return {"mode": "artistic auto-framed orbit; not a ground camera", "config": camera.to_dict(),
            "location_km": (center - forward * distance).tolist(), "target_km": center.tolist(),
            "forward": forward.tolist(), "right": right.tolist(), "up": up.tolist(),
            "coordinate_system": "chosen local Cartesian east/north/altitude; no recovered altitude", "aspect_ratio": resolution[0] / resolution[1],
            "framing": "fixed occupied-density bounds across the full sequence; 8.5% global density threshold", "framing_bounds_km": limits}


def _palette(position, style):
    import numpy as np
    t = np.clip(position, 0, 1)[..., None]
    low, mid, high = (np.asarray(style[name]) for name in ("low", "mid", "high"))
    return np.where(t < 0.55, low + (mid - low) * t / .55, mid + (high - mid) * (t - .55) / .45)


def volume_preview(dataset, *, frame_index=0, style="luminous", camera=None, resolution=(600, 400), density_scale=None):
    """Perspective emission/absorption integration of numerical voxels, not physics.

    Sampling and transfer opacity are artistic. This preview is a real 3-D
    projection, but does not pretend to be the identical Blender mesh material.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates, gaussian_filter
    if style not in STYLES:
        raise ValueError("Unknown volume style")
    if len(resolution) != 2 or any(type(v) is not int or not 32 <= v <= 1000 for v in resolution) or math.prod(resolution) > 600_000:
        raise ValueError("Volume preview must be bounded to 600,000 pixels")
    axes = _coordinates(dataset)
    density = np.asarray(dataset.density.isel(time=frame_index).values, dtype=np.float32)
    if not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("Imagined density must be finite and nonnegative")
    scale = float(density_scale if density_scale is not None else max(np.max(density), 1e-12))
    density = density / max(scale, 1e-12)
    view = _camera(dataset, camera, resolution)
    eye, forward, right, up = (np.asarray(view[key]) for key in ("location_km", "forward", "right", "up"))
    width, height = resolution
    xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(1, -1, height))
    tx = math.tan(math.radians(view["config"]["horizontal_fov_deg"]) / 2)
    rays = forward + xx[..., None] * tx * right + yy[..., None] * tx * height / width * up
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lower, upper = np.array([axis[0] for axis in axes]), np.array([axis[-1] for axis in axes])
    safe_rays = np.where(np.abs(rays) < 1e-10, 1e-10, rays)
    near_planes, far_planes = (lower - eye) / safe_rays, (upper - eye) / safe_rays
    near = np.maximum(np.min(np.stack((near_planes, far_planes)), axis=0).max(axis=-1), 0)
    far = np.max(np.stack((near_planes, far_planes)), axis=0).min(axis=-1)
    valid = far > near
    span = np.maximum(far - near, 0)
    color, alpha = np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32)
    material = STYLES[style]
    steps = 112
    for index in range(steps):
        distance = near + (index + .5) / steps * span
        xyz = eye + rays * distance[..., None]
        ijk = [(xyz[..., i] - axes[i][0]) / (axes[i][-1] - axes[i][0]) * (len(axes[i]) - 1) for i in (2, 1, 0)]
        value = map_coordinates(density, ijk, order=1, mode="constant", cval=0, prefilter=False)
        # Narrow transfer bands reveal folded inner density layers, while
        # weak continuous veil makes their 3-D support visible.
        outer = .35 * np.exp(-((value - .18) / .036) ** 2)
        middle = .72 * np.exp(-((value - .40) / .044) ** 2)
        inner = 1.05 * np.exp(-((value - .68) / .055) ** 2)
        bands = outer + middle + inner
        opacity = (1 - np.exp(-(bands + .08 * value) * span / (steps * max(np.ptp(axes[2]), 1)) * 2.8)) * valid * material["opacity"]
        height_fraction = (xyz[..., 2] - lower[2]) / max(upper[2] - lower[2], 1e-12)
        color_position = (outer * .07 + middle * .51 + inner * .92) / np.maximum(bands, 1e-12)
        rgb = _palette(color_position + .12 * (height_fraction - .5), material)
        brightness = .55 + .8 * value
        weight = (1 - alpha) * opacity
        color += weight[..., None] * rgb * brightness[..., None]
        alpha += weight
    # Restrained linear-light glow, never clipped-white entire volumes.
    glow = gaussian_filter(color, sigma=(2.5, 2.5, 0)) * .16
    rgb = 1 - np.exp(-(color + glow) * material["emission"] * 1.65)
    rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)
    # Preview matte is explicit; full transparent Blender PNG remains the
    # independent asset for compositing.
    matte = np.asarray([.019, .029, .055])
    rgb = np.clip(rgb + matte * (1 - alpha[..., None]), 0, 1)
    return np.rint(np.dstack((rgb, np.ones_like(alpha))) * 255).astype(np.uint8)


def _density_threads(density, axes, *, seed, count, scale):
    """Trace bright density ridges, not physical velocity streamlines.

    Deterministic seeds and fixed cross-plane coordinates avoid fresh random
    geometry each frame. A bounded local density ascent follows the evolving
    3-D field; topology can still change when the numerical field changes.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    rng = np.random.default_rng(seed)
    # Choose the widest horizontal dimension as the traversal direction.
    swapped = np.ptp(axes[1]) > np.ptp(axes[0])
    field = density.transpose(0, 2, 1) if swapped else density
    mid = field.shape[2] // 2
    plane = field[:, :, mid]
    total = float(plane.sum())
    if total <= 1e-12:
        return np.empty((0, 3), np.float32), np.array([0], np.int32), np.empty(0, np.float32)
    # Fixed quantiles of the cumulative density yield stable, reproducible
    # seeds; their positions respond to the actual current numerical field.
    probability = plane.ravel() / total
    seeds = np.searchsorted(np.cumsum(probability), (np.arange(count) + rng.uniform(.2, .8, count)) / count)
    curves, weights, offsets = [], [], [0]
    offsets2 = np.array([(z, y) for z in range(-2, 3) for y in range(-2, 3)])
    for seed_index in seeds:
        start = np.array(np.unravel_index(min(seed_index, plane.size - 1), plane.shape), dtype=float)
        paths = []
        for direction in (-1, 1):
            yz, path = start.copy(), []
            for along in range(mid, -1 if direction < 0 else field.shape[2], direction):
                candidates = yz + offsets2 * .6
                candidates[:, 0] = np.clip(candidates[:, 0], 0, field.shape[0] - 1)
                candidates[:, 1] = np.clip(candidates[:, 1], 0, field.shape[1] - 1)
                values = map_coordinates(field[:, :, along], candidates.T, order=1, mode="nearest") / scale
                best = int(np.argmax(values - .038 * np.sum((candidates - yz) ** 2, axis=1)))
                yz = candidates[best]
                if values[best] < .09:
                    break
                path.append((along, yz[1], yz[0], float(values[best])))
            paths.append(path)
        joined = paths[0][::-1] + paths[1][1:]
        if len(joined) < 6:
            continue
        points = np.asarray(joined)
        xyz_indices = points[:, [1, 0, 2]] if swapped else points[:, :3]
        xyz = np.column_stack([np.interp(xyz_indices[:, i], np.arange(len(axes[i])), axes[i]) for i in range(3)])
        xyz = gaussian_filter1d(xyz, sigma=.7, axis=0, mode="nearest")
        curves.extend(xyz.tolist())
        weights.extend(points[:, 3].tolist())
        offsets.append(len(curves))
    return np.asarray(curves, np.float32).reshape(-1, 3), np.asarray(offsets, np.int32), np.asarray(weights, np.float32)


def export_volume_package(dataset, destination, *, frame_index=0, style="luminous", camera=None, density_scale=None, seed=17, preview=True):
    """Portable arrays/JSON plus installed trusted Blender script; no pickles."""
    import numpy as np
    from PIL import Image
    from skimage.measure import marching_cubes
    if style not in STYLES:
        raise ValueError("Volume style must be ghost, luminous or quiet")
    axes = _coordinates(dataset)
    count = dataset.sizes["time"]
    if type(frame_index) is not int or not 0 <= frame_index < count:
        raise ValueError("frame_index is outside the imagined volume")
    density = np.asarray(dataset.density.isel(time=frame_index).values, np.float32)
    if not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("Imagined density must be finite and nonnegative")
    scale = float(density_scale if density_scale is not None else max(float(density.max()), 1e-12))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("density_scale must be finite and positive")
    normalized = density / scale
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    meshes, total_faces = [], 0
    for index, threshold in enumerate((.18, .40, .68)):
        if not float(normalized.min()) < threshold < float(normalized.max()):
            continue
        # Fixed step size for a sequence, not per-frame adaptive topology.
        step = max(1, int(math.ceil(max(density.shape) / 70)))
        vertices_zyx, faces, _, _ = marching_cubes(normalized, level=threshold, step_size=step, allow_degenerate=False)
        vertices = np.column_stack([np.interp(vertices_zyx[:, 2 - i], np.arange(len(axes[i])), axes[i]) for i in range(3)])
        total_faces += len(faces)
        if total_faces > MAX_MESH_FACES:
            raise ValueError("Volume isosurface complexity exceeds the bounded mesh budget")
        colors = _palette((.07, .51, .92)[index] + .16 * ((vertices[:, 2] - axes[2][0]) / np.ptp(axes[2]) - .5), STYLES[style])
        filename = f"isosurface_{index}.npz"
        np.savez_compressed(root / filename, vertices=vertices.astype(np.float32), faces=faces.astype(np.int32), colors=colors.astype(np.float32))
        meshes.append({"file": filename, "density_fraction": threshold, "vertices": len(vertices), "faces": len(faces), "layer": index})
    points, offsets, weights = _density_threads(density, axes, seed=seed, count=STYLES[style]["threads"], scale=scale)
    np.savez_compressed(root / "threads.npz", points=points, offsets=offsets, weights=weights)
    # Numeric source subset lets independent tools reproduce or remap the
    # chosen isosurfaces without running an untrusted Python object.
    np.savez_compressed(root / "volume.npz", density=density, x_km=axes[0], y_km=axes[1], z_km=axes[2])
    if preview:
        Image.fromarray(volume_preview(dataset, frame_index=frame_index, style=style, camera=camera, density_scale=scale)).save(root / "preview.png")
    source = Path(__file__).with_name("volume_blender_scene.py")
    shutil.copyfile(source, root / source.name)
    metadata = {"schema_version": SCHEMA, "category": "IMAGINED", "style": style,
                "material": STYLES[style], "frame_index": frame_index, "time": str(dataset.time.values[frame_index]),
                "camera": _camera(dataset, camera), "meshes": meshes, "threads": "threads.npz",
                "volume": "volume.npz", "density_scale": scale, "seed": int(seed),
                "bounds_km": [[float(axis[0]), float(axis[-1])] for axis in axes],
                "thread_count": len(offsets) - 1, "total_mesh_faces": total_faces,
                "semantic": "Entire 3-D density, chosen altitude, isosurfaces, ridge-following threads, color and radiance are imagined artistic constructions. Threads are not plasma trajectories; no volumetric reconstruction or fluid solution is claimed.",
                "preview_semantics": "Perspective ray integration of imagined numerical density, on a dark matte; approximate artistic transfer function, not the Blender surface material.",
                "files": {p.name: file_sha256(p) for p in sorted(root.iterdir()) if p.is_file()}}
    metadata["render_package_id"] = sha256(canonical_json(metadata)).hexdigest()
    write_json(root / "scene_metadata.json", metadata)
    return metadata


def _verify_volume_package(directory):
    """Fixed plain-data filenames, bounded inventory, trusted script identity."""
    return validate_package(directory)[0]


def _validate_arrays(root, metadata):
    """Reject pickles, zip bombs, malformed indices and nonfinite geometry."""
    return validate_arrays(Path(root), metadata)


def _read_hypothesis(directory):
    from ophanim.core.volumes import read_volume
    root = Path(directory).resolve()
    parent = verify_run(root, kind="hypothesis")
    volume = read_volume(root)
    _coordinates(volume.dataset)
    return root, parent, volume


def visualize_volume_run(volume_directory, style="luminous", *, camera=None, time_index=None, cancelled=None):
    """Publish a 3-D volume preview and independently replayable geometry."""
    import numpy as np
    root, parent, volume = _read_hypothesis(volume_directory)
    if style not in STYLES:
        raise ValueError("Unknown imagined-volume style")
    _cancel(cancelled)
    dataset = volume.dataset
    if time_index is None:
        time_index = dataset.sizes["time"] // 2
    if type(time_index) is not int or not 0 <= time_index < dataset.sizes["time"]:
        raise ValueError("time_index is outside the imagined sequence")
    view = _camera(dataset, camera)
    # One normalization for the entire evolution, not an independently
    # brightened frame that erases genuine numerical amplitude changes.
    scale = max(float(np.max(dataset.density.values)), 1e-12)
    seed = int(volume.recipe["parameters"]["seed"])
    identity = {"schema": SCHEMA, "hypothesis_run_id": parent["run_id"], "volume_source": "source_hypothesis",
                "hypothesis_manifest_sha256": file_sha256(root / "manifest.json"), "style": style,
                "camera": view["config"], "frame_index": time_index, "density_scale": scale,
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]
    def build(stage):
        _cancel(cancelled)
        # Copy only the verified upstream artifact inventory, not nested art
        # runs. A moved/downloaded visual remains independently animatable.
        source_copy = stage / "source_hypothesis"
        source_copy.mkdir()
        for relative in [*parent["files"], "manifest.json"]:
            destination = source_copy / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        package = export_volume_package(dataset, stage / "render_package", frame_index=time_index, style=style, camera=camera, density_scale=scale, seed=seed)
        _cancel(cancelled)
        write_json(stage / "style.json", {"style": style, "camera": view, "category": "IMAGINED"})
        write_json(stage / "visual_packet.json", {**identity, "category": "IMAGINED", "render_package_id": package["render_package_id"], "semantics": package["semantic"]})
        write_json(stage / "hypothesis.json", volume.recipe)
        write_json(stage / "volume_contract.json", {"provenance": volume.provenance})
        diagnostic = stage / "visual_diagnostics"
        diagnostic.mkdir()
        shutil.copyfile(stage / "render_package" / "preview.png", diagnostic / "preview.png")
        diagnostic.joinpath("report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>Imagined 3-D volume</title><style>body{max-width:900px;margin:3rem auto;background:#090f1b;color:#dbe9f5;font:17px/1.6 system-ui;padding:1rem}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Imagined 3-D structure</h1><p>" + html.escape(package["semantic"]) + "</p><img alt='Perspective numerical-volume preview' src='preview.png'><p>" + html.escape(package["preview_semantics"]) + "</p><h2>Reproducible recipe and chosen assumptions</h2><pre>" + html.escape(json.dumps(volume.recipe, indent=2)) + "</pre></html>", encoding="utf-8")
    return _publish(root / "visuals", run_id, "visual", identity, build)


def render_volume_blender(package_directory, output_path, *, blender_executable="blender", samples=24, resolution=(960, 640), threads=2, timeout_seconds=900, cancelled=None):
    from ophanim.shawtynet.render import _blender_argument_path
    root, output = Path(package_directory).resolve(), Path(output_path).resolve()
    metadata = _verify_volume_package(root)
    if output.exists() or output.suffix.lower() != ".png":
        raise ValueError("Volume rendering needs a new PNG output")
    if type(samples) is not int or not 1 <= samples <= 128 or type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Volume rendering supports 1..128 samples and 1..8 threads")
    if len(resolution) != 2 or any(type(v) is not int or not 64 <= v <= 2400 for v in resolution) or math.prod(resolution) > 4_000_000:
        raise ValueError("Volume rendering supports at most four megapixels")
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 3600:
        raise ValueError("Invalid volume rendering timeout")
    executable = shutil.which(blender_executable)
    if executable is None:
        raise ValueError("Blender is not installed; the numerical-volume preview and portable package remain available")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, "--background", "--factory-startup", "--threads", str(threads), "--python-exit-code", "1", "--python",
               _blender_argument_path(root / "volume_blender_scene.py", executable), "--", "--package", _blender_argument_path(root, executable),
               "--output", _blender_argument_path(output, executable), "--samples", str(samples), "--width", str(resolution[0]), "--height", str(resolution[1])]
    _cancel(cancelled)
    log = output.with_suffix(".blender.log")
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while child.poll() is None:
                _cancel(cancelled)
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("Volume renderer exceeded bounded runtime")
                time.sleep(.2)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            raise
    if child.returncode != 0 or not output.is_file():
        raise RuntimeError(f"Blender did not complete imagined-volume rendering; see {log}")
    _cancel(cancelled)
    write_json(output.with_suffix(".render.json"), {"schema": SCHEMA, "render_package_id": metadata["render_package_id"],
        "output_sha256": file_sha256(output), "samples": samples, "resolution": resolution, "threads": threads,
        "semantic": metadata["semantic"]})
    return output


def _blender_identity(executable):
    path = shutil.which(executable)
    if path is None:
        raise ValueError("Blender executable not found")
    version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    return {"executable": executable, "version": version.splitlines()[0] if version else "unknown", "build_sha256": sha256(version.encode()).hexdigest()}


def render_volume_run(visual_directory, *, blender_executable="blender", samples=24, resolution=(960, 640), threads=2, cancelled=None):
    visual = Path(visual_directory).resolve()
    parent = verify_run(visual, kind="visual")
    _verify_volume_package(visual / "render_package")
    identity = {"schema": SCHEMA, "visual_run_id": parent["run_id"], "visual_manifest_sha256": file_sha256(visual / "manifest.json"),
                "blender": _blender_identity(blender_executable), "samples": samples, "resolution": resolution, "threads": threads,
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]
    def build(stage):
        render_volume_blender(visual / "render_package", stage / "render.png", blender_executable=blender_executable,
                              samples=samples, resolution=resolution, threads=threads, cancelled=cancelled)
        write_json(stage / "render_packet.json", identity)
    return _publish(visual / "renders", run_id, "render", identity, build)


def animation_volume_run(visual_directory, *, frame_count=6, blender_executable=None, cancelled=None, progress=None):
    import numpy as np
    from PIL import Image
    if type(frame_count) is not int or not 3 <= frame_count <= 12:
        raise ValueError("Volume animation requires 3..12 frames")
    visual = Path(visual_directory).resolve()
    parent = verify_run(visual, kind="visual")
    packet = json.loads((visual / "visual_packet.json").read_text(encoding="utf-8"))
    if packet.get("schema") != SCHEMA:
        raise ValueError("Expected an imagined-volume visual run")
    if packet.get("volume_source") != "source_hypothesis":
        raise ValueError("Volume visual needs its portable source hypothesis; re-export this older visual")
    hypothesis, source, volume = _read_hypothesis(visual / "source_hypothesis")
    if source["run_id"] != packet["hypothesis_run_id"] or file_sha256(hypothesis / "manifest.json") != packet["hypothesis_manifest_sha256"]:
        raise ValueError("Imagined-volume lineage no longer matches")
    count = volume.dataset.sizes["time"]
    if count < 3:
        raise ValueError("Volume animation requires at least three numerical frames")
    indices = np.linspace(0, count - 1, min(frame_count, count), dtype=int).tolist()
    identity = {"schema": SCHEMA, "visual_run_id": parent["run_id"], "visual_manifest_sha256": file_sha256(visual / "manifest.json"),
                "frame_indices": indices, "style": packet["style"], "camera": packet["camera"],
                "blender": _blender_identity(blender_executable) if blender_executable else None,
                "resolution": (640, 426), "samples": 12, "threads": 2, "gif_frame_duration_ms": 180,
                "software": software_identity(extra_sections=("shawtynet",))}
    run_id = sha256(canonical_json(identity)).hexdigest()[:24]
    def build(stage):
        frames, rendered = [], []
        sequence = stage / "sequence"
        sequence.mkdir()
        for index, frame in enumerate(indices):
            _cancel(cancelled)
            if progress:
                progress(f"Extracting imagined 3-D frame {index + 1}/{len(indices)}…")
            package_path = sequence / f"frame_{index:04d}"
            package = export_volume_package(volume.dataset, package_path, frame_index=frame, style=packet["style"],
                camera=packet["camera"], density_scale=packet["density_scale"], seed=int(volume.recipe["parameters"]["seed"]), preview=False)
            frames.append({"path": package_path.name, "source_frame": frame, "time": package["time"], "render_package_id": package["render_package_id"]})
            if blender_executable:
                if progress:
                    progress(f"Rendering imagined 3-D frame {index + 1}/{len(indices)}…")
                output = stage / "frames" / f"frame_{index:04d}.png"
                render_volume_blender(package_path, output, blender_executable=blender_executable, resolution=(640, 426), samples=12, threads=2, cancelled=cancelled)
                with Image.open(output) as opened:
                    rgba = opened.convert("RGBA")
                    matte = Image.new("RGBA", rgba.size, (5, 8, 15, 255))
                    rendered.append(Image.alpha_composite(matte, rgba).convert("RGB"))
        _cancel(cancelled)
        write_json(sequence / "sequence.json", {"schema_version": SCHEMA, "frames": frames})
        if rendered:
            rendered[0].save(stage / "animation.gif", save_all=True, append_images=rendered[1:], duration=180, loop=0, disposal=2)
        write_json(stage / "animation_packet.json", {**identity, "rendered": bool(rendered), "frame_count": len(frames),
            "category": "IMAGINED", "selected_times": [frame["time"] for frame in frames],
            "semantics": "Actual evolution of the chosen imagined numerical 3-D density, not recovered plasma motion. Uniform GIF playback is artistic, not elapsed physical time."})
    return _publish(visual / "animations", run_id, "animation", identity, build)
