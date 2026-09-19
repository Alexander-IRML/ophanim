"""Portable shell/texture export and isolated Blender rendering boundary."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ophanim.shawtynet.config import CameraConfig
from ophanim.shawtynet.mapping import canonical_json


def geodetic_to_ecef(latitude_degrees, longitude_degrees, altitude_km):
    """WGS84 geodetic position in kilometres; artistic altitude is explicit."""
    import numpy as np

    lat, lon = np.radians(latitude_degrees), np.radians(longitude_degrees)
    a, flattening = 6378.137, 1 / 298.257223563
    eccentricity_squared = flattening * (2 - flattening)
    normal = a / np.sqrt(1 - eccentricity_squared * np.sin(lat) ** 2)
    altitude = np.asarray(altitude_km)
    return np.stack((
        (normal + altitude) * np.cos(lat) * np.cos(lon),
        (normal + altitude) * np.cos(lat) * np.sin(lon),
        (normal * (1 - eccentricity_squared) + altitude) * np.sin(lat),
    ), axis=-1)


def ecef_to_observer_enu(points, camera: CameraConfig):
    """Observer-relative east/north/up positions; one Blender unit is 1 km."""
    import numpy as np

    origin = geodetic_to_ecef(camera.observer_latitude, camera.observer_longitude, camera.observer_altitude_km)
    lat, lon = np.radians(camera.observer_latitude), np.radians(camera.observer_longitude)
    transform = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])
    return (np.asarray(points) - origin) @ transform.T


def shell_mesh(latitudes, longitudes, displacement_m, *, altitude_km: float, camera: CameraConfig):
    """Triangulate a curved artistic shell, retaining metric Earth curvature."""
    import numpy as np

    lat, lon = np.asarray(latitudes, dtype=float), np.asarray(longitudes, dtype=float)
    height = np.asarray(displacement_m, dtype=float)
    if lat.ndim != 2 or lat.shape != lon.shape or lat.shape != height.shape or min(lat.shape) < 2:
        raise ValueError("shell geometry needs matching 2D lat/lon/displacement arrays")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(height)):
        raise ValueError("shell coordinates and displacement must be finite")
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError("invalid geodetic shell coordinates")
    heights = altitude_km + height / 1000
    if np.any(heights <= 0):
        raise ValueError("artistic shell must stay above the ellipsoid")
    points = geodetic_to_ecef(lat, lon, heights)
    vertices = ecef_to_observer_enu(points, camera)
    points_above = geodetic_to_ecef(lat, lon, heights + 1)
    normals = ecef_to_observer_enu(points_above, camera) - vertices
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    rows, columns = lat.shape
    south_west = (np.arange(rows - 1)[:, None] * columns + np.arange(columns - 1)[None, :]).reshape(-1)
    faces = np.concatenate((
        np.stack((south_west, south_west + 1, south_west + columns + 1), axis=1),
        np.stack((south_west, south_west + columns + 1, south_west + columns), axis=1),
    ))
    uu, vv = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
    return {
        "vertices": vertices.reshape(-1, 3).astype(np.float32),
        "faces": faces.astype(np.int32),
        "uv": np.stack((uu, vv), axis=-1).reshape(-1, 2).astype(np.float32),
        "normals": normals.reshape(-1, 3).astype(np.float32),
    }


def _camera_for(visual, camera: CameraConfig | None) -> CameraConfig:
    import numpy as np

    if "lat" not in visual or "lon" not in visual:
        raise ValueError("render export requires 2D lat/lon coordinates")
    if camera is not None:
        if not isinstance(camera, CameraConfig):
            raise TypeError("camera must be a CameraConfig")
        return camera
    longitude = np.radians(np.asarray(visual.lon.values))
    mean_lon = float(np.degrees(np.arctan2(np.mean(np.sin(longitude)), np.mean(np.cos(longitude)))))
    return CameraConfig(float(np.mean(visual.lat.values)), mean_lon, pitch_deg=90)


def _frame_values(visual, name: str, frame_index: int):
    import numpy as np

    values = visual[name]
    if "time" in values.dims:
        values = values.isel(time=frame_index)
    return np.asarray(values.transpose("y", "x").values, dtype=np.float32)


def export_render_package(
    visual, destination: str | Path, camera: CameraConfig | None = None, *, frame_index: int = 0,
) -> dict[str, Any]:
    """Write a new portable render package; existing directories are refused.

    NPY textures preserve physical artistic control values. PNG textures are
    normalized shader previews with their exact scale/offset in the manifest.
    Blender consumes only this package, never xarray or the observation store.
    """
    import numpy as np
    from PIL import Image

    count = visual.sizes.get("time", 0)
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or not -count <= frame_index < count:
        raise ValueError("frame_index is outside the visual sequence")
    frame_index %= count
    required = {"height_displacement", "emission", "base_opacity", "hue_driver", "flow_texture", "flow_texture_strength"}
    if required - set(visual.data_vars):
        raise ValueError(f"visual fields missing: {sorted(required - set(visual.data_vars))}")
    camera = _camera_for(visual, camera)
    root = Path(destination)
    if root.exists():
        raise FileExistsError(f"render package already exists: {root}")
    altitude = float(visual.attrs.get("shell_altitude_km", 300))
    if not math.isfinite(altitude) or altitude <= 0:
        raise ValueError("invalid artistic shell altitude")
    geometry = shell_mesh(
        visual.lat.values, visual.lon.values, _frame_values(visual, "height_displacement", frame_index),
        altitude_km=altitude, camera=camera,
    )
    root.mkdir(parents=True)
    texture_root = root / "textures"
    texture_root.mkdir()
    np.savez_compressed(root / "mesh.npz", **geometry)
    texture_manifest = {}
    for name in sorted(visual.data_vars):
        if set(visual[name].dims) not in ({"time", "y", "x"}, {"y", "x"}):
            continue
        values = _frame_values(visual, name, frame_index)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"visual texture {name} contains nonfinite values")
        np.save(texture_root / f"{name}.npy", values, allow_pickle=False)
        minimum, maximum = float(np.min(values)), float(np.max(values))
        if name in ("wave_modulation", "hue_driver", "front_fold", "front_fold_displacement", "procedural_detail"):
            offset, scale = -1.0, 2.0
        elif name in ("height_displacement", "emission"):
            offset, scale = min(0.0, minimum), max(1e-12, maximum - min(0.0, minimum))
        else:
            offset, scale = 0.0, max(1.0, maximum)
        normalized = np.clip((values - offset) / scale, 0, 1)
        pixels = np.rint(np.flipud(normalized) * 65535).astype(np.uint16)
        Image.fromarray(pixels).save(texture_root / f"{name}.png")
        texture_manifest[name] = {
            "npy": f"textures/{name}.npy", "png": f"textures/{name}.png",
            "png_offset": offset, "png_scale": scale,
            "decode": "original = normalized_png * png_scale + png_offset",
            "semantic": visual[name].attrs.get("semantic", "artistic control field"),
            "units": visual[name].attrs.get("units", "1"),
        }
    provenance = json.loads(visual.attrs.get("visual_provenance", "{}"))
    style = provenance.get("style", {})
    positive = np.asarray(style.get("positive_color", (0.70, 0.80, 1)))
    negative = np.asarray(style.get("negative_color", (0.95, 0.60, 0.48)))
    hue = np.clip((_frame_values(visual, "hue_driver", frame_index) + 1) / 2, 0, 1)
    rgb = negative[None, None, :] * (1 - hue[..., None]) + positive[None, None, :] * hue[..., None]
    Image.fromarray(np.rint(np.clip(np.flipud(rgb), 0, 1) * 255).astype(np.uint8)).save(texture_root / "color.png")
    # A flat material preview is explicitly not camera/geometry rendering, but
    # does expose opacity, confident fibers, fraying and tone mapping instead
    # of presenting a hue swatch as though it were the finished material.
    preview = material_preview(visual, frame_index=frame_index)
    Image.fromarray(np.rint(np.flipud(preview) * 255).astype(np.uint8)).save(root / "preview.png")
    script_source = Path(__file__).with_name("blender_scene.py")
    shutil.copyfile(script_source, root / "blender_scene.py")
    manifest = {
        "schema_version": "ophanim-render/1", "visual_id": visual.attrs.get("visual_id"),
        "frame_index": frame_index, "time": str(visual.time.values[frame_index]),
        "geometry": "mesh.npz", "coordinate_system": "observer-local east/north/up",
        "units": "kilometres; one Blender unit equals one kilometre",
        "earth_model": "WGS84 geodetic ellipsoid", "camera": camera.to_dict(),
        "shell_altitude_km": altitude, "textures": texture_manifest,
        "color_texture": "textures/color.png", "visual_provenance": provenance,
        "material_preview": "preview.png",
        "material_preview_semantics": "Orthographic flat material preview with alpha, not a camera-calibrated shell render",
        "semantic": "Entire shell geometry and shader controls are artistic; altitude and vertical displacement are not measured.",
        "texture_origin": "PNG top row is north; mesh UV v=0 is south",
        "files": {},
    }
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest["files"][path.relative_to(root).as_posix()] = sha256(path.read_bytes()).hexdigest()
    manifest["render_package_id"] = sha256(canonical_json(manifest).encode()).hexdigest()
    (root / "scene_metadata.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def material_preview(visual, *, frame_index=0):
    """Return straight-alpha sRGB artistic material RGBA, without geometry.

    Mirrors the two-layer shader in linear light; useful for low-cost
    acceptance tests and previewing missing confidence/art channels.
    """
    import numpy as np
    from ophanim.shawtynet.composite import _linear_to_srgb, _srgb_to_linear

    style = json.loads(visual.attrs.get("visual_provenance", "{}")).get("style", {})
    hue = np.clip((_frame_values(visual, "hue_driver", frame_index) + 1) / 2, 0, 1)
    positive = np.asarray(style.get("positive_color", (0.70, 0.80, 1)))
    negative = np.asarray(style.get("negative_color", (0.95, 0.60, 0.48)))
    color = _srgb_to_linear(negative * (1 - hue[..., None]) + positive * hue[..., None])
    field = lambda name: _frame_values(visual, name, frame_index)
    emission = field("display_emission") if "display_emission" in visual else field("emission") / (1 + field("emission"))
    breakup = field("breakup") if "breakup" in visual else 0
    base_alpha = np.clip(field("base_opacity") * (1 - breakup), 0, 1)
    detail_alpha = field("flow_texture") * field("flow_texture_strength")
    if "ribbon_texture" in visual:
        detail_alpha += field("ribbon_texture") * field("ribbon_strength")
    detail_alpha = np.clip(detail_alpha * (1 - breakup), 0, 1)
    alpha = detail_alpha + base_alpha * (1 - detail_alpha)
    detail_emission = np.minimum(0.95, emission * style.get("detail_emission_gain", 2.0))
    radiance = detail_emission * detail_alpha + np.minimum(0.95, emission) * base_alpha * (1 - detail_alpha)
    straight = np.divide(radiance, alpha, out=np.zeros_like(radiance), where=alpha > 0)
    rgb = np.clip(_linear_to_srgb(color * straight[..., None]), 0, 1)
    return np.concatenate((rgb, alpha[..., None]), axis=-1).astype(np.float32)


def export_render_sequence(visual, destination: str | Path, camera: CameraConfig | None = None, *, cancelled=None) -> dict[str, Any]:
    """Export independently replayable mesh/texture packages for all times."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    camera = _camera_for(visual, camera)
    frames = []
    for index in range(visual.sizes.get("time", 0)):
        if cancelled is not None and cancelled():
            raise InterruptedError("Animation export cancelled")
        relative = f"frame_{index:06d}"
        manifest = export_render_package(visual, root / relative, camera, frame_index=index)
        frames.append({"path": relative, "time": manifest["time"], "render_package_id": manifest["render_package_id"]})
    if not frames:
        raise ValueError("cannot export an empty visual sequence")
    sequence = {"schema_version": "ophanim-render-sequence/1", "visual_id": visual.attrs.get("visual_id"), "frames": frames}
    (root / "sequence.json").write_text(canonical_json(sequence) + "\n", encoding="utf-8")
    return sequence


def _verify_package(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "scene_metadata.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ophanim-render/1":
        raise ValueError("unsupported render package schema")
    expected_id = manifest.get("render_package_id")
    unsigned = {key: value for key, value in manifest.items() if key != "render_package_id"}
    if sha256(canonical_json(unsigned).encode()).hexdigest() != expected_id:
        raise ValueError("render package metadata checksum mismatch")
    resolved = root.resolve()
    for relative, checksum in manifest.get("files", {}).items():
        file = (root / relative).resolve()
        if not file.is_relative_to(resolved) or not file.is_file():
            raise ValueError("render package references an invalid file")
        if sha256(file.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"render package file checksum mismatch: {relative}")
    if "blender_scene.py" not in manifest.get("files", {}):
        raise ValueError("render package has no verified Blender script")
    # Only an exact copy of the application-owned script is executable. A
    # self-consistent package is not permission to run arbitrary imported code.
    if sha256((root / "blender_scene.py").read_bytes()).hexdigest() != sha256(Path(__file__).with_name("blender_scene.py").read_bytes()).hexdigest():
        raise ValueError("render package uses a different Blender script version; re-export its saved visual state with this application before rendering")
    return manifest


def _exr_pass_channels(path: Path) -> list[str]:
    """Read bounded OpenEXR headers, without decoding pixels or loading an SDK.

    Blender 5 writes multipart EXR, older releases may use a single header.
    Reject malformed/oversized headers instead of trusting a filename as proof
    that the requested emission/depth/coverage passes were actually exported.
    """
    from io import BytesIO
    import struct

    limit = 1024 * 1024
    with Path(path).open("rb") as source:
        stream = BytesIO(source.read(limit))

    def take(count):
        if count < 0 or count > limit:
            raise ValueError("EXR header exceeds bounded inspection size")
        data = stream.read(count)
        if len(data) != count:
            raise ValueError("Truncated or oversized EXR header")
        return data

    def cstring(source=stream):
        value = bytearray()
        for _ in range(256):
            character = source.read(1)
            if not character:
                raise ValueError("Truncated EXR header string")
            if character == b"\0":
                try:
                    return value.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError("Invalid EXR header string") from error
            value.extend(character)
        raise ValueError("EXR header name exceeds 255 bytes")

    magic, version = struct.unpack("<II", take(8))
    if magic != 20000630 or version & 255 != 2:
        raise ValueError("Unsupported OpenEXR header")
    multipart = bool(version & 0x1000)
    channels = []
    for _ in range(64):
        name = cstring()
        if not name:
            break
        for _ in range(256):
            kind = cstring()
            size = struct.unpack("<I", take(4))[0]
            data = take(size)
            if kind == "chlist":
                sub = BytesIO(data)
                for _ in range(256):
                    channel = cstring(sub)
                    if not channel:
                        break
                    if len(sub.read(16)) != 16:
                        raise ValueError("Truncated EXR channel definition")
                    channels.append(channel)
                else:
                    raise ValueError("EXR has too many channels")
            name = cstring()
            if not name:
                break
        else:
            raise ValueError("EXR has too many header attributes")
        if not multipart:
            break
    else:
        raise ValueError("EXR has too many parts")
    if not channels:
        raise ValueError("EXR contains no channel definitions")
    return sorted(set(channels))


def render_blender(
    package_directory: str | Path, output_path: str | Path, *,
    blender_executable: str = "blender", samples: int = 32,
    resolution: tuple[int, int] = (1280, 720), timeout_seconds: float = 3600,
    threads: int = 2, passes: bool = False,
) -> Path:
    """Render a verified exported package in a separate Blender process.

    Science computation is complete before this function runs. No Blender API
    or library is imported into the scientific Python process.
    The packaged script must match the installed application exactly. Old
    packages retain their assets but require re-export from saved visual state
    after a renderer update; imported arbitrary Python is never executed.
    """
    root, output = Path(package_directory).resolve(), Path(output_path).resolve()
    _verify_package(root)
    if output.exists():
        raise FileExistsError(f"render output already exists: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("transparent Blender output must use .png")
    if not isinstance(samples, int) or isinstance(samples, bool) or not 1 <= samples <= 100_000:
        raise ValueError("samples must be a positive integer")
    if len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32768 for value in resolution):
        raise ValueError("invalid render resolution")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError("threads must be between 1 and 64")
    if not isinstance(passes, bool):
        raise ValueError("passes must be a boolean")
    if passes and (resolution[0] * resolution[1] > 4_000_000 or output.with_suffix(".passes.exr").exists()):
        raise ValueError("Pass export requires at most 4 megapixels and a new output path")
    executable = shutil.which(blender_executable)
    if executable is None:
        raise RuntimeError("Blender is not installed or not on PATH; the complete render package remains usable on a Blender machine")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Run the verified immutable package copy, not a live source file that
    # might change while the child Blender process is rendering.
    script_argument = _blender_argument_path(root / "blender_scene.py", executable)
    package_argument = _blender_argument_path(root, executable)
    output_argument = _blender_argument_path(output, executable)
    command = [
        executable, "--background", "--factory-startup", "--threads", str(threads), "--python-exit-code", "1",
        "--python", script_argument, "--",
        "--package", package_argument, "--output", output_argument, "--samples", str(samples),
        "--width", str(resolution[0]), "--height", str(resolution[1]),
    ]
    if passes:
        command.append("--passes")
    log = output.with_suffix(".blender.log")
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False)
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"Blender did not complete rendering; see {log}")
    receipt = {"render_package_id": _verify_package(root)["render_package_id"], "output_sha256": sha256(output.read_bytes()).hexdigest(), "samples": samples, "resolution": resolution, "threads": threads}
    if passes:
        pass_path = output.with_suffix(".passes.exr")
        if not pass_path.is_file():
            raise RuntimeError("Blender completed beauty but did not export requested render passes")
        channels = _exr_pass_channels(pass_path)
        expected = {"combined": (".Combined.",), "artistic_shell_depth": (".Depth.",),
                    "emission": (".Emit.", ".Emission."),
                    "base_coverage": (".ArtisticBaseCoverage.",),
                    "detail_coverage": (".ArtisticDetailCoverage.",)}
        semantic_passes = {name: [channel for channel in channels if any(marker in channel for marker in markers)]
                           for name, markers in expected.items()}
        if any(not values for values in semantic_passes.values()):
            raise RuntimeError("Blender EXR is missing one or more requested emission/depth/coverage passes")
        receipt["passes"] = {"file": pass_path.name, "sha256": sha256(pass_path.read_bytes()).hexdigest(),
            "channels": channels, "semantic_passes": semantic_passes,
            "semantics": "Artistic rendering channels. Depth is distance to visualization geometry in kilometres, not ionospheric or photographic depth; coverage is shader alpha, not confidence."}
    output.with_suffix(".render.json").write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return output


def _blender_argument_path(path: Path, executable: str) -> str:
    """Windows Blender launched from WSL needs Windows/UNC path arguments."""
    if sys.platform.startswith("linux") and executable.lower().endswith(".exe"):
        converter = shutil.which("wslpath")
        if converter is None:
            raise RuntimeError("Windows Blender from Linux requires WSL's wslpath converter")
        result = subprocess.run(
            [converter, "-w", str(path.resolve())], capture_output=True,
            text=True, check=True, timeout=10,
        )
        converted = result.stdout.strip()
        if not converted:
            raise RuntimeError("wslpath returned an empty Windows path")
        return converted
    return str(path.resolve())


def render_sequence(
    package_directory: str | Path, output_directory: str | Path, **render_options,
) -> tuple[Path, ...]:
    """Render all exported animation frames without imposing a video codec."""
    root, output = Path(package_directory).resolve(), Path(output_directory)
    sequence = json.loads((root / "sequence.json").read_text(encoding="utf-8"))
    if sequence.get("schema_version") != "ophanim-render-sequence/1" or not sequence.get("frames"):
        raise ValueError("invalid render sequence")
    packages = []
    for frame in sequence["frames"]:
        package = (root / frame["path"]).resolve()
        if not package.is_relative_to(root):
            raise ValueError("render frame escapes sequence directory")
        if _verify_package(package)["render_package_id"] != frame["render_package_id"]:
            raise ValueError("render frame identity mismatch")
        packages.append(package)
    output.mkdir(parents=True, exist_ok=False)
    return tuple(render_blender(package, output / f"frame_{index:06d}.png", **render_options) for index, package in enumerate(packages))
