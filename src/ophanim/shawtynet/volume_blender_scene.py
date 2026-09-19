"""Trusted standalone Blender renderer for plain-data imagined 3-D packages.

No OPHANIM, xarray or scientific inference executes inside Blender.
"""
from __future__ import annotations
import sys

# Exported packages contain plain data, not import search paths. Strip the
# script/current directory before importing even standard-library helpers.
if __name__ == "__main__":
    import os
    _excluded = {os.path.realpath(os.path.dirname(__file__)), os.path.realpath(os.getcwd())}
    sys.path[:] = [entry for entry in sys.path if entry and os.path.realpath(entry) not in _excluded]

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import zipfile

SCHEMA = "ophanim-imagined-volume-render/1"
MAX_FRAME_VOXELS = 1_000_000
MAX_MESH_FACES = 220_000
MAX_PACKAGE_BYTES = 48 * 1024 * 1024
STYLES = {
    "luminous": {"low": [0.24, 0.035, 0.54], "mid": [0.018, 0.57, 0.65],
                 "high": [1.0, 0.30, 0.055], "emission": 1.15, "opacity": 1.0, "threads": 64},
    "ghost": {"low": [0.09, 0.08, 0.29], "mid": [0.32, 0.55, 0.86],
              "high": [0.84, 0.94, 1.0], "emission": 1.0, "opacity": 0.68, "threads": 48},
    "quiet": {"low": [0.065, 0.11, 0.18], "mid": [0.24, 0.47, 0.48],
              "high": [0.89, 0.77, 0.53], "emission": 0.65, "opacity": 0.75, "threads": 28},
}


def _digest(path):
    with Path(path).open("rb") as stream:
        return sha256(stream.read(MAX_PACKAGE_BYTES + 1)).hexdigest()


def validate_arrays(root, metadata):
    """Bound NPY headers before NumPy allocates, then validate plain geometry.

    This runs both in the app and inside the standalone exported renderer.
    In particular, unsigned decreasing offsets must never wrap into huge
    positive spline point counts.
    """
    import numpy as np
    loaded, decoded, total_faces = {}, 0, 0
    for name in metadata["files"]:
        if not name.endswith(".npz"):
            continue
        expected = ({"vertices", "faces", "colors"} if name.startswith("isosurface_") else
                    {"points", "offsets", "weights"} if name == "threads.npz" else
                    {"density", "x_km", "y_km", "z_km"})
        with zipfile.ZipFile(root / name) as archive:
            members = archive.infolist()
            if len(members) != len(expected) or {entry.filename for entry in members} != {key + ".npy" for key in expected}:
                raise ValueError("Invalid volume numeric archive inventory")
            for entry in members:
                decoded += entry.file_size
                if decoded > MAX_PACKAGE_BYTES:
                    raise ValueError("Volume numeric archive exceeds its decoded size budget")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10_000)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10_000)
                    else:
                        raise ValueError("Unsupported volume numeric array encoding")
                    if (dtype.kind not in "fiu" or dtype.itemsize > 8 or len(shape) > 3
                            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                            or math.prod(shape) * dtype.itemsize != entry.file_size - stream.tell()):
                        raise ValueError("Volume numeric array header exceeds its declared payload")
        with np.load(root / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in expected}
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("Volume arrays must contain finite numeric data")
        loaded[name] = arrays
        if name.startswith("isosurface_"):
            v, f, c = (arrays[key] for key in ("vertices", "faces", "colors"))
            total_faces += len(f)
            if (v.ndim != 2 or v.shape[1] != 3 or c.shape != v.shape
                    or f.ndim != 2 or f.shape[1] != 3 or total_faces > MAX_MESH_FACES
                    or f.dtype.kind not in "iu" or np.any(f < 0) or np.any(f >= len(v))
                    or np.any(np.abs(v) > 1_000_000) or np.any(c < 0) or np.any(c > 1)):
                raise ValueError("Invalid volume mesh geometry or aggregate face budget")
        elif name == "threads.npz":
            p, o, w = (arrays[key] for key in ("points", "offsets", "weights"))
            if (p.ndim != 2 or p.shape[1] != 3 or w.shape != (len(p),)
                    or o.ndim != 1 or o.dtype.kind not in "iu" or not 1 <= len(o) <= 129
                    or len(p) > 20_000 or o[0] != 0 or o[-1] != len(p)
                    or np.any(o < 0) or np.any(o > len(p)) or np.any(o[1:] <= o[:-1])
                    or np.any(np.abs(p) > 1_000_000) or np.any(w < 0) or np.any(w > 1_000_000)):
                raise ValueError("Invalid volume thread geometry")
        else:
            d = arrays["density"]
            if d.ndim != 3 or min(d.shape) < 4 or d.size > MAX_FRAME_VOXELS or np.any(d < 0):
                raise ValueError("Invalid packaged numerical density")
            for index, axis_name in enumerate(("z_km", "y_km", "x_km")):
                axis = arrays[axis_name]
                if (axis.shape != (d.shape[index],) or np.any(np.abs(axis) > 1_000_000)
                        or np.any(axis[1:] <= axis[:-1])):
                    raise ValueError("Invalid packaged numerical coordinates")
    if (metadata.get("total_mesh_faces") != total_faces
            or metadata.get("thread_count") != len(loaded["threads.npz"]["offsets"]) - 1):
        raise ValueError("Volume geometry inventory does not match metadata")
    for layer in metadata["meshes"]:
        arrays = loaded[layer["file"]]
        if layer.get("faces") != len(arrays["faces"]) or layer.get("vertices") != len(arrays["vertices"]):
            raise ValueError("Volume mesh inventory does not match metadata")
    return loaded


def validate_package(directory):
    """Reject unsafe paths, imports, resource bombs and malformed plain data."""
    root = Path(directory).resolve()
    path = root / "scene_metadata.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError("Invalid volume package metadata")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported imagined-volume render package")
    identity = dict(metadata)
    claimed = identity.pop("render_package_id", None)
    if claimed != sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest():
        raise ValueError("Volume package identity mismatch")
    files = metadata.get("files", {})
    allowed = {"volume.npz", "threads.npz", "preview.png", "volume_blender_scene.py",
               "isosurface_0.npz", "isosurface_1.npz", "isosurface_2.npz"}
    if (not isinstance(files, dict) or not {"volume.npz", "threads.npz", "volume_blender_scene.py"} <= files.keys()
            or set(files) - allowed or {entry.name for entry in root.iterdir()} != {*files, "scene_metadata.json"}):
        raise ValueError("Invalid volume package inventory; extra files or imports are forbidden")
    total = 0
    for name, digest in files.items():
        asset = root / name
        if asset.is_symlink() or not asset.is_file():
            raise ValueError("Volume package files must be regular local files")
        total += asset.stat().st_size
        if total > MAX_PACKAGE_BYTES or _digest(asset) != digest:
            raise ValueError("Volume package checksum or size violation")
    if _digest(root / "volume_blender_scene.py") != _digest(__file__):
        raise ValueError("Packaged volume renderer differs from installed trusted renderer; re-export this saved volume before rendering")
    if metadata.get("style") not in STYLES or metadata.get("material") != STYLES[metadata["style"]]:
        raise ValueError("Invalid volume material controls")
    if metadata.get("threads") != "threads.npz" or metadata.get("volume") != "volume.npz":
        raise ValueError("Volume package reference is not a fixed asset")
    meshes = metadata.get("meshes")
    if (not isinstance(meshes, list) or len(meshes) > 3
            or any(not isinstance(mesh, dict) or type(mesh.get("layer")) is not int or mesh["layer"] not in (0, 1, 2)
                   or mesh.get("file") != f"isosurface_{mesh['layer']}.npz" or mesh["file"] not in files
                   or mesh.get("density_fraction") != (.18, .40, .68)[mesh["layer"]] for mesh in meshes)
            or len({mesh["file"] for mesh in meshes}) != len(meshes)
            or {name for name in files if name.startswith("isosurface_")} != {mesh["file"] for mesh in meshes}):
        raise ValueError("Invalid isosurface references")
    import numpy as np
    bounds = np.asarray(metadata.get("bounds_km"), dtype=float)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(np.abs(bounds) > 1_000_000) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid volume bounds")
    view = metadata.get("camera", {})
    vectors = np.asarray([view.get(name) for name in ("right", "up", "forward", "location_km", "target_km")], dtype=float)
    if (vectors.shape != (5, 3) or not np.all(np.isfinite(vectors)) or np.any(np.abs(vectors) > 1_000_000)
            or not np.allclose(vectors[:3] @ vectors[:3].T, np.eye(3), atol=1e-5)
            or np.linalg.norm(vectors[3] - vectors[4]) < .001):
        raise ValueError("Invalid volume camera vectors")
    fov, aspect = view.get("config", {}).get("horizontal_fov_deg"), view.get("aspect_ratio")
    if not isinstance(fov, (int, float)) or not 5 <= fov <= 150 or not isinstance(aspect, (int, float)) or not .05 <= aspect <= 20:
        raise ValueError("Invalid volume camera lens")
    arrays = validate_arrays(root, metadata)
    return metadata, arrays


def _surface_material(name, material, layer):
    import bpy
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes, links = result.node_tree.nodes, result.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = "Density color"
    links.new(color.outputs["Color"], principled.inputs["Base Color"])
    links.new(color.outputs["Color"], principled.inputs["Emission Color"])
    principled.inputs["Emission Strength"].default_value = material["emission"] * (.32, .46, .62)[layer]
    principled.inputs["Metallic"].default_value = .28
    principled.inputs["Roughness"].default_value = .32
    # Grazing faces brighten a coherent silhouette and expose 3-D folds.
    facing = nodes.new("ShaderNodeLayerWeight")
    facing.inputs["Blend"].default_value = .35
    amount = nodes.new("ShaderNodeMath")
    amount.operation = "MULTIPLY_ADD"
    amount.inputs[1].default_value = (.28, .40, .48)[layer] * material["opacity"]
    amount.inputs[2].default_value = (.065, .11, .23)[layer] * material["opacity"]
    links.new(facing.outputs["Fresnel"], amount.inputs[0])
    mix = nodes.new("ShaderNodeMixShader")
    links.new(amount.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(principled.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return result


def _thread_material(material, warm=False):
    import bpy
    result = bpy.data.materials.new("Golden density ridge" if warm else "Cyan density ridge")
    result.use_nodes = True
    nodes = result.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*material["high" if warm else "mid"], 1)
    emission.inputs["Strength"].default_value = material["emission"] * (1.7 if warm else 1.2)
    result.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return result


def build_scene(package, *, samples=24, width=960, height=640):
    if type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError("Volume rendering supports 1..128 samples")
    if any(type(value) is not int or not 64 <= value <= 2400 for value in (width, height)) or width * height > 4_000_000:
        raise ValueError("Volume rendering supports at most four megapixels")
    metadata, packages = validate_package(package)
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    material = metadata["material"]
    for layer in metadata["meshes"]:
        vertices, faces, colors = (packages[layer["file"]][name] for name in ("vertices", "faces", "colors"))
        mesh = bpy.data.meshes.new(f"Numerical density isosurface {layer['density_fraction']}")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        attribute = mesh.color_attributes.new(name="Density color", type="FLOAT_COLOR", domain="POINT")
        rgba = np.column_stack((colors, np.ones(len(colors)))).astype(np.float32)
        attribute.data.foreach_set("color", rgba.ravel())
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(_surface_material(mesh.name, material, layer["layer"]))
        obj["semantics"] = "Isosurface of imagined numerical density, not measured altitude or electron density"
    points, offsets, weights = (packages["threads.npz"][name] for name in ("points", "offsets", "weights"))
    bounds = np.asarray(metadata["bounds_km"])
    width_km = float(max(bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0]))
    for warm in (False, True):
        curves = bpy.data.curves.new("Imagined density ridge threads", "CURVE")
        curves.dimensions = "3D"
        curves.resolution_u = 2
        curves.bevel_depth = width_km * (.00065 if warm else .00045)
        curves.bevel_resolution = 2
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            start, end = int(start), int(end)
            if (index % 4 == 0) != warm or end - start < 2:
                continue
            spline = curves.splines.new("POLY")
            spline.points.add(int(end - start) - 1)
            coords = np.column_stack((points[start:end], np.ones(end - start)))
            spline.points.foreach_set("co", coords.ravel())
            for j, point in enumerate(spline.points):
                taper = math.sin(math.pi * (j + .5) / (end - start)) ** .5
                point.radius = max(.05, float(weights[start + j]) ** .5 * taper)
        obj = bpy.data.objects.new(curves.name, curves)
        bpy.context.collection.objects.link(obj)
        curves.materials.append(_thread_material(material, warm))
        obj["semantics"] = "Artistic ridge-following density curves; not physical streamlines or observed plasma trajectories"
    view = metadata["camera"]
    camera_data = bpy.data.cameras.new("Auto-framed imagined-volume orbit")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.collection.objects.link(camera)
    right, up, forward = (Vector(view[name]) for name in ("right", "up", "forward"))
    camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()
    center = Vector(view["target_km"])
    # Preserve safe framing when requesting a different render aspect ratio.
    original_distance = (Vector(view["location_km"]) - center).length
    aspect_gain = max(1.0, (width / height) / view.get("aspect_ratio", 1.5))
    camera.location = center - forward * original_distance * aspect_gain
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle = math.radians(view["config"]["horizontal_fov_deg"])
    camera_data.clip_start, camera_data.clip_end = .01, 100_000
    for name, position, energy, color, size in (
        ("Cool sculpting softbox", center + Vector((-width_km, -.4 * width_km, .8 * width_km)), 3.0 * width_km ** 2, (.34, .62, 1), .8 * width_km),
        ("Warm edge softbox", center + Vector((.7 * width_km, .4 * width_km, .5 * width_km)), 2.0 * width_km ** 2, (1, .43, .17), .55 * width_km),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.color, light_data.shape, light_data.size = energy, color, "DISK", size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = position
        light.rotation_euler = (center - position).to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.max_bounces = 6
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = "AgX"
    scene.world.color = (.012, .02, .04)
    return scene


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    output = Path(args.output).resolve()
    if output.exists() or output.suffix.lower() != ".png":
        raise ValueError("Volume rendering needs a new PNG output")
    scene = build_scene(args.package, samples=args.samples, width=args.width, height=args.height)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
