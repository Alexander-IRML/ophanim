"""Standalone Blender backend. Consume only an exported render package.

Usage: blender -b --python blender_scene.py -- --package PACKAGE --output out.png
This script runs inside Blender's Python; it has no OPHANIM/xarray dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _texture(nodes, root, relative, *, color=False):
    import bpy
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(root / relative), check_existing=True)
    node.image.colorspace_settings.name = "sRGB" if color else "Non-Color"
    node.extension = "CLIP"
    node.interpolation = "Linear"
    return node


def _material(root, metadata, *, detail=False):
    import bpy
    material = bpy.data.materials.new("Directional artistic fibers" if detail else "Supported artistic veil")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    mix = nodes.new("ShaderNodeMixShader")
    color = _texture(nodes, root, metadata["color_texture"], color=True)
    links.new(color.outputs["Color"], emission.inputs["Color"])
    textures = metadata["textures"]
    style = metadata.get("visual_provenance", {}).get("style", {})
    strength_field = "display_emission" if "display_emission" in textures else "emission"
    strength = _texture(nodes, root, textures[strength_field]["png"])
    decode = nodes.new("ShaderNodeMath")
    decode.operation = "MULTIPLY_ADD"
    gain = style.get("detail_emission_gain", 2.0) if detail else 1
    decode.inputs[1].default_value = textures[strength_field]["png_scale"] * gain
    decode.inputs[2].default_value = textures[strength_field]["png_offset"] * gain
    links.new(strength.outputs["Color"], decode.inputs[0])
    shoulder = nodes.new("ShaderNodeMath")
    shoulder.operation = "MINIMUM"
    shoulder.inputs[1].default_value = 0.95
    links.new(decode.outputs[0], shoulder.inputs[0])
    links.new(shoulder.outputs[0], emission.inputs["Strength"])
    if detail:
        fibers = _texture(nodes, root, textures["flow_texture"]["png"])
        confidence = _texture(nodes, root, textures["flow_texture_strength"]["png"])
        opacity = nodes.new("ShaderNodeMath")
        opacity.operation = "MULTIPLY"
        links.new(fibers.outputs["Color"], opacity.inputs[0])
        links.new(confidence.outputs["Color"], opacity.inputs[1])
        alpha = opacity.outputs[0]
        if "ribbon_texture" in textures and "ribbon_strength" in textures:
            ribbon = _texture(nodes, root, textures["ribbon_texture"]["png"])
            ribbon_support = _texture(nodes, root, textures["ribbon_strength"]["png"])
            product = nodes.new("ShaderNodeMath")
            product.operation = "MULTIPLY"
            links.new(ribbon.outputs["Color"], product.inputs[0])
            links.new(ribbon_support.outputs["Color"], product.inputs[1])
            combine = nodes.new("ShaderNodeMath")
            combine.operation = "ADD"
            combine.use_clamp = True
            links.new(alpha, combine.inputs[0])
            links.new(product.outputs[0], combine.inputs[1])
            alpha = combine.outputs[0]
    else:
        opacity = _texture(nodes, root, textures["base_opacity"]["png"])
        alpha = opacity.outputs["Color"]
    if "breakup" in textures:
        breakup = _texture(nodes, root, textures["breakup"]["png"])
        inverse = nodes.new("ShaderNodeMath")
        inverse.operation = "SUBTRACT"
        inverse.inputs[0].default_value = 1
        links.new(breakup.outputs["Color"], inverse.inputs[1])
        attenuate = nodes.new("ShaderNodeMath")
        attenuate.operation = "MULTIPLY"
        links.new(alpha, attenuate.inputs[0])
        links.new(inverse.outputs[0], attenuate.inputs[1])
        alpha = attenuate.outputs[0]
    links.new(alpha, mix.inputs[0])
    aov = nodes.new("ShaderNodeOutputAOV")
    aov.aov_name = "ArtisticDetailCoverage" if detail else "ArtisticBaseCoverage"
    links.new(alpha, aov.inputs["Value"])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return material


def build_scene(package, *, samples=32, width=1280, height=720, passes=False):
    import bpy
    import numpy as np
    from mathutils import Quaternion, Vector

    root = Path(package)
    metadata = json.loads((root / "scene_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "ophanim-render/1":
        raise ValueError("unsupported OPHANIM render package")
    with np.load(root / metadata["geometry"], allow_pickle=False) as arrays:
        vertices, faces, uv, normals = (arrays[name] for name in ("vertices", "faces", "uv", "normals"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for detail in (False, True):
        mesh = bpy.data.meshes.new("artistic_shell_detail" if detail else "artistic_shell_base")
        offset = metadata.get("visual_provenance", {}).get("style", {}).get("detail_layer_offset_km", 0.2)
        positions = vertices + normals * (offset if detail else 0)
        mesh.from_pydata(positions.tolist(), [], faces.tolist())
        mesh.update()
        uv_layer = mesh.uv_layers.new(name="Render UV")
        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = uv[loop.vertex_index].tolist()
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(_material(root, metadata, detail=detail))
        obj["semantics"] = "Artistic visualization shell; no measured altitude or plasma velocity"
    camera_data = bpy.data.cameras.new("Manual photograph camera")
    camera = bpy.data.objects.new("Manual photograph camera", camera_data)
    bpy.context.collection.objects.link(camera)
    config = metadata["camera"]
    heading, pitch = math.radians(config["heading_deg"]), math.radians(config["pitch_deg"])
    forward = Vector((math.sin(heading) * math.cos(pitch), math.cos(heading) * math.cos(pitch), math.sin(pitch)))
    rotation = forward.to_track_quat("-Z", "Y")
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Quaternion(forward, math.radians(config["roll_deg"])) @ rotation
    camera.location = (0, 0, 0)
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle = math.radians(config["horizontal_fov_deg"])
    camera_data.clip_start, camera_data.clip_end = 0.001, 30_000
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 16
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0, 0, 0)
    if passes:
        layer = scene.view_layers[0]
        layer.use_pass_z = True
        layer.use_pass_emit = True
        layer.pass_alpha_threshold = 0.001
        for name in ("ArtisticBaseCoverage", "ArtisticDetailCoverage"):
            aov = layer.aovs.add()
            aov.name, aov.type = name, "VALUE"
    return scene


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--passes", action="store_true")
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    scene = build_scene(arguments.package, samples=arguments.samples, width=arguments.width, height=arguments.height, passes=arguments.passes)
    scene.render.filepath = str(Path(arguments.output).resolve())
    bpy.ops.render.render(write_still=True)
    if arguments.passes:
        # Blender 5 splits media type from encoding; 4.x used one enum.
        if hasattr(scene.render.image_settings, "media_type"):
            scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_depth = "32"
        bpy.data.images["Render Result"].save_render(str(Path(arguments.output).with_suffix(".passes.exr").resolve()), scene=scene)


if __name__ == "__main__":
    main()
