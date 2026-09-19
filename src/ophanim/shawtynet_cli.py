"""Independently rerunnable scientific, artistic, rendering and composite stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any


def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Configuration exceeds 2 MiB")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def _path(value: str | Path, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_input(request: dict[str, Any], *, base: Path, as_of: str | None = None) -> Any:
    """Select an existing source; no model initialization or duplicate downloader."""
    from ophanim.sensing import read_archive_sequence, read_desktop_sequence, read_ionex_sequence

    settings = dict(request)
    kind = settings.pop("kind", "desktop")
    if kind == "synthetic":
        from ophanim.experiments import make_synthetic_dataset
        parameters = settings.pop("parameters", {})
        if settings:
            raise ValueError(f"Unknown synthetic input settings: {sorted(settings)}")
        return make_synthetic_dataset(**parameters)
    selection = {key: settings.pop(key) for key in ("start", "end", "bounds") if key in settings}
    selection["as_of"] = as_of
    if kind == "desktop":
        directory = _path(settings.pop("data_directory", "var/desktop"), base)
        artifact_ids = settings.pop("artifact_ids", None)
        if settings:
            raise ValueError(f"Unknown desktop input settings: {sorted(settings)}")
        return read_desktop_sequence(directory, artifact_ids, **selection)
    if kind == "archive":
        root = _path(settings.pop("zarr_root", "var/zarr"), base)
        grid_ids = settings.pop("grid_set_ids", None)
        if settings:
            raise ValueError(f"Unknown archive input settings: {sorted(settings)}")
        return read_archive_sequence(root, grid_ids, **selection)
    if kind == "ionex":
        from ophanim.domain import SourceArtifact
        from ophanim.ionex import IONEXV1Parser
        path = _path(settings.pop("path"), base)
        checksum = sha256(path.read_bytes()).hexdigest()
        availability = settings.pop("available_at", None)
        if as_of and availability is None:
            raise ValueError("Causal standalone IONEX requires known available_at provenance")
        ingested_at = (datetime.fromisoformat(availability.replace("Z", "+00:00"))
                       if availability else datetime.now(timezone.utc))
        artifact = SourceArtifact(
            artifact_id="shawtynet-ionex-" + checksum, provider=settings.pop("provider", "local"),
            product=settings.pop("product", "ionex"), parser_version=IONEXV1Parser.parser_version,
            revision_priority=settings.pop("revision_priority", 0), checksum_sha256=checksum,
            storage_ref=str(path), ingested_at=ingested_at, source_uri=path.as_uri(),
        )
        if settings:
            raise ValueError(f"Unknown IONEX input settings: {sorted(settings)}")
        return read_ionex_sequence(path, artifact=artifact, **selection)
    raise ValueError("Input kind must be desktop, archive, ionex or synthetic")


def analyze_request(request: dict[str, Any], *, base: Path, output_root: Path) -> Path:
    from ophanim.dynamics import AnalysisConfig
    from ophanim.science_runs import analyze_run

    unknown = set(request) - {"input", "analysis"}
    if unknown:
        raise ValueError(f"Unknown request sections: {sorted(unknown)}")
    config = AnalysisConfig.from_dict(request.get("analysis", {}))
    source = load_input(request.get("input", {}), base=base,
                        as_of=config.as_of if config.analysis_mode == "causal" else None)
    return analyze_run(source, config, output_root, request=request)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OPHANIM dynamics → ShawtyNet art pipeline")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "run"):
        item = commands.add_parser(name, help="Analyze a source" if name == "analyze" else "Run independent stages in sequence")
        item.add_argument("config", type=Path)
        item.add_argument("--output", type=Path, default=Path("var/shawtynet"))
    demo = commands.add_parser("demo", help="Synthetic validation/example, explicitly not observed TEC")
    demo.add_argument("--case", default="plane_wave",
                      choices=("quiet", "translating_gaussian", "growing_gaussian", "translating_growing_gaussian",
                               "plane_wave", "wave_packet", "moving_front", "crossing_waves", "noise", "missing"))
    demo.add_argument("--output", type=Path, default=Path("var/shawtynet"))
    demo.add_argument("--science-only", action="store_true")
    visual = commands.add_parser("visualize", help="New style; reuse immutable science")
    visual.add_argument("run", type=Path)
    for item in (visual, commands.choices["run"]):
        item.add_argument("--style", type=Path)
        item.add_argument("--camera", type=Path)
        item.add_argument("--frame", type=int, default=None,
                          help="frame index; defaults to the analyzed target timestamp")
        item.add_argument("--sequence", action="store_true",
                          help="explicitly generate all visual frames (higher memory use)")
    render = commands.add_parser("render", help="Transparent Blender render of an exported visual run")
    render.add_argument("run", type=Path)
    for item in (render, commands.choices["run"]):
        item.add_argument("--blender", default="blender")
        item.add_argument("--samples", type=int, default=32)
        item.add_argument("--width", type=int, default=1280)
        item.add_argument("--height", type=int, default=720)
        item.add_argument("--threads", type=int, default=2)
    commands.choices["run"].add_argument("--render", action="store_true")
    commands.choices["run"].add_argument("--photo", type=Path)
    composite = commands.add_parser("composite", help="Manually aligned photo and foreground/cloud mask composite")
    composite.add_argument("run", type=Path)
    composite.add_argument("photo", type=Path)
    for item in (composite, commands.choices["run"]):
        item.add_argument("--foreground-mask", type=Path)
        item.add_argument("--cloud-mask", type=Path)
        item.add_argument("--horizon-y", type=float, default=0.5)
        item.add_argument("--horizon-fade", type=float, default=0.6)
        item.add_argument("--saturation", type=float, default=0.85)
        item.add_argument("--exposure", type=float, default=1.0)
    inspect = commands.add_parser("inspect", help="Verify hashes and print provenance")
    inspect.add_argument("run", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if arguments.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        from ophanim.science_runs import (analyze_run, composite_run, json_value, render_run,
                                         verify_run, visualize_run)
        paths: dict[str, Any] = {}
        command = arguments.command
        if command == "inspect":
            manifest = verify_run(arguments.run)
            print(json.dumps(manifest, indent=2))
            return 0
        if command in ("analyze", "run"):
            paths["science"] = analyze_request(load_mapping(arguments.config),
                                                base=arguments.config.resolve().parent,
                                                output_root=arguments.output)
        elif command == "demo":
            from ophanim.dynamics import AnalysisConfig
            from ophanim.experiments import make_synthetic_dataset
            raw = make_synthetic_dataset(kind=arguments.case)
            config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20.0)
            paths["science"] = analyze_run(raw, config, arguments.output,
                                            request={"synthetic_case": arguments.case})
            if not arguments.science_only:
                paths["visual"] = visualize_run(paths["science"])
        if command in ("visualize", "run"):
            from ophanim.shawtynet import CameraConfig, VisualStyleConfig
            style = VisualStyleConfig.from_file(arguments.style) if arguments.style else VisualStyleConfig()
            camera = CameraConfig(**load_mapping(arguments.camera)) if arguments.camera else None
            source = paths["science"] if command == "run" else arguments.run
            paths["visual"] = visualize_run(source, style, camera=camera, frame_index=arguments.frame,
                                             sequence=arguments.sequence)
        if command == "render" or (command == "run" and (arguments.render or arguments.photo)):
            source = paths["visual"] if command == "run" else arguments.run
            paths["render"] = render_run(source, blender_executable=arguments.blender,
                                          samples=arguments.samples, resolution=(arguments.width, arguments.height),
                                          threads=arguments.threads)
        if command == "composite" or (command == "run" and arguments.photo):
            source = paths["render"] if command == "run" else arguments.run
            options = {name: getattr(arguments, name) for name in
                       ("foreground_mask", "cloud_mask", "horizon_y", "horizon_fade", "saturation", "exposure")}
            paths["composite"] = composite_run(source, arguments.photo, **options)
        print(json.dumps({"ok": True, **json_value(paths)}, indent=2))
        return 0
    except ImportError as error:
        print(f"ophanim-shawtynet: missing optional dependency: {error}. Install 'ophanim[shawtynet]'.", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        if arguments.verbose:
            logging.exception("Pipeline failed")
        print(f"ophanim-shawtynet: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
