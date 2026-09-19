"""Shared immutable artifact storage, serialization, and core code provenance.

This module never imports experimental generators or downstream rendering code.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from hashlib import sha256
from importlib import metadata
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from ophanim import __version__

LOGGER = logging.getLogger(__name__)
RUN_SCHEMA = "ophanim-science-run/1"


def json_value(value: Any) -> Any:
    """Convert metadata to strict JSON, with non-finite diagnostics as null."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError(f"Cannot serialize metadata of type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    """Write inside a private staging directory, before atomic publication."""
    path.write_text(json.dumps(json_value(payload), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_sha256(dataset: Any) -> str:
    """Hash coordinates, values, masks and metadata, not Zarr compression bytes."""
    import numpy as np

    digest = sha256(canonical_json(dataset.attrs))
    for name in sorted(dataset.variables):
        variable = dataset[name]
        values = np.asarray(variable.values)
        digest.update(canonical_json({"name": name, "dims": variable.dims,
                                     "dtype": str(values.dtype),
                                     "shape": values.shape, "attrs": variable.attrs}))
        if values.dtype.kind in "OU":
            digest.update(canonical_json(values.tolist()))
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def software_identity(*, extra_sections: tuple[str, ...] = ()) -> dict[str, Any]:
    """Hash core code, with explicit opt-in downstream source sections.

    Experiment or artistic code changes do not invalidate scientific core runs.
    """
    root = Path(__file__).parent.parent
    sections = [root / name for name in ("core", "sensing", "dynamics", "science_reports.py",
                "ionex.py", "artifacts.py", "tec_archive.py", "regional_grid.py")]
    for name in extra_sections:
        if name not in {"experiments", "shawtynet"}:
            raise ValueError("Unknown software identity section")
        sections.append(root / name)
    digest = sha256()
    files = sorted(file for section in sections
                   for file in ([section] if section.is_file() else section.rglob("*.py")))
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    versions = {}
    for package in ("numpy", "scipy", "xarray", "scikit-image", "pyproj", "zarr",
                    "matplotlib", "Pillow", "PyYAML"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"ophanim_version": __version__, "code_sha256": digest.hexdigest(),
            "dependencies": versions, "extra_sections": list(extra_sections)}


def verify_run(directory: str | Path, *, kind: str | None = None) -> dict[str, Any]:
    """Fail closed on incomplete, modified, escaped or symlinked artifacts."""
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Run manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported run manifest")
    if kind is not None and manifest.get("kind") != kind:
        raise ValueError(f"Expected {kind} run, got {manifest.get('kind')}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Run manifest has no artifact inventory")
    for relative, checksum in files.items():
        candidate = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Run artifact reference escapes run directory")
        if (candidate.is_symlink() or not candidate.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in candidate.parents if parent != root
                       and parent.is_relative_to(root))):
            raise ValueError("Run artifact reference must not traverse symlinks")
        if not candidate.is_file() or file_sha256(candidate) != checksum:
            raise ValueError(f"Run artifact checksum mismatch: {relative}")
    # Nested visual/render runs are intentionally not part of the science inventory.
    for name in ("dynamic.zarr", "visual.zarr", "native.zarr"):
        array_root = root / name
        if array_root.exists():
            actual = {p.relative_to(root).as_posix() for p in array_root.rglob("*") if p.is_file()}
            expected = {p for p in files if p.startswith(name + "/")}
            if actual != expected:
                raise ValueError(f"Untracked/missing files in {name}")
    return manifest


def _publish(root: Path, run_id: str, kind: str, identity: dict[str, Any],
             build: Callable[[Path], None]) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    if destination.exists():
        existing = verify_run(destination, kind=kind)
        if existing.get("identity") != json_value(identity):
            raise ValueError("Run identity collision")
        return destination
    stage = Path(tempfile.mkdtemp(prefix=f".{kind}-", dir=root))
    try:
        build(stage)
        inventory = {p.relative_to(stage).as_posix(): file_sha256(p)
                     for p in sorted(stage.rglob("*")) if p.is_file()}
        write_json(stage / "manifest.json", {
            "schema": RUN_SCHEMA, "kind": kind, "run_id": run_id,
            "status": "complete", "identity": identity, "files": inventory,
        })
        # The complete marker becomes visible only with the complete directory.
        try:
            stage.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            existing = verify_run(destination, kind=kind)
            if existing.get("identity") != json_value(identity):
                raise ValueError("Concurrent run publication identity mismatch")
        LOGGER.info("%s complete: %s", kind, destination)
        return destination
    finally:
        # Only this call's newly allocated private temporary directory is removed.
        if stage.exists():
            shutil.rmtree(stage)


def _write_dataset(dataset: Any, destination: Path) -> None:
    serializable = dataset.copy(deep=False)
    serializable.attrs = json_value(dataset.attrs)
    for name in serializable.variables:
        serializable[name].attrs = json_value(dataset[name].attrs)
        serializable[name].encoding = {}
    serializable.to_zarr(destination, mode="w", zarr_format=2, consolidated=True)
