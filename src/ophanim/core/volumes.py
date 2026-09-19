"""A bounded imagined volume, deliberately separate from scientific DynamicState.

The scalar named density is dimensionless material for visualization, NOT an
electron density, TEC reconstruction, observation, or calibrated prediction.
The contract is independent of its experimental producer and artistic consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

VOLUME_SCHEMA = "ophanim-imagined-volume/1"
MAX_VOLUME_CELLS = 3_000_000
MAX_FRAME_CELLS = 250_000
MAX_VOLUME_FRAMES = 24
VOLUME_SEMANTICS = ("Hypothetical dimensionless material evolving in an assumed altitude volume; "
                    "not measured plasma, not electron density, and not a reconstruction of TEC.")


@dataclass(frozen=True, slots=True)
class ImaginedVolume:
    dataset: Any
    recipe: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def validate(self) -> "ImaginedVolume":
        validate_volume(self)
        return self


def validate_volume(volume: ImaginedVolume) -> None:
    import numpy as np
    import xarray as xr

    if not isinstance(volume, ImaginedVolume) or not isinstance(volume.dataset, xr.Dataset):
        raise TypeError("imagined volume requires an xarray.Dataset")
    data = volume.dataset
    if data.attrs.get("volume_schema") != VOLUME_SCHEMA or data.attrs.get("source_kind") != "imagined":
        raise ValueError("volume must declare the imagined-volume schema and imagined source kind")
    if data.attrs.get("semantics") != VOLUME_SEMANTICS:
        raise ValueError("volume must preserve its explicitly hypothetical semantics")
    if "density" not in data or data.density.dims != ("time", "z", "y", "x"):
        raise ValueError("density dimensions must be (time,z,y,x)")
    if (not 3 <= data.sizes["time"] <= MAX_VOLUME_FRAMES
            or data.density.size > MAX_VOLUME_CELLS
            or data.sizes["z"] * data.sizes["y"] * data.sizes["x"] > MAX_FRAME_CELLS):
        raise ValueError("imagined volume exceeds the bounded frame/cell budget")
    for axis in ("time", "z", "y", "x"):
        if axis not in data.coords or data[axis].dims != (axis,):
            raise ValueError("volume axes must be one-dimensional coordinates")
        values = np.asarray(data[axis].values)
        if axis == "time":
            if values.dtype.kind != "M" or np.isnat(values).any() or data.time.attrs.get("timezone") != "UTC":
                raise ValueError("volume time must be explicitly UTC datetime64")
            increasing = np.diff(values) > np.timedelta64(0, "ns")
        else:
            if len(values) < 3 or data[axis].attrs.get("units") != "m" or not np.isfinite(values).all():
                raise ValueError("volume spatial axes require at least three finite coordinates in metres")
            increasing = np.diff(values) > 0
            if not np.allclose(np.diff(values), np.diff(values)[0], rtol=1e-5, atol=1e-5):
                raise ValueError("volume spatial axes must be regular")
        if not np.all(increasing):
            raise ValueError("volume axes must be strictly increasing")
    if float(data.z.min()) < 0 or data.z.attrs.get("semantic") != "chosen altitude above reference surface":
        raise ValueError("z must be an explicitly chosen nonnegative altitude")
    if data.density.attrs.get("units") != "1" or data.density.attrs.get("semantic_class") != "synthetic":
        raise ValueError("density must be dimensionless synthetic material")
    if set(data.data_vars) != {"density"}:
        raise ValueError("this volume schema only defines the density material channel")
    values = np.asarray(data.density.values)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("imagined density must be finite and nonnegative")
    for name, bounds in (("center_latitude", (-90, 90)), ("center_longitude", (-180, 180))):
        value = data.attrs.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"volume requires a valid {name}")
    if not isinstance(volume.recipe, Mapping) or not isinstance(volume.provenance, Mapping) or not volume.provenance.get("producer"):
        raise ValueError("volume requires a recipe and producer provenance")


def read_volume(directory) -> ImaginedVolume:
    """Verify and load a published volume; never reinterpret it as measured state."""
    import json
    from pathlib import Path
    import xarray as xr
    from .artifacts import verify_run

    root = Path(directory).resolve()
    manifest = verify_run(root, kind="hypothesis")
    contract = json.loads((root / "volume_contract.json").read_text())
    if contract.get("schema_version") != VOLUME_SCHEMA or contract.get("dataset") != "volume.zarr":
        raise ValueError("unsupported imagined volume contract")
    actual = {p.relative_to(root).as_posix() for p in (root / "volume.zarr").rglob("*") if p.is_file()}
    expected = {p for p in manifest["files"] if p.startswith("volume.zarr/")}
    if actual != expected:
        raise ValueError("untracked or missing volume array artifacts")
    recipe = json.loads((root / "hypothesis.json").read_text())
    with xr.open_zarr(root / "volume.zarr", consolidated=True) as data:
        if "density" not in data or data.density.size > MAX_VOLUME_CELLS:
            raise ValueError("published volume exceeds cell budget")
        result = data.load()
    return ImaginedVolume(result, recipe, contract["provenance"]).validate()


__all__ = ["ImaginedVolume", "validate_volume", "read_volume", "VOLUME_SCHEMA", "VOLUME_SEMANTICS"]
