"""Verified native TEC snapshots without a forecasting-model prerequisite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

from ophanim.tec_archive import (
    NATIVE_VALUE_KIND, TEC_GRID_LAYOUT_VERSION, TECGridManifest, ZarrTECGridStore,
)

if TYPE_CHECKING:
    import xarray as xr


class SensingError(ValueError):
    """A sequence has no usable support or violates the observation contract."""


class TECSequenceReader(Protocol):
    def read(self, *, start: datetime | str | None = None,
             end: datetime | str | None = None,
             as_of: datetime | str | None = None,
             bounds: Mapping[str, float] | None = None) -> xr.Dataset: ...


@dataclass(frozen=True)
class ArchiveTECSequenceReader:
    """A reusable read-only adapter over the existing immutable native store."""

    root: str | Path
    grid_set_ids: Sequence[str] | None = None

    def read(self, *, start=None, end=None, as_of=None, bounds=None):
        return read_archive_sequence(
            self.root, self.grid_set_ids, start=start, end=end,
            as_of=as_of, bounds=bounds,
        )


def _utc(value: datetime | str | None, name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise SensingError(f"{name} must be an ISO timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SensingError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _window(start, end, as_of):
    start, end, as_of = (_utc(start, "start"), _utc(end, "end"), _utc(as_of, "as_of"))
    if start is not None and end is not None and start > end:
        raise SensingError("start cannot follow end")
    return start, end, as_of


def _inside(observed_at, start, end, as_of):
    return ((start is None or observed_at >= start)
            and (end is None or observed_at <= end)
            and (as_of is None or observed_at <= as_of))


def _source_metadata(manifest: TECGridManifest) -> dict[str, Any]:
    return {
        "source_id": manifest.grid_set_id,
        "artifact_id": manifest.artifact_id,
        "provider": manifest.provider,
        "product": manifest.product,
        "revision": manifest.revision,
        "revision_priority": manifest.revision_priority,
        "available_at": manifest.source_ingested_at.isoformat(),
        "source_checksum_sha256": manifest.source_checksum_sha256,
        "logical_sha256": manifest.logical_sha256,
        "parser_version": manifest.parser_version,
        "value_kind": manifest.value_kind,
        "source_uri": manifest.source_uri,
        "quality_flag_bits": dict(manifest.quality_flag_bits),
    }


def _rank(manifest):
    return (manifest.revision_priority, manifest.source_ingested_at,
            manifest.artifact_id, manifest.grid_set_id)


def read_archive_sequence(
    root: str | Path,
    grid_set_ids: Sequence[str] | None = None,
    *, start: datetime | str | None = None,
    end: datetime | str | None = None,
    as_of: datetime | str | None = None,
    bounds: Mapping[str, float] | None = None,
) -> xr.Dataset:
    """Read a verified, deterministic native-grid snapshot into an xarray Dataset.

    Times and ranges are UTC (inclusive). ``as_of`` excludes observations and
    source revisions not yet available; availability conservatively means local
    source ingestion time, not an invented historical release time. Highest
    revision priority, then latest available ingestion, then stable IDs wins for
    overlapping epochs. No temporal/spatial interpolation is performed.

    Local discovery visits only raw/native groups. Every selected group is fully
    verified by the existing store before its arrays are exposed. Missing RMS in
    legacy archives remains unknown; use ``read_ionex_sequence`` on source bytes
    when source-supplied RMS is required.
    """
    import numpy as np
    import xarray as xr

    start, end, as_of = _window(start, end, as_of)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise SensingError("archive root does not exist")
    directory = root / "raw" / "tec-grid" / "v1"
    if grid_set_ids is None:
        ids = sorted(path.name[:-5] for path in directory.glob("*.zarr"))
    else:
        if isinstance(grid_set_ids, (str, bytes)):
            raise SensingError("grid_set_ids must be a sequence of IDs")
        ids = sorted(set(grid_set_ids))
    selected: dict[datetime, tuple[TECGridManifest, int]] = {}
    history: dict[str, dict[str, Any]] = {}
    for identity in ids:
        if (not isinstance(identity, str) or len(identity) != 64
                or any(ch not in "0123456789abcdef" for ch in identity)):
            raise SensingError("grid_set_id must be 64 lowercase hex characters")
        metadata_path = directory / f"{identity}.zarr" / "zarr.json"
        if not metadata_path.is_file():
            raise SensingError(f"native source grid is unavailable: {identity}")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = TECGridManifest.from_dict(payload["attributes"]["ophanim_manifest"])
        except (ValueError, KeyError, TypeError) as error:
            raise SensingError(f"invalid native source manifest: {identity}") from error
        if (manifest.grid_set_id != identity or manifest.value_kind != NATIVE_VALUE_KIND
                or manifest.layout_version != TEC_GRID_LAYOUT_VERSION):
            raise SensingError("analysis requires native source grids, not interpolated estimates")
        if as_of is not None and manifest.source_ingested_at > as_of:
            continue
        for index, epoch in enumerate(manifest.epochs):
            if not _inside(epoch.observed_at, start, end, as_of):
                continue
            history[identity] = _source_metadata(manifest)
            previous = selected.get(epoch.observed_at)
            if previous is None or _rank(manifest) > _rank(previous[0]):
                selected[epoch.observed_at] = (manifest, index)
    if not selected:
        raise SensingError("no native observations are available in the requested window")

    # Constructor sees an existing directory; read() opens each group in mode r.
    store = ZarrTECGridStore(root)
    loaded = {}
    latitudes = longitudes = None
    quality_names: set[str] = set()
    for manifest, _ in selected.values():
        if manifest.grid_set_id in loaded:
            continue
        result = store.read(manifest.grid_set_id)
        if result.manifest != manifest:
            raise SensingError("source manifest changed during snapshot acquisition")
        if latitudes is None:
            latitudes, longitudes = result.latitudes_degrees, result.longitudes_degrees
        elif (not np.array_equal(latitudes, result.latitudes_degrees)
              or not np.array_equal(longitudes, result.longitudes_degrees)):
            raise SensingError("native source axes differ; explicit regridding is required")
        quality_names.update(name for name, _ in manifest.quality_flag_bits)
        loaded[manifest.grid_set_id] = result
    if len(quality_names) > 32:
        raise SensingError("combined source quality flags exceed uint32 capacity")
    quality_bits = {name: index for index, name in enumerate(sorted(quality_names))}
    times = sorted(selected)
    frames, masks, qualities, available, source_ids = [], [], [], [], []
    for observed_at in times:
        manifest, index = selected[observed_at]
        result = loaded[manifest.grid_set_id]
        frames.append(result.vtec_tecu[index])
        masks.append(result.valid[index])
        mapped_quality = np.zeros(result.valid[index].shape, dtype=np.uint32)
        for name, old_bit in manifest.quality_flag_bits:
            flagged = (result.quality_mask[index] & np.uint32(1 << old_bit)) != 0
            mapped_quality[flagged] |= np.uint32(1 << quality_bits[name])
        qualities.append(mapped_quality)
        available.append(manifest.source_ingested_at)
        source_ids.append(manifest.grid_set_id)
    provenance = [history[identity] for identity in sorted(history)]
    dataset = xr.Dataset(
        {
            "tec": (("time", "lat", "lon"), np.stack(frames)),
            "observed_mask": (("time", "lat", "lon"), np.stack(masks)),
            "quality_mask": (("time", "lat", "lon"), np.stack(qualities)),
            "source_rms_tecu": (("time", "lat", "lon"), np.full(np.stack(frames).shape, np.nan, dtype=np.float32)),
            "available_at": ("time", _datetime_array(available)),
            "source_id": ("time", np.asarray(source_ids, dtype=str)),
        },
        coords={"time": _datetime_array(times), "lat": latitudes, "lon": longitudes},
    )
    return _finish_dataset(dataset, provenance, quality_bits, start, end, as_of, bounds)


def _datetime_array(values):
    import numpy as np
    return np.asarray([value.astimezone(UTC).replace(tzinfo=None) for value in values], dtype="datetime64[us]")


def _spacing(values):
    import numpy as np
    delta = np.diff(np.asarray(values, dtype=float))
    if not len(delta):
        return None
    if not np.allclose(delta, delta[0], rtol=1e-8, atol=1e-8):
        raise SensingError("native source axes must be regular")
    return float(abs(delta[0]))


def _finish_dataset(dataset, provenance, quality_bits, start, end, as_of, bounds):
    import numpy as np

    lat_spacing = _spacing(dataset.lat.values)
    lon_spacing = _spacing(dataset.lon.values)
    if bounds is not None:
        try:
            south, north, west, east = (float(bounds[name]) for name in ("south", "north", "west", "east"))
        except (TypeError, ValueError, KeyError) as error:
            raise SensingError("bounds require south, north, west, and east") from error
        if (not all(math.isfinite(v) for v in (south, north, west, east))
                or not -90 <= south <= north <= 90
                or not -180 <= west <= 180 or not -180 <= east <= 180):
            raise SensingError("invalid geographic bounds")
        lat_valid = (dataset.lat >= south) & (dataset.lat <= north)
        lon_valid = (((dataset.lon >= west) & (dataset.lon <= east)) if west <= east
                     else ((dataset.lon >= west) | (dataset.lon <= east)))
        dataset = dataset.isel(lat=np.flatnonzero(lat_valid.values), lon=np.flatnonzero(lon_valid.values))
        if not dataset.sizes["lat"] or not dataset.sizes["lon"]:
            raise SensingError("analysis region contains no native source nodes")
    mean_lat = float(dataset.lat.mean())
    radius_m = 6_371_008.8
    spacing_m = math.pi * radius_m / 180
    time_deltas = np.diff(dataset.time.values).astype("timedelta64[us]").astype(float) / 1e6
    cadence = (float(time_deltas[0]) if len(time_deltas) and np.allclose(time_deltas, time_deltas[0]) else None)
    selected_by_time = [
        {"time": np.datetime_as_string(time, unit="us") + "Z",
         "source_id": str(source),
         "available_at": np.datetime_as_string(available, unit="us") + "Z"}
        for time, source, available in zip(dataset.time.values, dataset.source_id.values,
                                           dataset.available_at.values)
    ]
    metadata = {
        "native_lat_spacing_deg": lat_spacing,
        "native_lon_spacing_deg": lon_spacing,
        "native_dx_m": None if lon_spacing is None else lon_spacing * spacing_m * math.cos(math.radians(mean_lat)),
        "native_dy_m": None if lat_spacing is None else lat_spacing * spacing_m,
        "spacing_reference_lat_deg": mean_lat,
        "spacing_method": "spherical grid spacing at region mean latitude; not resolving power",
        "effective_resolution_m": None,
        "effective_resolution_status": "unknown",
        "cadence_seconds": cadence,
        "observed_time_intervals_seconds": sorted(set(time_deltas.tolist())),
        "regular_time_axis": bool(cadence is not None),
        "source_ids": sorted(set(dataset.source_id.values.tolist())),
        "selected_by_time": selected_by_time,
        "revision_history": provenance,
        "quality_flag_bits": quality_bits,
        "available_at_definition": "local source ingestion time",
        "observed_mask_definition": "present in source product; not a direct instrument observation at each node",
    }
    identity_payload = {
        "contract": "ophanim-sensing-sequence/1", "sources": provenance,
        "selected": list(zip([str(t) for t in dataset.time.values], dataset.source_id.values.tolist())),
        "bounds": None if bounds is None else {k: float(v) for k, v in bounds.items()},
        "start": None if start is None else start.isoformat(),
        "end": None if end is None else end.isoformat(),
        "as_of": None if as_of is None else as_of.isoformat(),
    }
    dataset.attrs.update(
        schema_version="ophanim-sensing-sequence/1", source_kind="native",
        source_metadata=metadata, timezone="UTC",
        analysis_mode="causal" if as_of is not None else "retrospective",
        as_of=None if as_of is None else as_of.isoformat(),
        snapshot_id=sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
    )
    dataset.tec.attrs.update(units="TECU", semantic="native gridded source TEC product", origin="source_product")
    dataset.observed_mask.attrs.update(semantic=metadata["observed_mask_definition"], origin="source_availability")
    dataset.quality_mask.attrs.update(semantic="source quality flags; not a calibrated confidence", flag_bits=quality_bits)
    dataset.source_rms_tecu.attrs.update(units="TECU", semantic="source-supplied RMS estimate; NaN means unavailable; not confidence")
    dataset.available_at.attrs.update(timezone="UTC", semantic=metadata["available_at_definition"])
    dataset.time.attrs.update(timezone="UTC", semantic="source epoch")
    dataset.lat.attrs.update(units="degrees_north")
    dataset.lon.attrs.update(units="degrees_east")
    return dataset
