"""Immutable Zarr storage contracts for dense, multi-epoch TEC grids.

PostgreSQL is the discovery and transaction plane; this module owns the dense
numeric plane.  A grid is written once under a deterministic identifier and is
only made discoverable after a catalog successfully publishes its manifest.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

from ophanim.domain import SourceArtifact, TECObservation


TEC_GRID_LAYOUT_VERSION = "ophanim-tec-grid/1"
CORE_TEC_GRID_LAYOUT_VERSION = "ophanim-tec-core-half-degree/1"
CORE_TEC_GRID_DEFINITION_SOURCE = "ophanim-core-global-half-degree/1"
CORE_TEC_GRID_DERIVATION_METHOD = "ophanim-global-periodic-bilinear/1"
NATIVE_VALUE_KIND = "native_measurement"
INTERPOLATED_VALUE_KIND = "interpolated_estimate"
SUPPORTED_TEC_GRID_LAYOUT_VERSIONS = frozenset(
    {TEC_GRID_LAYOUT_VERSION, CORE_TEC_GRID_LAYOUT_VERSION}
)
ZARR_FORMAT = 3
VTEC_DTYPE = "float32"
QUALITY_MASK_DTYPE = "uint32"
MAX_QUALITY_FLAGS = 32
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class TECArchiveError(ValueError):
    """A TEC grid cannot be represented or safely published."""


class TECArchiveDependencyError(RuntimeError):
    """Optional array-storage dependencies are not installed."""


class TECArchiveConflictError(RuntimeError):
    """A published grid identifier already describes different content."""


@dataclass(frozen=True, slots=True)
class TECGridDefinition:
    """Explicit axes supplied by a grid-aware parser or product validator."""

    latitudes_degrees: tuple[float, ...]
    longitudes_degrees: tuple[float, ...]
    definition_source: str
    shell_height_km: float | None = None

    def __post_init__(self) -> None:
        _validate_axis(
            self.latitudes_degrees,
            name="latitude",
            minimum=-90.0,
            maximum=90.0,
            maximum_inclusive=True,
        )
        _validate_axis(
            self.longitudes_degrees,
            name="longitude",
            minimum=-180.0,
            maximum=180.0,
            maximum_inclusive=False,
        )
        if not isinstance(self.definition_source, str) or not (
            self.definition_source.strip()
        ):
            raise TECArchiveError("grid definition_source must not be empty")
        if self.shell_height_km is not None and (
            isinstance(self.shell_height_km, bool)
            or not isinstance(self.shell_height_km, (int, float))
            or not math.isfinite(self.shell_height_km)
            or self.shell_height_km < 0.0
        ):
            raise TECArchiveError(
                "grid shell_height_km must be a finite non-negative number or None"
            )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "definition_source": self.definition_source,
                "latitudes_degrees": list(self.latitudes_degrees),
                "longitudes_degrees": list(self.longitudes_degrees),
                "shell_height_km": self.shell_height_km,
            }
        )


@dataclass(frozen=True, slots=True)
class TECEpochSummary:
    """Small searchable statistics for one dense Zarr time slice."""

    time_index: int
    observed_at: datetime
    valid_cell_count: int
    missing_cell_count: int
    minimum_vtec_tecu: float | None
    maximum_vtec_tecu: float | None
    mean_vtec_tecu: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.time_index, int)
            or isinstance(self.time_index, bool)
            or self.time_index < 0
        ):
            raise TECArchiveError("epoch time_index must be a non-negative integer")
        _aware_utc(self.observed_at, name="epoch observed_at")
        for name, value in (
            ("valid_cell_count", self.valid_cell_count),
            ("missing_cell_count", self.missing_cell_count),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise TECArchiveError(f"epoch {name} must be a non-negative integer")
        statistics = (
            self.minimum_vtec_tecu,
            self.maximum_vtec_tecu,
            self.mean_vtec_tecu,
        )
        if self.valid_cell_count == 0:
            if any(value is not None for value in statistics):
                raise TECArchiveError(
                    "an empty epoch must not contain VTEC statistics"
                )
            return
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in statistics
        ):
            raise TECArchiveError(
                "a non-empty epoch requires finite VTEC statistics"
            )
        minimum, maximum, mean = statistics
        if not minimum <= mean <= maximum:
            raise TECArchiveError(
                "epoch VTEC statistics must satisfy minimum <= mean <= maximum"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_index": self.time_index,
            "observed_at": _datetime_text(self.observed_at),
            "valid_cell_count": self.valid_cell_count,
            "missing_cell_count": self.missing_cell_count,
            "minimum_vtec_tecu": self.minimum_vtec_tecu,
            "maximum_vtec_tecu": self.maximum_vtec_tecu,
            "mean_vtec_tecu": self.mean_vtec_tecu,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TECEpochSummary:
        return cls(
            time_index=int(payload["time_index"]),
            observed_at=_parse_datetime(payload["observed_at"]),
            valid_cell_count=int(payload["valid_cell_count"]),
            missing_cell_count=int(payload["missing_cell_count"]),
            minimum_vtec_tecu=_optional_float(payload["minimum_vtec_tecu"]),
            maximum_vtec_tecu=_optional_float(payload["maximum_vtec_tecu"]),
            mean_vtec_tecu=_optional_float(payload["mean_vtec_tecu"]),
        )


@dataclass(frozen=True, slots=True)
class TECGridManifest:
    """Self-contained catalog record for one immutable Zarr grid group."""

    grid_set_id: str
    artifact_id: str
    provider: str
    product: str
    parser_version: str
    revision_priority: int
    revision: str | None
    source_uri: str | None
    source_checksum_sha256: str
    source_storage_ref: str
    source_ingested_at: datetime
    storage_ref: str
    layout_version: str
    zarr_format: int
    grid_fingerprint: str
    logical_sha256: str
    shape: tuple[int, int, int]
    chunk_shape: tuple[int, int, int]
    vtec_dtype: str
    quality_mask_dtype: str
    quality_flag_bits: tuple[tuple[str, int], ...]
    definition_source: str
    shell_height_km: float | None
    archived_at: datetime
    epochs: tuple[TECEpochSummary, ...]
    value_kind: str = NATIVE_VALUE_KIND
    source_grid_set_id: str | None = None
    source_grid_logical_sha256: str | None = None
    derivation_method: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_id", self.artifact_id),
            ("provider", self.provider),
            ("product", self.product),
            ("parser_version", self.parser_version),
            ("source_storage_ref", self.source_storage_ref),
            ("storage_ref", self.storage_ref),
            ("definition_source", self.definition_source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TECArchiveError(f"manifest {name} must not be empty")
        for name, value in (
            ("revision", self.revision),
            ("source_uri", self.source_uri),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise TECArchiveError(
                    f"manifest {name} must be non-empty or null"
                )
        if (
            not isinstance(self.revision_priority, int)
            or isinstance(self.revision_priority, bool)
            or self.revision_priority < 0
        ):
            raise TECArchiveError(
                "manifest revision_priority must be a non-negative integer"
            )
        for name, value in (
            ("grid_set_id", self.grid_set_id),
            ("source_checksum_sha256", self.source_checksum_sha256),
            ("grid_fingerprint", self.grid_fingerprint),
            ("logical_sha256", self.logical_sha256),
        ):
            _validate_lower_hex(value, name=name)
        if self.layout_version not in SUPPORTED_TEC_GRID_LAYOUT_VERSIONS:
            raise TECArchiveError("manifest layout_version is unsupported")
        if self.value_kind not in {
            NATIVE_VALUE_KIND,
            INTERPOLATED_VALUE_KIND,
        }:
            raise TECArchiveError("manifest value_kind is unsupported")
        if self.layout_version == TEC_GRID_LAYOUT_VERSION:
            if self.value_kind != NATIVE_VALUE_KIND or any(
                value is not None
                for value in (
                    self.source_grid_set_id,
                    self.source_grid_logical_sha256,
                    self.derivation_method,
                )
            ):
                raise TECArchiveError(
                    "native-grid manifests cannot contain derived-grid provenance"
                )
        else:
            if self.value_kind != INTERPOLATED_VALUE_KIND:
                raise TECArchiveError(
                    "core-grid manifests must identify interpolated estimates"
                )
            for name, value in (
                ("source_grid_set_id", self.source_grid_set_id),
                ("source_grid_logical_sha256", self.source_grid_logical_sha256),
            ):
                _validate_lower_hex(value, name=name)
            if self.derivation_method != CORE_TEC_GRID_DERIVATION_METHOD:
                raise TECArchiveError(
                    "core-grid manifest derivation_method is unsupported"
                )
            expected_source_grid_set_id = sha256(
                f"{self.artifact_id}\0{TEC_GRID_LAYOUT_VERSION}".encode("utf-8")
            ).hexdigest()
            if self.source_grid_set_id != expected_source_grid_set_id:
                raise TECArchiveError(
                    "core-grid source_grid_set_id does not match the native layout"
                )
        if self.zarr_format != ZARR_FORMAT:
            raise TECArchiveError("manifest zarr_format is unsupported")
        _validate_integer_triple(self.shape, name="shape")
        _validate_integer_triple(self.chunk_shape, name="chunk_shape")
        if any(
            chunk > dimension
            for chunk, dimension in zip(self.chunk_shape, self.shape)
        ):
            raise TECArchiveError("manifest chunks cannot exceed array dimensions")
        if self.vtec_dtype != VTEC_DTYPE:
            raise TECArchiveError("manifest VTEC dtype is unsupported")
        if self.quality_mask_dtype != QUALITY_MASK_DTYPE:
            raise TECArchiveError("manifest quality-mask dtype is unsupported")
        _validate_quality_flag_bits(self.quality_flag_bits)
        if self.shell_height_km is not None and (
            isinstance(self.shell_height_km, bool)
            or not isinstance(self.shell_height_km, (int, float))
            or not math.isfinite(self.shell_height_km)
            or self.shell_height_km < 0.0
        ):
            raise TECArchiveError(
                "manifest shell_height_km must be finite and non-negative or null"
            )
        if self.layout_version == CORE_TEC_GRID_LAYOUT_VERSION:
            expected_core_grid = core_gim_grid_definition(
                shell_height_km=self.shell_height_km
            )
            if (
                self.definition_source != CORE_TEC_GRID_DEFINITION_SOURCE
                or self.shape[1:] != (351, 720)
                or self.grid_fingerprint != expected_core_grid.fingerprint
            ):
                raise TECArchiveError(
                    "core-grid manifest does not match the canonical 0.5-degree axes"
                )
        source_ingested_at = _aware_utc(
            self.source_ingested_at,
            name="manifest source_ingested_at",
        )
        archived_at = _aware_utc(
            self.archived_at,
            name="manifest archived_at",
        )
        if archived_at < source_ingested_at:
            raise TECArchiveError(
                "manifest archived_at cannot precede source ingestion"
            )
        if not isinstance(self.epochs, tuple) or len(self.epochs) != self.shape[0]:
            raise TECArchiveError(
                "manifest epoch count must match the time dimension"
            )
        expected_cells = self.shape[1] * self.shape[2]
        previous_time: datetime | None = None
        for expected_index, epoch in enumerate(self.epochs):
            if not isinstance(epoch, TECEpochSummary):
                raise TECArchiveError("manifest epochs must be epoch summaries")
            observed_at = _aware_utc(
                epoch.observed_at,
                name="manifest epoch observed_at",
            )
            if epoch.time_index != expected_index:
                raise TECArchiveError(
                    "manifest epoch indexes must be contiguous from zero"
                )
            if previous_time is not None and observed_at <= previous_time:
                raise TECArchiveError(
                    "manifest epoch timestamps must be strictly increasing"
                )
            if epoch.valid_cell_count + epoch.missing_cell_count != expected_cells:
                raise TECArchiveError(
                    "manifest epoch cell counts must match the spatial shape"
                )
            previous_time = observed_at
        expected_grid_set_id = sha256(
            f"{self.artifact_id}\0{self.layout_version}".encode("utf-8")
        ).hexdigest()
        if self.grid_set_id != expected_grid_set_id:
            raise TECArchiveError(
                "manifest grid_set_id does not match its artifact and layout"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_set_id": self.grid_set_id,
            "artifact_id": self.artifact_id,
            "provider": self.provider,
            "product": self.product,
            "parser_version": self.parser_version,
            "revision_priority": self.revision_priority,
            "revision": self.revision,
            "source_uri": self.source_uri,
            "source_checksum_sha256": self.source_checksum_sha256,
            "source_storage_ref": self.source_storage_ref,
            "source_ingested_at": _datetime_text(self.source_ingested_at),
            "storage_ref": self.storage_ref,
            "layout_version": self.layout_version,
            "zarr_format": self.zarr_format,
            "grid_fingerprint": self.grid_fingerprint,
            "logical_sha256": self.logical_sha256,
            "shape": list(self.shape),
            "chunk_shape": list(self.chunk_shape),
            "vtec_dtype": self.vtec_dtype,
            "quality_mask_dtype": self.quality_mask_dtype,
            "quality_flag_bits": [list(item) for item in self.quality_flag_bits],
            "definition_source": self.definition_source,
            "shell_height_km": self.shell_height_km,
            "archived_at": _datetime_text(self.archived_at),
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "value_kind": self.value_kind,
            "source_grid_set_id": self.source_grid_set_id,
            "source_grid_logical_sha256": self.source_grid_logical_sha256,
            "derivation_method": self.derivation_method,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TECGridManifest:
        return cls(
            grid_set_id=str(payload["grid_set_id"]),
            artifact_id=str(payload["artifact_id"]),
            provider=str(payload["provider"]),
            product=str(payload["product"]),
            parser_version=str(payload["parser_version"]),
            revision_priority=int(payload["revision_priority"]),
            revision=_optional_text(payload.get("revision")),
            source_uri=_optional_text(payload.get("source_uri")),
            source_checksum_sha256=str(payload["source_checksum_sha256"]),
            source_storage_ref=str(payload["source_storage_ref"]),
            source_ingested_at=_parse_datetime(payload["source_ingested_at"]),
            storage_ref=str(payload["storage_ref"]),
            layout_version=str(payload["layout_version"]),
            zarr_format=int(payload["zarr_format"]),
            grid_fingerprint=str(payload["grid_fingerprint"]),
            logical_sha256=str(payload["logical_sha256"]),
            shape=_integer_triple(payload["shape"], name="shape"),
            chunk_shape=_integer_triple(
                payload["chunk_shape"],
                name="chunk_shape",
            ),
            vtec_dtype=str(payload["vtec_dtype"]),
            quality_mask_dtype=str(payload["quality_mask_dtype"]),
            quality_flag_bits=_quality_flag_bits(payload["quality_flag_bits"]),
            definition_source=str(payload["definition_source"]),
            shell_height_km=_optional_float(payload.get("shell_height_km")),
            archived_at=_parse_datetime(payload["archived_at"]),
            epochs=tuple(
                TECEpochSummary.from_dict(item) for item in payload["epochs"]
            ),
            value_kind=str(payload.get("value_kind", NATIVE_VALUE_KIND)),
            source_grid_set_id=_optional_text(payload.get("source_grid_set_id")),
            source_grid_logical_sha256=_optional_text(
                payload.get("source_grid_logical_sha256")
            ),
            derivation_method=_optional_text(payload.get("derivation_method")),
        )


@dataclass(frozen=True, slots=True)
class TECGridWriteResult:
    manifest: TECGridManifest
    already_present: bool


@dataclass(frozen=True, slots=True)
class TECGridReadResult:
    """One verified immutable dense grid loaded for model consumption."""

    manifest: TECGridManifest
    times_utc_microseconds: Any
    latitudes_degrees: Any
    longitudes_degrees: Any
    vtec_tecu: Any
    valid: Any
    quality_mask: Any


@dataclass(frozen=True, slots=True)
class CatalogPublishResult:
    manifest: TECGridManifest
    already_present: bool


@dataclass(frozen=True, slots=True)
class TECArchiveResult:
    manifest: TECGridManifest
    grid_already_present: bool
    catalog_already_present: bool
    core_manifest: TECGridManifest | None = None
    core_grid_already_present: bool | None = None
    core_catalog_already_present: bool | None = None


class TECGridCatalog(Protocol):
    def publish(self, manifest: TECGridManifest) -> CatalogPublishResult: ...

    def close(self) -> None: ...


class ZarrTECGridStore:
    """Write immutable Zarr v3 groups to a local filesystem root."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Any | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        try:
            import numpy as np
            import zarr
            from zarr.codecs import BloscCodec, BytesCodec, Crc32cCodec
        except ImportError as error:
            raise TECArchiveDependencyError(
                "Zarr archive support requires the 'data' optional dependencies"
            ) from error
        self._np = np
        self._zarr = zarr
        self._blosc_codec = BloscCodec
        self._bytes_codec = BytesCodec
        self._crc32c_codec = Crc32cCodec

    def read(
        self,
        grid_set_id: str,
        *,
        layout_version: str = TEC_GRID_LAYOUT_VERSION,
    ) -> TECGridReadResult:
        """Load one immutable grid after verifying its complete contract."""

        path = self._grid_path(grid_set_id, layout_version)
        storage_ref = path.as_uri()
        result = self._read_manifest(
            path,
            expected_storage_ref=storage_ref,
            include_arrays=True,
        )
        assert isinstance(result, TECGridReadResult)
        if (
            result.manifest.grid_set_id != grid_set_id
            or result.manifest.layout_version != layout_version
        ):
            raise TECArchiveConflictError(
                "Zarr manifest identity differs from the requested grid"
            )
        return result

    def write(
        self,
        *,
        artifact: SourceArtifact,
        observations: Iterable[TECObservation],
        grid: TECGridDefinition,
    ) -> TECGridWriteResult:
        prepared = self._prepare(
            artifact=artifact,
            observations=observations,
            grid=grid,
        )
        return self._commit_prepared(
            artifact=artifact,
            grid=grid,
            prepared=prepared,
        )

    def write_core_half_degree(
        self,
        *,
        artifact: SourceArtifact,
        observations: Iterable[TECObservation],
        source_grid: TECGridDefinition,
        source_manifest: TECGridManifest,
    ) -> TECGridWriteResult:
        """Derive and persist the canonical 0.5-degree working grid.

        The source observations remain in their native immutable group.  This
        second group contains estimates, is deterministically linked to that
        native group, and never extrapolates past the source latitude support.
        """

        expected_grid = standard_gim_grid_definition()
        if (
            source_grid.latitudes_degrees != expected_grid.latitudes_degrees
            or source_grid.longitudes_degrees != expected_grid.longitudes_degrees
        ):
            raise TECArchiveError(
                "the core 0.5-degree grid requires the canonical 5x2.5 GIM source"
            )
        if source_manifest.layout_version != TEC_GRID_LAYOUT_VERSION:
            raise TECArchiveError("the core grid must reference a native-grid manifest")
        if source_manifest.artifact_id != artifact.artifact_id:
            raise TECArchiveError(
                "the core grid source manifest must match the source artifact"
            )
        native = self._prepare(
            artifact=artifact,
            observations=observations,
            grid=source_grid,
        )
        self._assert_same_grid(
            source_manifest,
            artifact=artifact,
            grid=source_grid,
            prepared=native,
            storage_ref=source_manifest.storage_ref,
        )
        core_grid = core_gim_grid_definition(
            shell_height_km=source_grid.shell_height_km
        )
        prepared = self._derive_core_half_degree(
            artifact=artifact,
            source=native,
            source_manifest=source_manifest,
            grid=core_grid,
        )
        return self._commit_prepared(
            artifact=artifact,
            grid=core_grid,
            prepared=prepared,
        )

    def _commit_prepared(
        self,
        *,
        artifact: SourceArtifact,
        grid: TECGridDefinition,
        prepared: dict[str, Any],
    ) -> TECGridWriteResult:
        final_path = self._grid_path(
            prepared["grid_set_id"],
            prepared["layout_version"],
        )
        if final_path.exists():
            manifest = self._read_manifest(
                final_path,
                expected_storage_ref=final_path.as_uri(),
            )
            self._assert_same_grid(
                manifest,
                artifact=artifact,
                grid=grid,
                prepared=prepared,
                storage_ref=final_path.as_uri(),
            )
            return TECGridWriteResult(manifest=manifest, already_present=True)

        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f".{prepared['grid_set_id']}.staging-",
                dir=final_path.parent,
            )
        )
        try:
            archived_at = self._clock()
            if archived_at.tzinfo is None or archived_at.utcoffset() is None:
                raise TECArchiveError("archive clock must return an aware datetime")
            manifest = self._manifest(
                artifact=artifact,
                grid=grid,
                prepared=prepared,
                storage_ref=final_path.as_uri(),
                archived_at=archived_at,
            )
            self._write_group(temporary_path, prepared, manifest)
            verified = self._read_manifest(temporary_path)
            if verified != manifest:
                raise RuntimeError("staged Zarr manifest did not round-trip")
            _fsync_tree(temporary_path)
            try:
                temporary_path.rename(final_path)
            except OSError:
                if not final_path.exists():
                    raise
                existing = self._read_manifest(
                    final_path,
                    expected_storage_ref=final_path.as_uri(),
                )
                self._assert_same_grid(
                    existing,
                    artifact=artifact,
                    grid=grid,
                    prepared=prepared,
                    storage_ref=final_path.as_uri(),
                )
                return TECGridWriteResult(
                    manifest=existing,
                    already_present=True,
                )
            _fsync_directory(final_path.parent)
            return TECGridWriteResult(manifest=manifest, already_present=False)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

    def observations_for_epoch(
        self,
        manifest: TECGridManifest,
        observed_at: datetime,
    ) -> tuple[TECObservation, ...]:
        if manifest.value_kind != NATIVE_VALUE_KIND:
            raise TECArchiveError(
                "derived grids cannot be converted to provenance-free TECObservation rows"
            )
        target = _aware_utc(observed_at, name="observed_at")
        epoch = next(
            (item for item in manifest.epochs if item.observed_at == target),
            None,
        )
        if epoch is None:
            return ()
        group = self._zarr.open_group(
            store=str(
                self._grid_path(manifest.grid_set_id, manifest.layout_version)
            ),
            mode="r",
            use_consolidated=False,
        )
        latitudes = group["latitude"][:]
        longitudes = group["longitude"][:]
        values = group["vtec"][epoch.time_index, :, :]
        valid = group["valid"][epoch.time_index, :, :]
        masks = group["quality_mask"][epoch.time_index, :, :]
        flags_by_bit = {bit: flag for flag, bit in manifest.quality_flag_bits}
        observations: list[TECObservation] = []
        for latitude_index, longitude_index in self._np.argwhere(valid):
            mask = int(masks[latitude_index, longitude_index])
            flags = tuple(
                flags_by_bit[bit]
                for bit in sorted(flags_by_bit)
                if mask & (1 << bit)
            )
            observations.append(
                TECObservation(
                    artifact_id=manifest.artifact_id,
                    observed_at=target,
                    latitude_degrees=float(latitudes[latitude_index]),
                    longitude_degrees=float(longitudes[longitude_index]),
                    vtec_tecu=float(values[latitude_index, longitude_index]),
                    quality_flags=flags,
                )
            )
        return tuple(observations)

    def _prepare(
        self,
        *,
        artifact: SourceArtifact,
        observations: Iterable[TECObservation],
        grid: TECGridDefinition,
    ) -> dict[str, Any]:
        candidates = tuple(observations)
        if not candidates:
            raise TECArchiveError("cannot archive an empty TEC observation set")
        if not isinstance(artifact.storage_ref, str) or not artifact.storage_ref.strip():
            raise TECArchiveError(
                "source artifact must have a committed non-empty storage_ref"
            )
        for item in candidates:
            if not isinstance(item.quality_flags, tuple):
                raise TECArchiveError("quality_flags must be a tuple of strings")
            if any(
                not isinstance(flag, str) or not flag
                for flag in item.quality_flags
            ):
                raise TECArchiveError("quality flags must be non-empty strings")
            if len(set(item.quality_flags)) != len(item.quality_flags):
                raise TECArchiveError(
                    "quality flags must be unique within each TEC observation"
                )
        latitudes = self._np.asarray(grid.latitudes_degrees, dtype="<f8")
        longitudes = self._np.asarray(grid.longitudes_degrees, dtype="<f8")
        latitude_indexes = {
            float(value): index for index, value in enumerate(latitudes)
        }
        longitude_indexes = {
            float(value): index for index, value in enumerate(longitudes)
        }
        epochs = tuple(
            sorted(
                {
                    _aware_utc(item.observed_at, name="observation observed_at")
                    for item in candidates
                }
            )
        )
        epoch_indexes = {value: index for index, value in enumerate(epochs)}
        shape = (len(epochs), len(latitudes), len(longitudes))
        vtec = self._np.full(shape, self._np.nan, dtype="<f4")
        valid = self._np.zeros(shape, dtype="|b1")
        all_flags = tuple(
            sorted({flag for item in candidates for flag in item.quality_flags})
        )
        if len(all_flags) > MAX_QUALITY_FLAGS:
            raise TECArchiveError(
                f"at most {MAX_QUALITY_FLAGS} distinct quality flags are supported"
            )
        quality_flag_bits = tuple(
            (flag, index) for index, flag in enumerate(all_flags)
        )
        bits_by_flag = dict(quality_flag_bits)
        quality = self._np.zeros(shape, dtype="<u4")

        for item in candidates:
            if item.artifact_id != artifact.artifact_id:
                raise TECArchiveError(
                    "all archived observations must match the source artifact"
                )
            latitude = float(item.latitude_degrees)
            longitude = float(item.longitude_degrees)
            if latitude not in latitude_indexes or longitude not in longitude_indexes:
                raise TECArchiveError(
                    "observation coordinate is absent from the explicit grid: "
                    f"({latitude:g}, {longitude:g})"
                )
            value = float(item.vtec_tecu)
            if not math.isfinite(value):
                raise TECArchiveError("archived VTEC values must be finite")
            time_index = epoch_indexes[
                _aware_utc(item.observed_at, name="observation observed_at")
            ]
            latitude_index = latitude_indexes[latitude]
            longitude_index = longitude_indexes[longitude]
            position = (time_index, latitude_index, longitude_index)
            if bool(valid[position]):
                raise TECArchiveError(
                    "duplicate TEC coordinate in one source epoch: "
                    f"{item.observed_at.isoformat()} ({latitude:g}, {longitude:g})"
                )
            converted = self._np.float32(value)
            if not self._np.isfinite(converted):
                raise TECArchiveError("VTEC value cannot be represented as float32")
            vtec[position] = converted
            valid[position] = True
            mask = 0
            for flag in item.quality_flags:
                if not isinstance(flag, str) or not flag:
                    raise TECArchiveError(
                        "quality flags must be non-empty strings"
                    )
                mask |= 1 << bits_by_flag[flag]
            quality[position] = mask

        times_us = self._np.asarray(
            [_datetime_microseconds(value) for value in epochs],
            dtype="<i8",
        )
        chunk_shape = (
            1,
            min(len(latitudes), 256),
            min(len(longitudes), 256),
        )
        logical_sha256 = _array_logical_sha256(
            artifact=artifact,
            grid=grid,
            layout_version=TEC_GRID_LAYOUT_VERSION,
            value_kind=NATIVE_VALUE_KIND,
            source_grid_set_id=None,
            source_grid_logical_sha256=None,
            derivation_method=None,
            times_us=times_us,
            latitudes=latitudes,
            longitudes=longitudes,
            vtec=vtec,
            valid=valid,
            quality=quality,
            quality_flag_bits=quality_flag_bits,
        )
        grid_set_id = sha256(
            f"{artifact.artifact_id}\0{TEC_GRID_LAYOUT_VERSION}".encode("utf-8")
        ).hexdigest()
        summaries = _epoch_summaries(
            vtec=vtec,
            valid=valid,
            epochs=epochs,
        )
        return {
            "grid_set_id": grid_set_id,
            "layout_version": TEC_GRID_LAYOUT_VERSION,
            "value_kind": NATIVE_VALUE_KIND,
            "source_grid_set_id": None,
            "source_grid_logical_sha256": None,
            "derivation_method": None,
            "logical_sha256": logical_sha256,
            "shape": shape,
            "chunk_shape": chunk_shape,
            "times_us": times_us,
            "latitudes": latitudes,
            "longitudes": longitudes,
            "vtec": vtec,
            "valid": valid,
            "quality": quality,
            "quality_flag_bits": quality_flag_bits,
            "epochs": summaries,
        }

    def _derive_core_half_degree(
        self,
        *,
        artifact: SourceArtifact,
        source: dict[str, Any],
        source_manifest: TECGridManifest,
        grid: TECGridDefinition,
    ) -> dict[str, Any]:
        """Vectorize bilinear estimates while respecting missing source support."""

        latitudes = self._np.asarray(grid.latitudes_degrees, dtype="<f8")
        longitudes = self._np.asarray(grid.longitudes_degrees, dtype="<f8")
        epoch_count = int(source["shape"][0])
        shape = (epoch_count, len(latitudes), len(longitudes))
        vtec = self._np.full(shape, self._np.nan, dtype="<f4")
        valid = self._np.zeros(shape, dtype="|b1")
        quality = self._np.zeros(shape, dtype="<u4")

        # Both canonical axes are exact integer multiples of the output step.
        # This avoids coordinate-search drift and makes native-node identity
        # explicit: every fifth latitude and every tenth longitude is measured.
        latitude_offsets = self._np.arange(len(latitudes), dtype="<i8")
        longitude_offsets = self._np.arange(len(longitudes), dtype="<i8")
        south_indexes = latitude_offsets // 5
        north_indexes = self._np.minimum(south_indexes + 1, 70)
        latitude_remainders = latitude_offsets % 5
        latitude_weights = latitude_remainders.astype("<f8") / 5.0
        latitude_exact = latitude_remainders == 0
        west_indexes = longitude_offsets // 10
        east_indexes = (west_indexes + 1) % 72
        longitude_remainders = longitude_offsets % 10
        longitude_weights = longitude_remainders.astype("<f8") / 10.0
        longitude_exact = longitude_remainders == 0

        for time_index in range(epoch_count):
            source_vtec = source["vtec"][time_index]
            source_valid = source["valid"][time_index]
            source_quality = source["quality"][time_index]

            southwest = source_vtec[
                south_indexes[:, None], west_indexes[None, :]
            ].astype("<f8")
            southeast = source_vtec[
                south_indexes[:, None], east_indexes[None, :]
            ].astype("<f8")
            northwest = source_vtec[
                north_indexes[:, None], west_indexes[None, :]
            ].astype("<f8")
            northeast = source_vtec[
                north_indexes[:, None], east_indexes[None, :]
            ].astype("<f8")
            south_values = self._np.where(
                longitude_exact[None, :],
                southwest,
                southwest
                + (southeast - southwest) * longitude_weights[None, :],
            )
            north_values = self._np.where(
                longitude_exact[None, :],
                northwest,
                northwest
                + (northeast - northwest) * longitude_weights[None, :],
            )
            estimates = self._np.where(
                latitude_exact[:, None],
                south_values,
                south_values
                + (north_values - south_values) * latitude_weights[:, None],
            ).astype("<f4")

            southwest_valid = source_valid[
                south_indexes[:, None], west_indexes[None, :]
            ]
            southeast_valid = source_valid[
                south_indexes[:, None], east_indexes[None, :]
            ]
            northwest_valid = source_valid[
                north_indexes[:, None], west_indexes[None, :]
            ]
            northeast_valid = source_valid[
                north_indexes[:, None], east_indexes[None, :]
            ]
            south_valid = southwest_valid & (
                longitude_exact[None, :] | southeast_valid
            )
            north_valid = northwest_valid & (
                longitude_exact[None, :] | northeast_valid
            )
            estimates_valid = south_valid & (
                latitude_exact[:, None] | north_valid
            )

            southwest_quality = source_quality[
                south_indexes[:, None], west_indexes[None, :]
            ]
            southeast_quality = source_quality[
                south_indexes[:, None], east_indexes[None, :]
            ]
            northwest_quality = source_quality[
                north_indexes[:, None], west_indexes[None, :]
            ]
            northeast_quality = source_quality[
                north_indexes[:, None], east_indexes[None, :]
            ]
            south_quality = self._np.where(
                longitude_exact[None, :],
                southwest_quality,
                southwest_quality | southeast_quality,
            )
            north_quality = self._np.where(
                longitude_exact[None, :],
                northwest_quality,
                northwest_quality | northeast_quality,
            )
            estimate_quality = self._np.where(
                latitude_exact[:, None],
                south_quality,
                south_quality | north_quality,
            ).astype("<u4")

            estimates[~estimates_valid] = self._np.nan
            estimate_quality[~estimates_valid] = 0

            # Guarantee a bit-for-bit copy at every native source node.
            estimates[::5, ::10] = source_vtec
            estimates_valid[::5, ::10] = source_valid
            estimate_quality[::5, ::10] = source_quality
            vtec[time_index] = estimates
            valid[time_index] = estimates_valid
            quality[time_index] = estimate_quality

        times_us = source["times_us"].copy()
        chunk_shape = (1, min(len(latitudes), 256), min(len(longitudes), 256))
        logical_sha256 = _array_logical_sha256(
            artifact=artifact,
            grid=grid,
            layout_version=CORE_TEC_GRID_LAYOUT_VERSION,
            value_kind=INTERPOLATED_VALUE_KIND,
            source_grid_set_id=source_manifest.grid_set_id,
            source_grid_logical_sha256=source_manifest.logical_sha256,
            derivation_method=CORE_TEC_GRID_DERIVATION_METHOD,
            times_us=times_us,
            latitudes=latitudes,
            longitudes=longitudes,
            vtec=vtec,
            valid=valid,
            quality=quality,
            quality_flag_bits=source["quality_flag_bits"],
        )
        return {
            "grid_set_id": sha256(
                f"{artifact.artifact_id}\0{CORE_TEC_GRID_LAYOUT_VERSION}".encode(
                    "utf-8"
                )
            ).hexdigest(),
            "layout_version": CORE_TEC_GRID_LAYOUT_VERSION,
            "value_kind": INTERPOLATED_VALUE_KIND,
            "source_grid_set_id": source_manifest.grid_set_id,
            "source_grid_logical_sha256": source_manifest.logical_sha256,
            "derivation_method": CORE_TEC_GRID_DERIVATION_METHOD,
            "logical_sha256": logical_sha256,
            "shape": shape,
            "chunk_shape": chunk_shape,
            "times_us": times_us,
            "latitudes": latitudes,
            "longitudes": longitudes,
            "vtec": vtec,
            "valid": valid,
            "quality": quality,
            "quality_flag_bits": source["quality_flag_bits"],
            "epochs": _epoch_summaries(
                vtec=vtec,
                valid=valid,
                epochs=tuple(item.observed_at for item in source["epochs"]),
            ),
        }

    def _manifest(
        self,
        *,
        artifact: SourceArtifact,
        grid: TECGridDefinition,
        prepared: dict[str, Any],
        storage_ref: str,
        archived_at: datetime,
    ) -> TECGridManifest:
        return TECGridManifest(
            grid_set_id=prepared["grid_set_id"],
            artifact_id=artifact.artifact_id,
            provider=artifact.provider,
            product=artifact.product,
            parser_version=artifact.parser_version,
            revision_priority=artifact.revision_priority,
            revision=artifact.revision,
            source_uri=artifact.source_uri,
            source_checksum_sha256=artifact.checksum_sha256,
            source_storage_ref=artifact.storage_ref,
            source_ingested_at=_aware_utc(
                artifact.ingested_at,
                name="source ingested_at",
            ),
            storage_ref=storage_ref,
            layout_version=prepared["layout_version"],
            zarr_format=ZARR_FORMAT,
            grid_fingerprint=grid.fingerprint,
            logical_sha256=prepared["logical_sha256"],
            shape=prepared["shape"],
            chunk_shape=prepared["chunk_shape"],
            vtec_dtype=VTEC_DTYPE,
            quality_mask_dtype=QUALITY_MASK_DTYPE,
            quality_flag_bits=prepared["quality_flag_bits"],
            definition_source=grid.definition_source,
            shell_height_km=grid.shell_height_km,
            archived_at=_aware_utc(archived_at, name="archived_at"),
            epochs=prepared["epochs"],
            value_kind=prepared["value_kind"],
            source_grid_set_id=prepared["source_grid_set_id"],
            source_grid_logical_sha256=prepared[
                "source_grid_logical_sha256"
            ],
            derivation_method=prepared["derivation_method"],
        )

    def _write_group(
        self,
        path: Path,
        prepared: dict[str, Any],
        manifest: TECGridManifest,
    ) -> None:
        group = self._zarr.create_group(
            store=str(path),
            zarr_format=ZARR_FORMAT,
            overwrite=True,
        )
        codecs = [
            self._blosc_codec(
                cname="zstd",
                clevel=5,
                shuffle="bitshuffle",
            ),
            self._crc32c_codec(),
        ]
        serializer = self._bytes_codec(endian="little")
        group.create_array(
            "time",
            data=prepared["times_us"],
            chunks=(min(len(prepared["times_us"]), 1024),),
            serializer=serializer,
            compressors=codecs,
            dimension_names=("time",),
            attributes={
                "standard_name": "time",
                "units": "microseconds since 1970-01-01T00:00:00Z",
                "calendar": "proleptic_gregorian",
            },
        )
        group.create_array(
            "latitude",
            data=prepared["latitudes"],
            chunks=(len(prepared["latitudes"]),),
            serializer=serializer,
            compressors=codecs,
            dimension_names=("latitude",),
            attributes={"standard_name": "latitude", "units": "degrees_north"},
        )
        group.create_array(
            "longitude",
            data=prepared["longitudes"],
            chunks=(len(prepared["longitudes"]),),
            serializer=serializer,
            compressors=codecs,
            dimension_names=("longitude",),
            attributes={"standard_name": "longitude", "units": "degrees_east"},
        )
        array_options = {
            "chunks": prepared["chunk_shape"],
            "serializer": serializer,
            "compressors": codecs,
            "dimension_names": ("time", "latitude", "longitude"),
        }
        group.create_array(
            "vtec",
            data=prepared["vtec"],
            fill_value=self._np.nan,
            attributes={
                "long_name": (
                    "vertical total electron content"
                    if manifest.value_kind == NATIVE_VALUE_KIND
                    else "bilinearly interpolated vertical total electron content"
                ),
                "units": "TECU",
            },
            **array_options,
        )
        group.create_array(
            "valid",
            data=prepared["valid"],
            fill_value=False,
            attributes={
                "long_name": (
                    "source value is present"
                    if manifest.value_kind == NATIVE_VALUE_KIND
                    else "all nonzero-weight native supports are present"
                )
            },
            **array_options,
        )
        group.create_array(
            "quality_mask",
            data=prepared["quality"],
            fill_value=0,
            attributes={
                "long_name": (
                    "source quality flags"
                    if manifest.value_kind == NATIVE_VALUE_KIND
                    else "bitwise union of required native-source quality flags"
                ),
                "flag_bits": {
                    flag: bit for flag, bit in manifest.quality_flag_bits
                },
            },
            **array_options,
        )
        group.attrs.update(
            {
                "title": (
                    "OPHANIM immutable native TEC grid"
                    if manifest.value_kind == NATIVE_VALUE_KIND
                    else "OPHANIM immutable derived 0.5-degree TEC grid"
                ),
                "layout_version": manifest.layout_version,
                "value_kind": manifest.value_kind,
                "ophanim_manifest": manifest.to_dict(),
            }
        )

    def _read_manifest(
        self,
        path: Path,
        *,
        expected_storage_ref: str | None = None,
        include_arrays: bool = False,
    ) -> TECGridManifest | TECGridReadResult:
        try:
            group = self._zarr.open_group(
                store=str(path),
                mode="r",
                use_consolidated=False,
            )
            payload = group.attrs["ophanim_manifest"]
            manifest = TECGridManifest.from_dict(payload)
            if group.metadata.zarr_format != ZARR_FORMAT:
                raise TECArchiveConflictError("Zarr group format is unsupported")
            if group.attrs.get("layout_version") != manifest.layout_version:
                raise TECArchiveConflictError(
                    "Zarr group layout version differs from its contract"
                )
            if group.attrs.get("value_kind", NATIVE_VALUE_KIND) != manifest.value_kind:
                raise TECArchiveConflictError(
                    "Zarr group value kind differs from its manifest"
                )
            if (
                expected_storage_ref is not None
                and manifest.storage_ref != expected_storage_ref
            ):
                raise TECArchiveConflictError(
                    "Zarr manifest storage reference differs from its location"
                )

            time_count, latitude_count, longitude_count = manifest.shape
            dense_dimensions = ("time", "latitude", "longitude")
            array_contracts = {
                "time": (
                    (time_count,),
                    "int64",
                    (min(time_count, 1024),),
                    ("time",),
                ),
                "latitude": (
                    (latitude_count,),
                    "float64",
                    (latitude_count,),
                    ("latitude",),
                ),
                "longitude": (
                    (longitude_count,),
                    "float64",
                    (longitude_count,),
                    ("longitude",),
                ),
                "vtec": (
                    manifest.shape,
                    VTEC_DTYPE,
                    manifest.chunk_shape,
                    dense_dimensions,
                ),
                "valid": (
                    manifest.shape,
                    "bool",
                    manifest.chunk_shape,
                    dense_dimensions,
                ),
                "quality_mask": (
                    manifest.shape,
                    QUALITY_MASK_DTYPE,
                    manifest.chunk_shape,
                    dense_dimensions,
                ),
            }
            if set(group.array_keys()) != set(array_contracts):
                raise TECArchiveConflictError(
                    "Zarr group does not contain the expected arrays"
                )
            for name, (shape, dtype, chunks, dimensions) in array_contracts.items():
                array = group[name]
                if (
                    tuple(array.shape) != shape
                    or str(array.dtype) != dtype
                    or tuple(array.chunks) != chunks
                    or tuple(array.metadata.dimension_names or ()) != dimensions
                ):
                    raise TECArchiveConflictError(
                        f"Zarr {name} metadata differs from its manifest"
                    )

            times_us = self._np.asarray(group["time"][:], dtype="<i8")
            latitudes = self._np.asarray(group["latitude"][:], dtype="<f8")
            longitudes = self._np.asarray(group["longitude"][:], dtype="<f8")
            vtec = self._np.asarray(group["vtec"][:], dtype="<f4")
            valid = self._np.asarray(group["valid"][:], dtype="|b1")
            quality = self._np.asarray(group["quality_mask"][:], dtype="<u4")
            epochs = tuple(item.observed_at for item in manifest.epochs)
            expected_times = self._np.asarray(
                [_datetime_microseconds(value) for value in epochs],
                dtype="<i8",
            )
            if not self._np.array_equal(times_us, expected_times):
                raise TECArchiveConflictError(
                    "Zarr time coordinates differ from its manifest"
                )
            grid = TECGridDefinition(
                latitudes_degrees=tuple(float(value) for value in latitudes),
                longitudes_degrees=tuple(float(value) for value in longitudes),
                definition_source=manifest.definition_source,
                shell_height_km=manifest.shell_height_km,
            )
            if grid.fingerprint != manifest.grid_fingerprint:
                raise TECArchiveConflictError(
                    "Zarr spatial coordinates differ from its manifest"
                )
            if self._np.any(~self._np.isfinite(vtec[valid])):
                raise TECArchiveConflictError(
                    "Zarr contains a non-finite value marked as valid"
                )
            if self._np.any(~self._np.isnan(vtec[~valid])):
                raise TECArchiveConflictError(
                    "Zarr missing cells must contain the NaN fill value"
                )
            if self._np.any(quality[~valid] != 0):
                raise TECArchiveConflictError(
                    "Zarr missing cells must not carry quality flags"
                )
            defined_mask = sum(
                1 << bit for _, bit in manifest.quality_flag_bits
            )
            undefined_mask = self._np.uint32((~defined_mask) & 0xFFFF_FFFF)
            if self._np.any(quality & undefined_mask):
                raise TECArchiveConflictError(
                    "Zarr quality mask uses an undefined flag bit"
                )
            if _epoch_summaries(
                vtec=vtec,
                valid=valid,
                epochs=epochs,
            ) != manifest.epochs:
                raise TECArchiveConflictError(
                    "Zarr epoch statistics differ from its manifest"
                )
            source = SourceArtifact(
                artifact_id=manifest.artifact_id,
                provider=manifest.provider,
                product=manifest.product,
                parser_version=manifest.parser_version,
                revision_priority=manifest.revision_priority,
                checksum_sha256=manifest.source_checksum_sha256,
                storage_ref=manifest.source_storage_ref,
                ingested_at=manifest.source_ingested_at,
                revision=manifest.revision,
                source_uri=manifest.source_uri,
            )
            logical_sha256 = _array_logical_sha256(
                artifact=source,
                grid=grid,
                layout_version=manifest.layout_version,
                value_kind=manifest.value_kind,
                source_grid_set_id=manifest.source_grid_set_id,
                source_grid_logical_sha256=manifest.source_grid_logical_sha256,
                derivation_method=manifest.derivation_method,
                times_us=times_us,
                latitudes=latitudes,
                longitudes=longitudes,
                vtec=vtec,
                valid=valid,
                quality=quality,
                quality_flag_bits=manifest.quality_flag_bits,
            )
            if logical_sha256 != manifest.logical_sha256:
                raise TECArchiveConflictError(
                    "Zarr content differs from its logical checksum"
                )
            if include_arrays:
                return TECGridReadResult(
                    manifest=manifest,
                    times_utc_microseconds=times_us,
                    latitudes_degrees=latitudes,
                    longitudes_degrees=longitudes,
                    vtec_tecu=vtec,
                    valid=valid,
                    quality_mask=quality,
                )
            return manifest
        except TECArchiveConflictError:
            raise
        except Exception as error:
            raise TECArchiveConflictError(
                f"cannot validate existing Zarr grid at {path}"
            ) from error

    def _grid_path(self, grid_set_id: str, layout_version: str) -> Path:
        if len(grid_set_id) != 64 or any(
            character not in "0123456789abcdef" for character in grid_set_id
        ):
            raise TECArchiveError("grid_set_id must be 64 lowercase hex characters")
        if layout_version == TEC_GRID_LAYOUT_VERSION:
            directory = self.root / "raw" / "tec-grid" / "v1"
        elif layout_version == CORE_TEC_GRID_LAYOUT_VERSION:
            directory = self.root / "derived" / "tec-grid" / "half-degree" / "v1"
        else:
            raise TECArchiveError("grid layout version is unsupported")
        return directory / f"{grid_set_id}.zarr"

    def _assert_same_grid(
        self,
        manifest: TECGridManifest,
        *,
        artifact: SourceArtifact,
        grid: TECGridDefinition,
        prepared: dict[str, Any],
        storage_ref: str,
    ) -> None:
        expected = self._manifest(
            artifact=artifact,
            grid=grid,
            prepared=prepared,
            storage_ref=storage_ref,
            archived_at=manifest.archived_at,
        )
        if manifest != expected:
            raise TECArchiveConflictError(
                "grid_set_id already exists with different content or provenance"
            )


class BulkTECArchive:
    """Publish native grids and, when requested, derived core grids."""

    def __init__(
        self,
        *,
        grid_store: ZarrTECGridStore,
        catalog: TECGridCatalog,
    ) -> None:
        self._grid_store = grid_store
        self._catalog = catalog

    def archive(
        self,
        *,
        artifact: SourceArtifact,
        observations: Iterable[TECObservation],
        grid: TECGridDefinition,
        include_core: bool = True,
    ) -> TECArchiveResult:
        """Archive one source grid.

        Interactive map loads keep the historical default and publish both
        representations.  Long Mamba backfills pass ``include_core=False``:
        persisting a 0.5-degree interpolation for twenty years would add no
        measurements while multiplying the dense cell count about 49 times.
        """

        if not isinstance(include_core, bool):
            raise TypeError("include_core must be a boolean")
        candidates = tuple(observations)
        stored = self._grid_store.write(
            artifact=artifact,
            observations=candidates,
            grid=grid,
        )
        published = self._catalog.publish(stored.manifest)
        if not include_core:
            return TECArchiveResult(
                manifest=published.manifest,
                grid_already_present=stored.already_present,
                catalog_already_present=published.already_present,
            )
        core_stored = self._grid_store.write_core_half_degree(
            artifact=artifact,
            observations=candidates,
            source_grid=grid,
            source_manifest=published.manifest,
        )
        core_published = self._catalog.publish(core_stored.manifest)
        return TECArchiveResult(
            manifest=published.manifest,
            grid_already_present=stored.already_present,
            catalog_already_present=published.already_present,
            core_manifest=core_published.manifest,
            core_grid_already_present=core_stored.already_present,
            core_catalog_already_present=core_published.already_present,
        )

    def read_native(self, grid_set_id: str) -> TECGridReadResult:
        """Load a published native measurement grid for spatial modeling."""

        return self._grid_store.read(
            grid_set_id,
            layout_version=TEC_GRID_LAYOUT_VERSION,
        )

    def close(self) -> None:
        self._catalog.close()


def standard_gim_grid_definition() -> TECGridDefinition:
    """Return the canonical 5-degree by 2.5-degree CODE/IGS grid."""

    return TECGridDefinition(
        latitudes_degrees=tuple(index * 2.5 - 87.5 for index in range(71)),
        longitudes_degrees=tuple(index * 5.0 - 180.0 for index in range(72)),
        definition_source="validated-standard-gim-5x2.5/1",
    )


def core_gim_grid_definition(
    *,
    shell_height_km: float | None = None,
) -> TECGridDefinition:
    """Return the canonical 0.5-degree grid over native GIM latitude support."""

    return TECGridDefinition(
        latitudes_degrees=tuple(index * 0.5 - 87.5 for index in range(351)),
        longitudes_degrees=tuple(index * 0.5 - 180.0 for index in range(720)),
        definition_source=CORE_TEC_GRID_DEFINITION_SOURCE,
        shell_height_km=shell_height_km,
    )


def _validate_axis(
    values: tuple[float, ...],
    *,
    name: str,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise TECArchiveError(f"grid {name} axis must be a non-empty tuple")
    normalized: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise TECArchiveError(f"grid {name} values must be finite numbers")
        number = float(value)
        in_range = minimum <= number <= maximum
        if not maximum_inclusive:
            in_range = minimum <= number < maximum
        if not in_range:
            bracket = "]" if maximum_inclusive else ")"
            raise TECArchiveError(
                f"grid {name} values must be in [{minimum:g}, {maximum:g}{bracket}"
            )
        normalized.append(number)
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise TECArchiveError(
            f"grid {name} axis must be strictly increasing and unique"
        )


def _array_logical_sha256(
    *,
    artifact: SourceArtifact,
    grid: TECGridDefinition,
    layout_version: str,
    value_kind: str,
    source_grid_set_id: str | None,
    source_grid_logical_sha256: str | None,
    derivation_method: str | None,
    times_us: Any,
    latitudes: Any,
    longitudes: Any,
    vtec: Any,
    valid: Any,
    quality: Any,
    quality_flag_bits: tuple[tuple[str, int], ...],
) -> str:
    metadata = {
        "artifact_id": artifact.artifact_id,
        "layout_version": layout_version,
        "source_checksum_sha256": artifact.checksum_sha256,
        "grid_fingerprint": grid.fingerprint,
        "quality_flag_bits": [list(item) for item in quality_flag_bits],
    }
    # Retain the exact v1 native hash contract so groups created by earlier
    # releases remain verifiable.  Derived layouts extend the digest metadata
    # with their explicit lineage and value semantics.
    if layout_version != TEC_GRID_LAYOUT_VERSION:
        metadata.update(
            {
                "value_kind": value_kind,
                "source_grid_set_id": source_grid_set_id,
                "source_grid_logical_sha256": source_grid_logical_sha256,
                "derivation_method": derivation_method,
            }
        )
    digest = sha256(_canonical_json(metadata).encode("utf-8"))
    for name, array in (
        ("time", times_us),
        ("latitude", latitudes),
        ("longitude", longitudes),
        ("vtec", vtec),
        ("valid", valid),
        ("quality_mask", quality),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _epoch_summaries(
    *,
    vtec: Any,
    valid: Any,
    epochs: tuple[datetime, ...],
) -> tuple[TECEpochSummary, ...]:
    cells_per_epoch = int(vtec.shape[1]) * int(vtec.shape[2])
    summaries: list[TECEpochSummary] = []
    for index, observed_at in enumerate(epochs):
        epoch_values = vtec[index][valid[index]].astype("<f8")
        count = int(epoch_values.size)
        summaries.append(
            TECEpochSummary(
                time_index=index,
                observed_at=observed_at,
                valid_cell_count=count,
                missing_cell_count=cells_per_epoch - count,
                minimum_vtec_tecu=(
                    None if count == 0 else float(epoch_values.min())
                ),
                maximum_vtec_tecu=(
                    None if count == 0 else float(epoch_values.max())
                ),
                mean_vtec_tecu=(
                    None if count == 0 else float(epoch_values.mean())
                ),
            )
        )
    return tuple(summaries)


def _datetime_microseconds(value: datetime) -> int:
    delta = _aware_utc(value, name="datetime") - _UNIX_EPOCH
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or (
        value.tzinfo is None or value.utcoffset() is None
    ):
        raise TECArchiveError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _datetime_text(value: datetime) -> str:
    return _aware_utc(value, name="datetime").isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TECArchiveError("manifest datetime must be a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise TECArchiveError("manifest datetime is invalid") from error
    return _aware_utc(parsed, name="manifest datetime")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TECArchiveError("manifest numeric values must be numbers or null")
    number = float(value)
    if not math.isfinite(number):
        raise TECArchiveError("manifest numeric values must be finite or null")
    return number


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TECArchiveError("manifest optional text must be non-empty or null")
    return value


def _integer_triple(value: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TECArchiveError(f"manifest {name} must contain three integers")
    result = tuple(value)
    _validate_integer_triple(result, name=name)
    return result  # type: ignore[return-value]


def _validate_integer_triple(value: Any, *, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TECArchiveError(f"manifest {name} must contain three integers")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value
    ):
        raise TECArchiveError(f"manifest {name} values must be positive integers")


def _validate_lower_hex(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TECArchiveError(
            f"manifest {name} must be 64 lowercase hexadecimal characters"
        )


def _validate_quality_flag_bits(value: Any) -> None:
    if not isinstance(value, tuple) or len(value) > MAX_QUALITY_FLAGS:
        raise TECArchiveError("manifest quality-flag mapping is invalid")
    flags: list[str] = []
    bits: list[int] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TECArchiveError("manifest quality-flag mapping is invalid")
        flag, bit = item
        if not isinstance(flag, str) or not flag:
            raise TECArchiveError("manifest quality flags must be non-empty strings")
        if (
            not isinstance(bit, int)
            or isinstance(bit, bool)
            or not 0 <= bit < MAX_QUALITY_FLAGS
        ):
            raise TECArchiveError("manifest quality-flag bits must be in [0, 31]")
        flags.append(flag)
        bits.append(bit)
    expected = tuple(
        (flag, bit) for bit, flag in enumerate(sorted(flags))
    )
    if len(set(flags)) != len(flags) or tuple(value) != expected:
        raise TECArchiveError(
            "manifest quality flags must use unique deterministic bit positions"
        )


def _quality_flag_bits(value: Any) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise TECArchiveError("manifest quality-flag mapping is invalid")
    result: list[tuple[str, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TECArchiveError("manifest quality-flag mapping is invalid")
        flag, bit = item
        result.append((flag, bit))
    normalized = tuple(result)
    _validate_quality_flag_bits(normalized)
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Flush every staged Zarr file and directory before publication."""

    for directory, _, filenames in os.walk(root, topdown=False):
        current = Path(directory)
        for filename in filenames:
            path = current / filename
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(current)


__all__ = [
    "BulkTECArchive",
    "CatalogPublishResult",
    "CORE_TEC_GRID_DEFINITION_SOURCE",
    "CORE_TEC_GRID_DERIVATION_METHOD",
    "CORE_TEC_GRID_LAYOUT_VERSION",
    "INTERPOLATED_VALUE_KIND",
    "NATIVE_VALUE_KIND",
    "QUALITY_MASK_DTYPE",
    "TECArchiveConflictError",
    "TECArchiveDependencyError",
    "TECArchiveError",
    "TECArchiveResult",
    "TECEpochSummary",
    "TECGridCatalog",
    "TECGridDefinition",
    "TECGridManifest",
    "TECGridWriteResult",
    "TEC_GRID_LAYOUT_VERSION",
    "VTEC_DTYPE",
    "ZARR_FORMAT",
    "ZarrTECGridStore",
    "core_gim_grid_definition",
    "standard_gim_grid_definition",
]
