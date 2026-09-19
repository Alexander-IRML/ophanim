"""Dependency-free localhost desktop interface for OPHANIM.

The laptop runs the project through WSL, so this module serves a loopback-only
web application and opens it in Windows Google Chrome when available, with the
Windows default browser as a fallback.  It deliberately keeps HTTP concerns
thin: :class:`DesktopController` composes the same public workflows used by
Python callers and is independently testable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import traceback
import webbrowser
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from ophanim import __version__
from ophanim.aggregation import (
    AggregationPolicy,
    GeographicBounds,
    boundary_checksum_sha256,
)
from ophanim.bootstrap import OphanimApplication, create_sqlite_application
from ophanim.domain import (
    AnomalyDecision,
    DetectorVersion,
    Forecast,
    ForecastMode,
    ForecastStatus,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    TECObservation,
    TargetMetric,
)
from ophanim.forecasting import ForecastRequest
from ophanim.gim import (
    CodeGIMSource,
    GIMAcquisitionError,
    GIMEdition,
    GIMFetchRequest,
    GIMNotFound,
)
from ophanim.ingestion import IngestionRequest, IngestionResult, LoadedSource
from ophanim.ionex import IONEXV1Parser
from ophanim.mamba_monitor import (
    MambaMonitor,
    MambaMonitorConflict,
    MambaMonitorUnavailable,
)
from ophanim.reconciliation import ReconciliationSummary
from ophanim.regional_grid import (
    DEFAULT_MAX_OUTPUT_CELLS,
    interpolate_regional_grid,
)
from ophanim.spatial_monitor import (
    SpatialMonitor,
    SpatialMonitorConflict,
    SpatialMonitorUnavailable,
)
from ophanim.shawtynet_desktop import (
    ShawtyNetBusy,
    ShawtyNetDesktopJobs,
    ShawtyNetUnavailable,
)
from ophanim.sqlite import SQLiteUnitOfWork
from ophanim.tec_archive import (
    BulkTECArchive,
    TECArchiveResult,
    ZarrTECGridStore,
    standard_gim_grid_definition,
)
from ophanim.workflows import AggregationSummary
from ophanim.workspace import Workspace, WorkspaceBusy, WorkspaceUnavailable


APPLICATION_ID = "ophanim-desktop"
DESKTOP_API_VERSION = 3
DEFAULT_PORT = 8765
PORT_ATTEMPTS = 10
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024
MAX_CONTEXT_BYTES = 4 * 1024 * 1024
CONTEXT_FILE_VERSION = 1
AGGREGATION_ALGORITHM_REVISION = "mean-median-regional/1"
MODEL_ALGORITHM_REVISION = "persistence/1"
DETECTOR_ALGORITHM_REVISION = "residual-zscore/1"
_SAFE_SLUG = re.compile(r"[^a-z0-9]+")
_SAFE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}")
_STANDARD_GIM_COORDINATES = frozenset(
    (latitude_index * 2.5 - 87.5, longitude_index * 5.0 - 180.0)
    for latitude_index in range(71)
    for longitude_index in range(72)
)


class DesktopInputError(ValueError):
    """A desktop form submission cannot be mapped to a valid experiment."""


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    dataset_id: str
    filename: str
    provider: str
    product: str
    revision: str | None
    revision_priority: int
    source_uri: str
    bounds: GeographicBounds
    expected_cell_count: int
    inferred_cell_count: int
    global_epoch_count: int
    global_cells_per_epoch_minimum: int
    global_cells_per_epoch_maximum: int
    region: RegionVersion
    processing_version: ProcessingVersion
    ingestion: IngestionResult
    aggregation: AggregationSummary
    observations: tuple[RegionalObservation, ...]
    bulk_archive: TECArchiveResult | None


class DesktopController:
    """Stateful adapter from desktop form values to OPHANIM workflows."""

    def __init__(
        self,
        *,
        data_directory: str | Path,
        clock: Callable[[], datetime] | None = None,
        gim_source: CodeGIMSource | None = None,
        bulk_archive: BulkTECArchive | None = None,
        mamba_monitor: MambaMonitor | None = None,
        spatial_monitor: SpatialMonitor | None = None,
        shawtynet_jobs: ShawtyNetDesktopJobs | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        self._data_directory = Path(data_directory).expanduser().resolve()
        self._database = self._data_directory / "ophanim.sqlite3"
        self._artifacts = self._data_directory / "artifacts"
        self._uploads = self._data_directory / "uploads"
        self._context_file = self._data_directory / "forecast-contexts.json"
        self._uploads.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._gim_source = gim_source or CodeGIMSource(clock=self._clock)
        self._bulk_archive = bulk_archive
        self._mamba_monitor = mamba_monitor
        self._spatial_monitor = spatial_monitor
        self._shawtynet_jobs = shawtynet_jobs
        self._workspace = workspace
        self._operation_lock = threading.RLock()
        self._loaded: LoadedDataset | None = None
        self._last_result: dict[str, Any] | None = None
        self._forecast_applications: dict[str, OphanimApplication] = {}
        self._last_forecast_id: str | None = None
        self._restore_forecast_contexts()

    def state(self) -> dict[str, Any]:
        with self._operation_lock:
            return {
                "ok": True,
                "loaded": (
                    None if self._loaded is None else self._loaded_payload(self._loaded)
                ),
                "result": self._last_result,
            }

    def load(
        self,
        *,
        content: bytes,
        filename: str,
        config: dict[str, Any],
        source_uri: str | None = None,
        require_standard_gim_grid: bool = False,
        expected_product_date: date | None = None,
    ) -> dict[str, Any]:
        with self._operation_lock:
            if not content:
                raise DesktopInputError("Choose a non-empty IONEX file")
            if len(content) > MAX_UPLOAD_BYTES:
                raise DesktopInputError(
                    f"IONEX upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
                )

            provider = _text(config, "provider", default="igs", maximum=80)
            product = _text(config, "product", default="gim", maximum=80)
            revision = _optional_text(config, "revision", maximum=120)
            revision_priority = _integer(
                config,
                "revision_priority",
                default=0,
                minimum=0,
                maximum=1_000_000,
            )
            region_name = _text(
                config,
                "region_name",
                default="Central Texas",
                maximum=120,
            )
            bounds = GeographicBounds(
                south_latitude_degrees=_number(
                    config, "south_latitude_degrees", default=29.0
                ),
                west_longitude_degrees=_number(
                    config, "west_longitude_degrees", default=-100.5
                ),
                north_latitude_degrees=_number(
                    config, "north_latitude_degrees", default=32.5
                ),
                east_longitude_degrees=_number(
                    config, "east_longitude_degrees", default=-96.0
                ),
            )
            expected_override = _optional_integer(
                config,
                "expected_cell_count",
                minimum=1,
                maximum=10_000_000,
            )

            display_filename = _display_filename(filename)
            acquisition_uri = source_uri
            if acquisition_uri is None:
                acquisition_uri = str(
                    self._store_upload(content, display_filename)
                )
            placeholder_detector_id = "desktop-detector-not-configured"
            preliminary = create_sqlite_application(
                database=self._database,
                artifact_directory=self._artifacts,
                region_bounds={"desktop-preflight": bounds},
                aggregation_policy=AggregationPolicy(expected_cell_count=1),
                detector_version_id=placeholder_detector_id,
                clock=self._clock,
            )
            ingestion_request = IngestionRequest(
                provider=provider,
                product=product,
                source_uri=acquisition_uri,
                revision=revision,
                revision_priority=revision_priority,
            )
            ingestion = preliminary.ingester.ingest_loaded(
                ingestion_request,
                LoadedSource(content=content, filename=display_filename),
                observation_validator=(
                    (
                        lambda observations: _validate_standard_gim_observations(
                            observations,
                            expected_product_date=expected_product_date,
                        )
                    )
                    if require_standard_gim_grid
                    else None
                ),
            )

            with preliminary.unit_of_work_factory() as unit_of_work:
                raw_observations = tuple(
                    unit_of_work.tec_observations.for_artifact(
                        ingestion.artifact.artifact_id
                    )
                )
            global_by_epoch: dict[
                datetime, set[tuple[float, float]]
            ] = defaultdict(set)
            selected_by_epoch: dict[
                datetime, set[tuple[float, float]]
            ] = defaultdict(set)
            selected_coordinates: set[tuple[float, float]] = set()
            for observation in raw_observations:
                coordinate = (
                    observation.latitude_degrees,
                    observation.longitude_degrees,
                )
                global_by_epoch[observation.observed_at].add(coordinate)
                if bounds.contains(
                    observation.latitude_degrees,
                    observation.longitude_degrees,
                ):
                    selected_by_epoch[observation.observed_at].add(coordinate)
                    selected_coordinates.add(coordinate)
            if not global_by_epoch:
                raise RuntimeError("ingestion produced no global TEC epochs")
            global_counts = tuple(len(cells) for cells in global_by_epoch.values())
            if require_standard_gim_grid:
                _validate_standard_gim_epochs(
                    global_by_epoch,
                    expected_product_date=expected_product_date,
                )
            bulk_archive_result = None
            if self._bulk_archive is not None and require_standard_gim_grid:
                bulk_archive_result = self._bulk_archive.archive(
                    artifact=ingestion.artifact,
                    observations=raw_observations,
                    grid=standard_gim_grid_definition(),
                )
            if not selected_coordinates:
                raise DesktopInputError(
                    "The selected region contains no cells from this IONEX file"
                )
            inferred_cell_count = len(selected_coordinates)
            largest_epoch = max(len(cells) for cells in selected_by_epoch.values())
            if require_standard_gim_grid:
                standard_regional_count = sum(
                    1
                    for latitude, longitude in _standard_gim_coordinates()
                    if bounds.contains(latitude, longitude)
                )
                if standard_regional_count == 0:
                    raise DesktopInputError(
                        "The selected region does not contain a standard GIM grid cell"
                    )
            else:
                standard_regional_count = inferred_cell_count
            expected_cell_count = expected_override or standard_regional_count
            if expected_cell_count < largest_epoch:
                raise DesktopInputError(
                    "Expected cell count is smaller than the cells present in at "
                    f"least one epoch ({largest_epoch})"
                )

            boundary_hash = boundary_checksum_sha256(bounds)
            now = _aware_now(self._clock, "desktop clock")
            region_config_hash = _stable_hash(
                {
                    "name": region_name,
                    "boundary_checksum_sha256": boundary_hash,
                }
            )
            region_id = (
                f"region-{_slug(region_name)}-{region_config_hash[:12]}"
            )
            candidate_region = RegionVersion(
                region_version_id=region_id,
                name=region_name,
                boundary_ref=f"bounds-{boundary_hash[:16]}",
                boundary_checksum_sha256=boundary_hash,
                created_at=now,
            )
            processing_config_hash = _stable_hash(
                {
                    "parser_version": IONEXV1Parser.parser_version,
                    "aggregation": "mean_median",
                    "algorithm_revision": AGGREGATION_ALGORITHM_REVISION,
                    "expected_cell_count": expected_cell_count,
                    "minimum_coverage_fraction": 0.5,
                    "preferred_coverage_fraction": 0.9,
                }
            )
            candidate_processing = ProcessingVersion(
                processing_version_id=(
                    f"processing-ionex-{processing_config_hash[:16]}"
                ),
                code_revision=(
                    f"ophanim-{__version__}:{AGGREGATION_ALGORITHM_REVISION}"
                ),
                configuration_hash=processing_config_hash,
                parser_version=IONEXV1Parser.parser_version,
                created_at=now,
            )
            with preliminary.unit_of_work_factory() as unit_of_work:
                region = (
                    unit_of_work.regions.get(candidate_region.region_version_id)
                    or candidate_region
                )
                processing_version = (
                    unit_of_work.processing_versions.get(
                        candidate_processing.processing_version_id
                    )
                    or candidate_processing
                )

            application = create_sqlite_application(
                database=self._database,
                artifact_directory=self._artifacts,
                region_bounds={region.boundary_ref: bounds},
                aggregation_policy=AggregationPolicy(
                    expected_cell_count=expected_cell_count
                ),
                detector_version_id=placeholder_detector_id,
                clock=self._clock,
            )
            aggregation = application.aggregation.aggregate_artifact(
                artifact_id=ingestion.artifact.artifact_id,
                region=region,
                processing_version=processing_version,
            )
            with application.unit_of_work_factory() as unit_of_work:
                observations = tuple(
                    observation
                    for observation in unit_of_work.regional_observations.for_artifact(
                        ingestion.artifact.artifact_id
                    )
                    if observation.region_version_id == region.region_version_id
                    and observation.processing_version_id
                    == processing_version.processing_version_id
                )
            observations = tuple(sorted(observations, key=lambda item: item.observed_at))
            if not observations:
                raise RuntimeError("regional aggregation produced no observations")

            dataset_id = _stable_hash(
                {
                    "artifact_id": ingestion.artifact.artifact_id,
                    "region_version_id": region.region_version_id,
                    "processing_version_id": processing_version.processing_version_id,
                }
            )[:20]
            loaded = LoadedDataset(
                dataset_id=dataset_id,
                filename=display_filename,
                provider=provider,
                product=product,
                revision=revision,
                revision_priority=revision_priority,
                source_uri=acquisition_uri,
                bounds=bounds,
                expected_cell_count=expected_cell_count,
                inferred_cell_count=inferred_cell_count,
                global_epoch_count=len(global_by_epoch),
                global_cells_per_epoch_minimum=min(global_counts),
                global_cells_per_epoch_maximum=max(global_counts),
                region=region,
                processing_version=processing_version,
                ingestion=ingestion,
                aggregation=aggregation,
                observations=observations,
                bulk_archive=bulk_archive_result,
            )
            self._loaded = loaded
            return {"ok": True, "loaded": self._loaded_payload(loaded)}

    def close(self) -> None:
        """Release optional external data-plane resources."""

        if self._workspace is not None:
            self._workspace.close()
        if self._shawtynet_jobs is not None:
            self._shawtynet_jobs.close()
        if self._spatial_monitor is not None:
            if self._spatial_monitor.close() is False:
                # A still-running spatial worker owns both dependencies.  Leave
                # them open rather than invalidating an in-flight checkpoint.
                return
        if self._mamba_monitor is not None:
            if self._mamba_monitor.close() is False:
                return
        if self._bulk_archive is not None:
            self._bulk_archive.close()

    def _get_shawtynet_jobs(self) -> ShawtyNetDesktopJobs:
        with self._operation_lock:
            if self._shawtynet_jobs is None:
                self._shawtynet_jobs = ShawtyNetDesktopJobs(self._data_directory)
            return self._shawtynet_jobs

    def _get_workspace(self) -> Workspace:
        with self._operation_lock:
            if self._workspace is None:
                self._workspace = Workspace(self._data_directory, source=self._gim_source)
            return self._workspace

    def workspace_status(self) -> dict[str, Any]:
        return self._get_workspace().status()

    def workspace_submit(self, kind: str, config: dict[str, Any]) -> dict[str, Any]:
        return self._get_workspace().submit(kind, config)

    def workspace_cancel(self, config: dict[str, Any]) -> dict[str, Any]:
        if set(config) - {"job_id"}:
            raise DesktopInputError("Cancel accepts only the selected job identifier")
        return self._get_workspace().cancel(config.get("job_id"))

    def workspace_record(self, identity: str, kind: str) -> dict[str, Any]:
        workspace = self._get_workspace()
        return {"ok": True, kind: workspace.candidate(identity) if kind == "event" else workspace.record(identity, kind)}

    def workspace_asset(self, relative_url: str) -> tuple[bytes, str]:
        return self._get_workspace().asset(relative_url)

    def workspace_photo(self, content: bytes) -> dict[str, Any]:
        return self._get_workspace().upload_photo(content)

    def workspace_mask(self, content: bytes) -> dict[str, Any]:
        return self._get_workspace().upload_mask(content)

    def shawtynet_status(self) -> dict[str, Any]:
        return self._get_shawtynet_jobs().status()

    def analyze_shawtynet(self, config: dict[str, Any]) -> dict[str, Any]:
        return self._get_shawtynet_jobs().analyze(config)

    def shawtynet_asset(self, relative_url: str) -> tuple[bytes, str]:
        return self._get_shawtynet_jobs().asset(relative_url)

    def mamba_status(self) -> dict[str, Any]:
        """Return the independent historical anomaly-monitor status."""

        return self._get_mamba_monitor().status()

    def initialize_mamba(self, config: dict[str, Any]) -> dict[str, Any]:
        """Queue the resumable historical baseline job."""

        return self._get_mamba_monitor().initialize(config)

    def scan_mamba(self) -> dict[str, Any]:
        """Queue an incremental scan from the last successful cursor."""

        return self._get_mamba_monitor().scan()

    def _get_mamba_monitor(self) -> MambaMonitor:
        # Construct lazily so Python/API users who never open the monitor do not
        # pay for a background worker.  The browser requests status at startup,
        # which activates it for the normal desktop experience.
        with self._operation_lock:
            if self._mamba_monitor is None:
                self._mamba_monitor = MambaMonitor(
                    data_directory=self._data_directory,
                    clock=self._clock,
                    gim_source=self._gim_source,
                    bulk_archive=self._bulk_archive,
                )
            return self._mamba_monitor

    def spatial_status(self) -> dict[str, Any]:
        """Return the causal native-grid spatial baseline status."""

        return self._get_spatial_monitor().status()

    def initialize_spatial(self, config: dict[str, Any]) -> dict[str, Any]:
        """Queue spatial model training against the frozen Mamba source set."""

        return self._get_spatial_monitor().initialize(config)

    def scan_spatial(self) -> dict[str, Any]:
        """Queue a spatial scan and comparison with the regional baseline."""

        return self._get_spatial_monitor().scan()

    def _get_spatial_monitor(self) -> SpatialMonitor:
        # The spatial monitor deliberately shares both Mamba's source ledger and
        # the immutable bulk archive.  This avoids downloading or persisting a
        # second copy of the historical native grids.
        with self._operation_lock:
            if self._spatial_monitor is None:
                self._spatial_monitor = SpatialMonitor(
                    data_directory=self._data_directory,
                    mamba_monitor=self._get_mamba_monitor(),
                    bulk_archive=self._bulk_archive,
                    clock=self._clock,
                )
            return self._spatial_monitor

    def fetch_gim(self, config: dict[str, Any]) -> dict[str, Any]:
        """Download, validate, ingest, and aggregate a public CODE GIM."""

        with self._operation_lock:
            try:
                edition = GIMEdition(
                    _text(config, "edition", default="rapid").casefold()
                )
            except ValueError as error:
                raise DesktopInputError(
                    "edition must be either rapid or final"
                ) from error
            product_date = _optional_date(config, "date")
            fetched = self._gim_source.fetch(
                GIMFetchRequest(
                    edition=edition,
                    product_date=product_date,
                )
            )
            controlled_config: dict[str, Any] = {
                "provider": fetched.ingestion_request.provider,
                "product": fetched.ingestion_request.product,
                "revision": fetched.ingestion_request.revision,
                "revision_priority": (
                    fetched.ingestion_request.revision_priority
                ),
            }
            for key in (
                "region_name",
                "south_latitude_degrees",
                "west_longitude_degrees",
                "north_latitude_degrees",
                "east_longitude_degrees",
                "expected_cell_count",
            ):
                if key in config:
                    controlled_config[key] = config[key]

            response = self.load(
                content=fetched.loaded_source.content,
                filename=fetched.loaded_source.filename,
                config=controlled_config,
                source_uri=fetched.source_uri,
                require_standard_gim_grid=True,
                expected_product_date=fetched.product_date,
            )
            response["acquisition"] = {
                "source": "CODE / University of Bern",
                "edition": fetched.edition.value,
                "requested_date": (
                    None if product_date is None else product_date.isoformat()
                ),
                "resolved_date": fetched.product_date.isoformat(),
                "source_uri": fetched.source_uri,
                "resolved_uri": fetched.resolved_uri,
                "filename": fetched.loaded_source.filename,
                "byte_count": len(fetched.loaded_source.content),
                "grid": {
                    "longitude_step_degrees": 5.0,
                    "latitude_step_degrees": 2.5,
                    "unique_cells_per_epoch": 5_112,
                },
            }
            return response

    def regional_grid(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build a table of explicitly interpolated regional VTEC estimates."""

        with self._operation_lock:
            loaded = self._require_loaded(config.get("dataset_id"))
            observed_at = _parse_datetime(
                config.get("observed_at"),
                field_name="observed_at",
            )
            available_epochs = {
                observation.observed_at for observation in loaded.observations
            }
            if observed_at not in available_epochs:
                raise DesktopInputError(
                    "Regional readout epoch must be one of the loaded epochs"
                )
            latitude_step = _number(
                config,
                "latitude_step_degrees",
                default=0.5,
                minimum_exclusive=0.0,
                maximum=2.5,
            )
            longitude_step = _number(
                config,
                "longitude_step_degrees",
                default=0.5,
                minimum_exclusive=0.0,
                maximum=5.0,
            )

            with SQLiteUnitOfWork(self._database) as unit_of_work:
                source_observations = tuple(
                    unit_of_work.tec_observations.for_artifact_at(
                        loaded.ingestion.artifact.artifact_id,
                        observed_at,
                    )
                )
            if not source_observations:
                raise RuntimeError(
                    "The stored source artifact has no cells at the selected epoch"
                )

            interpolation = interpolate_regional_grid(
                observations=source_observations,
                bounds=loaded.bounds,
                latitude_step_degrees=latitude_step,
                longitude_step_degrees=longitude_step,
                max_output_cells=DEFAULT_MAX_OUTPUT_CELLS,
            )
            native = interpolation.native_grid
            return {
                "ok": True,
                "readout": {
                    "dataset_id": loaded.dataset_id,
                    "artifact_id": native.artifact_id,
                    "observed_at": _datetime_payload(native.observed_at),
                    "method": interpolation.method,
                    "method_version": interpolation.method_version,
                    "value_kind": "interpolated_estimate",
                    "source": {
                        "provider": loaded.provider,
                        "product": loaded.product,
                        "revision": loaded.revision,
                        "source_uri": loaded.source_uri,
                    },
                    "native_grid": {
                        "latitude_step_degrees": (
                            native.latitude_step_degrees
                        ),
                        "longitude_step_degrees": (
                            native.longitude_step_degrees
                        ),
                        "longitude_is_cyclic": native.longitude_is_cyclic,
                        "latitude_count": native.latitude_count,
                        "longitude_count": native.longitude_count,
                        "source_point_count": native.source_point_count,
                    },
                    "requested_grid": {
                        "latitude_step_degrees": (
                            interpolation.requested_latitude_step_degrees
                        ),
                        "longitude_step_degrees": (
                            interpolation.requested_longitude_step_degrees
                        ),
                        "latitude_count": interpolation.latitude_count,
                        "longitude_count": interpolation.longitude_count,
                    },
                    "bounds": {
                        "south": loaded.bounds.south_latitude_degrees,
                        "west": loaded.bounds.west_longitude_degrees,
                        "north": loaded.bounds.north_latitude_degrees,
                        "east": loaded.bounds.east_longitude_degrees,
                    },
                    "cell_count": interpolation.cell_count,
                    "rows": [
                        {
                            "latitude_degrees": row.latitude_degrees,
                            "longitude_degrees": row.longitude_degrees,
                            "estimated_vtec_tecu": round(
                                row.estimated_vtec_tecu,
                                12,
                            ),
                            "is_native_node": row.is_native_node,
                            "native_support_count": len(row.native_support),
                        }
                        for row in interpolation.rows
                    ],
                },
            }

    def forecast(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._operation_lock:
            loaded = self._require_loaded(config.get("dataset_id"))
            origin_at = _parse_datetime(
                config.get("forecast_origin"),
                field_name="forecast_origin",
            )
            if origin_at not in {
                observation.observed_at for observation in loaded.observations
            }:
                raise DesktopInputError(
                    "Forecast origin must be one of the loaded regional epochs"
                )
            horizon_hours = _number(
                config,
                "forecast_horizon_hours",
                default=2.0,
                minimum_exclusive=0.0,
                maximum=24.0 * 31,
            )
            valid_at = origin_at + timedelta(hours=horizon_hours)
            try:
                target_metric = TargetMetric(
                    _text(config, "target_metric", default="median_vtec")
                )
            except ValueError as error:
                raise DesktopInputError("Unsupported target metric") from error
            try:
                mode = ForecastMode(_text(config, "mode", default="hindcast"))
            except ValueError as error:
                raise DesktopInputError("Unsupported forecast mode") from error

            detector_threshold = _number(
                config,
                "detector_threshold",
                default=3.0,
                minimum_exclusive=0.0,
            )
            calibration_mean = _number(
                config,
                "calibration_residual_mean_tecu",
                default=0.0,
            )
            calibration_standard_deviation = _number(
                config,
                "calibration_residual_standard_deviation_tecu",
                default=1.0,
                minimum_exclusive=0.0,
            )
            calibration_sample_count = _integer(
                config,
                "calibration_sample_count",
                default=2,
                minimum=2,
                maximum=2_147_483_647,
            )

            version_time = _aware_now(self._clock, "desktop clock")
            model_config_hash = _stable_hash(
                {
                    "model_type": "persistence",
                    "algorithm_revision": MODEL_ALGORITHM_REVISION,
                    "minimum_coverage_fraction": 0.5,
                }
            )
            candidate_model = ModelVersion(
                model_version_id=f"model-persistence-{model_config_hash[:16]}",
                model_type="persistence",
                configuration_hash=model_config_hash,
                created_at=version_time,
            )
            detector_config_hash = _stable_hash(
                {
                    "scoring_method": "residual_zscore",
                    "algorithm_revision": DETECTOR_ALGORITHM_REVISION,
                    "threshold": detector_threshold,
                    "calibration_mean_tecu": calibration_mean,
                    "calibration_standard_deviation_tecu": (
                        calibration_standard_deviation
                    ),
                    "calibration_sample_count": calibration_sample_count,
                }
            )
            candidate_detector = DetectorVersion(
                detector_version_id=f"detector-zscore-{detector_config_hash[:16]}",
                scoring_method="residual_zscore",
                threshold=detector_threshold,
                configuration_hash=detector_config_hash,
                created_at=version_time,
                calibration_data_version=f"desktop-{detector_config_hash[:16]}",
                calibration_residual_mean_tecu=calibration_mean,
                calibration_residual_standard_deviation_tecu=(
                    calibration_standard_deviation
                ),
                calibration_sample_count=calibration_sample_count,
            )
            application = create_sqlite_application(
                database=self._database,
                artifact_directory=self._artifacts,
                region_bounds={loaded.region.boundary_ref: loaded.bounds},
                aggregation_policy=AggregationPolicy(
                    expected_cell_count=loaded.expected_cell_count
                ),
                detector_version_id=candidate_detector.detector_version_id,
                clock=self._clock,
            )
            with application.unit_of_work_factory() as unit_of_work:
                model = (
                    unit_of_work.models.get(candidate_model.model_version_id)
                    or candidate_model
                )
                detector = (
                    unit_of_work.detectors.get(
                        candidate_detector.detector_version_id
                    )
                    or candidate_detector
                )
                unit_of_work.detectors.register(detector)
                unit_of_work.commit()

            issue_result = application.forecasting.issue(
                request=ForecastRequest(
                    origin_at=origin_at,
                    valid_at=valid_at,
                    source_provider=loaded.provider,
                    source_product=loaded.product,
                    region_version_id=loaded.region.region_version_id,
                    target_metric=target_metric,
                    mode=mode,
                ),
                model_version=model,
                processing_version=loaded.processing_version,
            )
            forecast_id = issue_result.forecast.forecast_id
            existing_application = self._forecast_applications.get(forecast_id)
            if (
                existing_application is not None
                and existing_application.reconciler.detector_version_id
                != candidate_detector.detector_version_id
            ):
                raise DesktopInputError(
                    "This forecast is already associated with a different "
                    "detector configuration"
                )
            forecast_application = existing_application or application
            self._forecast_applications[forecast_id] = forecast_application
            self._last_forecast_id = forecast_id
            self._persist_forecast_contexts()
            reconciliation = forecast_application.reconciler.reconcile_forecast(
                forecast_id=forecast_id,
                valid_at_or_before=valid_at,
            )
            result = self._result_payload(
                application=forecast_application,
                forecast_id=forecast_id,
                reconciliation=asdict(reconciliation),
            )
            self._last_result = result
            self._prune_forecast_contexts()
            self._persist_forecast_contexts()
            return {"ok": True, "result": result}

    def reconcile(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._operation_lock:
            if not self._forecast_applications or self._last_forecast_id is None:
                raise DesktopInputError("Run a forecast before checking pending work")
            raw_cutoff = config.get("valid_at_or_before")
            cutoff = (
                _aware_now(self._clock, "desktop clock")
                if raw_cutoff is None or raw_cutoff == ""
                else _parse_datetime(raw_cutoff, field_name="valid_at_or_before")
            )
            summaries = tuple(
                application.reconciler.reconcile_forecast(
                    forecast_id=forecast_id,
                    valid_at_or_before=cutoff,
                )
                for forecast_id, application in self._forecast_applications.items()
            )
            summary = ReconciliationSummary(
                considered=sum(item.considered for item in summaries),
                scored=sum(item.scored for item in summaries),
                insufficient_data=sum(
                    item.insufficient_data for item in summaries
                ),
                retryable_errors=sum(item.retryable_errors for item in summaries),
            )
            last_application = self._forecast_applications[
                self._last_forecast_id
            ]
            self._last_result = self._result_payload(
                application=last_application,
                forecast_id=self._last_forecast_id,
                reconciliation=asdict(summary),
            )
            self._prune_forecast_contexts()
            self._persist_forecast_contexts()
            return {
                "ok": True,
                "summary": asdict(summary),
                "result": self._last_result,
            }

    def _restore_forecast_contexts(self) -> None:
        if not self._context_file.is_file() or not self._database.is_file():
            return
        try:
            with self._context_file.open("rb") as source:
                content = source.read(MAX_CONTEXT_BYTES + 1)
            if len(content) > MAX_CONTEXT_BYTES:
                raise ValueError("context file is too large")
            payload = json.loads(content.decode("utf-8"))
            contexts, last_forecast_id = _parse_context_file(payload)

            applications_by_detector: dict[str, OphanimApplication] = {}
            restored: dict[str, OphanimApplication] = {}
            for forecast_id, detector_version_id in contexts:
                application = applications_by_detector.get(detector_version_id)
                if application is None:
                    application = self._reconciliation_application(
                        detector_version_id
                    )
                    applications_by_detector[detector_version_id] = application
                with application.unit_of_work_factory() as unit_of_work:
                    forecast = unit_of_work.forecasts.get(forecast_id)
                    detector = unit_of_work.detectors.get(detector_version_id)
                if forecast is not None and detector is not None:
                    restored[forecast_id] = application

            self._forecast_applications = restored
            if not restored:
                return
            self._last_forecast_id = (
                last_forecast_id
                if last_forecast_id in restored
                else next(reversed(restored))
            )
            last_application = restored[self._last_forecast_id]
            self._last_result = self._result_payload(
                application=last_application,
                forecast_id=self._last_forecast_id,
                reconciliation=asdict(_empty_reconciliation_summary()),
            )
        except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError):
            print(
                "Warning: saved desktop forecast context could not be restored.",
                file=sys.stderr,
            )
            self._forecast_applications = {}
            self._last_forecast_id = None
            self._last_result = None

    def _persist_forecast_contexts(self) -> None:
        payload = {
            "version": CONTEXT_FILE_VERSION,
            "last_forecast_id": self._last_forecast_id,
            "contexts": [
                {
                    "forecast_id": forecast_id,
                    "detector_version_id": (
                        application.reconciler.detector_version_id
                    ),
                }
                for forecast_id, application in self._forecast_applications.items()
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_CONTEXT_BYTES:
            raise RuntimeError("too many desktop forecast contexts to persist")
        temporary = self._context_file.with_name(
            f".{self._context_file.name}.{secrets.token_hex(6)}.tmp"
        )
        try:
            with temporary.open("xb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self._context_file)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _prune_forecast_contexts(self) -> None:
        for forecast_id, application in tuple(
            self._forecast_applications.items()
        ):
            if forecast_id == self._last_forecast_id:
                continue
            with application.unit_of_work_factory() as unit_of_work:
                forecast = unit_of_work.forecasts.get(forecast_id)
            if forecast is None or forecast.status is not ForecastStatus.PENDING:
                del self._forecast_applications[forecast_id]

    def _reconciliation_application(
        self,
        detector_version_id: str,
    ) -> OphanimApplication:
        placeholder_bounds = GeographicBounds(
            south_latitude_degrees=-90.0,
            west_longitude_degrees=-180.0,
            north_latitude_degrees=90.0,
            east_longitude_degrees=179.999999,
        )
        return create_sqlite_application(
            database=self._database,
            artifact_directory=self._artifacts,
            region_bounds={"desktop-reconciliation": placeholder_bounds},
            aggregation_policy=AggregationPolicy(expected_cell_count=1),
            detector_version_id=detector_version_id,
            clock=self._clock,
        )

    def _require_loaded(self, dataset_id: Any) -> LoadedDataset:
        if self._loaded is None:
            raise DesktopInputError("Load an IONEX file first")
        if not isinstance(dataset_id, str) or dataset_id != self._loaded.dataset_id:
            raise DesktopInputError("The loaded dataset changed; reload the file")
        return self._loaded

    def _store_upload(self, content: bytes, filename: str) -> Path:
        checksum = sha256(content).hexdigest()
        suffixes = Path(filename).suffixes[-2:]
        safe_suffixes = [suffix.lower() for suffix in suffixes if _SAFE_SUFFIX.fullmatch(suffix)]
        suffix = "".join(safe_suffixes) or ".ionex"
        destination = self._uploads / f"{checksum}{suffix}"
        if destination.exists():
            if sha256(destination.read_bytes()).hexdigest() != checksum:
                raise RuntimeError("cached upload content is corrupted")
            return destination
        temporary = self._uploads / f".{checksum}.{secrets.token_hex(6)}.tmp"
        try:
            with temporary.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @staticmethod
    def _loaded_payload(loaded: LoadedDataset) -> dict[str, Any]:
        observed_times = {item.observed_at for item in loaded.observations}
        scorable = [
            item.observed_at
            for item in loaded.observations
            if item.observed_at + timedelta(hours=2) in observed_times
        ]
        default_origin = (scorable or [loaded.observations[-1].observed_at])[-1]
        return {
            "dataset_id": loaded.dataset_id,
            "filename": loaded.filename,
            "provider": loaded.provider,
            "product": loaded.product,
            "revision": loaded.revision,
            "revision_priority": loaded.revision_priority,
            "source_uri": loaded.source_uri,
            "artifact_id": loaded.ingestion.artifact.artifact_id,
            "observation_count": loaded.ingestion.observation_count,
            "already_present": loaded.ingestion.already_present,
            "expected_cell_count": loaded.expected_cell_count,
            "inferred_cell_count": loaded.inferred_cell_count,
            "global_epoch_count": loaded.global_epoch_count,
            "global_cells_per_epoch_minimum": (
                loaded.global_cells_per_epoch_minimum
            ),
            "global_cells_per_epoch_maximum": (
                loaded.global_cells_per_epoch_maximum
            ),
            "bounds": {
                "south": loaded.bounds.south_latitude_degrees,
                "west": loaded.bounds.west_longitude_degrees,
                "north": loaded.bounds.north_latitude_degrees,
                "east": loaded.bounds.east_longitude_degrees,
            },
            "region_version_id": loaded.region.region_version_id,
            "processing_version_id": (
                loaded.processing_version.processing_version_id
            ),
            "aggregation": asdict(loaded.aggregation),
            "bulk_archive": (
                None
                if loaded.bulk_archive is None
                else _bulk_archive_payload(loaded.bulk_archive)
            ),
            "epochs": [_observation_payload(item) for item in loaded.observations],
            "default_origin": _datetime_payload(default_origin),
        }

    @staticmethod
    def _result_payload(
        *,
        application: OphanimApplication,
        forecast_id: str,
        reconciliation: dict[str, Any],
    ) -> dict[str, Any]:
        with application.unit_of_work_factory() as unit_of_work:
            forecast = unit_of_work.forecasts.get(forecast_id)
            if forecast is None:
                raise RuntimeError("stored forecast disappeared")
            decisions = tuple(unit_of_work.decisions.for_forecast(forecast_id))
            decision = decisions[-1] if decisions else None
            actual = (
                None
                if decision is None
                else unit_of_work.regional_observations.get(
                    decision.observation_id
                )
            )
            source = unit_of_work.regional_observations.get(
                forecast.input_observation_ids[-1]
            )
        return {
            "forecast": _forecast_payload(forecast),
            "decision": None if decision is None else _decision_payload(decision),
            "actual": None if actual is None else _observation_payload(actual),
            "source_observation": (
                None if source is None else _observation_payload(source)
            ),
            "reconciliation": reconciliation,
        }


class OphanimHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        controller: DesktopController,
        token: str,
    ) -> None:
        super().__init__(server_address, OphanimRequestHandler)
        self.controller = controller
        self.token = token


class OphanimRequestHandler(BaseHTTPRequestHandler):
    server: OphanimHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            self._require_local_host()
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "application": APPLICATION_ID,
                        "api_version": DESKTOP_API_VERSION,
                        "version": __version__,
                    }
                )
                return
            if parsed.path == "/api/state":
                self._send_json(self.server.controller.state())
                return
            if parsed.path == "/api/workspace":
                self._send_json(self.server.controller.workspace_status())
                return
            if parsed.path.startswith("/api/workspace/scans/"):
                self._send_json(self.server.controller.workspace_record(parsed.path.removeprefix("/api/workspace/scans/"), "scan"))
                return
            if parsed.path.startswith("/api/workspace/events/"):
                self._send_json(self.server.controller.workspace_record(parsed.path.removeprefix("/api/workspace/events/"), "event"))
                return
            if parsed.path.startswith("/workspace-assets/"):
                content, content_type = self.server.controller.workspace_asset(parsed.path.removeprefix("/workspace-assets/"))
                self._send_bytes(content, content_type, inline_styles=content_type.startswith("text/html"))
                return
            if parsed.path == "/api/mamba/status":
                self._send_json(self.server.controller.mamba_status())
                return
            if parsed.path == "/api/spatial/status":
                self._send_json(self.server.controller.spatial_status())
                return
            if parsed.path == "/api/shawtynet/status":
                self._send_json(self.server.controller.shawtynet_status())
                return
            if parsed.path.startswith("/shawtynet-assets/"):
                content, content_type = self.server.controller.shawtynet_asset(
                    parsed.path.removeprefix("/shawtynet-assets/")
                )
                self._send_bytes(content, content_type,
                                 inline_styles=content_type.startswith("text/html"))
                return
            if parsed.path in {"/", "/index.html"}:
                html = _web_asset("index.html").decode("utf-8").replace(
                    "__OPHANIM_TOKEN__",
                    self.server.token,
                )
                self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/styles.css":
                self._send_bytes(
                    _web_asset("styles.css"),
                    "text/css; charset=utf-8",
                )
                return
            if parsed.path == "/app.js":
                self._send_bytes(
                    _web_asset("app.js"),
                    "text/javascript; charset=utf-8",
                )
                return
            if parsed.path == "/shawtynet.js":
                self._send_bytes(_web_asset("shawtynet.js"), "text/javascript; charset=utf-8")
                return
            if parsed.path == "/workspace.js":
                self._send_bytes(_web_asset("workspace.js"), "text/javascript; charset=utf-8")
                return
            if parsed.path == "/favicon.svg":
                self._send_bytes(
                    _web_asset("favicon.svg"),
                    "image/svg+xml; charset=utf-8",
                )
                return
            self._send_json({"ok": False, "error": "Not found"}, status=404)
        except ValueError as error:
            self._send_json({"ok": False, "error": str(error)}, status=400)
        except (MambaMonitorConflict, SpatialMonitorConflict, ShawtyNetBusy, WorkspaceBusy) as error:
            self._send_json({"ok": False, "error": str(error)}, status=409)
        except (MambaMonitorUnavailable, SpatialMonitorUnavailable, ShawtyNetUnavailable, WorkspaceUnavailable) as error:
            self._send_json({"ok": False, "error": str(error)}, status=503)
        except Exception as error:
            self._handle_exception(error)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            self._require_local_host()
            self._require_mutation_token()
            if parsed.path in {"/api/workspace/scan", "/api/workspace/scenario", "/api/workspace/imagine", "/api/workspace/artify", "/api/workspace/style", "/api/workspace/render", "/api/workspace/animation", "/api/workspace/composite"}:
                self._send_json(self.server.controller.workspace_submit(parsed.path.rsplit("/", 1)[1], self._read_json()), status=202)
                return
            if parsed.path in {"/api/workspace/photo", "/api/workspace/mask"}:
                length = self._content_length(maximum=8*1024*1024)
                content = self.rfile.read(length)
                if len(content) != length:
                    raise DesktopInputError("Image upload is incomplete")
                handler = self.server.controller.workspace_mask if parsed.path.endswith("/mask") else self.server.controller.workspace_photo
                self._send_json(handler(content), status=201)
                return
            if parsed.path == "/api/workspace/cancel":
                self._send_json(self.server.controller.workspace_cancel(self._read_json()))
                return
            if parsed.path == "/api/shawtynet/analyze":
                self._send_json(self.server.controller.analyze_shawtynet(self._read_json()), status=202)
                return
            if parsed.path == "/api/gim/fetch":
                self._send_json(
                    self.server.controller.fetch_gim(self._read_json())
                )
                return
            if parsed.path == "/api/mamba/initialize":
                self._send_json(
                    self.server.controller.initialize_mamba(self._read_json()),
                    status=202,
                )
                return
            if parsed.path == "/api/mamba/scan":
                # Consume and validate the bounded JSON object even though v3
                # intentionally accepts no client-selected cursor/window.
                payload = self._read_json()
                if payload:
                    raise DesktopInputError("Mamba scan does not accept a client cursor")
                self._send_json(
                    self.server.controller.scan_mamba(),
                    status=202,
                )
                return
            if parsed.path == "/api/spatial/initialize":
                self._send_json(
                    self.server.controller.initialize_spatial(self._read_json()),
                    status=202,
                )
                return
            if parsed.path == "/api/spatial/scan":
                # The durable source/model cursors are server-owned.  Accepting
                # client-selected windows here could accidentally introduce
                # look-ahead into an otherwise causal comparison.
                payload = self._read_json()
                if payload:
                    raise DesktopInputError(
                        "Spatial scan does not accept a client cursor"
                    )
                self._send_json(
                    self.server.controller.scan_spatial(),
                    status=202,
                )
                return
            if parsed.path == "/api/regional-grid":
                self._send_json(
                    self.server.controller.regional_grid(self._read_json())
                )
                return
            if parsed.path == "/api/load":
                length = self._content_length(maximum=MAX_UPLOAD_BYTES)
                content = self.rfile.read(length)
                if len(content) != length:
                    raise DesktopInputError("IONEX upload ended unexpectedly")
                values = {
                    key: items[-1]
                    for key, items in parse_qs(
                        parsed.query,
                        keep_blank_values=True,
                    ).items()
                }
                filename = values.pop("filename", "upload.ionex")
                self._send_json(
                    self.server.controller.load(
                        content=content,
                        filename=filename,
                        config=values,
                    )
                )
                return
            if parsed.path == "/api/forecast":
                self._send_json(
                    self.server.controller.forecast(self._read_json())
                )
                return
            if parsed.path == "/api/reconcile":
                self._send_json(
                    self.server.controller.reconcile(self._read_json())
                )
                return
            if parsed.path == "/api/shutdown":
                self._send_json({"ok": True, "message": "OPHANIM is stopping"})
                threading.Thread(
                    target=self.server.shutdown,
                    name="ophanim-shutdown",
                    daemon=True,
                ).start()
                return
            self._send_json({"ok": False, "error": "Not found"}, status=404)
        except GIMNotFound as error:
            self._send_json({"ok": False, "error": str(error)}, status=404)
        except GIMAcquisitionError as error:
            self._send_json({"ok": False, "error": str(error)}, status=502)
        except (MambaMonitorConflict, SpatialMonitorConflict, ShawtyNetBusy, WorkspaceBusy) as error:
            self._send_json({"ok": False, "error": str(error)}, status=409)
        except (MambaMonitorUnavailable, SpatialMonitorUnavailable, ShawtyNetUnavailable, WorkspaceUnavailable) as error:
            self._send_json({"ok": False, "error": str(error)}, status=503)
        except ValueError as error:
            self._send_json({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            self._handle_exception(error)

    def _read_json(self) -> dict[str, Any]:
        length = self._content_length(maximum=MAX_JSON_BYTES)
        content = self.rfile.read(length)
        if len(content) != length:
            raise DesktopInputError("JSON request ended unexpectedly")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DesktopInputError("Request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise DesktopInputError("Request JSON must be an object")
        return payload

    def _content_length(self, *, maximum: int) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise DesktopInputError("Content-Length is required")
        try:
            length = int(raw)
        except ValueError as error:
            raise DesktopInputError("Invalid Content-Length") from error
        if length < 0 or length > maximum:
            raise DesktopInputError(f"Request body exceeds {maximum} bytes")
        return length

    def _require_mutation_token(self) -> None:
        supplied = self.headers.get("X-OPHANIM-Token", "")
        if not secrets.compare_digest(supplied, self.server.token):
            raise DesktopInputError("Invalid local application token")
        origin = self.headers.get("Origin")
        if origin:
            request_host, request_port = self._local_request_authority()
            try:
                parsed = urlparse(origin)
                origin_port = parsed.port
            except ValueError as error:
                raise DesktopInputError("Invalid request origin") from error
            if (
                parsed.scheme != "http"
                or (parsed.hostname or "").casefold() != request_host
                or origin_port != request_port
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise DesktopInputError("Cross-origin requests are not allowed")

    def _require_local_host(self) -> None:
        self._local_request_authority()

    def _local_request_authority(self) -> tuple[str, int]:
        raw_host = self.headers.get("Host", "")
        try:
            parsed = urlparse(f"//{raw_host}")
            port = parsed.port
        except ValueError as error:
            raise DesktopInputError("Invalid request host") from error
        hostname = (parsed.hostname or "").casefold()
        if (
            hostname not in {"127.0.0.1", "localhost"}
            or port is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise DesktopInputError("Requests must use the local application address")
        return hostname, port

    def _handle_exception(self, error: Exception) -> None:
        traceback.print_exception(error, file=sys.stderr)
        self._send_json(
            {
                "ok": False,
                "error": (
                    "Unexpected application error. See the OPHANIM launcher "
                    "window for details."
                ),
            },
            status=500,
        )

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            ),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(
        self,
        content: bytes,
        content_type: str,
        *,
        status: int = 200,
        inline_styles: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'"
            + (" 'unsafe-inline'" if inline_styles else "") + "; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def _web_asset(name: str) -> bytes:
    return resources.files("ophanim").joinpath("web", name).read_bytes()


def _forecast_payload(forecast: Forecast) -> dict[str, Any]:
    return {
        "forecast_id": forecast.forecast_id,
        "status": forecast.status.value,
        "mode": forecast.mode.value,
        "target_metric": forecast.target_metric.value,
        "origin_at": _datetime_payload(forecast.origin_at),
        "issued_at": _datetime_payload(forecast.issued_at),
        "valid_at": _datetime_payload(forecast.valid_at),
        "source_provider": forecast.source_provider,
        "source_product": forecast.source_product,
        "region_version_id": forecast.region_version_id,
        "model_version_id": forecast.model_version_id,
        "processing_version_id": forecast.processing_version_id,
        "predicted_vtec_tecu": forecast.predicted_vtec_tecu,
        "input_observation_ids": list(forecast.input_observation_ids),
    }


def _decision_payload(decision: AnomalyDecision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "detector_version_id": decision.detector_version_id,
        "scored_at": _datetime_payload(decision.scored_at),
        "residual_tecu": decision.residual_tecu,
        "anomaly_score": decision.anomaly_score,
        "threshold": decision.threshold,
        "is_forecast_anomaly": decision.is_forecast_anomaly,
        "disturbance_assessment": decision.disturbance_assessment.value,
        "assessment_source": decision.assessment_source,
        "supersedes_decision_id": decision.supersedes_decision_id,
    }


def _observation_payload(observation: RegionalObservation) -> dict[str, Any]:
    return {
        "observation_id": observation.observation_id,
        "artifact_id": observation.artifact_id,
        "observed_at": _datetime_payload(observation.observed_at),
        "produced_at": _datetime_payload(observation.produced_at),
        "mean_vtec_tecu": observation.mean_vtec_tecu,
        "median_vtec_tecu": observation.median_vtec_tecu,
        "cell_count": observation.cell_count,
        "coverage_fraction": observation.coverage_fraction,
        "quality_flags": list(observation.quality_flags),
    }


def _bulk_archive_payload(result: TECArchiveResult) -> dict[str, Any]:
    manifest = result.manifest
    payload = {
        "grid_set_id": manifest.grid_set_id,
        "layout_version": manifest.layout_version,
        "value_kind": getattr(manifest, "value_kind", "native_measurement"),
        "storage_ref": manifest.storage_ref,
        "zarr_format": manifest.zarr_format,
        "shape": list(manifest.shape),
        "chunk_shape": list(manifest.chunk_shape),
        "vtec_dtype": manifest.vtec_dtype,
        "valid_cell_count": sum(
            epoch.valid_cell_count for epoch in manifest.epochs
        ),
        "missing_cell_count": sum(
            epoch.missing_cell_count for epoch in manifest.epochs
        ),
        "grid_already_present": result.grid_already_present,
        "catalog_already_present": result.catalog_already_present,
    }
    core_manifest = getattr(result, "core_manifest", None)
    payload["core_grid"] = (
        None
        if core_manifest is None
        else {
            "grid_set_id": core_manifest.grid_set_id,
            "layout_version": core_manifest.layout_version,
            "value_kind": core_manifest.value_kind,
            "storage_ref": core_manifest.storage_ref,
            "zarr_format": core_manifest.zarr_format,
            "shape": list(core_manifest.shape),
            "chunk_shape": list(core_manifest.chunk_shape),
            "vtec_dtype": core_manifest.vtec_dtype,
            "valid_cell_count": sum(
                epoch.valid_cell_count for epoch in core_manifest.epochs
            ),
            "missing_cell_count": sum(
                epoch.missing_cell_count for epoch in core_manifest.epochs
            ),
            "source_grid_set_id": core_manifest.source_grid_set_id,
            "source_grid_logical_sha256": (
                core_manifest.source_grid_logical_sha256
            ),
            "derivation_method": core_manifest.derivation_method,
            "grid_already_present": getattr(
                result, "core_grid_already_present", None
            ),
            "catalog_already_present": getattr(
                result, "core_catalog_already_present", None
            ),
        }
    )
    return payload


def _datetime_payload(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DesktopInputError(f"{field_name} is required")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise DesktopInputError(f"{field_name} is not a valid date/time") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _aware_now(clock: Callable[[], datetime], name: str) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must return a timezone-aware datetime")
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _standard_gim_coordinates() -> frozenset[tuple[float, float]]:
    return _STANDARD_GIM_COORDINATES


def _validate_standard_gim_observations(
    observations: tuple[TECObservation, ...],
    *,
    expected_product_date: date | None,
) -> None:
    by_epoch: dict[datetime, set[tuple[float, float]]] = defaultdict(set)
    for observation in observations:
        by_epoch[observation.observed_at].add(
            (
                observation.latitude_degrees,
                observation.longitude_degrees,
            )
        )
    _validate_standard_gim_epochs(
        by_epoch,
        expected_product_date=expected_product_date,
    )


def _validate_standard_gim_epochs(
    by_epoch: dict[datetime, set[tuple[float, float]]],
    *,
    expected_product_date: date | None,
) -> None:
    if not by_epoch:
        raise DesktopInputError("Downloaded GIM contains no TEC maps")
    expected_grid = _standard_gim_coordinates()
    incomplete_epochs = tuple(
        observed_at
        for observed_at, coordinates in by_epoch.items()
        if coordinates != expected_grid
    )
    if incomplete_epochs:
        first = min(incomplete_epochs)
        actual = len(by_epoch[first])
        raise DesktopInputError(
            "Downloaded file is not a complete standard 5° × 2.5° GIM: "
            f"expected 5,112 cells at every epoch, found {actual} at "
            f"{first.isoformat()}"
        )
    if expected_product_date is None:
        return
    expected_start = datetime(
        expected_product_date.year,
        expected_product_date.month,
        expected_product_date.day,
        tzinfo=UTC,
    )
    product_end = expected_start + timedelta(days=1)
    if min(by_epoch) != expected_start or max(by_epoch) > product_end:
        raise DesktopInputError(
            "Downloaded GIM epochs do not match its resolved UTC date"
        )


def _parse_context_file(
    payload: Any,
) -> tuple[tuple[tuple[str, str], ...], str | None]:
    if not isinstance(payload, dict):
        raise ValueError("context payload must be an object")
    version = payload.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CONTEXT_FILE_VERSION
    ):
        raise ValueError("unsupported context file version")
    raw_contexts = payload.get("contexts")
    if not isinstance(raw_contexts, list) or len(raw_contexts) > 10_000:
        raise ValueError("contexts must be a bounded list")

    contexts: dict[str, str] = {}
    for entry in raw_contexts:
        if not isinstance(entry, dict):
            raise ValueError("context entries must be objects")
        forecast_id = _context_identifier(entry.get("forecast_id"), "forecast_id")
        detector_id = _context_identifier(
            entry.get("detector_version_id"),
            "detector_version_id",
        )
        prior = contexts.get(forecast_id)
        if prior is not None and prior != detector_id:
            raise ValueError("forecast has conflicting detector contexts")
        contexts[forecast_id] = detector_id

    raw_last = payload.get("last_forecast_id")
    last_forecast_id = (
        None
        if raw_last is None
        else _context_identifier(raw_last, "last_forecast_id")
    )
    return tuple(contexts.items()), last_forecast_id


def _context_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 500:
        raise ValueError(f"{field_name} must be a non-empty identifier")
    return value


def _empty_reconciliation_summary() -> ReconciliationSummary:
    return ReconciliationSummary(
        considered=0,
        scored=0,
        insufficient_data=0,
        retryable_errors=0,
    )


def _slug(value: str) -> str:
    slug = _SAFE_SLUG.sub("-", value.casefold()).strip("-")
    return slug[:48] or "region"


def _display_filename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        return "upload.ionex"
    return "".join(character for character in name if character.isprintable())[:240]


def _text(
    config: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
    maximum: int = 200,
) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise DesktopInputError(f"{key} must not be empty")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise DesktopInputError(f"{key} is longer than {maximum} characters")
    return normalized


def _optional_text(
    config: dict[str, Any], key: str, *, maximum: int = 200
) -> str | None:
    value = config.get(key)
    if value is None or value == "":
        return None
    return _text(config, key, maximum=maximum)


def _optional_date(config: dict[str, Any], key: str) -> date | None:
    value = config.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise DesktopInputError(f"{key} must be a YYYY-MM-DD date")
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as error:
        raise DesktopInputError(f"{key} must be a valid YYYY-MM-DD date") from error
    if parsed.isoformat() != normalized:
        raise DesktopInputError(f"{key} must be a YYYY-MM-DD date")
    return parsed


def _number(
    config: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    minimum_exclusive: float | None = None,
    maximum: float | None = None,
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise DesktopInputError(f"{key} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DesktopInputError(f"{key} must be a number") from error
    if not math.isfinite(number):
        raise DesktopInputError(f"{key} must be finite")
    if minimum is not None and number < minimum:
        raise DesktopInputError(f"{key} must be at least {minimum}")
    if minimum_exclusive is not None and number <= minimum_exclusive:
        raise DesktopInputError(f"{key} must be greater than {minimum_exclusive}")
    if maximum is not None and number > maximum:
        raise DesktopInputError(f"{key} must not exceed {maximum}")
    return number


def _integer(
    config: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise DesktopInputError(f"{key} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DesktopInputError(f"{key} must be an integer") from error
    if str(value).strip() not in {str(number), f"+{number}"}:
        raise DesktopInputError(f"{key} must be an integer")
    if minimum is not None and number < minimum:
        raise DesktopInputError(f"{key} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise DesktopInputError(f"{key} must not exceed {maximum}")
    return number


def _optional_integer(
    config: dict[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = config.get(key)
    if value is None or value == "":
        return None
    return _integer(config, key, minimum=minimum, maximum=maximum)


def _existing_server(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.3) as response:
            content = response.read(MAX_JSON_BYTES + 1)
            if len(content) > MAX_JSON_BYTES:
                return False
            payload = json.loads(content.decode("utf-8"))
    except (OSError, URLError, UnicodeDecodeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("application") == APPLICATION_ID
        and payload.get("api_version") == DESKTOP_API_VERSION
    )


def _open_browser(url: str) -> None:
    windows_browsers = (
        Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("/mnt/c/Windows/explorer.exe"),
    )
    for executable in windows_browsers:
        if not executable.is_file():
            continue
        try:
            subprocess.Popen(
                [str(executable), url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return
        except OSError:
            continue
    webbrowser.open(url)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the OPHANIM desktop UI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--bind-address",
        choices=("127.0.0.1", "0.0.0.0"),
        default="127.0.0.1",
        help="bind to 0.0.0.0 only behind a localhost-only container port",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_directory(),
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser


def _default_data_directory() -> Path:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "ophanim"
    return Path.home() / ".local" / "share" / "ophanim"


def _bulk_archive_from_environment() -> BulkTECArchive | None:
    enabled = os.environ.get("OPHANIM_BULK_ARCHIVE", "").strip().casefold()
    if enabled in {"", "0", "false", "no"}:
        return None
    if enabled not in {"1", "true", "yes"}:
        raise SystemExit(
            "OPHANIM_BULK_ARCHIVE must be true/false, yes/no, or 1/0"
        )
    zarr_root = os.environ.get("OPHANIM_ZARR_ROOT", "").strip()
    if not zarr_root:
        raise SystemExit(
            "OPHANIM_ZARR_ROOT is required when the bulk archive is enabled"
        )
    from ophanim.postgres_catalog import PostgresTECCatalog

    catalog = PostgresTECCatalog(
        os.environ.get("OPHANIM_POSTGRES_DSN", "")
    )
    try:
        catalog.wait()
        catalog.initialize_schema()
        return BulkTECArchive(
            grid_store=ZarrTECGridStore(zarr_root),
            catalog=catalog,
        )
    except Exception:
        catalog.close()
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    controller = DesktopController(
        data_directory=arguments.data_dir,
        bulk_archive=_bulk_archive_from_environment(),
    )
    token = secrets.token_urlsafe(32)
    server: OphanimHTTPServer | None = None
    selected_port: int | None = None
    for port in range(arguments.port, min(arguments.port + PORT_ATTEMPTS, 65536)):
        if _existing_server(port):
            url = f"http://127.0.0.1:{port}/"
            if not arguments.no_browser:
                _open_browser(url)
            print(f"OPHANIM is already running at {url}")
            controller.close()
            return 0
        try:
            server = OphanimHTTPServer(
                (arguments.bind_address, port),
                controller=controller,
                token=token,
            )
        except OSError:
            continue
        selected_port = port
        break
    if server is None or selected_port is None:
        controller.close()
        raise SystemExit("No available localhost port was found")

    url = f"http://127.0.0.1:{selected_port}/"
    print("OPHANIM desktop is ready.")
    print(f"Open: {url}")
    print("Close it with the Stop application button or Ctrl+C.")
    if not arguments.no_browser:
        threading.Timer(0.2, _open_browser, args=(url,)).start()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_for_signal(signum: int, frame: Any) -> None:
        del signum, frame
        threading.Thread(
            target=server.shutdown,
            name="ophanim-sigterm-shutdown",
            daemon=True,
        ).start()

    signal.signal(signal.SIGTERM, stop_for_signal)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping OPHANIM.")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.server_close()
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
