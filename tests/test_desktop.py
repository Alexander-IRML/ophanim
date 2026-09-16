"""End-to-end tests for the dependency-free desktop application."""

from __future__ import annotations

import json
import gzip
import subprocess
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ophanim.desktop import (
    APPLICATION_ID,
    DESKTOP_API_VERSION,
    DesktopController,
    DesktopInputError,
    OphanimHTTPServer,
    _open_browser,
)
from ophanim.gim import FetchedGIM, GIMEdition
from ophanim.ingestion import IngestionRequest, LoadedSource
from ophanim.mamba_monitor import MambaMonitorConflict, MambaMonitorUnavailable
from ophanim.spatial_monitor import SpatialMonitorConflict, SpatialMonitorUnavailable
from ophanim.sqlite import SQLiteUnitOfWork


UTC = timezone.utc
ORIGIN = datetime(2024, 1, 2, 0, tzinfo=UTC)
RUN_AT = datetime(2024, 1, 3, 12, tzinfo=UTC)


def _record(payload: str, label: str) -> str:
    return f"{payload:<60}{label:<20}\n"


def _grid_payload(*values: float) -> str:
    return "  " + "".join(f"{value:6.1f}" for value in values)


def _data_record(*values: int) -> str:
    return "".join(f"{value:5d}" for value in values).ljust(80) + "\n"


def _map(number: int, hour: int, value: int) -> str:
    lines = [
        _record(f"{number:6d}", "START OF TEC MAP"),
        _record(
            f"{2024:6d}{1:6d}{2:6d}{hour:6d}{0:6d}{0:6d}",
            "EPOCH OF CURRENT MAP",
        ),
    ]
    for latitude in (32.5, 31.5):
        lines.append(
            _record(
                _grid_payload(latitude, -100.0, -98.0, 1.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
        )
        lines.append(_data_record(value, value, value))
    lines.append(_record(f"{number:6d}", "END OF TEC MAP"))
    return "".join(lines)


def _ionex(*, epochs: tuple[tuple[int, int], ...]) -> bytes:
    header = "".join(
        (
            _record(f"{1.0:8.1f}{'':12}I{'':39}", "IONEX VERSION / TYPE"),
            _record(f"{len(epochs):6d}", "# OF MAPS IN FILE"),
            _record(f"{2:6d}", "MAP DIMENSION"),
            _record(_grid_payload(450.0, 450.0, 0.0), "HGT1 / HGT2 / DHGT"),
            _record(_grid_payload(32.5, 31.5, -1.0), "LAT1 / LAT2 / DLAT"),
            _record(_grid_payload(-100.0, -98.0, 1.0), "LON1 / LON2 / DLON"),
            _record(f"{-1:6d}", "EXPONENT"),
            _record("", "END OF HEADER"),
        )
    )
    maps = "".join(
        _map(number, hour, value)
        for number, (hour, value) in enumerate(epochs, start=1)
    )
    return (header + maps + _record("", "END OF FILE")).encode("ascii")


def _standard_gim(product_date: date) -> bytes:
    header = "".join(
        (
            _record(f"{1.0:8.1f}{'':12}I{'':39}", "IONEX VERSION / TYPE"),
            _record(f"{1:6d}", "# OF MAPS IN FILE"),
            _record(f"{2:6d}", "MAP DIMENSION"),
            _record(_grid_payload(450.0, 450.0, 0.0), "HGT1 / HGT2 / DHGT"),
            _record(_grid_payload(87.5, -87.5, -2.5), "LAT1 / LAT2 / DLAT"),
            _record(_grid_payload(-180.0, 180.0, 5.0), "LON1 / LON2 / DLON"),
            _record(f"{-1:6d}", "EXPONENT"),
            _record("", "END OF HEADER"),
            _record(f"{1:6d}", "START OF TEC MAP"),
            _record(
                f"{product_date.year:6d}{product_date.month:6d}"
                f"{product_date.day:6d}{0:6d}{0:6d}{0:6d}",
                "EPOCH OF CURRENT MAP",
            ),
        )
    )
    rows: list[str] = []
    for latitude_index in range(71):
        latitude = 87.5 - latitude_index * 2.5
        rows.append(
            _record(
                _grid_payload(latitude, -180.0, 180.0, 5.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
        )
        values = [100] * 73
        for offset in range(0, len(values), 16):
            rows.append(_data_record(*values[offset : offset + 16]))
    trailer = _record(f"{1:6d}", "END OF TEC MAP") + _record("", "END OF FILE")
    return gzip.compress((header + "".join(rows) + trailer).encode("ascii"))


class _FakeGIMSource:
    def __init__(self, *, content: bytes | None = None) -> None:
        self.content = content
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        product_date = request.product_date or date(2024, 1, 2)
        edition = request.edition
        filename = (
            f"COD0OPS{edition.filename_code}_"
            f"{product_date.year:04d}{product_date.timetuple().tm_yday:03d}0000_"
            "01D_01H_GIM.INX.gz"
        )
        source_uri = f"https://example.test/CODE/{filename}"
        return FetchedGIM(
            product_date=product_date,
            edition=edition,
            ingestion_request=IngestionRequest(
                provider="code",
                product="gim",
                source_uri=source_uri,
                revision=edition.value,
                revision_priority=edition.revision_priority,
            ),
            loaded_source=LoadedSource(
                content=self.content or _standard_gim(product_date),
                filename=filename,
            ),
            resolved_uri=source_uri,
        )


class _FakeBulkArchive:
    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    def archive(self, **request):
        self.requests.append(request)
        manifest = SimpleNamespace(
            grid_set_id="1" * 64,
            layout_version="ophanim-tec-grid/1",
            storage_ref="file:///zarr/grid.zarr",
            zarr_format=3,
            shape=(1, 71, 72),
            chunk_shape=(1, 71, 72),
            vtec_dtype="float32",
            epochs=(
                SimpleNamespace(
                    valid_cell_count=5_112,
                    missing_cell_count=0,
                ),
            ),
            value_kind="native_measurement",
        )
        core_manifest = SimpleNamespace(
            grid_set_id="2" * 64,
            layout_version="ophanim-tec-core-half-degree/1",
            value_kind="interpolated_estimate",
            storage_ref="file:///zarr/core-grid.zarr",
            zarr_format=3,
            shape=(1, 351, 720),
            chunk_shape=(1, 256, 256),
            vtec_dtype="float32",
            source_grid_set_id=manifest.grid_set_id,
            source_grid_logical_sha256="3" * 64,
            derivation_method="ophanim-global-periodic-bilinear/1",
            epochs=(
                SimpleNamespace(
                    valid_cell_count=252_720,
                    missing_cell_count=0,
                ),
            ),
        )
        return SimpleNamespace(
            manifest=manifest,
            grid_already_present=False,
            catalog_already_present=True,
            core_manifest=core_manifest,
            core_grid_already_present=False,
            core_catalog_already_present=False,
        )

    def close(self) -> None:
        self.closed = True


class _FakeMambaMonitor:
    def __init__(self) -> None:
        self.initializations = []
        self.scan_count = 0
        self.closed = False
        self.failure: Exception | None = None

    def status(self):
        return {
            "ok": True,
            "available": True,
            "state": "uninitialized",
            "model": None,
            "active_job": None,
            "scan": {"pending_readout_count": 0, "last_result": None},
        }

    def initialize(self, config):
        if self.failure is not None:
            raise self.failure
        self.initializations.append(config)
        return {**self.status(), "state": "initializing"}

    def scan(self):
        if self.failure is not None:
            raise self.failure
        self.scan_count += 1
        return {**self.status(), "state": "scanning"}

    def close(self):
        self.closed = True


class _FakeSpatialMonitor:
    def __init__(self) -> None:
        self.initializations = []
        self.scan_count = 0
        self.closed = False
        self.close_result = None
        self.failure: Exception | None = None

    def status(self):
        return {
            "ok": True,
            "available": True,
            "state": "uninitialized",
            "model": None,
            "active_job": None,
            "scan": {"pending_frame_count": 0, "last_result": None},
        }

    def initialize(self, config):
        if self.failure is not None:
            raise self.failure
        self.initializations.append(config)
        return {**self.status(), "state": "initializing"}

    def scan(self):
        if self.failure is not None:
            raise self.failure
        self.scan_count += 1
        return {**self.status(), "state": "scanning"}

    def close(self):
        self.closed = True
        return self.close_result


def _load_config() -> dict[str, str]:
    return {
        "provider": "test-provider",
        "product": "test-ionex",
        "revision": "r1",
        "revision_priority": "4",
    }


class BrowserLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        root = Path(self._temporary_directory.name)
        self.chrome = root / "Program Files" / "Google" / "Chrome" / "chrome.exe"
        self.explorer = root / "Windows" / "explorer.exe"
        self.chrome.parent.mkdir(parents=True)
        self.explorer.parent.mkdir(parents=True)

    def test_open_browser_prefers_windows_chrome_over_default_browser(self) -> None:
        self.chrome.touch()
        self.explorer.touch()
        url = "http://127.0.0.1:8765/"

        with (
            patch(
                "ophanim.desktop.Path",
                side_effect=lambda value: {
                    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe": self.chrome,
                    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe": (
                        self.chrome.with_name("missing-chrome.exe")
                    ),
                    "/mnt/c/Windows/explorer.exe": self.explorer,
                }.get(value, Path(value)),
            ),
            patch("ophanim.desktop.subprocess.Popen") as popen,
            patch("ophanim.desktop.webbrowser.open") as fallback,
        ):
            _open_browser(url)

        popen.assert_called_once_with(
            [str(self.chrome), url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        fallback.assert_not_called()

    def test_open_browser_falls_back_to_explorer_when_chrome_fails(self) -> None:
        self.chrome.touch()
        self.explorer.touch()
        url = "http://127.0.0.1:8765/"

        with (
            patch(
                "ophanim.desktop.Path",
                side_effect=lambda value: {
                    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe": self.chrome,
                    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe": (
                        self.chrome.with_name("missing-chrome.exe")
                    ),
                    "/mnt/c/Windows/explorer.exe": self.explorer,
                }.get(value, Path(value)),
            ),
            patch(
                "ophanim.desktop.subprocess.Popen",
                side_effect=(OSError("Chrome failed"), object()),
            ) as popen,
            patch("ophanim.desktop.webbrowser.open") as fallback,
        ):
            _open_browser(url)

        self.assertEqual(popen.call_count, 2)
        self.assertEqual(popen.call_args_list[0].args[0], [str(self.chrome), url])
        self.assertEqual(popen.call_args_list[1].args[0], [str(self.explorer), url])
        fallback.assert_not_called()

    def test_open_browser_uses_python_fallback_without_windows_launchers(self) -> None:
        url = "http://127.0.0.1:8765/"
        missing = Path(self._temporary_directory.name) / "missing.exe"

        with (
            patch("ophanim.desktop.Path", side_effect=lambda _value: missing),
            patch("ophanim.desktop.subprocess.Popen") as popen,
            patch("ophanim.desktop.webbrowser.open") as fallback,
        ):
            _open_browser(url)

        popen.assert_not_called()
        fallback.assert_called_once_with(url)


class DesktopControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.data_directory = Path(self._temporary_directory.name) / "desktop"
        self.controller = DesktopController(
            data_directory=self.data_directory,
            clock=lambda: RUN_AT,
        )
        self.content = _ionex(epochs=((0, 100), (2, 200)))

    def _load(self) -> dict:
        return self.controller.load(
            content=self.content,
            filename=r"C:\fakepath\central-texas.24i",
            config=_load_config(),
        )["loaded"]

    def test_initial_state_and_preconditions_are_clear(self) -> None:
        self.assertEqual(
            self.controller.state(),
            {"ok": True, "loaded": None, "result": None},
        )
        with self.assertRaisesRegex(DesktopInputError, "non-empty IONEX"):
            self.controller.load(
                content=b"",
                filename="empty.ionex",
                config={},
            )
        with self.assertRaisesRegex(DesktopInputError, "Load an IONEX"):
            self.controller.forecast({"dataset_id": "missing"})
        with self.assertRaisesRegex(DesktopInputError, "Run a forecast"):
            self.controller.reconcile({})

    def test_load_infers_grid_aggregates_epochs_and_is_idempotent(self) -> None:
        loaded = self._load()

        self.assertEqual(loaded["filename"], "central-texas.24i")
        self.assertEqual(loaded["provider"], "test-provider")
        self.assertEqual(loaded["product"], "test-ionex")
        self.assertEqual(loaded["revision"], "r1")
        self.assertEqual(loaded["revision_priority"], 4)
        self.assertEqual(loaded["observation_count"], 12)
        self.assertFalse(loaded["already_present"])
        self.assertEqual(loaded["inferred_cell_count"], 6)
        self.assertEqual(loaded["expected_cell_count"], 6)
        self.assertEqual(
            loaded["bounds"],
            {"south": 29.0, "west": -100.5, "north": 32.5, "east": -96.0},
        )
        self.assertEqual(
            loaded["aggregation"],
            {
                "artifact_id": loaded["artifact_id"],
                "epochs_seen": 2,
                "observations_created": 2,
                "observations_reused": 0,
            },
        )
        self.assertEqual(len(loaded["epochs"]), 2)
        self.assertEqual(
            [epoch["observed_at"] for epoch in loaded["epochs"]],
            ["2024-01-02T00:00:00Z", "2024-01-02T02:00:00Z"],
        )
        self.assertEqual(
            [epoch["mean_vtec_tecu"] for epoch in loaded["epochs"]],
            [10.0, 20.0],
        )
        self.assertEqual(
            [epoch["median_vtec_tecu"] for epoch in loaded["epochs"]],
            [10.0, 20.0],
        )
        self.assertEqual(loaded["default_origin"], "2024-01-02T00:00:00Z")
        self.assertEqual(self.controller.state()["loaded"], loaded)

        repeated = self._load()

        self.assertEqual(repeated["dataset_id"], loaded["dataset_id"])
        self.assertEqual(repeated["artifact_id"], loaded["artifact_id"])
        self.assertTrue(repeated["already_present"])
        self.assertEqual(repeated["aggregation"]["observations_created"], 0)
        self.assertEqual(repeated["aggregation"]["observations_reused"], 2)

    def test_fetch_gim_uses_controlled_source_and_validates_standard_grid(self) -> None:
        gim_source = _FakeGIMSource()
        controller = DesktopController(
            data_directory=self.data_directory / "automatic",
            clock=lambda: RUN_AT,
            gim_source=gim_source,
        )

        response = controller.fetch_gim(
            {
                "edition": "rapid",
                "date": "2024-01-02",
                "provider": "attacker-controlled",
                "product": "not-used",
            }
        )
        loaded = response["loaded"]

        self.assertEqual(len(gim_source.requests), 1)
        self.assertEqual(gim_source.requests[0].edition, GIMEdition.RAPID)
        self.assertEqual(gim_source.requests[0].product_date, date(2024, 1, 2))
        self.assertEqual(loaded["provider"], "code")
        self.assertEqual(loaded["product"], "gim")
        self.assertEqual(loaded["revision"], "rapid")
        self.assertEqual(loaded["revision_priority"], 10)
        self.assertEqual(loaded["global_epoch_count"], 1)
        self.assertEqual(loaded["global_cells_per_epoch_minimum"], 5_112)
        self.assertEqual(loaded["global_cells_per_epoch_maximum"], 5_112)
        self.assertEqual(loaded["observation_count"], 5_112)
        self.assertEqual(loaded["expected_cell_count"], 2)
        self.assertEqual(loaded["inferred_cell_count"], 2)
        self.assertTrue(loaded["source_uri"].startswith("https://example.test/"))
        self.assertEqual(response["acquisition"]["resolved_date"], "2024-01-02")
        self.assertEqual(
            response["acquisition"]["grid"]["unique_cells_per_epoch"],
            5_112,
        )
        self.assertEqual(list((self.data_directory / "automatic" / "uploads").iterdir()), [])

    def test_validated_gim_is_mirrored_to_bulk_archive_but_local_load_is_not(self) -> None:
        bulk_archive = _FakeBulkArchive()
        controller = DesktopController(
            data_directory=self.data_directory / "bulk-archive",
            clock=lambda: RUN_AT,
            gim_source=_FakeGIMSource(),
            bulk_archive=bulk_archive,
        )

        response = controller.fetch_gim(
            {"edition": "rapid", "date": "2024-01-02"}
        )
        archived = response["loaded"]["bulk_archive"]

        self.assertEqual(len(bulk_archive.requests), 1)
        request = bulk_archive.requests[0]
        self.assertEqual(len(request["observations"]), 5_112)
        self.assertEqual(
            request["artifact"].artifact_id,
            response["loaded"]["artifact_id"],
        )
        self.assertEqual(len(request["grid"].latitudes_degrees), 71)
        self.assertEqual(len(request["grid"].longitudes_degrees), 72)
        self.assertEqual(
            archived,
            {
                "grid_set_id": "1" * 64,
                "layout_version": "ophanim-tec-grid/1",
                "value_kind": "native_measurement",
                "storage_ref": "file:///zarr/grid.zarr",
                "zarr_format": 3,
                "shape": [1, 71, 72],
                "chunk_shape": [1, 71, 72],
                "vtec_dtype": "float32",
                "valid_cell_count": 5_112,
                "missing_cell_count": 0,
                "grid_already_present": False,
                "catalog_already_present": True,
                "core_grid": {
                    "grid_set_id": "2" * 64,
                    "layout_version": "ophanim-tec-core-half-degree/1",
                    "value_kind": "interpolated_estimate",
                    "storage_ref": "file:///zarr/core-grid.zarr",
                    "zarr_format": 3,
                    "shape": [1, 351, 720],
                    "chunk_shape": [1, 256, 256],
                    "vtec_dtype": "float32",
                    "valid_cell_count": 252_720,
                    "missing_cell_count": 0,
                    "source_grid_set_id": "1" * 64,
                    "source_grid_logical_sha256": "3" * 64,
                    "derivation_method": "ophanim-global-periodic-bilinear/1",
                    "grid_already_present": False,
                    "catalog_already_present": False,
                },
            },
        )

        local = controller.load(
            content=self.content,
            filename="local.ionex",
            config=_load_config(),
        )["loaded"]

        self.assertEqual(len(bulk_archive.requests), 1)
        self.assertIsNone(local["bulk_archive"])
        controller.close()
        self.assertTrue(bulk_archive.closed)

    def test_spatial_monitor_is_lazy_and_shares_mamba_and_bulk_archive(self) -> None:
        bulk_archive = _FakeBulkArchive()
        mamba_monitor = _FakeMambaMonitor()
        spatial_monitor = _FakeSpatialMonitor()
        clock = lambda: RUN_AT
        data_directory = self.data_directory / "spatial"
        controller = DesktopController(
            data_directory=data_directory,
            clock=clock,
            bulk_archive=bulk_archive,
            mamba_monitor=mamba_monitor,
        )

        with patch(
            "ophanim.desktop.SpatialMonitor",
            return_value=spatial_monitor,
        ) as constructor:
            self.assertEqual(controller.spatial_status()["state"], "uninitialized")
            initialized = controller.initialize_spatial(
                {"history_years": 20, "input_frame_count": 12}
            )
            scanned = controller.scan_spatial()

        constructor.assert_called_once_with(
            data_directory=data_directory.resolve(),
            mamba_monitor=mamba_monitor,
            bulk_archive=bulk_archive,
            clock=clock,
        )
        self.assertEqual(initialized["state"], "initializing")
        self.assertEqual(scanned["state"], "scanning")
        self.assertEqual(
            spatial_monitor.initializations,
            [{"history_years": 20, "input_frame_count": 12}],
        )
        self.assertEqual(spatial_monitor.scan_count, 1)

        controller.close()
        self.assertTrue(spatial_monitor.closed)
        self.assertTrue(mamba_monitor.closed)
        self.assertTrue(bulk_archive.closed)

    def test_close_keeps_shared_dependencies_open_while_spatial_worker_runs(self) -> None:
        bulk_archive = _FakeBulkArchive()
        mamba_monitor = _FakeMambaMonitor()
        spatial_monitor = _FakeSpatialMonitor()
        spatial_monitor.close_result = False
        controller = DesktopController(
            data_directory=self.data_directory / "busy-spatial",
            clock=lambda: RUN_AT,
            bulk_archive=bulk_archive,
            mamba_monitor=mamba_monitor,
            spatial_monitor=spatial_monitor,
        )

        controller.close()

        self.assertTrue(spatial_monitor.closed)
        self.assertFalse(mamba_monitor.closed)
        self.assertFalse(bulk_archive.closed)

    def test_fetch_gim_rejects_wrong_grid_before_artifact_commit(self) -> None:
        gim_source = _FakeGIMSource(content=self.content)
        data_directory = self.data_directory / "wrong-grid"
        controller = DesktopController(
            data_directory=data_directory,
            clock=lambda: RUN_AT,
            gim_source=gim_source,
        )

        with self.assertRaisesRegex(DesktopInputError, "5,112 cells"):
            controller.fetch_gim(
                {"edition": "rapid", "date": "2024-01-02"}
            )

        artifact_files = [
            path
            for path in (data_directory / "artifacts").rglob("*")
            if path.is_file()
        ]
        self.assertEqual(artifact_files, [])
        self.assertIsNone(controller.state()["loaded"])

    def test_fetch_gim_rejects_invalid_catalog_fields(self) -> None:
        gim_source = _FakeGIMSource()
        controller = DesktopController(
            data_directory=self.data_directory / "invalid-fetch",
            clock=lambda: RUN_AT,
            gim_source=gim_source,
        )
        with self.assertRaisesRegex(DesktopInputError, "rapid or final"):
            controller.fetch_gim({"edition": "custom"})
        with self.assertRaisesRegex(DesktopInputError, "valid YYYY-MM-DD"):
            controller.fetch_gim({"edition": "rapid", "date": "2024-02-30"})
        self.assertEqual(gim_source.requests, [])

    def test_regional_grid_builds_a_fine_explicitly_derived_readout(self) -> None:
        controller = DesktopController(
            data_directory=self.data_directory / "regional-grid",
            clock=lambda: RUN_AT,
            gim_source=_FakeGIMSource(),
        )
        loaded = controller.fetch_gim(
            {"edition": "rapid", "date": "2024-01-02"}
        )["loaded"]

        readout = controller.regional_grid(
            {
                "dataset_id": loaded["dataset_id"],
                "observed_at": loaded["default_origin"],
                "latitude_step_degrees": 0.5,
                "longitude_step_degrees": 0.5,
            }
        )["readout"]

        self.assertEqual(readout["value_kind"], "interpolated_estimate")
        self.assertEqual(readout["method"], "bilinear")
        self.assertEqual(readout["artifact_id"], loaded["artifact_id"])
        self.assertEqual(readout["native_grid"]["source_point_count"], 5_112)
        self.assertEqual(readout["native_grid"]["latitude_step_degrees"], 2.5)
        self.assertEqual(readout["native_grid"]["longitude_step_degrees"], 5.0)
        self.assertTrue(readout["native_grid"]["longitude_is_cyclic"])
        self.assertEqual(readout["requested_grid"]["latitude_count"], 8)
        self.assertEqual(readout["requested_grid"]["longitude_count"], 10)
        self.assertEqual(readout["cell_count"], 80)
        self.assertEqual(len(readout["rows"]), 80)
        self.assertTrue(
            all(row["estimated_vtec_tecu"] == 10.0 for row in readout["rows"])
        )
        self.assertEqual(
            sum(row["is_native_node"] for row in readout["rows"]),
            2,
        )

    def test_regional_grid_rejects_stale_epoch_dataset_and_bad_steps(self) -> None:
        controller = DesktopController(
            data_directory=self.data_directory / "regional-grid-errors",
            clock=lambda: RUN_AT,
            gim_source=_FakeGIMSource(),
        )
        loaded = controller.fetch_gim(
            {"edition": "rapid", "date": "2024-01-02"}
        )["loaded"]
        request = {
            "dataset_id": loaded["dataset_id"],
            "observed_at": loaded["default_origin"],
        }

        with self.assertRaisesRegex(DesktopInputError, "dataset changed"):
            controller.regional_grid({**request, "dataset_id": "stale"})
        with self.assertRaisesRegex(DesktopInputError, "loaded epochs"):
            controller.regional_grid(
                {**request, "observed_at": "2024-01-02T01:00:00Z"}
            )
        with self.assertRaisesRegex(DesktopInputError, "greater than 0"):
            controller.regional_grid(
                {**request, "latitude_step_degrees": 0}
            )
        with self.assertRaisesRegex(DesktopInputError, "not exceed 2.5"):
            controller.regional_grid(
                {**request, "latitude_step_degrees": 3}
            )

    def test_forecast_scores_actual_and_exposes_provenance(self) -> None:
        loaded = self._load()

        response = self.controller.forecast(
            {
                "dataset_id": loaded["dataset_id"],
                "forecast_origin": loaded["default_origin"],
                "forecast_horizon_hours": "2",
                "target_metric": "median_vtec",
                "mode": "hindcast",
                "detector_threshold": "3",
                "calibration_residual_mean_tecu": "0",
                "calibration_residual_standard_deviation_tecu": "2",
                "calibration_sample_count": "100",
            }
        )

        result = response["result"]
        forecast = result["forecast"]
        self.assertTrue(response["ok"])
        self.assertEqual(forecast["status"], "scored")
        self.assertEqual(forecast["mode"], "hindcast")
        self.assertEqual(forecast["target_metric"], "median_vtec")
        self.assertEqual(forecast["origin_at"], "2024-01-02T00:00:00Z")
        self.assertEqual(forecast["valid_at"], "2024-01-02T02:00:00Z")
        self.assertEqual(forecast["predicted_vtec_tecu"], 10.0)
        self.assertEqual(forecast["source_provider"], loaded["provider"])
        self.assertEqual(
            forecast["processing_version_id"],
            loaded["processing_version_id"],
        )
        self.assertEqual(len(forecast["input_observation_ids"]), 1)

        self.assertEqual(result["source_observation"]["median_vtec_tecu"], 10.0)
        self.assertEqual(result["actual"]["median_vtec_tecu"], 20.0)
        decision = result["decision"]
        self.assertEqual(decision["residual_tecu"], 10.0)
        self.assertEqual(decision["anomaly_score"], 5.0)
        self.assertEqual(decision["threshold"], 3.0)
        self.assertTrue(decision["is_forecast_anomaly"])
        self.assertEqual(decision["disturbance_assessment"], "candidate")
        self.assertEqual(decision["assessment_source"], "residual_zscore")
        self.assertEqual(
            result["reconciliation"],
            {
                "considered": 1,
                "scored": 1,
                "insufficient_data": 0,
                "retryable_errors": 0,
            },
        )
        self.assertEqual(self.controller.state()["result"], result)

        follow_up = self.controller.reconcile({})
        self.assertEqual(follow_up["summary"]["considered"], 0)
        self.assertEqual(
            follow_up["result"]["forecast"]["forecast_id"],
            result["forecast"]["forecast_id"],
        )
        self.assertEqual(follow_up["result"]["decision"], result["decision"])
        self.assertEqual(
            follow_up["result"]["reconciliation"],
            follow_up["summary"],
        )

    def test_forecast_can_remain_pending_without_an_actual(self) -> None:
        current_time = [ORIGIN + timedelta(hours=1)]
        controller = DesktopController(
            data_directory=Path(self._temporary_directory.name) / "pending",
            clock=lambda: current_time[0],
        )
        loaded = controller.load(
            content=_ionex(epochs=((0, 100),)),
            filename="one-epoch.ionex",
            config=_load_config(),
        )["loaded"]

        result = controller.forecast(
            {
                "dataset_id": loaded["dataset_id"],
                "forecast_origin": loaded["default_origin"],
                "forecast_horizon_hours": 2,
            }
        )["result"]

        self.assertEqual(result["forecast"]["status"], "pending")
        self.assertIsNone(result["decision"])
        self.assertIsNone(result["actual"])
        self.assertEqual(result["reconciliation"]["considered"], 1)
        self.assertEqual(result["reconciliation"]["scored"], 0)

        current_time[0] = ORIGIN + timedelta(hours=9)
        updated = controller.reconcile({})
        self.assertEqual(updated["summary"]["insufficient_data"], 1)
        self.assertEqual(updated["result"]["forecast"]["status"], "insufficient_data")

    def test_later_file_scores_pending_forecast_with_exact_original_input(self) -> None:
        current_time = [ORIGIN + timedelta(hours=1)]
        data_directory = Path(self._temporary_directory.name) / "later-actual"
        controller = DesktopController(
            data_directory=data_directory,
            clock=lambda: current_time[0],
        )
        initial = controller.load(
            content=_ionex(epochs=((0, 100),)),
            filename="initial.ionex",
            config=_load_config(),
        )["loaded"]
        pending = controller.forecast(
            {
                "dataset_id": initial["dataset_id"],
                "forecast_origin": initial["default_origin"],
                "forecast_horizon_hours": 2,
                "calibration_residual_standard_deviation_tecu": 2,
                "calibration_sample_count": 100,
            }
        )["result"]
        original_input_id = pending["source_observation"]["observation_id"]

        current_time[0] = ORIGIN + timedelta(hours=3)
        controller = DesktopController(
            data_directory=data_directory,
            clock=lambda: current_time[0],
        )
        self.assertIsNone(controller.state()["loaded"])
        self.assertEqual(
            controller.state()["result"]["forecast"]["forecast_id"],
            pending["forecast"]["forecast_id"],
        )
        revised_config = {**_load_config(), "revision": "r2", "revision_priority": "9"}
        controller.load(
            content=_ionex(epochs=((0, 150), (2, 200))),
            filename="later-final.ionex",
            config=revised_config,
        )
        updated = controller.reconcile({})

        self.assertEqual(updated["summary"]["scored"], 1)
        self.assertEqual(updated["result"]["forecast"]["status"], "scored")
        self.assertEqual(updated["result"]["actual"]["median_vtec_tecu"], 20.0)
        self.assertEqual(
            updated["result"]["source_observation"]["observation_id"],
            original_input_id,
        )
        self.assertEqual(
            updated["result"]["source_observation"]["median_vtec_tecu"],
            10.0,
        )

    def test_pending_forecasts_keep_their_own_detector_configuration(self) -> None:
        current_time = [ORIGIN + timedelta(hours=1)]
        data_directory = Path(self._temporary_directory.name) / "two-detectors"
        controller = DesktopController(
            data_directory=data_directory,
            clock=lambda: current_time[0],
        )
        loaded = controller.load(
            content=_ionex(epochs=((0, 100),)),
            filename="initial.ionex",
            config=_load_config(),
        )["loaded"]

        def issue(standard_deviation: float) -> dict:
            return controller.forecast(
                {
                    "dataset_id": loaded["dataset_id"],
                    "forecast_origin": loaded["default_origin"],
                    "forecast_horizon_hours": 2,
                    "calibration_residual_standard_deviation_tecu": (
                        standard_deviation
                    ),
                    "calibration_sample_count": 100,
                }
            )["result"]

        first = issue(2)
        current_time[0] += timedelta(minutes=1)
        second = issue(5)
        current_time[0] = ORIGIN + timedelta(hours=3)
        controller = DesktopController(
            data_directory=data_directory,
            clock=lambda: current_time[0],
        )
        controller.load(
            content=_ionex(epochs=((2, 200),)),
            filename="actual.ionex",
            config={**_load_config(), "revision": "r2"},
        )

        updated = controller.reconcile({})

        self.assertEqual(updated["summary"]["considered"], 2)
        self.assertEqual(updated["summary"]["scored"], 2)
        with SQLiteUnitOfWork(data_directory / "ophanim.sqlite3") as unit_of_work:
            first_decision = unit_of_work.decisions.for_forecast(
                first["forecast"]["forecast_id"]
            )[-1]
            second_decision = unit_of_work.decisions.for_forecast(
                second["forecast"]["forecast_id"]
            )[-1]
        self.assertNotEqual(
            first_decision.detector_version_id,
            second_decision.detector_version_id,
        )
        self.assertEqual(first_decision.anomaly_score, 5.0)
        self.assertTrue(first_decision.is_forecast_anomaly)
        self.assertEqual(second_decision.anomaly_score, 2.0)
        self.assertFalse(second_decision.is_forecast_anomaly)

    def test_form_validation_rejects_mismatched_or_impossible_values(self) -> None:
        loaded = self._load()
        with self.assertRaisesRegex(DesktopInputError, "dataset changed"):
            self.controller.forecast(
                {
                    "dataset_id": "stale-dataset",
                    "forecast_origin": loaded["default_origin"],
                }
            )
        with self.assertRaisesRegex(DesktopInputError, "loaded regional epochs"):
            self.controller.forecast(
                {
                    "dataset_id": loaded["dataset_id"],
                    "forecast_origin": "2024-01-02T01:00:00Z",
                }
            )
        with self.assertRaisesRegex(DesktopInputError, "Unsupported target metric"):
            self.controller.forecast(
                {
                    "dataset_id": loaded["dataset_id"],
                    "forecast_origin": loaded["default_origin"],
                    "target_metric": "peak_vtec",
                }
            )
        with self.assertRaisesRegex(DesktopInputError, "must be a number"):
            self.controller.forecast(
                {
                    "dataset_id": loaded["dataset_id"],
                    "forecast_origin": loaded["default_origin"],
                    "forecast_horizon_hours": 10**10_000,
                }
            )
        with self.assertRaisesRegex(DesktopInputError, "must be an integer"):
            self.controller.forecast(
                {
                    "dataset_id": loaded["dataset_id"],
                    "forecast_origin": loaded["default_origin"],
                    "calibration_sample_count": float("inf"),
                }
            )


class DesktopHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.gim_source = _FakeGIMSource()
        self.controller = DesktopController(
            data_directory=Path(self._temporary_directory.name) / "desktop",
            clock=lambda: RUN_AT,
            gim_source=self.gim_source,
        )
        self.token = "test-local-token"
        try:
            self.server = OphanimHTTPServer(
                ("127.0.0.1", 0),
                controller=self.controller,
                token=self.token,
            )
        except PermissionError as error:
            raise unittest.SkipTest(
                f"loopback sockets are unavailable: {error}"
            ) from error
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="ophanim-http-test",
            daemon=True,
        )
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _json_request(
        self,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict]:
        request = Request(self.base_url + path, data=data, headers=headers or {})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read().decode("utf-8"))
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_health_state_and_static_assets_are_served_locally(self) -> None:
        status, health = self._json_request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])
        self.assertEqual(health["application"], APPLICATION_ID)
        self.assertEqual(health["api_version"], DESKTOP_API_VERSION)

        with urlopen(self.base_url + "/", timeout=5) as response:
            index = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
            self.assertIn(
                "default-src 'self'",
                response.headers["Content-Security-Policy"],
            )
        self.assertIn(self.token, index)
        self.assertNotIn("__OPHANIM_TOKEN__", index)

        for asset_path, content_type in (
            ("/styles.css", "text/css"),
            ("/app.js", "text/javascript"),
            ("/favicon.svg", "image/svg+xml"),
        ):
            with self.subTest(asset_path=asset_path):
                with urlopen(self.base_url + asset_path, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(content_type, response.headers["Content-Type"])
                    self.assertGreater(len(response.read()), 100)

        status, state = self._json_request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state, {"ok": True, "loaded": None, "result": None})

    def test_mutations_require_token_and_reject_cross_origin_requests(self) -> None:
        status, payload = self._json_request("/api/reconcile", data=b"{}")
        self.assertEqual(status, 400)
        self.assertIn("token", payload["error"])

        status, payload = self._json_request(
            "/api/reconcile",
            data=b"{}",
            headers={
                "X-OPHANIM-Token": self.token,
                "Origin": "https://example.test",
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("Cross-origin", payload["error"])

        status, payload = self._json_request(
            "/api/health",
            headers={"Host": "example.test"},
        )
        self.assertEqual(status, 400)
        self.assertIn("local application address", payload["error"])

    def test_loopback_proxy_port_is_allowed_when_host_and_origin_match(self) -> None:
        proxy_authority = "127.0.0.1:18876"
        status, health = self._json_request(
            "/api/health",
            headers={"Host": proxy_authority},
        )
        self.assertEqual(status, 200)
        self.assertEqual(health["api_version"], DESKTOP_API_VERSION)

        monitor = _FakeMambaMonitor()
        self.controller._mamba_monitor = monitor
        status, payload = self._json_request(
            "/api/mamba/scan",
            data=b"{}",
            headers={
                "Host": proxy_authority,
                "Origin": f"http://{proxy_authority}",
                "Content-Type": "application/json",
                "X-OPHANIM-Token": self.token,
            },
        )
        self.assertEqual(status, 202)
        self.assertEqual(payload["state"], "scanning")

    def test_ionex_can_be_loaded_through_the_http_boundary(self) -> None:
        query = urlencode({**_load_config(), "filename": "uploaded.ionex"})
        status, payload = self._json_request(
            f"/api/load?{query}",
            data=_ionex(epochs=((0, 100), (2, 200))),
            headers={
                "Content-Type": "application/octet-stream",
                "X-OPHANIM-Token": self.token,
                "Origin": self.base_url,
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["loaded"]["filename"], "uploaded.ionex")
        self.assertEqual(payload["loaded"]["inferred_cell_count"], 6)
        status, state = self._json_request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(
            state["loaded"]["dataset_id"],
            payload["loaded"]["dataset_id"],
        )

    def test_gim_can_be_fetched_through_token_protected_http_boundary(self) -> None:
        request_body = json.dumps(
            {"edition": "final", "date": "2024-01-02"}
        ).encode("utf-8")
        status, payload = self._json_request(
            "/api/gim/fetch",
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "X-OPHANIM-Token": self.token,
                "Origin": self.base_url,
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["loaded"]["provider"], "code")
        self.assertEqual(payload["loaded"]["revision"], "final")
        self.assertEqual(payload["loaded"]["global_cells_per_epoch_minimum"], 5_112)
        self.assertEqual(payload["acquisition"]["resolved_date"], "2024-01-02")
        self.assertEqual(len(self.gim_source.requests), 1)

        readout_body = json.dumps(
            {
                "dataset_id": payload["loaded"]["dataset_id"],
                "observed_at": payload["loaded"]["default_origin"],
                "latitude_step_degrees": 0.5,
                "longitude_step_degrees": 0.5,
            }
        ).encode("utf-8")
        status, readout = self._json_request(
            "/api/regional-grid",
            data=readout_body,
            headers={
                "Content-Type": "application/json",
                "X-OPHANIM-Token": self.token,
                "Origin": self.base_url,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(readout["readout"]["cell_count"], 80)
        self.assertEqual(
            readout["readout"]["value_kind"],
            "interpolated_estimate",
        )

    def test_mamba_status_initialize_and_scan_are_wired_as_async_operations(self) -> None:
        monitor = _FakeMambaMonitor()
        self.controller._mamba_monitor = monitor

        status, payload = self._json_request("/api/mamba/status")
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "uninitialized")

        region = {
            "name": "Central Texas",
            "south": 29.0,
            "west": -100.5,
            "north": 32.5,
            "east": -96.0,
        }
        status, payload = self._json_request(
            "/api/mamba/initialize",
            data=json.dumps({"history_years": 20, "region": region}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-OPHANIM-Token": self.token,
                "Origin": self.base_url,
            },
        )
        self.assertEqual(status, 202)
        self.assertEqual(payload["state"], "initializing")
        self.assertEqual(monitor.initializations, [{"history_years": 20, "region": region}])

        status, payload = self._json_request(
            "/api/mamba/scan",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-OPHANIM-Token": self.token,
                "Origin": self.base_url,
            },
        )
        self.assertEqual(status, 202)
        self.assertEqual(payload["state"], "scanning")
        self.assertEqual(monitor.scan_count, 1)

        status, payload = self._json_request(
            "/api/mamba/scan",
            data=b'{"from": 1}',
            headers={
                "Content-Type": "application/json",
                "X-OPHANIM-Token": self.token,
                "Origin": self.base_url,
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("does not accept", payload["error"])

    def test_mamba_expected_lifecycle_errors_have_specific_http_statuses(self) -> None:
        monitor = _FakeMambaMonitor()
        self.controller._mamba_monitor = monitor
        headers = {
            "Content-Type": "application/json",
            "X-OPHANIM-Token": self.token,
            "Origin": self.base_url,
        }

        monitor.failure = MambaMonitorConflict("already running")
        status, payload = self._json_request(
            "/api/mamba/initialize",
            data=b"{}",
            headers=headers,
        )
        self.assertEqual(status, 409)
        self.assertIn("already running", payload["error"])

        monitor.failure = MambaMonitorUnavailable("initialize first")
        status, payload = self._json_request(
            "/api/mamba/scan",
            data=b"{}",
            headers=headers,
        )
        self.assertEqual(status, 503)
        self.assertIn("initialize first", payload["error"])

    def test_spatial_status_initialize_and_scan_are_wired_as_async_operations(self) -> None:
        monitor = _FakeSpatialMonitor()
        self.controller._spatial_monitor = monitor

        status, payload = self._json_request("/api/spatial/status")
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "uninitialized")

        config = {
            "history_years": 20,
            "architecture": "predictive-convlstm",
            "input_frame_count": 12,
            "forecast_horizon_hours": 2,
        }
        headers = {
            "Content-Type": "application/json",
            "X-OPHANIM-Token": self.token,
            "Origin": self.base_url,
        }
        status, payload = self._json_request(
            "/api/spatial/initialize",
            data=json.dumps(config).encode(),
            headers=headers,
        )
        self.assertEqual(status, 202)
        self.assertEqual(payload["state"], "initializing")
        self.assertEqual(monitor.initializations, [config])

        status, payload = self._json_request(
            "/api/spatial/scan",
            data=b"{}",
            headers=headers,
        )
        self.assertEqual(status, 202)
        self.assertEqual(payload["state"], "scanning")
        self.assertEqual(monitor.scan_count, 1)

        status, payload = self._json_request(
            "/api/spatial/scan",
            data=b'{"from": "2024-01-01"}',
            headers=headers,
        )
        self.assertEqual(status, 400)
        self.assertIn("does not accept", payload["error"])

    def test_spatial_expected_lifecycle_errors_have_specific_http_statuses(self) -> None:
        monitor = _FakeSpatialMonitor()
        self.controller._spatial_monitor = monitor
        headers = {
            "Content-Type": "application/json",
            "X-OPHANIM-Token": self.token,
            "Origin": self.base_url,
        }

        monitor.failure = SpatialMonitorConflict("spatial training already running")
        status, payload = self._json_request(
            "/api/spatial/initialize",
            data=b"{}",
            headers=headers,
        )
        self.assertEqual(status, 409)
        self.assertIn("already running", payload["error"])

        monitor.failure = SpatialMonitorUnavailable("train spatial baseline first")
        status, payload = self._json_request(
            "/api/spatial/scan",
            data=b"{}",
            headers=headers,
        )
        self.assertEqual(status, 503)
        self.assertIn("train spatial baseline first", payload["error"])


if __name__ == "__main__":
    unittest.main()
