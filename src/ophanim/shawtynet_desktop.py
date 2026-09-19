"""Bounded local research jobs; scientific and artistic algorithms live elsewhere."""

from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
from urllib.parse import unquote


CASES = ("quiet", "translating_gaussian", "growing_gaussian", "translating_growing_gaussian",
         "plane_wave", "wave_packet", "moving_front", "crossing_waves", "noise", "missing")
DEFAULT_BOUNDS = {"south": 20.0, "north": 42.0, "west": -112.0, "east": -88.0}
_ASSET = re.compile(
    r"(?P<science>[0-9a-f]{24})/(?:(?P<science_file>diagnostics/(?:report\.html|[a-z0-9_-]+\.png))|"
    r"visuals/(?P<visual>[0-9a-f]{24})/(?P<visual_file>(?:visual_diagnostics/(?:report\.html|[a-z0-9_-]+\.png)|render_package/textures/color\.png)))\Z"
)


class ShawtyNetBusy(RuntimeError):
    """A job already owns this desktop research worker."""


class ShawtyNetUnavailable(RuntimeError):
    """The optional scientific environment is not installed."""


def _availability():
    names = ("numpy", "scipy", "xarray", "skimage", "pyproj", "zarr", "matplotlib", "PIL")
    missing = [name for name in names if find_spec(name) is None]
    return not missing


def _validated_request(payload):
    if not isinstance(payload, dict) or set(payload) - {"source", "case", "bounds"}:
        raise ValueError("ShawtyNet accepts only source, case and regional bounds")
    source = payload.get("source", "synthetic")
    if source not in ("synthetic", "desktop"):
        raise ValueError("source must be synthetic or desktop")
    case = payload.get("case", "plane_wave")
    if not isinstance(case, str) or case not in CASES:
        raise ValueError("unknown synthetic case")
    raw_bounds = payload.get("bounds", DEFAULT_BOUNDS)
    if not isinstance(raw_bounds, dict) or set(raw_bounds) != set(DEFAULT_BOUNDS):
        raise ValueError("bounds require south, north, west and east")
    bounds = {}
    for name, value in raw_bounds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("bounds must be finite numeric degrees")
        bounds[name] = float(value)
    if (not -75 <= bounds["south"] < bounds["north"] <= 75
            or not -180 <= bounds["west"] < bounds["east"] <= 180
            or not 5 <= bounds["north"] - bounds["south"] <= 30
            or not 10 <= bounds["east"] - bounds["west"] <= 40):
        raise ValueError("Select a region 5–30° tall and 10–40° wide, within ±75° latitude")
    return {"source": source, "case": case, "bounds": bounds}


class ShawtyNetDesktopJobs:
    """One background job with durable last status and immutable result links.

    Construction/status require no scientific imports or model initialization.
    A restart marks an unfinished job interrupted; submitting again safely reuses
    completed content-addressed science/art stages through science_runs.
    """

    def __init__(self, data_directory, *, runner=None, availability=None):
        self._data_directory = Path(data_directory).expanduser().resolve()
        self.root = self._data_directory / "shawtynet"
        self._state_path = self.root / "desktop-state.json"
        self._lock = threading.RLock()
        self._thread = None
        self._closed = False
        self._runner = runner or self._run
        self._availability = availability or _availability
        self._state = {"state": "idle", "message": "Ready for a synthetic example or acquired TEC.",
                       "job_id": None, "result": None, "error": None}
        if self._state_path.is_file() and self._state_path.stat().st_size < 256 * 1024:
            try:
                saved = json.loads(self._state_path.read_text(encoding="utf-8"))
                if saved.get("schema") == "ophanim-shawtynet-desktop/1" and saved.get("state") in {"running", "complete", "failed", "interrupted"}:
                    self._state = saved
                    if saved["state"] == "running":
                        self._state.update(state="interrupted", message="The previous job was interrupted. Run it again to continue.")
            except (ValueError, OSError):
                pass

    def status(self):
        with self._lock:
            result = deepcopy(self._state)
        available = self._availability()
        return {"ok": True, "available": available,
                "unavailable_reason": None if available else "Install OPHANIM's shawtynet optional dependencies in the application's Python environment.",
                "cases": list(CASES), "default_bounds": dict(DEFAULT_BOUNDS), **result}

    def analyze(self, payload):
        request = _validated_request(payload)
        if not self._availability():
            raise ShawtyNetUnavailable("Install OPHANIM's shawtynet optional dependencies to run research analysis")
        with self._lock:
            if self._closed:
                raise ShawtyNetUnavailable("The application is stopping")
            if self._thread is not None and self._thread.is_alive():
                raise ShawtyNetBusy("A ShawtyNet research job is already running")
            self._state = {"state": "running", "message": "Preparing source data…",
                           "job_id": secrets.token_hex(12), "request": request,
                           "started_at": datetime.now(UTC).isoformat(), "result": None, "error": None}
            self._persist()
            self._thread = threading.Thread(target=self._work, args=(request,),
                                            name="ophanim-shawtynet", daemon=True)
            self._thread.start()
        return self.status()

    def _persist(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".desktop-state-{secrets.token_hex(8)}.tmp"
        payload = {"schema": "ophanim-shawtynet-desktop/1", **self._state}
        temporary.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self._state_path)

    def _progress(self, message):
        with self._lock:
            self._state["message"] = message
            self._persist()

    def _work(self, request):
        try:
            from ophanim.core.jobs import heavy_work
            self._progress("Waiting for the shared compute slot…")
            with heavy_work(self._data_directory, lambda: self._closed):
                result = self._runner(request, self._progress)
            with self._lock:
                self._state.update(state="complete", message="Scientific packet and artistic fields are ready.",
                                   result=result, finished_at=datetime.now(UTC).isoformat())
                self._persist()
        except InterruptedError:
            with self._lock:
                self._state.update(state="interrupted", message="Interrupted while waiting for compute; retry when ready.",
                                   finished_at=datetime.now(UTC).isoformat())
                self._persist()
        except Exception as error:
            message = str(error).replace(str(self._data_directory), "local data")[:1500]
            with self._lock:
                self._state.update(state="failed", message="Analysis could not finish.", error=message,
                                   finished_at=datetime.now(UTC).isoformat())
                self._persist()

    def close(self, timeout=0.1):
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return thread is None or not thread.is_alive()

    def _run(self, request, progress):
        from ophanim.dynamics import AnalysisConfig
        from ophanim.science_runs import analyze_run, verify_run, visualize_run

        if request["source"] == "synthetic":
            from ophanim.experiments import make_synthetic_dataset
            source = make_synthetic_dataset(kind=request["case"])
            config = AnalysisConfig(baseline_method="constant",
                                    constant_background_tecu=source.attrs["synthetic_truth"]["background_tecu"])
        else:
            from ophanim.sensing import read_desktop_sequence
            database = self._data_directory / "ophanim.sqlite3"
            if not database.is_file():
                raise ValueError("Download or load a TEC source map in OPHANIM first.")
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
                latest = connection.execute("SELECT MAX(observed_at) FROM tec_observations").fetchone()[0]
            if latest is None:
                raise ValueError("Download or load a TEC source map in OPHANIM first.")
            end = datetime.fromisoformat(latest.replace("Z", "+00:00"))
            source = read_desktop_sequence(self._data_directory, start=end - timedelta(hours=96),
                                           end=end, bounds=request["bounds"])
            config = AnalysisConfig(science_spacing_km=75, baseline_method="rolling_median",
                                    baseline_window_minutes=720, short_window_minutes=360,
                                    wave_window_hours=24,
                                    smooth_sigma_km=100, structure_sigma_km=200,
                                    target_time=(end - timedelta(hours=6)).isoformat())
        progress("Computing structure, apparent motion and source capability checks…")
        science = analyze_run(source, config, self.root, request=request)
        verify_run(science, kind="science")
        progress("Generating artistic fields and diagnostics…")
        visual = visualize_run(science)
        verify_run(visual, kind="visual")
        packet = json.loads((science / "science_packet.json").read_text(encoding="utf-8"))
        event = json.loads((science / "event.json").read_text(encoding="utf-8"))
        prefix = f"/shawtynet-assets/{science.name}"
        visual_prefix = f"{prefix}/visuals/{visual.name}"
        return {
            "science_run_id": science.name, "visual_run_id": visual.name,
            "source_kind": packet.get("source_kind"), "time_start": packet.get("time_start"),
            "time_end": packet.get("time_end"), "capabilities": packet.get("capabilities", {}),
            "analysis_mode": "retrospective", "target_time": config.target_time or event.get("time"),
            "event": event,
            "science_report_url": f"{prefix}/diagnostics/report.html",
            "visual_report_url": f"{visual_prefix}/visual_diagnostics/report.html",
            "preview_url": f"{visual_prefix}/render_package/textures/color.png",
        }

    def asset(self, relative_url):
        """Serve only inventory-verified diagnostic HTML/PNG, never raw paths."""
        decoded = unquote(relative_url)
        match = _ASSET.fullmatch(decoded)
        if match is None:
            raise ValueError("Unknown research artifact")
        root = self.root.resolve()
        candidate = root / decoded
        if (not candidate.resolve().is_relative_to(root) or candidate.is_symlink()
                or any(parent.is_symlink() for parent in candidate.parents if parent != root and parent.is_relative_to(root))):
            raise ValueError("Research artifact cannot traverse symlinks")
        run = root / match["science"]
        relative = match["science_file"]
        if match["visual"]:
            run = run / "visuals" / match["visual"]
            relative = match["visual_file"]
        manifest_path = run / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Research run has no valid manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        expected = files.get(relative) if isinstance(files, dict) else None
        if (manifest.get("schema") != "ophanim-science-run/1" or manifest.get("status") != "complete"
                or manifest.get("run_id") != run.name
                or manifest.get("kind") != ("visual" if match["visual"] else "science")
                or not isinstance(expected, str) or not candidate.is_file()
                or candidate.stat().st_size > 32 * 1024 * 1024):
            raise ValueError("Research artifact is unavailable or incomplete")
        content = candidate.read_bytes()
        if sha256(content).hexdigest() != expected:
            raise ValueError("Research artifact checksum mismatch")
        return content, "text/html; charset=utf-8" if candidate.suffix == ".html" else "image/png"
