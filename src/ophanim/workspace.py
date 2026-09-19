"""Local application composition: reusable core → experiments → optional art.

This is deliberately outside all three subsystems. It owns durable user journeys,
not acquisition algorithms, model training, scientific claims, or visual recipes.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import sqlite3
import threading
from urllib.parse import unquote
import zipfile

from ophanim.core.jobs import WorkCancelled, heavy_work


class WorkspaceBusy(RuntimeError):
    pass


class WorkspaceUnavailable(RuntimeError):
    pass


_ID = re.compile(r"[0-9a-f]{24}\Z")
_ASSET = re.compile(r"(?P<id>[0-9a-f]{24})/(?P<file>(?:science|visual)/(?:report\.html|[a-z0-9_-]+\.png)|preview\.png|render\.png|composite\.png|animation\.gif|animation\.zip|package\.zip|recipe\.json)\Z")
_ACTIVE = ("queued", "running", "cancelling")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _now():
    return datetime.now(UTC).isoformat()


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid workspace identifier")
    return value


def _available():
    return all(find_spec(name) is not None for name in ("numpy", "scipy", "xarray", "zarr", "pyproj", "skimage", "matplotlib", "PIL"))


def _blender():
    # Only server configuration chooses an executable; never an HTTP request.
    configured = os.environ.get("OPHANIM_BLENDER")
    candidates = ([configured] if configured else []) + ["blender", "/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"]
    return next((path for candidate in candidates if (path := shutil.which(candidate))), None)


def _camera_payload(value):
    if value is None:
        return None
    allowed = {"heading_deg": (-360, 360), "pitch_deg": (-90, 90), "roll_deg": (-180, 180),
               "horizontal_fov_deg": (5, 150), "observer_latitude": (-90, 90),
               "observer_longitude": (-180, 180), "observer_altitude_km": (-1, 1000)}
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Camera accepts observer latitude/longitude/altitude, heading, pitch, roll and horizontal field of view")
    for key, number in value.items():
        low, high = allowed[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"Invalid camera {key}")
    return value


def _analysis_frame_times(times, *, representative=0, maximum=12):
    """Bounded exact epochs, including display and final analysis targets.

    This chooses analysis timestamps, not interpolated animation observations.
    The selected display frame never shortens the primary scientific window.
    """
    import numpy as np
    indices = set(np.linspace(0, len(times)-1, min(maximum-1, len(times)), dtype=int).tolist())
    indices.add(int(representative))
    return [np.datetime_as_string(times[index], unit="s") + "Z" for index in sorted(indices)]


def _representative_frame(source, kind):
    """Choose a visible feature, not a late uniform plateau or an event time."""
    import numpy as np
    anomaly = np.asarray(source.tec.values) - 20
    if kind == "moving_front":
        gy, gx = np.gradient(anomaly, axis=(1, 2))
        energy = np.mean(gx*gx + gy*gy, axis=(1, 2))
        method = "near-maximum spatial gradient energy; keeps the front transition in view"
    else:
        energy = np.mean(anomaly*anomaly, axis=(1, 2))
        method = "near-maximum departure energy from the chosen background; middle of supported energetic interval"
    eligible = np.arange(1, len(energy)-1)
    if not len(eligible):
        return 0, method
    peak = float(np.max(energy[eligible]))
    strongest = eligible[energy[eligible] >= .98*peak]
    return int(strongest[len(strongest)//2]), method


def _observed_wave_settings(times):
    """Choose a cadence-compatible search band without relaxing admission.

    Coarse GIM products may only support multi-hour periodic image structure,
    not the 30–180 minute band used for finer synthetic examples. Spatial and
    gap/occupancy gates still apply independently and can reject this band.
    """
    import numpy as np
    from ophanim.dynamics.schemas import WaveConfig
    cadence = float(np.median(np.diff(times).astype("timedelta64[ns]").astype(float))/60e9)
    minimum = max(30.0, 6*cadence)
    maximum = max(180.0, 2*minimum)
    return WaveConfig(min_period_minutes=minimum, max_period_minutes=maximum), max(24.0, 4*minimum/60)


class Workspace:
    """One durable workspace queue with saved events, scenarios and render lineage.

    Completed results survive cancellation/failure of later work. A restart marks
    unfinished jobs interrupted; retries reuse immutable completed stages. No
    download, model initialization, or numerical import happens on a status read.
    """

    def __init__(self, data_directory, *, runner=None, availability=None, source=None):
        self.data_directory = Path(data_directory).expanduser().resolve()
        self.root = self.data_directory / "workspace"
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "workspace.sqlite3"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._thread = None
        self._queue = queue.Queue()
        self._runner = runner or self._run
        self._availability = availability or _available
        self._source = source
        self._process_lock = None
        try:
            import fcntl
            self._process_lock = (self.root / "worker.lock").open("a+b")
            try:
                fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self._process_lock.close()
                raise WorkspaceBusy("Another application owns this workspace") from error
        except ImportError:
            pass
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                    stage TEXT NOT NULL, message TEXT NOT NULL, progress REAL,
                    request_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    finished_at TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS records (
                    record_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS workspace_records_kind ON records(kind, created_at);
            """)
            connection.execute("UPDATE jobs SET state='interrupted',stage='interrupted',message='Interrupted by restart; retry the saved request.',finished_at=? WHERE state IN ('queued','running','cancelling')", (_now(),))
            connection.commit()

    def _connection(self):
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        return closing(connection)

    def _job(self, row):
        if row is None:
            return None
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        return result

    def status(self):
        with self._connection() as connection:
            job = self._job(connection.execute("SELECT * FROM jobs ORDER BY created_at DESC,rowid DESC LIMIT 1").fetchone())
            rows = []
            for kind in ("scan", "scenario", "project"):
                rows.extend(connection.execute("SELECT * FROM records WHERE kind=? ORDER BY created_at DESC,rowid DESC LIMIT 30", (kind,)).fetchall())
        scans, scenarios, projects = [], [], []
        last_scan = None
        for row in rows:
            value = json.loads(row["payload_json"])
            if row["kind"] == "scan":
                if last_scan is None:
                    last_scan = value
                scans.append({"scan_id": row["record_id"], "created_at": row["created_at"],
                              "status": value.get("status"), "candidate_count": len(value.get("candidates", []))})
            elif row["kind"] == "scenario":
                scenarios.append({"scenario_id": row["record_id"], **value})
            elif row["kind"] == "project":
                projects.append(self._public_project(value))
        return {"ok": True, "available": bool(self._availability()),
                "capabilities": {"core": bool(self._availability()),
                                 "experiments": "2D scenarios and imagined 3D volumes; model evaluation and agent training are extension contracts",
                                 "imagined_3d": bool(self._availability()),
                                 "ifm": {"status": "planned", "milestone": "v0.3"},
                                 "remote_simulation": {"status": "not_configured", "automatic_uploads": False},
                                 "agent_training": "not_configured", "novel_model": "not_configured",
                                 "blender": _blender() is not None},
                "job": job, "last_scan": last_scan, "scans": scans[:30],
                "scenarios": scenarios[:30], "projects": projects[:30]}

    def record(self, identity, kind):
        with self._connection() as connection:
            row = connection.execute("SELECT payload_json FROM records WHERE record_id=? AND kind=?", (_identifier(identity), kind)).fetchone()
        if row is None:
            raise ValueError(f"Saved {kind} not found")
        return json.loads(row[0])

    def _save(self, kind, payload):
        identity = sha256(_json(payload).encode()).hexdigest()[:24]
        payload = {**payload, f"{kind}_id": identity}
        with self._connection() as connection:
            # Reusing an immutable artifact is still a new user selection. Move
            # its history entry to the front without changing its payload/ID.
            connection.execute("INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET created_at=excluded.created_at", (identity, kind, _json(payload), _now()))
            connection.commit()
        return payload

    def candidate(self, identity):
        _identifier(identity)
        with self._connection() as connection:
            rows = connection.execute("SELECT payload_json FROM records WHERE kind='scan' ORDER BY created_at DESC").fetchall()
        for row in rows:
            scan = json.loads(row[0])
            for candidate in scan.get("candidates", []):
                if candidate["candidate_id"] == identity:
                    return {**candidate, "scan_id": scan["scan_id"], "snapshot_id": scan["snapshot_id"], "source_kind": scan.get("source", {}).get("source_kind")}
        raise ValueError("Saved candidate not found")

    def submit(self, kind, payload):
        if not self._availability():
            raise WorkspaceUnavailable("Install OPHANIM's science dependencies to use discovery and scenarios")
        request = self._validate(kind, payload)
        with self._lock:
            if self._stop.is_set():
                raise WorkspaceUnavailable("Application is stopping")
            with self._connection() as connection:
                if connection.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancelling') LIMIT 1").fetchone():
                    raise WorkspaceBusy("A workspace job is already active")
                identity = secrets.token_hex(12)
                connection.execute("INSERT INTO jobs VALUES (?,?,?,'queued',?,0,?,?,NULL,NULL)",
                                   (identity, kind, "queued", "Queued; completed results remain available.", _json(request), _now()))
                connection.commit()
            self._cancel.clear()
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name="ophanim-workspace", daemon=True)
                self._thread.start()
            self._queue.put((identity, kind, request))
        return self.status()

    def _validate(self, kind, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request must be an object")
        allowed = {"scan": {"source", "days"}, "scenario": {"candidate_id", "parameters", "style", "camera"},
                   "imagine": {"candidate_id", "source_project_id", "parameters", "style", "camera"},
                   "artify": {"candidate_id", "style", "camera"}, "style": {"project_id", "style", "camera"},
                   "render": {"project_id"}, "animation": {"project_id", "frame_count"},
                   "composite": {"project_id", "photo_id", "opacity", "horizon_y", "horizon_fade",
                                 "saturation", "exposure", "grade_rgb", "foreground_mask_id", "cloud_mask_id"}}
        if kind not in allowed or set(payload) - allowed[kind]:
            raise ValueError("Unsupported workspace request fields")
        request = dict(payload)
        if kind == "scan":
            request.setdefault("source", "latest")
            request.setdefault("days", 8)
            if request["source"] not in ("latest", "cached", "demo"):
                raise ValueError("Scan source must be latest, cached or demo")
            if isinstance(request["days"], bool) or not isinstance(request["days"], int) or not 4 <= request["days"] <= 14:
                raise ValueError("Scan lookback must be 4–14 days")
        if "candidate_id" in request:
            self.candidate(request["candidate_id"])
        if kind == "artify" and "candidate_id" not in request:
            raise ValueError("Choose a saved event first")
        if kind in ("style", "render", "animation", "composite"):
            self.record(request.get("project_id"), "project")
        if kind == "animation":
            request.setdefault("frame_count", 6)
            count = request["frame_count"]
            if isinstance(count, bool) or not isinstance(count, int) or not 3 <= count <= 12:
                raise ValueError("Animation must contain 3–12 frames")
        if kind == "composite":
            photo = self.record(request.get("photo_id"), "photo")
            if not self.record(request["project_id"], "project").get("render_directory"):
                raise ValueError("Render a still frame before making a photo composite")
            controls = {"opacity": (.65, 0, 1), "horizon_y": (.5, .001, 1),
                        "horizon_fade": (.6, 0, 1), "saturation": (.85, 0, 2), "exposure": (1, 0, 4)}
            for key, (default, low, high) in controls.items():
                number = request.setdefault(key, default)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
                    raise ValueError(f"Photo {key} must be finite and between {low} and {high}")
            grade = request.setdefault("grade_rgb", [1, 1, 1])
            if (not isinstance(grade, list) or len(grade) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 2 for v in grade)):
                raise ValueError("Photo grade_rgb requires three gains between zero and two")
            for key in ("foreground_mask_id", "cloud_mask_id"):
                if request.get(key) is not None:
                    mask = self.record(request[key], "mask")
                    if (mask["width"], mask["height"]) != (photo["width"], photo["height"]):
                        raise ValueError("Masks must match the oriented photograph's original dimensions")
        if kind in ("scenario", "imagine", "artify", "style"):
            request.setdefault("style", "luminous" if kind == "imagine" else "ghost")
            if request["style"] not in ("ghost", "luminous", "quiet"):
                raise ValueError("Unknown artistic style")
            request["camera"] = _camera_payload(request.get("camera"))
        if kind == "scenario":
            from ophanim.experiments.scenarios import ScenarioConfig
            # Validate supplied fields without filling defaults over inherited observations.
            ScenarioConfig.from_mapping(request.get("parameters", {}))
        if kind == "imagine":
            from ophanim.experiments.hypotheses import hypothesis_from_evidence
            candidate = self.candidate(request["candidate_id"]) if request.get("candidate_id") else None
            hypothesis_from_evidence(candidate=candidate, overrides=request.get("parameters", {}))
            if request.get("source_project_id"):
                source = self.record(request["source_project_id"], "project")
                if (not source.get("science_directory") or source.get("kind") != "observations"
                        or not candidate or source.get("candidate_id") != candidate["candidate_id"]):
                    raise ValueError("Scientific input must be an observation project for the same candidate")
        imagined = kind == "imagine" or (kind == "style" and self.record(request["project_id"], "project").get("kind") == "imagined_3d")
        if imagined and request.get("camera") and any(key.startswith("observer_") for key in request["camera"]):
            raise ValueError("3D art uses an auto-framed orbit camera, not a ground observer; supply only heading, pitch, roll and field of view")
        if imagined and request.get("camera"):
            request["camera"] = {"heading_deg": -35, "pitch_deg": -18, "roll_deg": 0,
                                 "horizontal_fov_deg": 55, **request["camera"]}
        if kind in ("render", "animation") and _blender() is None:
            raise WorkspaceUnavailable("Blender is unavailable; the export package can be rendered elsewhere")
        return request

    def upload_photo(self, content):
        if find_spec("PIL") is None:
            raise WorkspaceUnavailable("Install OPHANIM's shawtynet dependencies to upload photographs")
        from ophanim.shawtynet.studio import upload_photo
        result = upload_photo(self.root / "photographs", content)
        result.pop("photo_id", None)
        saved = self._save("photo", result)
        return {"ok": True, "photo": {key: saved[key] for key in ("photo_id", "width", "height")}}

    def upload_mask(self, content):
        if find_spec("PIL") is None:
            raise WorkspaceUnavailable("Install OPHANIM's shawtynet dependencies to upload masks")
        from ophanim.shawtynet.studio import upload_mask
        result = upload_mask(self.root / "masks", content)
        result.pop("mask_id", None)
        saved = self._save("mask", result)
        return {"ok": True, "mask": {key: saved[key] for key in ("mask_id", "width", "height")}}

    def cancel(self, identity=None):
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT job_id FROM jobs WHERE state IN ('queued','running','cancelling') ORDER BY created_at DESC LIMIT 1").fetchone()
            if row is not None:
                if identity is not None and _identifier(identity) != row[0]:
                    raise ValueError("The selected job is no longer active")
                self._cancel.set()
                connection.execute("UPDATE jobs SET state='cancelling',message='Cancellation requested; finishing the current safe stage.' WHERE job_id=?", (row[0],))
                connection.commit()
        return self.status()

    def close(self, timeout=.2):
        self._stop.set()
        self._cancel.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)
        stopped = self._thread is None or not self._thread.is_alive()
        if stopped and self._process_lock is not None:
            self._process_lock.close()
            self._process_lock = None
        return stopped

    def _check(self):
        if self._stop.is_set() or self._cancel.is_set():
            raise WorkCancelled("Cancelled at a safe processing boundary")

    def _worker(self):
        try:
            while not self._stop.is_set():
                item = self._queue.get()
                if item is None:
                    return
                identity, kind, request = item
                try:
                    self._check()
                    def progress(stage, message=None, fraction=None):
                        self._check()
                        with self._connection() as connection:
                            connection.execute("UPDATE jobs SET state='running',stage=?,message=?,progress=? WHERE job_id=?", (str(stage), str(message or stage), fraction, identity))
                            connection.commit()
                    progress("starting", "Preparing the requested stage…", 0)
                    self._runner(kind, request, progress)
                    self._check()
                    state, message, error = "complete", "Ready. Results and provenance are saved.", None
                except InterruptedError:
                    state, message, error = "interrupted" if self._stop.is_set() else "cancelled", "Stopped safely; completed stages remain reusable.", None
                except Exception as failure:
                    state, message = "failed", "This stage could not finish. Earlier results are unchanged."
                    error = str(failure).replace(str(self.data_directory), "local data")[:1200]
                with self._connection() as connection:
                    connection.execute("UPDATE jobs SET state=?,stage=?,message=?,error=?,finished_at=?,progress=? WHERE job_id=?", (state,state,message,error,_now(),1 if state=="complete" else None,identity))
                    connection.commit()
        finally:
            if self._process_lock is not None:
                self._process_lock.close()
                self._process_lock = None

    def _snapshot(self, dataset):
        from ophanim.core.artifacts import _publish, dataset_sha256
        from ophanim.core.runs import _write_dataset
        identity = {"input_sha256": dataset_sha256(dataset), "schema": "native-snapshot/1"}
        name = sha256(_json(identity).encode()).hexdigest()[:24]
        return _publish(self.root / "snapshots", name, "snapshot", identity,
                        lambda stage: _write_dataset(dataset, stage / "native.zarr"))

    def _open_snapshot(self, identity):
        import xarray as xr
        from ophanim.core.artifacts import verify_run
        path = self.root / "snapshots" / _identifier(identity)
        verify_run(path, kind="snapshot")
        return xr.open_zarr(path / "native.zarr", consolidated=True)

    def _run(self, kind, request, progress):
        if kind == "scan":
            from ophanim.core.acquisition import acquire_recent, read_recent
            from ophanim.core.discovery import survey_dataset
            if request["source"] == "latest":
                progress("acquiring", "Checking a bounded recent window; no model training is started.", .05)
                source = acquire_recent(self.data_directory, days=request["days"],
                                        progress=lambda *args: progress("acquiring", " · ".join(map(str,args)), .15),
                                        cancelled=lambda: self._cancel.is_set() or self._stop.is_set(), source=self._source)
            elif request["source"] == "cached":
                source = read_recent(self.data_directory, days=request["days"])
            else:
                from ophanim.experiments.discovery_demo import make_discovery_demo
                source = make_discovery_demo()
            self._check()
            progress("waiting", "Waiting for the shared compute slot…", .35)
            with heavy_work(self.data_directory, lambda: self._cancel.is_set() or self._stop.is_set()):
                progress("surveying", "Comparing native cells and grouping persistent candidate regions…", .45)
                result = survey_dataset(source)
                self._check()
                snapshot = self._snapshot(source)
                result["snapshot_id"] = snapshot.name
                result.setdefault("source", {})["source_kind"] = source.attrs.get("source_kind", "unknown")
                result["request"] = request
            if request["source"] == "latest" and result.get("candidates"):
                from ophanim.core.context import annotate_candidates, fetch_geomagnetic_context
                progress("context", "Checking optional, timestamp-matched geomagnetic context…", .9)
                context = fetch_geomagnetic_context()
                result["geomagnetic_context"] = context
                result["candidates"] = annotate_candidates(result["candidates"], context)
            self._check()
            self._save("scan", result)
            return
        progress("waiting", "Waiting for the shared compute slot…", .1)
        with heavy_work(self.data_directory, lambda: self._cancel.is_set() or self._stop.is_set()):
            if kind == "scenario":
                self._scenario(request, progress)
            elif kind == "imagine":
                self._imagine(request, progress)
            elif kind == "artify":
                self._artify(request, progress)
            elif kind == "style":
                original = self.record(request["project_id"], "project")
                if original.get("kind") == "imagined_3d":
                    metadata = {key: value for key, value in original.items()
                                if key not in {"project_id", "visual_directory", "render_directory", "animation_directory",
                                               "composite_directory", "photo_id", "photo_settings", "photo_opacity"}}
                    metadata["parent_project_id"] = original["project_id"]
                    self._volume_project(Path(original["volume_directory"]), request, progress, metadata)
                else:
                    self._visual_project(Path(original["science_directory"]), request, progress,
                                     {"kind": original["kind"], "title": original["title"],
                                      "parent_project_id": original["project_id"],
                                      **{key:original[key] for key in ("scenario_id","candidate_id","parameters","provenance","representative_frame") if key in original}})
            elif kind == "render":
                from ophanim.shawtynet.runs import render_run
                project = self.record(request["project_id"], "project")
                progress("rendering", "Rendering one frame with two Blender threads…", .3)
                if project.get("kind") == "imagined_3d":
                    from ophanim.shawtynet.volume import render_volume_run
                    directory = render_volume_run(project["visual_directory"], blender_executable=_blender(),
                        samples=24, resolution=(960,640), threads=2,
                        cancelled=lambda: self._cancel.is_set() or self._stop.is_set())
                else:
                    directory = render_run(project["visual_directory"], blender_executable=_blender(), samples=16, resolution=(960,640), threads=2)
                self._check()
                child = {key:value for key,value in project.items() if key != "project_id"}
                child.update(parent_project_id=project["project_id"], render_directory=str(directory), active_view="render")
                self._save("project", child)
            elif kind in ("animation", "composite"):
                from ophanim.shawtynet.studio import animation_run, studio_composite_run
                project = self.record(request["project_id"], "project")
                if kind == "animation":
                    progress("animating", "Rendering a bounded sequence; cancellation is checked between frames…", .2)
                    options = dict(frame_count=request["frame_count"], blender_executable=_blender(),
                                   cancelled=lambda: self._cancel.is_set() or self._stop.is_set(),
                                   progress=lambda *args: progress("animating", " · ".join(map(str,args)), .5))
                    if project.get("kind") == "imagined_3d":
                        from ophanim.shawtynet.volume import animation_volume_run
                        directory = animation_volume_run(project["visual_directory"], **options)
                    else:
                        style, camera = self._visual_settings(Path(project["science_directory"]), project)
                        directory = animation_run(project["science_directory"], style, camera, **options)
                else:
                    progress("compositing", "Center-cropping the photo for a manually aligned artistic overlay…", .4)
                    photo = self.record(request["photo_id"], "photo")
                    options = {key: request[key] for key in ("opacity", "horizon_y", "horizon_fade", "saturation", "exposure", "grade_rgb")}
                    for key in ("foreground_mask", "cloud_mask"):
                        if request.get(key+"_id"):
                            options[key+"_directory"] = self.record(request[key+"_id"], "mask")["directory"]
                    directory = studio_composite_run(project["render_directory"], photo["directory"], **options)
                self._check()
                child = {key:value for key,value in project.items() if key != "project_id"}
                child.update(parent_project_id=project["project_id"], active_view=kind, **{kind+"_directory":str(directory)})
                if kind == "composite":
                    child.update(photo_id=request["photo_id"],photo_opacity=request["opacity"],
                                 photo_settings={key:value for key,value in request.items() if key not in ("project_id", "photo_id")})
                self._save("project", child)

    def _imagine(self, request, progress):
        from ophanim.core.artifacts import verify_run
        from ophanim.experiments.hypotheses import hypothesis_from_evidence
        from ophanim.experiments.volume_model import publish_hypothesis_run
        candidate = self.candidate(request["candidate_id"]) if request.get("candidate_id") else None
        event = None
        source_project = None
        if request.get("source_project_id"):
            source_project = self.record(request["source_project_id"], "project")
            science = Path(source_project["science_directory"])
            if not science.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Scientific input is outside this workspace")
            verify_run(science, kind="science")
            event = json.loads((science / "event.json").read_text())
        progress("hypothesizing", "Freezing source evidence and recording a revisable 3D hypothesis…", .2)
        recipe = hypothesis_from_evidence(candidate=candidate, event=event, overrides=request.get("parameters", {}))
        progress("simulating", "Evolving an imagined 3D field; altitude and material density are assumptions…", .35)
        volume = publish_hypothesis_run(recipe, self.root / "volumes",
                                       cancelled=lambda: self._cancel.is_set() or self._stop.is_set())
        self._check()
        saved = self._save("scenario", recipe)
        metadata = {"kind": "imagined_3d", "scenario_id": saved["scenario_id"],
                    "title": "Imagined 3D · " + recipe["parameters"]["kind"].replace("_", " "),
                    "parameters": recipe["parameters"], "provenance": recipe["provenance"],
                    "hypothesis_semantics": recipe["semantics"], "evidence_sha256": recipe["evidence_sha256"],
                    "simulation_backend": "local_procedural", "ifm": recipe["ifm"]}
        if candidate:
            metadata["candidate_id"] = candidate["candidate_id"]
        if source_project:
            metadata.update(source_project_id=source_project["project_id"], science_directory=source_project["science_directory"])
        self._volume_project(volume, request, progress, metadata)

    def _volume_project(self, volume, request, progress, metadata):
        from ophanim.shawtynet import CameraConfig
        from ophanim.shawtynet.volume import visualize_volume_run
        parameters = metadata["parameters"]
        camera = None
        if request.get("camera"):
            camera = CameraConfig(**{"observer_latitude": parameters["center_latitude"],
                                     "observer_longitude": parameters["center_longitude"], **request["camera"]})
        progress("visualizing", "Shaping depth-rich ribbons and filaments from the imagined volume…", .7)
        visual = visualize_volume_run(volume, style=request["style"], camera=camera,
                                      cancelled=lambda: self._cancel.is_set() or self._stop.is_set())
        self._check()
        self._save("project", {**metadata, "volume_directory": str(volume), "visual_directory": str(visual),
                               "style": request["style"], "camera": request.get("camera"), "active_view": "preview"})

    def _scenario(self, request, progress):
        import numpy as np
        from ophanim.experiments.scenarios import scenario_from_candidate, make_scenario
        from ophanim.dynamics import AnalysisConfig
        from ophanim.core.runs import analyze_run
        candidate = self.candidate(request["candidate_id"]) if request.get("candidate_id") else None
        recipe = scenario_from_candidate(candidate, overrides=request.get("parameters", {}))
        saved = self._save("scenario", recipe)
        source = make_scenario(recipe)
        progress("simulating", "Generating a hypothetical mathematical scenario; not reconstructed observations.", .25)
        # A moving feature may have left the small domain by the final epoch.
        # Choose a representative interior frame by signal energy, explicitly
        # as a display choice, never a truncation of scientific history.
        frame, method = _representative_frame(source, recipe["parameters"]["kind"])
        selected_time = np.datetime_as_string(source.time.values[frame], unit="s") + "Z"
        config = AnalysisConfig(baseline_method="constant", constant_background_tecu=20,
                                wave_window_hours=max(4, recipe["parameters"]["duration_minutes"]/60),
                                max_cube_cells=200_000)
        selection = {"status":"chosen", "method":method,
                     "frame_index":frame, "time":selected_time}
        science = analyze_run(source, config, self.root / "science",
                              frame_times=_analysis_frame_times(source.time.values, representative=frame),
                              cancelled=lambda: self._cancel.is_set() or self._stop.is_set(),
                              request={"scenario_id":saved["scenario_id"],"recipe":recipe,"representative_frame":selection,
                                       "window_policy":"final epoch primary analysis; separate trailing wave windows at display/animation epochs; shared retrospective baseline"})
        self._check()
        self._visual_project(science, request, progress, {
            "kind":"hypothetical", "title":"Hypothetical " + str(recipe["parameters"]["kind"]).replace("_"," "),
            "scenario_id":saved["scenario_id"], "candidate_id":request.get("candidate_id"),
            "parameters":recipe["parameters"], "provenance":recipe.get("provenance", {}),
            "representative_frame":selection,
        })

    def _artify(self, request, progress):
        import numpy as np
        from ophanim.dynamics import AnalysisConfig
        from ophanim.core.runs import analyze_run
        candidate = self.candidate(request["candidate_id"])
        center = candidate["centroid"]
        lat, lon = center["latitude"], center["longitude"]
        if abs(lat) > 70:
            raise ValueError("Observed local mapping currently supports event centers within ±70° latitude. The native event remains available; use an explicitly hypothetical scenario for polar experiments.")
        with self._open_snapshot(candidate["snapshot_id"]) as opened:
            # A broad context window is separate from the selected event's footprint.
            longitude_distance = (opened.lon.values-lon+180)%360-180
            indices = np.flatnonzero(np.abs(longitude_distance) <= 15)
            source = opened.isel(lon=indices).sel(lat=slice(max(-75,lat-12),min(75,lat+12))).load()
        if min(source.sizes["lat"], source.sizes["lon"]) < 3:
            raise ValueError("This candidate lacks enough regional support for scientific mapping")
        progress("analyzing", "Analyzing the source snapshot; scientific gaps stay explicit.", .3)
        wave, wave_hours = _observed_wave_settings(source.time.values)
        bounds = {key: candidate["bounds"][key] for key in ("south", "north", "west", "east")}
        dy = float(np.median(np.abs(np.diff(source.lat.values)))) / 2
        unwrapped_longitude = np.degrees(np.unwrap(np.radians(source.lon.values)))
        dx = float(np.median(np.abs(np.diff(unwrapped_longitude)))) / 2
        bounds.update(south=max(-90,bounds["south"]-dy),north=min(90,bounds["north"]+dy),
                      west=(bounds["west"]-dx+180)%360-180,east=(bounds["east"]+dx+180)%360-180)
        config = AnalysisConfig(science_spacing_km=100, smooth_sigma_km=100, structure_sigma_km=200,
                                projection_center_lat=lat,projection_center_lon=lon,
                                baseline_method="savgol",baseline_window_minutes=720,
                                short_window_minutes=360,wave_window_hours=wave_hours,wave=wave,event_focus_bounds=bounds,
                                target_time=candidate["peak_time"],max_cube_cells=300_000)
        from ophanim.dynamics.regularize import utc64
        peak = utc64(candidate["peak_time"])
        display_times = source.time.values[(source.time.values <= peak) & (source.time.values >= peak-np.timedelta64(24,"h"))]
        science = analyze_run(source, config, self.root / "science",
                              frame_times=_analysis_frame_times(display_times, representative=len(display_times)-1),
                              cancelled=lambda: self._cancel.is_set() or self._stop.is_set(),
                              request={"candidate_id":candidate["candidate_id"],"snapshot_id":candidate["snapshot_id"],
                                       "window_policy":"cadence-compatible periodicity band; native spatial, coverage and effective-cycle gates unchanged"})
        self._check()
        self._visual_project(science,request,progress,{"kind":"observations" if candidate.get("source_kind") != "synthetic" else "synthetic_demo",
                              "title":"Artistic view · "+candidate["title"], "candidate_id":candidate["candidate_id"]})

    def _visual_settings(self, science, request):
        from dataclasses import replace
        import xarray as xr
        import numpy as np
        from ophanim.shawtynet import VisualStyleConfig, CameraConfig
        style = VisualStyleConfig(seed=42,render_spacing_km=20,lic_streamline_steps=15)
        if request["style"] == "luminous":
            style = replace(style, name="luminous",emission_gain=4,base_opacity=.55)
        elif request["style"] == "quiet":
            style = replace(style,name="quiet",emission_gain=1,base_opacity=.2,flow_texture_strength=.25)
        camera = None
        if request.get("camera"):
            with xr.open_zarr(science / "dynamic.zarr", consolidated=True) as data:
                lat = float(np.mean(data.lat.values))
                radians = np.radians(data.lon.values)
                lon = float(np.degrees(np.arctan2(np.mean(np.sin(radians)),np.mean(np.cos(radians)))))
            camera = CameraConfig(**{"observer_latitude":lat,"observer_longitude":lon,**request["camera"]})
        return style, camera

    def _visual_project(self, science, request, progress, metadata):
        from ophanim.shawtynet.runs import visualize_run
        style, camera = self._visual_settings(science, request)
        progress("visualizing", "ShawtyNet is mapping a single selected frame into artistic fields…", .7)
        frame = metadata.get("representative_frame", {}).get("frame_index")
        visual = visualize_run(science,style,camera=camera,frame_index=frame)
        event = json.loads((science / "event.json").read_text())
        timeline = json.loads((science / "events.json").read_text()) if (science / "events.json").exists() else {}
        selected_time = metadata.get("representative_frame", {}).get("time")
        if selected_time:
            from ophanim.dynamics.regularize import utc64
            event = next((item["event"] for item in timeline.get("frames", [])
                          if utc64(item["time"]) == utc64(selected_time)), event)
        flow = event.get("channel_status",{}).get("flow",{})
        summary = {"time":event.get("time"), "flow_status":flow.get("interpretation_status", "unavailable"),
                   "flow_fit_status":flow.get("status", "unavailable"),
                   "eligible_vector_fraction":flow.get("interpretation_eligible_fraction", 0),
                   "wave_status":event.get("wave",{}).get("status", "unavailable"),
                   "event_confidence":event.get("event_confidence"), "timeline_frames":len(timeline.get("frames", [])),
                   "source_uncertainty_status":"unknown" if event.get("metrics",{}).get("measurement_reliability") is None else "provided_rms_proxy",
                   "confidence_semantics":"uncalibrated evidence scores, not probabilities"}
        self._check()
        self._save("project",{**metadata,"science_directory":str(science),"visual_directory":str(visual),
                             "style":request["style"],"camera":request.get("camera"),"analysis_summary":summary,"active_view":"preview"})

    def _public_project(self, project):
        identity = project["project_id"]
        prefix = f"/workspace-assets/{identity}"
        result = {key:value for key,value in project.items() if not key.endswith("_directory")}
        result.update(preview_url=prefix+"/preview.png",
                      visual_report_url=prefix+"/visual/report.html",export_url=prefix+"/package.zip",
                      recipe_url=prefix+"/recipe.json")
        if project.get("science_directory"):
            result["science_report_url"] = prefix+"/science/report.html"
        if project.get("render_directory"):
            result["render_url"] = prefix+"/render.png"
        if project.get("composite_directory"):
            result["composite_url"] = prefix+"/composite.png"
        if project.get("animation_directory"):
            result.update(animation_url=prefix+"/animation.gif",animation_export_url=prefix+"/animation.zip")
        return result

    def asset(self, relative):
        from ophanim.core.artifacts import file_sha256, verify_run
        import io
        match = _ASSET.fullmatch(unquote(relative))
        if match is None:
            raise ValueError("Unknown workspace artifact")
        project = self.record(match["id"],"project")
        asset = match["file"]
        if asset == "recipe.json":
            if project.get("scenario_id"):
                recipe = self.record(project["scenario_id"], "scenario")
                recipe.pop("scenario_id", None)
                return _json(recipe).encode(),"application/json"
            return _json(self._public_project(project)).encode(),"application/json"
        if asset.startswith("science/"):
            if not project.get("science_directory"):
                raise ValueError("This imagined project has no scientific reconstruction report")
            root, kind, name = Path(project["science_directory"]),"science","diagnostics/"+asset.split("/",1)[1]
        elif asset.startswith("visual/"):
            root, kind, name = Path(project["visual_directory"]),"visual","visual_diagnostics/"+asset.split("/",1)[1]
        elif asset == "render.png":
            if not project.get("render_directory"):
                raise ValueError("Render is not ready")
            root, kind, name = Path(project["render_directory"]),"render","render.png"
        elif asset == "composite.png":
            if not project.get("composite_directory"):
                raise ValueError("Photo composite is not ready")
            root, kind, name = Path(project["composite_directory"]), "composite", "composite.png"
        elif asset in ("animation.gif", "animation.zip"):
            if not project.get("animation_directory"):
                raise ValueError("Animation is not ready")
            root, kind, name = Path(project["animation_directory"]), "animation", "animation.gif"
        else:
            root, kind, name = Path(project["visual_directory"]),"visual","render_package/textures/color.png"
        if not root.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Artifact path is outside the workspace")
        manifest = verify_run(root,kind=kind)
        if asset == "preview.png" and "render_package/preview.png" in manifest["files"]:
            name = "render_package/preview.png"
        if asset in ("package.zip", "animation.zip"):
            names = [key for key in manifest["files"] if key.startswith("render_package/")] if asset == "package.zip" else list(manifest["files"])
            if sum((root/key).stat().st_size for key in names) > 64*1024*1024:
                raise ValueError("Package is too large for browser export")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
                for key in names:
                    # Keep the strict 3D package inventory separate from the
                    # human-facing project/recipe metadata in the outer bundle.
                    packed = key if project.get("kind") == "imagined_3d" else key.removeprefix("render_package/")
                    archive.write(root/key,packed)
                archive.writestr("project.json",_json(self._public_project(project)))
                if project.get("kind") == "imagined_3d" and asset == "package.zip":
                    archive.writestr("README.txt", "OPHANIM imagined 3D scene (not measured plasma).\n"
                        "Keep render_package intact: its validated inventory excludes this outer metadata.\n"
                        "From the extracted outer folder, with a trusted Blender installation:\n"
                        "blender --background --factory-startup --threads 2 --python render_package/volume_blender_scene.py "
                        "-- --package render_package --output render.png --samples 24 --width 960 --height 640\n"
                        "Only run renderer scripts from an OPHANIM installation you trust, never arbitrary downloaded code.\n")
                if project.get("scenario_id"):
                    recipe = self.record(project["scenario_id"], "scenario")
                    recipe.pop("scenario_id", None)
                    archive.writestr("scenario-recipe.json",_json(recipe))
            return buffer.getvalue(),"application/zip"
        expected = manifest["files"].get(name)
        path = root/name
        if not expected or path.stat().st_size > 32*1024*1024 or file_sha256(path) != expected:
            raise ValueError("Artifact is missing or has changed")
        return path.read_bytes(),{".html":"text/html; charset=utf-8", ".gif":"image/gif"}.get(path.suffix,"image/png")
