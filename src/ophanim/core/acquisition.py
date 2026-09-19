"""Model-independent bounded GIM acquisition and immutable native cache.

SQLite indexes whole sources and epochs, not millions of repeated cell rows.
Original bytes and normalized arrays are content addressed and verified on read.
No forecast, artistic, desktop-controller, or experiment modules are imported.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sqlite3

from ophanim.artifacts import FilesystemArtifactStore
from ophanim.domain import SourceArtifact
from ophanim.gim import CodeGIMSource, GIMAcquisitionError, GIMFetchRequest, GIMNotFound
from ophanim.ionex import IONEXV1Parser
from ophanim.sensing import SensingError, read_desktop_sequence, read_ionex_sequence
from ophanim.sensing.sequence import _finish_dataset, _utc


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
 artifact_id TEXT PRIMARY KEY, product_date TEXT NOT NULL,
 first_epoch TEXT NOT NULL, last_epoch TEXT NOT NULL,
 artifact_json TEXT NOT NULL, array_ref TEXT NOT NULL,
 array_checksum TEXT NOT NULL, dataset_attrs TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sources_dates ON sources(product_date);
CREATE INDEX IF NOT EXISTS sources_range ON sources(first_epoch, last_epoch);
CREATE TABLE IF NOT EXISTS epochs (
 observed_at TEXT NOT NULL, artifact_id TEXT NOT NULL REFERENCES sources(artifact_id),
 PRIMARY KEY (observed_at, artifact_id)
);
CREATE TABLE IF NOT EXISTS product_checks (
 product_date TEXT PRIMARY KEY, last_checked TEXT NOT NULL
);
"""

NATIVE_CACHE_VERSION = "ophanim-native-cache/1"


def _days(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 31:
        raise ValueError("days must be an integer from 1 through 31")
    return value


def _cancelled(callback):
    if callback is not None and callback():
        raise InterruptedError("Data acquisition cancelled; completed source downloads remain cached")


def _notify(callback, message):
    if callback is not None:
        callback(message)


def _database(data_directory, *, create=False):
    root = Path(data_directory).expanduser().resolve() / "core"
    database = root / "catalog.sqlite3"
    if create:
        root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(_SCHEMA)
    elif database.is_file():
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    else:
        return None
    connection.row_factory = sqlite3.Row
    return connection


def _artifact_payload(artifact):
    payload = asdict(artifact)
    payload["ingested_at"] = artifact.ingested_at.isoformat()
    return payload


def _artifact_from_payload(payload):
    payload = dict(payload)
    payload["ingested_at"] = _utc(payload["ingested_at"], "source availability")
    return SourceArtifact(**payload)


def _persist(data_directory, raw, artifact, dataset, product_date):
    import numpy as np

    root = Path(data_directory).expanduser().resolve() / "core"
    original = FilesystemArtifactStore(root / "artifacts").put(raw, suffix=".ionex")
    if original.checksum_sha256 != artifact.checksum_sha256:
        raise SensingError("source checksum differs before cache commit")
    with closing(_database(data_directory, create=True)) as connection:
        existing = connection.execute("SELECT artifact_json FROM sources WHERE artifact_id=?", (artifact.artifact_id,)).fetchone()
    if existing is not None:
        previous = json.loads(existing[0])
        immutable_fields = ("checksum_sha256", "parser_version", "provider", "product", "revision", "revision_priority")
        if any(previous[name] != getattr(artifact, name) for name in immutable_fields):
            raise SensingError("immutable native artifact identity was reused for different source metadata")
        # Successful unchanged refreshes update product_checks only: first
        # availability and normalized arrays remain immutable, with no orphans.
        return
    payload = _artifact_payload(artifact)
    payload["storage_ref"] = original.storage_ref
    buffer = BytesIO()
    arrays = {name: dataset[name].values for name in dataset.data_vars}
    arrays.update({name: dataset[name].values for name in ("time", "lat", "lon")})
    np.savez_compressed(buffer, **arrays)
    stored = FilesystemArtifactStore(root / "arrays").put(buffer.getvalue(), suffix=".npz")
    times = [np.datetime_as_string(v, unit="us") + "Z" for v in dataset.time.values]
    attributes = dict(dataset.attrs, native_cache_version=NATIVE_CACHE_VERSION)
    with closing(_database(data_directory, create=True)) as connection, connection:
        connection.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,?,?,?)", (
            artifact.artifact_id, str(product_date), times[0], times[-1],
            json.dumps(payload, sort_keys=True, allow_nan=False), stored.storage_ref,
            stored.checksum_sha256, json.dumps(attributes, sort_keys=True, allow_nan=False),
        ))
        connection.executemany("INSERT OR IGNORE INTO epochs VALUES (?,?)", [(v, artifact.artifact_id) for v in times])


def _read_cached(data_directory, start, end):
    import numpy as np
    import xarray as xr

    connection = _database(data_directory)
    if connection is None:
        return []
    with closing(connection):
        rows = connection.execute("SELECT * FROM sources WHERE last_epoch>=? AND first_epoch<=? ORDER BY artifact_id", (
            start.isoformat(timespec="microseconds").replace("+00:00", "Z"), end.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        )).fetchall()
    store = FilesystemArtifactStore(Path(data_directory) / "core" / "arrays")
    result = []
    for row in rows:
        raw = store.read(row["array_ref"])
        if sha256(raw).hexdigest() != row["array_checksum"]:
            raise SensingError("native cache checksum differs from catalog")
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            variables = {name: (("time",) if archive[name].ndim == 1 else ("time", "lat", "lon"), archive[name])
                         for name in archive.files if name not in {"time", "lat", "lon"}}
            dataset = xr.Dataset(variables, coords={name: archive[name] for name in ("time", "lat", "lon")},
                                 attrs=json.loads(row["dataset_attrs"]))
        dataset = dataset.sel(time=slice(start.replace(tzinfo=None), end.replace(tzinfo=None)))
        if dataset.sizes["time"]:
            result.append(dataset)
    return result


def _legacy_latest(data_directory):
    database = Path(data_directory).expanduser().resolve() / "ophanim.sqlite3"
    if not database.is_file():
        return None
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='tec_observations'").fetchone():
            return None
        value = connection.execute("SELECT MAX((SELECT observed_at FROM tec_observations o WHERE o.artifact_id=s.artifact_id ORDER BY observed_at DESC LIMIT 1)) FROM source_artifacts s").fetchone()[0]
    return _utc(value, "cached epoch") if value else None


def _merge(datasets, start, end):
    import numpy as np
    import xarray as xr

    if not datasets:
        raise SensingError("no native observations are available in the requested window")
    history = {}
    for dataset in datasets:
        if not np.array_equal(dataset.lat, datasets[0].lat) or not np.array_equal(dataset.lon, datasets[0].lon):
            raise SensingError("native source axes differ; explicit regridding is required")
        for item in dataset.attrs["source_metadata"]["revision_history"]:
            history[item["source_id"]] = item
    combined = xr.concat(datasets, dim="time", join="exact", combine_attrs="drop")
    selected = {}
    ranks = []
    for index, time in enumerate(combined.time.values):
        identity = str(combined.source_id.values[index])
        item = history[identity]
        rank = (item.get("revision_priority", 0), _utc(item["available_at"], "availability"), identity)
        ranks.append(rank)
        if time not in selected or rank > ranks[selected[time]]:
            selected[time] = index
    combined = combined.isel(time=[selected[t] for t in sorted(selected)])
    return _finish_dataset(combined, [history[key] for key in sorted(history)], {}, start, end, None, None)


def _window_data(data_directory, start, end):
    datasets = _read_cached(data_directory, start, end)
    database = Path(data_directory) / "ophanim.sqlite3"
    if database.is_file():
        try:
            datasets.append(read_desktop_sequence(data_directory, start=start, end=end))
        except SensingError as error:
            if "no native observations" not in str(error):
                raise
    return _merge(datasets, start, end)


def read_recent(data_directory, *, days=8):
    """Read bounded latest cached history without network or model initialization.

    A stale cache remains inspectable; its actual last epoch is never relabeled
    as current. The consumer must display age. No sources are modified.
    """
    _days(days)
    latest = _legacy_latest(data_directory)
    connection = _database(data_directory)
    if connection is not None:
        with closing(connection):
            value = connection.execute("SELECT MAX(observed_at) FROM epochs").fetchone()[0]
        if value:
            cached = _utc(value, "cached epoch")
            latest = cached if latest is None else max(latest, cached)
    if latest is None:
        raise SensingError("no cached native observations; pull ionosphere data first")
    return _window_data(data_directory, latest - timedelta(days=days), latest)


def acquire_recent(data_directory, *, days=8, progress=None, cancelled=None, source=None):
    """Download missing recent daily CODE sources, independent of trained models.

    Exact completed UTC days are bounded to 1..31 (default 8). A successful
    source is fully parsed and atomically cached before cancellation is checked
    again. Missing/unavailable days remain visible warnings and are retryable.
    Existing source revisions are retained. Only the two most recent completed
    product days are rechecked, at most once per 24 hours after success. Check
    time is separate from first local source availability; identical bytes do
    not acquire a fabricated later availability or continually trigger retries.
    """
    import numpy as np

    _days(days)
    _cancelled(cancelled)
    now = datetime.now(UTC)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    with closing(_database(data_directory, create=True)) as connection:
        complete_days = {row[0] for row in connection.execute("SELECT DISTINCT product_date FROM sources")}
        checked = {row[0]: _utc(row[1], "last product check") for row in connection.execute("SELECT product_date,last_checked FROM product_checks")}
    # Existing laptop bytes are reusable without their old model being present.
    legacy_database = Path(data_directory) / "ophanim.sqlite3"
    if legacy_database.is_file():
        _notify(progress, "Checking existing native source history…")
        try:
            legacy = read_desktop_sequence(data_directory, start=start, end=end)
        except SensingError as error:
            if "no native observations" not in str(error):
                raise
        else:
            # Count a day only if multiple native epochs exist, not the next-day
            # midnight endpoint commonly included in the previous IONEX file.
            stamps = legacy.time.values.astype("datetime64[D]")
            for day in np.unique(stamps):
                if np.count_nonzero(stamps == day) > 1:
                    complete_days.add(str(day))
    provider = source or CodeGIMSource()
    warnings = []
    for offset in range(days):
        _cancelled(cancelled)
        day = (start + timedelta(days=offset)).date()
        recent = day >= (end - timedelta(days=2)).date()
        refresh = recent and (str(day) not in checked or now - checked[str(day)] >= timedelta(hours=24))
        if str(day) in complete_days and not refresh:
            _notify(progress, f"Using cached CODE source for {day}")
            continue
        _notify(progress, f"Pulling CODE native ionosphere data for {day} ({offset + 1}/{days})…")
        try:
            fetched = provider.fetch(GIMFetchRequest(product_date=day))
        except GIMNotFound:
            warnings.append(f"No completed CODE product was available for {day}.")
            continue
        except GIMAcquisitionError as error:
            warnings.append(f"CODE download for {day} failed: {error}")
            continue
        raw = fetched.loaded_source.content
        request = fetched.ingestion_request
        digest = sha256(raw).hexdigest()
        identity = sha256(json.dumps([NATIVE_CACHE_VERSION, IONEXV1Parser.parser_version, "ophanim-sensing-rms/1",
                                     request.provider, request.product, request.revision,
                                     request.revision_priority, digest], separators=(",", ":")).encode()).hexdigest()
        artifact = SourceArtifact(
            artifact_id=identity, provider=request.provider, product=request.product,
            parser_version=IONEXV1Parser.parser_version, revision_priority=request.revision_priority,
            checksum_sha256=digest, storage_ref="pending", ingested_at=datetime.now(UTC),
            revision=request.revision, source_uri=fetched.resolved_uri,
        )
        dataset = read_ionex_sequence(raw, artifact=artifact)
        _persist(data_directory, raw, artifact, dataset, day)
        with closing(_database(data_directory, create=True)) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO product_checks VALUES (?,?)", (str(day), datetime.now(UTC).isoformat()))
    _cancelled(cancelled)
    try:
        result = _window_data(data_directory, start, end)
    except SensingError as error:
        if warnings and "no native observations" in str(error):
            raise SensingError(str(error) + ". " + " ".join(warnings[-3:])) from error
        raise
    result.attrs["acquisition_warnings"] = warnings
    result.attrs["requested_window"] = {"start": start.isoformat(), "end": end.isoformat(), "days": days}
    return result
