"""Read desktop acquisitions using SQLite metadata and verified original bytes."""

from contextlib import closing
from pathlib import Path
import sqlite3

from ophanim.artifacts import FilesystemArtifactStore
from ophanim.domain import SourceArtifact

from .ionex import read_ionex_sequence
from .sequence import SensingError, _finish_dataset, _inside, _utc, _window


def read_desktop_sequence(
    data_directory, artifact_ids=None, *, start=None, end=None, as_of=None, bounds=None,
):
    """Read existing desktop acquisitions without initializing or writing a DB.

    Reads ``ophanim.sqlite3`` in SQLite read-only mode and content-addressed
    ``artifacts/`` bytes. Historical revisions unavailable at ``as_of`` are
    excluded before deduplication. Missing RMS stays NaN.
    """
    import numpy as np
    import xarray as xr

    start, end, as_of = _window(start, end, as_of)
    root = Path(data_directory).expanduser().resolve()
    database = root / "ophanim.sqlite3"
    artifact_root = root / "artifacts"
    if not database.is_file() or not artifact_root.is_dir():
        raise SensingError("desktop data requires ophanim.sqlite3 and artifacts/")
    if isinstance(artifact_ids, (str, bytes)):
        raise SensingError("artifact_ids must be a sequence of IDs")
    requested = None if artifact_ids is None else set(artifact_ids)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        available_ids = {row[0] for row in connection.execute("SELECT artifact_id FROM source_artifacts")}
        if requested is not None and requested - available_ids:
            raise SensingError("requested source artifact is absent from desktop metadata")
        # Use the existing (artifact_id, observed_at) index before decompressing
        # any source. Minimal legacy metadata-only fixtures still work.
        indexed = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tec_observations'").fetchone()
        terms, arguments = [], []
        if requested is not None:
            if not requested:
                raise SensingError("no native observations are available in the requested window")
            terms.append("s.artifact_id IN (" + ",".join("?" for _ in requested) + ")")
            arguments.extend(sorted(requested))
        if indexed and (start is not None or end is not None or as_of is not None):
            epoch_terms = ["o.artifact_id=s.artifact_id"]
            if start is not None:
                epoch_terms.append("o.observed_at>=?")
                arguments.append(start.isoformat(timespec="microseconds"))
            stop = min(v for v in (end, as_of) if v is not None) if end or as_of else None
            if stop is not None:
                epoch_terms.append("o.observed_at<=?")
                arguments.append(stop.isoformat(timespec="microseconds"))
            terms.append("EXISTS (SELECT 1 FROM tec_observations o WHERE " + " AND ".join(epoch_terms) + ")")
        query = "SELECT s.* FROM source_artifacts s" + (" WHERE " + " AND ".join(terms) if terms else "") + " ORDER BY s.artifact_id"
        rows = connection.execute(query, arguments).fetchall()
    store = FilesystemArtifactStore(artifact_root)
    datasets = []
    ranks = []
    provenance = []
    for row in rows:
        if requested is not None and row["artifact_id"] not in requested:
            continue
        available_at = _utc(row["ingested_at"], "source ingested_at")
        if as_of is not None and available_at > as_of:
            continue
        artifact = SourceArtifact(
            artifact_id=row["artifact_id"], provider=row["provider"], product=row["product"],
            parser_version=row["parser_version"], revision_priority=row["revision_priority"],
            checksum_sha256=row["checksum_sha256"], storage_ref=row["storage_ref"],
            ingested_at=available_at, revision=row["revision"], source_uri=row["source_uri"],
        )
        dataset = read_ionex_sequence(store.read(artifact.storage_ref), artifact=artifact)
        keep = [index for index, time in enumerate(dataset.time.values)
                if _inside(_utc(np.datetime_as_string(time, unit="us") + "Z", "source epoch"), start, end, as_of)]
        if not keep:
            continue
        dataset = dataset.isel(time=keep)
        if datasets and (not np.array_equal(dataset.lat, datasets[0].lat)
                         or not np.array_equal(dataset.lon, datasets[0].lon)):
            raise SensingError("native source axes differ; explicit regridding is required")
        datasets.append(dataset)
        ranks.extend([(artifact.revision_priority, available_at, artifact.artifact_id)] * len(keep))
        provenance.extend(dataset.attrs["source_metadata"]["revision_history"])
    if not datasets:
        raise SensingError("no native observations are available in the requested window")
    combined = xr.concat(datasets, dim="time", join="exact", combine_attrs="drop")
    chosen = {}
    for index, time in enumerate(combined.time.values):
        if time not in chosen or ranks[index] > ranks[chosen[time]]:
            chosen[time] = index
    combined = combined.isel(time=[chosen[time] for time in sorted(chosen)])
    return _finish_dataset(combined, sorted(provenance, key=lambda item: item["source_id"]),
                           {}, start, end, as_of, bounds)
