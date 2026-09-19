"""Additive source-RMS access without changing immutable IONEX parse identities."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path

from ophanim.domain import SourceArtifact
from ophanim.ingestion import decompress_ionex
from ophanim import ionex as codec

from .sequence import SensingError, _datetime_array, _finish_dataset, _inside, _utc, _window


def _source_bytes(source):
    if isinstance(source, Path):
        return source.read_bytes()
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        if "\n" not in source and "\r" not in source:
            return Path(source).read_bytes()
        return source.encode("ascii")
    raise TypeError("source must contain bytes, text, or a local path")


def _record(payload, label):
    return f"{payload:<60}{label:<20}\n"


def _parse_source(source, artifact):
    raw = _source_bytes(source)
    if sha256(raw).hexdigest() != artifact.checksum_sha256:
        raise SensingError("original source checksum differs from artifact provenance")
    content = decompress_ionex(raw)
    parser = codec.IONEXV1Parser()
    observations = list(parser.parse(artifact=artifact, stream=BytesIO(content)))
    # Reuse the same validated fixed-width header/coordinate helpers as the
    # pinned parser, rather than implementing a subtly different IONEX grid.
    lines = list(codec._decode_lines(BytesIO(content)))
    grid = parser._parse_header(iter(lines))
    header_end = next(i for i, line in enumerate(lines) if codec._split_record(line)[1] == "ENDOFHEADER")
    header = [line.text + "\n" for line in lines[:header_end + 1]]
    tec_epochs = {}
    rms_blocks = []
    exponent = grid.exponent
    active_tec = None
    active_rms = None
    rms_number = None
    rms_start_exponent = exponent
    eof = False
    for line in lines[header_end + 1:]:
        payload, label = codec._split_record(line)
        if eof:
            if label in {"STARTOFRMSMAP", "ENDOFRMSMAP"}:
                raise codec.IONEXParseError("RMS map appears after END OF FILE")
            continue
        if label == "EXPONENT":
            exponent = codec._parse_integer_field(payload, "exponent", line)
        if label == "STARTOFTECMAP":
            if active_rms is not None:
                raise codec.IONEXParseError("TEC map nested in RMS map")
            active_tec = codec._parse_integer_field(payload, "TEC map number", line)
        elif label == "EPOCHOFCURRENTMAP" and active_tec is not None:
            tec_epochs[active_tec] = codec._parse_epoch(payload, line)
        elif label == "ENDOFTECMAP":
            active_tec = None
        elif label == "STARTOFRMSMAP":
            if active_rms is not None or active_tec is not None:
                raise codec.IONEXParseError("nested RMS map")
            rms_number = codec._parse_integer_field(payload, "RMS map number", line)
            rms_start_exponent = exponent
            active_rms = [_record("1", "START OF TEC MAP")]
            continue
        elif label == "ENDOFRMSMAP":
            if active_rms is None:
                raise codec.IONEXParseError("END OF RMS MAP without a start")
            if codec._parse_integer_field(payload, "RMS map number", line) != rms_number:
                raise codec.IONEXParseError("RMS start/end map numbers differ")
            active_rms.append(_record("1", "END OF TEC MAP"))
            rms_blocks.append((rms_number, rms_start_exponent, active_rms))
            active_rms = None
            continue
        elif label == "ENDOFFILE":
            if active_rms is not None:
                raise codec.IONEXParseError("unterminated RMS map")
            eof = True
        if active_rms is not None:
            active_rms.append(line.text + "\n")
    if active_rms is not None:
        raise codec.IONEXParseError("unterminated RMS map")
    rms = {}
    seen = set()
    for number, start_exponent, block in rms_blocks:
        if number in seen or number not in tec_epochs:
            raise codec.IONEXParseError("RMS map number must uniquely match a TEC map")
        seen.add(number)
        rms_header = []
        for entry in header:
            label = codec._split_record(codec._Line(1, entry.rstrip("\n")))[1]
            if label == "#OFMAPSINFILE":
                rms_header.append(_record("1", "# OF MAPS IN FILE"))
            elif label == "ENDOFHEADER":
                rms_header.append(_record(str(start_exponent), "EXPONENT"))
                rms_header.append(entry)
            else:
                rms_header.append(entry)
        rms_text = "".join(rms_header + block + [_record("", "END OF FILE")])
        rms_values = list(parser.parse(artifact=artifact, stream=StringIO(rms_text)))
        # Even an all-9999 RMS map must have the matching epoch.
        block_epoch_lines = [entry for entry in block
                             if codec._split_record(codec._Line(1, entry.rstrip("\n")))[1] == "EPOCHOFCURRENTMAP"]
        if len(block_epoch_lines) != 1:
            raise codec.IONEXParseError("RMS map requires one epoch")
        epoch_line = codec._Line(1, block_epoch_lines[0].rstrip("\n"))
        epoch = codec._parse_epoch(codec._split_record(epoch_line)[0], epoch_line)
        if epoch != tec_epochs[number]:
            raise codec.IONEXParseError("RMS epoch does not match its TEC map")
        for value in rms_values:
            if value.vtec_tecu < 0:
                raise codec.IONEXParseError("source RMS must be non-negative")
            rms[(value.observed_at, value.latitude_degrees, value.longitude_degrees)] = value.vtec_tecu
    return grid, observations, rms, sorted(tec_epochs.values())


def parse_ionex_rms(source, *, artifact: SourceArtifact):
    """Return source RMS TECU keyed by (UTC epoch, latitude, longitude).

    Original bytes (possibly gzip/.Z) must match the artifact checksum. TEC and
    RMS grids are validated by the existing IONEX parser. Missing ``9999`` RMS
    cells and absent RMS maps are omitted, never replaced by zero/confidence.
    """
    return _parse_source(source, artifact)[2]


def read_ionex_sequence(
    source, *, artifact: SourceArtifact, start=None, end=None, as_of=None, bounds=None,
):
    """Read original source bytes into the shared native observation contract."""
    import numpy as np
    import xarray as xr

    start, end, as_of = _window(start, end, as_of)
    available = _utc(artifact.ingested_at, "artifact ingested_at")
    if as_of is not None and available > as_of:
        raise SensingError("source artifact was unavailable at as_of")
    grid, observations, rms, all_times = _parse_source(source, artifact)
    times = [t for t in all_times if _inside(t, start, end, as_of)]
    if not times:
        raise SensingError("no native observations are available in the requested window")
    latitudes = sorted(codec._coordinate(grid.latitude[0], grid.latitude[2], index)
                       for index in range(codec._grid_count(*grid.latitude, name="latitude grid")))
    longitudes = sorted({codec._normalise_longitude(codec._coordinate(grid.longitude[0], grid.longitude[2], index))
                         for index in range(codec._grid_count(*grid.longitude, name="longitude grid"))})
    shape = (len(times), len(latitudes), len(longitudes))
    tec = np.full(shape, np.nan, dtype=np.float64)
    rms_values = np.full(shape, np.nan, dtype=np.float64)
    time_index = {value: index for index, value in enumerate(times)}
    lat_index = {value: index for index, value in enumerate(latitudes)}
    lon_index = {value: index for index, value in enumerate(longitudes)}
    for value in observations:
        if value.observed_at not in time_index:
            continue
        idx = (time_index[value.observed_at], lat_index[value.latitude_degrees], lon_index[value.longitude_degrees])
        tec[idx] = value.vtec_tecu
        rms_values[idx] = rms.get((value.observed_at, value.latitude_degrees, value.longitude_degrees), np.nan)
    dataset = xr.Dataset(
        {
            "tec": (("time", "lat", "lon"), tec),
            "observed_mask": (("time", "lat", "lon"), np.isfinite(tec)),
            "quality_mask": (("time", "lat", "lon"), np.zeros(shape, dtype=np.uint32)),
            "source_rms_tecu": (("time", "lat", "lon"), rms_values),
            "available_at": ("time", _datetime_array([available] * len(times))),
            "source_id": ("time", np.asarray([artifact.artifact_id] * len(times), dtype=str)),
        },
        coords={"time": _datetime_array(times), "lat": latitudes, "lon": longitudes},
    )
    metadata = {
        "source_id": artifact.artifact_id, "artifact_id": artifact.artifact_id,
        "provider": artifact.provider, "product": artifact.product,
        "revision": artifact.revision, "revision_priority": artifact.revision_priority,
        "available_at": available.isoformat(), "source_checksum_sha256": artifact.checksum_sha256,
        "parser_version": artifact.parser_version, "rms_parser_version": "ophanim-sensing-rms/1",
        "value_kind": "native_source_product", "source_uri": artifact.source_uri,
    }
    return _finish_dataset(dataset, [metadata], {}, start, end, as_of, bounds)
