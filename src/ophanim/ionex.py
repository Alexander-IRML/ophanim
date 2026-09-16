"""Dependency-free parsing for IONEX v1 TEC maps.

The parser consumes an already-open, decompressed ASCII stream. Acquisition,
compression handling, and persistence belong to the ingestion layer. Only
two-dimensional TEC maps can be represented by :class:`TECObservation`; RMS,
height, and auxiliary blocks are deliberately ignored.

IONEX records use a 60-character value area followed by a 20-character label.
Grid headers and data values are fixed-width, so whitespace splitting alone is
not sufficient (``87.5-180.0`` is a valid pair of adjacent fields).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import BinaryIO, Protocol, TextIO

from ophanim.domain import SourceArtifact, TECObservation


IONEXStream = BinaryIO | TextIO


class IONEXParser(Protocol):
    """Parse an IONEX artifact without owning durable state."""

    @property
    def parser_version(self) -> str:
        """Return the immutable revision identity for this parse behavior."""

        ...

    def parse(
        self,
        *,
        artifact: SourceArtifact,
        stream: IONEXStream,
    ) -> Iterable[TECObservation]: ...


class IONEXError(ValueError):
    """Base class for invalid or unsupported IONEX input."""


class IONEXParseError(IONEXError):
    """The source is malformed or internally inconsistent."""

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        self.line_number = line_number
        location = f" at line {line_number}" if line_number is not None else ""
        super().__init__(f"{message}{location}")


class UnsupportedIONEXError(IONEXError):
    """The source is valid IONEX but cannot be represented by this parser."""


@dataclass(frozen=True, slots=True)
class _Line:
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class _Grid:
    map_count: int
    dimension: int
    height: tuple[float, float, float]
    latitude: tuple[float, float, float]
    longitude: tuple[float, float, float]
    exponent: int


@dataclass(frozen=True, slots=True)
class _Row:
    latitude: float
    longitude_start: float
    longitude_end: float
    longitude_step: float
    height: float


_INTEGER = re.compile(r"[+-]?\d+")

_VERSION_LABEL = "IONEXVERSION/TYPE"
_MAP_COUNT_LABEL = "#OFMAPSINFILE"
_MAP_DIMENSION_LABEL = "MAPDIMENSION"
_HEIGHT_GRID_LABEL = "HGT1/HGT2/DHGT"
_LATITUDE_GRID_LABEL = "LAT1/LAT2/DLAT"
_LONGITUDE_GRID_LABEL = "LON1/LON2/DLON"
_EXPONENT_LABEL = "EXPONENT"
_START_AUX_LABEL = "STARTOFAUXDATA"
_END_AUX_LABEL = "ENDOFAUXDATA"
_END_HEADER_LABEL = "ENDOFHEADER"
_START_TEC_LABEL = "STARTOFTECMAP"
_END_TEC_LABEL = "ENDOFTECMAP"
_EPOCH_LABEL = "EPOCHOFCURRENTMAP"
_ROW_LABEL = "LAT/LON1/LON2/DLON/H"
_END_FILE_LABEL = "ENDOFFILE"

_KNOWN_LABELS = {
    _VERSION_LABEL,
    _MAP_COUNT_LABEL,
    _MAP_DIMENSION_LABEL,
    _HEIGHT_GRID_LABEL,
    _LATITUDE_GRID_LABEL,
    _LONGITUDE_GRID_LABEL,
    _EXPONENT_LABEL,
    _START_AUX_LABEL,
    _END_AUX_LABEL,
    _END_HEADER_LABEL,
    _START_TEC_LABEL,
    _END_TEC_LABEL,
    _EPOCH_LABEL,
    _ROW_LABEL,
    _END_FILE_LABEL,
}


class IONEXV1Parser:
    """Parse validated, two-dimensional IONEX v1 TEC grids.

    The official raw value ``9999`` means that no TEC value is available. Such
    cells are omitted, allowing downstream coverage logic to distinguish them
    from real measurements. Longitudes are normalized into the domain's
    half-open ``[-180, 180)`` convention. Equivalent wrap endpoints
    (for example, 0 and 360 degrees) are emitted once when their values agree;
    conflicting endpoint values make the map invalid.
    """

    missing_value = 9999
    default_exponent = -1
    parser_version = "ophanim-ionex-v1-parser/2"

    def parse(
        self,
        *,
        artifact: SourceArtifact,
        stream: IONEXStream,
    ) -> Iterator[TECObservation]:
        if artifact.parser_version != self.parser_version:
            raise IONEXParseError(
                "artifact parser_version does not match the active parser: "
                f"{artifact.parser_version!r} != {self.parser_version!r}"
            )
        lines = iter(_decode_lines(stream))
        grid = self._parse_header(lines)
        yield from self._parse_maps(
            artifact_id=artifact.artifact_id,
            lines=lines,
            grid=grid,
        )

    def _parse_header(self, lines: Iterator[_Line]) -> _Grid:
        try:
            first = next(lines)
        except StopIteration as exc:
            raise IONEXParseError("empty IONEX stream") from exc

        payload, label = _split_record(first)
        if label != _VERSION_LABEL:
            raise IONEXParseError(
                "first record must be IONEX VERSION / TYPE",
                line_number=first.number,
            )

        version = _parse_float_field(payload[:8], "IONEX version", first)
        if not (1.0 <= version < 2.0):
            raise UnsupportedIONEXError(
                f"IONEX version {version:g} is unsupported; expected version 1.x"
            )
        file_type = payload[20:21].strip().upper()
        if file_type != "I":
            raise UnsupportedIONEXError(
                f"IONEX file type {file_type or '<blank>'!r} is not an ionosphere map"
            )

        map_count: int | None = None
        dimension: int | None = None
        height: tuple[float, float, float] | None = None
        latitude: tuple[float, float, float] | None = None
        longitude: tuple[float, float, float] | None = None
        exponent = self.default_exponent
        in_auxiliary_data = False

        for line in lines:
            payload, label = _split_record(line)

            if label == _START_AUX_LABEL:
                in_auxiliary_data = True
                continue
            if label == _END_AUX_LABEL:
                in_auxiliary_data = False
                continue
            if in_auxiliary_data:
                continue
            if label == _END_HEADER_LABEL:
                break
            if label == _MAP_COUNT_LABEL:
                map_count = _parse_integer_field(payload, "map count", line)
            elif label == _MAP_DIMENSION_LABEL:
                dimension = _parse_integer_field(payload, "map dimension", line)
            elif label == _HEIGHT_GRID_LABEL:
                height = _parse_fixed_floats(payload, 2, 6, 3, line)
            elif label == _LATITUDE_GRID_LABEL:
                latitude = _parse_fixed_floats(payload, 2, 6, 3, line)
            elif label == _LONGITUDE_GRID_LABEL:
                longitude = _parse_fixed_floats(payload, 2, 6, 3, line)
            elif label == _EXPONENT_LABEL:
                exponent = _parse_integer_field(payload, "exponent", line)
        else:
            raise IONEXParseError("missing END OF HEADER record")

        missing: list[str] = []
        if map_count is None:
            missing.append("# OF MAPS IN FILE")
        if dimension is None:
            missing.append("MAP DIMENSION")
        if height is None:
            missing.append("HGT1 / HGT2 / DHGT")
        if latitude is None:
            missing.append("LAT1 / LAT2 / DLAT")
        if longitude is None:
            missing.append("LON1 / LON2 / DLON")
        if missing:
            raise IONEXParseError(
                "missing required header record(s): " + ", ".join(missing)
            )

        assert map_count is not None
        assert dimension is not None
        assert height is not None
        assert latitude is not None
        assert longitude is not None

        if map_count <= 0:
            raise IONEXParseError("# OF MAPS IN FILE must be positive")
        if dimension != 2:
            raise UnsupportedIONEXError(
                "only two-dimensional IONEX TEC maps are supported; "
                f"MAP DIMENSION is {dimension}"
            )
        if not (_close(height[0], height[1]) and _close(height[2], 0.0)):
            raise IONEXParseError(
                "two-dimensional maps require one fixed height and DHGT = 0"
            )

        _grid_count(*latitude, name="latitude grid")
        _grid_count(*longitude, name="longitude grid")
        if not (-90.0 <= latitude[0] <= 90.0 and -90.0 <= latitude[1] <= 90.0):
            raise IONEXParseError("latitude grid lies outside -90..90 degrees")

        return _Grid(
            map_count=map_count,
            dimension=dimension,
            height=height,
            latitude=latitude,
            longitude=longitude,
            exponent=exponent,
        )

    def _parse_maps(
        self,
        *,
        artifact_id: str,
        lines: Iterator[_Line],
        grid: _Grid,
    ) -> Iterator[TECObservation]:
        active_map_number: int | None = None
        active_epoch: datetime | None = None
        active_observations: list[TECObservation] = []
        active_latitudes: list[float] = []
        exponent = grid.exponent
        previous_epoch: datetime | None = None
        parsed_map_count = 0
        found_end_of_file = False

        for line in lines:
            payload, label = _split_record(line)

            if active_map_number is None:
                if label == _START_TEC_LABEL:
                    active_map_number = _parse_integer_field(
                        payload, "TEC map number", line
                    )
                    expected_map_number = parsed_map_count + 1
                    if active_map_number != expected_map_number:
                        raise IONEXParseError(
                            f"unexpected TEC map number {active_map_number}; "
                            f"expected {expected_map_number}",
                            line_number=line.number,
                        )
                    active_epoch = None
                    active_observations = []
                    active_latitudes = []
                elif label == _EXPONENT_LABEL:
                    exponent = _parse_integer_field(payload, "exponent", line)
                elif label == _END_FILE_LABEL:
                    found_end_of_file = True
                    break
                continue

            if label == _EPOCH_LABEL:
                if active_epoch is not None or active_latitudes:
                    raise IONEXParseError(
                        "EPOCH OF CURRENT MAP must occur once before TEC rows",
                        line_number=line.number,
                    )
                active_epoch = _parse_epoch(payload, line)
                if previous_epoch is not None and active_epoch <= previous_epoch:
                    raise IONEXParseError(
                        "TEC map epochs must be strictly chronological",
                        line_number=line.number,
                    )
                continue

            if label == _EXPONENT_LABEL:
                exponent = _parse_integer_field(payload, "exponent", line)
                continue

            if label == _ROW_LABEL:
                if active_epoch is None:
                    raise IONEXParseError(
                        "TEC row appears before EPOCH OF CURRENT MAP",
                        line_number=line.number,
                    )
                row = _parse_row(payload, line)
                self._validate_row(row, grid, active_latitudes, line)
                raw_values = _read_row_values(lines, row, line)
                active_latitudes.append(row.latitude)
                scale = _scale_for_exponent(exponent, line)
                row_observations: dict[
                    float, tuple[float, TECObservation] | None
                ] = {}

                for index, raw_value in enumerate(raw_values):
                    source_longitude = _coordinate(
                        row.longitude_start,
                        row.longitude_step,
                        index,
                    )
                    longitude = _normalise_longitude(source_longitude)
                    if raw_value == self.missing_value:
                        # Keep the logical cell's insertion position. A value at
                        # the equivalent wrap endpoint may fill it later.
                        row_observations.setdefault(longitude, None)
                        continue
                    value = raw_value * scale
                    if exponent < 0:
                        value = round(value, -exponent)
                    if not math.isfinite(value):
                        raise IONEXParseError(
                            "scaled TEC value is not finite",
                            line_number=line.number,
                        )
                    observation = TECObservation(
                        artifact_id=artifact_id,
                        observed_at=active_epoch,
                        latitude_degrees=row.latitude,
                        longitude_degrees=longitude,
                        vtec_tecu=value,
                    )
                    previous = row_observations.get(longitude)
                    if previous is None:
                        row_observations[longitude] = (
                            source_longitude,
                            observation,
                        )
                        continue

                    previous_longitude, previous_observation = previous
                    if previous_observation.vtec_tecu != value:
                        raise IONEXParseError(
                            "conflicting TEC values at equivalent longitude "
                            f"endpoints {previous_longitude:g} and "
                            f"{source_longitude:g} (both normalize to "
                            f"{longitude:g}) for latitude {row.latitude:g}: "
                            f"{previous_observation.vtec_tecu:g} versus "
                            f"{value:g}",
                            line_number=line.number,
                        )

                active_observations.extend(
                    entry[1]
                    for entry in row_observations.values()
                    if entry is not None
                )
                continue

            if label == _END_TEC_LABEL:
                end_map_number = _parse_integer_field(
                    payload, "TEC map number", line
                )
                if end_map_number != active_map_number:
                    raise IONEXParseError(
                        "START OF TEC MAP and END OF TEC MAP numbers differ",
                        line_number=line.number,
                    )
                if active_epoch is None:
                    raise IONEXParseError(
                        "TEC map has no EPOCH OF CURRENT MAP",
                        line_number=line.number,
                    )
                self._validate_complete_latitude_grid(
                    active_latitudes, grid, line_number=line.number
                )

                parsed_map_count += 1
                previous_epoch = active_epoch
                yield from active_observations
                active_map_number = None
                active_epoch = None
                active_observations = []
                active_latitudes = []
                continue

            if label == _START_TEC_LABEL:
                raise IONEXParseError(
                    "nested START OF TEC MAP record", line_number=line.number
                )
            if label == _END_FILE_LABEL:
                raise IONEXParseError(
                    "END OF FILE occurs inside a TEC map", line_number=line.number
                )
            if line.text.strip():
                description = label or "unlabelled data"
                raise IONEXParseError(
                    f"unexpected {description!r} record inside TEC map",
                    line_number=line.number,
                )

        if active_map_number is not None:
            raise IONEXParseError(
                f"TEC map {active_map_number} is missing END OF TEC MAP"
            )
        if not found_end_of_file:
            raise IONEXParseError("missing END OF FILE record")
        if parsed_map_count != grid.map_count:
            raise IONEXParseError(
                "parsed TEC map count does not match # OF MAPS IN FILE: "
                f"expected {grid.map_count}, got {parsed_map_count}"
            )

    @staticmethod
    def _validate_row(
        row: _Row,
        grid: _Grid,
        latitudes_seen: list[float],
        line: _Line,
    ) -> None:
        expected_index = len(latitudes_seen)
        expected_count = _grid_count(*grid.latitude, name="latitude grid")
        if expected_index >= expected_count:
            raise IONEXParseError(
                "TEC map contains too many latitude rows", line_number=line.number
            )
        expected_latitude = _coordinate(
            grid.latitude[0], grid.latitude[2], expected_index
        )
        if not _close(row.latitude, expected_latitude):
            raise IONEXParseError(
                f"unexpected latitude {row.latitude:g}; "
                f"expected {expected_latitude:g}",
                line_number=line.number,
            )
        if not (
            _close(row.longitude_start, grid.longitude[0])
            and _close(row.longitude_end, grid.longitude[1])
            and _close(row.longitude_step, grid.longitude[2])
        ):
            raise IONEXParseError(
                "TEC row longitude grid differs from the file header",
                line_number=line.number,
            )
        if not _close(row.height, grid.height[0]):
            raise IONEXParseError(
                "TEC row height differs from the 2-D map height",
                line_number=line.number,
            )
        if not -90.0 <= row.latitude <= 90.0:
            raise IONEXParseError(
                "TEC row latitude lies outside -90..90 degrees",
                line_number=line.number,
            )

    @staticmethod
    def _validate_complete_latitude_grid(
        latitudes_seen: list[float],
        grid: _Grid,
        *,
        line_number: int,
    ) -> None:
        expected = _grid_count(*grid.latitude, name="latitude grid")
        if len(latitudes_seen) != expected:
            raise IONEXParseError(
                f"incomplete latitude grid: expected {expected} rows, "
                f"got {len(latitudes_seen)}",
                line_number=line_number,
            )


def _decode_lines(stream: IONEXStream) -> Iterator[_Line]:
    for number, raw_line in enumerate(stream, start=1):
        if isinstance(raw_line, bytes):
            if number == 1 and raw_line.startswith(b"\xef\xbb\xbf"):
                raw_line = raw_line[3:]
            try:
                text = raw_line.decode("ascii")
            except UnicodeDecodeError as exc:
                raise IONEXParseError(
                    "IONEX input must be ASCII", line_number=number
                ) from exc
        elif isinstance(raw_line, str):
            text = raw_line.lstrip("\ufeff") if number == 1 else raw_line
            try:
                text.encode("ascii")
            except UnicodeEncodeError as exc:
                raise IONEXParseError(
                    "IONEX input must be ASCII", line_number=number
                ) from exc
        else:
            raise TypeError("IONEX stream iteration must yield bytes or str")
        yield _Line(number=number, text=text.rstrip("\r\n"))


def _split_record(line: _Line) -> tuple[str, str]:
    text = line.text
    if len(text) >= 60:
        label = _normalise_label(text[60:80])
        # Unlabelled mI5 data records also occupy all 80 columns. Their final
        # 20 columns are numeric values, not a record label.
        if label in _KNOWN_LABELS or any(character.isalpha() for character in label):
            return text[:60], label
        return text, ""

    stripped = text.rstrip()
    for display_label in (
        "IONEX VERSION / TYPE",
        "# OF MAPS IN FILE",
        "MAP DIMENSION",
        "HGT1 / HGT2 / DHGT",
        "LAT1 / LAT2 / DLAT",
        "LON1 / LON2 / DLON",
        "START OF AUX DATA",
        "END OF AUX DATA",
        "END OF HEADER",
        "START OF TEC MAP",
        "END OF TEC MAP",
        "EPOCH OF CURRENT MAP",
        "LAT/LON1/LON2/DLON/H",
        "END OF FILE",
        "EXPONENT",
    ):
        if stripped.endswith(display_label):
            payload = stripped[: -len(display_label)].rstrip()
            return payload, _normalise_label(display_label)
    return text, ""


def _normalise_label(value: str) -> str:
    return "".join(value.upper().split())


def _parse_float_field(value: str, name: str, line: _Line) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise IONEXParseError(
            f"invalid {name}: {value.strip()!r}", line_number=line.number
        ) from exc
    if not math.isfinite(parsed):
        raise IONEXParseError(f"{name} must be finite", line_number=line.number)
    return parsed


def _parse_integer_field(payload: str, name: str, line: _Line) -> int:
    field = payload[:6].strip()
    if not _INTEGER.fullmatch(field):
        tokens = payload.split()
        field = tokens[0] if tokens else ""
    if not _INTEGER.fullmatch(field):
        raise IONEXParseError(
            f"invalid {name}: {field!r}", line_number=line.number
        )
    return int(field)


def _parse_fixed_floats(
    payload: str,
    offset: int,
    width: int,
    count: int,
    line: _Line,
) -> tuple[float, ...]:
    fields = [
        payload[offset + index * width : offset + (index + 1) * width]
        for index in range(count)
    ]
    try:
        values = tuple(float(field.strip()) for field in fields)
    except ValueError:
        tokens = payload.split()
        if len(tokens) < count:
            raise IONEXParseError(
                f"expected {count} numeric fields, got {len(tokens)}",
                line_number=line.number,
            ) from None
        try:
            values = tuple(float(token) for token in tokens[:count])
        except ValueError as exc:
            raise IONEXParseError(
                "invalid numeric grid field", line_number=line.number
            ) from exc
    if not all(math.isfinite(value) for value in values):
        raise IONEXParseError(
            "grid fields must be finite", line_number=line.number
        )
    return values


def _parse_epoch(payload: str, line: _Line) -> datetime:
    tokens = payload.split()
    if len(tokens) != 6 or not all(_INTEGER.fullmatch(token) for token in tokens):
        raise IONEXParseError(
            "EPOCH OF CURRENT MAP must contain six integers",
            line_number=line.number,
        )
    try:
        return datetime(*(int(token) for token in tokens), tzinfo=timezone.utc)
    except ValueError as exc:
        raise IONEXParseError(
            f"invalid EPOCH OF CURRENT MAP: {exc}", line_number=line.number
        ) from exc


def _parse_row(payload: str, line: _Line) -> _Row:
    values = _parse_fixed_floats(payload, 2, 6, 5, line)
    return _Row(*values)


def _read_row_values(
    lines: Iterator[_Line], row: _Row, row_line: _Line
) -> list[int]:
    expected_count = _grid_count(
        row.longitude_start,
        row.longitude_end,
        row.longitude_step,
        name="TEC row longitude grid",
        line_number=row_line.number,
    )
    values: list[int] = []

    while len(values) < expected_count:
        try:
            line = next(lines)
        except StopIteration as exc:
            raise IONEXParseError(
                f"TEC row is incomplete: expected {expected_count} values, "
                f"got {len(values)}",
                line_number=row_line.number,
            ) from exc

        _, label = _split_record(line)
        if label:
            raise IONEXParseError(
                f"TEC row is incomplete before {label!r}: expected "
                f"{expected_count} values, got {len(values)}",
                line_number=line.number,
            )
        line_values = _parse_data_values(line)
        if not line_values:
            raise IONEXParseError(
                "blank TEC data record", line_number=line.number
            )
        if len(values) + len(line_values) > expected_count:
            raise IONEXParseError(
                f"TEC row contains more than {expected_count} values",
                line_number=line.number,
            )
        values.extend(line_values)

    return values


def _parse_data_values(line: _Line) -> list[int]:
    text = line.text
    fixed_fields = [text[index : index + 5] for index in range(0, len(text), 5)]
    nonempty_fields = [field.strip() for field in fixed_fields if field.strip()]
    if nonempty_fields and all(_INTEGER.fullmatch(field) for field in nonempty_fields):
        return [int(field) for field in nonempty_fields]

    tokens = text.split()
    if not all(_INTEGER.fullmatch(token) for token in tokens):
        raise IONEXParseError(
            "TEC data records must contain fixed-width integers",
            line_number=line.number,
        )
    return [int(token) for token in tokens]


def _grid_count(
    start: float,
    end: float,
    step: float,
    *,
    name: str,
    line_number: int | None = None,
) -> int:
    if _close(step, 0.0):
        if _close(start, end):
            return 1
        raise IONEXParseError(
            f"{name} has zero increment but different endpoints",
            line_number=line_number,
        )

    intervals = (end - start) / step
    rounded_intervals = round(intervals)
    if intervals < 0 or not math.isclose(
        intervals, rounded_intervals, rel_tol=0.0, abs_tol=1e-7
    ):
        raise IONEXParseError(
            f"{name} endpoints are not reachable by its increment",
            line_number=line_number,
        )
    return rounded_intervals + 1


def _coordinate(start: float, step: float, index: int) -> float:
    value = start + step * index
    if _close(value, 0.0):
        return 0.0
    return round(value, 10)


def _normalise_longitude(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 0.0 if _close(normalized, 0.0) else round(normalized, 10)


def _scale_for_exponent(exponent: int, line: _Line) -> float:
    try:
        scale = 10.0**exponent
    except OverflowError as exc:
        raise IONEXParseError(
            f"IONEX exponent {exponent} is outside the numeric range",
            line_number=line.number,
        ) from exc
    if not math.isfinite(scale) or scale == 0.0:
        raise IONEXParseError(
            f"IONEX exponent {exponent} is outside the numeric range",
            line_number=line.number,
        )
    return scale


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7)


__all__ = [
    "IONEXError",
    "IONEXParseError",
    "IONEXParser",
    "IONEXStream",
    "IONEXV1Parser",
    "UnsupportedIONEXError",
]
