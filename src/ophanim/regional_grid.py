"""Deterministic regional readout grids derived from one native TEC epoch.

The source observations remain the measurements.  Values produced here are
explicitly estimates from regular-grid bilinear interpolation; the module does
not attach or imply an uncertainty that the source product does not provide.

Callers must supply the full native epoch rather than observations already
cropped to the requested bounds.  Interpolation at a regional edge commonly
needs source nodes immediately outside that region.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor, fsum, isclose, isfinite, ulp

from ophanim.aggregation import GeographicBounds
from ophanim.domain import TECObservation


BILINEAR_METHOD = "bilinear"
BILINEAR_METHOD_VERSION = "ophanim-regular-grid-bilinear/1"
DEFAULT_MAX_OUTPUT_CELLS = 10_000

_COORDINATE_DECIMAL_PLACES = 10
_MINIMUM_COORDINATE_STEP = 10**-_COORDINATE_DECIMAL_PLACES
_COORDINATE_ABS_TOLERANCE = _MINIMUM_COORDINATE_STEP / 2.0
_GRID_REGULARITY_QUANTA = 2.0
_GRID_REGULARITY_ULPS = 8


class RegionalGridInterpolationError(ValueError):
    """Base class for invalid regional-grid interpolation operations."""


class InterpolationRequestError(RegionalGridInterpolationError):
    """The requested output bounds or spacing cannot be produced safely."""


class NativeGridError(RegionalGridInterpolationError):
    """The supplied observations do not describe one usable native grid."""


class MissingNativeSupportError(NativeGridError):
    """A requested estimate lacks at least one required native source node."""


@dataclass(frozen=True, slots=True)
class NativeGridPoint:
    """One native source point that contributed to an output value."""

    latitude_degrees: float
    longitude_degrees: float
    vtec_tecu: float
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NativeGridProvenance:
    """Identity and inferred geometry of the native source grid."""

    artifact_id: str
    observed_at: datetime
    latitude_step_degrees: float
    longitude_step_degrees: float
    south_latitude_degrees: float
    north_latitude_degrees: float
    longitude_origin_degrees: float
    longitude_is_cyclic: bool
    latitude_count: int
    longitude_count: int
    source_point_count: int


@dataclass(frozen=True, slots=True)
class RegionalGridEstimate:
    """One table-ready VTEC estimate and its exact native support."""

    latitude_degrees: float
    longitude_degrees: float
    estimated_vtec_tecu: float
    method: str
    is_native_node: bool
    native_support: tuple[NativeGridPoint, ...]


@dataclass(frozen=True, slots=True)
class RegionalGridInterpolation:
    """An immutable regional readout and enough metadata to audit its origin."""

    bounds: GeographicBounds
    requested_latitude_step_degrees: float
    requested_longitude_step_degrees: float
    latitude_count: int
    longitude_count: int
    method: str
    method_version: str
    native_grid: NativeGridProvenance
    rows: tuple[RegionalGridEstimate, ...]

    @property
    def cell_count(self) -> int:
        """Return the number of table rows in this readout."""

        return len(self.rows)


@dataclass(frozen=True, slots=True)
class _NativeGrid:
    latitudes: tuple[float, ...]
    # Eastward, unwrapped coordinates beginning at provenance.longitude_origin.
    longitudes: tuple[float, ...]
    longitude_is_cyclic: bool
    points: dict[tuple[float, float], NativeGridPoint]
    provenance: NativeGridProvenance


def interpolate_regional_grid(
    *,
    observations: Iterable[TECObservation],
    bounds: GeographicBounds,
    latitude_step_degrees: float,
    longitude_step_degrees: float,
    max_output_cells: int = DEFAULT_MAX_OUTPUT_CELLS,
) -> RegionalGridInterpolation:
    """Produce a fine regional table from one regular native TEC grid.

    Output coordinates begin at the south/west bounds and advance by the
    requested step.  The exact north/east bounds are appended when a span is
    not evenly divisible by its step.  Longitudes advance eastward through the
    antimeridian when ``bounds.west_longitude_degrees > bounds.east...``.

    Exact native nodes are copied without arithmetic.  On a native latitude or
    longitude line, the bilinear calculation naturally reduces to a one-axis
    linear interpolation.  Every distinct source node actually needed by that
    calculation must be present. Request coordinates and steps are normalized
    for harmless floating-point noise; values with meaningful precision beyond
    the parser's ten decimal places are rejected.
    """

    if not isinstance(bounds, GeographicBounds):
        raise InterpolationRequestError("bounds must be GeographicBounds")
    normalized_bounds = _normalize_request_bounds(bounds)
    latitude_step = _validate_step(
        latitude_step_degrees,
        field_name="latitude_step_degrees",
        maximum=180.0,
    )
    longitude_step = _validate_step(
        longitude_step_degrees,
        field_name="longitude_step_degrees",
        maximum=360.0,
    )
    if (
        not isinstance(max_output_cells, int)
        or isinstance(max_output_cells, bool)
        or max_output_cells <= 0
    ):
        raise InterpolationRequestError("max_output_cells must be a positive integer")

    latitude_span = (
        normalized_bounds.north_latitude_degrees
        - normalized_bounds.south_latitude_degrees
    )
    longitude_span = _eastward_longitude_span(
        normalized_bounds.west_longitude_degrees,
        normalized_bounds.east_longitude_degrees,
    )
    latitude_count = _axis_sample_count(latitude_span, latitude_step)
    longitude_count = _axis_sample_count(longitude_span, longitude_step)
    if (
        latitude_count > max_output_cells
        or longitude_count > max_output_cells // latitude_count
    ):
        requested_count = latitude_count * longitude_count
        raise InterpolationRequestError(
            f"requested regional grid has {requested_count} cells; "
            f"max_output_cells is {max_output_cells}"
        )

    native = _build_native_grid(observations)
    _validate_latitude_support(normalized_bounds, native.latitudes)
    _validate_longitude_support(normalized_bounds, native)

    latitude_targets = _axis_values(
        start=normalized_bounds.south_latitude_degrees,
        span=latitude_span,
        step=latitude_step,
        count=latitude_count,
        axis_name="latitude",
    )
    unwrapped_longitude_targets = _axis_values(
        start=normalized_bounds.west_longitude_degrees,
        span=longitude_span,
        step=longitude_step,
        count=longitude_count,
        axis_name="longitude",
    )

    rows: list[RegionalGridEstimate] = []
    for latitude in latitude_targets:
        for unwrapped_longitude in unwrapped_longitude_targets:
            longitude = _canonical_longitude(unwrapped_longitude)
            rows.append(
                _interpolate_point(
                    native=native,
                    latitude=latitude,
                    longitude=longitude,
                )
            )

    return RegionalGridInterpolation(
        bounds=normalized_bounds,
        requested_latitude_step_degrees=latitude_step,
        requested_longitude_step_degrees=longitude_step,
        latitude_count=latitude_count,
        longitude_count=longitude_count,
        method=BILINEAR_METHOD,
        method_version=BILINEAR_METHOD_VERSION,
        native_grid=native.provenance,
        rows=tuple(rows),
    )


def _validate_step(value: float, *, field_name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterpolationRequestError(f"{field_name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise InterpolationRequestError(
            f"{field_name} must be a finite number"
        ) from error
    if not isfinite(normalized):
        raise InterpolationRequestError(f"{field_name} must be finite")
    if not 0.0 < normalized <= maximum:
        raise InterpolationRequestError(
            f"{field_name} must be in (0, {maximum:g}]"
        )
    if normalized < _MINIMUM_COORDINATE_STEP:
        raise InterpolationRequestError(
            f"{field_name} must be at least {_MINIMUM_COORDINATE_STEP:g} "
            "at the supported coordinate precision"
        )
    return _normalize_request_coordinate(normalized, field_name=field_name)


def _normalize_request_bounds(bounds: GeographicBounds) -> GeographicBounds:
    fields = {
        "south_latitude_degrees": bounds.south_latitude_degrees,
        "west_longitude_degrees": bounds.west_longitude_degrees,
        "north_latitude_degrees": bounds.north_latitude_degrees,
        "east_longitude_degrees": bounds.east_longitude_degrees,
    }
    normalized = {
        name: _normalize_request_coordinate(value, field_name=name)
        for name, value in fields.items()
    }
    if normalized["west_longitude_degrees"] == 180.0 or (
        normalized["east_longitude_degrees"] == 180.0
    ):
        raise InterpolationRequestError(
            "longitude bounds cannot round to 180 at the supported coordinate "
            "precision; use canonical -180 instead"
        )
    return GeographicBounds(**normalized)


def _normalize_request_coordinate(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterpolationRequestError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise InterpolationRequestError(
            f"{field_name} must be a finite number"
        ) from error
    if not isfinite(number):
        raise InterpolationRequestError(f"{field_name} must be finite")
    rounded = _round_coordinate(number)
    floating_tolerance = _GRID_REGULARITY_ULPS * ulp(max(abs(number), 1.0))
    if abs(number - rounded) > floating_tolerance:
        raise InterpolationRequestError(
            f"{field_name} exceeds the supported "
            f"{_COORDINATE_DECIMAL_PLACES}-decimal coordinate precision"
        )
    return rounded


def _eastward_longitude_span(west: float, east: float) -> float:
    if west <= east:
        return east - west
    return east + 360.0 - west


def _axis_sample_count(span: float, step: float) -> int:
    if _close(span, 0.0):
        return 1
    ratio = span / step
    if not isfinite(ratio):
        raise InterpolationRequestError(
            "requested step is too small to construct a finite output grid"
        )
    whole_steps = floor(ratio)
    last_regular = whole_steps * step
    return whole_steps + 1 if _close(last_regular, span) else whole_steps + 2


def _axis_values(
    *,
    start: float,
    span: float,
    step: float,
    count: int,
    axis_name: str,
) -> tuple[float, ...]:
    if count == 1:
        return (_round_coordinate(start),)
    values = [start + index * step for index in range(count - 1)]
    endpoint = start + span
    if _close(values[-1], endpoint):
        values[-1] = endpoint
    else:
        values.append(endpoint)
    if len(values) != count:
        raise AssertionError("axis count and generated values diverged")
    rounded = tuple(_round_coordinate(value) for value in values)
    if any(
        second <= first
        for first, second in zip(rounded, rounded[1:], strict=False)
    ):
        raise InterpolationRequestError(
            f"requested {axis_name} axis contains duplicate coordinates at the "
            f"supported {_COORDINATE_DECIMAL_PLACES}-decimal precision"
        )
    return rounded


def _build_native_grid(observations: Iterable[TECObservation]) -> _NativeGrid:
    candidates = tuple(observations)
    if not candidates:
        raise NativeGridError("no native TEC observations were supplied")

    first = candidates[0]
    _validate_observation(first)
    artifact_id = first.artifact_id
    observed_at = first.observed_at.astimezone(timezone.utc)
    points: dict[tuple[float, float], NativeGridPoint] = {}

    for observation in candidates:
        latitude, longitude, value = _validate_observation(observation)
        if observation.artifact_id != artifact_id:
            raise NativeGridError(
                "one interpolation call may contain only one source artifact"
            )
        if observation.observed_at.astimezone(timezone.utc) != observed_at:
            raise NativeGridError(
                "one interpolation call may contain only one observed_at instant"
            )
        coordinate = (
            _round_coordinate(latitude),
            _canonical_longitude(longitude),
        )
        if coordinate in points:
            raise NativeGridError(
                "duplicate native latitude/longitude coordinate "
                f"{_format_coordinate(*coordinate)}"
            )
        points[coordinate] = NativeGridPoint(
            latitude_degrees=coordinate[0],
            longitude_degrees=coordinate[1],
            vtec_tecu=value,
            quality_flags=tuple(observation.quality_flags),
        )

    latitudes = tuple(sorted({coordinate[0] for coordinate in points}))
    canonical_longitudes = tuple(sorted({coordinate[1] for coordinate in points}))
    latitude_step = _regular_axis_step(
        latitudes,
        axis_name="latitude",
    )
    longitudes, longitude_step, longitude_is_cyclic = _longitude_axis(
        canonical_longitudes
    )
    provenance = NativeGridProvenance(
        artifact_id=artifact_id,
        observed_at=observed_at,
        latitude_step_degrees=latitude_step,
        longitude_step_degrees=longitude_step,
        south_latitude_degrees=latitudes[0],
        north_latitude_degrees=latitudes[-1],
        longitude_origin_degrees=_canonical_longitude(longitudes[0]),
        longitude_is_cyclic=longitude_is_cyclic,
        latitude_count=len(latitudes),
        longitude_count=len(longitudes),
        source_point_count=len(points),
    )
    return _NativeGrid(
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_is_cyclic=longitude_is_cyclic,
        points=points,
        provenance=provenance,
    )


def _validate_observation(
    observation: TECObservation,
) -> tuple[float, float, float]:
    if not isinstance(observation.artifact_id, str) or not observation.artifact_id:
        raise NativeGridError("native observation artifact_id must not be empty")
    if not isinstance(observation.observed_at, datetime) or (
        observation.observed_at.tzinfo is None
        or observation.observed_at.utcoffset() is None
    ):
        raise NativeGridError("native observation observed_at must be timezone-aware")

    latitude = _native_number(
        observation.latitude_degrees,
        label="latitude",
    )
    longitude = _native_number(
        observation.longitude_degrees,
        label="longitude",
    )
    value = _native_number(
        observation.vtec_tecu,
        label="VTEC",
    )
    if not -90.0 <= latitude <= 90.0:
        raise NativeGridError("native latitude must be finite and in [-90, 90]")
    if not -180.0 <= longitude < 180.0:
        raise NativeGridError(
            "native longitude must be finite and in canonical [-180, 180)"
        )
    if not isinstance(observation.quality_flags, tuple) or any(
        not isinstance(flag, str) or not flag for flag in observation.quality_flags
    ):
        raise NativeGridError(
            "native quality_flags must be a tuple of non-empty strings"
        )
    return latitude, longitude, value


def _native_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeGridError(f"native {label} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise NativeGridError(f"native {label} must be finite") from error
    if not isfinite(normalized):
        raise NativeGridError(f"native {label} must be finite")
    return normalized


def _regular_axis_step(
    values: tuple[float, ...],
    *,
    axis_name: str,
) -> float:
    if len(values) < 2:
        raise NativeGridError(
            f"native {axis_name} axis requires at least two distinct coordinates"
        )
    differences = [
        second - first for first, second in zip(values, values[1:], strict=False)
    ]
    step = fsum(differences) / len(differences)
    if step <= 0.0 or not all(
        _regular_gap_close(difference, step) for difference in differences
    ):
        raise NativeGridError(
            f"native {axis_name} coordinates do not form a regular axis"
        )
    return _round_coordinate(step)


def _longitude_axis(
    canonical_values: tuple[float, ...],
) -> tuple[tuple[float, ...], float, bool]:
    """Return an eastward regular axis and whether it covers the full cycle.

    A local axis has one uncovered circular gap.  Starting immediately after
    that gap makes both ordinary local strips and antimeridian-crossing strips
    a monotonically increasing, non-cyclic axis.  A complete global axis has
    no larger gap and is safe to interpolate periodically.
    """

    if len(canonical_values) < 2:
        raise NativeGridError(
            "native longitude axis requires at least two distinct coordinates"
        )
    circular_gaps = [
        second - first
        for first, second in zip(
            canonical_values,
            canonical_values[1:],
            strict=False,
        )
    ]
    circular_gaps.append(canonical_values[0] + 360.0 - canonical_values[-1])
    if any(gap <= 0.0 for gap in circular_gaps):
        raise NativeGridError("native longitude axis has duplicate coordinates")

    cyclic_step = 360.0 / len(canonical_values)
    if all(_regular_gap_close(gap, cyclic_step) for gap in circular_gaps):
        return canonical_values, _round_coordinate(cyclic_step), True

    local_candidates: list[tuple[int, float]] = []
    for cut_index in range(len(circular_gaps)):
        retained_gaps = [
            gap
            for index, gap in enumerate(circular_gaps)
            if index != cut_index
        ]
        candidate_step = fsum(retained_gaps) / len(retained_gaps)
        if all(
            _regular_gap_close(gap, candidate_step) for gap in retained_gaps
        ):
            local_candidates.append((cut_index, candidate_step))
    if len(local_candidates) != 1:
        raise NativeGridError(
            "native longitude coordinates do not form one unambiguous regular "
            "local or cyclic axis"
        )

    gap_index, step = local_candidates[0]
    start_index = (gap_index + 1) % len(canonical_values)
    ordered_canonical = (
        canonical_values[start_index:] + canonical_values[:start_index]
    )
    unwrapped: list[float] = []
    previous: float | None = None
    for value in ordered_canonical:
        candidate = value
        if previous is not None and candidate < previous:
            candidate += 360.0
        unwrapped.append(candidate)
        previous = candidate
    if not all(
        _regular_gap_close(second - first, step)
        for first, second in zip(unwrapped, unwrapped[1:], strict=False)
    ):
        raise NativeGridError(
            "native longitude coordinates do not form one regular local or cyclic axis"
        )
    return tuple(unwrapped), _round_coordinate(step), False


def _validate_latitude_support(
    bounds: GeographicBounds,
    native_latitudes: tuple[float, ...],
) -> None:
    minimum = native_latitudes[0]
    maximum = native_latitudes[-1]
    if (
        bounds.south_latitude_degrees < minimum
        and not _close(bounds.south_latitude_degrees, minimum)
    ) or (
        bounds.north_latitude_degrees > maximum
        and not _close(bounds.north_latitude_degrees, maximum)
    ):
        raise MissingNativeSupportError(
            "requested latitude bounds lie outside native grid support "
            f"[{minimum:g}, {maximum:g}]"
        )


def _validate_longitude_support(
    bounds: GeographicBounds,
    native: _NativeGrid,
) -> None:
    if native.longitude_is_cyclic:
        return
    requested_start = _unwrap_longitude(
        bounds.west_longitude_degrees,
        native.longitudes[0],
    )
    requested_end = requested_start + _eastward_longitude_span(
        bounds.west_longitude_degrees,
        bounds.east_longitude_degrees,
    )
    native_start = native.longitudes[0]
    native_end = native.longitudes[-1]
    if (
        requested_start < native_start and not _close(requested_start, native_start)
    ) or (requested_end > native_end and not _close(requested_end, native_end)):
        canonical_start = _canonical_longitude(native_start)
        canonical_end = _canonical_longitude(native_end)
        raise MissingNativeSupportError(
            "requested longitude bounds lie outside non-cyclic native grid support "
            f"[{canonical_start:g}, {canonical_end:g}] eastward"
        )


def _interpolate_point(
    *,
    native: _NativeGrid,
    latitude: float,
    longitude: float,
) -> RegionalGridEstimate:
    south, north = _linear_bracket(native.latitudes, latitude, cyclic=False)
    longitude_unwrapped = _unwrap_longitude(longitude, native.longitudes[0])
    west_unwrapped, east_unwrapped = _linear_bracket(
        native.longitudes,
        longitude_unwrapped,
        cyclic=native.longitude_is_cyclic,
    )
    west = _canonical_longitude(west_unwrapped)
    east = _canonical_longitude(east_unwrapped)

    support_coordinates: list[tuple[float, float]] = []
    for support_latitude in _distinct_pair(south, north):
        for support_longitude in _distinct_pair(west, east):
            support_coordinates.append((support_latitude, support_longitude))

    missing = [
        coordinate
        for coordinate in support_coordinates
        if coordinate not in native.points
    ]
    if missing:
        formatted_missing = ", ".join(
            _format_coordinate(*coordinate) for coordinate in missing
        )
        raise MissingNativeSupportError(
            "cannot estimate target "
            f"{_format_coordinate(latitude, longitude)}; "
            f"missing native support: {formatted_missing}"
        )

    support = tuple(native.points[coordinate] for coordinate in support_coordinates)
    if _close(south, north) and _close(west_unwrapped, east_unwrapped):
        value = support[0].vtec_tecu
        is_native_node = True
    elif _close(south, north):
        west_value = native.points[(south, west)].vtec_tecu
        east_value = native.points[(south, east)].vtec_tecu
        longitude_weight = (longitude_unwrapped - west_unwrapped) / (
            east_unwrapped - west_unwrapped
        )
        value = _linear_value(west_value, east_value, longitude_weight)
        is_native_node = False
    elif _close(west_unwrapped, east_unwrapped):
        south_value = native.points[(south, west)].vtec_tecu
        north_value = native.points[(north, west)].vtec_tecu
        latitude_weight = (latitude - south) / (north - south)
        value = _linear_value(south_value, north_value, latitude_weight)
        is_native_node = False
    else:
        south_west = native.points[(south, west)].vtec_tecu
        south_east = native.points[(south, east)].vtec_tecu
        north_west = native.points[(north, west)].vtec_tecu
        north_east = native.points[(north, east)].vtec_tecu
        latitude_weight = (latitude - south) / (north - south)
        longitude_weight = (longitude_unwrapped - west_unwrapped) / (
            east_unwrapped - west_unwrapped
        )
        south_value = _linear_value(
            south_west,
            south_east,
            longitude_weight,
        )
        north_value = _linear_value(
            north_west,
            north_east,
            longitude_weight,
        )
        value = _linear_value(south_value, north_value, latitude_weight)
        is_native_node = False

    if not isfinite(value):
        raise NativeGridError("bilinear interpolation produced non-finite VTEC")
    output_latitude = (
        south if _close(south, north) else _round_coordinate(latitude)
    )
    output_longitude = (
        west
        if _close(west_unwrapped, east_unwrapped)
        else _canonical_longitude(longitude)
    )
    return RegionalGridEstimate(
        latitude_degrees=output_latitude,
        longitude_degrees=output_longitude,
        estimated_vtec_tecu=value,
        method=BILINEAR_METHOD,
        is_native_node=is_native_node,
        native_support=support,
    )


def _linear_bracket(
    axis: tuple[float, ...],
    target: float,
    *,
    cyclic: bool,
) -> tuple[float, float]:
    index = bisect_left(axis, target)
    if index < len(axis) and _close(axis[index], target):
        return axis[index], axis[index]
    if index and _close(axis[index - 1], target):
        return axis[index - 1], axis[index - 1]

    if not cyclic:
        if index == 0 or index == len(axis):
            raise MissingNativeSupportError(
                f"target coordinate {target:g} lies outside native axis support"
            )
        return axis[index - 1], axis[index]
    if index == 0:
        return axis[-1] - 360.0, axis[0]
    if index == len(axis):
        return axis[-1], axis[0] + 360.0
    return axis[index - 1], axis[index]


def _distinct_pair(first: float, second: float) -> tuple[float, ...]:
    return (first,) if _close(first, second) else (first, second)


def _linear_value(first: float, second: float, weight: float) -> float:
    if first == second:
        return first
    return (1.0 - weight) * first + weight * second


def _canonical_longitude(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    rounded = _round_coordinate(normalized)
    return -180.0 if rounded == 180.0 else rounded


def _unwrap_longitude(value: float, origin: float) -> float:
    canonical = _canonical_longitude(value)
    if canonical < origin and not _close(canonical, origin):
        return canonical + 360.0
    return canonical


def _round_coordinate(value: float) -> float:
    return _clean_zero(round(value, _COORDINATE_DECIMAL_PLACES))


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _close(first: float, second: float) -> bool:
    return isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=_COORDINATE_ABS_TOLERANCE,
    )


def _regular_gap_close(first: float, second: float) -> bool:
    floating_margin = _GRID_REGULARITY_ULPS * max(
        ulp(max(abs(first), 1.0)),
        ulp(max(abs(second), 1.0)),
    )
    tolerance = (
        _GRID_REGULARITY_QUANTA * _MINIMUM_COORDINATE_STEP
        + floating_margin
    )
    return abs(first - second) <= tolerance


def _format_coordinate(latitude: float, longitude: float) -> str:
    return f"({latitude:g}, {longitude:g})"


__all__ = [
    "BILINEAR_METHOD",
    "BILINEAR_METHOD_VERSION",
    "DEFAULT_MAX_OUTPUT_CELLS",
    "InterpolationRequestError",
    "MissingNativeSupportError",
    "NativeGridError",
    "NativeGridPoint",
    "NativeGridProvenance",
    "RegionalGridEstimate",
    "RegionalGridInterpolation",
    "RegionalGridInterpolationError",
    "interpolate_regional_grid",
]
