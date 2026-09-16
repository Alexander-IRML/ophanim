"""Regional selection, aggregation, coverage, and quality handling.

The concrete v0 aggregator deliberately receives a spatial selection strategy.
That keeps ``RegionVersion.boundary_ref`` as provenance instead of quietly
assuming that every region is a rectangular latitude/longitude crop.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import fsum, isfinite
from statistics import median
from typing import Protocol

from ophanim.domain import (
    ProcessingVersion,
    RegionVersion,
    RegionalObservation,
    TECObservation,
)


class AggregationError(ValueError):
    """Base class for invalid or ambiguous aggregation inputs."""


class BoundaryDefinitionError(AggregationError):
    """A region boundary cannot be interpreted safely."""


class UnknownRegionBoundaryError(AggregationError):
    """No configured boundary matches a region's ``boundary_ref``."""


class InsufficientRegionalDataError(AggregationError):
    """A regional value cannot be produced at the configured quality floor."""


class RegionalAggregator(Protocol):
    """Derive both mean and median VTEC for a versioned region."""

    def aggregate(
        self,
        *,
        observations: Iterable[TECObservation],
        region: RegionVersion,
        processing_version: ProcessingVersion,
        produced_at: datetime,
    ) -> RegionalObservation: ...


class RegionSelectionStrategy(Protocol):
    """Decide whether a grid observation belongs to a versioned region."""

    def contains(
        self, *, region: RegionVersion, observation: TECObservation
    ) -> bool: ...


class RegionalObservationIdFactory(Protocol):
    """Injectable observation identity policy for repository-specific IDs."""

    def __call__(
        self,
        *,
        artifact_id: str,
        processing_version_id: str,
        region_version_id: str,
        observed_at: datetime,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class GeographicBounds:
    """Inclusive geographic bounds; west > east crosses the antimeridian."""

    south_latitude_degrees: float
    west_longitude_degrees: float
    north_latitude_degrees: float
    east_longitude_degrees: float

    def __post_init__(self) -> None:
        values = (
            self.south_latitude_degrees,
            self.west_longitude_degrees,
            self.north_latitude_degrees,
            self.east_longitude_degrees,
        )
        if not all(isfinite(value) for value in values):
            raise BoundaryDefinitionError("bounds coordinates must be finite")
        if not -90.0 <= self.south_latitude_degrees <= 90.0:
            raise BoundaryDefinitionError("south latitude is outside [-90, 90]")
        if not -90.0 <= self.north_latitude_degrees <= 90.0:
            raise BoundaryDefinitionError("north latitude is outside [-90, 90]")
        if self.south_latitude_degrees > self.north_latitude_degrees:
            raise BoundaryDefinitionError("south latitude must not exceed north")
        if not -180.0 <= self.west_longitude_degrees < 180.0:
            raise BoundaryDefinitionError("west longitude is outside [-180, 180)")
        if not -180.0 <= self.east_longitude_degrees < 180.0:
            raise BoundaryDefinitionError("east longitude is outside [-180, 180)")

    def contains(self, latitude_degrees: float, longitude_degrees: float) -> bool:
        if not (
            self.south_latitude_degrees
            <= latitude_degrees
            <= self.north_latitude_degrees
        ):
            return False

        west = self.west_longitude_degrees
        east = self.east_longitude_degrees
        if west <= east:
            return west <= longitude_degrees <= east
        return longitude_degrees >= west or longitude_degrees <= east


@dataclass(frozen=True, slots=True)
class PolygonBoundary:
    """A simple latitude/longitude polygon for a local, non-wrapping region.

    Vertices are ``(latitude_degrees, longitude_degrees)`` pairs. Boundary
    points count as inside. The v0 planar implementation intentionally rejects
    edges that cross the antimeridian; a geodesic strategy can be introduced
    later without changing the aggregator.
    """

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3 or len(set(self.vertices)) < 3:
            raise BoundaryDefinitionError("a polygon requires three distinct vertices")
        for latitude, longitude in self.vertices:
            if not isfinite(latitude) or not isfinite(longitude):
                raise BoundaryDefinitionError("polygon coordinates must be finite")
            if not -90.0 <= latitude <= 90.0:
                raise BoundaryDefinitionError("polygon latitude is outside [-90, 90]")
            if not -180.0 <= longitude < 180.0:
                raise BoundaryDefinitionError(
                    "polygon longitude is outside [-180, 180)"
                )

        for first, second in _polygon_edges(self.vertices):
            if abs(first[1] - second[1]) > 180.0:
                raise BoundaryDefinitionError(
                    "v0 polygon boundaries may not cross the antimeridian"
                )

    def contains(self, latitude_degrees: float, longitude_degrees: float) -> bool:
        point = (latitude_degrees, longitude_degrees)
        inside = False
        for first, second in _polygon_edges(self.vertices):
            if _point_on_segment(point, first, second):
                return True

            first_latitude, first_longitude = first
            second_latitude, second_longitude = second
            crosses_latitude = (first_latitude > latitude_degrees) != (
                second_latitude > latitude_degrees
            )
            if not crosses_latitude:
                continue
            crossing_longitude = first_longitude + (
                (latitude_degrees - first_latitude)
                * (second_longitude - first_longitude)
                / (second_latitude - first_latitude)
            )
            if longitude_degrees < crossing_longitude:
                inside = not inside
        return inside


def _polygon_edges(
    vertices: tuple[tuple[float, float], ...],
) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    previous = vertices[-1]
    for current in vertices:
        yield previous, current
        previous = current


def _point_on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    latitude, longitude = point
    first_latitude, first_longitude = first
    second_latitude, second_longitude = second
    cross_product = (latitude - first_latitude) * (
        second_longitude - first_longitude
    ) - (longitude - first_longitude) * (second_latitude - first_latitude)
    if abs(cross_product) > 1e-10:
        return False
    return (
        min(first_latitude, second_latitude) - 1e-10
        <= latitude
        <= max(first_latitude, second_latitude) + 1e-10
        and min(first_longitude, second_longitude) - 1e-10
        <= longitude
        <= max(first_longitude, second_longitude) + 1e-10
    )


class BoundsRegionSelector:
    """Resolve ``boundary_ref`` values to inclusive rectangular bounds."""

    def __init__(self, boundaries: Mapping[str, GeographicBounds]) -> None:
        if not boundaries:
            raise BoundaryDefinitionError("at least one bounds definition is required")
        self._boundaries = dict(boundaries)
        self._boundary_checksums = {
            reference: boundary_checksum_sha256(boundary)
            for reference, boundary in self._boundaries.items()
        }

    def contains(
        self, *, region: RegionVersion, observation: TECObservation
    ) -> bool:
        try:
            boundary = self._boundaries[region.boundary_ref]
        except KeyError as error:
            raise UnknownRegionBoundaryError(
                f"no bounds configured for boundary_ref={region.boundary_ref!r}"
            ) from error
        _verify_boundary_checksum(
            region,
            self._boundary_checksums[region.boundary_ref],
        )
        return boundary.contains(
            observation.latitude_degrees, observation.longitude_degrees
        )


class PolygonRegionSelector:
    """Resolve ``boundary_ref`` values to explicit polygon boundaries."""

    def __init__(self, boundaries: Mapping[str, PolygonBoundary]) -> None:
        if not boundaries:
            raise BoundaryDefinitionError("at least one polygon definition is required")
        self._boundaries = dict(boundaries)
        self._boundary_checksums = {
            reference: boundary_checksum_sha256(boundary)
            for reference, boundary in self._boundaries.items()
        }

    def contains(
        self, *, region: RegionVersion, observation: TECObservation
    ) -> bool:
        try:
            boundary = self._boundaries[region.boundary_ref]
        except KeyError as error:
            raise UnknownRegionBoundaryError(
                f"no polygon configured for boundary_ref={region.boundary_ref!r}"
            ) from error
        _verify_boundary_checksum(
            region,
            self._boundary_checksums[region.boundary_ref],
        )
        return boundary.contains(
            observation.latitude_degrees, observation.longitude_degrees
        )


# Names that emphasize the interchangeable strategy role.
BoundsSelectionStrategy = BoundsRegionSelector
PolygonSelectionStrategy = PolygonRegionSelector


def boundary_checksum_sha256(
    boundary: GeographicBounds | PolygonBoundary,
) -> str:
    """Return a stable checksum for a concrete geographic definition."""

    if isinstance(boundary, GeographicBounds):
        payload: dict[str, object] = {
            "type": "geographic_bounds",
            "coordinates": [
                float(boundary.south_latitude_degrees),
                float(boundary.west_longitude_degrees),
                float(boundary.north_latitude_degrees),
                float(boundary.east_longitude_degrees),
            ],
        }
    elif isinstance(boundary, PolygonBoundary):
        payload = {
            "type": "polygon",
            "vertices": [
                [float(latitude), float(longitude)]
                for latitude, longitude in boundary.vertices
            ],
        }
    else:
        raise TypeError(f"unsupported boundary type: {type(boundary).__name__}")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _verify_boundary_checksum(
    region: RegionVersion,
    actual: str,
) -> None:
    if region.boundary_checksum_sha256 != actual:
        raise BoundaryDefinitionError(
            f"boundary checksum does not match region version {region.region_version_id!r}"
        )


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    """Quality and completeness rules for one expected regional grid."""

    expected_cell_count: int
    minimum_coverage_fraction: float = 0.5
    preferred_coverage_fraction: float = 0.9
    excluded_quality_flags: frozenset[str] = field(
        default_factory=lambda: frozenset({"invalid", "missing", "unusable"})
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.expected_cell_count, int)
            or isinstance(self.expected_cell_count, bool)
            or self.expected_cell_count <= 0
        ):
            raise ValueError("expected_cell_count must be a positive integer")
        if not 0.0 < self.minimum_coverage_fraction <= 1.0:
            raise ValueError("minimum_coverage_fraction must be in (0, 1]")
        if not (
            self.minimum_coverage_fraction
            <= self.preferred_coverage_fraction
            <= 1.0
        ):
            raise ValueError(
                "preferred_coverage_fraction must be between the minimum and 1"
            )
        if any(not flag for flag in self.excluded_quality_flags):
            raise ValueError("quality flags must be non-empty strings")


class MeanMedianRegionalAggregator:
    """Aggregate one artifact/time slice into a deterministic regional record."""

    def __init__(
        self,
        *,
        selection_strategy: RegionSelectionStrategy,
        policy: AggregationPolicy,
        observation_id_factory: RegionalObservationIdFactory | None = None,
    ) -> None:
        self._selection_strategy = selection_strategy
        self._policy = policy
        self._observation_id_factory = (
            observation_id_factory or _regional_observation_id
        )

    def aggregate(
        self,
        *,
        observations: Iterable[TECObservation],
        region: RegionVersion,
        processing_version: ProcessingVersion,
        produced_at: datetime,
    ) -> RegionalObservation:
        candidates = tuple(observations)
        if not candidates:
            raise InsufficientRegionalDataError("no grid observations were supplied")

        _require_aware(region.created_at, "region.created_at")
        _require_aware(processing_version.created_at, "processing_version.created_at")
        _require_aware(produced_at, "produced_at")

        first = candidates[0]
        _validate_grid_observation(first)
        observed_at = first.observed_at
        artifact_id = first.artifact_id
        if produced_at < observed_at:
            raise AggregationError(
                "regional observation cannot be produced before it is observed"
            )
        for observation in candidates[1:]:
            _validate_grid_observation(observation)
            if observation.observed_at != observed_at:
                raise AggregationError(
                    "one aggregation call may contain only one observed_at instant"
                )
            if observation.artifact_id != artifact_id:
                raise AggregationError(
                    "one aggregation call may contain only one source artifact"
                )

        selected = tuple(
            observation
            for observation in candidates
            if self._selection_strategy.contains(
                region=region, observation=observation
            )
        )
        if not selected:
            raise InsufficientRegionalDataError(
                "no observations fall inside the region"
            )
        if len(selected) > self._policy.expected_cell_count:
            raise AggregationError(
                "selected cell count exceeds the configured expected grid size"
            )

        coordinates: set[tuple[float, float]] = set()
        usable_values: list[float] = []
        aggregate_flags: set[str] = set()
        excluded_count = 0
        for observation in selected:
            coordinate = (
                observation.latitude_degrees,
                observation.longitude_degrees,
            )
            if coordinate in coordinates:
                raise AggregationError(
                    "duplicate latitude/longitude cell in one artifact/time slice"
                )
            coordinates.add(coordinate)

            flags = set(observation.quality_flags)
            if flags & self._policy.excluded_quality_flags or not isfinite(
                observation.vtec_tecu
            ):
                aggregate_flags.update(
                    f"excluded_cell:{flag}" for flag in sorted(flags)
                )
                if not isfinite(observation.vtec_tecu):
                    aggregate_flags.add("excluded_cell:non_finite_vtec")
                excluded_count += 1
                continue
            aggregate_flags.update(flags)
            usable_values.append(observation.vtec_tecu)

        coverage_fraction = len(usable_values) / self._policy.expected_cell_count
        if not usable_values or (
            coverage_fraction < self._policy.minimum_coverage_fraction
        ):
            raise InsufficientRegionalDataError(
                "usable regional coverage "
                f"{coverage_fraction:.3f} is below minimum "
                f"{self._policy.minimum_coverage_fraction:.3f}"
            )

        if len(selected) < self._policy.expected_cell_count:
            aggregate_flags.add("missing_cells")
        if excluded_count:
            aggregate_flags.add("excluded_cells")
        if coverage_fraction < self._policy.preferred_coverage_fraction:
            aggregate_flags.add("low_coverage")

        mean_value = fsum(usable_values) / len(usable_values)
        median_value = float(median(usable_values))
        observation_id = self._observation_id_factory(
            artifact_id=artifact_id,
            processing_version_id=processing_version.processing_version_id,
            region_version_id=region.region_version_id,
            observed_at=observed_at,
        )
        if not observation_id:
            raise AggregationError("observation_id_factory returned an empty ID")
        return RegionalObservation(
            observation_id=observation_id,
            artifact_id=artifact_id,
            processing_version_id=processing_version.processing_version_id,
            region_version_id=region.region_version_id,
            observed_at=observed_at,
            produced_at=produced_at,
            mean_vtec_tecu=mean_value,
            median_vtec_tecu=median_value,
            cell_count=len(usable_values),
            coverage_fraction=coverage_fraction,
            quality_flags=tuple(sorted(aggregate_flags)),
        )


def _validate_grid_observation(observation: TECObservation) -> None:
    _require_aware(observation.observed_at, "observation.observed_at")
    if not observation.artifact_id:
        raise AggregationError("observation artifact_id must not be empty")
    latitude = observation.latitude_degrees
    longitude = observation.longitude_degrees
    if not isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise AggregationError("observation latitude must be finite and in [-90, 90]")
    if not isfinite(longitude) or not -180.0 <= longitude < 180.0:
        raise AggregationError(
            "observation longitude must be finite and in [-180, 180)"
        )
    if any(not isinstance(flag, str) or not flag for flag in observation.quality_flags):
        raise AggregationError("observation quality flags must be non-empty strings")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AggregationError(f"{field_name} must be timezone-aware")


def _regional_observation_id(
    *,
    artifact_id: str,
    processing_version_id: str,
    region_version_id: str,
    observed_at: datetime,
) -> str:
    instant = observed_at.astimezone(timezone.utc).isoformat()
    payload = "\x1f".join(
        (artifact_id, processing_version_id, region_version_id, instant)
    ).encode("utf-8")
    return f"regional-{sha256(payload).hexdigest()[:24]}"
