"""Tests for deterministic, dependency-free regional TEC interpolation."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

from ophanim.aggregation import GeographicBounds
from ophanim.domain import TECObservation
from ophanim.regional_grid import (
    BILINEAR_METHOD,
    BILINEAR_METHOD_VERSION,
    InterpolationRequestError,
    MissingNativeSupportError,
    NativeGridError,
    interpolate_regional_grid,
)


EPOCH = datetime(2026, 9, 11, 12, tzinfo=timezone.utc)


def observation(latitude: float, longitude: float, value: float) -> TECObservation:
    return TECObservation(
        artifact_id="artifact-grid-v1",
        observed_at=EPOCH,
        latitude_degrees=latitude,
        longitude_degrees=longitude,
        vtec_tecu=value,
    )


def native_grid() -> tuple[TECObservation, ...]:
    """A complete 10 by 90 degree grid with a planar local test surface."""

    return tuple(
        observation(
            latitude,
            longitude,
            2.0 * latitude + (longitude + 180.0) / 10.0,
        )
        for latitude in (0.0, 10.0)
        for longitude in (-180.0, -90.0, 0.0, 90.0)
    )


class RegionalGridInterpolationTests(unittest.TestCase):
    def test_bilinear_plane_exact_nodes_and_provenance(self) -> None:
        result = interpolate_regional_grid(
            observations=reversed(native_grid()),
            bounds=GeographicBounds(0.0, -180.0, 10.0, -90.0),
            latitude_step_degrees=5.0,
            longitude_step_degrees=45.0,
        )

        self.assertEqual(result.method, BILINEAR_METHOD)
        self.assertEqual(result.method_version, BILINEAR_METHOD_VERSION)
        self.assertEqual(result.latitude_count, 3)
        self.assertEqual(result.longitude_count, 3)
        self.assertEqual(result.cell_count, 9)
        self.assertEqual(
            [
                (row.latitude_degrees, row.longitude_degrees)
                for row in result.rows
            ],
            [
                (0.0, -180.0),
                (0.0, -135.0),
                (0.0, -90.0),
                (5.0, -180.0),
                (5.0, -135.0),
                (5.0, -90.0),
                (10.0, -180.0),
                (10.0, -135.0),
                (10.0, -90.0),
            ],
        )

        exact = result.rows[0]
        self.assertTrue(exact.is_native_node)
        self.assertEqual(exact.estimated_vtec_tecu, 0.0)
        self.assertEqual(len(exact.native_support), 1)

        center = result.rows[4]
        self.assertFalse(center.is_native_node)
        self.assertAlmostEqual(center.estimated_vtec_tecu, 14.5)
        self.assertEqual(
            [
                (point.latitude_degrees, point.longitude_degrees)
                for point in center.native_support
            ],
            [
                (0.0, -180.0),
                (0.0, -90.0),
                (10.0, -180.0),
                (10.0, -90.0),
            ],
        )

        provenance = result.native_grid
        self.assertEqual(provenance.artifact_id, "artifact-grid-v1")
        self.assertEqual(provenance.observed_at, EPOCH)
        self.assertEqual(provenance.latitude_step_degrees, 10.0)
        self.assertEqual(provenance.longitude_step_degrees, 90.0)
        self.assertTrue(provenance.longitude_is_cyclic)
        self.assertEqual(provenance.latitude_count, 2)
        self.assertEqual(provenance.longitude_count, 4)
        self.assertEqual(provenance.source_point_count, 8)

    def test_asymmetric_bilinear_and_latitude_linear_weights(self) -> None:
        values = {
            (0.0, -180.0): 0.0,
            (0.0, -90.0): 10.0,
            (0.0, 0.0): 0.0,
            (0.0, 90.0): 0.0,
            (10.0, -180.0): 20.0,
            (10.0, -90.0): 40.0,
            (10.0, 0.0): 0.0,
            (10.0, 90.0): 0.0,
        }
        inputs = tuple(
            observation(latitude, longitude, value)
            for (latitude, longitude), value in values.items()
        )
        bilinear = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(2.5, -157.5, 2.5, -157.5),
            latitude_step_degrees=1.0,
            longitude_step_degrees=1.0,
        )
        latitude_linear = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(2.5, -180.0, 2.5, -180.0),
            latitude_step_degrees=1.0,
            longitude_step_degrees=1.0,
        )

        self.assertAlmostEqual(bilinear.rows[0].estimated_vtec_tecu, 8.125)
        self.assertEqual(len(bilinear.rows[0].native_support), 4)
        self.assertAlmostEqual(
            latitude_linear.rows[0].estimated_vtec_tecu,
            5.0,
        )
        self.assertEqual(len(latitude_linear.rows[0].native_support), 2)

    def test_non_divisible_spans_append_exact_north_and_east_bounds(self) -> None:
        result = interpolate_regional_grid(
            observations=native_grid(),
            bounds=GeographicBounds(2.0, -170.0, 9.0, -100.0),
            latitude_step_degrees=4.0,
            longitude_step_degrees=30.0,
        )

        self.assertEqual(result.latitude_count, 3)
        self.assertEqual(result.longitude_count, 4)
        self.assertEqual(
            sorted({row.latitude_degrees for row in result.rows}),
            [2.0, 6.0, 9.0],
        )
        self.assertEqual(
            [row.longitude_degrees for row in result.rows[:4]],
            [-170.0, -140.0, -110.0, -100.0],
        )

    def test_antimeridian_uses_periodic_native_neighbors(self) -> None:
        seam_values = {
            -180.0: 20.0,
            -90.0: 30.0,
            0.0: 40.0,
            90.0: 10.0,
        }
        inputs = tuple(
            observation(latitude, longitude, seam_values[longitude])
            for latitude in (0.0, 10.0)
            for longitude in (-180.0, -90.0, 0.0, 90.0)
        )

        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(0.0, 135.0, 0.0, -135.0),
            latitude_step_degrees=1.0,
            longitude_step_degrees=45.0,
        )

        self.assertEqual(
            [row.longitude_degrees for row in result.rows],
            [135.0, -180.0, -135.0],
        )
        self.assertEqual(
            [row.estimated_vtec_tecu for row in result.rows],
            [15.0, 20.0, 25.0],
        )
        self.assertFalse(result.rows[0].is_native_node)
        self.assertTrue(result.rows[1].is_native_node)
        self.assertEqual(
            [
                point.longitude_degrees
                for point in result.rows[0].native_support
            ],
            [90.0, -180.0],
        )

    def test_regular_local_longitude_axis_is_not_treated_as_periodic(self) -> None:
        inputs = tuple(
            observation(latitude, longitude, latitude + longitude + 100.0)
            for latitude in (0.0, 10.0)
            for longitude in (-100.0, -95.0, -90.0)
        )
        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(5.0, -97.5, 5.0, -92.5),
            latitude_step_degrees=1.0,
            longitude_step_degrees=2.5,
        )

        self.assertFalse(result.native_grid.longitude_is_cyclic)
        self.assertEqual(result.native_grid.longitude_origin_degrees, -100.0)
        self.assertEqual(
            [row.estimated_vtec_tecu for row in result.rows],
            [7.5, 10.0, 12.5],
        )

        with self.assertRaisesRegex(
            MissingNativeSupportError,
            "outside non-cyclic native grid support",
        ):
            interpolate_regional_grid(
                observations=inputs,
                bounds=GeographicBounds(5.0, -95.0, 5.0, -100.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=5.0,
            )

    def test_wide_local_axis_uses_its_regular_step_not_the_shorter_gap(self) -> None:
        longitude_values = {
            -180.0: 0.0,
            -80.0: 100.0,
            20.0: 200.0,
            120.0: 300.0,
        }
        inputs = tuple(
            observation(latitude, longitude, value)
            for latitude in (0.0, 10.0)
            for longitude, value in longitude_values.items()
        )
        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(0.0, -130.0, 0.0, -130.0),
            latitude_step_degrees=1.0,
            longitude_step_degrees=1.0,
        )

        self.assertFalse(result.native_grid.longitude_is_cyclic)
        self.assertEqual(result.native_grid.longitude_step_degrees, 100.0)
        self.assertAlmostEqual(result.rows[0].estimated_vtec_tecu, 50.0)
        with self.assertRaisesRegex(
            MissingNativeSupportError,
            "outside non-cyclic native grid support",
        ):
            interpolate_regional_grid(
                observations=inputs,
                bounds=GeographicBounds(0.0, 150.0, 0.0, 150.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=1.0,
            )

    def test_two_point_local_longitude_axis_is_rejected_as_ambiguous(self) -> None:
        inputs = tuple(
            observation(latitude, longitude, latitude + longitude)
            for latitude in (0.0, 10.0)
            for longitude in (-100.0, -90.0)
        )

        with self.assertRaisesRegex(NativeGridError, "unambiguous"):
            interpolate_regional_grid(
                observations=inputs,
                bounds=GeographicBounds(5.0, -95.0, 5.0, -95.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=1.0,
            )

    def test_local_native_axis_may_cross_the_antimeridian(self) -> None:
        unwrapped_values = {
            170.0: 0.0,
            175.0: 5.0,
            -180.0: 10.0,
            -175.0: 15.0,
            -170.0: 20.0,
        }
        inputs = tuple(
            observation(latitude, longitude, unwrapped_values[longitude])
            for latitude in (0.0, 10.0)
            for longitude in (170.0, 175.0, -180.0, -175.0, -170.0)
        )
        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(0.0, 175.0, 0.0, -175.0),
            latitude_step_degrees=1.0,
            longitude_step_degrees=2.5,
        )

        self.assertFalse(result.native_grid.longitude_is_cyclic)
        self.assertEqual(result.native_grid.longitude_origin_degrees, 170.0)
        self.assertEqual(
            [row.longitude_degrees for row in result.rows],
            [175.0, 177.5, -180.0, -177.5, -175.0],
        )
        self.assertEqual(
            [row.estimated_vtec_tecu for row in result.rows],
            [5.0, 7.5, 10.0, 12.5, 15.0],
        )

    def test_missing_required_corner_is_rejected(self) -> None:
        inputs = tuple(
            cell
            for cell in native_grid()
            if not (
                cell.latitude_degrees == 10.0
                and cell.longitude_degrees == -90.0
            )
        )

        with self.assertRaisesRegex(
            MissingNativeSupportError,
            r"target \(5, -135\).*\(10, -90\)",
        ):
            interpolate_regional_grid(
                observations=inputs,
                bounds=GeographicBounds(5.0, -135.0, 5.0, -135.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=1.0,
            )

    def test_native_grid_identity_and_coordinates_are_unambiguous(self) -> None:
        inputs = native_grid()
        scenarios = {
            "source artifact": inputs + (
                replace(inputs[-1], artifact_id="artifact-other"),
            ),
            "observed_at instant": inputs + (
                replace(inputs[-1], observed_at=EPOCH + timedelta(hours=1)),
            ),
            "duplicate native": inputs + (inputs[-1],),
        }
        for message, candidate in scenarios.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(NativeGridError, message):
                    interpolate_regional_grid(
                        observations=candidate,
                        bounds=GeographicBounds(5.0, -135.0, 5.0, -135.0),
                        latitude_step_degrees=1.0,
                        longitude_step_degrees=1.0,
                    )

    def test_non_finite_values_and_noncanonical_longitudes_are_rejected(self) -> None:
        for changed, message in (
            (replace(native_grid()[0], vtec_tecu=math.nan), "VTEC must be finite"),
            (
                replace(native_grid()[0], latitude_degrees=math.inf),
                "latitude must be finite",
            ),
            (
                replace(native_grid()[0], longitude_degrees=180.0),
                "canonical.*180",
            ),
        ):
            with self.subTest(message=message):
                inputs = (changed,) + native_grid()[1:]
                with self.assertRaisesRegex(NativeGridError, message):
                    interpolate_regional_grid(
                        observations=inputs,
                        bounds=GeographicBounds(5.0, -135.0, 5.0, -135.0),
                        latitude_step_degrees=1.0,
                        longitude_step_degrees=1.0,
                    )

    def test_irregular_native_axes_are_rejected(self) -> None:
        inputs = tuple(
            replace(cell, longitude_degrees=-80.0)
            if cell.longitude_degrees == -90.0
            else cell
            for cell in native_grid()
        )

        with self.assertRaisesRegex(NativeGridError, "regular local or cyclic axis"):
            interpolate_regional_grid(
                observations=inputs,
                bounds=GeographicBounds(5.0, -135.0, 5.0, -135.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=1.0,
            )

    def test_request_validation_and_cell_cap_precede_source_consumption(self) -> None:
        class MustNotIterate:
            def __iter__(self):
                raise AssertionError("observations consumed before output cap check")

        with self.assertRaisesRegex(InterpolationRequestError, "1001 cells"):
            interpolate_regional_grid(
                observations=MustNotIterate(),
                bounds=GeographicBounds(0.0, -180.0, 10.0, -90.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=1.0,
                max_output_cells=1_000,
            )

        for field, value, message in (
            ("latitude", 0.0, r"latitude_step_degrees.*\(0, 180\]"),
            ("latitude", math.inf, "latitude_step_degrees must be finite"),
            ("longitude", 361.0, r"longitude_step_degrees.*\(0, 360\]"),
        ):
            with self.subTest(field=field, value=value):
                keyword_arguments = {
                    "observations": native_grid(),
                    "bounds": GeographicBounds(0.0, -180.0, 0.0, -180.0),
                    "latitude_step_degrees": 1.0,
                    "longitude_step_degrees": 1.0,
                }
                keyword_arguments[f"{field}_step_degrees"] = value
                with self.assertRaisesRegex(InterpolationRequestError, message):
                    interpolate_regional_grid(**keyword_arguments)

    def test_latitudes_outside_native_support_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            MissingNativeSupportError,
            r"outside native grid support \[0, 10\]",
        ):
            interpolate_regional_grid(
                observations=native_grid(),
                bounds=GeographicBounds(-1.0, -135.0, 5.0, -135.0),
                latitude_step_degrees=1.0,
                longitude_step_degrees=1.0,
            )

    def test_public_result_rows_and_support_are_immutable(self) -> None:
        result = interpolate_regional_grid(
            observations=native_grid(),
            bounds=GeographicBounds(5.0, -135.0, 5.0, -135.0),
            latitude_step_degrees=1.0,
            longitude_step_degrees=1.0,
        )

        with self.assertRaises(FrozenInstanceError):
            result.method = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.rows[0].estimated_vtec_tecu = -1.0
        with self.assertRaises(FrozenInstanceError):
            result.rows[0].native_support[0].vtec_tecu = -1.0

    def test_generated_coordinates_use_parser_precision(self) -> None:
        inputs = tuple(
            observation(latitude, longitude, latitude)
            for latitude in (29.0, 30.0)
            for longitude in (-100.0, -95.0, -90.0)
        )
        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(29.1, -100.0, 29.3, -100.0),
            latitude_step_degrees=0.1,
            longitude_step_degrees=1.0,
        )

        self.assertEqual(
            [row.latitude_degrees for row in result.rows],
            [29.1, 29.2, 29.3],
        )

    def test_request_coordinates_enforce_precision_after_float_normalization(
        self,
    ) -> None:
        machine_noisy = interpolate_regional_grid(
            observations=native_grid(),
            bounds=GeographicBounds(
                0.1 + 0.2,
                -180.0,
                0.1 + 0.2,
                -180.0,
            ),
            latitude_step_degrees=0.1 + 0.2,
            longitude_step_degrees=1.0,
        )

        self.assertEqual(machine_noisy.bounds.south_latitude_degrees, 0.3)
        self.assertEqual(machine_noisy.requested_latitude_step_degrees, 0.3)
        self.assertEqual(machine_noisy.rows[0].latitude_degrees, 0.3)
        self.assertAlmostEqual(machine_noisy.rows[0].estimated_vtec_tecu, 0.6)

        for bounds in (
            GeographicBounds(0.00000000005, -180.0, 0.00000000055, -180.0),
            GeographicBounds(
                0.0,
                179.99999999995,
                0.0,
                -179.99999999995,
            ),
        ):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(
                    InterpolationRequestError,
                    "10-decimal coordinate precision",
                ):
                    interpolate_regional_grid(
                        observations=native_grid(),
                        bounds=bounds,
                        latitude_step_degrees=0.0000000001,
                        longitude_step_degrees=0.0000000001,
                    )

    def test_fractional_rounded_global_axes_are_recognized_as_cyclic(self) -> None:
        for longitude_count in (7, 11, 13):
            with self.subTest(longitude_count=longitude_count):
                longitudes = tuple(
                    round(-180.0 + index * 360.0 / longitude_count, 10)
                    for index in range(longitude_count)
                )
                inputs = tuple(
                    observation(latitude, longitude, 12.0)
                    for latitude in (0.0, 10.0)
                    for longitude in longitudes
                )
                result = interpolate_regional_grid(
                    observations=inputs,
                    bounds=GeographicBounds(0.0, 179.0, 0.0, -179.0),
                    latitude_step_degrees=1.0,
                    longitude_step_degrees=1.0,
                )

                self.assertTrue(result.native_grid.longitude_is_cyclic)
                self.assertAlmostEqual(
                    result.native_grid.longitude_step_degrees,
                    360.0 / longitude_count,
                    places=9,
                )
                self.assertEqual(
                    [row.longitude_degrees for row in result.rows],
                    [179.0, -180.0, -179.0],
                )
                self.assertEqual(
                    [row.estimated_vtec_tecu for row in result.rows],
                    [12.0, 12.0, 12.0],
                )

    def test_interpolation_remains_finite_for_opposite_extreme_values(self) -> None:
        inputs = tuple(
            observation(
                latitude,
                longitude,
                -1e308 if longitude == -180.0 else 1e308,
            )
            for latitude in (0.0, 10.0)
            for longitude in (-180.0, -90.0, 0.0, 90.0)
        )
        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(0.0, -135.0, 0.0, -135.0),
            latitude_step_degrees=1.0,
            longitude_step_degrees=1.0,
        )

        self.assertTrue(math.isfinite(result.rows[0].estimated_vtec_tecu))
        self.assertEqual(result.rows[0].estimated_vtec_tecu, 0.0)

    def test_distinct_coordinates_at_parser_precision_are_not_collapsed(self) -> None:
        inputs = tuple(
            observation(latitude, longitude, latitude * 10_000_000_000)
            for latitude in (0.0, 0.000000001)
            for longitude in (-180.0, -90.0, 0.0, 90.0)
        )
        result = interpolate_regional_grid(
            observations=inputs,
            bounds=GeographicBounds(
                0.0000000005,
                -180.0,
                0.0000000005,
                -180.0,
            ),
            latitude_step_degrees=0.0000000001,
            longitude_step_degrees=1.0,
        )

        self.assertFalse(result.rows[0].is_native_node)
        self.assertAlmostEqual(result.rows[0].estimated_vtec_tecu, 5.0)
        self.assertEqual(len(result.rows[0].native_support), 2)

        tiny_grid = interpolate_regional_grid(
            observations=tuple(
                observation(latitude, longitude, latitude)
                for latitude in (0.0, 0.00000001)
                for longitude in (-180.0, -90.0, 0.0, 90.0)
            ),
            bounds=GeographicBounds(0.0, -180.0, 0.00000001, -180.0),
            latitude_step_degrees=0.0000000001,
            longitude_step_degrees=1.0,
        )
        short_span = interpolate_regional_grid(
            observations=tuple(
                observation(latitude, longitude, latitude)
                for latitude in (0.0, 0.00000001)
                for longitude in (-180.0, -90.0, 0.0, 90.0)
            ),
            bounds=GeographicBounds(0.0, -180.0, 0.0000000005, -180.0),
            latitude_step_degrees=0.0000000001,
            longitude_step_degrees=1.0,
        )

        self.assertEqual(tiny_grid.latitude_count, 101)
        self.assertEqual(tiny_grid.rows[-1].latitude_degrees, 0.00000001)
        self.assertEqual(short_span.latitude_count, 6)
        self.assertEqual(short_span.rows[-1].latitude_degrees, 0.0000000005)


if __name__ == "__main__":
    unittest.main()
