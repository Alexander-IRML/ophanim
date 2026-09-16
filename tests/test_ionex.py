"""Focused tests for the dependency-free IONEX v1 parser."""

from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO, StringIO
import unittest

from ophanim.domain import SourceArtifact
from ophanim.ionex import (
    IONEXParseError,
    IONEXV1Parser,
    UnsupportedIONEXError,
)


def _record(payload: str, label: str) -> str:
    if len(payload) > 60 or len(label) > 20:
        raise ValueError("IONEX test record is wider than 80 columns")
    return f"{payload:<60}{label:<20}\n"


def _version_record(version: float = 1.0, file_type: str = "I") -> str:
    payload = f"{version:8.1f}{'':12}{file_type}{'':39}"
    return _record(payload, "IONEX VERSION / TYPE")


def _grid_payload(*values: float) -> str:
    return "  " + "".join(f"{value:6.1f}" for value in values)


def _data_record(*values: int) -> str:
    if len(values) > 16:
        raise ValueError("IONEX permits at most 16 I5 values per record")
    return "".join(f"{value:5d}" for value in values).ljust(80) + "\n"


def _header(
    *,
    map_count: int = 1,
    dimension: int = 2,
    height: tuple[float, float, float] = (450.0, 450.0, 0.0),
    latitude: tuple[float, float, float] = (32.5, 31.5, -1.0),
    longitude: tuple[float, float, float] = (-100.0, -98.0, 1.0),
    exponent: int | None = -1,
) -> str:
    lines = [
        _version_record(),
        _record(f"{map_count:6d}", "# OF MAPS IN FILE"),
        _record(f"{dimension:6d}", "MAP DIMENSION"),
        _record(_grid_payload(*height), "HGT1 / HGT2 / DHGT"),
        _record(_grid_payload(*latitude), "LAT1 / LAT2 / DLAT"),
        _record(_grid_payload(*longitude), "LON1 / LON2 / DLON"),
    ]
    if exponent is not None:
        lines.append(_record(f"{exponent:6d}", "EXPONENT"))
    lines.append(_record("", "END OF HEADER"))
    return "".join(lines)


def _row(latitude: float, *values: int) -> str:
    return _record(
        _grid_payload(latitude, -100.0, -98.0, 1.0, 450.0),
        "LAT/LON1/LON2/DLON/H",
    ) + _data_record(*values)


def _map(
    number: int,
    hour: int,
    *,
    first_row: tuple[int, ...],
    second_row: tuple[int, ...],
    exponent: int | None = None,
) -> str:
    lines = [
        _record(f"{number:6d}", "START OF TEC MAP"),
        _record(
            f"{2024:6d}{1:6d}{2:6d}{hour:6d}{0:6d}{0:6d}",
            "EPOCH OF CURRENT MAP",
        ),
    ]
    if exponent is not None:
        lines.append(_record(f"{exponent:6d}", "EXPONENT"))
    lines.extend(
        [
            _row(32.5, *first_row),
            _row(31.5, *second_row),
            _record(f"{number:6d}", "END OF TEC MAP"),
        ]
    )
    return "".join(lines)


class IONEXV1ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = IONEXV1Parser()
        self.artifact = SourceArtifact(
            artifact_id="artifact-1",
            provider="test-provider",
            product="test-gim",
            parser_version=self.parser.parser_version,
            revision_priority=0,
            checksum_sha256="0" * 64,
            storage_ref="memory://fixture",
            ingested_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

    def parse(self, source: str, *, binary: bool = True):
        stream = BytesIO(source.encode("ascii")) if binary else StringIO(source)
        return list(self.parser.parse(artifact=self.artifact, stream=stream))

    def test_rejects_artifact_parser_provenance_mismatch(self) -> None:
        artifact = replace(self.artifact, parser_version="ophanim-ionex-v1-parser/1")

        with self.assertRaisesRegex(IONEXParseError, "parser_version does not match"):
            list(self.parser.parse(artifact=artifact, stream=StringIO("")))

    def test_parses_fixed_width_rows_epochs_exponents_and_missing_cells(self) -> None:
        source = (
            _header(map_count=2)
            + _map(
                1,
                0,
                first_row=(100, 9999, -20),
                second_row=(30, 40, 50),
            )
            + _map(
                2,
                2,
                exponent=-2,
                first_row=(100, 200, 300),
                second_row=(400, 500, 600),
            )
            + _record("1", "START OF RMS MAP")
            + _record("RMS CONTENT IS IGNORED", "COMMENT")
            + _record("1", "END OF RMS MAP")
            + _record("", "END OF FILE")
        )

        observations = self.parse(source)

        self.assertEqual(len(observations), 11)
        self.assertEqual(
            observations[0].observed_at,
            datetime(2024, 1, 2, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(observations[0].latitude_degrees, 32.5)
        self.assertEqual(observations[0].longitude_degrees, -100.0)
        self.assertAlmostEqual(observations[0].vtec_tecu, 10.0)

        # The raw 9999 cell at longitude -99 is omitted, not scaled or emitted.
        first_epoch_points = [
            item
            for item in observations
            if item.observed_at.hour == 0 and item.latitude_degrees == 32.5
        ]
        self.assertEqual(
            [item.longitude_degrees for item in first_epoch_points],
            [-100.0, -98.0],
        )
        self.assertAlmostEqual(first_epoch_points[-1].vtec_tecu, -2.0)

        second_epoch = [item for item in observations if item.observed_at.hour == 2]
        self.assertEqual(
            [round(item.vtec_tecu, 6) for item in second_epoch],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        self.assertTrue(
            all(item.artifact_id == self.artifact.artifact_id for item in observations)
        )

    def test_uses_default_exponent_and_accepts_text_streams(self) -> None:
        source = (
            _header(exponent=None)
            + _map(
                1,
                0,
                first_row=(10, 20, 30),
                second_row=(40, 50, 60),
            )
            + _record("", "END OF FILE")
        )

        observations = self.parse(source, binary=False)

        self.assertEqual(
            [item.vtec_tecu for item in observations],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def test_reads_a_row_across_multiple_data_records(self) -> None:
        source = (
            _header(latitude=(32.5, 32.5, 0.0))
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, -100.0, -98.0, 1.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20)
            + _data_record(30)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        observations = self.parse(source)

        self.assertEqual([item.vtec_tecu for item in observations], [1.0, 2.0, 3.0])

    def test_normalizes_zero_to_360_longitudes_for_domain_consumers(self) -> None:
        source = (
            _header(
                latitude=(32.5, 32.5, 0.0),
                longitude=(255.0, 257.0, 1.0),
            )
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, 255.0, 257.0, 1.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20, 30)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        observations = self.parse(source)

        self.assertEqual(
            [item.longitude_degrees for item in observations],
            [-105.0, -104.0, -103.0],
        )

    def test_deduplicates_identical_zero_and_360_endpoint_values(self) -> None:
        source = (
            _header(
                latitude=(32.5, 32.5, 0.0),
                longitude=(0.0, 360.0, 120.0),
            )
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, 0.0, 360.0, 120.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20, 30, 10)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        observations = self.parse(source)

        self.assertEqual(len(observations), 3)
        self.assertEqual(
            [item.longitude_degrees for item in observations],
            [0.0, 120.0, -120.0],
        )
        self.assertEqual(
            [item.vtec_tecu for item in observations],
            [1.0, 2.0, 3.0],
        )

    def test_rejects_conflicting_zero_and_360_endpoint_values(self) -> None:
        source = (
            _header(
                latitude=(32.5, 32.5, 0.0),
                longitude=(0.0, 360.0, 120.0),
            )
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, 0.0, 360.0, 120.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20, 30, 11)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        with self.assertRaisesRegex(
            IONEXParseError,
            r"conflicting TEC values at equivalent longitude endpoints "
            r"0 and 360 .* at line \d+",
        ):
            self.parse(source)

    def test_deduplicates_identical_negative_and_positive_180_endpoints(
        self,
    ) -> None:
        source = (
            _header(
                latitude=(32.5, 32.5, 0.0),
                longitude=(-180.0, 180.0, 180.0),
            )
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, -180.0, 180.0, 180.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20, 10)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        observations = self.parse(source)

        self.assertEqual(
            [item.longitude_degrees for item in observations],
            [-180.0, 0.0],
        )

    def test_rejects_conflicting_negative_and_positive_180_endpoints(
        self,
    ) -> None:
        source = (
            _header(
                latitude=(32.5, 32.5, 0.0),
                longitude=(-180.0, 180.0, 180.0),
            )
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, -180.0, 180.0, 180.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20, 11)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        with self.assertRaisesRegex(
            IONEXParseError,
            r"equivalent longitude endpoints -180 and 180",
        ):
            self.parse(source)

    def test_exponent_can_change_between_data_blocks(self) -> None:
        source = (
            _header()
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _row(32.5, 10, 20, 30)
            + _record("-2", "EXPONENT")
            + _row(31.5, 100, 200, 300)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        observations = self.parse(source)

        self.assertEqual(
            [item.vtec_tecu for item in observations],
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )

    def test_rejects_incomplete_longitude_row_with_line_number(self) -> None:
        source = (
            _header(latitude=(32.5, 32.5, 0.0))
            + _record("1", "START OF TEC MAP")
            + _record("2024 1 2 0 0 0", "EPOCH OF CURRENT MAP")
            + _record(
                _grid_payload(32.5, -100.0, -98.0, 1.0, 450.0),
                "LAT/LON1/LON2/DLON/H",
            )
            + _data_record(10, 20)
            + _record("1", "END OF TEC MAP")
            + _record("", "END OF FILE")
        )

        with self.assertRaisesRegex(
            IONEXParseError, r"TEC row is incomplete.*at line \d+"
        ):
            self.parse(source)

    def test_rejects_three_dimensional_maps_without_mislabeling_them_as_vtec(
        self,
    ) -> None:
        source = _header(
            dimension=3,
            height=(200.0, 800.0, 50.0),
        ) + _record("", "END OF FILE")

        with self.assertRaisesRegex(
            UnsupportedIONEXError, "only two-dimensional"
        ):
            self.parse(source)

    def test_rejects_map_count_mismatch(self) -> None:
        source = (
            _header(map_count=2)
            + _map(
                1,
                0,
                first_row=(10, 20, 30),
                second_row=(40, 50, 60),
            )
            + _record("", "END OF FILE")
        )

        with self.assertRaisesRegex(IONEXParseError, "map count does not match"):
            self.parse(source)


if __name__ == "__main__":
    unittest.main()
