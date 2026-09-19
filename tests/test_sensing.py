"""Native sensing contracts: provenance, causal revisions, masks and source RMS."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ophanim.artifacts import FilesystemArtifactStore
from ophanim.domain import SourceArtifact, TECObservation
from ophanim.ionex import IONEXParseError, IONEXV1Parser
from ophanim.sensing import (
    SensingError, parse_ionex_rms, read_archive_sequence,
    read_desktop_sequence, read_ionex_sequence,
)
from ophanim.tec_archive import TECArchiveConflictError, TECGridDefinition, ZarrTECGridStore


HAS_ARRAYS = all(find_spec(name) is not None for name in ("numpy", "xarray", "zarr"))
EPOCH = datetime(2024, 1, 2, tzinfo=UTC)


def record(payload, label):
    return f"{payload:<60}{label:<20}\n"


def grid_payload(*values):
    return "  " + "".join(f"{value:6.1f}" for value in values)


def fixture(*, rms=True, rms_values=(10, 9999, 30), rms_epoch="2024 1 2 0 0 0"):
    header = (
        record(f"{1.0:8.1f}{'':12}I", "IONEX VERSION / TYPE")
        + record("1", "# OF MAPS IN FILE")
        + record("2", "MAP DIMENSION")
        + record(grid_payload(450, 450, 0), "HGT1 / HGT2 / DHGT")
        + record(grid_payload(30, 32.5, 2.5), "LAT1 / LAT2 / DLAT")
        + record(grid_payload(-100, -90, 5), "LON1 / LON2 / DLON")
        + record("-1", "EXPONENT")
        + record("", "END OF HEADER")
    )
    def block(kind, values, epoch):
        parts = [record("1", f"START OF {kind} MAP"), record(epoch, "EPOCH OF CURRENT MAP")]
        for latitude in (30, 32.5):
            parts.append(record(grid_payload(latitude, -100, -90, 5, 450), "LAT/LON1/LON2/DLON/H"))
            parts.append("".join(f"{value:5d}" for value in values).ljust(80) + "\n")
        parts.append(record("1", f"END OF {kind} MAP"))
        return "".join(parts)
    text = header + block("TEC", (100, 200, 9999), "2024 1 2 0 0 0")
    if rms:
        text += block("RMS", rms_values, rms_epoch)
    return (text + record("", "END OF FILE")).encode("ascii")


def artifact(raw=None, *, identity="source-1", priority=10, ingested=None):
    return SourceArtifact(
        artifact_id=identity, provider="code", product="gim",
        parser_version=IONEXV1Parser.parser_version, revision_priority=priority,
        checksum_sha256=sha256(raw or b"fixture").hexdigest(),
        storage_ref="fixture.inx", ingested_at=ingested or EPOCH + timedelta(days=1),
    )


class SourceRMSTests(unittest.TestCase):
    def test_rms_is_scaled_and_missing_values_are_unknown(self):
        raw = fixture()
        values = parse_ionex_rms(raw, artifact=artifact(raw))
        self.assertEqual(values[(EPOCH, 30, -100)], 1.0)
        self.assertEqual(values[(EPOCH, 32.5, -90)], 3.0)
        self.assertNotIn((EPOCH, 30, -95), values)
        self.assertEqual(IONEXV1Parser.parser_version, "ophanim-ionex-v1-parser/2")

    def test_absent_rms_is_an_empty_mapping(self):
        raw = fixture(rms=False)
        self.assertEqual(parse_ionex_rms(raw, artifact=artifact(raw)), {})

    def test_rms_exponent_override_is_preserved(self):
        raw = fixture().replace(record("1", "START OF RMS MAP").encode(),
                                (record("1", "START OF RMS MAP") + record("-2", "EXPONENT")).encode())
        self.assertEqual(parse_ionex_rms(raw, artifact=artifact(raw))[(EPOCH, 30, -100)], 0.1)

    def test_malformed_rms_is_rejected_not_silently_discarded(self):
        cases = (
            fixture(rms_values=(-10, 20, 30)),
            fixture(rms_epoch="2024 1 2 2 0 0"),
            fixture().replace(record("1", "END OF RMS MAP").encode(), b""),
            fixture().replace(record("1", "END OF RMS MAP").encode(), record("2", "END OF RMS MAP").encode()),
        )
        for raw in cases:
            with self.subTest(raw=raw[-100:]):
                with self.assertRaises(IONEXParseError):
                    parse_ionex_rms(raw, artifact=artifact(raw))

    def test_source_bytes_must_match_provenance(self):
        with self.assertRaisesRegex(SensingError, "checksum"):
            parse_ionex_rms(fixture(), artifact=artifact())


@unittest.skipUnless(HAS_ARRAYS, "optional sensing array dependencies unavailable")
class SensingSequenceTests(unittest.TestCase):
    def test_raw_source_keeps_native_axes_masks_rms_and_source_availability(self):
        import numpy as np
        raw = fixture()
        result = read_ionex_sequence(raw, artifact=artifact(raw))
        self.assertEqual(result.tec.dims, ("time", "lat", "lon"))
        self.assertEqual(result.tec.shape, (1, 2, 3))
        self.assertEqual(result.tec.values[0, 0, 0], 10)
        self.assertFalse(result.observed_mask.values[0, 0, 2])
        self.assertTrue(np.isnan(result.source_rms_tecu.values[0, 0, 1]))
        self.assertTrue(np.isnan(result.source_rms_tecu.values[0, 0, 2]))
        self.assertEqual(result.source_rms_tecu.values[0, 0, 0], 1)
        self.assertEqual(result.attrs["source_metadata"]["native_lat_spacing_deg"], 2.5)
        self.assertEqual(result.attrs["source_metadata"]["native_lon_spacing_deg"], 5)
        self.assertIsNone(result.attrs["source_metadata"]["effective_resolution_m"])
        self.assertEqual(result.available_at.values[0], np.datetime64("2024-01-03"))
        self.assertIn("not a direct instrument", result.observed_mask.attrs["semantic"])

    def test_causal_time_must_be_explicit_and_exclude_unavailable_data(self):
        raw = fixture()
        for cutoff in (datetime(2024, 1, 3), "2024-01-03"):
            with self.assertRaisesRegex(SensingError, "timezone-aware"):
                read_ionex_sequence(raw, artifact=artifact(raw), as_of=cutoff)
        with self.assertRaisesRegex(SensingError, "unavailable"):
            read_ionex_sequence(raw, artifact=artifact(raw), as_of=EPOCH)
        result = read_ionex_sequence(raw, artifact=artifact(raw), as_of="2024-01-03T00:00:00Z")
        self.assertEqual(result.attrs["analysis_mode"], "causal")

    def _write(self, store, source, value=10, flags=("edge",)):
        definition = TECGridDefinition((30.0, 32.5), (-100.0, -95.0, -90.0), "sensing-test/1")
        observations = [TECObservation(source.artifact_id, EPOCH + timedelta(hours=hour), lat, lon, value,
                                       quality_flags=flags)
                        for hour in (0, 2) for lat in definition.latitudes_degrees
                        for lon in definition.longitudes_degrees if (lat, lon) != (32.5, -90)]
        return store.write(artifact=source, observations=observations, grid=definition).manifest

    def test_archive_revision_selection_is_deterministic_and_causal(self):
        import numpy as np
        with TemporaryDirectory() as directory:
            store = ZarrTECGridStore(directory)
            early = self._write(store, artifact(identity="early"), value=10)
            later = self._write(store, artifact(identity="later", priority=20,
                                               ingested=EPOCH + timedelta(days=2)), value=30)
            ids = [later.grid_set_id, early.grid_set_id]
            historical = read_archive_sequence(directory, ids, as_of=EPOCH + timedelta(days=1))
            current = read_archive_sequence(directory, list(reversed(ids)))
            self.assertEqual(historical.tec.values[0, 0, 0], 10)
            self.assertEqual(current.tec.values[0, 0, 0], 30)
            self.assertEqual(current.attrs["snapshot_id"], read_archive_sequence(directory, ids).attrs["snapshot_id"])
            self.assertEqual(historical.attrs["source_metadata"]["source_ids"], [early.grid_set_id])
            self.assertEqual(current.attrs["source_metadata"]["cadence_seconds"], 7200)
            self.assertTrue(np.isnan(current.tec.values[0, 1, 2]))
            self.assertTrue(np.isnan(current.source_rms_tecu.values).all())
            self.assertEqual(current.quality_mask.values[0, 0, 0], 1)

    def test_archive_read_checks_arrays_and_ignores_derived_discovery(self):
        import zarr
        with TemporaryDirectory() as directory:
            store = ZarrTECGridStore(directory)
            source = self._write(store, artifact())
            derived_id = "b" * 64
            derived = Path(directory) / "derived" / "tec-grid" / "half-degree" / "v1" / f"{derived_id}.zarr"
            derived.mkdir(parents=True)
            self.assertEqual(read_archive_sequence(directory).attrs["source_metadata"]["native_lon_spacing_deg"], 5)
            with self.assertRaisesRegex(SensingError, "native source grid"):
                read_archive_sequence(directory, [derived_id])
            path = Path(directory) / "raw" / "tec-grid" / "v1" / f"{source.grid_set_id}.zarr"
            group = zarr.open_group(str(path), mode="r+")
            group["vtec"][0, 0, 0] = 99
            with self.assertRaises(TECArchiveConflictError):
                read_archive_sequence(directory)

    def test_bounds_do_not_turn_cropped_spacing_into_fine_resolution(self):
        raw = fixture()
        result = read_ionex_sequence(raw, artifact=artifact(raw), bounds={"south":30,"north":30,"west":-100,"east":-100})
        self.assertEqual(result.tec.shape, (1, 1, 1))
        self.assertEqual(result.attrs["source_metadata"]["native_lat_spacing_deg"], 2.5)
        with self.assertRaisesRegex(SensingError, "no native source nodes"):
            read_ionex_sequence(raw, artifact=artifact(raw), bounds={"south":0,"north":1,"west":0,"east":1})

    def test_missing_archive_root_does_not_create_a_store(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "absent"
            with self.assertRaisesRegex(SensingError, "does not exist"):
                read_archive_sequence(root)
            self.assertFalse(root.exists())

    def test_desktop_reads_original_artifacts_without_model_or_archive(self):
        with TemporaryDirectory() as directory:
            raw = fixture()
            store = FilesystemArtifactStore(Path(directory) / "artifacts")
            stored = store.put(raw, suffix=".inx")
            early = replace(artifact(raw), storage_ref=stored.storage_ref)
            late = replace(early, artifact_id="late", revision_priority=20, ingested_at=EPOCH + timedelta(days=2))
            database = Path(directory) / "ophanim.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE source_artifacts (artifact_id TEXT, provider TEXT, product TEXT, parser_version TEXT, revision_priority INTEGER, checksum_sha256 TEXT, storage_ref TEXT, ingested_at TEXT, revision TEXT, source_uri TEXT)")
                for source in (early, late):
                    connection.execute("INSERT INTO source_artifacts VALUES (?,?,?,?,?,?,?,?,?,?)", (
                        source.artifact_id, source.provider, source.product, source.parser_version,
                        source.revision_priority, source.checksum_sha256, source.storage_ref,
                        source.ingested_at.isoformat(), source.revision, source.source_uri,
                    ))
            result = read_desktop_sequence(directory, as_of=EPOCH + timedelta(days=1))
            self.assertEqual(result.source_id.values.tolist(), [early.artifact_id])
            self.assertEqual(result.source_rms_tecu.values[0, 0, 0], 1)
            current = read_desktop_sequence(directory)
            self.assertEqual(current.source_id.values.tolist(), [late.artifact_id])
            with self.assertRaisesRegex(SensingError, "absent"):
                read_desktop_sequence(directory, artifact_ids=["unknown"])


if __name__ == "__main__":
    unittest.main()
