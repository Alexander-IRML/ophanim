"""Unit tests for the PostgreSQL TEC catalog without a database server."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import unittest

from ophanim.postgres_catalog import (
    POSTGRES_SCHEMA_VERSION,
    PostgresTECCatalog,
)
from ophanim.tec_archive import (
    CORE_TEC_GRID_DEFINITION_SOURCE,
    CORE_TEC_GRID_DERIVATION_METHOD,
    CORE_TEC_GRID_LAYOUT_VERSION,
    INTERPOLATED_VALUE_KIND,
    TECArchiveConflictError,
    TECArchiveError,
    TECEpochSummary,
    TECGridManifest,
    core_gim_grid_definition,
)


EPOCH = datetime(2026, 9, 11, tzinfo=UTC)
GRID_SET_ID = sha256(
    b"artifact-1\0ophanim-tec-grid/1"
).hexdigest()


def manifest() -> TECGridManifest:
    return TECGridManifest(
        grid_set_id=GRID_SET_ID,
        artifact_id="artifact-1",
        provider="code",
        product="gim",
        parser_version="ophanim-ionex-v1-parser/2",
        revision_priority=10,
        revision="rapid",
        source_uri="https://example.test/source.inx.gz",
        source_checksum_sha256="a" * 64,
        source_storage_ref="file:///archive/source.inx.gz",
        source_ingested_at=EPOCH,
        storage_ref="file:///zarr/grid.zarr",
        layout_version="ophanim-tec-grid/1",
        zarr_format=3,
        grid_fingerprint="b" * 64,
        logical_sha256="c" * 64,
        shape=(2, 2, 3),
        chunk_shape=(1, 2, 3),
        vtec_dtype="float32",
        quality_mask_dtype="uint32",
        quality_flag_bits=(("edge", 0),),
        definition_source="test-grid/1",
        shell_height_km=450.0,
        archived_at=EPOCH + timedelta(hours=3),
        epochs=(
            TECEpochSummary(
                time_index=0,
                observed_at=EPOCH + timedelta(hours=1),
                valid_cell_count=5,
                missing_cell_count=1,
                minimum_vtec_tecu=10.0,
                maximum_vtec_tecu=12.0,
                mean_vtec_tecu=11.0,
            ),
            TECEpochSummary(
                time_index=1,
                observed_at=EPOCH + timedelta(hours=2),
                valid_cell_count=6,
                missing_cell_count=0,
                minimum_vtec_tecu=11.0,
                maximum_vtec_tecu=13.0,
                mean_vtec_tecu=12.0,
            ),
        ),
    )


def core_manifest(source: TECGridManifest) -> TECGridManifest:
    core_grid = core_gim_grid_definition(
        shell_height_km=source.shell_height_km
    )
    return replace(
        source,
        grid_set_id=sha256(
            f"{source.artifact_id}\0{CORE_TEC_GRID_LAYOUT_VERSION}".encode()
        ).hexdigest(),
        layout_version=CORE_TEC_GRID_LAYOUT_VERSION,
        storage_ref="file:///zarr/core-grid.zarr",
        grid_fingerprint=core_grid.fingerprint,
        logical_sha256="e" * 64,
        shape=(len(source.epochs), 351, 720),
        chunk_shape=(1, 256, 256),
        definition_source=CORE_TEC_GRID_DEFINITION_SOURCE,
        epochs=tuple(
            replace(
                epoch,
                valid_cell_count=252_720,
                missing_cell_count=0,
            )
            for epoch in source.epochs
        ),
        value_kind=INTERPOLATED_VALUE_KIND,
        source_grid_set_id=source.grid_set_id,
        source_grid_logical_sha256=source.logical_sha256,
        derivation_method=CORE_TEC_GRID_DERIVATION_METHOD,
    )


class _Result:
    def __init__(self, row=None) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, database: "_FakePool") -> None:
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def transaction(self):
        return nullcontext()

    def cursor(self):
        return self

    def execute(self, statement, parameters=None, **options):
        sql = " ".join(statement.split())
        self.database.calls.append((sql, parameters, options))
        if "SELECT COALESCE(max(version), 0)" in sql:
            return _Result((self.database.schema_version,))
        if "INSERT INTO ophanim.schema_migrations" in sql:
            self.database.schema_version = int(parameters[0])
        elif "SELECT logical_sha256, manifest" in sql:
            record = self.database.grids.get(parameters[0])
            if record is None:
                return _Result()
            return _Result((record["logical_sha256"], record["manifest"]))
        elif "SELECT artifact_id, logical_sha256, value_kind" in sql:
            record = self.database.grids.get(parameters[0])
            if record is None:
                return _Result()
            return _Result(
                (
                    record["artifact_id"],
                    record["logical_sha256"],
                    record["value_kind"],
                )
            )
        elif "INSERT INTO ophanim.source_artifacts" in sql:
            artifact_id = parameters[0]
            self.database.sources.setdefault(artifact_id, tuple(parameters[1:]))
        elif "SELECT provider, product, parser_version" in sql:
            return _Result(self.database.sources.get(parameters[0]))
        elif "INSERT INTO ophanim.tec_grid_sets" in sql:
            self.database.grids[parameters[0]] = {
                "artifact_id": parameters[1],
                "layout_version": parameters[2],
                "logical_sha256": parameters[6],
                "value_kind": parameters[7],
                "source_grid_set_id": parameters[8],
                "source_grid_logical_sha256": parameters[9],
                "derivation_method": parameters[10],
                "time_count": parameters[11],
                "valid_cell_count": parameters[21],
                "missing_cell_count": parameters[22],
                "manifest": json.loads(parameters[24]),
            }
        elif "SELECT manifest FROM ophanim.tec_grid_sets" in sql:
            artifact_id, layout_version = parameters
            for record in self.database.grids.values():
                if (
                    record["artifact_id"] == artifact_id
                    and record["layout_version"] == layout_version
                ):
                    return _Result((record["manifest"],))
            return _Result()
        elif "SELECT count(*) FROM ophanim.source_artifacts" in sql:
            return _Result(
                (
                    len(self.database.sources),
                    len(self.database.grids),
                    sum(row["time_count"] for row in self.database.grids.values()),
                    sum(
                        row["valid_cell_count"]
                        for row in self.database.grids.values()
                    ),
                    sum(
                        row["missing_cell_count"]
                        for row in self.database.grids.values()
                    ),
                )
            )
        return _Result()

    def executemany(self, statement, rows) -> None:
        sql = " ".join(statement.split())
        materialized = list(rows)
        self.database.calls.append((sql, materialized, {}))
        self.database.epochs.extend(materialized)


class _FakePool:
    def __init__(self, *, schema_version: int = 0) -> None:
        self.schema_version = schema_version
        self.sources = {}
        self.grids = {}
        self.epochs = []
        self.calls = []
        self.waited = []
        self.closed = False
        self._connection = _FakeConnection(self)

    def connection(self):
        return self._connection

    def wait(self, *, timeout) -> None:
        self.waited.append(timeout)

    def close(self) -> None:
        self.closed = True


class PostgresTECCatalogTests(unittest.TestCase):
    def test_schema_initialization_is_migrated_once_and_wait_is_forwarded(self) -> None:
        pool = _FakePool()
        catalog = PostgresTECCatalog(pool=pool)

        catalog.wait(4.5)
        catalog.initialize_schema()
        calls_after_first_initialization = len(pool.calls)
        catalog.initialize_schema()

        self.assertEqual(pool.waited, [4.5])
        self.assertEqual(pool.schema_version, POSTGRES_SCHEMA_VERSION)
        self.assertEqual(len(pool.calls), calls_after_first_initialization)
        migration_calls = [
            call for call in pool.calls if call[2] == {"prepare": False}
        ]
        self.assertEqual(len(migration_calls), 2)
        self.assertIn("CREATE TABLE ophanim.tec_grid_sets", migration_calls[0][0])
        self.assertIn("ALTER TABLE ophanim.tec_grid_sets", migration_calls[1][0])

    def test_schema_initialization_upgrades_an_existing_v1_catalog(self) -> None:
        pool = _FakePool(schema_version=1)

        PostgresTECCatalog(pool=pool).initialize_schema()

        self.assertEqual(pool.schema_version, POSTGRES_SCHEMA_VERSION)
        migration_calls = [
            call for call in pool.calls if call[2] == {"prepare": False}
        ]
        self.assertEqual(len(migration_calls), 1)
        self.assertIn("ADD COLUMN value_kind", migration_calls[0][0])
        self.assertNotIn("CREATE TABLE ophanim.source_artifacts", migration_calls[0][0])

    def test_schema_initialization_rejects_an_unknown_version(self) -> None:
        catalog = PostgresTECCatalog(pool=_FakePool(schema_version=99))

        with self.assertRaisesRegex(RuntimeError, "unsupported.*schema version"):
            catalog.initialize_schema()

    def test_publish_lookup_statistics_and_retry_are_idempotent(self) -> None:
        pool = _FakePool()
        catalog = PostgresTECCatalog(pool=pool)
        candidate = manifest()

        first = catalog.publish(candidate)
        second = catalog.publish(candidate)
        restored = catalog.get_for_artifact(
            candidate.artifact_id,
            layout_version=candidate.layout_version,
        )

        self.assertFalse(first.already_present)
        self.assertTrue(second.already_present)
        self.assertEqual(second.manifest, candidate)
        self.assertEqual(restored, candidate)
        self.assertEqual(len(pool.sources), 1)
        self.assertEqual(len(pool.grids), 1)
        self.assertEqual(len(pool.epochs), 2)
        self.assertEqual(
            pool.grids[candidate.grid_set_id]["value_kind"],
            "native_measurement",
        )
        self.assertIsNone(pool.grids[candidate.grid_set_id]["source_grid_set_id"])
        self.assertEqual(
            catalog.statistics().to_dict(),
            {
                "artifact_count": 1,
                "grid_set_count": 1,
                "epoch_count": 2,
                "valid_cell_count": 11,
                "missing_cell_count": 1,
            },
        )

    def test_publish_persists_derived_grid_lineage_columns(self) -> None:
        pool = _FakePool()
        catalog = PostgresTECCatalog(pool=pool)
        source = manifest()
        derived = core_manifest(source)

        catalog.publish(source)
        result = catalog.publish(derived)

        self.assertFalse(result.already_present)
        row = pool.grids[derived.grid_set_id]
        self.assertEqual(row["value_kind"], INTERPOLATED_VALUE_KIND)
        self.assertEqual(row["source_grid_set_id"], source.grid_set_id)
        self.assertEqual(
            row["source_grid_logical_sha256"],
            source.logical_sha256,
        )
        self.assertEqual(
            row["derivation_method"],
            CORE_TEC_GRID_DERIVATION_METHOD,
        )

    def test_derived_publish_requires_a_matching_native_parent(self) -> None:
        source = manifest()
        derived = core_manifest(source)

        missing_pool = _FakePool()
        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "source grid is not published",
        ):
            PostgresTECCatalog(pool=missing_pool).publish(derived)
        self.assertNotIn(derived.grid_set_id, missing_pool.grids)

        mismatches = (
            (
                {"artifact_id": "different-artifact"},
                "different artifact",
            ),
            (
                {"logical_sha256": "f" * 64},
                "different logical checksum",
            ),
            (
                {"value_kind": INTERPOLATED_VALUE_KIND},
                "not a native measurement grid",
            ),
        )
        for changes, message in mismatches:
            with self.subTest(message=message):
                pool = _FakePool()
                catalog = PostgresTECCatalog(pool=pool)
                catalog.publish(source)
                pool.grids[source.grid_set_id].update(changes)

                with self.assertRaisesRegex(TECArchiveConflictError, message):
                    catalog.publish(derived)
                self.assertNotIn(derived.grid_set_id, pool.grids)

    def test_grid_and_source_identity_conflicts_fail_closed(self) -> None:
        pool = _FakePool()
        catalog = PostgresTECCatalog(pool=pool)
        candidate = manifest()
        catalog.publish(candidate)

        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "different logical content",
        ):
            catalog.publish(replace(candidate, logical_sha256="d" * 64))
        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "different immutable metadata",
        ):
            catalog.publish(
                replace(
                    candidate,
                    archived_at=candidate.archived_at + timedelta(seconds=1),
                )
            )

        conflicting_pool = _FakePool()
        conflicting_pool.sources[candidate.artifact_id] = (
            candidate.provider,
            candidate.product,
            candidate.parser_version,
            candidate.revision_priority,
            candidate.revision,
            "https://example.test/different.inx.gz",
            candidate.source_checksum_sha256,
            candidate.source_storage_ref,
            candidate.source_ingested_at,
        )
        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "different source provenance",
        ):
            PostgresTECCatalog(pool=conflicting_pool).publish(candidate)

    def test_supplied_pool_lifetime_remains_with_caller(self) -> None:
        pool = _FakePool()

        PostgresTECCatalog(pool=pool).close()

        self.assertFalse(pool.closed)


class TECGridManifestTests(unittest.TestCase):
    def test_manifest_invariants_reject_inconsistent_catalog_records(self) -> None:
        candidate = manifest()
        scenarios = (
            (
                {"epochs": candidate.epochs[:1]},
                "epoch count",
            ),
            (
                {"chunk_shape": (3, 2, 3)},
                "chunks cannot exceed",
            ),
            (
                {"quality_flag_bits": (("edge", 1),)},
                "deterministic bit positions",
            ),
            (
                {"grid_set_id": "d" * 64},
                "does not match its artifact",
            ),
        )
        for changes, message in scenarios:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TECArchiveError, message):
                    replace(candidate, **changes)

        payload = candidate.to_dict()
        payload["shape"] = ["2", 2, 3]
        with self.assertRaisesRegex(TECArchiveError, "positive integers"):
            TECGridManifest.from_dict(payload)

        derived = core_manifest(candidate)
        for changes in (
            {"definition_source": "not-the-core-grid"},
            {"grid_fingerprint": "f" * 64},
            {"shape": (2, 350, 720)},
        ):
            with self.subTest(core_contract=changes):
                with self.assertRaisesRegex(
                    TECArchiveError,
                    "canonical 0.5-degree axes",
                ):
                    replace(derived, **changes)


if __name__ == "__main__":
    unittest.main()
