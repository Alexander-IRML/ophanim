"""Tests for immutable dense TEC grid archiving."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.util import find_spec
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ophanim.domain import SourceArtifact, TECObservation
from ophanim.tec_archive import (
    BulkTECArchive,
    CatalogPublishResult,
    CORE_TEC_GRID_DEFINITION_SOURCE,
    CORE_TEC_GRID_DERIVATION_METHOD,
    CORE_TEC_GRID_LAYOUT_VERSION,
    TECArchiveConflictError,
    TECArchiveError,
    TECGridDefinition,
    TECGridManifest,
    TECGridWriteResult,
    ZarrTECGridStore,
    core_gim_grid_definition,
    standard_gim_grid_definition,
)


HAS_ARRAY_DEPENDENCIES = (
    find_spec("numpy") is not None and find_spec("zarr") is not None
)
EPOCH = datetime(2026, 9, 11, tzinfo=UTC)


def artifact() -> SourceArtifact:
    return SourceArtifact(
        artifact_id="artifact-grid-1",
        provider="code",
        product="gim",
        parser_version="ophanim-ionex-v1-parser/2",
        revision_priority=10,
        checksum_sha256="a" * 64,
        storage_ref="artifacts/source.inx.gz",
        ingested_at=EPOCH + timedelta(days=1),
        revision="rapid-2026-09-11",
        source_uri="https://example.test/source.inx.gz",
    )


def grid() -> TECGridDefinition:
    return TECGridDefinition(
        latitudes_degrees=(0.0, 2.5),
        longitudes_degrees=(-180.0, -175.0, -170.0),
        definition_source="unit-test-grid/1",
        shell_height_km=450.0,
    )


def observations() -> tuple[TECObservation, ...]:
    rows = []
    for epoch_offset in (0, 1):
        observed_at = EPOCH + timedelta(hours=epoch_offset)
        for latitude in grid().latitudes_degrees:
            for longitude in grid().longitudes_degrees:
                if epoch_offset == 0 and (latitude, longitude) == (2.5, -170.0):
                    continue
                rows.append(
                    TECObservation(
                        artifact_id=artifact().artifact_id,
                        observed_at=observed_at,
                        latitude_degrees=latitude,
                        longitude_degrees=longitude,
                        vtec_tecu=10.0 + epoch_offset + latitude,
                        quality_flags=("edge",) if longitude == -180.0 else (),
                    )
                )
    return tuple(rows)


def standard_observations(
    *,
    omit: tuple[float, float] | None = None,
) -> tuple[TECObservation, ...]:
    """One standard global epoch with a deliberately non-periodic seam."""

    definition = standard_gim_grid_definition()
    return tuple(
        TECObservation(
            artifact_id=artifact().artifact_id,
            observed_at=EPOCH,
            latitude_degrees=latitude,
            longitude_degrees=longitude,
            vtec_tecu=10.0 * latitude_index + longitude_index,
        )
        for latitude_index, latitude in enumerate(definition.latitudes_degrees)
        for longitude_index, longitude in enumerate(
            definition.longitudes_degrees
        )
        if (latitude, longitude) != omit
    )


class TECGridDefinitionTests(unittest.TestCase):
    def test_standard_grid_is_canonical_and_fingerprinted(self) -> None:
        definition = standard_gim_grid_definition()

        self.assertEqual(len(definition.latitudes_degrees), 71)
        self.assertEqual(len(definition.longitudes_degrees), 72)
        self.assertEqual(definition.latitudes_degrees[0], -87.5)
        self.assertEqual(definition.latitudes_degrees[-1], 87.5)
        self.assertEqual(definition.longitudes_degrees[0], -180.0)
        self.assertEqual(definition.longitudes_degrees[-1], 175.0)
        self.assertEqual(len(definition.fingerprint), 64)

    def test_core_grid_is_half_degree_and_stays_inside_native_latitudes(self) -> None:
        definition = core_gim_grid_definition()

        self.assertEqual(
            definition.definition_source,
            CORE_TEC_GRID_DEFINITION_SOURCE,
        )
        self.assertEqual(len(definition.latitudes_degrees), 351)
        self.assertEqual(len(definition.longitudes_degrees), 720)
        self.assertEqual(definition.latitudes_degrees[0], -87.5)
        self.assertEqual(definition.latitudes_degrees[-1], 87.5)
        self.assertEqual(definition.longitudes_degrees[0], -180.0)
        self.assertEqual(definition.longitudes_degrees[-1], 179.5)
        self.assertEqual(
            definition.latitudes_degrees[1]
            - definition.latitudes_degrees[0],
            0.5,
        )
        self.assertEqual(
            definition.longitudes_degrees[1] - definition.longitudes_degrees[0],
            0.5,
        )
        self.assertEqual(
            len(definition.latitudes_degrees)
            * len(definition.longitudes_degrees),
            252_720,
        )
        self.assertNotIn(-90.0, definition.latitudes_degrees)
        self.assertNotIn(90.0, definition.latitudes_degrees)

    def test_axes_must_be_explicit_canonical_and_sorted(self) -> None:
        scenarios = (
            ((1.0, 0.0), (-180.0,), "latitude axis"),
            ((0.0,), (-180.0, -180.0), "longitude axis"),
            ((0.0,), (0.0, 180.0), r"\[-180, 180\)"),
        )
        for latitudes, longitudes, message in scenarios:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TECArchiveError, message):
                    TECGridDefinition(
                        latitudes_degrees=latitudes,
                        longitudes_degrees=longitudes,
                        definition_source="test",
                    )


@unittest.skipUnless(HAS_ARRAY_DEPENDENCIES, "Zarr data extras are not installed")
class ZarrTECGridStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.store = ZarrTECGridStore(
            self.temporary.name,
            clock=lambda: EPOCH + timedelta(days=2),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_sparse_dense_roundtrip_and_idempotent_publication(self) -> None:
        first = self.store.write(
            artifact=artifact(),
            observations=reversed(observations()),
            grid=grid(),
        )

        self.assertFalse(first.already_present)
        self.assertEqual(
            first.manifest.logical_sha256,
            "616ee40a6dd6c36b4f0422e3a91b02e4d65b06fcb9a2a15d1262be6ecbe2081e",
        )
        self.assertEqual(first.manifest.shape, (2, 2, 3))
        self.assertEqual(first.manifest.chunk_shape, (1, 2, 3))
        self.assertEqual(first.manifest.vtec_dtype, "float32")
        self.assertEqual(first.manifest.quality_flag_bits, (("edge", 0),))
        self.assertEqual(first.manifest.epochs[0].valid_cell_count, 5)
        self.assertEqual(first.manifest.epochs[0].missing_cell_count, 1)
        self.assertEqual(first.manifest.epochs[1].valid_cell_count, 6)
        self.assertTrue(Path(first.manifest.storage_ref.removeprefix("file://")).is_dir())

        restored = self.store.observations_for_epoch(first.manifest, EPOCH)
        self.assertEqual(len(restored), 5)
        self.assertEqual(restored[0].quality_flags, ("edge",))
        self.assertEqual(restored[-1].longitude_degrees, -175.0)
        self.assertEqual(restored[-1].vtec_tecu, 12.5)

        dense = self.store.read(first.manifest.grid_set_id)
        self.assertEqual(dense.manifest, first.manifest)
        self.assertEqual(tuple(dense.vtec_tecu.shape), (2, 2, 3))
        self.assertEqual(int(dense.valid[0].sum()), 5)
        self.assertTrue(math.isnan(float(dense.vtec_tecu[0, 1, 2])))
        self.assertEqual(float(dense.vtec_tecu[1, 1, 2]), 13.5)

        second = self.store.write(
            artifact=artifact(),
            observations=observations(),
            grid=grid(),
        )
        self.assertTrue(second.already_present)
        self.assertEqual(second.manifest, first.manifest)
        staging = list(
            Path(self.temporary.name).rglob("*.staging-*")
        )
        self.assertEqual(staging, [])

    def test_conflicts_duplicates_and_unknown_coordinates_fail_closed(self) -> None:
        original = observations()
        self.store.write(
            artifact=artifact(),
            observations=original,
            grid=grid(),
        )
        changed = (replace(original[0], vtec_tecu=999.0),) + original[1:]
        with self.assertRaises(TECArchiveConflictError):
            self.store.write(
                artifact=artifact(),
                observations=changed,
                grid=grid(),
            )
        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "different content or provenance",
        ):
            self.store.write(
                artifact=replace(artifact(), provider="different-provider"),
                observations=original,
                grid=grid(),
            )

        stored = self.store.write(
            artifact=artifact(),
            observations=original,
            grid=grid(),
        )
        self.store.read(stored.manifest.grid_set_id)
        group = self.store._zarr.open_group(
            store=stored.manifest.storage_ref.removeprefix("file://"),
            mode="r+",
            use_consolidated=False,
        )
        group["vtec"][0, 0, 0] = 999.0
        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "logical checksum|statistics",
        ):
            self.store.read(stored.manifest.grid_set_id)
        with self.assertRaisesRegex(
            TECArchiveConflictError,
            "logical checksum|statistics",
        ):
            self.store.write(
                artifact=artifact(),
                observations=original,
                grid=grid(),
            )

        with self.assertRaisesRegex(TECArchiveError, "duplicate TEC coordinate"):
            ZarrTECGridStore(Path(self.temporary.name) / "duplicate").write(
                artifact=artifact(),
                observations=original + (original[0],),
                grid=grid(),
            )
        with self.assertRaisesRegex(TECArchiveError, "absent from the explicit"):
            ZarrTECGridStore(Path(self.temporary.name) / "outside").write(
                artifact=artifact(),
                observations=(replace(original[0], latitude_degrees=5.0),),
                grid=grid(),
            )

    def test_manifest_json_roundtrip_is_lossless(self) -> None:
        result = self.store.write(
            artifact=artifact(),
            observations=observations(),
            grid=grid(),
        )

        restored = TECGridManifest.from_dict(result.manifest.to_dict())
        legacy_payload = result.manifest.to_dict()
        for field in (
            "value_kind",
            "source_grid_set_id",
            "source_grid_logical_sha256",
            "derivation_method",
        ):
            legacy_payload.pop(field)
        restored_legacy = TECGridManifest.from_dict(legacy_payload)

        self.assertEqual(restored, result.manifest)
        self.assertEqual(restored_legacy, result.manifest)

    def test_failed_archive_clock_does_not_leave_a_staging_group(self) -> None:
        root = Path(self.temporary.name) / "bad-clock"
        store = ZarrTECGridStore(
            root,
            clock=lambda: datetime(2026, 9, 11),
        )

        with self.assertRaisesRegex(TECArchiveError, "clock must return an aware"):
            store.write(
                artifact=artifact(),
                observations=observations(),
                grid=grid(),
            )

        self.assertEqual(list(root.rglob("*.staging-*")), [])

    def test_core_half_degree_grid_interpolates_and_preserves_provenance(self) -> None:
        source_grid = standard_gim_grid_definition()
        source_observations = standard_observations()
        source = self.store.write(
            artifact=artifact(),
            observations=source_observations,
            grid=source_grid,
        )

        first = self.store.write_core_half_degree(
            artifact=artifact(),
            observations=reversed(source_observations),
            source_grid=source_grid,
            source_manifest=source.manifest,
        )

        manifest = first.manifest
        self.assertFalse(first.already_present)
        self.assertEqual(manifest.layout_version, CORE_TEC_GRID_LAYOUT_VERSION)
        self.assertEqual(
            manifest.definition_source,
            CORE_TEC_GRID_DEFINITION_SOURCE,
        )
        self.assertEqual(manifest.shape, (1, 351, 720))
        self.assertEqual(manifest.epochs[0].valid_cell_count, 252_720)
        self.assertEqual(manifest.epochs[0].missing_cell_count, 0)
        self.assertEqual(manifest.value_kind, "interpolated_estimate")
        self.assertEqual(manifest.derivation_method, CORE_TEC_GRID_DERIVATION_METHOD)
        self.assertEqual(manifest.source_grid_set_id, source.manifest.grid_set_id)
        self.assertEqual(
            manifest.source_grid_logical_sha256,
            source.manifest.logical_sha256,
        )
        self.assertNotEqual(manifest.grid_set_id, source.manifest.grid_set_id)
        self.assertNotEqual(manifest.storage_ref, source.manifest.storage_ref)
        self.assertEqual(TECGridManifest.from_dict(manifest.to_dict()), manifest)
        with self.assertRaisesRegex(
            TECArchiveError,
            "derived grids cannot be converted",
        ):
            self.store.observations_for_epoch(manifest, EPOCH)

        source_group = self.store._zarr.open_group(
            store=source.manifest.storage_ref.removeprefix("file://"),
            mode="r",
            use_consolidated=False,
        )
        core_group = self.store._zarr.open_group(
            store=manifest.storage_ref.removeprefix("file://"),
            mode="r",
            use_consolidated=False,
        )
        self.assertEqual(
            tuple(core_group["latitude"][:]),
            core_gim_grid_definition().latitudes_degrees,
        )
        self.assertEqual(
            tuple(core_group["longitude"][:]),
            core_gim_grid_definition().longitudes_degrees,
        )

        # A native node is copied bit-for-bit into the derived float32 array.
        self.assertEqual(
            float(core_group["vtec"][0, 175, 10]),
            float(source_group["vtec"][0, 35, 1]),
        )
        # (-87, -179.5) is 20% north and 10% east in its native cell.
        self.assertAlmostEqual(float(core_group["vtec"][0, 1, 1]), 2.1, places=5)
        # 179.5 lies 90% across the periodic 175 -> -180 seam.
        self.assertAlmostEqual(float(core_group["vtec"][0, 0, 719]), 7.1, places=5)

        second = self.store.write_core_half_degree(
            artifact=artifact(),
            observations=source_observations,
            source_grid=source_grid,
            source_manifest=source.manifest,
        )
        self.assertTrue(second.already_present)
        self.assertEqual(second.manifest, manifest)

        conflicting_source = replace(
            source.manifest,
            logical_sha256="f" * 64,
        )
        with self.assertRaises(TECArchiveConflictError):
            self.store.write_core_half_degree(
                artifact=artifact(),
                observations=source_observations,
                source_grid=source_grid,
                source_manifest=conflicting_source,
            )

    def test_core_half_degree_grid_uses_weight_aware_missing_support(self) -> None:
        source_grid = standard_gim_grid_definition()
        source_observations = standard_observations(omit=(-85.0, -175.0))
        source = self.store.write(
            artifact=artifact(),
            observations=source_observations,
            grid=source_grid,
        )

        core = self.store.write_core_half_degree(
            artifact=artifact(),
            observations=source_observations,
            source_grid=source_grid,
            source_manifest=source.manifest,
        )
        group = self.store._zarr.open_group(
            store=core.manifest.storage_ref.removeprefix("file://"),
            mode="r",
            use_consolidated=False,
        )

        # Points on the south and west native lines do not need the absent
        # northeast corner, while a genuine interior point does.
        self.assertTrue(bool(group["valid"][0, 0, 1]))
        self.assertAlmostEqual(float(group["vtec"][0, 0, 1]), 0.1, places=5)
        self.assertTrue(bool(group["valid"][0, 1, 0]))
        self.assertAlmostEqual(float(group["vtec"][0, 1, 0]), 2.0, places=5)
        self.assertFalse(bool(group["valid"][0, 1, 1]))
        self.assertTrue(math.isnan(float(group["vtec"][0, 1, 1])))

        # The missing native node remains missing; interpolation never invents it.
        self.assertFalse(bool(group["valid"][0, 5, 10]))
        self.assertTrue(math.isnan(float(group["vtec"][0, 5, 10])))

    def test_core_interpolation_avoids_float32_intermediate_overflow(self) -> None:
        source_grid = standard_gim_grid_definition()
        source_observations = tuple(
            replace(
                item,
                vtec_tecu=(
                    3.0e38
                    if (
                        item.latitude_degrees,
                        item.longitude_degrees,
                    )
                    == (-87.5, -180.0)
                    else -3.0e38
                ),
            )
            if item.latitude_degrees == -87.5
            and item.longitude_degrees in {-180.0, -175.0}
            else item
            for item in standard_observations()
        )
        source = self.store.write(
            artifact=artifact(),
            observations=source_observations,
            grid=source_grid,
        )

        core = self.store.write_core_half_degree(
            artifact=artifact(),
            observations=source_observations,
            source_grid=source_grid,
            source_manifest=source.manifest,
        )
        group = self.store._zarr.open_group(
            store=core.manifest.storage_ref.removeprefix("file://"),
            mode="r",
            use_consolidated=False,
        )
        estimate = float(group["vtec"][0, 0, 1])

        self.assertTrue(math.isfinite(estimate))
        self.assertLess(abs(estimate / 2.4e38 - 1.0), 1e-6)


class BulkTECArchiveTests(unittest.TestCase):
    def test_catalog_publication_happens_after_each_grid_write(self) -> None:
        events = []
        native_manifest = SimpleNamespace(grid_set_id="a" * 64)
        core_manifest = SimpleNamespace(grid_set_id="b" * 64)

        class Store:
            def write(self, **kwargs):
                events.append(("grid", kwargs))
                return TECGridWriteResult(
                    manifest=native_manifest,
                    already_present=False,
                )

            def write_core_half_degree(self, **kwargs):
                events.append(("core_grid", kwargs))
                return TECGridWriteResult(
                    manifest=core_manifest,
                    already_present=True,
                )

        class Catalog:
            publication_count = 0

            def publish(self, candidate):
                event = "catalog" if self.publication_count == 0 else "core_catalog"
                self.publication_count += 1
                events.append((event, candidate))
                return CatalogPublishResult(
                    manifest=candidate,
                    already_present=event == "core_catalog",
                )

            def close(self):
                events.append(("close", None))

        archive = BulkTECArchive(grid_store=Store(), catalog=Catalog())
        result = archive.archive(
            artifact=artifact(),
            observations=observations(),
            grid=grid(),
        )
        archive.close()

        self.assertEqual(
            [event[0] for event in events],
            ["grid", "catalog", "core_grid", "core_catalog", "close"],
        )
        self.assertIs(result.manifest, native_manifest)
        self.assertFalse(result.grid_already_present)
        self.assertFalse(result.catalog_already_present)
        self.assertIs(result.core_manifest, core_manifest)
        self.assertTrue(result.core_grid_already_present)
        self.assertTrue(result.core_catalog_already_present)
        core_arguments = events[2][1]
        self.assertIs(core_arguments["source_manifest"], native_manifest)

    def test_native_only_archive_skips_interpolated_core_grid(self) -> None:
        events = []
        native_manifest = SimpleNamespace(grid_set_id="a" * 64)

        class Store:
            def write(self, **kwargs):
                events.append(("grid", kwargs))
                return TECGridWriteResult(
                    manifest=native_manifest,
                    already_present=True,
                )

            def write_core_half_degree(self, **kwargs):
                raise AssertionError("native-only history must not derive a core grid")

        class Catalog:
            def publish(self, candidate):
                events.append(("catalog", candidate))
                return CatalogPublishResult(manifest=candidate, already_present=False)

            def close(self):
                pass

        result = BulkTECArchive(grid_store=Store(), catalog=Catalog()).archive(
            artifact=artifact(),
            observations=observations(),
            grid=grid(),
            include_core=False,
        )

        self.assertEqual([event[0] for event in events], ["grid", "catalog"])
        self.assertIs(result.manifest, native_manifest)
        self.assertIsNone(result.core_manifest)


if __name__ == "__main__":
    unittest.main()
