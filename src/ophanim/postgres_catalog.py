"""PostgreSQL catalog for immutable Zarr TEC grid groups."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from threading import Lock
from typing import Any

from ophanim.tec_archive import (
    CatalogPublishResult,
    INTERPOLATED_VALUE_KIND,
    NATIVE_VALUE_KIND,
    TECArchiveConflictError,
    TECArchiveDependencyError,
    TECGridManifest,
)


POSTGRES_SCHEMA_VERSION = 2
_SCHEMA_LOCK_ID = 6_727_404_716_911_829_797
_MIGRATION_FILES = {
    1: "0001_tec_archive.sql",
    2: "0002_core_grid.sql",
}


@dataclass(frozen=True, slots=True)
class TECCatalogStatistics:
    artifact_count: int
    grid_set_count: int
    epoch_count: int
    valid_cell_count: int
    missing_cell_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "artifact_count": self.artifact_count,
            "grid_set_count": self.grid_set_count,
            "epoch_count": self.epoch_count,
            "valid_cell_count": self.valid_cell_count,
            "missing_cell_count": self.missing_cell_count,
        }


class PostgresTECCatalog:
    """Small PostgreSQL catalog; dense cell values remain in Zarr."""

    def __init__(
        self,
        conninfo: str = "",
        *,
        pool: Any | None = None,
        min_size: int = 1,
        max_size: int = 4,
    ) -> None:
        if pool is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as error:
                raise TECArchiveDependencyError(
                    "PostgreSQL catalog support requires the 'data' dependencies"
                ) from error
            pool = ConnectionPool(
                conninfo=conninfo,
                min_size=min_size,
                max_size=max_size,
                open=True,
                name="ophanim-tec-catalog",
            )
            self._owns_pool = True
        else:
            self._owns_pool = False
        self._pool = pool
        self._schema_ready = False
        self._schema_lock = Lock()

    def wait(self, timeout_seconds: float = 30.0) -> None:
        self._pool.wait(timeout=timeout_seconds)

    def initialize_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._pool.connection() as connection:
                with connection.transaction():
                    connection.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_SCHEMA_LOCK_ID,),
                    )
                    connection.execute("CREATE SCHEMA IF NOT EXISTS ophanim")
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS ophanim.schema_migrations (
                            version INTEGER PRIMARY KEY CHECK (version > 0),
                            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                    row = connection.execute(
                        "SELECT COALESCE(max(version), 0) "
                        "FROM ophanim.schema_migrations"
                    ).fetchone()
                    current = int(row[0])
                    if current < 0 or current > POSTGRES_SCHEMA_VERSION:
                        raise RuntimeError(
                            "unsupported PostgreSQL catalog schema version "
                            f"{current}; expected {POSTGRES_SCHEMA_VERSION}"
                        )
                    for version in range(current + 1, POSTGRES_SCHEMA_VERSION + 1):
                        migration = (
                            resources.files("ophanim")
                            .joinpath(
                                "sql",
                                "postgres",
                                _MIGRATION_FILES[version],
                            )
                            .read_text(encoding="utf-8")
                        )
                        connection.execute(migration, prepare=False)
                        connection.execute(
                            "INSERT INTO ophanim.schema_migrations (version) "
                            "VALUES (%s)",
                            (version,),
                        )
            self._schema_ready = True

    def publish(self, manifest: TECGridManifest) -> CatalogPublishResult:
        self.initialize_schema()
        manifest_json = json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (manifest.grid_set_id,),
                )
                existing = connection.execute(
                    "SELECT logical_sha256, manifest "
                    "FROM ophanim.tec_grid_sets WHERE grid_set_id = %s",
                    (manifest.grid_set_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != manifest.logical_sha256:
                        raise TECArchiveConflictError(
                            "PostgreSQL grid_set_id has different logical content"
                        )
                    stored_manifest = TECGridManifest.from_dict(
                        _json_object(existing[1])
                    )
                    if stored_manifest != manifest:
                        raise TECArchiveConflictError(
                            "PostgreSQL grid_set_id has different immutable metadata"
                        )
                    return CatalogPublishResult(
                        manifest=stored_manifest,
                        already_present=True,
                    )

                self._publish_source(connection, manifest)
                self._verify_source_grid(connection, manifest)
                connection.execute(
                    """
                    INSERT INTO ophanim.tec_grid_sets (
                        grid_set_id, artifact_id, layout_version, storage_ref,
                        zarr_format, grid_fingerprint, logical_sha256,
                        value_kind, source_grid_set_id,
                        source_grid_logical_sha256, derivation_method,
                        time_count, latitude_count, longitude_count,
                        time_chunk, latitude_chunk, longitude_chunk,
                        vtec_dtype, quality_mask_dtype,
                        first_observed_at, last_observed_at,
                        valid_cell_count, missing_cell_count, archived_at,
                        manifest
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s::jsonb
                    )
                    """,
                    (
                        manifest.grid_set_id,
                        manifest.artifact_id,
                        manifest.layout_version,
                        manifest.storage_ref,
                        manifest.zarr_format,
                        manifest.grid_fingerprint,
                        manifest.logical_sha256,
                        manifest.value_kind,
                        manifest.source_grid_set_id,
                        manifest.source_grid_logical_sha256,
                        manifest.derivation_method,
                        manifest.shape[0],
                        manifest.shape[1],
                        manifest.shape[2],
                        manifest.chunk_shape[0],
                        manifest.chunk_shape[1],
                        manifest.chunk_shape[2],
                        manifest.vtec_dtype,
                        manifest.quality_mask_dtype,
                        manifest.epochs[0].observed_at,
                        manifest.epochs[-1].observed_at,
                        sum(item.valid_cell_count for item in manifest.epochs),
                        sum(item.missing_cell_count for item in manifest.epochs),
                        manifest.archived_at,
                        manifest_json,
                    ),
                )
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO ophanim.tec_grid_epochs (
                            grid_set_id, time_index, observed_at,
                            valid_cell_count, missing_cell_count,
                            minimum_vtec_tecu, maximum_vtec_tecu, mean_vtec_tecu
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                manifest.grid_set_id,
                                epoch.time_index,
                                epoch.observed_at,
                                epoch.valid_cell_count,
                                epoch.missing_cell_count,
                                epoch.minimum_vtec_tecu,
                                epoch.maximum_vtec_tecu,
                                epoch.mean_vtec_tecu,
                            )
                            for epoch in manifest.epochs
                        ],
                    )
        return CatalogPublishResult(manifest=manifest, already_present=False)

    def get_for_artifact(
        self,
        artifact_id: str,
        *,
        layout_version: str,
    ) -> TECGridManifest | None:
        self.initialize_schema()
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT manifest FROM ophanim.tec_grid_sets "
                "WHERE artifact_id = %s AND layout_version = %s",
                (artifact_id, layout_version),
            ).fetchone()
        if row is None:
            return None
        return TECGridManifest.from_dict(_json_object(row[0]))

    def statistics(self) -> TECCatalogStatistics:
        self.initialize_schema()
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM ophanim.source_artifacts),
                    count(*),
                    COALESCE(sum(time_count), 0),
                    COALESCE(sum(valid_cell_count), 0),
                    COALESCE(sum(missing_cell_count), 0)
                FROM ophanim.tec_grid_sets
                """
            ).fetchone()
        return TECCatalogStatistics(*(int(value) for value in row))

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()

    @staticmethod
    def _verify_source_grid(
        connection: Any,
        manifest: TECGridManifest,
    ) -> None:
        if manifest.value_kind == NATIVE_VALUE_KIND:
            return
        if manifest.value_kind != INTERPOLATED_VALUE_KIND:
            raise TECArchiveConflictError(
                "derived grid has an unsupported value kind"
            )

        row = connection.execute(
            """
            SELECT artifact_id, logical_sha256, value_kind
            FROM ophanim.tec_grid_sets
            WHERE grid_set_id = %s
            FOR KEY SHARE
            """,
            (manifest.source_grid_set_id,),
        ).fetchone()
        if row is None:
            raise TECArchiveConflictError(
                "derived grid source grid is not published"
            )
        source_artifact_id, source_logical_sha256, source_value_kind = (
            str(value) for value in row
        )
        if source_artifact_id != manifest.artifact_id:
            raise TECArchiveConflictError(
                "derived grid source grid belongs to a different artifact"
            )
        if source_logical_sha256 != manifest.source_grid_logical_sha256:
            raise TECArchiveConflictError(
                "derived grid source grid has a different logical checksum"
            )
        if source_value_kind != NATIVE_VALUE_KIND:
            raise TECArchiveConflictError(
                "derived grid source grid is not a native measurement grid"
            )

    @staticmethod
    def _publish_source(connection: Any, manifest: TECGridManifest) -> None:
        connection.execute(
            """
            INSERT INTO ophanim.source_artifacts (
                artifact_id, provider, product, parser_version,
                revision_priority, revision, source_uri, checksum_sha256,
                storage_ref, ingested_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (artifact_id) DO NOTHING
            """,
            (
                manifest.artifact_id,
                manifest.provider,
                manifest.product,
                manifest.parser_version,
                manifest.revision_priority,
                manifest.revision,
                manifest.source_uri,
                manifest.source_checksum_sha256,
                manifest.source_storage_ref,
                manifest.source_ingested_at,
            ),
        )
        row = connection.execute(
            """
            SELECT provider, product, parser_version, revision_priority,
                   revision, source_uri, checksum_sha256, storage_ref,
                   ingested_at
            FROM ophanim.source_artifacts
            WHERE artifact_id = %s
            """,
            (manifest.artifact_id,),
        ).fetchone()
        expected = (
            manifest.provider,
            manifest.product,
            manifest.parser_version,
            manifest.revision_priority,
            manifest.revision,
            manifest.source_uri,
            manifest.source_checksum_sha256,
            manifest.source_storage_ref,
            manifest.source_ingested_at,
        )
        if tuple(row) != expected:
            raise TECArchiveConflictError(
                "artifact_id already exists with different source provenance"
            )


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise RuntimeError("PostgreSQL catalog manifest is not a JSON object")
    return value


__all__ = [
    "POSTGRES_SCHEMA_VERSION",
    "PostgresTECCatalog",
    "TECCatalogStatistics",
]
