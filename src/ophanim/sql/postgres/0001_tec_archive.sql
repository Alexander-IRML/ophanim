CREATE TABLE ophanim.source_artifacts (
    artifact_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    product TEXT NOT NULL CHECK (length(trim(product)) > 0),
    parser_version TEXT NOT NULL CHECK (length(trim(parser_version)) > 0),
    revision_priority INTEGER NOT NULL CHECK (revision_priority >= 0),
    revision TEXT CHECK (revision IS NULL OR length(trim(revision)) > 0),
    source_uri TEXT CHECK (source_uri IS NULL OR length(trim(source_uri)) > 0),
    checksum_sha256 CHAR(64) NOT NULL
        CHECK (checksum_sha256 ~ '^[0-9a-f]{64}$'),
    storage_ref TEXT NOT NULL CHECK (length(trim(storage_ref)) > 0),
    ingested_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX source_artifacts_checksum_idx
    ON ophanim.source_artifacts (checksum_sha256);

CREATE INDEX source_artifacts_series_time_idx
    ON ophanim.source_artifacts (provider, product, ingested_at DESC);

CREATE TABLE ophanim.tec_grid_sets (
    grid_set_id CHAR(64) PRIMARY KEY
        CHECK (grid_set_id ~ '^[0-9a-f]{64}$'),
    artifact_id TEXT NOT NULL
        REFERENCES ophanim.source_artifacts (artifact_id),
    layout_version TEXT NOT NULL CHECK (length(trim(layout_version)) > 0),
    storage_ref TEXT NOT NULL UNIQUE CHECK (length(trim(storage_ref)) > 0),
    zarr_format SMALLINT NOT NULL CHECK (zarr_format = 3),
    grid_fingerprint CHAR(64) NOT NULL
        CHECK (grid_fingerprint ~ '^[0-9a-f]{64}$'),
    logical_sha256 CHAR(64) NOT NULL
        CHECK (logical_sha256 ~ '^[0-9a-f]{64}$'),
    time_count INTEGER NOT NULL CHECK (time_count > 0),
    latitude_count INTEGER NOT NULL CHECK (latitude_count > 0),
    longitude_count INTEGER NOT NULL CHECK (longitude_count > 0),
    time_chunk INTEGER NOT NULL CHECK (time_chunk > 0),
    latitude_chunk INTEGER NOT NULL CHECK (latitude_chunk > 0),
    longitude_chunk INTEGER NOT NULL CHECK (longitude_chunk > 0),
    vtec_dtype TEXT NOT NULL CHECK (length(trim(vtec_dtype)) > 0),
    quality_mask_dtype TEXT NOT NULL
        CHECK (length(trim(quality_mask_dtype)) > 0),
    first_observed_at TIMESTAMPTZ NOT NULL,
    last_observed_at TIMESTAMPTZ NOT NULL,
    valid_cell_count BIGINT NOT NULL CHECK (valid_cell_count >= 0),
    missing_cell_count BIGINT NOT NULL CHECK (missing_cell_count >= 0),
    archived_at TIMESTAMPTZ NOT NULL,
    manifest JSONB NOT NULL CHECK (jsonb_typeof(manifest) = 'object'),
    CHECK (last_observed_at >= first_observed_at),
    CHECK (
        time_chunk <= time_count
        AND latitude_chunk <= latitude_count
        AND longitude_chunk <= longitude_count
    ),
    CHECK (
        valid_cell_count + missing_cell_count
        = time_count::BIGINT
            * latitude_count::BIGINT
            * longitude_count::BIGINT
    ),
    UNIQUE (artifact_id, layout_version)
);

CREATE INDEX tec_grid_sets_source_time_idx
    ON ophanim.tec_grid_sets (
        artifact_id,
        first_observed_at,
        last_observed_at
    );

CREATE TABLE ophanim.tec_grid_epochs (
    grid_set_id CHAR(64) NOT NULL
        REFERENCES ophanim.tec_grid_sets (grid_set_id) ON DELETE CASCADE,
    time_index INTEGER NOT NULL CHECK (time_index >= 0),
    observed_at TIMESTAMPTZ NOT NULL,
    valid_cell_count INTEGER NOT NULL CHECK (valid_cell_count >= 0),
    missing_cell_count INTEGER NOT NULL CHECK (missing_cell_count >= 0),
    minimum_vtec_tecu DOUBLE PRECISION,
    maximum_vtec_tecu DOUBLE PRECISION,
    mean_vtec_tecu DOUBLE PRECISION,
    PRIMARY KEY (grid_set_id, time_index),
    UNIQUE (grid_set_id, observed_at),
    CHECK (
        (valid_cell_count = 0
            AND minimum_vtec_tecu IS NULL
            AND maximum_vtec_tecu IS NULL
            AND mean_vtec_tecu IS NULL)
        OR
        (valid_cell_count > 0
            AND minimum_vtec_tecu IS NOT NULL
            AND maximum_vtec_tecu IS NOT NULL
            AND mean_vtec_tecu IS NOT NULL
            AND minimum_vtec_tecu > '-Infinity'::DOUBLE PRECISION
            AND maximum_vtec_tecu < 'Infinity'::DOUBLE PRECISION
            AND mean_vtec_tecu > '-Infinity'::DOUBLE PRECISION
            AND mean_vtec_tecu < 'Infinity'::DOUBLE PRECISION
            AND mean_vtec_tecu >= minimum_vtec_tecu
            AND mean_vtec_tecu <= maximum_vtec_tecu)
    )
);

CREATE INDEX tec_grid_epochs_observed_at_idx
    ON ophanim.tec_grid_epochs (observed_at DESC);

CREATE INDEX tec_grid_epochs_observed_at_brin_idx
    ON ophanim.tec_grid_epochs USING BRIN (observed_at);
