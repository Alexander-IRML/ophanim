ALTER TABLE ophanim.tec_grid_sets
    ADD COLUMN value_kind TEXT NOT NULL DEFAULT 'native_measurement',
    ADD COLUMN source_grid_set_id CHAR(64),
    ADD COLUMN source_grid_logical_sha256 CHAR(64),
    ADD COLUMN derivation_method TEXT,
    ADD CONSTRAINT tec_grid_sets_source_grid_fk
        FOREIGN KEY (source_grid_set_id)
        REFERENCES ophanim.tec_grid_sets (grid_set_id),
    ADD CONSTRAINT tec_grid_sets_value_kind_check
        CHECK (value_kind IN ('native_measurement', 'interpolated_estimate')),
    ADD CONSTRAINT tec_grid_sets_source_grid_not_self_check
        CHECK (
            source_grid_set_id IS NULL
            OR source_grid_set_id <> grid_set_id
        ),
    ADD CONSTRAINT tec_grid_sets_lineage_check
        CHECK (
            (
                value_kind = 'native_measurement'
                AND source_grid_set_id IS NULL
                AND source_grid_logical_sha256 IS NULL
                AND derivation_method IS NULL
            )
            OR
            (
                value_kind = 'interpolated_estimate'
                AND source_grid_set_id IS NOT NULL
                AND source_grid_logical_sha256 IS NOT NULL
                AND source_grid_logical_sha256 ~ '^[0-9a-f]{64}$'
                AND derivation_method IS NOT NULL
                AND length(trim(derivation_method)) > 0
            )
        );

CREATE INDEX tec_grid_sets_value_kind_time_idx
    ON ophanim.tec_grid_sets (value_kind, first_observed_at DESC);

CREATE INDEX tec_grid_sets_source_grid_idx
    ON ophanim.tec_grid_sets (source_grid_set_id)
    WHERE source_grid_set_id IS NOT NULL;
