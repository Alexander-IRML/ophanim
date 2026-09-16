# OPHANIM

OPHANIM is an experimental pipeline for turning regional ionospheric TEC
observations into forecasts and disturbance assessments.

> **Research software:** v0.1.0 produces anomaly candidates for experiment and
> review. It is not an operational space-weather warning service, and model
> agreement is not confirmation of a physical disturbance.

The v0.1.0 deployment has a dependency-free Python implementation of the first
local vertical slice, a CPU selective-state-space abnormality monitor, a causal
spatial ConvLSTM baseline, and a containerized bulk-data plane. It parses
two-dimensional IONEX v1 products, stores operational provenance and
observations in SQLite, computes regional mean/median VTEC, issues a persistence
forecast, and reconciles that forecast with a residual z-score. Validated CODE
downloads can also be archived as immutable native and derived 0.5° × 0.5°
Zarr grids indexed by PostgreSQL. The Mamba monitor can resumably acquire about
twenty years of CODE final maps, fit a ridge readout over compact regional
mean/median sequences using a fixed CPU selective-SSM reservoir, and later check
only publications after its last successful scan cursor. The ConvLSTM reuses
those same immutable native grids to predict each next two-hour map and retain
spatial residual evidence for side-by-side comparison.

## Core flow

```text
CODE GIM download or local IONEX source
    -> ingestion
    -> regional aggregation (mean and median)
    -> forecast (PENDING)

stored native GIM epoch + selected bounds
    -> bilinear regional estimate grid
    -> readout table / CSV

validated standard-grid CODE artifact
    -> immutable native Zarr v3 grid
    -> immutable 0.5° × 0.5° core Zarr v3 grid
    -> PostgreSQL source / grid / epoch catalog with lineage

new observation / scheduled sweep / backfill
    -> reconciliation
    -> anomaly scoring
    -> SCORED / INSUFFICIENT_DATA / pending retry

historical CODE final maps (day checkpoints)
    -> compact native regional mean/median sequence
    -> fixed CPU selective-state-space reservoir + fitted ridge readout artifact
    -> scan publications after the last successful cursor
    -> candidate readouts for review

the same immutable native 71 x 72 maps
    -> twelve earlier two-hour frames (target excluded)
    -> prior-day-residual CNN-ConvLSTM prediction of the next native map
    -> masked spatial residual score + localized candidate cells
    -> timestamp-aligned comparison with the Mamba candidate result
```

A forecast miss and an actual ionospheric disturbance are represented
separately. A residual-based detector may produce a disturbance `CANDIDATE`;
later evidence can confirm it or leave it unknown.

## Package map

```text
src/ophanim/
├── domain/           Shared records and lifecycle enums
├── artifacts.py      Content-addressed immutable file storage
├── bootstrap.py      Concrete SQLite application assembly
├── ionex.py          IONEX parsing boundary
├── ingestion.py      Local/HTTP acquisition, validation, and ingestion
├── gim.py            Bounded public CODE GIM discovery and HTTPS download
├── mamba_model.py    Dependency-free CPU selective-SSM model and artifacts
├── mamba_monitor.py  Resumable history jobs, scan cursor, and candidate ledger
├── convlstm_model.py Causal native-grid predictor, artifacts, and checkpoints
├── spatial_monitor.py Durable spatial training, scan, and comparison lifecycle
├── aggregation.py    Region crop, mean/median, coverage, and quality
├── regional_grid.py  Fine regional estimate tables from native GIM epochs
├── forecasting.py    Persistence forecaster and model runner
├── detection.py      Residual calibration, scoring, and assessment
├── reconciliation.py Resolve forecasts once observations are available
├── repositories.py   Durable-state and model-registry contracts
├── sqlite.py         Concrete repositories and transaction boundary
├── tec_archive.py    Immutable native and derived Zarr v3 grids
├── postgres_catalog.py PostgreSQL grid discovery and lineage catalog
├── data_cli.py       Bulk-data initialization and status commands
├── workflows.py      End-to-end application orchestration boundary
├── desktop.py        Loopback-only desktop web server and UI adapter
├── sql/postgres/     Versioned PostgreSQL catalog migrations
└── web/              Dependency-free desktop interface assets
```

These are modules inside one application, not separately deployed services.
The native desktop workflow uses SQLite plus immutable source/model artifact
files. The optional bulk-data plane adds PostgreSQL and Zarr without moving the
forecast workflow off SQLite yet.

## Boundary rules

- Repositories own durable state; processors transform inputs into outputs.
- Time fields are explicit: `observed_at`, derived-data `produced_at`, forecast
  `origin_at`, `issued_at`, `valid_at`, and `scored_at` are not interchangeable.
- `ingested_at` and `produced_at` are availability stamps captured after the
  final SQLite writer lock is acquired, not pre-computation start times.
- `LIVE` forecasts enforce source availability at issuance. `HINDCAST`
  forecasts deliberately relax that constraint and are labeled durably.
- Live issuance takes the SQLite publication lock before stamping `issued_at`
  and reads its inputs in that same transaction, so concurrent work cannot
  make an uncommitted observation appear historically available.
- Forecast history and actuals are bound to one provider/product series.
- Raw parses record their parser revision. Re-parsing identical bytes under a
  new parser version creates separate observation provenance while sharing the
  same content-addressed blob.
- Higher `revision_priority` wins within a source series; availability time
  breaks ties. This keeps a late low-priority backfill from replacing a final
  product.
- Scalar TEC fields include the unit in their name (`*_vtec_tecu`).
- Grid and boundary longitudes use one half-open `[-180, 180)` convention;
  the parser canonicalizes equivalent wrap endpoints before persistence.
- Source, processing, region, model, and detector versions remain traceable.
- Every forecast begins `PENDING`; scoring happens through reconciliation.
- `forecast_anomaly` does not by itself mean `CONFIRMED` disturbance.
- Result searches return only the tip of each revision chain by default;
  superseded decisions remain available as audit history.

## Desktop application

On this laptop, double-click **OPHANIM Dashboard** on the Windows desktop. A
small launcher window stays open while OPHANIM runs and the workspace opens in
Google Chrome when it is installed in a standard Windows location. If Chrome
cannot be started, the launcher falls back to the Windows default browser. Use
**Stop application** in the workspace, or close the launcher window, when
finished.

The desktop workflow is deliberately simple:

1. Click **Download & inspect GIM** to use the latest CODE rapid product, optionally
   choose a final/exact-date product, or switch to a local IONEX file. The
   Central Texas crop remains editable.
2. Generate a fine regional readout for any loaded epoch and download its table
   as CSV if useful.
3. Optionally initialize the independent 20-year Mamba monitor. The background
   job saves one checkpoint per UTC day and resumes after restart.
4. Once ready, use **Check new readouts** to download and score data since the
   last successful check.
5. In Docker mode, train the **Predictive ConvLSTM monitor** from those archived
   native maps, then use **Check & compare** for localized residual evidence.
6. Choose a forecast origin, mean or median VTEC, horizon, and run mode.
7. Run the persistence forecast and inspect the separate forecast-anomaly and
   physical-disturbance assessments.
8. Use **Check pending** later when a forecast did not yet have an actual at its
   valid time.

All computation and durable state stay local. Automatic mode downloads the
source over restricted HTTPS from CODE at the University of Bern; local-file
mode never sends the selected file anywhere. The SQLite database, immutable
artifacts, and uploaded source cache live under `var/desktop/`. Pending
forecasts retain their detector configuration across application restarts, so
a later IONEX file can be loaded and reconciled without changing the original
experiment. To launch the same application from a WSL terminal, run:

```bash
scripts/launch_desktop.sh
```

The server listens only on `127.0.0.1` and requires a per-launch token for any
operation that changes state.

### Docker and bulk TEC archive

The Compose stack runs the desktop application, PostgreSQL 18, a one-shot
schema migration, a persistent Zarr v3 store, and the CPU PyTorch runtime used
by the spatial baseline. On Windows/WSL, Docker Desktop
must be running and **Settings → Resources → WSL Integration** must be
enabled for this Linux distribution. Confirm the connection first with
`docker compose version`.

For the first launch:

```bash
cp .env.example .env
# Keep .env local, and replace POSTGRES_PASSWORD before using the stack.
docker compose up --build --detach
```

Then open <http://127.0.0.1:8765>. PostgreSQL has no host-published port, and
the application port is published only on the laptop's loopback interface.
If another local process already owns 8765, set `OPHANIM_PORT` in `.env` and
open that loopback port instead; the container still listens on 8765 internally.
Startup waits for PostgreSQL to become healthy, runs the catalog migration, and
starts the application only after that migration succeeds. Check the services
or inspect the catalog with:

```bash
docker compose ps
docker compose run --rm migrate status
docker compose logs --follow app
```

Use either this stack or the native **OPHANIM Dashboard** launcher at one time;
they otherwise compete for the same browser port and desktop data directory.
Stop the stack safely with `docker compose down`. That preserves all three data
locations:

- `var/desktop/` on the host contains SQLite state, immutable source artifacts,
  and the desktop source cache.
- The `ophanim-postgres` named volume contains searchable bulk-catalog records.
- The `ophanim-zarr` named volume contains dense TEC arrays.

`docker compose down --volumes` deliberately deletes the PostgreSQL and Zarr
named volumes; it does **not** delete `var/desktop/`. Use it only when discarding
the bulk archive is intended. A complete backup must keep the PostgreSQL
catalog, Zarr volume, and referenced immutable source artifacts together.

#### Storage ownership

| Store | Owns | Does not own |
| --- | --- | --- |
| SQLite + artifact files | Current ingestion, forecasts, decisions, native-cell queries, Mamba jobs/readouts/models, and original source bytes | The scalable bulk-grid catalog |
| PostgreSQL `ophanim` schema | Source provenance, grid manifests, epoch timestamps/statistics, checksums, and searchable indexes | Millions of individual TEC cells |
| Zarr | Dense native measurements and derived 0.5° core estimates, coordinates, presence masks, and source quality masks | Forecast lifecycle or catalog transactions |

In the Docker stack, a successfully validated automatic CODE download follows
the existing SQLite/artifact ingestion path and is also mirrored to Zarr and
PostgreSQL. The native measurements remain the source of truth; a separate
0.5° × 0.5° group is the standard working grid and is explicitly marked as
derived. PostgreSQL is the discoverability boundary: readers should use only
grids with a committed catalog row, rather than scanning the Zarr directory.
Zarr has no global resolution setting; each group's coordinate arrays and
versioned layout define its resolution.

Each archived source artifact first gets one immutable native group at:

```text
raw/tec-grid/v1/<grid_set_id>.zarr/
├── time
├── latitude
├── longitude
├── vtec
├── valid
└── quality_mask
```

The same artifact then gets one immutable derived core group at:

```text
derived/tec-grid/half-degree/v1/<grid_set_id>.zarr/
├── time
├── latitude
├── longitude
├── vtec
├── valid
└── quality_mask
```

The dimensions are `(T, Y, X)` for time, latitude, and longitude. Numeric
values use little-endian storage.

| Array | Shape | Type | Chunk shape |
| --- | --- | --- | --- |
| `time` | `(T,)` | signed 64-bit UTC microseconds since Unix epoch | `(min(T, 1024),)` |
| `latitude` | `(Y,)` | 64-bit float, degrees north | `(Y,)` |
| `longitude` | `(X,)` | 64-bit float, degrees east | `(X,)` |
| `vtec` | `(T, Y, X)` | 32-bit float, TECU; missing values are `NaN` | `(1, min(Y, 256), min(X, 256))` |
| `valid` | `(T, Y, X)` | boolean presence mask | Same as `vtec` |
| `quality_mask` | `(T, Y, X)` | unsigned 32-bit source-quality bit mask | Same as `vtec` |

The `valid` array is authoritative for missingness. Quality flags are assigned
deterministic bit positions in the group manifest. Arrays use Blosc Zstandard
compression with bit shuffle and CRC32C checksums. A normal standard CODE group
has shape `(T, 71, 72)`, 5,112 cells per epoch, and dense-array chunks of
`(1, 71, 72)`. Its `value_kind` is `native_measurement`.

The core group has shape `(T, 351, 720)`, **252,720 cells per epoch**, and
dense-array chunks of `(1, 256, 256)`. Its latitude axis runs from -87.5°
through 87.5° and its half-open longitude axis from -180° through 179.5°,
both at 0.5° spacing. OPHANIM does not extrapolate to ±90° because the native
source has no support there. The denser layer contains about 49.4 times as many
cells per epoch as the native layer; actual disk growth varies with compression
and missingness.

Core values use deterministic bilinear interpolation with periodic longitude
at the antimeridian. Native nodes are copied exactly. A derived cell is valid
only when every native corner with nonzero interpolation weight is valid;
unsupported cells remain invalid with `NaN` VTEC rather than being silently
filled. Quality bits are combined from the native cells that actually
participate. The core manifest uses `value_kind=interpolated_estimate` and
records the native `source_grid_set_id`, its logical checksum, and the versioned
derivation method.

PostgreSQL catalog schema v2 adds that explicit parent-child lineage and
enforces that a derived row references a previously published native grid for
the same artifact and checksum. Publication therefore happens in this order:
verify and atomically install native Zarr, publish native catalog rows, verify
and atomically install core Zarr, then publish core catalog rows. If a later
step fails, an already published native grid remains usable while no partial
derived grid becomes discoverable.

Catalog status totals count stored materializations, so one source epoch now
contributes one native epoch and one core epoch; cell totals will consequently
be dominated by the denser layer. The migration preserves existing native
groups but does not synthesize core siblings for them. Re-fetch an old source,
or use a future explicit backfill command, to materialize its core grid.

Each Zarr publication is crash-conscious and idempotent. OPHANIM writes a
sibling staging directory, reopens and reads every array (exercising its
CRC32C), validates the array contract, recomputes the logical checksum and
epoch statistics, fsyncs every staged file and directory, atomically renames
the group into place, and fsyncs the parent directory. Only then does it publish
the source, grid, and epoch rows in one PostgreSQL transaction. A crash between
the rename and catalog commit can leave an undiscoverable Zarr orphan, but
cannot leave a catalog row pointing to a partial group. Retrying the same
artifact/layout reuses only fully identical content and provenance; a mismatch
is a hard conflict.

### Automatic GIM source

The zero-setup feed is CODE's
[public product archive](https://www.aiub.unibe.ch/download/CODE/), not the IGS
combined product. CODE is an IGS Ionosphere Associate Analysis Center; its
daily compressed IONEX files typically contain global snapshots at a two-hour
cadence on the standard 5° longitude × 2.5° latitude lattice. Modern files use
`.INX.gz`; historical files use legacy Unix-compress `.Z`. The serialized ±180°
endpoints describe the same seam, so OPHANIM validates and persists exactly
**5,112 unique cells per epoch**. Rapid and final files share the `code/gim`
source series, with final assigned a higher revision priority.

“Latest” searches only a bounded set of completed UTC days, newest first. An
exact date is never silently replaced with another date. The downloader accepts
only known CODE HTTPS hosts and paths, caps compressed and decompressed sizes,
rejects login/HTML responses, validates gzip or legacy Unix-compress data and
IONEX structure, and then sends
the already-downloaded bytes through the normal atomic ingestion boundary while
retaining the public source URL as provenance.

Exact-date acquisition supports CODE's year archive back to 2002. Dates before
the IGS long-filename transition on 2022-11-26 use the historic final
`CODGddd0.yyI.Z` / rapid `CORGddd0.yyI.Z` convention and bounded Unix-compress
decoding. The historical Mamba initializer deliberately uses final products;
the old rapid series is not consistently retained in the public archive.

The [IGS product catalog](https://igs.org/products/) identifies the official
combined rapid/final GIM, but its CDDIS download path requires a
[NASA Earthdata Login](https://www.earthdata.nasa.gov/centers/cddis-daac/archive-access).
That source is intentionally not presented as zero-configuration. The local-file
path remains available for combined IGS or other IONEX products.

### Mamba abnormality monitor

The monitor is independent of the single-map forecast form and has three local
API operations: `GET /api/mamba/status`, `POST /api/mamba/initialize`, and
`POST /api/mamba/scan`. Initialize and scan return immediately with a durable
background job. The UI polls its phase and day-level progress. A shutdown or
crash leaves completed days reusable, and an unfinished job resumes on the next
launch.

Initialization defaults to the previous twenty complete UTC years and the
current Central Texas bounds. It downloads one compressed daily CODE final
IONEX product (`.INX.gz` or legacy `.Z`), retains the original bytes in a
content-addressed store, and writes only compact regional mean/median readouts
to `mamba-monitor.sqlite3`. The source maps are typically spaced two hours
apart. When the Docker bulk archive is enabled, initialization also publishes
the historical **native** grids to Zarr/PostgreSQL. It intentionally does not
materialize the 0.5-degree interpolated history: that layer has about 49.4 times
as many cells and contains no additional measurements.

`ophanim-mamba-selective-ssm/1` is a small, deterministic, Mamba-inspired CPU
reference. Its selective state-space recurrence (input-dependent step, input,
and output gates) is a fixed feature reservoir; initialization fits only a
ridge next-readout head and residual calibration. It predicts both mean and
median VTEC and calibrates scores on a chronological held-out suffix. This is
not the official CUDA-fused `mamba-ssm` package, nor does initialization train
an end-to-end Mamba network. Model bytes are canonical JSON in the immutable
model artifact store, including scaling, recurrence, readout, calibration,
training-span, and training-data hashes.

Each scan rechecks a bounded overlap for late final publications, snapshots an
immutable publication cursor, predicts each new regional readout before giving
that readout to the recurrence, and commits all scores before advancing the
cursor. A download, scoring, or database failure leaves the prior successful
cursor untouched. Missing two-hour slots are represented to the recurrent model
with near-zero coverage and carried values so elapsed time still advances the
state. These placeholders are excluded from ridge supervision and residual
calibration, and they are not reported as observed or candidate readouts.

### Predictive ConvLSTM baseline

The spatial monitor exposes `GET /api/spatial/status`,
`POST /api/spatial/initialize`, and `POST /api/spatial/scan`. It requires the
Docker bulk-data plane and reuses the immutable native grids already acquired by
the Mamba history job; it never redownloads a second twenty-year copy and never
trains on the denser interpolated display layer.

`ophanim-convlstm-prior-day-residual/1` consumes twelve strictly earlier
two-hour native maps. Each input frame has a normalized VTEC channel and an
explicit valid-cell mask. The compact convolutional LSTM wraps convolution at
longitude's antimeridian, does not wrap latitude, and predicts the next map's
residual from the same UTC map one day earlier. The target map is accepted only
by the loss/scoring boundary, not by the predictor. Loss and residual scoring
exclude missing target/reference cells.

Training and evaluation use chronological source snapshots. Model weights,
normalization, training-data identity, and optimizer resume state use bounded,
canonical, pickle-free artifacts. Days are split in order into 70% training,
10% validation, 10% residual calibration, and 10% final test partitions;
causal windows never cross those boundaries. Repeated next-day midnight epochs
in daily CODE files are removed by product-day ownership before the exact
two-hour cadence is assembled. A successful scan saves native-grid residual
candidates and compares matching timestamps with the regional Mamba result.
Either model can flag a candidate independently; agreement is supporting
evidence, not a `CONFIRMED` disturbance. PyTorch is optional for the native
application (`pip install ".[spatial]"`) and is included as a CPU runtime in the
v0.1.0 Docker image.

The comparison is allowed only when both models use identical region bounds.
It reads persisted Mamba scores for the spatial run's exact time interval, so a
separate newer Mamba check cannot silently replace the evidence being compared.

### Fine regional readout

After a source is loaded, OPHANIM can sample one native map epoch on a 1°,
0.5°, or 0.25° regional grid. The UI labels 0.5° × 0.5° as the **Core**
output and selects it by default; the other choices are coarser or experimental
views. The default Central Texas bounds produce 80 rows at 0.5° spacing. Each
value is a deterministic bilinear interpolation from the surrounding native
GIM cells; exact native nodes retain their source value.
Longitude interpolation observes the same half-open antimeridian convention as
the parser, and requests without the necessary source corners fail rather than
silently extrapolating.

These rows are explicitly labeled **interpolated estimates**, not additional
measurements. The table is computed on demand from the stored source epoch and
can be exported to CSV; it does not duplicate derived rows in SQLite or inflate
forecast coverage. The native artifact, epoch, grid spacing, interpolation
version, and requested bounds remain attached to the API result so the readout
is reproducible. A later regional GNSS correction layer can use this surface as
its global background.

## Running the vertical slice from Python

The package currently exposes a Python composition root rather than a CLI:

```python
from datetime import UTC, datetime, timedelta

from ophanim.aggregation import (
    AggregationPolicy,
    GeographicBounds,
    boundary_checksum_sha256,
)
from ophanim.bootstrap import create_sqlite_application
from ophanim.domain import (
    DetectorVersion,
    ForecastMode,
    ModelVersion,
    ProcessingVersion,
    RegionVersion,
)
from ophanim.ingestion import IngestionRequest
from ophanim.ionex import IONEXV1Parser
from ophanim.workflows import VerticalSliceRequest

run_at = datetime.now(UTC)
origin_at = datetime(2026, 8, 31, 0, tzinfo=UTC)

bounds = GeographicBounds(
    south_latitude_degrees=29.0,
    west_longitude_degrees=-100.5,
    north_latitude_degrees=32.5,
    east_longitude_degrees=-96.0,
)
region = RegionVersion(
    region_version_id="central-texas-v1",
    name="Central Texas",
    boundary_ref="central-texas-bounds-v1",
    boundary_checksum_sha256=boundary_checksum_sha256(bounds),
    created_at=run_at,
)
processing = ProcessingVersion(
    processing_version_id="ionex-regional-v1",
    code_revision="replace-with-git-revision",
    configuration_hash="replace-with-config-hash",
    parser_version=IONEXV1Parser.parser_version,
    created_at=run_at,
)
model = ModelVersion(
    model_version_id="persistence-v1",
    model_type="persistence",
    configuration_hash="replace-with-config-hash",
    created_at=run_at,
)
detector = DetectorVersion(
    detector_version_id="residual-z-v1",
    scoring_method="residual_zscore",
    threshold=3.0,
    configuration_hash="replace-with-config-hash",
    created_at=run_at,
    calibration_data_version="replace-with-calibration-version",
    calibration_residual_mean_tecu=0.0,
    calibration_residual_standard_deviation_tecu=1.0,
    calibration_sample_count=2,
)

app = create_sqlite_application(
    database="var/ophanim.sqlite3",
    artifact_directory="var/artifacts",
    region_bounds={region.boundary_ref: bounds},
    # Set this to the number of cells the selected IONEX grid should contain.
    aggregation_policy=AggregationPolicy(expected_cell_count=4),
    detector_version_id=detector.detector_version_id,
)

result = app.vertical_slice.run(
    VerticalSliceRequest(
        ingestion=IngestionRequest(
            provider="igs",
            product="gim",
            source_uri="path/to/source.INX.gz",
            # Use a larger value for products that should supersede
            # preliminary or backfilled revisions of the same observations.
            revision_priority=0,
        ),
        region=region,
        processing_version=processing,
        model_version=model,
        detector_version=detector,
        forecast_origin=origin_at,
        forecast_horizon=timedelta(hours=2),
        mode=ForecastMode.HINDCAST,
    )
)
```

The pending/revision side entrance is the same reconciler used by the vertical
slice. Targeted reconciliation preserves the detector selected for that
forecast:

```python
summary = app.reconciler.reconcile_forecast(
    forecast_id=result.forecast.forecast.forecast_id,
    valid_at_or_before=datetime.now(UTC),
)

# A bulk sweep is also available when every selected pending forecast belongs
# to this reconciler's configured detector version.
bulk_summary = app.reconciler.reconcile(valid_at_or_before=datetime.now(UTC))

# After a revised actual has been ingested and aggregated:
replacement = app.reconciler.reassess_scored(
    forecast_id=result.forecast.forecast.forecast_id,
)

# If a forecast exhausted its grace period before an actual arrived:
late_decision = app.reconciler.retry_insufficient(
    forecast_id=result.forecast.forecast.forecast_id,
)
```

The composition root also exposes the durable-store boundary for model
registration and result queries:

```python
with app.unit_of_work_factory() as unit_of_work:
    decisions = unit_of_work.decisions.search(
        valid_at_or_after=origin_at,
        valid_before=datetime.now(UTC) + timedelta(microseconds=1),
    )
```

The calibration numbers above are structural placeholders, not scientifically
meaningful defaults. A real run should derive them from a chronologically
separate residual-calibration set.

For v0, live `issued_at` is the serialized run-start and data-availability
cutoff: the workflow stamps it only after acquiring the database transaction
lock. It rechecks the clock after model execution and refuses to persist a live
forecast that did not finish before `valid_at`. Hindcast/replay requests may
carry an explicit issuance time; omitting it asks the workflow to timestamp the
run. The pure model runner requires an explicit issuance time because it owns
no clock or persistence boundary. A separate publication-completion timestamp
can be added when the pipeline gains a delivery surface.

Run the test suite with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Deliberate v0 limits

- Automatic CODE discovery supports public final/rapid products from 2002,
  including legacy `.Z` and modern `.INX.gz` files. CDDIS/Earthdata
  authentication, conditional requests, and a scheduled unattended collector
  are not yet implemented.
- Raw global cells are still normalized into SQLite rows for the current
  forecast/readout workflow. Docker additionally stores both the immutable
  native grid and its 0.5° derived core grid in Zarr and PostgreSQL, but
  forecasts do not read that bulk archive yet. Local IONEX uploads are not
  mirrored until the parser boundary exposes their grid axes explicitly.
- Ordinary single-map bulk ingestion is synchronous with its interactive CODE
  download. The historical Mamba backfill runs in a resumable background job,
  but a scheduler, retention policy, Zarr-orphan garbage collection, and cloud
  or object-store backend remain future work. Atomic group publication
  currently relies on a local filesystem rename.
- Fine regional tables currently use spatial interpolation only. They make the
  field easier to inspect but do not add independent observations, calibrated
  uncertainty, or the future regional GNSS correction layer.
- The residual z-score is a baseline and is not yet stratified by local time,
  season, location, or solar conditions.
- The Mamba monitor is a laptop-scale CPU reference whose ridge readout and
  residual calibration are fitted on one region's mean/median sequence; the
  selective-SSM reservoir itself remains fixed. It does not yet localize
  abnormality within the region, incorporate solar/geomagnetic covariates,
  compare against independent GNSS stations, or promote a candidate readout to
  a confirmed disturbance. A full twenty-year first initialization requires
  thousands of public downloads and can run for hours; progress is resumable
  rather than instantaneous.
- The ConvLSTM is likewise an experimental laptop-scale baseline. Its native
  71 × 72 inputs localize residual evidence only to the source grid spacing;
  the 0.5-degree display grid is not extra evidence. CPU training over twenty
  years is intentionally a long-running job. It does not yet ingest solar-wind
  or geomagnetic covariates, quantify calibrated physical-event probability,
  or turn a candidate into a confirmed disturbance.
- Mamba's SQLite source-day ledger and the optional PostgreSQL/Zarr bulk archive
  are a coupled backup set once native grids have been published. If the bulk
  named volumes are deleted while `var/desktop/` is retained, v0 does not yet
  verify and automatically republish a source day whose SQLite row already has
  a native Zarr grid identifier. Restore those stores together, or deliberately
  rebuild the affected monitor/archive state.
- `CONFIRMED` disturbance evidence and automatic calibration/backtesting remain
  future workflows.
- SQLite schema changes currently require an explicit migration; unknown
  versions are rejected rather than guessed at.
- Source bytes are fsynced before database commit. A database failure can leave
  an unreferenced content-addressed file for later garbage collection, but not
  a committed record pointing at an incompletely written file.
