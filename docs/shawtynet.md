# OPHANIM dynamics and ShawtyNet V1

ShawtyNet turns an inspectable scientific state into explicitly artistic
visualization fields, a curved high-altitude shell, a transparent Blender render,
and a manually aligned photograph composite. The science also works on its own.
It does not reconstruct a physical 3-D ionosphere or measure plasma velocity.

## Start on this laptop

Restart the desktop application using its existing launcher. The new
**ShawtyNet · dynamics & visual research** panel offers a synthetic example or
analysis of already acquired TEC. No Mamba initialization is needed. The panel
runs work in the background and provides scientific/visual diagnostic reports.
The preview is an artistic color-control texture, not a finished Blender scene.

The project launcher uses `.venv-shawty/bin/python` when present. Set
`OPHANIM_PYTHON` to select a different Python explicitly. The isolated environment
contains the original PostgreSQL/model dependencies as well as the science extra.
No system Python packages or browser defaults are changed.

For another installation:

```bash
python3 -m venv .venv-shawty
.venv-shawty/bin/python -m pip install -e '.[data,shawtynet]'
# Optional existing ConvLSTM runtime, CPU-only:
.venv-shawty/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3'
scripts/launch_desktop.sh
```

The Docker build includes the extra. Rebuild the image to get the new code;
existing source databases, native grids and model checkpoints are unchanged.
Blender is an external rendering backend, not bundled in the application image.

## Independently rerunnable stages

Commands below use the project virtual environment. The installed entry point
`ophanim-shawtynet` is equivalent to `python -m ophanim.shawtynet_cli`.

```bash
.venv-shawty/bin/python -m ophanim.shawtynet_cli demo --case plane_wave
.venv-shawty/bin/python -m ophanim.shawtynet_cli analyze examples/shawtynet/synthetic-wave.yaml
.venv-shawty/bin/python -m ophanim.shawtynet_cli analyze examples/shawtynet/desktop-archive.yaml
.venv-shawty/bin/python -m ophanim.shawtynet_cli visualize SCIENCE_RUN --style examples/shawtynet/ghost-atmospheric.yaml
.venv-shawty/bin/python -m ophanim.shawtynet_cli render VISUAL_RUN --blender /path/to/blender
.venv-shawty/bin/python -m ophanim.shawtynet_cli composite RENDER_RUN photograph.jpg --foreground-mask mask.png
.venv-shawty/bin/python -m ophanim.shawtynet_cli inspect SCIENCE_RUN
```

Each stage prints its exact output directory as JSON. `SCIENCE_RUN`, `VISUAL_RUN`
and `RENDER_RUN` mean the corresponding directories printed by earlier commands,
not the containing output root. `--verbose` precedes the command.

The `run` command composes stages without merging their identities:

```bash
.venv-shawty/bin/python -m ophanim.shawtynet_cli run examples/shawtynet/synthetic-wave.yaml --render --blender /path/to/blender
```

Pass `--photo` to add compositing, with `--camera` describing the photograph's
manually chosen camera. Render dimensions must match the photograph; the program
does not silently stretch either image. `--frame` explicitly chooses a frame;
otherwise export uses the scientific event's target timestamp.
Normal visual runs process just that selected frame to keep laptop memory use
bounded. `visualize --sequence` explicitly generates the full visual time series;
render-sequence helpers can then export each frame. Original science always keeps
the complete analysis sequence. The desktop launcher defaults numerical workers
to two threads, respecting explicit `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` settings.
Scientific cubes and visual sequences also have configurable total-cell budgets
(`max_cube_cells`, `max_render_voxels`, each two million by default). Blender
defaults to two render threads; `render --threads` explicitly changes that limit.

This laptop already has a Windows Blender installation; from WSL use:

```bash
.venv-shawty/bin/python scripts/shawtynet_photo_demo.py --blender '/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe'
```

That example uses the credited public-domain NPS photo and manual skyline in
`examples/shawtynet/`. Its TEC is synthetic, its camera is approximate and its
time is not matched to the photograph. It is **an artistic demonstration, not
evidence of an observed ionospheric event**. See `PHOTO-CREDIT.md` there.
Windows Blender arguments are translated with `wslpath`; rendering runs headless
without modifying an existing Blender scene or user preferences.

Animation is exposed through `export_render_sequence` / `render_sequence` in
`ophanim.shawtynet`: each source time produces its own package/frame. LIC noise
is seeded and advected for temporal continuity. Automatic camera calibration,
cloud segmentation, video encoding and event-ranking automation are not required
by this V1 and are not claimed.

## Inputs and configuration

Analysis requests are JSON or safe-loaded YAML with `input` and `analysis` keys.
Paths inside a request are relative to that request file. Unknown configuration
keys are rejected. The resolved scientific configuration is saved in every run.

Input kinds:

| Kind | Required location | Selection |
|---|---|---|
| `desktop` | `data_directory` containing existing `ophanim.sqlite3` and `artifacts/` | Optional artifact IDs, UTC start/end, bounds |
| `archive` | `zarr_root` of the existing verified native store | Optional grid-set IDs, UTC start/end, bounds |
| `ionex` | `path` to existing IONEX/gzip bytes | Provider/product and known `available_at` for causal use |
| `synthetic` | `parameters` for `make_synthetic_dataset` | Explicit mathematical truth, seed, cadence, scales |

The sensing adapter never downloads a second copy, initializes a trained model,
or writes into the native source archive. It selects revisions deterministically
and preserves the input checksums and actual local availability timestamps.
Original IONEX RMS is retained where supplied; older TEC-only Zarr arrays do not
magically acquire uncertainty. The desktop adapter can recover RMS from retained
original bytes without changing the immutable legacy parser identity.

Source bounds describe the **analysis region**, not necessarily the eventual
photographic crop. A small scene may need a much larger analysis region to contain
enough original grid cells and spatial cycles. Local projection extent/distortion
checks reject unsuitable global windows.

## Scientific contract

`ophanim.dynamics.analyze_dataset(raw, AnalysisConfig(...))` returns an
`AnalysisResult(dataset, event, capabilities)`. Numerical dependencies are optional
and do not load when importing the base OPHANIM application.

Raw datasets have `tec`, `observed_mask` and UTC `time`, either on native
`(time,lat,lon)` axes or explicit metric `(time,y,x)` axes. Causal inputs also
require `available_at(time)`. `source_metadata` records **original** support.

Dynamic datasets use ascending `x` and `y` in metres, UTC time and 2-D geographic
coordinates. `x_km`/`y_km` are presentation coordinates. Variables include:

- Source TEC, observed/temporal/spatial/missing masks, source RMS/uncertainty and
  observation support. GIM values are already estimated source products.
- Smoothed/background TEC, absolute/relative/robust-normalized dTEC, saved
  normalization scale and center. dTEC is not a forecast-model residual.
- Physical temporal/spatial derivatives, gradient magnitude, Laplacian, tensor
  eigenvalues, band orientation and coherence.
- Apparent TEC-feature velocity, bearing, directional observability, fit,
  forward/backward and warp diagnostics, growth ambiguity and confidence.
- Apparent-flow divergence/vorticity/strain and advection residual. These are
  descriptive image-motion quantities, not plasma-fluid measurements.
- Regional wave/event confidence; the event document retains the complete
  heuristic score vector and individual channel statuses.

Every scientific variable records units, semantics and its semantic class.
Unavailable physical values are NaN with explicit statuses, not fake zero motion
or an automatically quiet event. Source presence, source RMS, observability and
fit quality are not interchangeable.

Wave estimates retain signed temporal/spatial frequencies, phase and its origin,
period/wavelength/speed/bearing, amplitude, concentration and component evidence.
Conjugate FFT peaks represent the same propagation vector. Optical flow is an
independent consistency check. Waves smaller/faster than source support are
withheld even when an interpolated array has many more cells or frames.

Six actual samples per period and three observed cycles are configurable
conservative temporal gates. Directional native spatial support, source resolving
scale when known, geographic extent and missingness also constrain admission.
These rules are safeguards, not proof of physical resolving power. A 0.5-degree
interpolated GIM remains limited by its original 5-by-2.5-degree support.

Smoothing uses physical lengths. Background methods include Savitzky–Golay,
rolling median and an explicitly supplied constant (useful for synthetic truth).
Missing support is propagated through operators. Centered rolling medians lack
full support at sequence edges: choose a target inside that support, or accept
an unavailable anomaly channel. The desktop preset explicitly targets six hours
before its latest source time for its 12-hour retrospective median.

## Causal versus retrospective

Retrospective analysis is the art default and may use surrounding observations.
Causal analysis requires an explicit timezone-aware `as_of` and source availability
metadata. Later observations and later-available revisions are excluded before
normalization, interpolation and inference. Causal baselines/derivatives use past
support. Local ingestion time is conservative availability evidence; it is not
an invented historical publication time.

Do not feed retrospective results into live forecast scoring as though they were
available at the target time. The run records its mode, cutoff, target, support
interval and input revisions so this distinction remains visible.

## Artifact layout and firewall

```text
RUN_ID/
  config.json
  request.json                 # when a request was supplied
  dynamic.zarr                 # xarray-compatible Zarr format 2
  event.json
  science_packet.json
  diagnostics/report.html     # plus scientific plots
  manifest.json               # complete marker and every artifact checksum
  visuals/STYLE_RUN_ID/
    style.json
    visual.zarr
    visual_packet.json
    visual_diagnostics/report.html
    render_package/           # curved mesh, PNG/NPY controls, standalone script
    manifest.json
    renders/RENDER_ID/
      render.png
      render_packet.json
      manifest.json
      composites/COMPOSITE_ID/
        composite.png
        composite.composite.json
        composite_packet.json
        manifest.json
```

Run identity incorporates exact input arrays/provenance, resolved configuration,
algorithm code digest and dependency versions. Each stage publishes atomically
from a private staging directory. Reruns reuse matching completed results;
corrupted or incomplete results fail verification. The source archive's existing
Zarr format/schema is untouched. Visual recipes, Blender builds/render settings
and photo/mask bytes have separate identities.

The artistic layer consumes science but never modifies it. Quiet supported TEC
can retain a faint veil through an explicit artistic opacity floor. Weak flow
confidence removes fibers, not the base structure. Weak wave evidence removes
wave deformation. Shell altitude, displacement exaggeration, procedural detail,
colors, boundary feathering and atmospheric fading are explicitly artistic.

## Verification and limits

```bash
.venv-shawty/bin/python -m unittest discover -s tests -v
```

Tests cover units/signs, known translations/growth/fronts/waves, sampling
degradation, masks, causal replay, source uncertainty, immutable stage replay,
confidence-separated art, seeded LIC, geometry, manual occlusion, and desktop
authentication/lifecycle. Scientific diagnostics expose the intermediate fields;
they are not replaced by a plausible-looking render.

Run the bounded numerical sensitivity report independently:

```bash
.venv-shawty/bin/python scripts/benchmark_dynamics.py
```

It records 31 flow cases and eight wave cases, including finite envelopes,
amplitude changes, weaker crossing waves, measurement noise and missing support.
Each result retains known-truth errors alongside confidence, exact configuration
and seeds. The default 0.4 flow-status gate rejects the inaccurate high-displacement
cases in this finite sweep; it is not a general accuracy guarantee. The saved
report also separates full-vector observability from the normal motion of bands.

Confidence/event values remain experimental heuristics, not calibrated
probabilities. Synthetic validation does not establish real-world detection
accuracy or explain a disturbance's physical cause. Operational validation with
independent events/quiet controls and better-resolved sources remains research
work. The architecture deliberately permits new source adapters and learned
dynamic-state providers without importing Blender into OPHANIM sensing.

See the [recorded implementation checks and example outputs](shawtynet-implementation.md)
and the [actual May 2024 storm sanity check](shawtynet-known-storm.md). The latter
uses independently identified event timing and real CODE products, with an
explicitly descriptive preceding-day comparison rather than a claimed quiet-day
calibration.
