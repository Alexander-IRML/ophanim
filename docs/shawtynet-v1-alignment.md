# ShawtyNet V1 alignment after the implementation audit

The original 72-section brief remains the specification. Earlier checked lists
recorded that pipeline stages existed; they did **not** prove that every stage
met its scientific or aesthetic acceptance criterion. This record separates
implemented behavior, reproducible evidence, and limitations.

## What changed

| Brief area | Implemented correction | Verification |
| --- | --- | --- |
| §§2–3, 27–30, 53: semantics and confidence | Synthetic TEC/RMS stay synthetic; observation support, source RMS, measurement reliability, motion observability/fit, wave and event confidence stay separate. | State-contract and reliability tests; saved field-level metadata. |
| §§15–19, 47–48: motion and mechanics | Stationary photometric change competes with translation locally and regionally. Raw vectors/scores stay auditable. A separate interpretation channel requires pair admission, supplied RMS and the existing local gate before vectors drive residuals, kinematics, wave consistency or directional art. | Gradual growth, mixed motion/growth, correlated-noise safeguards, known translation and analytic divergence/vorticity/strain/residual tests. |
| §§20–24, 50–51: waves | Signed Fourier phase remains intact; temporal occupancy/effective cycles, structural orientation, competing peaks, supported directional agreement and source RMS affect evidence. | Sustained wave versus short burst, uncertainty sensitivity, direction/phase and existing sampling-gate tests. |
| §§25–28: event interpretation | Explicit candidate focus or a dominant connected anomaly avoids dilution by a large quiet crop. Compactness, elongation and competing interpretations are recorded. | Compact anomaly/front fixtures, focus-boundary and crop-context tests. |
| §§6, 28–30, 58–59: temporal state | Bounded exact-time event/wave windows and selected adjacent flow pairs; display frame separate from the primary scientific target. Unsupported intervals are never filled with a copied event. | Sequence tests, causal-prefix invariance, cancellation/budget tests, per-epoch diagnostic plots. |
| §§28, 30, 66–67: replaceable representation | Versioned validated `DynamicState`, concrete engineered adapter and explicit decoded learned-output adapter. Renderer accepts this state without selecting a scientific model. | Units, confidence, source semantics, checkpoint/input references and invalid mappings tested. |
| §§31–40, 54–56, 64F–H: visual language | Orientation-conditioned front ribbons, flow-aligned fibers, coherent wave modulation, kinematic detail, temporally stable noise, quiet veil and disturbed fraying; bounded radiance replaces clipped emission. | Six analytical known-state fixtures, portable packages, non-saturation/distinctness/elongation/no-flicker assertions; actual Blender gallery reviewed separately. |
| §§41–42, 64I: photo integration | Manual observer coordinates/altitude, heading/pitch/roll/FOV; smooth horizon fade, saturation, exposure, RGB grade, foreground and cloud masks. | Upload safety, matching crop/mask geometry, continuous fade, foreground preservation and UI/API tests. |
| §§29, 52, 55: diagnostics | Spatial spectrum with selected candidate and competing directions, carrier envelope/occupancy, individual confidence components, morphology metadata, kinematics and event timeline. | Actual PNG/HTML report generation tests. |

## State and timeline contract

`ophanim.core.state.DynamicState` carries a metric, spatially indexed dataset,
event interpretation, optional exact-time timeline and producer provenance.
The deterministic `EngineeredModelAdapter` is executable through the shared
model interface. `LearnedStateAdapter` only accepts **already decoded** fields
with explicit names, units, semantics and model/input artifact references. It
does not magically turn a latent vector into physical state, train a model, or
validate a research model's scientific claims.

New science runs retain `dynamic.zarr`, `event.json`, `science_packet.json` and
add `state_contract.json`; sequence runs also publish `events.json`. Each timeline
frame records its own status, wave window, actual observation support and event.
The ordinary single-target API is preserved. Old saved runs stay immutable and
usable as stills, but must be reanalyzed before a time-specific animation.

Retrospective analysis reuses preprocessing and derivatives. Causal sequence
analysis recomputes each eligible prefix so later samples, revisions and
normalization cannot influence earlier states. The desktop limits the number
of selected inference epochs to 12; it does not analyze every frame of a large
archive or silently launch training.

## Window and resolution policy

The scenario's energetic interior frame is a **display choice**, not the end of
the main scientific record. Front previews use spatial-gradient energy so a
uniform plateau after the transition leaves view is not selected as the event.
Each requested animation/display epoch still needs
its own sufficient prior wave support. The first frames may legitimately lack
wave or motion evidence.

Native observational products use a cadence-compatible periodicity search band:
for hourly observations, the six-samples-per-period gate starts at six hours.
This is not a claim that hourly GIM can recover short-period TIDs. Original
spatial support, missing epochs, baseline attenuation, actual occupancy and
effective-cycle gates can still reject every candidate. Interpolation to a
finer grid never changes those limits.

## Reproduce the acceptance evidence

Use the existing science environment, with bounded thread counts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-shawty/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-shawty/bin/python scripts/shawtynet_visual_acceptance.py --blender /path/to/blender --passes
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-shawty/bin/python scripts/shawtynet_photo_demo.py --blender /path/to/blender
```

Omit `--blender` from the visual acceptance script for material-only previews.
The gallery uses **analytical known-state** inputs to isolate the rendering
boundary; it is not a test that the detector recovered those states. Numerical
recovery is tested separately. Generated galleries and reports live in ignored
`var/` directories and preserve producing code/settings rather than replacing
old evidence.

## Limits that remain deliberate

- These are heuristic scientific interpretations, not calibrated probabilities,
  verified physical diagnoses or plasma velocities. Brightness/transport
  ambiguity cannot be universally resolved from TEC images alone.
- Source RMS sensitivity is not full covariance/error propagation; unknown RMS
  stays unknown. A numerical model cannot manufacture missing observations.
- Held-out spatially smooth, temporally correlated noise can still produce
  confidently wrong **raw** apparent-motion fits. In the retained adverse
  fixtures, 80–82% of locally admitted raw scores have large directional error,
  even though pair-level summaries with an RMS proxy abstain. Raw fit scores
  are therefore not used directly for directional art or motion-derived
  diagnostics: the separate conservative eligibility channel also requires
  pair-level admission and known RMS. These conjunctive gates are **not** an
  accuracy certification, and the adverse raw estimates remain inspectable. The
  earlier independent-noise test family must not be treated as universal
  validation or an operational error bound.
- Existing real-storm evidence checks source acquisition and qualitative TEC
  disturbance, not independently measured ground-truth motion or wave accuracy.
- The renderer uses curved, layered sheets. Ribbon and jet forms are artistic
  language; this is not a volumetric plasma/3-D reconstruction or fluid solver.
- Learned-output adapters and agent-training interfaces are extension points.
  No new learned dynamics, operational classifier or RL training system is
  claimed. Those are explicitly deferred by §§65–67 of the brief.
- Manual camera/mask controls do not provide automatic photo registration,
  segmentation or a claim that the ionosphere would be visible in a photograph.
- Automated visual metrics detect regressions, not aesthetic success. Blender
  output and the curated photograph require visual review in addition to tests.

## Reviewed local evidence — September 18, 2026

- [Six-state actual Blender gallery](../var/shawtynet-visual-acceptance/e04925a1b485efeab37cd8f3/index.html),
  also available as a [comparison image](../var/shawtynet-visual-acceptance/e04925a1b485efeab37cd8f3/gallery.png).
  All six used the same camera/style, 640×426 pixels, 12 samples and two threads.
  The front was revised after the first real render showed a dark seam between
  bright plateaus; the reviewed version emphasizes a localized curtain/fold.
  Translation has visible aligned fibers; uncertainty removes those fibers while
  retaining the broad envelope. None of the six contains white-clipped pixels
  under the recorded near-white/alpha metric.
- The translation's multilayer EXR was inspected for actual channel names:
  Combined, Depth, Emission, ArtisticBaseCoverage and ArtisticDetailCoverage are
  present. Depth is distance to artistic shell geometry, not ionospheric depth.
  The receipt parser records and validates these actual headers.
- [Updated curated photographic composite](../var/shawtynet-photo-final/science/cfb3283056405d3b39303778/visuals/e6a42ec0c3cf3b9c65ab8d3d/renders/b9e2e9ef0e606a2e59ff6634/composites/b2895e6bb70c226eaa417bb2/composite.png).
  The source is the existing credited NPS photograph, manually aligned with a
  synthetic wave packet. All four stage manifests verified; all 557,717 fully
  masked foreground pixels were unchanged. The scientific wave was admitted
  with heuristic confidence 0.926; global full-vector flow remained low-confidence.
  The final composite retains broad wave modulation but withholds unsupported
  flow-aligned detail. It was rendered at 1200×802 pixels with 24 samples and
  visually reviewed after the interpretation gate was introduced.
- [Current numerical sensitivity report](../var/shawtynet-validation-v2/dynamics-confidence-sensitivity-with-interpretation-eligibility.json).
  In the original 31-case motion family, confidence/error Spearman correlation
  is −0.818; low/high-confidence tertiles have mean endpoint errors 104.30/2.24
  m/s. All seven admitted pair estimates stay below the test's half-true-speed
  error threshold. **The additional correlated-noise counterexamples do not
  share that result**; both pair-level and local errors must be considered.
  The separate presentation gate admits zero pixels across all 12 retained
  adverse correlated-error cases. The clean, known-RMS translation control
  retains all 1,024 interior vectors with mean endpoint error 0.972 m/s. Raw
  failures remain in the report; this is not independent operational validation.
- [Authenticated application workflow](../var/shawtynet-v1-workflow-final.json): cached
  native candidate → science/art, hypothetical front → still → three-frame
  animation → uploaded-photo/foreground-mask composite. All nine report/image/
  scene/recipe download endpoints returned real nonempty artifacts. The native
  example correctly abstained from motion/wave interpretation (low confidence /
  invalid support); pipeline completion is not a scientific detection claim.

## Final regression and application checks

- Frozen-source scientific-environment suite: 448 tests run, 446 passed and
  two skipped. Dependency-light suite: 448 tests run, 268 passed and 180 skipped
  because their optional scientific/artistic dependencies are unavailable.
  The full run emitted two unclosed-SQLite-connection `ResourceWarning`s; these
  did not fail the suite and are not evidence that resource lifecycle cleanup
  has been fully audited.
- Actual Chrome desktop and 390-pixel mobile checks passed across all five
  workspace tabs, with no horizontal overflow or JavaScript exceptions.
  Isolated browser checks verified form serialization/authentication, camera
  defaults, masks, unsaved-setting render guards and latest-output selection.
  Browser request interception tests UI wiring; the separate authenticated
  workflow above verifies the real backend.
- Actual Blender 5.1.2 rendered the six-state gallery, curated photograph,
  application still and three-frame animation. Tests also verified the real
  optional multilayer EXR channels, not just requested settings.
- Offline wheel packaging, Python compilation and `git diff --check` passed.
  Docker Desktop's WSL integration was unavailable, so no container rebuild is
  claimed. No archive download or model retraining was launched for acceptance.
- The updated desktop application responded successfully at
  `http://127.0.0.1:8765/` after the final workflow. Existing project history and
  source/scientific artifacts were preserved.

The acceptance record above was completed before the v0.2.0 release; those
historical artifacts retain their original version metadata. The current
application is v0.2.0, with release changes recorded in `CHANGELOG.md`.
Earlier saved artifacts remain immutable. Re-export old visual
packages if their embedded Blender script differs from the installed trusted
renderer; the application does not execute arbitrary uploaded Python scripts.
