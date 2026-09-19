# Subsystem boundaries

OPHANIM remains one local application with three independently usable subsystems.
The desktop/workspace layer composes them; it is not an algorithm library.

```text
                        desktop / workspace / CLI
                          │        │        │
                          ▼        ▼        ▼
                       core ◀─ experiments  ShawtyNet
                         ▲                     │
                         └─────────────────────┘
```

- **Core:** acquisition, native-source mapping, discovery, reusable model
  interfaces, scientific analysis and immutable scientific artifacts. Existing
  implementations in `sensing`, `dynamics`, forecasting, monitoring and storage
  remain usable; the `core` namespace is a stable boundary, not a second copy.
- **Experiments:** explicitly hypothetical scenarios, known-truth test fields,
  imagined 3D hypothesis volumes, degradation tests, model-evaluation contracts and future agent-training seams.
  Experiments can consume the core without importing or running ShawtyNet.
- **ShawtyNet:** artistic interpretation, visual fields, camera/render packages,
  Blender rendering and photo compositing. It consumes a scientific artifact or
  a separately typed imagined-volume artifact;
  it neither selects a simulation nor acquires source observations.

The composition layer owns `ScanRun → CandidateEvent → Scenario → VisualProject`
handoffs and may connect these branches. Neither the core nor ShawtyNet imports
the experiments package. AST and dependency-free-import tests enforce this.

## Public entry points

| Purpose | Entry point |
| --- | --- |
| Publish scientific analysis | `ophanim.core.runs.analyze_run` |
| Verify immutable artifacts | `ophanim.core.artifacts.verify_run` |
| Implement an inference adapter | `ophanim.core.models.ModelAdapter` |
| Execute the engineered baseline through the model interface | `ophanim.core.models.EngineeredModelAdapter` |
| Validate engineered or explicitly decoded learned state | `ophanim.core.state.DynamicState`, `EngineeredStateAdapter`, `LearnedStateAdapter` |
| Analyze bounded exact-time state evolution | `ophanim.dynamics.sequence.analyze_sequence` |
| Build a hypothetical candidate recipe | `ophanim.experiments.scenario_from_candidate` |
| Generate its numerical field | `ophanim.experiments.make_scenario` |
| Propose a revisable 3D hypothesis | `ophanim.experiments.hypotheses.hypothesis_from_evidence` |
| Evolve/publish imagined 3D material | `ophanim.experiments.volume_model.publish_hypothesis_run` |
| Exchange a strictly non-scientific volume | `ophanim.core.volumes.ImaginedVolume`, `read_volume` |
| Preview/render an imagined volume | `ophanim.shawtynet.volume.visualize_volume_run`, `render_volume_run` |
| Define frozen evaluation partitions | `ophanim.experiments.DatasetSplit`, `ExperimentSpec` |
| Create/revise an artistic interpretation | `ophanim.shawtynet.runs.visualize_run` |
| Render or composite an interpretation | `ophanim.shawtynet.runs.render_run`, `composite_run` |

`ophanim.science_runs` remains a lazy compatibility facade for existing scripts.
Scientific run identities hash core code, not experimental or artistic code.
Generated scenarios carry their generator identity; downstream visual identities
include their own code and the immutable parent scientific manifest.

## Scenario semantics

A recipe records all editable parameters and their per-field provenance:
`observed`, `estimated`, or `chosen`. Current candidate-derived recipes inherit
only a bounded amplitude proxy and centroid, both marked **estimated**. Width,
velocity, period, time span, finer spacing and event shape are **chosen**. User
overrides are always marked chosen, even when replacing an inherited estimate.
The candidate and native snapshot IDs remain attached to the recipe.

These are mathematical TEC fields, not a recovered small-scale event and not a
physical ionosphere simulation. Selecting finer scenario spacing adds no new
observations. Negative native deviations can suggest a hypothetical localized
depletion; that selected shape is still not a causal diagnosis.

The scenario is limited to 200,000 space-time cells before allocating arrays,
uses a chosen 20 TECU background and five-minute cadence, and records a seed.
`width_km` is the Gaussian scale (or front width parameter), not a source-derived
physical event size. For waves the chosen wavelength is
`max(4 × spacing_km, 2 × width_km)`; period sets phase speed, while `speed_m_s`
sets envelope translation. A front uses speed normal to its orientation.
Quiet and growth-only fixtures do not use translational speed. Core capability
checks still report when the chosen duration/cadence cannot support a wave fit.

## Independent experimentation and future agents

`DatasetSplit` declares a half-open UTC time interval and checksum-identified
partition artifacts. `ExperimentSpec` requires ordered, disjoint training,
validation and test intervals, separate partition references and content hashes,
a model identity, seed and named metrics. References are never automatically
downloaded or opened.
Structural validation prevents obvious split mistakes; it cannot verify the
truth of an external artifact's stated time range or a model's training history.

The scientific-state adapter additionally enforces metric coordinates, declared
units and confidence channels, source semantics and exact event/timeline epochs.
Decoded learned state retains its checkpoint and input artifact references; a
latent vector alone is not accepted as physical state. The renderer consumes
this contract without importing the experiment or training subsystem.

`ModelAdapter` is an inference protocol that can wrap a deterministic baseline,
an existing trained model, or a later research model. `MetricResult` carries an
explicit split and unit. These contracts do not implement training or benchmark
execution by themselves.

`AgentEnvironment` and `AgentTrainer` are extension protocols only. The runtime
capability explicitly reports **not configured**. No reinforcement-learning
framework, agent policy, reward model, training run or trained agent is included.
When implemented later, a trainer must return a real checksum-identified model
artifact and use the same declared splits, seeds and evaluation conventions.
