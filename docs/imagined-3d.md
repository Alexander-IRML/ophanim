# From evidence to an imagined 3D event

In Event Lab, select **Imagine in 3D**. Studio can propose a shape from the saved
evidence or use an explicitly illustrative fallback. Alternatively start directly
in Studio. Choose folded wave ribbons, corrugated front curtains or a twisting
jet bundle, a variation seed and an artistic style. Optional mechanics overrides
are recorded as chosen values. Generate the preview, then render with Blender or
make a short animation. Saved projects retain recipes and source links.

## Ownership and semantics

1. **Core:** preserves native TEC, discovery evidence and scientific confidence/
   abstention. Its separate `ImaginedVolume` interchange contract does not make a
   volume scientific: it requires dimensionless synthetic material semantics.
2. **Experiments:** `hypotheses.py` freezes available evidence, records provenance
   and chooses a revisable recipe. `volume_model.py` creates `density(time,z,y,x)`
   through seeded forcing, advection and diffusion. The z coordinate is a chosen
   altitude, not an estimated altitude profile. Apparent scientific flow is never
   relabeled as plasma velocity or silently required to permit imagined motion.
3. **ShawtyNet:** consumes the saved volume, creates a depth-integrated preview,
   extracts geometric surfaces and filaments, and renders artistic emission. It
   imports the core contract, not the experimental simulator.
4. **Workspace:** composes those stages, queues bounded work and saves lineage.
   A prior observation-derived science project for the same candidate may supply
   supported wave estimates. Missing evidence remains missing.

The default model has 12 timestamps on a 41 × 41 × 25 grid, spanning an assumed
800 km square and 180–480 km altitude. It evolves forced passive material, **not
mass-conserving electron transport, electrodynamics or a calibrated forecast**.
There is no observation-derived vertical solution. Conservative server-side
cell/step/work limits apply before arrays are allocated. Seeds reproduce the
same recipe with the same software version; manifests identify input and code.

The 3D camera uses an automatically distanced artistic orbit. Heading, pitch,
roll and field of view can be chosen; ground-observer coordinates are only
supported by the older 2D rendering path. A quick volume preview and a geometric
Blender render are different representations, not identical image promises.

The downloaded scene ZIP keeps a strict `render_package/` folder separate from
the outer project/recipe JSON. Its README gives a bounded Blender replay command.
Do not add files inside that package or execute scripts from untrusted downloads.

Scientific uncertainty does not cap artistic ambition: when a wave/front/jet is
unconfirmed, it may still be explored as a hypothesis, clearly marked as such.
Trying alternative shapes does not generate additional scientific evidence.

## Scope and next milestone

Everything here executes locally, including Blender. Exports are available;
there is no automatic cloud fallback, remote job submission or credential setup.
IFM is explicitly deferred to **v0.3**. Remote simulations are an accepted future
direction, subject to the [security checklist](simulation-security.md) and an
explicit provider/data/cost decision. The [roadmap](../ROADMAP.md) retains that
milestone rather than promising a reminder outside this application.
