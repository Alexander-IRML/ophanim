# Simulation boundary and remote execution security

## Current behavior

The imagination workflow runs locally. It accepts a bounded numeric recipe and
saved workspace identifiers, not code, model downloads, arbitrary URLs or shell
commands. It does not upload observations, photographs, masks or credentials.
The local application uses its existing loopback host checks, mutation token,
single-worker queue, resource budgets and manifest-verified artifact serving.

Blender is a server-configured executable. Renderer scripts are application-owned;
simulation packages are plain data. Hash verification detects modification, not
trustworthiness: a downloaded script is not safe merely because it has a hash.
Never execute a returned `.blend`, Python script or pickled array automatically.

## Required before enabling a remote backend

1. Choose a provider and document account, region, retention, costs and the exact
   data leaving the laptop. Obtain explicit approval for those choices. No
   automatic remote fallback when local computation fails.
2. Use a server-configured HTTPS endpoint allowlist, verify certificates, reject
   unsafe redirects/private-network targets, and keep credentials out of browser
   state, recipes, logs and exported packages. Use minimum-scope short-lived tokens.
3. Send only the necessary numeric recipe and approved model inputs. Photographs,
   local paths and unrelated source archives are excluded by default.
4. Set upload/download, array shape, decompression, runtime, concurrency and cost
   limits. Bound retries; cancellation must cancel the remote job as well as local
   waiting. Surface unknown remote state instead of resubmitting paid work blindly.
5. Parse responses as schema-validated inert data with finite values, units,
   provenance and expected coordinates. Reject symlinks, path traversal, object
   arrays/pickle, executable files and unbounded archives. Stage and verify outputs
   before atomic publication; isolate any format conversion from credentials.
6. Record provider/job/model version and input/output digests. Show the user where
   data went, deletion/retention behavior and failures. Keep scientific evidence
   separate from simulated output even when the remote model is physics-based.

This is a readiness checklist, not a claim that a remote backend has been audited
or configured. IFM is a **v0.3** milestone; see [the roadmap](../ROADMAP.md).
