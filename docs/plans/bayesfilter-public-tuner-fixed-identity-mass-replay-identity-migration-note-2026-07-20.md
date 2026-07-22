# Public-Tuner Replay Identity Migration Note

Date: `2026-07-20`

## Decision

The active NeuTra frozen-transport validation route no longer uses the broad
public `final_kernel_hash` as an executable-kernel admission gate. That hash
contains run-instance lineage (bootstrap, windowed-stage, fixed-mass-stage,
and Phase 7 artifact hashes) and therefore cannot establish mechanics identity
across otherwise equivalent tuning runs.

The route now persists and replays
`bayesfilter.admitted_hmc_kernel_replay_artifact.v1`. Its fingerprint covers
the transition-affecting mechanics: target and scope identity, base/Phase 4/
final adapter signatures, both fixed-mass payloads, step size, leapfrog count,
acceptance policy, and TensorFlow/TFP/XLA/TF32 execution settings. Tuning
lineage remains in `tuning_provenance` for audit only and is not consulted for
replay admission.

## Governance Classification

This is a proportionality repair under the BayesFilter trusted-local academic
governance profile. It removes a procedural proxy that blocked a mechanics-
equivalent retry; it does not remove the scientific gates. Replay still fails
closed on target, adapter, mass, initial position, step, leapfrog, acceptance,
schema, fingerprint, or execution-setting mismatch. Sequential HMC still uses
the existing warm-up, convergence, target-health, truth-tail, and nonclaim
requirements.

No new compute authorization is implied. The Phase 4 tuning budget remains
closed, and preserved Attempt 01/02/03 artifacts are historical evidence.
The next sampling attempt, if desired, requires a fresh versioned output root
and explicit user authorization under the unchanged scientific contract.

## Skeptical Audit

- Baseline: the admitted executable kernel from the passed fixed-identity
  public tuner, not its lineage-heavy public summary hash.
- Promotion proxy risk: resolved by excluding intermediate artifact hashes
  from the mechanics fingerprint; acceptance and short tuning diagnostics
  remain tuning evidence, not posterior promotion criteria.
- Unfair comparison risk: the replay consumer requires the same target,
  adapter stack, mass payloads, step size, leapfrog count, acceptance policy,
  dtype/backend/XLA/TF32 settings, and frozen-transport identity.
- Environment mismatch risk: replay fails on the declared execution settings;
  serious sampling remains subject to the repository GPU-memory and XLA gates.
- Stop conditions: schema, fingerprint, target, adapter, mass, initialization,
  acceptance, or execution mismatch stops before sequential sampling.
- Artifact adequacy: the replay JSON contains the mechanics required to rebuild
  the retained adapter; provenance is retained separately and cannot decide
  admission.

Audit verdict: passed for implementation and CPU-safe regression only. It does
not authorize or answer the second-seed scientific question.

## Verification

- `python -m py_compile` passed for the modified Python modules.
- `git diff --check` passed.
- Focused CPU-safe regression before executable fixture expansion: `125
  passed, 2 warnings` across the NeuTra contract, fixed-transport tuning, HMC
  artifact, and HMC identity suites.
- Executable admitted-replay contract suite: `27 passed, 2 warnings`, including
  adapter reconstruction, lineage-only changes, and fail-closed corruption and
  context mismatch checks.
- No GPU, tuner, or sampling campaign was launched during this repair.

## Migrated Artifact

The preserved Attempt 03 cell result was migrated successfully without tuning
or sampling:

- artifact: `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260720-replay-migration/admitted_kernel_mechanics.json`;
- mechanics fingerprint:
  `477f9b321817fd292f569913ac6d5233fedf0133df85a8fb0c7bd8c570be28cb`;
- artifact SHA-256:
  `fddfe81e12c7a542b2491db573de645672a5304e1e0bba5fcfc37e4b3ded4e20`;
- selected step: `0.7779889586003162`;
- selected leapfrog count: `6`;
- fixed-identity mass signature:
  `25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe`;
- replay reconstruction: passed with GPU hidden;
- migration manifest: `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260720-replay-migration/migration_manifest.json`.

The next run, if authorized, should consume this artifact directly. It should
not invoke the tuner or retrain the transport; only the missing sequential
warm-up and retained sampling stages remain.
