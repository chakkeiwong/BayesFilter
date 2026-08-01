# LGSSM NeuTra Gap Closure Phase 0 - Compatibility And Policy Repair

Date: 2026-07-15  
Status: `COMPLETE_PASS`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Objective

Create and prove the smallest TensorFlow/TFP-only route that joins current
strict graph-native training artifacts to frozen-transport HMC tuning and
confirmatory validation without importing the legacy NumPy-backed HMC stack.

## Entry Conditions

- The four-arm 500-step screen passed and selected `wide_2x_lr5e3`.
- Selection artifact file SHA-256 is
  `1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97`.
- Strict graph-native training tests and GPU memory-growth probe passed.
- No fresh 5,000-step artifact is assumed to exist.
- Current `AGENTS.md` NumPy diagnostic-only and GPU/XLA policies govern.

## Required Artifacts

- `bayesfilter/testing/lgssm_neutra_gap_closure_tf.py` with no NumPy import and
  no dependency on `bayesfilter.inference.hmc`,
  `fixed_transport_hmc_tuning`, or the old serious-validation orchestrator.
- `docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py`.
- focused tests for strict-result schema consumption, fixed grid selection,
  rank/folded R-hat admission, tensor archive round-trip, comparator summary
  binding, and no-overwrite behavior.
- Phase 0 close record and refreshed Phase 1 subplan.

## Required Checks

- AST/import closure contains no repository NumPy dependency on the new route.
- CPU-hidden TensorFlow/TFP Gaussian four-chain XLA sampler produces finite,
  moving samples, finite log acceptance, zero energy-error divergences, and a
  valid modern diagnostic payload.
- Synthetic pass/fail fixtures prove that acceptance alone cannot admit a
  candidate and that folded R-hat is part of the maximum R-hat gate.
- TensorFlow serialized sample tensor round-trips with exact shape/dtype/value.
- Strict-result adapter rejects wrong schema, failed result, wrong job/steps,
  recipe drift, target drift, missing or hash-mismatched payload, failed parity,
  non-GPU/non-XLA execution, or failed memory-growth evidence.
- Comparator summary reader binds exact file hash, schema, parameter order,
  finite means/SD/MCSE, and the already-passed plain-HMC modern diagnostic.
- compile/import checks and `git diff --check` pass for touched files.

The first fresh candidate's GPU/CPU frozen-objective smoke is a Phase 0/Phase 3
bridge check because no current 5,000-step payload exists. Its absence does not
authorize HMC tuning; the smoke must pass before Phase 3 closes.

## Evidence Contract

Phase 0 establishes engineering validity only: schema compatibility, backend
policy compliance, HMC mechanics on a Gaussian fixture, diagnostic-gate
behavior, and artifact integrity. It cannot establish NeuTra training quality,
LGSSM posterior convergence, agreement, recovery, or superiority.

## Forbidden Claims And Actions

- Do not run a 5,000-step training job until the local Phase 0 gate passes.
- Do not import or wrap the legacy NumPy-backed HMC/tuning modules.
- Do not convert a tensor to a NumPy array for inference, tuning, selection,
  admission, serialization, or posterior computation.
- Do not treat a Gaussian fixture as LGSSM or NeuTra evidence.
- Do not change the selected recipe, target, comparator, thresholds, seeds, or
  campaign budget.
- Do not mutate packages/environments or overwrite existing artifacts.

## Handoff Conditions

Phase 1 may start when all focused local checks pass, the new route can consume
a valid strict-result fixture, and the Phase 0 close record says
`PASS_PHASE0_COMPATIBILITY_GATE`. The Phase 1 subplan must contain the exact
seed1201 command, selected recipe hash, GPU/memory-growth/XLA checks, time
budget, and failure-repair rule.

## Stop Conditions

Stop for an unavoidable NumPy dependency in the required active path, invalid
target/comparator identity, a TensorFlow/TFP HMC mechanics failure that remains
after a focused repair, or a plan change requiring new scientific direction.
Localized test, schema, XLA, or serialization failures trigger repair within
this phase and do not require new authorization.

## Suitability Review

The subplan directly addresses both material audit failures before spending GPU
budget. It is bounded to the NeuTra LGSSM campaign instead of attempting a risky
repository-wide migration of the generic legacy HMC module. The Gaussian gate
tests mechanics but is explicitly prevented from supporting the scientific
claim. Frozen-candidate parity remains mandatory at the earliest point a fresh
payload exists. Verdict: `SUITABLE_TO_EXECUTE`.
