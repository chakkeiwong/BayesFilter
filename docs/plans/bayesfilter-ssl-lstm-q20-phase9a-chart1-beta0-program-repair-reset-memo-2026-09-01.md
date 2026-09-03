# Phase 9A chart-1/beta-0 program-repair reset memo

Date: 2026-09-01  
State: `PHASE9A_LOCALIZED_MECHANICS_REPAIR_COMPLETE_FULL_REPLAY_PENDING`

## What changed

The chart-1/beta-0 continuation veto from the 2026-08-31 Phase 9A run was
treated as a runner/tuning-boundary problem. The executable runner now supports
source-owned profiles and scope selection, fresh seed namespaces, explicit
measured joint `(epsilon,L)` grids, durable pre-import and signal failure
manifests, per-call progress records, and a GPU0 memory-growth wrapper. The
repair keeps the q=20 bridge, strict backend, C5 compact-high architecture,
acceptance semantics, and target signature unchanged.

## Current evidence

The final fresh run is:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-01/phase9a-chart1-beta0-repair/attempt-05/`

It used profile `chart1_beta0_repair_v4_fresh`, scope index `3` (chart-1,
beta `0.0`), and measured all four declared pairs with a seed namespace
distinct from attempts 01--04. All four pairs were finite and mobile in this
realization. The provisional mechanics handoff selected `epsilon=0.55`,
`L=3`. Held-out values were finite and mobile, but acceptance was `0.941146`,
which fired the acceptance repair trigger. The run status is
`PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL`.

Attempt-01 and attempt-02 are preserved resource timeouts. Attempt-03 is
preserved as a successful mechanics run with a repaired provenance defect;
attempt-04 is a deterministic provenance replay that reused attempt-03 seeds;
neither is the final fresh record. Attempt-05 is the corrected fresh-seed
record. The difference in movement outcomes between attempts 04 and 05 shows
that the short schedule is seed-sensitive. No attempt is a posterior,
whitening, convergence, mode-discovery, or HMC-readiness result.

## Guardrails for the next agent

- Do not open Phase 9B or run the shared transition controller from this state.
- Do not use the selected handoff as a claim-run tuning artifact.
- Do not infer a tuning ranking from the four-draw-per-chain selection runs.
- Preserve the target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` and backend `tensorflow_eigh_strict`.
- Keep `measured_joint_grid_v1`; never restore the old directional acceptance
  ladder or infer unmeasured neighboring step sizes.
- Keep fresh output directories and fresh seed namespaces for every replay.
- Treat TensorFlow trainer-retracing warnings as a performance repair item.

## Next justified work

Draft and audit a new full Phase 9A replay or performance subplan. It should
measure all six `(chart,beta)` scopes, retain disjoint selection and held-out
data, address repeated trainer construction, and require six scope-specific
handoffs before any transition or Phase 9B consideration. Increase draw counts
only with an explicit target-specific budget and evidence contract.

The detailed result, decision table, and exact run manifest are in:

- `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md`
- `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-01/phase9a-chart1-beta0-repair/attempt-05/run_manifest.json`
