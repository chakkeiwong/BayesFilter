# Fixed-Variant Value/Score Multimodel Campaign Plan

Date: 2026-08-04

Status: `PLANNED_AND_PARTIALLY_EXECUTED`

Anti-drift authority:
`docs/plans/bayesfilter-zhao-cui-fixed-variant-all-model-campaign-control-master-program-2026-08-06.md`


## Purpose

Compare two fixed-variant methods for **likelihood value** and **score** across five model families:

- LGSSM
- KSC SV
- actual SV
- predator-prey
- Austria SIR

The campaign order is frozen:

1. **Method A** — original single-fitted frozen version: fit once, freeze, no interpolation. Method A admission in this campaign requires a manual or analytical score route for the declared scalar; same-scalar autodiff tie-out is supporting diagnostic evidence only.
2. **Method B** — tangent/interpolation extension: only after Method A exists and passes its manual/analytical-score admission gates for that model.

## Scope and nonclaims

This campaign is about **declared fixed-variant scalars and their scores**, not about posterior correctness, HMC convergence, source faithfulness, or production readiness.

Passing this campaign does **not** establish:
- exact physical observed-data likelihood correctness,
- exact posterior correctness,
- HMC readiness,
- production/default readiness,
- superiority of Method B over Method A beyond the declared same-scalar/local-extension evidence.

## Evidence contract

For each model × method row, the campaign records:
- the declared value scalar,
- the declared score meaning,
- the derivative backend,
- replay determinism,
- same-scalar diagnostic status,
- row status (`ready`, `blocked`, `pending`).

Under the active Method A policy, a row is Method-A-admitted only when its score provenance is manual or analytical for the declared scalar. An autodiff same-scalar tie-out may support a row diagnostically, but it does not satisfy the Method A admission rule by itself.

Method B is admitted model-by-model only after Method A for that same model passes its gates.

## Models and intended Method A anchors

- **LGSSM**: `bayesfilter/testing/deterministic_lgssm_exact_target_tf.py`
- **KSC SV**: `bayesfilter/testing/ksc_ukf_neutra_target_tf.py`
- **actual SV**: `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_analytic_tf.py`
- **predator-prey**: `bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py`
- **Austria SIR**: `bayesfilter/testing/sir_filter_neutra_target_design_tf.py` (`make_sir_sgqf_neutra_adapter` for the single fitted frozen baseline)

## Models and intended Method B anchors

- **Austria SIR**: `bayesfilter/highdim/zhao_cui_austria_sir_parameter_child_tf.py`, `...lane_b_t1_score_tf.py`, `...lane_b_t2_score_tf.py`, `...training_jvp_tf.py`
- **LGSSM**: thin persisted wrapper over existing exact/tangent machinery if feasible
- **KSC SV / actual SV / predator-prey**: only if a minimal frozen-parent/tangent route can be created without breaking scalar comparability or governance; otherwise explicitly blocked

## Phases

### Phase 0 — Governed artifacts
Create:
- this plan,
- a frozen target registry,
- a uniform adapter/result schema.

### Phase 1 — Method A baseline
Expose one common adapter contract across all five models for value + score.

### Phase 2 — Method A gates
For each Method A row require:
- finite value,
- finite score,
- deterministic replay,
- same-scalar FD or same-scalar autodiff tie-out,
- frozen branch / fixture identity across perturbations.

### Phase 3 — Method B extension
Admit Method B per model only after Method A is ready.

Austria SIR is the first expected B-ready model.
LGSSM is the second candidate if a thin wrapper is feasible.
All others may remain blocked.

### Phase 4 — Reporting
Emit:
- one full Method A cross-model table,
- one Method B extension table,
- explicit blocked/pending rows (no silent omissions).

## Verification strategy

### Method A
For each model:
- deterministic replay on frozen object,
- value and score finite,
- same-scalar check,
- explicit derivative-backend metadata.

### Method B
For each eligible model:
- parent identity frozen and recorded,
- tangent/interpolation state persisted and hash-recorded,
- child scalar replayable,
- same-scalar child check,
- local comparison to Method A on a predeclared held-out theta design.

## Skeptical audit

The campaign must explicitly check for:
- wrong-scalar comparisons between A and B,
- derivative-label drift (`analytical` vs `manual-runtime-from-offline-tangents`),
- hidden adaptation in Method A,
- false symmetry from silently missing Method B rows,
- accidental Austria-route drift back into the later frozen-importance/APF score-completion route.

## Expected deliverables

- `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-plan-2026-08-04.md`
- `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-target-registry-2026-08-04.json`
- `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-adapter-schema-2026-08-04.json`
- one runner script under `scripts/`
- one result note under `docs/plans/`
