# Reset memo: fixed-variant multimodel value/score campaign reboot

## Date
2026-08-04

## Context
The current task is a cross-model fixed-variant campaign comparing two methods
for **likelihood value** and **score** across five model families:

- LGSSM
- KSC SV
- actual SV
- predator-prey
- Austria SIR

The intended scientific order was clarified during this session and should not be
re-litigated on reboot:

1. **Method A** = original single-fitted frozen version (fit once, freeze, no
   interpolation).
2. **Method B** = tangent/interpolation extension, only after Method A is
   admitted for that model.

The main conceptual resolution of this pass was that the campaign should not
start from a source-native Zhao-Cui reimplementation, nor from a global tangent
extension. The baseline object to test first is the once-fitted frozen parent.

## Decision / policy
Future sessions should assume the following.

Anti-drift authority for the current execution block:
`docs/plans/bayesfilter-zhao-cui-fixed-variant-all-model-campaign-control-master-program-2026-08-06.md`

- The campaign is about **declared fixed-variant scalars and their scores**, not posterior correctness, exact physical likelihood, source faithfulness, HMC readiness, or production readiness.
- The exact mathematical likelihood remains the unique observed-data marginal density implied by the model. The fixed-variant TT / filtering routes in this campaign are finite deterministic approximations to that exact likelihood, and the score backend must differentiate the same finite program the value route computes.
- Method A is the primary cross-model baseline and must be implemented/reported for all five models before Method B is interpreted as anything more than a per-model empirical extension.
- Method A admission now requires a manual or analytical score route for the declared scalar; same-scalar autodiff agreement is diagnostic evidence only.
- Method B must be admitted **model by model**; missing Method B routes are to be reported explicitly as `blocked` or `missing`, never silently omitted.
- Austria SIR Method B may be runtime-analytical for the child scalar while still
  relying on empirical/offline tangent issuance. Do not promote that into an
  end-to-end analytical-derivative claim without new evidence.
- The Austria SIR Method B loader paths are currently the main execution blocker;
  do not let them block Method A completion across the full model set.
- The actual-SV transformed-SV helper route in `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_analytic_tf.py` is not the campaign path and should not be reintroduced as a substitute target.
- The actual-SV Method A route now uses a same-program manual replay backend for the active batch TT finite program; do not revert to the autodiff wrapper unless the same-program manual path regresses.

## What changed
- File: `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-plan-2026-08-04.md`
  - Created the main campaign plan.
  - Froze the method order, scope, verification strategy, and skeptical audit.
- File: `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-target-registry-2026-08-04.json`
  - Created the initial registry of model × method rows.
  - Recorded current Method A / Method B readiness and derivative backend labels.
- File: `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-adapter-schema-2026-08-04.json`
  - Created the uniform result payload schema for the campaign.
- File: `scripts/run_fixed_variant_value_score_multimodel_20260804.py`
  - Created an initial cross-model runner that binds Method A adapters for all
    five models and attempts Austria SIR Method B via persisted artifacts.
  - Fixed two early script bugs:
    - wrong import location for `load_selected_t2_parameter_parent_compat`
    - tensor telemetry casting for int32 status tensors
- File: `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
  - Added the same-program manual score backend for the actual-SV batch TT finite program.
- File: `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
  - Rewired the actual-SV adapter to the same-program manual score backend and updated metadata accordingly.

## Bugs / blockers resolved
- Symptom: the initial runner imported
  `load_selected_t2_parameter_parent_compat` from the wrong module.
- Root cause: the function lives in
  `bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf`, not in
  `...lane_b_t2_tf`.
- Resolution: updated the import in
  `scripts/run_fixed_variant_value_score_multimodel_20260804.py`.

- Symptom: the runner crashed while serializing LGSSM telemetry because int32
  status tensors were forced into float64 conversion.
- Root cause: `_tensor_list()` used `tf.convert_to_tensor(..., dtype=float64)` on
  int tensors.
- Resolution: changed `_tensor_list()` to cast after conversion.

- Symptom: actual SV remained blocked under Method A because the route score backend was autodiff-backed.
- Root cause: the active batch TT finite program had no same-program manual backend; an earlier analytic helper route was a different scalar family and therefore inadmissible.
- Resolution: implemented a same-program manual directional replay in `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py` and rewired the adapter to it without changing the value scalar.

## Verification already run

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py tests/test_zhao_cui_actual_sv_neutra_target.py
CUDA_VISIBLE_DEVICES=-1 python scripts/run_fixed_variant_value_score_multimodel_20260804.py
```

Observed:
- The actual-SV manual-score tests pass on the same active batch TT scalar.
- The CPU-only multimodel campaign rerun reports actual SV as `manual` with a passing same-scalar finite-difference check.
- Austria SIR Method B T1 and T2 still load and execute from the fresh current-source artifact chain.
- LGSSM, KSC SV, predator-prey, and Austria SIR remain admitted Method A rows with manual score provenance.

## Current policy
- Preserve the multimodel plan, registry, adapter schema, and runner as the campaign-control anchor.
- Method A admission requires manual or analytical score provenance for the declared scalar.
- The actual-SV Method A route is now the active same-program manual score backend in `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py` and `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`.
- The actual-SV route remains a deterministic approximation to the exact model likelihood, and the manual score is the derivative of that same finite program.
- Austria Method B work remains historical context only for this reset except where explicitly reported in the current runner summary.

## Known limitations / cautions
- The actual-SV Method A route is still not an exact physical-likelihood route; it is a same-program manual score backend for the current deterministic approximation.
- The machine-readable multimodel result now has live Austria Method B rows, but their `same_scalar_check` fields are still replay-loader statuses (`NOT_RUN` in the campaign summary), not newly rerun held-out local-comparison diagnostics.
- The registry still likely underspecifies current Austria Method B freshness and should be refreshed if it is going to remain the long-lived authority.
- LGSSM Method B remains only a wrapper candidate, not an implemented persisted route.
- KSC SV, actual SV, and predator-prey still have no real Method B path in this campaign.
- The fresh Austria reissue artifacts were created under trusted GPU execution; preserve them and do not silently fall back to the stale 20260731/20260801 paths in future runner edits.

## Suggested next steps
1. Decide whether the campaign should stop with Austria-only live Method B
   evidence and honest blocked rows elsewhere, or whether a follow-on phase
   should build a truly same-scalar persisted LGSSM Method B wrapper.
2. If LGSSM Method B is pursued, do **not** reuse the existing persisted
   frozen-transport / NeuTra artifacts as Method B rows, because they change the
   scalar via transport pullback plus log-Jacobian relative to Method A.
3. If stronger Austria Method B interpretation is needed, add the promised
   held-out local-comparison diagnostics rather than inferring them from issuer
   replay passes alone.
4. If stronger actual-SV Method A evidence is needed, add a slightly wider same-scalar finite-difference ladder around the current campaign probe rather than changing the route family.
