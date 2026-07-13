# Phase 5 Refreshed Subplan: Actual-SV Full-Row Score/Memory Gate

metadata_date: 2026-07-07
status: `DRAFT_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-full-row-after-streaming-parity-repair

## Phase Objective

Attempt, or explicitly block, full-row actual-SV LEDH score admission after the
streaming-flow parity repair passed tiny same-target diagnostics.

The full-row target remains:

```text
row_id = zhao_cui_sv_actual_nongaussian_T1000
target_scalar = observed_data_log_likelihood_estimator
target_output_tensor_field = log_likelihood
target_observation_policy = transformed_actual_sv_log_y_square
theta_coordinate_system = synthetic_unconstrained
score_parameter_names = [gamma_unconstrained, log_beta]
```

## Entry Conditions Inherited From Previous Phase

- Actual-SV value artifact is admitted at `N=10000,T=1000`.
- Phase 5 streaming-flow parity repair result exists:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-result-2026-07-07.md`.
- Repaired tiny diagnostic exists and is not full admission:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.json`.
- Required local checks passed with `30 passed, 2 warnings`.
- LGSSM, fixed-SIR, and predator-prey remain blocked/not admitted.

## Required Artifacts

Input artifacts:

- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-result-2026-07-07.md`

Code/tests:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`

Expected artifacts:

- full-row plan/review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-refresh-review-bundle-2026-07-07.md`;
- ladder/full score artifacts if attempted;
- full-row score result or blocker:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-refresh-result-2026-07-07.md`;
- refreshed Phase 6 generalized-SV subplan or explicit stop handoff.

## Required Checks/Tests/Reviews

Before any full run:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Review:

- bounded read-only review of this refreshed full-row score/memory subplan before any trusted GPU full-row command;
- bounded read-only review of the full-row result/blocker and next subplan.

If reviewed execution proceeds, commands that initialize or use GPU/CUDA/TensorFlow GPU must run in trusted/escalated context.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the repaired actual-SV no-tape total score route scale to full `N=10000,T=1000` with replayable correctness and memory evidence? |
| Baseline/comparator | Admitted actual-SV value artifact, repaired tiny score diagnostic, Phase 1 score validator, same-forward-scalar parity tests. |
| Primary criterion | Full score artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)`, both parameters have approved correctness evidence, memory passes under the stated budget, and no-tape provenance remains clear. |
| Veto diagnostics | Unsafe full-history storage, OOM, tape/autodiff, stopped partial, target substitution, missing all-parameter correctness, missing memory pass, nonfinite score/value, or artifact validation failure. |
| Explanatory diagnostics | Runtime, memory peak, per-seed score dispersion, FD step sensitivity, dtype/TF32 mode, device placement, chunk sizes, per-ladder memory growth. |
| Not concluded | HMC readiness, posterior correctness, exact raw-observation likelihood, KSC/generalized-SV score admission, runtime ranking, scientific superiority, or all-algorithm comparison. |
| Artifact | Full-row score artifact or blocker result, review bundle, next subplan. |

## Step-By-Step Plan

1. Re-run the focused Phase 5 local checks.
2. Audit the repaired score implementation's retained records and estimate memory for `T=1000,N=10000`.
3. If memory risk is unsafe, write a blocker result requiring checkpointed reverse/recompute or online adjoint before full admission.
4. If memory risk is bounded by review, run a trusted GPU ladder first:
   - `T=5,N=256`;
   - `T=20,N=1024`;
   - only then consider full `T=1000,N=10000`.
5. For each ladder, record command, dtype, TF32 mode, device placement, peak memory, score correctness status, and nonclaims.
6. Do not use ladder runtime/memory as correctness admission by itself.
7. For any full score artifact, validate with `validate_ledh_score_artifact(..., require_admitted=True)`.
8. Write full-row result or blocker result.
9. Draft or refresh the Phase 6 generalized-SV subplan.
10. Review result and next subplan.

## Forbidden Claims/Actions

- Do not run full `N=10000,T=1000` before this refreshed subplan is reviewed.
- Do not run full `N=10000,T=1000` if retained records imply unsafe memory.
- Do not claim full admission from tiny diagnostics or ladder runs.
- Do not use tape, `ForwardAccumulator`, hidden autodiff, or stopped partials.
- Do not substitute KSC, raw Gaussian, augmented-noise, or matrix-flow evidence.
- Do not change score pass/fail criteria after seeing results.
- Do not claim HMC readiness, posterior correctness, runtime ranking, scientific superiority, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

Phase 6 generalized-SV may start only if:

- full-row actual-SV score is admitted or explicitly blocked;
- Phase 6 subplan exists;
- local checks pass;
- read-only review agrees the Phase 5 decision and Phase 6 handoff are boundary-safe.

## Stop Conditions

Stop and write a blocker result if:

- full-row memory risk is unsafe under the current stored-record reverse scan;
- trusted GPU ladder OOMs or produces nonfinite values;
- score correctness cannot be checked for both parameters;
- score artifact cannot validate against the admitted value artifact;
- no-tape provenance becomes ambiguous;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | The admitted actual-SV value artifact and repaired parity tests anchor the scalar. |
| Proxy promotion | Tiny correctness and ladder memory cannot admit full score alone. |
| Missing stop condition | Stop on unsafe memory, OOM, validation failure, no-tape ambiguity, or target substitution. |
| Hidden assumption | Explicitly audit stored records before full run. |
| Stale context | Starts from the streaming-flow parity repair result, not the older matrix-flow tiny result. |
| Environment mismatch | Full-row and ladder commands must use trusted GPU execution. |
| Useless artifact | Admission requires validator replay with `require_admitted=True`. |
