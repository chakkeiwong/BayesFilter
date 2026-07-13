# Phase 5 Repair Subplan: Actual-SV Full-Row Score/Memory Gate

metadata_date: 2026-07-07
status: `DRAFT_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-full-row

## Phase Objective

Attempt, or explicitly block, full-row actual-SV LEDH score admission after the
tiny no-tape same-target diagnostic passed.

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
- Actual-SV tiny score diagnostic passed all-coordinate same-scalar FD.
- Actual-SV full score is not admitted.
- LGSSM, fixed-SIR, and predator-prey remain blocked/not admitted.

## Required Artifacts

Input artifacts:

- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-2026-07-07.md`

Code/tests:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

Expected artifacts:

- full-row plan/review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-review-bundle-2026-07-07.md`
- full-row score artifact if attempted and passed:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-score-artifact-2026-07-07.json`
- full-row score summary if attempted:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-score-artifact-2026-07-07.md`
- full-row blocker/result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-result-2026-07-07.md`
- Phase 6 generalized-SV subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase6-generalized-sv-subplan-2026-07-07.md`

## Required Checks/Tests/Reviews

Before any full run:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Review:

- bounded read-only review of this full-row score/memory subplan before any
  trusted GPU full-row command;
- bounded read-only review of the full-row result/blocker and Phase 6 subplan.

If reviewed execution proceeds, the full-row command must be trusted GPU and
must write JSON/Markdown artifacts. The first admissible command must preserve:

- `batch_seeds = 81120,81121,81122,81123,81124`;
- `time_steps = 1000`;
- `num_particles = 10000`;
- `transport_policy = active-all`;
- `transport_plan_mode = streaming`;
- manual streaming finite transport VJP with `transport_ad_mode=full`;
- no `GradientTape`, `ForwardAccumulator`, hidden autodiff, or stopped partials;
- same target scalar and parameter order.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the tiny-passing actual-SV no-tape total score route scale to full `N=10000,T=1000` with replayable correctness and memory evidence? |
| Baseline/comparator | Tiny actual-SV score diagnostic, admitted actual-SV value artifact, Phase 1 score artifact validator, and same-scalar correctness checks. |
| Primary criterion | Full score artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)`, all two parameters have approved correctness evidence, memory passes under the stated budget, and no-tape provenance remains clear. |
| Veto diagnostics | Unsafe full-history storage, OOM, tape/autodiff, stopped partial, target substitution, missing all-parameter correctness, missing memory pass, nonfinite score/value, or artifact validation failure. |
| Explanatory diagnostics | Runtime, memory peak, per-seed score dispersion, FD step sensitivity, dtype/TF32 mode, device placement, and chunk sizes. |
| Not concluded | HMC readiness, posterior correctness, exact raw-observation likelihood, KSC/generalized-SV score admission, runtime ranking, scientific superiority, or all-algorithm comparison. |
| Artifact | Full-row score artifact or blocker result, review bundle, Phase 6 subplan. |

## Step-By-Step Plan

1. Re-run focused Phase 5 local checks.
2. Audit whether the current score implementation stores all time-step records
   and estimate memory risk for `T=1000,N=10000`.
3. If memory risk is unsafe, write a blocker result requiring a checkpointed or
   online reverse/recompute design before full admission.
4. If memory risk is bounded by review, run a smaller trusted GPU ladder first:
   - `T=5,N=256`;
   - `T=20,N=1024`;
   - only then full `T=1000,N=10000`.
5. Do not use ladder runtime/memory as correctness admission by itself.
6. For any full score artifact, validate with
   `validate_ledh_score_artifact(..., require_admitted=True)`.
7. Write full-row result or blocker result.
8. Draft Phase 6 generalized-SV subplan.
9. Review result and Phase 6 subplan.

## Forbidden Claims/Actions

- Do not run full `N=10000,T=1000` if the route still stores an unsafe full
  per-time history.
- Do not claim full admission from the tiny diagnostic.
- Do not use tape, `ForwardAccumulator`, hidden autodiff, or stopped partials.
- Do not substitute KSC, raw Gaussian, or augmented-noise target evidence.
- Do not change score pass/fail criteria after seeing results.
- Do not claim HMC readiness, posterior correctness, runtime ranking,
  scientific superiority, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

Phase 6 generalized-SV may start only if:

- full-row actual-SV score is admitted or explicitly blocked;
- Phase 6 subplan exists;
- local checks pass;
- read-only review agrees the Phase 5 decision and Phase 6 handoff are
  boundary-safe.

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
| Wrong baseline | Only the admitted actual-SV value artifact anchors the full score. |
| Proxy promotion | Tiny correctness and runtime/memory cannot admit full score alone. |
| Missing stop condition | Stop on unsafe memory, OOM, validation failure, or no-tape ambiguity. |
| Hidden assumption | Explicitly audit stored records before full run. |
| Stale context | Start from Phase 5 tiny diagnostic and current implementation. |
| Environment mismatch | Full-row and ladder commands must use trusted GPU execution. |
| Useless artifact | Admission requires validator replay with `require_admitted=True`. |
