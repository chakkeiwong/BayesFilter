# Phase 4 Repair Subplan: Predator-Prey Full-Row Correctness Calibration

metadata_date: 2026-07-07
status: `DRAFT_AFTER_RUNG3_GPU_SMOKE_MIXED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4-repair-calibration

## Phase Objective

Define and execute a correctness policy for predator-prey full-row score
admission after the no-tape tiny route passed but float32/TF32 coordinate FD
smoke failed.

The score target remains the no-tape total derivative of:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

for:

```text
zhao_cui_predator_prey_T20
```

with physical parameter order:

```text
r, K, a, s, u, v
```

## Entry Conditions Inherited From Previous Phase

- Rung 1 dynamics VJP passed all-parameter/state finite-difference checks.
- Rung 2 tiny total-score route passed no-autodiff sentinel and
  all-coordinate same-scalar FD checks.
- Rung 3 GPU smoke showed:
  - FP64 GPU smoke passed tightly;
  - float32/TF32 GPU smoke failed strict FD correctness.
- No predator-prey score is admitted yet.

## Required Artifacts

Source artifacts:

- `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-rung3-gpu-smoke-result-2026-07-07.md`

Code artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Expected artifacts:

- calibration result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-full-row-correctness-calibration-result-2026-07-07.md`
- if calibrated policy passes, full score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.json`
- if calibrated policy passes, full score summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.md`
- Phase 5 actual-SV subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md`
- review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase4-predator-prey-calibration-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Pre-calibration checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Required calibration ladder:

1. FP64 bounded GPU correctness ladder:
   - `T=2,N=64`;
   - `T=5,N=256`;
   - all six coordinate finite differences;
   - strict tolerance default `atol=5e-3, rtol=5e-3`.
2. FP32/TF32 production-memory smoke:
   - same route and chunking;
   - correctness diagnostic is explanatory unless a reviewed tolerance ladder
     passes;
   - memory and runtime only cannot admit correctness.
3. If FP64 bounded correctness passes but FP32/TF32 FD remains noisy:
   - write a policy result that separates:
     - FP64 correctness evidence;
     - FP32/TF32 production runtime/memory evidence;
     - full-row admission status.
   - do not admit the full score unless the policy explicitly requires and
     obtains a validating score artifact.

Review:

- bounded read-only review of this calibration subplan before execution;
- bounded read-only review of the calibration result and Phase 5 subplan before
  Phase 5 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What correctness evidence is required before predator-prey full-row score admission after FP64 passes but FP32/TF32 FD is noisy? |
| Baseline/comparator | Same-scalar coordinate FD in FP64 bounded GPU mode; FP32/TF32 production-memory smoke; admitted value artifact. |
| Primary criterion | A score is admitted only if a full artifact validates with `require_admitted=True`, all six parameters have approved same-scalar correctness evidence, and memory passes. |
| Veto diagnostics | Treating FP32/TF32 noisy FD as pass without reviewed tolerance; treating FP64 tiny correctness as full-row admission; memory/runtime-only promotion; wrong target; wrong parameter order; tape/autodiff; stopped partial. |
| Explanatory diagnostics | FD step ladder, dtype comparison, runtime, memory, per-coordinate errors, device placement. |
| Not concluded | HMC readiness, posterior correctness, exact nonlinear likelihood correctness, source-faithfulness, runtime ranking, or scientific superiority. |
| Artifact | Calibration result, optional full score artifact, review bundle, Phase 5 subplan. |

## Step-By-Step Plan

1. Run pre-calibration local checks.
2. Run FP64 bounded GPU `T=2,N=64` correctness ladder and compare with existing
   Rung 3 FP64 result.
3. Run FP64 bounded GPU `T=5,N=256` correctness ladder if the smaller rung
   passes.
4. Run FP32/TF32 `T=5,N=256` production-memory smoke with recorded FD errors,
   but do not promote it unless the tolerance policy is reviewed.
5. Decide whether full-row admission can be supported:
   - if full-row all-coordinate correctness and memory are both available,
     write a validating score artifact;
   - otherwise write a blocker/calibration result and continue to Phase 5 with
     predator-prey not admitted.
6. Draft Phase 5 actual-SV subplan.
7. Review the calibration result and Phase 5 subplan.

## Forbidden Claims/Actions

- Do not claim predator-prey score admission from tiny FP64 checks alone.
- Do not treat FP32/TF32 noisy FD as pass by loosening tolerances after seeing
  failures without reviewed justification.
- Do not use memory/runtime as correctness evidence.
- Do not run full `N=10000,T=20` score admission before this calibration
  subplan is reviewed.
- Do not use tape, ForwardAccumulator, hidden autodiff, or stopped partials.
- Do not change row target, theta coordinate/order, horizon, seed list, or
  observation policy.

## Exact Next-Phase Handoff Conditions

Phase 5 actual-SV may start only if:

- calibration writes an admitted predator-prey score result or explicit blocker
  result;
- Phase 5 actual-SV subplan exists;
- review agrees the predator-prey decision is boundary-safe.

## Stop Conditions

Stop and write a blocker result if:

- FP64 bounded correctness fails;
- FP32/TF32 tolerance policy cannot be justified;
- full-row memory/correctness evidence cannot be produced within bounded
  execution;
- score artifact cannot validate against the admitted value artifact;
- no-tape provenance becomes ambiguous;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | All checks target the admitted predator-prey value artifact and same scalar. |
| Proxy promotion | Runtime/memory and noisy FD cannot admit correctness. |
| Missing stop condition | Stop on FP64 correctness failure, unjustified tolerance, or missing full artifact. |
| Hidden assumption | FP64 correctness and FP32/TF32 memory are separate ledgers unless reviewed otherwise. |
| Stale context | Start from Rung 3 smoke results and current score module/tests. |
| Environment mismatch | GPU runs require trusted execution; CPU-only tests hide GPU. |
| Useless artifact | Final admission requires `validate_ledh_score_artifact(..., require_admitted=True)`. |
