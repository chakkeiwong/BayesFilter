# Phase 1 Subplan: Score Artifact Schema And Guards

metadata_date: 2026-07-07
status: `DRAFT_AFTER_PHASE0_LOCAL_PASS_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 1

## Phase Objective

Define the replayable LEDH score artifact schema and guard tests required
before any model score can be admitted.

The score target is the no-tape total derivative of the realized finite-`N`
LEDH estimator:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

## Entry Conditions Inherited From Previous Phase

- Phase 0 freezes exactly six eligible value rows.
- Phase 0 admits no scores.
- Phase 0 excludes the parameterized SIR diagnostic row.
- Phase 0 records that score means realized finite-`N` estimator derivative,
  not unproven true-likelihood derivative.
- Phase 0 result and this Phase 1 subplan must pass read-only review before
  execution.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-subplan-2026-07-07.md`
- Score contract/schema implementation, expected path:
  `bayesfilter/highdim/ledh_score_contract.py`
- Score schema guard tests, expected path:
  `tests/highdim/test_ledh_score_contract_phase1.py`
- Phase 1 result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-result-2026-07-07.md`
- Phase 2 LGSSM subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-subplan-2026-07-07.md`
- Phase 1 review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase1-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Implementation checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Focused schema tests:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Combined baseline/schema checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Diff hygiene:

```text
git diff --check -- \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-result-2026-07-07.md \
  docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-subplan-2026-07-07.md
```

Review:

- bounded read-only review of Phase 1 result and Phase 2 subplan before Phase
  2 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the repo reject score artifacts that are not no-tape same-scalar total derivatives of admitted value artifacts? |
| Baseline/comparator | Phase 0 baseline, Phase 8 value artifact schema, existing LGSSM/fixed-SIR diagnostic fields. |
| Primary criterion | A validator accepts only artifacts with an admitted source value artifact, same row id, same target scalar, same output field, same target observation policy, same theta coordinate system, same parameter names/order, finite score vector, no-tape derivative provenance, same-route value/score identity, tiny/full admission distinction, memory fields for `N=10000`, and no forbidden autodiff flags. |
| Veto diagnostics | Missing source value artifact; row-set mismatch; diagnostic SIR row; score for a different scalar; target observation policy mismatch; theta coordinate mismatch; parameter-order mismatch; tape/autodiff provenance; stopped partial derivative; missing parameter names; nonfinite score; tiny artifact admitted as full; missing memory gate for full row; KSC exact-SV overclaim. |
| Explanatory diagnostics | FD/exact errors, runtime, memory, per-seed score, component decomposition, and output devices. |
| Not concluded | No model score admission, no HMC readiness, posterior correctness, scientific superiority, runtime ranking, or all-algorithm comparison. |
| Artifact | Score contract module, tests, Phase 1 result, Phase 2 subplan. |

## Step-By-Step Plan

1. Add score schema constants:
   - `bayesfilter.highdim.ledh_score_artifact.v1`;
   - `tiny_score_diagnostic_not_admitted`;
   - `n10000_same_target_no_tape_score_admitted`;
   - `blocked_score_not_run`;
   - no-tape derivative provenance allowlist.
2. Add `validate_ledh_score_artifact(...)`:
   - requires a normalized admitted value artifact or source value artifact
     metadata;
   - requires row id, target scalar, output field, target observation policy,
     theta coordinate system, and parameter names/order to match the admitted
     value artifact;
   - requires `score_target_kind =
     realized_finite_N_ledh_log_likelihood_estimator`;
   - requires `value_score_route_status = same_route_value_score`;
   - rejects `GradientTape`, `ForwardAccumulator`, hidden autodiff, stopped
     partials, and diagnostic SIR row ids;
   - rejects full admission without `N=10000`, finite score, memory fields, and
     replayed correctness gate.
3. Add negative tests for:
   - score before value;
   - wrong row id;
   - wrong target scalar;
   - wrong target output field;
   - target observation policy mismatch;
   - theta coordinate system mismatch;
   - parameter names/order mismatch;
   - tape provenance;
   - stopped partial derivative;
   - parameterized SIR diagnostic row;
   - KSC exact-SV target overclaim;
   - tiny artifact admitted as full.
4. Add positive minimal fixture tests for:
   - tiny LGSSM diagnostic accepted as not admitted;
   - full-row fixture accepted only when memory/correctness fields are present.
5. Run required checks.
6. Write Phase 1 result.
7. Draft Phase 2 LGSSM subplan.
8. Send Phase 1 result and Phase 2 subplan for bounded read-only review.

## Forbidden Claims/Actions

- Do not admit any model score in Phase 1.
- Do not run `N=10000` score benchmarks in Phase 1.
- Do not use or authorize tape/autodiff score routes.
- Do not promote diagnostic SIR row evidence.
- Do not claim exact true-likelihood derivative unless separately proved for a
  row.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  runtime ranking, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

Phase 2 LGSSM may start only if:

- score schema implementation and tests pass;
- Phase 1 result exists;
- Phase 2 LGSSM subplan exists;
- read-only review agrees that the schema prevents score/value target mismatch,
  target-policy mismatch, theta-coordinate mismatch, parameter-order mismatch,
  tape/autodiff admission, diagnostic-row promotion, and tiny-as-full
  admission.

## Stop Conditions

Stop and write a blocker result if:

- the schema cannot tie a score artifact to a specific admitted value artifact;
- the schema cannot enforce target observation policy, theta coordinate system,
  and parameter names/order against the admitted value artifact;
- the schema cannot distinguish tiny diagnostic from full `N=10000` score
  admission;
- the no-tape/tape provenance cannot be made explicit;
- a required negative test cannot be written;
- review finds a material issue that does not converge after five rounds;
- a human approval boundary is reached.
