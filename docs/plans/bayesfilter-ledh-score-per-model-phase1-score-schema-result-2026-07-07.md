# Phase 1 Result: Score Artifact Schema And Guards

metadata_date: 2026-07-07
status: `PASSED_LOCAL_PENDING_READONLY_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 1

## Phase Objective

Define and locally test a replayable LEDH score artifact contract before any
model score row can be admitted.

The score target remains the no-tape total derivative of the realized
finite-`N` LEDH estimator:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Phase 1 score artifact schema and guard tests pass locally. |
| Primary criterion status | Passed locally: score artifacts are accepted only when tied to an admitted value artifact with matching row id, target scalar, output field, target observation policy, theta coordinate system, parameter names/order, finite score, same-route status, no-tape provenance, correctness pass, and full-row memory gate when admitted. |
| Veto diagnostic status | Negative tests cover row mismatch, scalar mismatch, output-field mismatch, target-policy mismatch, theta-coordinate mismatch, parameter-order mismatch, tape/forward-accumulator/stopped-partial flags, tiny-as-full admission, missing memory gate, KSC exact-SV overclaim, diagnostic SIR row, and nonfinite score. |
| Main uncertainty | Phase 1 validates score artifacts but does not compute a model score. Each model still needs its own implementation/replay phase. |
| Next justified action | Review this result and the Phase 2 LGSSM subplan; if review agrees, execute the LGSSM score phase. |
| What is not concluded | No model score is admitted; no leaderboard score integration, HMC readiness, posterior correctness, scientific superiority, or runtime ranking is concluded. |

## Implementation Artifacts

- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`

Key exported contract fields:

- `schema_version = bayesfilter.highdim.ledh_score_artifact.v1`
- `score_target_kind = realized_finite_N_ledh_log_likelihood_estimator`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `score_admission_status = n10000_same_target_no_tape_score_admitted`
- `score_admission_status = blocked_score_not_run`
- `value_score_route_status = same_route_value_score`

## Local Checks

CPU-hidden checks intentionally set `CUDA_VISIBLE_DEVICES=-1` because Phase 1
is a schema/contract phase and is not GPU evidence.

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result: passed.

Focused schema tests:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
19 passed, 2 warnings in 2.73s
```

Combined value/schema replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
22 passed, 2 warnings in 2.75s
```

Diff hygiene:

```text
git diff --check -- \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result: passed.

## Repair Record

The first focused schema run failed because the provenance guard banned the
substring `autodiff`, while the approved no-tape route identifiers deliberately
contain `no_autodiff`. This was a contract implementation error, not a model
score failure.

Repair:

- keep explicit route allowlisting for approved no-tape provenance strings;
- keep explicit forbidden flags for `uses_gradient_tape`,
  `uses_forward_accumulator`, and `uses_stopped_partial_derivative`;
- keep explicit forbidden provenance tokens for `GradientTape`,
  `ForwardAccumulator`, `stopped`, `stop_gradient`, and
  `partial_derivative`;
- remove the overbroad substring ban that rejected `no_autodiff`.

Focused checks were rerun after the repair and passed.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the repo reject score artifacts that are not no-tape same-scalar total derivatives of admitted value artifacts? |
| Answer | Locally yes: the validator accepts the positive LGSSM fixtures and rejects the targeted mismatch/provenance/admission/overclaim fixtures. |
| Baseline/comparator | Phase 0 baseline, Phase 8 value artifact schema, admitted LGSSM/KSC value artifacts, and existing LGSSM/fixed-SIR diagnostic fields. |
| Primary criterion | Passed locally through score contract tests and combined value/schema replay. |
| Veto diagnostics | No veto remains open in Phase 1 local checks. |
| Explanatory diagnostics | Warnings are TensorFlow Probability deprecation warnings and do not affect the schema result. |
| Not concluded | No model score admission, no full `N=10000` score evidence, no HMC/posterior/scientific/runtime claim. |

## Phase 2 Handoff

Phase 2 LGSSM may begin only after bounded read-only review agrees with this
Phase 1 result and the Phase 2 LGSSM subplan.

Phase 2 must not admit an LGSSM score unless the produced score artifact passes
`validate_ledh_score_artifact(..., require_admitted=True)` against:

```text
docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json
```

The active LGSSM runner currently has a stale full-row constant:

```text
FULL_ROW_NUM_PARTICLES = 1000
```

while the admitted value artifact is:

```text
num_particles = 10000
```

Phase 2 must repair or override that mismatch before any full-row score
admission can be accepted.

## Nonclaims

- No model score row is admitted by Phase 1.
- Tiny fixtures are not full score admission.
- Schema acceptance is not score correctness.
- CPU-hidden schema tests are not GPU/CUDA/XLA evidence.
- No HMC readiness, posterior correctness, scientific superiority, runtime
  ranking, all-algorithm comparison, or public-release claim is made.
