# Phase 0 Result: Contract And Route Taxonomy Repair

Date: 2026-07-09

Status: `PASSED_CONTRACT_GATE_WITH_CODEX_FALLBACK_REVIEW`

## Objective

Make the LEDH score contract operationally distinguish memory-style admissible
routes from compact/historical diagnostic routes, and prevent bare memory flags
or wrong exact-reference baselines from full score admission.

## Review

Claude review gate was attempted with the approved bounded review bundle:

`docs/reviews/bayesfilter-ledh-score-tangent-materialization-plan-review-bundle-2026-07-09.md`

The approval reviewer rejected the Claude command as an external data-transfer
risk. Per runbook fallback, a fresh Codex read-only review was used. The review
returned `VERDICT: REVISE` with two material blockers:

- memory gate lacked numeric source/peak/budget derivation;
- full admission could still accept generic `exact_reference`, which could
  accidentally admit exact Kalman or other non-LEDH baselines.

Both blockers were patched visibly before execution continued.

## Implementation

Updated:

- `docs/plans/bayesfilter-ledh-score-tangent-materialization-root-cause-repair-plan-2026-07-09.md`
- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`
- `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`

The contract now requires full score admission memory diagnostics to include:

- `n10000_memory_pass == true`;
- `source` in:
  - `score_gpu_memory_info_after`;
  - `max_per_seed_score_gpu_memory_info_after`;
  - `trusted_gpu_score_memory_artifact`;
- finite `peak_mib`;
- finite `budget_mib`;
- `peak_mib <= budget_mib`.

The contract also rejects `score_correctness.kind == exact_reference` for full
admission. Full admission requires `same_scalar_finite_difference`.

## Checks

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py -q
```

Result:

```text
37 passed, 2 warnings
```

## Evidence Contract Status

| Requirement | Status |
| --- | --- |
| Compact/historical routes rejected for full admission | Passed |
| Bare `n10000_memory_pass` rejected | Passed |
| Numeric memory source/peak/budget required | Passed |
| Peak above budget rejected | Passed |
| Generic `exact_reference` rejected for full admission | Passed |
| Tiny diagnostics may remain non-admitted | Passed |

## Nonclaims

This phase does not admit any model score, does not prove memory pass for
`N=10000`, does not establish HMC readiness, and does not complete leaderboard
score computation.

## Handoff

Phase 1 may begin: wire predator-prey and actual-SV default score wrappers to
existing memory-style reverse/VJP routes and keep compact forward sensitivity
diagnostic only.
