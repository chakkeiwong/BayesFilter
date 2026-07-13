# Phase 3 Result: Fixed-SIR Compact Default And Legacy Normalizer Demotion

Date: 2026-07-10

## Decision Table

| Decision item | Status |
| --- | --- |
| Fixed-SIR compact score helper default/admissible path | Passed |
| Full admission production `score_precision` | Added and tested |
| Historical memory/manual result full admission | Blocked |
| Nested historical/manual relabeling fixture | Added and tested |
| Tiny diagnostic non-admission | Preserved |
| Local py-compile | Passed |
| Focused fixed-SIR/shared tests | Passed: `67 passed, 2 warnings` |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for fixed-SIR wiring: compact forward-sensitivity is the only full-admissible score path. |
| Baseline/comparator | Existing fixed-SIR compact helper and historical fixed-SIR score-memory artifact. |
| Primary criterion | Met: tests cover compact no-autodiff execution, tiny same-scalar FD, full-admission precision/shape/memory gates, historical memory/manual demotion, and adversarial nested relabeling rejection. |
| Veto diagnostics | No historical memory/manual result can full-admit; no compact full artifact can omit production precision; compact artifact rejects nested manual/memory-style route metadata. |
| Not concluded | No new fixed-SIR N=10000 GPU score run, no Zhao-Cui source-faithfulness claim, no nonlinear exact likelihood claim, no leaderboard rebuild. |

## Code Changes

- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
  - Added compact score precision propagation from diagnostic precision metadata.
  - Required compact diagnostic base to declare compact score route,
    `no_autodiff_score_route`, and same-route value/score status before artifact
    construction.
  - Changed `_fixed_sir_score_artifact_from_memory_result` so historical
    memory/manual score artifacts are diagnostic-only and cannot full-admit.
  - Preserved historical memory/manual provenance on legacy normalized artifacts
    instead of relabeling them as compact.
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
  - Added production precision full-admission fixture.
  - Added full-admission rejection for TF32-disabled precision.
  - Added nested historical/manual relabeling rejection tests.
  - Updated historical memory-result tests to expect diagnostic-only behavior.

## Local Checks

Py-compile:

```bash
python -m py_compile docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py
```

Result: passed.

Focused tests:

```bash
pytest -q tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/highdim/test_ledh_score_contract_phase1.py
```

Result: `67 passed, 2 warnings`.

Route/precision search:

```bash
rg -n "_fixed_sir_compact_score_artifact_from_diagnostic|_fixed_sir_score_artifact_from_memory_result|score_precision|FIXED_SIR_COMPACT_SCORE_ROUTE_ID|FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID|FIXED_SIR_MANUAL_SCORE_ROUTE_ID|historical memory/manual|diagnostic only|compact score route" docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py
```

Result: `51` matches. Classification: compact diagnostic path is the only
full-admission path; historical memory/manual result normalization is
diagnostic-only.

## Review Status

Claude review remains unavailable under the Phase 0 policy rejection. Phase 3
result and Phase 4 predator-prey subplan require substitute read-only review
before Phase 4 execution.

Review status at result write: pending.

## Plain-Language Gate

- Target: fixed-SIR same-target LEDH finite-`N` observed-data log likelihood
  score wiring.
- Computed quantity: local CPU-hidden tiny diagnostics and metadata/artifact
  validation, not a new N=10000 GPU score run.
- Direct classification: fixed-SIR compact full-admission wiring is `correct`
  relative to the Phase 3 target.
- Wrong relative to target: full admission from historical memory/manual score
  artifacts is historical/wrong and now blocked.
- Unsupported: predator-prey/actual-SV repair, all-model readiness, leaderboard
  completion.

## Next Phase Handoff

Phase 4 may start only after review of this result and the Phase 4 subplan.
Phase 4 is predator-prey-specific: switch default score/admission to compact
forward-sensitivity, demote reverse/manual memory-style route to diagnostic,
and enforce production precision metadata for any full admission.
