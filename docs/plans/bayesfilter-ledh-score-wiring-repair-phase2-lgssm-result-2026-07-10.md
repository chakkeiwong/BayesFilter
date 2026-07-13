# Phase 2 Result: LGSSM Compact Default Cleanup

Date: 2026-07-10

## Decision Table

| Decision item | Status |
| --- | --- |
| LGSSM default compact score path | Passed |
| Historical full-history reverse route demotion | Passed |
| Stale `memory_style` raw admission label | Demoted to legacy/wrong; no longer admits |
| Nested historical diagnostic relabeling | Blocked after substitute-review repair |
| Full score artifact `score_precision` | Added and tested |
| Seed-sharded score artifact `score_precision` | Added and tested |
| Score timing fields | Added: `score_call_seconds`, `score_materialize_seconds` |
| Local py-compile | Passed |
| Focused LGSSM/shared tests | Passed: `77 passed, 2 warnings` |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for LGSSM wiring: default `compact-sensitivity` uses compact forward-sensitivity and not the full-history reverse helper. |
| Baseline/comparator | Phase 1 repaired shared score contract and prior LGSSM compact score-only memory artifact. |
| Primary criterion | Met: tests prove compact dispatch, historical reverse diagnostic status, score precision in admitted artifacts, and stale `memory_style` demotion. |
| Veto diagnostics | No default path calls `_manual_value_and_score_from_components`; stale raw `admitted_same_target_memory_style_score` no longer full-admits; full artifacts include production precision metadata. |
| Not concluded | No new GPU score run, no exact Kalman score claim, no HMC readiness, no repair for non-LGSSM model score paths. |

## Code Changes

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
  - Added `RAW_COMPACT_ADMITTED_STATUS = "admitted_same_target_compact_score"`.
  - Kept `RAW_MEMORY_STYLE_ADMITTED_STATUS` only as a legacy wrong label that
    is recognized and rejected/demoted.
  - Made compact full-row raw admission depend on `RAW_COMPACT_ADMITTED_STATUS`.
  - Added `_score_precision_metadata` and propagated production precision into
    monolithic and seed-sharded score artifacts.
  - Added score timing fields separate from value timing. `score_call_seconds`
    measures the score diagnostic wall-clock region. `score_materialize_seconds`
    is retained as a compatibility field but is not a clean post-call tensor
    materialization split because the diagnostic helper materializes internally.
- `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
  - Updated raw full-row fixture to use compact admitted status and production
    `float32`/TF32 precision metadata.
  - Added stale memory-style status demotion coverage.
  - Added adversarial nested-diagnostic mismatch tests to prevent historical
    reverse/manual provenance from being relabeled as compact.
  - Added precision assertions on monolithic and sharded score artifacts.
- `tests/test_ledh_lgssm_manual_score_phase4.py`
  - Updated admission decision assertion to the compact raw status constant.

## Local Checks

Py-compile:

```bash
python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py
```

Result: passed.

Focused tests:

```bash
pytest -q tests/test_ledh_lgssm_manual_score_phase4.py tests/highdim/test_ledh_lgssm_score_phase2_contract.py tests/highdim/test_ledh_score_contract_phase1.py
```

Result after substitute-review repairs: `83 passed, 2 warnings`.

Route/precision search:

```bash
rg -n "compact-sensitivity|_compact_value_and_score_from_components|_manual_value_and_score_from_components|RAW_MEMORY_STYLE_ADMITTED_STATUS|RAW_COMPACT_ADMITTED_STATUS|score_precision|score_call_seconds|score_materialize_seconds" docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py tests/highdim/test_ledh_lgssm_score_phase2_contract.py tests/test_ledh_lgssm_manual_score_phase4.py
```

Result before substitute-review repairs: `60` matches. A focused repair search
after substitute-review repairs is recorded at
`docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-review-repair-rg-2026-07-10.log`
with `44` matches. Relevant classification: compact default dispatches to
`_compact_value_and_score_from_components`; manual reverse remains an explicit
historical diagnostic branch; `RAW_MEMORY_STYLE_ADMITTED_STATUS` appears only in
legacy demotion/rejection coverage.

## Review Status

Claude review remains unavailable under the external-data disclosure policy
rejection recorded in Phase 0. Phase 2 result and Phase 3 subplan require a
fresh Codex substitute read-only review before Phase 3 execution.

Initial substitute review returned `VERDICT: REVISE` because the LGSSM artifact
adapter could hardcode compact provenance without checking nested diagnostic
provenance. The adapter and tests were patched, and focused local checks passed
with `83 passed, 2 warnings`. A focused re-review is required before Phase 2 is
closed.

## Plain-Language Gate

- Target: LGSSM same-target LEDH finite-`N` observed-data log likelihood score
  wiring.
- Computed quantity: local import/CPU-hidden tests and metadata/artifact
  validation, not a new GPU score.
- Direct classification: LGSSM compact default wiring is `correct` relative to
  the Phase 2 target.
- Wrong relative to target: old `admitted_same_target_memory_style_score` as a
  full admission trigger is historical/wrong and now demoted.
- Unsupported: all-model score readiness and leaderboard score completion.

## Next Phase Handoff

Phase 3 may start only after review of this result and the Phase 3 subplan.
Phase 3 is fixed-SIR-specific: preserve the compact helper as default, demote
legacy memory-result normalizers to diagnostic import-only status, and add
production precision coverage.
