# Phase 1 Result: Shared Score Contract And Precision Gate

Date: 2026-07-10

## Decision Table

| Decision item | Status |
| --- | --- |
| Historical route full-admission guard | Passed |
| Compact route full-admissible provenance guard | Passed |
| Production score precision validator | Added and tested |
| Full admission requires score precision metadata | Passed |
| Explicit precision fields required | Passed after substitute-review repair |
| Row/model compact provenance consistency | Passed after substitute-review repair |
| Shared artifact builder propagates precision metadata | Passed |
| Local py-compile | Passed |
| Shared tests | Passed: `55 passed, 2 warnings` |
| Per-model score wiring repair | Not claimed |
| GPU score memory | Not run in Phase 1 |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for shared contract: full admission now requires compact no-tape provenance and production precision metadata. |
| Baseline/comparator | Phase 0 inventory and existing `ledh_score_contract.py` behavior. |
| Primary criterion | Met: tests cover historical-route rejection, compact-route admission eligibility, and production `float32` plus TF32 metadata. |
| Veto diagnostics | No historical full-admission pass found; no float64/TF32-disabled full admission accepted. |
| Not concluded | No per-model default path repair; no model full score admission; no leaderboard completion. |

## Code Changes

- `bayesfilter/highdim/ledh_score_contract.py`
  - Added `LEDH_SCORE_PRODUCTION_DTYPE = "float32"`.
  - Added `LEDH_SCORE_PRODUCTION_TF32_MODE = "enabled"`.
  - Added `validate_ledh_score_production_precision`.
  - Full score admission now requires `score_precision` metadata with:
    - `dtype=float32`;
    - `active_dtype=float32`;
    - `tf_dtype=float32`;
    - `tf32_mode=enabled`;
    - `tf32_execution_enabled=true`.
- `bayesfilter/highdim/ledh_score_artifact.py`
  - Added optional `score_precision` propagation in
    `build_ledh_score_artifact`.
- `tests/highdim/test_ledh_score_contract_phase1.py`
  - Added direct precision validator tests.
  - Added full-admission rejection tests for missing precision, missing
    explicit `active_dtype`/`tf_dtype`, `float64`, TF32-disabled metadata, and
    wrong-row compact provenance.
- `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`
  - Added production precision metadata to the full-admission artifact fixture.

## Local Checks

Py-compile:

```bash
python -m py_compile bayesfilter/highdim/ledh_score_contract.py bayesfilter/highdim/ledh_score_artifact.py
```

Result: passed.

Shared tests:

```bash
pytest -q tests/highdim/test_ledh_score_contract_phase1.py tests/highdim/test_ledh_score_artifact_emitter_phase1.py
```

Result: `57 passed, 2 warnings` after substitute-review repairs.

Route/precision search:

```bash
rg -n "manual_total_vjp|memory_style|compact_forward_sensitivity|score_precision|validate_ledh_score_production_precision" bayesfilter/highdim tests/highdim
```

Result: `65` matches recorded for audit.

## Review Status

Claude review is not available because the launch Claude gate was rejected by
execution policy as external data disclosure. A substitute Codex review found
two fixable blockers:

- `score_precision.active_dtype` and `score_precision.tf_dtype` were defaulted
  instead of explicitly required.
- Full admission accepted any compact provenance string instead of requiring
  the compact provenance for the same row/model.

Both blockers were patched in `ledh_score_contract.py` and covered by focused
tests. Review status after repair: `REPAIRED_AFTER_REVISE`; focused local
checks passed.

## Plain-Language Gate

- Target: shared score artifact admission guard.
- Computed quantity: validation of serialized score artifacts, not score
  computation.
- Direct classification: shared contract is `correct` for blocking historical
  full admission and missing production precision metadata.
- Unsupported: model runners are not yet repaired by Phase 1.

## Next Phase Handoff

Phase 2 can start after review if the Phase 2 subplan is accepted. Phase 2 is
limited to LGSSM cleanup: preserve compact default, remove misleading naming
where safe, and add score timing instrumentation.
