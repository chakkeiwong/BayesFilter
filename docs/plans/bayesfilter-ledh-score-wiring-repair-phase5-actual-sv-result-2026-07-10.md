# Phase 5 Result: Actual-SV Compact Default And Precision Gate

Date: 2026-07-10

Status: `PASSED_ACTUAL_SV_COMPACT_WIRING_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Actual-SV score wiring now treats compact forward sensitivity as the current/admissible route for the transformed actual-SV scalar. | Passed focused CPU-hidden wiring tests: compact score executes without autodiff sentinel, same-scalar FD passes at tiny scale, target policy remains transformed actual-SV, artifact provenance is compact, and full fixtures require production precision. | Historical reverse/manual routes cannot be nested under full compact admission; CLI defaults no longer request `float64` or TF32 disabled; exact native likelihood claim remains false. | No trusted full `N=10000,T=1000` actual-SV score-memory run was launched in this phase. | Proceed to Phase 6 generalized-SV compact precision gate. | No full actual-SV score admission, no exact native likelihood claim, no leaderboard rebuild, no HMC readiness, no posterior correctness, no scientific superiority claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is actual-SV wired so compact forward-sensitivity is the only full-admissible score path for the transformed actual-SV same scalar? |
| Baseline/comparator | Previous actual-SV tests and coordinate-FD diagnostic routed through memory-style reverse/manual score. |
| Primary criterion | Passed. `_coordinate_fd_score_diagnostic` uses `_compact_value_and_score_from_components` as the score base and value-only same-scalar objectives for FD; `_score_artifact_from_diagnostic` rejects nested non-compact bases. |
| Veto diagnostics | Passed. Historical/manual route full admission is rejected, target substitution is rejected, full-admission fixtures require row shape match, memory pass, `float32`/TF32-enabled `score_precision`, and `claims_exact_native_actual_sv_likelihood = false`. |
| Explanatory diagnostics | Existing value-route and particle-chunk parity checks continue to pass at tiny scale. |
| Artifact | This result plus focused tests; no new score JSON admission artifact was produced. |

## Changed Files

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

## Implementation Summary

- Set actual-SV score module default dtype to `tf.float32`.
- Changed CLI defaults to `--dtype float32` and `--tf32-mode enabled`.
- Added explicit `score_precision` metadata with `dtype`, `active_dtype`,
  `tf_dtype`, `tf32_mode`, and `tf32_execution_enabled`.
- Changed `_coordinate_fd_score_diagnostic` so its score base is
  `_compact_value_and_score_from_components`.
- Kept the finite-difference comparator as a value-only same-scalar route.
- Added compact base metadata: `batch_seeds`, `time_steps`, `num_particles`,
  and transport settings.
- Hardened `_score_artifact_from_diagnostic` against nested historical/manual
  relabeling and against tiny-shape promotion.
- Preserved `_manual_value_and_score_from_components` and
  `_manual_value_and_score_across_seeds` as historical diagnostic-only routes.
- Preserved the transformed actual-SV observation policy and exact-native
  likelihood nonclaim.

## Local Checks

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result:

```text
70 passed, 2 warnings in 48.54s
```

Source search:

```bash
rg -n "base = _compact_value_and_score_from_components|_value_objective_across_seeds|score_precision|default=\"float32\"|default=\"enabled\"|Historical reverse/manual|compact diagnostic must use compact score route|N=10000 diagnostic shape|transformed_actual_sv_log_y_square|claims_exact_native_actual_sv_likelihood" \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

Result: found compact diagnostic base, value-only FD comparator, production
precision defaults, historical-route demotion text, nested compact route guards,
transformed actual-SV target policy, and exact-native likelihood nonclaim.

## Boundary Notes

- CPU-hidden local checks are wiring evidence only.
- Full score admission still requires trusted GPU memory evidence at
  `N=10000,T=1000` and row-matched compact provenance.
- The historical memory-style/manual route remains in the module only as a
  diagnostic comparator and must not be used for full score admission.
- This phase does not claim exact native actual-SV likelihood correctness.

## Next Phase Handoff

Phase 6 generalized-SV may start after review of this result and the Phase 6
subplan. Phase 6 should preserve the existing compact route and source-route
prior-mean generalized-SV target, while adding the same production precision
and full-admission boundary hardening required by the shared score contract.
