# Phase 4 Result: Predator-Prey Compact Default And Precision Gate

Date: 2026-07-10

Status: `PASSED_PREDATOR_PREY_COMPACT_WIRING_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Predator-prey score wiring now treats compact forward sensitivity as the current/admissible route. | Passed focused CPU-hidden wiring tests: compact score executes without autodiff sentinel, same-scalar FD passes at tiny scale, artifact provenance is compact, and full fixtures require production precision. | Historical reverse/manual routes cannot be nested under full compact admission; CLI defaults no longer request `float64` or TF32 disabled. | No trusted full `N=10000,T=20` predator-prey score-memory run was launched in this phase. | Proceed to Phase 5 actual-SV compact-default repair. | No full predator-prey score admission, no leaderboard rebuild, no HMC readiness, no posterior correctness, no scientific superiority claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is predator-prey wired so compact forward-sensitivity is the only full-admissible score path, with reverse/manual route demoted and precision enforced? |
| Baseline/comparator | Previous predator-prey tests asserted reverse/manual default; Phase 4 flips that stale expectation to the compact helper already present in the module. |
| Primary criterion | Passed. `_coordinate_fd_score_diagnostic` uses `_compact_value_and_score_from_components` as the score base and value-only same-scalar objectives for FD; `_score_artifact_from_diagnostic` rejects nested non-compact bases. |
| Veto diagnostics | Passed. Historical/manual route full admission is rejected, full-admission fixtures require row shape match, memory pass, and `float32`/TF32-enabled `score_precision`. |
| Explanatory diagnostics | Low-level historical reverse/manual VJP tests remain diagnostic-only; their FD step was adjusted for the new float32 default. |
| Artifact | This result plus focused tests; no new score JSON admission artifact was produced. |

## Changed Files

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

## Implementation Summary

- Set predator-prey score module default dtype to `tf.float32`.
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

## Local Checks

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result:

```text
70 passed, 2 warnings in 123.61s
```

Source search:

```bash
rg -n "base = _compact_value_and_score_from_components|_value_objective_across_seeds|score_precision|default=\"float32\"|default=\"enabled\"|Historical reverse/manual|compact diagnostic must use compact score route|N=10000 diagnostic shape" \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py
```

Result: found compact diagnostic base, value-only FD comparator, production
precision defaults, historical-route demotion text, and nested compact route
guards.

## Boundary Notes

- CPU-hidden local checks are wiring evidence only.
- Full score admission still requires trusted GPU memory evidence at the full
  row shape and row-matched compact provenance.
- The historical memory-style/manual route remains in the module only as a
  diagnostic comparator and must not be used for full score admission.

## Next Phase Handoff

Phase 5 actual-SV may start after review of this result and the Phase 5 subplan.
Phase 5 must repair the same stale default class in
`docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`: compact route
as current/admissible, reverse/manual route diagnostic-only, production
precision metadata required, and transformed actual-SV target preserved.
