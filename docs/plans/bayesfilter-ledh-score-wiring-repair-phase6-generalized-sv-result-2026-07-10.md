# Phase 6 Result: Generalized-SV Compact Precision Gate

Date: 2026-07-10

Status: `PASSED_GENERALIZED_SV_COMPACT_PRECISION_GATE_AFTER_BATCHING_REPAIR`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Generalized-SV score wiring now exposes its existing compact forward-sensitivity route directly and enforces the shared production precision and full-admission boundaries. | Passed focused CPU-hidden wiring tests: the compact score executes without autodiff, the tiny same-scalar finite-difference check passes, the FD perturbations use a value-only objective, and artifact target semantics remain source-route prior-mean generalized-SV. | Full admission rejects non-compact nested provenance, false no-autodiff/same-route declarations, non-production precision, and mismatched particles, time steps, or seeds. | No trusted full `N=10000,T=1008` generalized-SV score-memory run was launched. | Review this result and the Phase 7 KSC-SV subplan. | No full generalized-SV score admission, leaderboard rebuild, HMC readiness, posterior correctness, runtime ranking, or scientific superiority claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is generalized-SV wired so its existing compact score route is compatible with the shared production precision and full-admission gates? |
| Baseline/comparator | The pre-phase generalized-SV compact implementation, whose module and CLI defaulted to `float64`/TF32 disabled and whose artifact constructor lacked precision and nested full-row checks. |
| Primary criterion | Passed. `_coordinate_fd_score_diagnostic` now takes its score from `_compact_value_and_score_across_seeds`, which invokes `_compact_value_and_score_from_components` sequentially for each fixed-randomness seed; finite-difference perturbations call a value-only same-scalar objective; focused tiny checks and the no-autodiff sentinel pass. |
| Veto diagnostics | Passed. Production defaults are `float32` with TF32 enabled; full artifact construction requires compact nested provenance, production `score_precision`, admitted-value row shape, matching seeds, and trusted memory diagnostics. Target substitution is rejected by the shared contract. |
| Explanatory diagnostics | Tiny CPU-hidden compact/value parity and finite-difference checks. These do not establish full-row memory behavior or scientific validity. |
| Artifact | This result plus focused tests; no new score JSON admission artifact was produced. |

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed target | Score of the realized finite-`N` observed-data LEDH log-likelihood estimator for the admitted `source_route_prior_mean_generalized_sv` row. |
| Quantity checked locally | Compact forward sensitivity of the tiny fixed-randomness generalized-SV scalar, compared with central finite differences of the value-only route. |
| Relationship | Equal within the declared tiny finite-difference tolerance in the focused CPU-hidden test. Full-row equality and memory admissibility were not checked. |
| Support | `tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py` and the final focused pytest result below. |

## Changed Files

- `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py`
- `tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py`
- `bayesfilter/linear/kalman_qr_tf.py` (one-line prerequisite syntax repair)

## Implementation Summary

- Set the generalized-SV score module default dtype to `tf.float32`.
- Changed CLI defaults to `--dtype float32` and `--tf32-mode enabled`.
- Added explicit `score_precision` metadata with `dtype`, `active_dtype`,
  `tf_dtype`, `tf32_mode`, and `tf32_execution_enabled`.
- Made `_coordinate_fd_score_diagnostic` call the explicitly named compact
  across-seed wrapper for the score base. The wrapper invokes
  `_compact_value_and_score_from_components` sequentially per seed, preserving
  the pre-phase bounded-memory seed schedule.
- Kept finite-difference perturbations on a value-only same-scalar objective.
- Added compact base metadata for batch seeds, time steps, particle count, and
  transport settings.
- Hardened `_score_artifact_from_diagnostic` against nested non-compact
  relabeling and mismatched full-row particles, time steps, or seeds.
- Preserved the KSC-free raw generalized-SV target route and
  `source_route_prior_mean_generalized_sv` observation policy.

## Engineering Correctness Ledger

The first focused pytest attempt did not run any Phase 6 test. Collection found
a committed syntax error in `bayesfilter/linear/kalman_qr_tf.py`: one
`tf.while_loop` call repeated `maximum_iterations=n_timesteps`. The working tree
matched `HEAD`, so this was baseline state rather than a concurrent user edit.
The prerequisite repair removed only the duplicate keyword. Independent
`py_compile` then passed and the exact focused test command collected normally.
This repair is engineering-harness evidence only and is not generalized-SV
numerical or scientific evidence.

## Local Checks

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  bayesfilter/linear/kalman_qr_tf.py \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Final result:

```text
68 passed, 2 warnings in 39.86s
```

The earlier `36.69s` run preceded the sequential-seed repair and is
superseded. The `39.86s` result is the binding post-repair focused run.

Prerequisite QR-module focused check:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/test_linear_kalman_qr_tf.py
```

Result: `8 passed, 2 warnings in 6.04s`.

Source search found the sequential compact across-seed score base, value-only FD objective,
production precision defaults, explicit score precision, nested compact route
guards, exact full-row shape/seed guards, and preserved
`source_route_prior_mean_generalized_sv` policy.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the uncommitted scoped changes listed above |
| Command | Final focused pytest command shown above |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; TensorFlow `2.19.1` |
| CPU/GPU status | CPU-only diagnostic; `CUDA_VISIBLE_DEVICES=-1` intentionally hid GPU devices before framework import |
| Data version | Admitted generalized-SV forward artifact dated 2026-07-07 |
| Random seeds | Tiny test seed `81120`; full-row admission fixtures use `81120` through `81124` without running the full computation |
| Wall time | `39.86s` for the final post-batching-repair focused pytest run |
| Output artifacts | This result and pytest terminal result; no score JSON admission artifact |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md` |
| Result file | This file |

## Post-Run Red Team

- Strongest alternative explanation: the tiny finite-difference test may pass
  while a full `N=10000,T=1008` GPU execution exceeds memory or exposes a
  scale-dependent error.
- Result that would overturn the wiring decision: a focused test showing the
  default diagnostic reaches the compatibility wrapper or a score-bearing
  route for FD perturbations, or an admitted artifact that accepts mismatched
  nested provenance or shape.
- Weakest evidence: no full-row trusted GPU score-memory artifact exists for
  generalized-SV in this phase.

## Boundary Notes

- CPU-hidden checks are wiring evidence only.
- No stochastic method ranking was performed; no ranking is supported.
- Full score admission still requires trusted GPU memory evidence at
  `N=10000,T=1008` with row-matched compact provenance.
- This result does not establish KSC-SV wiring, native actual-SV likelihood
  correctness, posterior correctness, HMC readiness, or scientific validity.

## Next Phase Handoff

Phase 7 KSC-SV may start only after review of this result and the Phase 7
subplan. Phase 7 must preserve the admitted KSC Gaussian-mixture surrogate
target and keep `claims_exact_native_actual_sv_likelihood = false` while adding
the same production precision and full-admission hardening.

The first Codex substitute review returned `VERDICT: AGREE`. A subsequent
cross-model audit found that calling the component helper once with all seeds
would change the prior sequential seed schedule and could inflate full-row
memory. The implementation was repaired to use an explicitly named compact
sequential-seed wrapper; the first verdict is superseded pending focused
re-review. This is a batching/memory-boundary repair, not a target or score-math
change.

Fresh post-repair substitute re-review returned `VERDICT: AGREE` after checking
the sequential-seed wrapper and the `68 passed` post-repair suite.
