# Phase 4 Repair Rung 2 Result: Predator-Prey Tiny Total Score

metadata_date: 2026-07-07
status: `RUNG2_TINY_TOTAL_SCORE_PASSED_FULL_SCORE_NOT_YET_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4-repair-rung2

## Objective

Wire a tiny predator-prey no-tape total-score route for the same finite-`N`
LEDH scalar:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

This rung targets a tiny diagnostic scale only. It is not full-row score
admission.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Rung 2 passed at tiny scale. Full score remains not admitted. |
| Primary criterion status | Met for this rung: tiny no-tape route runs under runtime autodiff sentinel and matches coordinate-wise same-scalar finite differences. |
| Veto diagnostic status | Full-row memory/correctness gate not run, so score admission remains blocked. |
| Main uncertainty | Need full `T=20,N=10000` score artifact and memory diagnostics, or a reviewed smaller-to-full evidence bridge. |
| Next justified action | Prepare a full-row GPU score/memory command only after reviewing the tiny route and artifact boundary. |
| What is not concluded | No predator-prey full score admission; no N=10000 memory/correctness pass; no HMC, posterior, runtime, source-faithfulness, exact-likelihood, or scientific claim. |

## What Was Implemented

Updated:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

The tiny score route now:

- builds the same fixed-randomness initial particles and process-noise stream
  as the value runner;
- replays the same target correction formula;
- reverses through manual streaming transport VJP;
- reverses through log-weight normalization;
- reverses through transition and observation Gaussian log-density VJPs;
- reverses through LEDH linearized flow VJP;
- reverses through pre-flow process-noise push;
- uses the Rung 1 RK4 transition-mean VJP to accumulate all six physical
  parameter scores.

The score artifact normalizer now:

- writes Phase 1 schema fields;
- marks tiny diagnostics as `tiny_score_diagnostic_not_admitted`;
- refuses full admission without explicit all-parameter correctness and
  `N=10000` memory pass.

## Local Checks

Focused tiny score checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

Result:

```text
8 passed, 2 warnings
```

Combined Phase 4 checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
29 passed, 2 warnings
```

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the predator-prey no-tape total-score route run at tiny scale and match all-coordinate same-scalar finite differences? |
| Answer | Yes at tiny scale. |
| Baseline/comparator | Coordinate-wise central finite differences with fixed randomness and the same scalar. |
| Primary criterion | Passed for tiny scale only. |
| Veto diagnostics | Full-row memory/correctness gate remains absent; no score admission. |
| Artifact | This result, score helper module, and tests. |

## Next Gate

Before any full `N=10000,T=20` GPU score/memory run:

- review the tiny route and full-run command;
- ensure logs/artifacts are bounded;
- ensure the full artifact will validate through
  `validate_ledh_score_artifact(..., require_admitted=True)`;
- preserve the option to write a blocker result if full execution is too slow,
  runs out of memory, or lacks all-parameter correctness.

## Nonclaims

- Tiny score correctness is not full-row score admission.
- Full-row `N=10000,T=20` score/memory evidence has not been produced.
- No exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, HMC
  readiness, posterior correctness, scientific superiority, runtime ranking,
  or all-algorithm comparison is claimed.
