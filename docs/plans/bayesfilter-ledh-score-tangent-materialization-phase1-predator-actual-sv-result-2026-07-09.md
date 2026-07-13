# Phase 1 Result: Predator-Prey And Actual-SV Default Wrapper Repair

Date: 2026-07-09

Status: `PASSED_FOCUSED_GATE`

## Objective

Wire predator-prey and actual-SV default/across-seed score diagnostics to the
existing memory-style reverse/VJP routes instead of compact
forward-sensitivity routes, while preserving the same realized finite-`N`
LEDH `observed_data_log_likelihood_estimator` / `log_likelihood` scalar.

## Implementation

Updated:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

The default/across-seed wrappers now emit:

- predator-prey:
  `memory_style_reverse_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`;
- actual-SV:
  `memory_style_reverse_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot`.

The old compact forward-sensitivity route and old `manual_total_vjp*` aliases
remain historical/diagnostic only. Tests check that wrapper source calls the
manual reverse/VJP helper rather than the compact forward-sensitivity helper.

## Checks

Focused model checks were included in the combined gate:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

Result:

```text
79 passed, 2 warnings
```

## Evidence Contract Status

| Requirement | Status |
| --- | --- |
| Predator-prey default score route is memory-style reverse/VJP | Passed |
| Actual-SV default score route is memory-style reverse/VJP | Passed |
| Compact forward-sensitivity helper remains diagnostic/historical | Passed |
| Old `manual_total_vjp*` aliases remain blocked for full admission | Passed |
| No production `GradientTape`/`ForwardAccumulator` default path introduced | Passed by focused sentinel tests |

## Nonclaims

This phase does not claim `N=10000` memory pass, full score admission,
leaderboard completion, HMC readiness, posterior correctness, scientific
superiority, or exact likelihood correctness.

## Handoff

Phase 2 may proceed: fixed-SIR route reclassification can use the same contract
gate, but full admission still requires all-parameter same-scalar correctness
and numeric `N=10000` memory evidence.
