# Phase 4 Result: Fixed-SIR Compact Score Port

Date: 2026-07-08

Status: `PASSED_TINY_COMPACT_FIXED_SIR_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Fixed-SIR now has a compact forward-sensitivity score route for the tiny same-scalar gate. | Passed for tiny `N=8,T=2`, all three `sir_log_scale_theta` coordinates. | No tape/autodiff runtime sentinel failure; no reverse-record source in compact default symbols; historical `manual_total_vjp*` remains blocked from full admission. | No full `N=10000,T=20` memory evidence yet. | Proceed to Phase 5 predator-prey compact migration; later integration must require full-row memory evidence before score admission. | Full fixed-SIR leaderboard score admission, HMC readiness, posterior correctness, exact nonlinear likelihood correctness, or Zhao-Cui source-faithfulness. |

## Evidence Contract

| Field | Result |
| --- | --- |
| Question | Can fixed-SIR compute the same finite-`N` LEDH `log_likelihood` score in `sir_log_scale_theta` coordinates using compact forward sensitivity instead of the historical p8p reverse/manual-total-VJP route? |
| Baseline/comparator | Admitted fixed-SIR value artifact, p8p historical same-target diagnostic, LGSSM/actual-SV compact style, and tiny same-scalar finite differences. |
| Primary criterion | Passed at tiny scale: compact route carries particles, weights, tangents, and log-likelihood tangents forward; emits compact provenance; passes all-coordinate FD; old route cannot full-admit. |
| Veto diagnostics | No wrong scalar, autodiff, stopped partial derivative, reverse-record default, parameterized diagnostic row promotion, or historical full admission was accepted. |
| Explanatory diagnostics | Tiny compact objective matched the historical same-target diagnostic within `7.80006322997906e-06`; FD max absolute error was `0.0005332655891479021`; FD max relative error was `1.7359599675765744e-05`. |
| Artifact | `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-tiny-compact-score-2026-07-08.json` |

## Implementation Summary

Changed files:

- `bayesfilter/highdim/ledh_score_contract.py`
- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`

Key implementation points:

- Added compact fixed-SIR provenance:
  `compact_forward_sensitivity_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`.
- Implemented compact fixed-SIR forward sensitivities for:
  - SIR RK4 transition mean;
  - Gaussian transition and observation log densities;
  - LEDH linearized flow;
  - normalized log-weight update;
  - streaming finite Sinkhorn transport value+JVP.
- Added a tangent-specific SIR flatten helper so `[batch, particle, region, parameter]` tangents preserve the parameter axis as `[batch, particle, 18, parameter]`.
- Fixed the compact FD diagnostic to configure precision before constructing `theta` and `step`.
- Added a compact score artifact builder separate from the historical memory-result normalizer.
- Kept `_fixed_sir_score_artifact_from_memory_result` historical and blocked from full admission.

## Tiny Artifact

Artifact path:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-tiny-compact-score-2026-07-08.json`

Key fields:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `theta_values = [0.0, 0.0, 0.0]`
- `batch_seeds = [81120]`
- `time_steps = 2`
- `num_particles = 8`
- `max_abs_error = 0.0005332655891479021`
- `max_rel_error = 1.7359599675765744e-05`

The artifact validates with `validate_ledh_score_artifact(..., require_admitted=False)` and correctly fails `require_admitted=True` because no full-row memory gate has run.

## Local Checks

Passed:

```bash
python -m py_compile \
  bayesfilter/highdim/ledh_score_contract.py \
  docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py
```

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
37 passed, 2 warnings
```

Passed artifact readback:

```text
compact_forward_sensitivity_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot
tiny_score_diagnostic_not_admitted
False
```

## Repair Notes

Two fixable implementation issues were found and repaired:

1. The first tiny compact diagnostic failed because the value flatten helper was used on tangent tensors and collapsed the parameter axis. A compact-only tangent flatten helper fixed this.
2. A fresh artifact-generation process exposed a dtype-order bug in the FD helper. `_fixed_sir_compact_coordinate_fd_diagnostic` now configures precision before constructing tensors.

Both repairs were followed by focused tests and artifact regeneration.

## Boundary Notes

- The historical `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot` route remains diagnostic only.
- The fixed-SIR compact tiny result is not a full leaderboard score admission.
- No full `N=10000,T=20` fixed-SIR score run was launched in this phase.
- No public benchmark, HMC, posterior, or scientific-superiority claim is made.

## Next Phase Handoff

Phase 5 predator-prey may start after read-only review of this result and the Phase 5 subplan. The Phase 5 starting point is the existing predator-prey same-target score module, which currently uses a historical manual-total-VJP score route and must be migrated to compact forward sensitivity or blocked.
