# Phase 5 Result: Predator-Prey Compact Score Port

Date: 2026-07-08

Status: `PASSED_TINY_COMPACT_PREDATOR_PREY_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Predator-prey now has a compact forward-sensitivity score route for the tiny same-scalar gate. | Passed for tiny `N=2,T=1`, all six physical `(r,K,a,s,u,v)` coordinates. | No tape/autodiff runtime sentinel failure; no reverse-record source in compact default symbols; historical `manual_total_vjp*` remains blocked from full admission. | No full `N=10000,T=20` memory evidence yet. | Proceed to Phase 6 generalized-SV compact score planning/implementation. | Full predator-prey leaderboard score admission, HMC readiness, posterior correctness, exact nonlinear likelihood correctness, or Zhao-Cui source-faithfulness. |

## Evidence Contract

| Field | Result |
| --- | --- |
| Question | Can predator-prey compute the same finite-`N` LEDH `log_likelihood` score in physical `(r,K,a,s,u,v)` coordinates using compact forward sensitivity instead of the historical reverse/manual-total-VJP route? |
| Baseline/comparator | Admitted predator-prey value artifact, existing historical score diagnostic, fixed-SIR compact port pattern, LGSSM compact reference, and tiny same-scalar finite differences. |
| Primary criterion | Passed at tiny scale: compact route carries particles, weights, tangents, and log-likelihood tangents forward; emits compact provenance; passes all-coordinate FD; old route cannot full-admit. |
| Veto diagnostics | No wrong scalar, autodiff, stopped partial derivative, reverse-record default, non-predator-prey row promotion, or historical full admission was accepted. |
| Explanatory diagnostics | Compact objective matched the value-only route exactly at the tiny gate; FD max absolute error was `1.4451367178480723e-06`; FD max relative error was `7.776625302494132e-09`. |
| Artifact | `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-tiny-compact-score-2026-07-08.json` |

## Implementation Summary

Changed files:

- `bayesfilter/highdim/ledh_score_contract.py`
- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Key implementation points:

- Added compact predator-prey provenance:
  `compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`.
- Implemented compact predator-prey forward sensitivities for:
  - predator-prey RHS;
  - RK4 transition mean;
  - Gaussian transition and observation log densities;
  - LEDH linearized flow;
  - normalized log-weight update;
  - streaming finite Sinkhorn transport value+JVP.
- Updated default across-seed score dispatch to use compact forward sensitivity.
- Updated score artifact provenance to use the route carried by the diagnostic.
- Kept `_manual_value_and_score_from_components` as historical/diagnostic only.

## Tiny Artifact

Artifact path:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-tiny-compact-score-2026-07-08.json`

Key fields:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `batch_seeds = [81120]`
- `time_steps = 1`
- `num_particles = 2`
- `max_abs_error = 1.4451367178480723e-06`
- `max_rel_error = 7.776625302494132e-09`

The artifact validates with `validate_ledh_score_artifact(..., require_admitted=False)` and correctly fails `require_admitted=True` because no full-row memory gate has run.

## Local Checks

Baseline precheck before compact migration:

```text
35 passed, 2 warnings
```

Post-implementation checks passed:

```bash
python -m py_compile \
  bayesfilter/highdim/ledh_score_contract.py \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
38 passed, 2 warnings
```

Tiny artifact command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  --source-value-artifact docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-tiny-compact-score-2026-07-08.json \
  --batch-seeds 81120 \
  --time-steps 1 \
  --num-particles 2 \
  --transport-policy active-all \
  --sinkhorn-iterations 1 \
  --row-chunk-size 2 \
  --col-chunk-size 2 \
  --particle-chunk-size 2 \
  --dtype float64 \
  --tf32-mode disabled
```

Artifact readback:

```text
compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot
tiny_score_diagnostic_not_admitted
False
```

## Boundary Notes

- The historical `manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot` route remains diagnostic only.
- The predator-prey compact tiny result is not a full leaderboard score admission.
- No full `N=10000,T=20` predator-prey score run was launched in this phase.
- No public benchmark, HMC, posterior, source-faithfulness, or scientific-superiority claim is made.

## Next Phase Handoff

Phase 6 generalized-SV may start after read-only review of this result and the Phase 6 subplan. Phase 6 must preserve the source-route prior-mean generalized-SV raw-y target likelihood and active transformed coordinate system; it must not substitute actual-SV or KSC score targets.
