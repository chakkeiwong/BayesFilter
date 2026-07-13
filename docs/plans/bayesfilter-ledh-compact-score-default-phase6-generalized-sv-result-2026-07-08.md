# Phase 6 Result: Generalized-SV Compact Score Port

Date: 2026-07-08

Status: `PASSED_TINY_COMPACT_GENERALIZED_SV_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Generalized-SV now has a compact forward-sensitivity score route for the tiny same-scalar gate. | Passed for tiny `N=8,T=2`, all three active transformed coordinates. | No tape/autodiff runtime sentinel failure; compact default symbols contain no reverse records; historical `manual_total_vjp*` remains blocked from full admission. | No full `N=10000,T=1008` generalized-SV score memory evidence yet. | Proceed to Phase 7 KSC-SV compact score planning/implementation. | Full generalized-SV leaderboard score admission, HMC readiness, posterior correctness, SP500 validity, author-default truth validity, or scientific superiority. |

## Evidence Contract

| Field | Result |
| --- | --- |
| Question | Can generalized-SV compute the same finite-`N` source-route prior-mean raw-y LEDH `log_likelihood` score in active transformed coordinates using compact forward sensitivity? |
| Baseline/comparator | Admitted generalized-SV value artifact, generalized-SV value runner, previous compact model ports, and tiny same-scalar finite differences. |
| Primary criterion | Passed at tiny scale: compact route carries particles, log weights, particle tangents, log-weight tangents, log-likelihood, and log-likelihood tangents forward; emits compact provenance; matches the value route; passes all-coordinate FD; does not substitute actual-SV or KSC semantics. |
| Veto diagnostics | No wrong target scalar, actual-SV/KSC substitution, KSC surrogate promotion, wrong coordinate order, stopped partial derivative, tape/autodiff, reverse-record default, nonfinite score, or tiny FD failure was accepted. |
| Explanatory diagnostics | Compact objective matched both the local value replay and `benchmark_ledh_same_target_generalized_sv_value._generalized_sv_value_core` at the tiny gate. FD max absolute error was `4.1007384898997246e-05`; FD max relative error was `0.001305349464063811`. |
| Artifact | `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-tiny-compact-score-2026-07-08.json` |

## Implementation Summary

Changed files:

- `bayesfilter/highdim/ledh_score_contract.py`
- `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py`
- `tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py`

Key implementation points:

- Added compact generalized-SV provenance:
  `compact_forward_sensitivity_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot`.
- Implemented compact generalized-SV forward sensitivities for:
  - `gamma = Phi(gamma_unconstrained)`;
  - `tau = exp(log_tau)`;
  - stationary initial particles;
  - prior-mean AR(1) transition `mu + gamma * (x - mu)`;
  - proposal flow observation surface `tau * x`;
  - raw-y target observation density `y_t | x_t ~ Normal(0, exp(tau*x_t))`;
  - LEDH linearized flow with parameterized observation Jacobian;
  - normalized log-weight update;
  - streaming finite Sinkhorn transport value+JVP.
- Generated initial particles and proposal noises from the same stateless seeds
  used by the value runner, so finite differences perturb theta while holding
  estimator randomness fixed.
- Kept the log-square Gaussianized observation surface proposal-only; it is not
  the score target likelihood.

## Tiny Artifact

Artifact path:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-tiny-compact-score-2026-07-08.json`

Key fields:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `score_parameter_names = [gamma_unconstrained, log_tau, mu]`
- `batch_seeds = [81120]`
- `time_steps = 2`
- `num_particles = 8`
- `max_abs_error = 4.1007384898997246e-05`
- `max_rel_error = 0.001305349464063811`

The artifact validates with `validate_ledh_score_artifact(..., require_admitted=False)` and correctly remains non-admitted because no full-row memory gate has run.

## Local Checks

Precheck:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py
```

Post-implementation checks passed:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py \
  bayesfilter/highdim/ledh_score_contract.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
34 passed, 2 warnings
```

Tiny artifact command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py \
  --source-value-artifact docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-tiny-compact-score-2026-07-08.json \
  --markdown-output docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-tiny-compact-score-2026-07-08.md \
  --batch-seeds 81120 \
  --time-steps 2 \
  --num-particles 8 \
  --particle-chunk-size 4 \
  --row-chunk-size 4 \
  --col-chunk-size 4 \
  --sinkhorn-iterations 1 \
  --fd-step 1e-5 \
  --score-fd-atol 1e-3 \
  --score-fd-rtol 2e-3 \
  --dtype float64
```

Artifact readback:

```text
compact_forward_sensitivity_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot
tiny_score_diagnostic_not_admitted
False
```

## Boundary Notes

- The target scalar remains `observed_data_log_likelihood_estimator`, reported
  as `log_likelihood`.
- The target observation policy remains `source_route_prior_mean_generalized_sv`.
- The coordinate order remains `gamma_unconstrained, log_tau, mu`.
- The flow observation policy remains `log_square_gaussian_surrogate_for_ledh_flow_only`.
- No full `N=10000,T=1008` generalized-SV score run was launched in this phase.
- No KSC, actual-SV, SP500, author-default, HMC, posterior, or scientific-superiority claim is made.

## Next Phase Handoff

Phase 7 KSC-SV may start after read-only review of this result and the Phase 7 subplan. Phase 7 must preserve the KSC finite-mixture surrogate target and must not infer KSC score correctness from generalized-SV or actual-SV evidence.
