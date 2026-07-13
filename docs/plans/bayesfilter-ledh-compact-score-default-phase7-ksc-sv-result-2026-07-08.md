# Phase 7 Result: KSC-SV Compact Score Port

Date: 2026-07-08

Status: `PASSED_TINY_COMPACT_KSC_SV_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| KSC-SV now has a compact forward-sensitivity score route for the tiny same-scalar gate. | Passed for tiny `N=8,T=2`, both synthetic unconstrained coordinates. | No tape/autodiff runtime sentinel failure; compact default symbols contain no reverse records; historical `manual_total_vjp*` remains blocked from full admission; exact native actual-SV overclaim is rejected. | No full `N=10000,T=1000` KSC-SV score memory evidence yet. | Proceed to Phase 8 integration planning/checks. | Full KSC-SV leaderboard score admission, exact native actual-SV likelihood, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |

## Evidence Contract

| Field | Result |
| --- | --- |
| Question | Can KSC-SV compute the same finite-`N` KSC finite-mixture surrogate LEDH `log_likelihood` score in `synthetic_unconstrained` coordinates using compact forward sensitivity? |
| Baseline/comparator | Admitted KSC-SV value artifact, KSC-SV value runner, previous compact model ports, and tiny same-scalar finite differences. |
| Primary criterion | Passed at tiny scale: compact route carries particles, log weights, particle tangents, log-weight tangents, log-likelihood, and log-likelihood tangents forward; emits compact provenance; matches the value route; passes all-coordinate FD; does not substitute exact actual-SV, generalized-SV, or raw Gaussian semantics. |
| Veto diagnostics | No wrong target scalar, exact actual-SV target substitution, generalized-SV substitution, raw Gaussian callback promotion, wrong coordinate order, stopped partial derivative, tape/autodiff, reverse-record default, nonfinite score, or tiny FD failure was accepted. |
| Explanatory diagnostics | Compact objective matched both the local value replay and `benchmark_ledh_same_target_ksc_sv_value._ksc_sv_value_core` at the tiny gate. FD max absolute error was `1.688629603341374e-05`; FD max relative error was `6.364022943512981e-05`. |
| Artifact | `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.json` |

## Implementation Summary

Changed files:

- `bayesfilter/highdim/ledh_score_contract.py`
- `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py`
- `tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py`

Key implementation points:

- Added compact KSC-SV provenance:
  `compact_forward_sensitivity_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot`.
- Implemented compact KSC-SV forward sensitivities for:
  - `gamma = Phi(gamma_unconstrained)`;
  - `beta = exp(log_beta)`;
  - stationary initial particles;
  - initial and positive-time transition densities;
  - LEDH proposal flow surface `x + 2 log(beta)`;
  - KSC finite-mixture target density using component responsibilities;
  - normalized log-weight update;
  - streaming finite Sinkhorn transport value+JVP.
- Generated initial particles and proposal noises from the same stateless seeds
  used by the value runner, so finite differences perturb theta while holding
  estimator randomness fixed.
- Kept the KSC finite mixture as the target likelihood and explicitly rejected
  exact native actual-SV likelihood overclaims.

## Tiny Artifact

Artifact path:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.json`

Key fields:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `score_parameter_names = [gamma_unconstrained, log_beta]`
- `batch_seeds = [81120]`
- `time_steps = 2`
- `num_particles = 8`
- `max_abs_error = 1.688629603341374e-05`
- `max_rel_error = 6.364022943512981e-05`

The artifact validates with `validate_ledh_score_artifact(..., require_admitted=False)` and correctly remains non-admitted because no full-row memory gate has run.

## Local Checks

Precheck:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py
```

Post-implementation checks passed:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
33 passed, 2 warnings
```

Tiny artifact command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py \
  --source-value-artifact docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.json \
  --markdown-output docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.md \
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
compact_forward_sensitivity_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot
tiny_score_diagnostic_not_admitted
False
```

## Boundary Notes

- The target scalar remains `observed_data_log_likelihood_estimator`, reported
  as `log_likelihood`.
- The target observation policy remains `ksc_log_chi_square_gaussian_mixture_surrogate`.
- The coordinate order remains `gamma_unconstrained, log_beta`.
- The KSC finite mixture is the target likelihood for this row; it is not exact
  native actual-SV likelihood.
- No full `N=10000,T=1000` KSC-SV score run was launched in this phase.
- No HMC, posterior, runtime ranking, or scientific-superiority claim is made.

## Next Phase Handoff

Phase 8 integration may start after read-only review of this result and the Phase 8 subplan. Phase 8 must wire compact score provenance as the default admitted route and keep historical `manual_total_vjp*` routes diagnostic-only.
