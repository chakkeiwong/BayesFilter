# Phase 3 Result: Fixed-SIR Score

metadata_date: 2026-07-07
status: `CLOSED_BLOCKED_FULL_SCORE_NOT_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 3

## Phase Objective

Admit, or explicitly block, the LEDH score for:

```text
zhao_cui_spatial_sir_austria_j9_T20
```

The target score is the no-tape total derivative of the same realized
finite-`N` LEDH estimator admitted by the fixed-SIR value artifact:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Fixed-SIR score is blocked, not admitted, for the current runbook. |
| Primary criterion status | Not met: existing `N=10000` memory evidence has only a directional same-scalar FD diagnostic, not all-parameter score correctness. |
| Veto diagnostic status | No row-id/target/no-tape/memory veto found for the existing diagnostic; directional-only correctness vetoes full score admission. |
| Main uncertainty | Need explicit all-parameter same-scalar correctness for `[log_kappa_scale, log_nu_scale, log_obs_noise_scale]`. |
| Next justified action | Proceed to Phase 4 predator-prey with fixed-SIR recorded as not admitted. |
| What is not concluded | No fixed-SIR score admission; no rejection of the manual total-VJP mathematics; no exact nonlinear likelihood, HMC, posterior, source-faithfulness, runtime, or scientific claim. |

## What Passed

The existing fixed-SIR score-memory artifact is tied to the main fixed-SIR row:

- row id: `zhao_cui_spatial_sir_austria_j9_T20`;
- target scope: `main_observed_data_filtering_row`;
- `N=10000`;
- finite three-parameter score vector;
- manual route:
  `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`;
- memory peak: `3166.76904296875 MiB` under a `14000 MiB` budget;
- directional same-scalar FD diagnostic passed with absolute error
  `0.001092374324798584` and relative error `0.001272708410397172`.

Artifact:

- `docs/plans/ledh-phase5-fixed-sir-score-memory-n10000-2026-07-06.json`

## What Blocked Admission

The artifact's correctness evidence is only:

```text
same_scalar_directional_finite_difference
```

A single directional finite difference checks one projection of the
three-dimensional score. It does not establish all-parameter correctness for:

```text
log_kappa_scale, log_nu_scale, log_obs_noise_scale
```

The Phase 3 adapter now refuses flag-only promotion. Full admission requires an
explicit all-parameter correctness record before
`validate_ledh_score_artifact(..., require_admitted=True)` may pass.

## Code And Test Changes

Changed:

- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`

The adapter now requires an explicit `all_parameter_score_correctness` mapping
with:

- kind: `same_scalar_finite_difference`;
- status: `pass`;
- parameter names exactly matching
  `[log_kappa_scale, log_nu_scale, log_obs_noise_scale]`.

Without that record, a caller cannot promote directional evidence by passing a
Boolean flag.

## Local Checks

Command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/test_ledh_fixed_sir_manual_score_phase4.py tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
29 passed, 2 warnings
```

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the fixed-SIR main row produce an admitted no-tape total derivative of the same finite-`N` LEDH `log_likelihood` scalar as the value artifact? |
| Answer | Not yet. The route and memory evidence are promising diagnostics, but all-parameter correctness evidence was not produced. |
| Baseline/comparator | Admitted fixed-SIR value artifact, existing manual total-VJP route, `N=10000` memory artifact, and directional same-scalar FD diagnostic. |
| Primary criterion | Failed/not met because no admitted score artifact exists. |
| Veto diagnostics | Directional-only correctness evidence vetoes admission. |
| Explanatory diagnostics | Runtime, memory, finite score, and directional FD are diagnostic only. |
| Artifact | This Phase 3 result; no admitted fixed-SIR score artifact. |

## Phase 4 Handoff

Phase 4 predator-prey may begin with these inherited conditions:

- LGSSM score is blocked/not admitted.
- Fixed-SIR score is blocked/not admitted.
- Directional FD remains diagnostic only.
- Predator-prey starts from its admitted value artifact and no score route is
  assumed admitted.

Required next subplan:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-subplan-2026-07-07.md`

## Nonclaims

- Fixed-SIR score is not admitted.
- Directional FD is not all-parameter correctness.
- Existing memory evidence is not score correctness evidence by itself.
- No exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, HMC
  readiness, posterior correctness, scientific superiority, runtime ranking,
  or all-algorithm comparison is claimed.
