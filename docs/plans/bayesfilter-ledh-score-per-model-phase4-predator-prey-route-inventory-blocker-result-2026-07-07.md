# Phase 4 Blocker Result: Predator-Prey Score Route Inventory

metadata_date: 2026-07-07
status: `BLOCKED_FIXABLE_REPAIR_LOOP_OPEN`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4

## Phase Objective

Admit, or explicitly block, the LEDH score for:

```text
zhao_cui_predator_prey_T20
```

The target score remains the no-tape total derivative of the same realized
finite-`N` LEDH estimator admitted by the predator-prey value artifact:

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
| Decision | Predator-prey score is blocked by a fixable missing-adapter problem; enter Phase 4 repair loop. |
| Primary criterion status | Not met: no same-target no-tape total LEDH score adapter exists for predator-prey. |
| Veto diagnostic status | Value artifact replay passes; score admission is vetoed because current code exposes value-only transport and model-local density parameter scores, not total LEDH score. |
| Main uncertainty | Need a reviewed manual reverse scan through transport, log-weight normalization, LEDH flow, and RK4 transition dynamics for all six physical parameters. |
| Next justified action | Draft and review a Phase 4 repair subplan for the predator-prey manual total-VJP adapter. |
| What is not concluded | No predator-prey score admission; no rejection of the feasibility of the manual route; no exact nonlinear likelihood, HMC, posterior, source-faithfulness, runtime, or scientific claim. |

## What Passed

Preflight checks passed:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py \
  bayesfilter/highdim/models.py \
  bayesfilter/highdim/ledh_score_contract.py
```

Forward value/schema replay passed:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
21 passed, 2 warnings
```

Additional Phase 4 boundary tests passed:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
23 passed, 2 warnings
```

## Inventory Findings

The admitted value path is:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py`

It uses:

- row id: `zhao_cui_predator_prey_T20`;
- theta coordinate: `physical`;
- parameter order: `[r,K,a,s,u,v]`;
- target observation policy: `additive_gaussian_predator_prey`;
- target scalar: `observed_data_log_likelihood_estimator`;
- output field: `log_likelihood`;
- streaming LEDH-PFPF-OT value core;
- target-density correction:
  `transition_log_density + observation_log_density - pre_flow_log_density + forward_log_det`.

The model exposes local manual density parameter scores:

- `PredatorPreySSM.transition_mean_parameter_jacobian`;
- `PredatorPreySSM.transition_log_density_parameter_score`;
- `PredatorPreySSM.observation_log_density_parameter_score`.

Those local methods are useful ingredients, but they are not an admitted LEDH
score route because they do not propagate the total derivative through:

- relaxed Sinkhorn transport;
- log-weight normalization;
- LEDH linearized flow;
- pre-flow process-noise map;
- prior mean dependence across the filtering time loop;
- state cotangents through the RK4 dynamics.

## Blocker

There is no current predator-prey function equivalent to the fixed-SIR
manual total reverse scan:

```text
manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot
```

Therefore the value artifact, local-density scores, runtime, finite output,
and any directional finite difference are diagnostic only.

## Guard Added

Added:

- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

The test confirms:

- the admitted predator-prey value artifact is not a score artifact;
- directional-only predator-prey score evidence cannot pass
  `validate_ledh_score_artifact(..., require_admitted=True)`.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can predator-prey currently produce an admitted no-tape total derivative of the same finite-`N` LEDH scalar as the value artifact? |
| Answer | Not before repair. The required total-score adapter is missing. |
| Baseline/comparator | Admitted predator-prey value artifact, value runner, model local-density score helpers, and score schema guards. |
| Primary criterion | Failed/not met because no score artifact exists and no total-score adapter exists. |
| Veto diagnostics | Missing total-score adapter vetoes admission. |
| Explanatory diagnostics | Local-density score helpers and value replay are ingredients only. |
| Artifact | This blocker result and the Phase 4 repair subplan. |

## Repair Handoff

Open repair subplan:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-subplan-2026-07-07.md`

The repair loop must not start full `N=10000,T=20` score execution until tiny
manual total-VJP and artifact-schema checks pass.

## Nonclaims

- Predator-prey score is not admitted.
- Local-density parameter scores are not total LEDH score evidence.
- Value-only replay is not score evidence.
- No exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, HMC
  readiness, posterior correctness, scientific superiority, runtime ranking,
  or all-algorithm comparison is claimed.
