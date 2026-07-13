# Phase 3 Subplan: Fixed SIR Score

metadata_date: 2026-07-07
status: `DRAFT_AFTER_PHASE2_BLOCKER`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 3

## Phase Objective

Admit, or explicitly block, the LEDH score for:

```text
zhao_cui_spatial_sir_austria_j9_T20
```

The score target is the no-tape total derivative of the same realized
finite-`N` LEDH estimator admitted by the fixed-SIR value artifact:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

The parameter vector and order are:

```text
log_kappa_scale, log_nu_scale, log_obs_noise_scale
```

in coordinate system:

```text
sir_log_scale_theta
```

## Entry Conditions Inherited From Previous Phase

- Phase 1 score schema exists and passed review.
- Phase 2 LGSSM is blocked, not admitted.
- Phase 3 must preserve the main fixed-SIR row and must not promote the
  legacy parameterized diagnostic row.
- Directional FD evidence is diagnostic only unless separately justified by an
  all-parameter correctness gate.

## Required Artifacts

Source value artifact:

- `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json`

Current implementation/test artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py`
- `tests/test_ledh_fixed_sir_manual_score_phase4.py`
- `tests/test_ledh_score_memory_n10000.py`
- `bayesfilter/highdim/ledh_score_contract.py`

Expected Phase 3 artifacts:

- score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-score-artifact-2026-07-07.json`
- score summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-score-artifact-2026-07-07.md`
- Phase 3 result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-result-2026-07-07.md`
- Phase 4 predator-prey subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-subplan-2026-07-07.md`
- Phase 3 review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase3-fixed-sir-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

CPU-hidden preflight:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py \
  tests/test_ledh_fixed_sir_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Tiny/no-tape checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_fixed_sir_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Phase 3 schema tests to add or refresh:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py -q
```

Trusted GPU run may reuse or rerun the N=10000 fixed-SIR score-memory test only
as diagnostic input unless it is upgraded to all-parameter correctness:

```text
BAYESFILTER_RUN_LEDHD_SCORE_MEMORY_N10000=1 \
BAYESFILTER_LEDHD_SCORE_MEMORY_BUDGET_MIB=14000 \
MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_score_memory_n10000.py::test_fixed_spatial_sir_ledh_full_row_score_correctness_and_memory_n10000 -q
```

Before admission, the Phase 3 score artifact must pass:

```text
validate_ledh_score_artifact(..., require_admitted=True)
```

against the fixed-SIR value artifact.

Review:

- bounded read-only review of Phase 3 result and Phase 4 predator-prey subplan
  before Phase 4 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the fixed-SIR main row produce an admitted no-tape total derivative of the same finite-`N` LEDH `log_likelihood` scalar as the value artifact? |
| Baseline/comparator | Admitted fixed-SIR value artifact, existing manual total VJP score adapter, tiny same-scalar checks, N=10000 memory/score artifact, and all-parameter correctness if available. |
| Primary criterion | Score artifact validates with `require_admitted=True`; row id is the main fixed-SIR row; parameter order is `[log_kappa_scale, log_nu_scale, log_obs_noise_scale]`; score route is `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`; score is finite; no tape/stopped partial route is used; full-row T=20,N=10000 identity matches value artifact; memory gate passes; and all-parameter correctness is established or the row remains blocked. |
| Veto diagnostics | Parameterized diagnostic row promotion; no-free-theta row promotion; directional FD used as sole admission evidence; wrong target scalar; wrong observation policy; wrong theta coordinate; wrong parameter order; tape/ForwardAccumulator/stopped partial; nonfinite score; memory/device failure. |
| Explanatory diagnostics | Directional FD, score decomposition, runtime, memory, device placement, and old diagnostic comparisons. |
| Not concluded | Exact nonlinear likelihood correctness, Zhao-Cui TT/SIRT source-faithfulness, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or all-algorithm comparison. |
| Artifact | Fixed-SIR score artifact/result or blocker, tests, review bundle, Phase 4 subplan. |

## Step-By-Step Plan

1. Re-read the fixed-SIR value artifact and score adapter.
2. Add or refresh a Phase 3 fixed-SIR score contract test:
   - accepts only the main row id;
   - rejects the legacy parameterized diagnostic row;
   - rejects old no-free-theta semantics;
   - requires T=20,N=10000, batch seeds `81120..81124`;
   - requires target observation policy
     `fixed_sir_infectious_components_gaussian_observation_density`;
   - requires parameter order
     `[log_kappa_scale, log_nu_scale, log_obs_noise_scale]`.
3. Add a schema adapter for fixed-SIR score artifacts if missing.
4. Classify existing N=10000 score-memory evidence:
   - memory and finite score evidence may carry over only if T=20,N=10000 and
     row id match;
   - directional FD is diagnostic only unless an all-parameter correctness
     gate is added and reviewed.
5. If a bounded all-parameter correctness route exists, run it and validate the
   artifact.
6. If only directional FD evidence exists, write a Phase 3 blocker result and
   keep fixed-SIR score not admitted.
7. Draft the Phase 4 predator-prey subplan.
8. Review the Phase 3 result/blocker and Phase 4 subplan.

## Forbidden Claims/Actions

- Do not promote
  `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale`.
- Do not promote old `no_free_theta` semantics.
- Do not use directional FD alone as full score admission evidence.
- Do not use tape, ForwardAccumulator, hidden autodiff, or stopped partials.
- Do not claim Zhao-Cui source-faithfulness for the amended log-scale theta
  score.
- Do not claim exact nonlinear likelihood correctness, HMC readiness, posterior
  correctness, scientific superiority, runtime ranking, or all-algorithm
  comparison.

## Exact Next-Phase Handoff Conditions

Phase 4 predator-prey may start only if:

- Phase 3 writes an admitted fixed-SIR score result or an explicit blocker
  result;
- Phase 4 predator-prey subplan exists;
- review agrees the fixed-SIR decision is boundary-safe and does not promote
  diagnostic/scoped rows.

## Stop Conditions

Stop and write a blocker result if:

- fixed-SIR score cannot be tied to the admitted value artifact;
- all-parameter correctness cannot be established;
- no-tape provenance becomes ambiguous;
- parameterized or no-free-theta evidence is required for admission;
- memory/device gates fail;
- review finds a material issue that does not converge after five rounds.
