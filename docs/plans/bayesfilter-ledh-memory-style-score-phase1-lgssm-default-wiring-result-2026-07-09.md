# LEDH Memory-Style Score Phase 1 Result: LGSSM Default Wiring

Date: 2026-07-09

Status: `PARTIAL_PASS_WITH_REPAIR_BLOCKER`

## Phase Objective

Route the LGSSM score mode through the memory-style reverse/VJP recurrence by
default, while preserving the same realized finite-`N` LEDH
`observed_data_log_likelihood_estimator` / `log_likelihood` scalar and keeping
the old compact forward-sensitivity route as historical/diagnostic only.

## Entry Conditions

- Root-cause plan:
  `docs/plans/bayesfilter-ledh-memory-style-score-root-cause-and-repair-plan-2026-07-09.md`
- Prior shared compact-score contraction repair had already removed several
  avoidable 5D JVP temporaries but still left old forward-sensitivity route
  artifacts and tests in place.
- Claude review was not available for this phase because external Claude
  review execution was policy-rejected; a fresh local Codex review was used as
  a read-only substitute and returned `VERDICT: AGREE` for the root-cause plan.

## Work Completed

- `bayesfilter/highdim/ledh_score_contract.py`
  - Added memory-style LGSSM provenance:
    `memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`.
  - Full admission now requires the memory-style no-tape provenance rather than
    the old compact forward-sensitivity provenance.
  - Old compact forward-sensitivity route strings remain historical/tiny-only.
- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
  - Repointed default `compact-sensitivity` score mode to
    `_manual_value_and_score_from_components`.
  - Kept `_compact_value_and_score_from_components` as historical diagnostic.
  - Emitted `score_execution_style =
    memory_style_reverse_vjp_no_particle_param_axis`.
- Focused contract tests were updated to expect memory-style provenance.

## Checks Run

CPU-hidden focused checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py -q
```

Result: `57 passed, 2 warnings`.

CPU-hidden broader score/contract suite:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py -q
```

Result: `117 passed, 2 warnings`.

Trusted GPU score-only smoke:

- `N=256,T=3`, score-only, one seed, emitted:
  `docs/plans/artifacts/ledh-memory-style-score-phase1-lgssm-score-only-n256-t3-2026-07-09.json`
- The artifact reports:
  - `score_derivative_provenance =
    memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`
  - `score_execution_style =
    memory_style_reverse_vjp_no_particle_param_axis`
  - `score_admission_status =
    blocked_score_diagnostic_stage_not_admitted`

## New Blocker

Trusted GPU `N=1000,T=10` score-only using the memory-style route did not emit
an artifact after roughly 4.5 minutes and was interrupted. GPU memory had
climbed to about `15760 MiB / 16376 MiB`, above the reviewed `14000 MiB`
score-memory budget.

The interrupted stack was inside the reverse transport pullback:

- `benchmark_ledh_same_target_lgssm_m3_t50_value.py::_manual_value_and_score_from_components`
- `_manual_transport_vjp_tf`
- `annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_pullback`
- `_filterflow_streaming_finite_sinkhorn_potentials_vjp_total`
- `_filterflow_streaming_softmin_vjp`
- `_scatter_axis1_add_3d`

This means the Phase 1 default wiring is correct, but the reverse/VJP route is
not yet memory-sufficient. It avoids full `[batch,N,state_dim,param_dim]`
particle tangents, but the transport pullback still performs repeated
full-sized scatter-add updates and the LGSSM reverse pass stores full per-time
flow/state records.

## Decision Table

| Field | Decision |
| --- | --- |
| Primary criterion status | Partial pass: default score route now uses memory-style reverse/VJP and tiny GPU artifact confirms provenance. |
| Veto diagnostic status | Veto for full-score scale remains: `N=1000,T=10` memory-style score-only exceeded memory budget/no artifact. |
| Main uncertainty | Whether removing transport-pullback scatter/full-accumulator lifetime is sufficient before also checkpointing LGSSM flow history. |
| Next justified action | Phase 1R: repair transport VJP memory lifetime and rerun CPU correctness plus GPU rungs. |
| Not concluded | No full LGSSM score admission, no `N=10000,T=50` score readiness, no HMC readiness, no posterior correctness. |

## Next-Phase Handoff

Proceed to:

`docs/plans/bayesfilter-ledh-memory-style-score-phase1r-transport-vjp-memory-repair-subplan-2026-07-09.md`

Phase 1R inherits:

- same scalar target;
- memory-style reverse/VJP route as the default LGSSM score route;
- old compact forward-sensitivity route remains historical/tiny-only;
- score-only rungs are diagnostic and cannot admit a leaderboard row.
