# Phase 6 Subplan: Generalized-SV Forward Scalar Build

metadata_date: 2026-07-07
status: `DRAFT_AFTER_PHASE5_LOCAL_PASS_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 6

## Phase Objective

Build, admit, or explicitly block the LEDH observed-data forward scalar row:

```text
zhao_cui_generalized_sv_synthetic_from_estimated_values
```

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase is forward-scalar-only. It must not implement, run, or admit scores.

## Entry Conditions Inherited From Previous Phase

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV target evidence is model-specific and cannot admit
  generalized-SV.
- Phase 5 result and this Phase 6 subplan must pass read-only review before
  execution.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase6-generalized-sv-subplan-2026-07-07.md`
- Generalized-SV runner or blocker result. If implemented, expected path:
  `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py`
- Tiny-smoke JSON artifact:
  `docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`
- Tiny-smoke markdown summary:
  `docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md`
- Full-row canonical JSON artifact, only if tiny smoke passes:
  `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`
- Full-row markdown summary, only if full row runs:
  `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md`
- Tiny replay test, if tiny artifact is produced:
  `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py`
- Full-row replay test, if full artifact is produced:
  `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py`
- Phase 6 result or blocker result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase6-generalized-sv-result-2026-07-07.md`
- Phase 7 KSC-SV subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-subplan-2026-07-07.md`
- Phase 6 review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase6-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Before execution:

- bounded read-only review of Phase 5 result and this subplan.

If a runner is implemented, run:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py
```

Tiny smoke must run before any full-row run. Proposed tiny command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  --run-scope tiny-smoke \
  --time-steps 4 \
  --num-particles 128 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --history-mode full \
  --warmups 0 \
  --repeats 1 \
  --device /GPU:0 \
  --expect-device-kind gpu \
  --output docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md
```

Tiny replay and guard check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py -q
```

Full trusted GPU run may start only after tiny smoke passes, the full-row
settings are explicitly guarded, and the runner has a full-row replay test.
Proposed full-row command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  --run-scope full-row-admission \
  --time-steps 1008 \
  --num-particles 10000 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --row-chunk-size 512 \
  --col-chunk-size 512 \
  --particle-chunk-size 512 \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --device /GPU:0 \
  --expect-device-kind gpu \
  --output docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md
```

After artifact creation, run the through-Phase-6 replay set:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py -q
```

If Phase 6 blocks before a runner or full-row artifact exists, replace
nonexistent test modules in the check list with the exact blocker-replay or
contract tests written in this phase.

Before Phase 7:

- bounded read-only review of the Phase 6 result and Phase 7 KSC-SV subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the source-route prior-mean generalized-SV LEDH adapter produce an executable same-target observed-data likelihood artifact? |
| Baseline/comparator | `make_generalized_sv_forward_contract(...)`, `scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py::_generalized_sv_prior_mean_dataset`, `scripts/filtering_value_gradient_benchmark_run_p8d_numeric.py::_dpf_generalized_sv_callbacks`, `docs/plans/bayesfilter-filtering-value-gradient-benchmark-generalized-sv-testing-spec-2026-06-11.md`, and Phase 1 schema validator. |
| Primary criterion | Full-row JSON artifact validates with `validate_ledh_forward_scalar_artifact(..., expected_row_id=GENERALIZED_SV_ROW_ID, require_admitted=True)`, has finite `log_likelihood_by_seed`, row id `zhao_cui_generalized_sv_synthetic_from_estimated_values`, theta coordinate `source_route_active_transformed_prior_mean`, theta values `[1.0824113944610982,-2.076793740349318,0.0]`, `T=1008`, `N=10000`, seeds `[81120,81121,81122,81123,81124]`, `target_observation_policy=source_route_prior_mean_generalized_sv`, `target_density_used_for_correction=true`, raw zero-mean generalized-SV target observation density, and GPU output device. |
| Veto diagnostics | Actual-SV evidence borrowed; KSC evidence borrowed; native generalized-SV dense fixture substituted; SP500 returns used as benchmark observations; author defaults used as truth; log-square Gaussianized flow observation treated as target likelihood; raw target density missing; wrong theta/seeds/T/N; nonfinite output; missing replay test; score fields used as value evidence; runtime or memory promoted as correctness. |
| Explanatory diagnostics | Runtime, compile time, memory, tiny smoke artifacts, ESS, Monte Carlo variability, and prior dense/native references. |
| Not concluded | No score admission, score correctness, KSC admission, actual-SV admission, HMC readiness, posterior correctness, Zhao-Cui source-faithfulness, scientific superiority, or runtime ranking. |
| Artifact | Tiny artifact, optional full-row artifact, replay tests, Phase 6 result/blocker result, Phase 7 KSC subplan. |

## Step-By-Step Plan

1. Inventory generalized-SV route and source contracts:
   - row id `zhao_cui_generalized_sv_synthetic_from_estimated_values`;
   - dataset seed `81105`;
   - horizon `1008`;
   - theta coordinate `source_route_active_transformed_prior_mean`;
   - theta values `[1.0824113944610982,-2.076793740349318,0.0]`;
   - target observation policy `source_route_prior_mean_generalized_sv`;
   - target density
     `raw_zero_mean_generalized_sv_prior_mean_normal_log_density`;
   - flow observation policy `log_square_gaussian_surrogate_for_ledh_flow_only`.
2. Decide whether existing actual-SV runner can be adapted by code reuse without
   target reuse:
   - allowed: copy structural loop/transport mechanics and replace model
     equations;
   - forbidden: reuse actual-SV target density, actual-SV row id, or
     actual-SV artifact evidence.
3. Implement the smallest generalized-SV runner if feasible:
   - explicit `--run-scope tiny-smoke|full-row-admission`;
   - exact full-row guard requiring `T=1008`, `N=10000`, and seeds
     `[81120,81121,81122,81123,81124]`;
  - stationary previous-state sampling for the source-route AR(1);
  - transition density `Normal(mu + gamma * (x_{t-1} - mu), 1)` for each
    recorded observation, including `t=0`, matching
    `_generalized_sv_prior_mean_dataset(...)` and the existing DPF callback
    convention;
   - raw observation density using log scale `0.5 * tau * x_t`;
   - LEDH proposal surface based on log-square observations with declared
     positive offset only if inherited from the existing callback route and
     clearly marked proposal-only.
4. Add tiny replay test:
   - validate tiny artifact with `require_admitted=False`;
   - assert `require_admitted=True` rejects it;
   - assert target policy, flow policy, theta, and target-density flags.
5. Run compile checks.
6. Run trusted GPU tiny smoke.
7. If tiny smoke fails from implementation bug, patch and rerun focused checks.
8. If tiny smoke fails from target ambiguity, numerical nonfiniteness, or OOM
   that cannot be repaired within this phase, write blocker result and stop.
9. If tiny smoke passes, add full-row replay test and run the trusted GPU
   full-row command.
10. Validate the canonical full-row artifact with `require_admitted=True`.
11. Run through-Phase-6 replay checks.
12. Write Phase 6 result or blocker result.
13. Draft or refresh Phase 7 KSC-SV subplan.
14. Send Phase 6 result and Phase 7 subplan for bounded read-only review.

## Forbidden Claims/Actions

- Do not use actual-SV target evidence for generalized-SV admission.
- Do not use KSC finite-mixture likelihood as generalized-SV target evidence.
- Do not use the native generalized-SV dense fixture as a substitute for the
  source-route prior-mean row.
- Do not use SP500 returns as the benchmark observations.
- Do not use author-code defaults as the truth vector.
- Do not treat the log-square Gaussianized observation as the target
  likelihood; it may be proposal-only.
- Do not admit unless `T=1008`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, theta, target policy, and target density
  match the frozen contract.
- Do not implement or admit scores.
- Do not rebuild the leaderboard.
- Do not claim Zhao-Cui source-faithfulness, HMC readiness, posterior
  correctness, scientific superiority, or runtime ranking.

## Exact Next-Phase Handoff Conditions

Phase 7 KSC-SV may begin only if:

- the full generalized-SV artifact validates with `require_admitted=True`, or a
  blocker result explains why it cannot yet be produced;
- replay tests read any produced generalized-SV JSON artifacts from disk;
- the tiny replay confirms any tiny artifact is not admitted;
- Phase 6 result records the target bridge, source route, full-row status, and
  nonclaims;
- local checks pass, or the blocker result explains why they cannot apply;
- Phase 7 KSC-SV subplan is drafted or refreshed;
- bounded read-only review agrees.

## Stop Conditions

Stop and write a blocker result if:

- the source-route generalized-SV target cannot be expressed without changing
  the row target;
- the implementation would require score work;
- the flow surface and target likelihood cannot be kept distinct;
- the runner cannot safely separate tiny smoke from full-row admission;
- the full artifact is nonfinite or fails schema validation;
- output is not on the requested GPU for trusted full-row execution;
- theta, seeds, `T`, or `N` differ from the frozen full-row contract;
- actual-SV, KSC, native-generalized-SV fixture, SP500 returns, or author
  defaults are needed as substitutes;
- the run fails from OOM or runtime failure after the implemented route is
  otherwise correct;
- a human approval boundary is reached.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | The baseline is the generalized-SV source-route prior-mean contract, not actual-SV or KSC. |
| Proxy metrics | Runtime, memory, and finite output cannot admit the row without schema validation and replay. |
| Missing stop condition | Stops cover target ambiguity, actual-SV/KSC/native substitution, wrong row settings, nonfinite output, OOM/runtime failure, score creep, and boundary crossing. |
| Hidden assumption | Full-row admission is behind explicit `--run-scope full-row-admission`. |
| Stale context | Entry conditions cite Phase 5 actual-SV as complete but non-transferable. |
| Environment mismatch | Full-row command is trusted GPU/XLA; CPU-only checks declare device hiding. |
| Useless artifact | Required artifact must validate with `require_admitted=True`; stdout alone cannot pass. |

Audit status: passed for draft. Execution may start only after bounded read-only
review of the Phase 5 result and this subplan agrees.
