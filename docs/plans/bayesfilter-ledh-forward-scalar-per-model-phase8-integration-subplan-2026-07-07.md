# Phase 8 Subplan: Value-Only Integration From Admitted Forward Scalars

metadata_date: 2026-07-07
status: `DRAFT_AFTER_PHASE7_LOCAL_PASS_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 8

## Phase Objective

Build a value-only LEDH integration artifact from admitted same-target
forward-scalar artifacts for the main observed-data model rows.

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase is forward-scalar-only. It must not implement, run, import, admit,
or merge scores.

## Entry Conditions Inherited From Previous Phase

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally with `require_admitted=True`.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV artifact validates locally with `require_admitted=True`.
- Phase 6 generalized-SV artifact validates locally with `require_admitted=True`.
- Phase 7 KSC-SV artifact validates locally with `require_admitted=True`.
- Phase 7 result and this Phase 8 subplan must pass read-only review before
  execution.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-subplan-2026-07-07.md`
- Value-only integration builder:
  `docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py`
- Value-only JSON artifact:
  `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- Value-only markdown artifact:
  `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md`
- Integration replay test:
  `tests/highdim/test_ledh_phase8_value_integration_artifact.py`
- Phase 8 result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-result-2026-07-07.md`
- Phase 8 result review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase8-review-bundle-2026-07-07.md`

## Required Input Artifacts

The builder must read these admitted forward-scalar JSON artifacts:

- `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`

The builder must also record the legacy diagnostic status of:

```text
zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale
```

as excluded from the main value leaderboard unless a separate admitted artifact
exists for that exact diagnostic row. This phase must not promote the
diagnostic row to a main observed-data row.

## Required Checks/Tests/Reviews

Before execution:

- bounded read-only review of the Phase 7 result and this subplan.

Implementation checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py
```

Run builder:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  --output docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md
```

Focused replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Through-Phase-8 replay:

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
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Diff hygiene:

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json \
  docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md \
  docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-result-2026-07-07.md
```

Before final close:

- bounded read-only review of the Phase 8 result, integration artifact, and
  replay test.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the six main LEDH high-dimensional model rows be assembled into one value-only integration artifact using only admitted same-target forward-scalar artifacts? |
| Baseline/comparator | Phase 1 schema validator; Phase 2-7 admitted artifacts listed above; old score-inclusive leaderboard builders only as negative examples of what not to reuse unchanged. |
| Primary criterion | The JSON artifact has exactly six main LEDH value rows, each row validates with `validate_ledh_forward_scalar_artifact(..., require_admitted=True)`, each row has finite `log_likelihood_by_seed`, target scalar `observed_data_log_likelihood_estimator`, output field `log_likelihood`, score fields absent or explicitly blocked, runtime cross-ranking disabled, and the parameterized SIR diagnostic row excluded from main rows. |
| Veto diagnostics | Missing admitted artifact; stale artifact path; row id mismatch; target policy mismatch; nonfinite `log_likelihood`; tiny artifact admitted; score fields merged; score-inclusive builder reused unchanged; old score artifacts read; runtime ranking enabled; diagnostic SIR row promoted to main row; KSC exact native actual-SV claim; value result inferred from metadata, callback existence, runtime, memory, or finite output without schema replay. |
| Explanatory diagnostics | Mean log likelihood, mean per-time-step log likelihood, Monte Carlo standard error across seeds, compile/warm timing copied from source artifacts, GPU output device copied from source artifacts, and source artifact paths. |
| Not concluded | No score admission, score correctness, HMC readiness, posterior correctness, scientific superiority, fair runtime ranking, or all-algorithm comparison. |
| Artifact | Value-only JSON/markdown integration artifacts, replay test, Phase 8 result, review bundle, and ledger entry. |

## Step-By-Step Plan

1. Inventory all six required source artifacts and map each to its expected row
   id:
   - LGSSM: `benchmark_lgssm_exact_oracle_m3_T50`;
   - fixed SIR: `zhao_cui_spatial_sir_austria_j9_T20`;
   - predator-prey: `zhao_cui_predator_prey_T20`;
   - actual-SV: `zhao_cui_sv_actual_nongaussian_T1000`;
   - generalized-SV:
     `zhao_cui_generalized_sv_synthetic_from_estimated_values`;
   - KSC-SV: `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`.
2. Implement the smallest value-only builder:
   - read each source artifact;
   - validate each with Phase 1 schema and `require_admitted=True`;
   - normalize row summaries from the validator output;
   - compute mean, sample standard deviation, and Monte Carlo standard error
     for total and average log likelihoods;
   - copy timing/device diagnostics only as explanatory fields;
   - copy source nonclaims and add integration nonclaims;
   - set `runtime_cross_ranking_allowed=false`;
   - set `score_integration_status=blocked_out_of_scope_forward_scalar_only`;
   - record the parameterized SIR diagnostic row as excluded.
3. Add a replay test that reads the integration JSON from disk and checks:
   - schema version and runbook phase;
   - exactly six main rows;
   - expected row ids and source artifact paths;
   - every row is admitted and finite;
   - no row has score values or admitted score status;
   - runtime ranking is disabled;
   - parameterized SIR diagnostic row is not in `rows`;
   - KSC row states finite-mixture surrogate target and does not claim exact
     native actual-SV likelihood.
4. Run compile checks.
5. Run the builder.
6. Run focused replay.
7. Run through-Phase-8 replay.
8. Run diff hygiene.
9. Write Phase 8 result with a decision table, evidence contract result, exact
   artifact paths, checks, and nonclaims.
10. Send Phase 8 result, artifact summary, and replay test for bounded
    read-only review.
11. If review returns `VERDICT: REVISE`, patch the same phase visibly and rerun
    focused checks. Stop after five review rounds for the same blocker.

## Forbidden Claims/Actions

- Do not implement, run, import, admit, or merge score routes.
- Do not read old score artifacts as Phase 8 evidence.
- Do not reuse `benchmark_two_lane_highdim_ledh_inclusive_results.py`
  unchanged, because it is score-aware and expects score admission for LGSSM
  and SIR.
- Do not rank LEDH runtime against frozen non-LEDH rows.
- Do not claim all-algorithm leaderboard readiness.
- Do not promote the parameterized SIR diagnostic row to a main row.
- Do not claim KSC exact native actual-SV likelihood.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  Zhao-Cui source-faithfulness, or runtime ranking.
- Do not use runtime, memory, finite output, metadata, or callback existence as
  a substitute for schema replay of admitted artifacts.

## Exact Next-Phase Handoff Conditions

This is the last phase of the forward-scalar value-only runbook. It may close
as passed only if:

- the integration JSON and markdown artifacts are written;
- every main row in the integration JSON comes from an admitted Phase 2-7
  forward-scalar artifact;
- replay and through-Phase-8 checks pass;
- read-only review agrees; and
- the final result explicitly states score and all-algorithm comparison work
  remain out of scope.

If any condition fails, write a Phase 8 blocker result and stop for repair or
human direction.

## Stop Conditions

Stop and write a blocker result if any of the following occurs:

- any required source artifact is missing;
- any source artifact fails Phase 1 validation with `require_admitted=True`;
- any row id, target policy, theta, `T`, `N`, or seed contract mismatches its
  source artifact;
- any source artifact has nonfinite `log_likelihood_by_seed`;
- the only available integration path requires score artifacts;
- the builder would need to promote the parameterized SIR diagnostic row as a
  main row;
- the builder would need to rank runtimes against frozen non-LEDH baselines;
- local replay fails after artifact generation;
- a human approval boundary is reached;
- Claude/Codex review finds a material issue that does not converge after at
  most five repair rounds for the same blocker.
