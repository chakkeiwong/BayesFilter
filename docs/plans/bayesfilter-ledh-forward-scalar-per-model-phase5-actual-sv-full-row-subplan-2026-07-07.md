# Phase 5 Subplan: Actual-SV Full-Row Forward Scalar Admission

metadata_date: 2026-07-07
status: `DRAFT_PRE_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
parent_phase: 5

## Phase Objective

Run and admit, or explicitly block, the full actual-SV LEDH observed-data
forward scalar row using the exact transformed adapter smoke route that already
passed:

```text
zhao_cui_sv_actual_nongaussian_T1000
```

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase is forward-scalar-only. It must not implement, run, or admit scores.

## Entry Conditions Inherited From Previous Phase

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally.
- Phase 3 fixed SIR artifact validates locally.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 exact transformed actual-SV tiny adapter smoke passed with:
  - artifact:
    `docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`;
  - `admission_status=tiny_executed_not_full_row`;
  - exact `log(y^2)` transform with offset `0.0`;
  - exact `exact_log_chi_square_log_density` target correction;
  - no raw Gaussian, KSC, or augmented-noise target substitution;
  - implementation/result review `VERDICT: AGREE`.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-subplan-2026-07-07.md`
- Patched runner with explicit full-row admission mode:
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- Full-row canonical JSON artifact:
  `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- Full-row markdown summary:
  `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md`
- Full-row replay test:
  `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py`
- Phase 5 full-row result or blocker result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-result-2026-07-07.md`
- Phase 6 generalized-SV subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase6-generalized-sv-subplan-2026-07-07.md`
- Plan review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-plan-review-bundle-2026-07-07.md`
- Result review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-result-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Before execution:

- bounded read-only review of this subplan.

After implementation:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py
```

Full trusted GPU run:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  --run-scope full-row-admission \
  --time-steps 1000 \
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
  --output docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md
```

Replay and guard checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py -q
```

Run the existing through-Phase-4 replay set plus Phase 5 after artifact
creation:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py -q
```

Before Phase 6:

- bounded read-only review of the full-row result and Phase 6 subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the exact transformed actual-SV LEDH adapter produce a full-row executable same-target observed-data likelihood artifact? |
| Baseline/comparator | Phase 5 tiny adapter smoke result, `make_actual_sv_forward_contract(...)`, `StochasticVolatilitySSM`, `exact_transformed_sv_observations`, `exact_log_chi_square_log_density`, and Phase 1 schema validator. |
| Primary criterion | Full-row JSON artifact validates with `validate_ledh_forward_scalar_artifact(..., expected_row_id=ACTUAL_SV_ROW_ID, require_admitted=True)`, has finite `log_likelihood_by_seed`, row id `zhao_cui_sv_actual_nongaussian_T1000`, theta coordinate `synthetic_unconstrained`, theta values `[0.2533471031357997,-0.916290731874155]`, `T=1000`, `N=10000`, seeds `[81120,81121,81122,81123,81124]`, `target_observation_policy=transformed_actual_sv_log_y_square`, `target_density_used_for_correction=true`, and GPU output device. |
| Veto diagnostics | Tiny artifact admitted as full row; full run uses raw Gaussian observation likelihood as target correction; KSC finite mixture used as actual-SV evidence; augmented-noise Gaussian closure used; transform offset other than `0.0`; target density not used for correction; wrong theta/seeds/T/N; nonfinite output; missing replay test; score fields used as value evidence; runtime or memory promoted as correctness. |
| Explanatory diagnostics | Runtime, compile time, memory, ESS placeholders in value-only mode, warm-call timing, and Monte Carlo variability across seeds. |
| Not concluded | No score admission, score correctness, generalized-SV admission, KSC admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |
| Artifact | Full-row JSON/markdown artifact, replay test, Phase 5 result, Phase 6 subplan. |

## Step-By-Step Plan

1. Patch the actual-SV runner to add an explicit `--run-scope` argument:
   - default `tiny-smoke`;
   - optional `full-row-admission`.
2. Keep tiny-smoke behavior unchanged:
   - full-row settings rejected unless `--run-scope full-row-admission`;
   - status remains `tiny_executed_not_full_row`;
   - `full_leaderboard_row=false`.
3. In `full-row-admission` mode, require exact full-row settings before
   admission:
   - seeds `[81120,81121,81122,81123,81124]`;
   - `T=1000`;
   - `N=10000`;
   - exact theta and target policy;
   - finite output.
4. Emit `n10000_same_target_value_admitted` only when all full-row checks pass.
5. Add a full-row replay test that reads the full JSON artifact and validates
   with `require_admitted=True`.
6. Run compile checks.
7. Run the trusted GPU full-row command with `history-mode value-only`.
8. Run focused replay/guard checks and the through-Phase-5 replay set.
9. Write the Phase 5 full-row result or blocker result.
10. Draft or refresh Phase 6 generalized-SV subplan.
11. Send result and Phase 6 subplan for bounded read-only review.

## Forbidden Claims/Actions

- Do not use raw Gaussian observation likelihood as actual-SV target
  correction.
- Do not use KSC finite-mixture likelihood as actual-SV target evidence.
- Do not use augmented-noise Gaussian closure as same-target evidence.
- Do not use a positive log-square transform offset.
- Do not admit unless `T=1000`, `N=10000`, and seeds
  `[81120,81121,81122,81123,81124]` are used.
- Do not implement or admit scores.
- Do not rebuild the leaderboard.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  or runtime ranking.

## Exact Next-Phase Handoff Conditions

Phase 6 may begin only if:

- the full actual-SV artifact validates with `require_admitted=True`, or a
  blocker result explains why it cannot yet be produced;
- replay tests read the actual full-row JSON artifact from disk;
- the tiny replay still confirms the tiny artifact is not admitted;
- Phase 5 result records the exact target bridge and full-row status;
- local checks pass, or a blocker result explains why they cannot apply;
- Phase 6 generalized-SV subplan is drafted or refreshed;
- bounded read-only review agrees.

## Stop Conditions

Stop and write a blocker result if:

- the runner cannot safely separate tiny smoke from full-row admission;
- the full artifact is nonfinite or fails schema validation;
- output is not on the requested GPU;
- theta, seeds, `T`, or `N` differ from the frozen full-row contract;
- the target correction becomes raw Gaussian, KSC, augmented-noise, or offset
  transformed;
- the run fails from OOM or runtime failure after the implemented route is
  otherwise correct;
- score work is required;
- a human approval boundary is reached.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | The full run must use the exact transformed adapter that passed tiny smoke. |
| Proxy metrics | Runtime and memory cannot admit the row without schema validation and replay. |
| Missing stop condition | Stops cover wrong target, nonfinite output, wrong row settings, OOM/runtime failure, score creep, and boundary crossing. |
| Hidden assumption | Full-row admission is behind explicit `--run-scope full-row-admission`. |
| Stale context | Entry conditions cite the actual tiny artifact and review result. |
| Environment mismatch | Full-row command is trusted GPU/XLA; CPU-only checks declare device hiding. |
| Useless artifact | Required artifact must validate with `require_admitted=True`; stdout alone cannot pass. |

Audit status: passed for pre-review. Execution may start only after bounded
read-only review agrees.
