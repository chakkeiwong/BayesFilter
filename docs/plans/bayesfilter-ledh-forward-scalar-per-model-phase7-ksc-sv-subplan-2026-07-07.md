# Phase 7 Subplan: KSC-SV Forward Scalar Build

metadata_date: 2026-07-07
status: `DRAFT_AFTER_PHASE6_LOCAL_PASS_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 7

## Phase Objective

Build, admit, or explicitly block the LEDH observed-data forward scalar row:

```text
zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000
```

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

The target observation policy is:

```text
ksc_log_chi_square_gaussian_mixture_surrogate
```

This phase is forward-scalar-only. It must not implement, run, or admit scores.

## Entry Conditions Inherited From Previous Phase

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV artifact validates locally with `require_admitted=True`.
- Phase 6 generalized-SV artifact validates locally with `require_admitted=True`.
- Actual-SV and generalized-SV target evidence is model-specific and cannot
  admit KSC-SV.
- Phase 6 result and this Phase 7 subplan must pass read-only review before
  execution.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-subplan-2026-07-07.md`
- KSC-SV runner or blocker result. If implemented, expected path:
  `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py`
- Tiny-smoke JSON artifact:
  `docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`
- Tiny-smoke markdown summary:
  `docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md`
- Full-row canonical JSON artifact, only if tiny smoke passes:
  `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`
- Full-row markdown summary, only if full row runs:
  `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.md`
- Tiny replay test, if tiny artifact is produced:
  `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py`
- Full-row replay test, if full artifact is produced:
  `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py`
- Phase 7 result or blocker result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-result-2026-07-07.md`
- Phase 8 value-integration subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-subplan-2026-07-07.md`
- Phase 7 review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase7-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Before execution:

- bounded read-only review of Phase 6 result and this subplan.

If a runner is implemented, run:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py
```

Tiny smoke must run before any full-row run. Proposed tiny command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
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
  --output docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md
```

Tiny replay and guard check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py -q
```

Full trusted GPU run may start only after tiny smoke passes, the full-row
settings are explicitly guarded, and the runner has a full-row replay test.
Proposed full-row command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
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
  --output docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.md
```

After artifact creation, run the through-Phase-7 replay set:

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
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py -q
```

If Phase 7 blocks before a runner or full-row artifact exists, replace
nonexistent test modules in the check list with the exact blocker-replay or
contract tests written in this phase.

Before Phase 8:

- bounded read-only review of the Phase 7 result and Phase 8 integration
  subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the declared KSC finite Gaussian-mixture surrogate SV row produce an executable same-target observed-data likelihood artifact under LEDH? |
| Baseline/comparator | `make_ksc_sv_forward_contract(...)`, `KSCMixtureTransformedSVSSM`, `ksc_1998_log_chi_square_mixture()`, `transformed_sv_observations(..., offset=1e-8)`, the KSC dataset manifest row derived from actual-SV observations with transform offset `1e-8`, and Phase 1 schema validator. |
| Primary criterion | Full-row JSON artifact validates with `validate_ledh_forward_scalar_artifact(..., expected_row_id=KSC_SV_ROW_ID, require_admitted=True)`, has finite `log_likelihood_by_seed`, row id `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`, theta coordinate `synthetic_unconstrained`, theta values `[0.2533471031357997,-0.916290731874155]`, `T=1000`, `N=10000`, seeds `[81120,81121,81122,81123,81124]`, `target_observation_policy=ksc_log_chi_square_gaussian_mixture_surrogate`, target density correction from the finite KSC Gaussian mixture, transform offset `1e-8`, and GPU output device. |
| Veto diagnostics | Actual-SV exact log-chi-square target used; generalized-SV raw target used; raw Gaussian SV callback used as target; KSC mixture used as proposal-only rather than target correction; transform offset missing or changed without reviewed approval; wrong theta/seeds/T/N; nonfinite output; missing replay test; score fields used as value evidence; runtime or memory promoted as correctness. |
| Explanatory diagnostics | Runtime, compile time, memory, tiny smoke artifacts, ESS, Monte Carlo variability, and Kalman/mixture-enumeration references. |
| Not concluded | No score admission, score correctness, exact native actual-SV likelihood, generalized-SV admission, HMC readiness, posterior correctness, Zhao-Cui source-faithfulness, scientific superiority, or runtime ranking. |
| Artifact | Tiny artifact, optional full-row artifact, replay tests, Phase 7 result/blocker result, Phase 8 integration subplan. |

## Step-By-Step Plan

1. Inventory the KSC route and source contracts:
   - row id `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`;
   - dataset source `zhao_cui_sv_actual_nongaussian_T1000` with transform
     `log(y_t^2 + offset)`;
   - transform offset `1e-8`;
   - horizon `1000`;
   - theta coordinate `synthetic_unconstrained`;
   - theta values `[0.2533471031357997,-0.916290731874155]`;
   - target observation policy `ksc_log_chi_square_gaussian_mixture_surrogate`;
   - target density `finite_ksc_log_chi_square_gaussian_mixture_log_density`.
2. Decide whether the actual-SV runner can be adapted by code reuse without
   target reuse:
   - allowed: copy structural loop/transport mechanics and replace model
     equations;
   - forbidden: reuse exact log-chi-square target density, exact actual-SV
     artifact evidence, or actual-SV nonclaims as KSC evidence.
3. Implement the smallest KSC runner if feasible:
   - explicit `--run-scope tiny-smoke|full-row-admission`;
   - exact full-row guard requiring `T=1000`, `N=10000`, and seeds
     `[81120,81121,81122,81123,81124]`;
   - raw observations from `_sv_dataset(81101)` transformed as
     `log(y_t^2 + 1e-8)`;
   - stationary initial state matching the SV latent model;
   - transition density matching the synthetic SV AR(1);
   - observation density as `logsumexp_k log w_k + log Normal(z_t; log(beta^2)+x_t+m_k, sqrt(v_k))`;
   - LEDH proposal surface may use the same transformed scalar with a declared
     Gaussianized/moment-matched flow variance, but target correction must use
     the finite mixture density.
4. Add tiny replay test:
   - validate tiny artifact with `require_admitted=False`;
   - assert `require_admitted=True` rejects it;
   - assert target policy, mixture component count, transform offset, theta, and
     target-density flags.
5. Run compile checks.
6. Run trusted GPU tiny smoke.
7. If tiny smoke fails from implementation bug, patch and rerun focused checks.
8. If tiny smoke fails from target ambiguity, numerical nonfiniteness, or OOM
   that cannot be repaired within this phase, write blocker result and stop.
9. If tiny smoke passes, add full-row replay test and run the trusted GPU
   full-row command.
10. Validate the canonical full-row artifact with `require_admitted=True`.
11. Run through-Phase-7 replay checks.
12. Write Phase 7 result or blocker result.
13. Draft or refresh Phase 8 integration subplan.
14. Send Phase 7 result and Phase 8 subplan for bounded read-only review.

## Forbidden Claims/Actions

- Do not use exact actual-SV target evidence for KSC admission.
- Do not use generalized-SV target evidence for KSC admission.
- Do not use a raw Gaussian SV callback as the KSC target likelihood.
- Do not treat the KSC finite mixture as exact native actual-SV likelihood.
- Do not change the transform offset from `1e-8` without a reviewed plan and
  human approval.
- Do not admit unless `T=1000`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, theta
  `[0.2533471031357997,-0.916290731874155]`, and target policy all match.
- Do not admit a tiny-smoke artifact.
- Do not implement, run, or admit score routes in this phase.
- Do not rebuild the leaderboard in this phase.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  exact actual-SV likelihood, or runtime ranking.
- Do not use runtime, memory, or finite output as a substitute for schema and
  target-identity replay.

## Exact Next-Phase Handoff Conditions

Phase 8 integration may start only if either:

- Phase 7 writes a result showing the KSC-SV full-row artifact validates with
  `require_admitted=True`, through-Phase-7 replay checks pass, and read-only
  review agrees; or
- Phase 7 writes a blocker result that precisely states why KSC remains
  blocked, through-Phase-7 blocker checks pass, and read-only review agrees the
  integration phase must exclude the blocked row or mark it blocked.

In both cases, Phase 8 must consume only admitted forward-scalar artifacts or
explicit blocker records. It must not infer KSC admission from actual-SV,
generalized-SV, metadata, callback existence, runtime, memory, or finite output.

## Stop Conditions

Stop and write a blocker result if any of the following occurs:

- KSC target density cannot be stated as the finite Gaussian mixture in the
  current code path.
- The runner uses exact actual-SV log-chi-square density as the KSC target.
- The runner uses generalized-SV raw density as the KSC target.
- The runner uses a raw Gaussian callback as the KSC target.
- The transform offset is absent, ambiguous, or differs from `1e-8`.
- The full-row guard cannot enforce exact `T=1000`, `N=10000`, seeds, theta, and
  target policy.
- Tiny smoke is nonfinite, OOMs, or reveals target ambiguity that is not fixable
  inside this phase.
- Full-row run OOMs or exceeds available runtime and no smaller admitted scope
  is permitted by the evidence contract.
- A replay test fails after artifact generation.
- A score route is needed to proceed.
- A human approval boundary is reached.
- Claude/Codex review finds a material issue that does not converge after at
  most five repair rounds for the same blocker.
