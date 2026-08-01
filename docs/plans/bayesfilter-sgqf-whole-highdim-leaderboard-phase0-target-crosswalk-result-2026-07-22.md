# SGQF Whole High-Dimensional Leaderboard Phase 0 Target Crosswalk Result

Date: 2026-07-22

Status: `PASS_SCHEMA_RESET; SOURCE_DATA_RESET_REQUIRED; NUMERICAL_PROMOTION_BLOCKED`

Governing plan:
`docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-repair-master-program-2026-07-22.md`

## Decision

The live seven-row inventory is exhaustive and the SGQF execution lane is
separate from the concurrent GenUT and Zhao--Cui APF worktree edits. The
leaderboard readiness schema may be repaired now. Source-named numerical rows
must not be promoted from the old artifacts yet: the current actual-SV,
fixed-SIR, and predator-prey fixtures observe the initial state and perform
only `T-1` transitions, whereas the checked Zhao--Cui executable source performs
`T` transition-then-observe steps.

This is a target mismatch, not a numerical failure. Old results are preserved
as evidence for explicit BayesFilter initial-observation-first amended targets.
They are not evidence for the reset source-scope rows.

## Frozen Execution Context

| Field | Frozen value |
| --- | --- |
| Git HEAD | `71f659aa2620adfaa9fdb34d66c0816543365c82` |
| Branch relation | `main` ahead of `origin/main` by one commit |
| Python | `3.11.14` |
| TensorFlow | `2.19.1` |
| TensorFlow Probability | `0.25.0` |
| Diagnostic device choice | CPU-only; `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import |
| Live leaderboard runner blob | `9c201bd517efedea2414700fc340a0cc048502e3` |
| Dataset generator blob | `dcd0fdd2e77756c9783c1931be8101ce10a621de` |
| Highdim model blob | `ea54bd89ed58d337a72913d2512e5556854620f7` |
| Fixed-SGQF kernel blob | `f49d472b7dd8c2a78f3f01aaf4e89866b446344c` |
| Existing SIR-SGQF design blob | `c58d8e2bfbe83f51e8e94c490664591196a89d34` |
| Zhao--Cui `ssmodel.m` blob | `d8ec527be90798e21d85f73aaa2845805f8b9da1` |
| Relevant pre-edit diff SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty diff) |
| Latest complete historical pair | July 3 JSON SHA-256 `b44fd1ccc8a0132d45ea4f64925bd92930a17c11f7b62bc8f0a15f66631985e7`; Markdown SHA-256 `90873f67192e1a1da5a43ad8ea39fc301a96de8ed10a5b59efb2259cb485091f` |

Concurrent dirty GenUT, transport, and Zhao--Cui APF files are out of this lane
and must not be modified or reverted. Shared SGQF/leaderboard anchors were clean
at the freeze.

## Source-Target Crosswalk

The author-code authority for executable time order is
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m:34`:
`X(:,1)=x0`, then each `t=1:T` computes `X(:,t+1)=st_process(...)` and
`Y(:,t)=ob_process(...,X(:,t+1))`.

| Row | Source timing / target | Current local generator | Classification and action |
| --- | --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | Project exact-oracle row, not a Zhao--Cui RNG reproduction | Explicit transition-then-observe loop | Target-consistent project baseline; retain subject to focused regression. |
| `zhao_cui_sv_actual_nongaussian_T1000` | Author program has 1000 transition-then-observe steps | `StochasticVolatilitySSM.simulate(final_time=999)` emits `y0:y999`, hence 19/999 analogous transition drift | Wrong relative to the declared source timing. Regenerate a 1000-transition fixture or relabel the old evidence as amended-target history. |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | KSC surrogate must share the audited actual-SV timing convention while remaining a different likelihood | Reuses the current initial-observation-first SV fixture | Wrong relative to source timing. Reset on the same source-consistent latent/observation indexing as the actual-SV row; never merge the likelihoods. |
| `zhao_cui_spatial_sir_austria_j9_T20` | Paper/source fixed `kappa_j=0.1`, `nu_j=18`; no free theta; 20 transition-then-observe steps | Seed 81103, `simulate(final_time=19)`, 20 states/observations including `x0/y0` | Wrong source timing and wrong score taxonomy. Regenerate 20 transitions and emit value only. |
| `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale` | Explicit BayesFilter three-log-scale local complete-data component | Byte-identical seed-81103 state/observation fixture to fixed SIR, but a different parameter/conditioning contract | Keep scoped and not applicable to SGQF/UKF full-filter comparison. Byte equality does not make target scopes interchangeable. |
| `zhao_cui_predator_prey_T20` | Source program has 20 transition-then-observe RK4 steps and six physical parameters | Seed 81104, `simulate(final_time=19)`, 20 states/observations including `x0/y0` | Wrong source timing. Preserve the corrected `-103.13789` family only under an amended-target identity; build a fresh source-timing route/data identity. |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | Scalar `svmodels` prior-mean amendment; raw observations; active `(gamma,tau,mu)` coordinates | Seed 81105 loop transitions before each of 1008 observations | Timing-consistent local amendment, but source/amendment identity and scalar route still require qualification. It is not `NativeGeneralizedSVSSM`. |

## Frozen Current Dataset Evidence

Hashes are SHA-256 over contiguous float64 tensor bytes. These identify the
current amended fixtures; they are sentinels to prevent accidental reuse in the
reset source rows.

| Fixture | Seed | State shape / SHA-256 | Observation shape / SHA-256 | Time order |
| --- | ---: | --- | --- | --- |
| Actual SV | 81101 | `(1000,1)` / `4d42af24f704a3e588071d77a09434be430f5af399eef1c5f7df29d447319627` | `(1000,1)` / `f597f9b18090da7aa251a284b1bbc9f3829f6942d42e10dcffc70c30078ea290` | initial observation plus 999 transitions |
| Fixed SIR | 81103 | `(20,18)` / `df4bc05ec9f9f78b9ef198fd18c0fba374e918317db904be8e8114e8973bf59b` | `(20,9)` / `882edb4b9492bea337d0c7466e733010a037bb90bf4211bee35d9a57fd923701` | initial observation plus 19 transitions |
| Parameterized SIR | 81103 | same as fixed SIR | same as fixed SIR | same bytes, different scoped scientific target |
| Predator-prey | 81104 | `(20,2)` / `e0dc312153429ce6010d6a900aab73ed65bc421a52ec8bcd82ca11658df8cd34` | `(20,2)` / `88c3662bc49f7c92cf3969c907feb3f1af3d2da8237afc8b1e53c3631c05ec55` | initial observation plus 19 transitions |
| Generalized SV amendment | 81105 | `(1008,1)` / `7ddf06cc036b2267d3826a368bd718a7760998528beb6d49ec007767ae34898f` | `(1008,1)` / `74e88ee251dc1bc5b274746bc60e58fedbe7c7da03f0a3cd1c2966adb0b6a8f8` | 1008 transition-then-observe steps |

TensorFlow seeds define reproducible source-model synthetic replications. They
do not claim bitwise reproduction of MATLAB random streams.

## Applicability Taxonomy

| Row class | Fixed SGQF | UKF | Zhao--Cui | Readiness rule |
| --- | --- | --- | --- | --- |
| Free-theta main row | applicable, `value_score` | applicable, `value_score` | applicable, `value_score` | all applicable cells need an admitted value and manual/analytical score |
| Fixed SIR main row | applicable, `value_only_no_free_theta` | applicable, `value_only_no_free_theta` | applicable, `value_only_no_free_theta` | all applicable cells need an admitted value; no score may be required |
| Parameterized-SIR scoped component | not applicable | not applicable | applicable scoped component, `value_score` | excluded from main comparison and SGQF denominator; scoped readiness is reported separately |

Unknown rows or algorithms fail closed. `full_three_way_ready` remains a
deprecated compatibility field with its historical meaning; it is not the
SGQF completion criterion.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Source timing follows author executable code | Checked `ssmodel.complete`; reviewed target fact | Plausible values for a different conditioning sequence | T=1 and T=2 transition-count tests plus reset hashes |
| TensorFlow RNG replaces MATLAB RNG for reset synthetic data | Convenience choice, explicitly not source replay | False bitwise reproduction claim | Record language/seed and forbid MATLAB replay claim |
| Level-2 SIR axis cloud (37 points in `d=18`) | Existing design hypothesis | Center weight `1-d/3=-5` and missing mixed fourth moments may bias coupled quadratic dynamics | Exact moment checks, same-target UKF/PF diagnostics, refinement where feasible |
| Existing SV/SGQF numerical values | Historical amended-target evidence | Stale values silently promoted after target reset | Fresh row/data identities and same-target reruns |
| Fixed SIR has no score | Paper/source target fact | Artificial parameter extension changes scientific row | Schema rejects score requirement; parameterized component remains separate |

## Phase 0 Gate And Next Action

| Decision | Status |
| --- | --- |
| Live inventory exhaustive | pass: seven rows, three algorithm identifiers |
| Concurrent-lane isolation | pass at freeze |
| Source/index mismatch explained | pass; affected rows explicitly classified |
| Old evidence quarantined | pass by target classification; no files deleted |
| Applicability schema ready to implement | pass |
| Source-row numerical promotion | blocked pending regenerated identities and reruns |

The next justified action is the applicability-aware schema patch and focused
CPU-only tests. Then build the fixed-SIR transition-first dataset and value-only
SGQF route. No GPU or claim-bearing numerical run is justified before those CPU
gates pass.
