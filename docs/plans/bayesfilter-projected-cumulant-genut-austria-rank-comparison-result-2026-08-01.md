# Projected-Cumulant GenUT Austria Rank Comparison Result

Date: 2026-08-01  
Plan: `docs/plans/bayesfilter-projected-cumulant-genut-austria-rank-comparison-plan-2026-08-01.md`  
Artifacts: `docs/benchmarks/artifacts/projected_cumulant_genut_austria_20260801/`  
Status: `PROJECTED_CUMULANT_CANDIDATE_REJECTED`

## Executive Verdict

The implementation is mechanically executable on the trusted RTX 4080 SUPER
GPU with TensorFlow/XLA/TF32 and verified memory growth. The projected
third/fourth-moment correction is not a viable Austria candidate under the
predeclared contract:

- At `N=1008`, ranks `r=4` and `r=8` had no mechanically valid setting in the
  six-arm validation grid. Rank `r=6` had valid validation rows for all six
  controls, but no control passed the score-variance veto; its selected arm was
  explicitly an ineligible diagnostic representative and failed `2/16` claim
  seeds.
- At `N=4032`, ranks `r=4` and `r=8` again had no mechanically valid setting.
  Rank `r=6` became eligible on the three-seed validation scope, but its claim
  score dispersion was much worse than pairwise and included a catastrophic
  common-seed displacement.
- The heldout calibration/validation bases are unstable: maximum principal
  angles are approximately `88.5--90.0` degrees for every tested rank and
  particle count. This is a promotion veto and supports the interpretation that
  the low-rank structure is mostly particle/cloud-specific noise.

This rejects the current projected-cumulant candidate and its present basis
construction/control grid. It does not prove that every possible low-rank
teacher is impossible, but the research direction should not continue to a
default or HMC path without a new, independently justified basis mechanism.

## Evidence And Scope

The claim-bearing `N=1008` run used 16 common particle seeds `98201..98216`.
The `N=4032` run used three descriptive seeds `98201..98203`; it is not
statistical ranking evidence. Calibration observation seeds were `91141,91142`,
validation observation seeds `91241,91242`, and tuning particle seeds were
`98301,98302`. Both scopes used the legal Austria exact cubature counts `1008`
and `4032`, FP32 with TF32 enabled, XLA JIT, and the same target observation
hash. `N=4000` was not used because it violates the replicated 36-point
cubature divisibility requirement.

The SGQF values in the runner are a descriptive comparator, not an exact
nonlinear Austria oracle:

```text
value = -682.3480055392419
score = [28.739453057371584, -106.65885657030441, 9.43117639262833]
```

## Value And Score Results

### N=1008, 16 Seeds

| Arm | Valid rows | Value mean (SD) | Score SD `(0,1,2)` | Status |
|---|---:|---:|---|---|
| Diagonal | 16/16 | `-683.3638 (0.6367)` | `[3435.63, 1272.44, 301.97]` | mechanically valid, severe score outliers |
| Pairwise | 16/16 | `-682.1039 (0.5647)` | `[33.94, 17.99, 19.72]` | mechanically valid, descriptive variance repair |
| Projected `r=4` | 0/0 run | unavailable | unavailable | no mechanically valid tuned control |
| Projected `r=6` | 14/16 | `-683.1267 (0.5480)` over finite rows | `[134.15, 57.56, 132.84]` over finite rows | invalid claim arm; no ranking interval |
| Projected `r=8` | 0/0 run | unavailable | unavailable | no mechanically valid tuned control |

The projected `r=6` rows are reported descriptively only because two claim rows
were non-finite. Its selected validation control was
`projected_steps=2, strength=0.0025`, marked
`INELIGIBLE_DIAGNOSTIC_REPRESENTATIVE` because it failed the validation
score-variance criterion. The two invalid seeds were `98206` and `98210`.

### N=4032, Three Descriptive Seeds

| Arm | Valid rows | Value mean (SD) | Score SD `(0,1,2)` | Status |
|---|---:|---:|---|---|
| Diagonal | 2/3 | `-683.5815 (0.1294)` over finite rows | `[23.31, 21.90, 53.46]` over finite rows | one non-finite seed |
| Pairwise | 3/3 | `-682.1668 (0.4634)` | `[34.25, 20.88, 5.49]` | mechanically valid, descriptive only |
| Projected `r=4` | 0/0 run | unavailable | unavailable | no mechanically valid tuned control |
| Projected `r=6` | 3/3 | `-683.8201 (0.4219)` | `[2453.75, 1012.20, 312.01]` | descriptive finite arm, severe score dispersion |
| Projected `r=8` | 0/0 run | unavailable | unavailable | no mechanically valid tuned control |

The `N=4032` rank-6 selected control was `projected_steps=1,
strength=0.01`. The three-seed score SD is not a promotion or superiority
claim. It is nevertheless decisive as a diagnostic: the largest score SD is
roughly 72 times the pairwise value for coordinate 0 and 48 times for
coordinate 1.

### Cross-N Displacement

For common seeds `98201..98203`, pairwise value displacements from `N=1008` to
`N=4032` were `[-0.931, +0.116, +0.047]`; score displacements remained on the
order of tens. Rank-6 projected value displacements were `[-1.897, -0.595,
-0.779]`, but the common seed `98202` had score-0 displacement `+3922.46`,
with corresponding score-1 and score-2 displacements `-1608.99` and `-499.88`.
This is a descriptive instability diagnostic, not an exact bias estimate.

## Basis And Residual Diagnostics

| Scope/rank | Validation explained energy | Mean principal angle (deg) | Max principal angle (deg) |
|---|---:|---:|---:|
| `N=1008,r=4` | `0.284` | `46.81` | `89.94` |
| `N=1008,r=6` | `0.403` | `39.18` | `89.96` |
| `N=1008,r=8` | `0.517` | `34.08` | `89.67` |
| `N=4032,r=4` | `0.310` | `38.87` | `88.51` |
| `N=4032,r=6` | `0.433` | `35.53` | `89.91` |
| `N=4032,r=8` | `0.559` | `28.68` | `89.12` |

Explained energy increases with rank by construction, but the near-90-degree
principal angles show that the calibration eigenspace is not reproducible on
the untouched validation clouds. Projected residual norms for the finite rank-6
claim rows were approximately `17--49` at `N=1008` and `17--33` at `N=4032`;
residual reduction is explanatory only and did not translate into score
stability.

## Engineering Evidence

| Check | Result |
|---|---|
| Focused projected/higher-moment tests | `17 passed`, CPU-hidden diagnostic lane |
| LaTeX note | `latexmk` succeeded; 15-page PDF produced |
| Trusted GPU probe | RTX 4080 SUPER visible; TensorFlow logical GPU created |
| Memory policy | growth verified before logical-device initialization |
| XLA/TF32 | XLA clusters compiled; TF32 enabled; no CPU fallback in claim rows |
| Smoke | attempt 01 exposed an opaque harness gate; attempt 02 completed with rank-local invalid arms |
| `N=1008` campaign | completed in `316.64 s`; peak allocator `90,661,120` bytes |
| `N=4032` campaign | completed in `428.86 s`; peak allocator `577,713,920` bytes |
| Artifact provenance | JSON, Markdown, checkpoint, manifest, source hashes, target hash, seeds, controls, and nonclaims preserved |

The first smoke and first claim attempts are retained as separate historical
attempts. Their failures were localized harness/reporting defects and were
repaired without changing the target, method, data split, promotion criteria,
hardware class, or campaign budget.

## Decision Table

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject projected `r=4` | No valid tuning candidate at either `N` | Mechanics tuning veto | Grid may be too narrow, but all six controls failed | Do not expand grid under this campaign; require a new basis/control proposal | Not a theorem that rank 4 can never work |
| Reject projected `r=6` for promotion | `N=1008` score/value selection failed; 2/16 claim invalid; `N=4032` score SD exploded | Claim validity and score-variance veto | Rank-6 finite value SD looks small, but score is the target metric and no oracle exists | Archive as diagnostic only | No score-bias estimate |
| Reject projected `r=8` | No valid tuning candidate at either `N` | Mechanics tuning veto | Possible overfitting/high-order amplification | Do not promote or run HMC | Not a theorem that all rank 8 maps fail |
| Retain pairwise as prior opt-in comparator | Full validity and much smaller score SD than diagonal at `N=1008` | Existing value no-regression gate from prior pairwise result remains failed | Absolute score accuracy remains unknown | Keep opt-in only; no default/HMC promotion | No claim that pairwise is exact or superior |
| Reject current low-rank direction for continuation | Heldout basis instability and rank-6 score blow-up | Research-direction promotion veto, not harness invalidity | A different low-rank basis mechanism might behave differently | Require a new independently justified mechanism and fresh plan before more compute | No universal impossibility result |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Baselines pass at `N=1008`; projected rank 6 fails 2/16 claim seeds; diagonal fails 1/3 at `N=4032`; ranks 4/8 have no valid tuned arms |
| Statistically supported ranking | None for projected ranks. No complete common-seed variance-ratio interval exists for an invalid/incomplete projected arm |
| Descriptive-only differences | Pairwise versus diagonal score SD, rank-6 finite value/score means, SGQF gaps, basis energy/angles, and cross-N displacement |
| Default readiness | Not established; projected candidate is rejected and pairwise remains opt-in only |
| Next evidence needed | A new basis mechanism with heldout stability, full valid claim rows, and an absolute/reference score or value diagnostic before any ranking claim |

## Post-Run Red Team

The strongest alternative explanation is that the frozen eigenspace is being
fit to particle-cloud residual noise, not to a stable Austria transition
geometry. The nearly orthogonal validation bases and rank-6 catastrophic score
seed support that explanation. A weaker alternative is that the six-control grid
was underpowered; that could motivate a new experiment, but it cannot rescue
this candidate or justify silently expanding the grid after seeing claim data.

The result would be overturned only by a fresh, preplanned mechanism that uses
disjoint basis/tuning data, achieves heldout subspace stability, and passes a
complete untouched claim run with uncertainty evidence. Nothing here establishes
an exact nonlinear Austria likelihood, absolute score bias, HMC readiness,
source-faithful Zhao--Cui equivalence, a universal rank law, or a default
numerical policy.
