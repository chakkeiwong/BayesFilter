# Zhao-Cui Bounded GenUT Austria T2 Crossed Validation Plan

Date: 2026-08-06

Status: `EXECUTED_STOPPED_CALIBRATION_VETO`

Terminal result:
`docs/plans/bayesfilter-zhao-cui-bounded-genut-austria-t2-crossed-validation-result-2026-08-06.md`

## Research intent ledger

| Field | Frozen decision |
|---|---|
| Main question | After target-specific control calibration, is the sampled bounded Zhao-Cui teacher sufficiently numerically valid and insensitive to teacher Monte Carlo seed to justify extending the teacher sequence beyond T2? |
| Candidate | T2 Austria Lane-B latent-preclip GenUT with four diagonal and four pairwise third/fourth-moment correction steps, a calibrated pairwise strength and smooth radial cap, and an independently sampled Zhao-Cui bounded-coordinate teacher. |
| Exact baseline | The identical finite Austria program, observations, particles, Contract-E controls, FP32/no-TF32/GPU/XLA arithmetic, and particle seeds with no higher-moment correction. |
| Expected failure | Bounded-chart exit, failed physical mean/covariance restoration, same-program score/finite-difference disagreement, or teacher-seed variation comparable with or larger than ordinary particle Monte Carlo variation. |
| Promotion criterion | Every frozen-candidate validation row and teacher-specific FD check passes, and for value plus every score coordinate the between-teacher SD of teacher marginal means is at most `0.5` times the pooled within-teacher particle-seed SD. Passing promotes only construction of T3+ teachers. |
| Promotion veto | Non-finite value/score, invalid finite program, non-GPU row, any corrected bounded coordinate with absolute value `>=1`, normalized physical affine mean/covariance residual above `2e-4`, FD absolute residual above `0.08`, or FD normalized residual above `0.03`. |
| Continuation veto | Strict teacher identity failure, invalid sealed observations, no calibration candidate passing the numerical gates, corrupted/missing diagnostics, inability to execute GPU/XLA with verified memory growth, or exhausted campaign budget. |
| Repair trigger | A localized CLI, tracing, serialization, instrumentation, or resource failure that leaves target, data, candidates, criteria, partitions, hardware class, and total budget unchanged. |
| Explanatory only | Candidate-baseline value/score differences, moment residuals, cap activity, runtimes, raw affine residuals, and signs of paired differences. |
| Must not be concluded | Exact physical moments, exact Austria score, improved likelihood or score accuracy, posterior correctness, HMC/NeuTra readiness, T20 validity, statistical superiority, or default readiness. |

The computed score is the total JVP of each executed finite GenUT program. It is
checked against finite differences of that same scalar. There is no independent
exact T2 Austria score authority, so derivative parity proves internal
consistency, not accuracy relative to the true filtering likelihood.

## Source and target boundary

The T1/T2 Lane-B parent and issued score-child identities remain those strictly
loaded by `zhao_cui_austria_sir_bounded_teacher_tf.py`. The bounded teacher is a
self-normalized fixed-sample estimator using exact `log p_TT - log q` weights
and issued marginal-score tangents. It never uses GenUT particle moments.

The composition of bounded Zhao-Cui moments with GenUT correction is
`extension_or_invention`. It is not a source-faithful Zhao-Cui filtering
operation and is not an estimator of the divergent physical high moments of
the Lane-B defensive mixture.

## Frozen partitions and candidates

| Role | Seeds or choices |
|---|---|
| Calibration teacher | time seeds `(98541,98542)`, 128 samples |
| Calibration particles | `98701,98702` |
| Validation teachers | time-seed pairs `(98611,98612)`, `(98621,98622)`, `(98631,98632)`, 128 samples each |
| Validation particles | `98801..98806`, common across all teachers and baseline |
| Fixed diagonal controls | steps `4`, strength `0.2` |
| Pairwise tuning grid | steps `4`, strength in `{0.005,0.01,0.02}`, cap in `{1,2,4}` |
| Fixed filter scope | `T=2`, `N=1008`, FP32, TF32 disabled, GPU/XLA |

The calibration selection rule is lexicographic and frozen: discard every
candidate failing a gate, then minimize the mean normalized pairwise residual
objective over the two calibration particle seeds; break an exact tie by lower
pairwise strength and then lower cap. Value and score displacement are not used
for tuning because no external accuracy authority exists.

For each validation quantity, teacher sensitivity is

\[
 s_{\rm teacher}=\operatorname{SD}_a\left(\bar y_{a\cdot}\right),
 \qquad
 s_{\rm particle}=\sqrt{\frac{1}{3}\sum_a
                 \operatorname{Var}_b(y_{ab})}.
\]

The predeclared screen is `s_teacher / s_particle <= 0.5`. Here
`s_particle` estimates the Monte Carlo standard error of one finite-program
evaluation across particle seeds. It is not the MCSE of the six-row average,
which would be smaller.

## Evidence contract

The exact comparator is no shape correction under common particle random
numbers. The primary decision is whether the frozen bounded-teacher candidate
is eligible for the next teacher-construction phase, not whether it is closer
to an unknown true score. Numerical vetoes are finite-program validity,
bounded-chart validity, physical affine restoration, GPU/XLA placement, and
same-program FD. Teacher sensitivity is the promotion criterion. Moment
residual, cap, candidate-baseline displacement, and runtime are explanatory.

The result is preserved under the unique root
`docs/benchmarks/artifacts/zhao_cui_bounded_genut_austria_t2_crossed_validation_20260806/`
with strict teacher manifests, a selection ledger, the complete crossed table,
source hashes, environment, seeds, command, GPU/memory policy, wall time, and a
result note.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| 128 teacher samples | doubled repair hypothesis relative to the prior 64-sample smoke | improves high-moment precision while four artifacts remain bounded | still too noisy, or build cost grows unexpectedly | three-way teacher sensitivity and ESS/log-correction rows |
| Three teacher seeds | bounded validation design | separates teacher Monte Carlo variation from particle variation | imprecise teacher-SD estimate | report ratios as a hard screen, not a ranking |
| Six particle seeds | user-requested multi-seed validation | estimates ordinary finite-filter Monte Carlo scale with common random numbers | still weak for stochastic ranking | explicit non-ranking and full per-seed table |
| Ratio `0.5` | user-approved adequacy threshold | teacher noise at half particle MC error is not the dominant uncertainty source | threshold does not imply negligible error for an averaged estimator | report numerator, denominator, and ratio separately |
| Diagonal strength `0.2` | inherited warm start, not a default | pairwise correction is incremental on the existing diagonal pass | all candidates could inherit diagonal over-correction | boundary and FD gates; no-candidate pass is a tuning failure |
| Pairwise grid | cap-2/strength-0.02 smoke plus nearby weaker hypotheses | small discriminating grid around the only passing smoke | optimum may lie outside the grid | candidate rejection triggers a new plan, not extrapolation |
| Moment residual selection | mechanism-specific calibration criterion | directly measures the declared bounded moment-matching goal | can select a score-inaccurate program | no score-accuracy claim; FD checks consistency only |
| `N=1008,T=2` | current exact cubature-compatible diagnostic scope | answers the immediate repair question | cannot transfer to N=4096 or T20 | scope-bound result and no transfer claim |
| FP32/no TF32/XLA | prior derivative-parity scope | avoids the earlier TF32 derivative deterioration | FP32 may still amplify a near-boundary inverse | bounded maximum and FD gates |

## Skeptical plan audit

Verdict: `PASS_AFTER_MATERIAL_REVISION`.

- Wrong baseline: repaired by exact no-shape bypass under common particles.
- Proxy promotion: moment residual selects controls only; it cannot establish
  score accuracy. Extension beyond T2 additionally requires numerical and
  teacher-sensitivity gates.
- Missing stop conditions: explicit artifact, boundary, affine, FD, device,
  calibration, and budget vetoes are frozen.
- Unfair comparison: teachers are crossed with the same six particle seeds;
  the baseline shares every non-candidate input.
- Hidden assumption: raw affine restoration errors are scale dependent.
  Scale-normalized residuals are now exposed and gated rather than assuming
  that internal restoration succeeded.
- Stale context: historical recursive T20 and empirical-teacher artifacts are
  excluded. Only current strict T1/T2 issuer chains are eligible.
- Environment mismatch: serious evaluation is FP32/no-TF32/GPU/XLA with memory
  growth configured and verified before device initialization. Teacher builds
  are deliberately CPU-only and recorded as such.
- Artifact sufficiency: the full crossed table and variance decomposition
  directly answer numerical validity and teacher-seed sensitivity.

## Pre-mortem

- The campaign could pass FD while matching the wrong target. The result must
  state the bounded-coordinate estimator and `extension_or_invention` status.
- It could select an aggressive control using validation rows. The script
  freezes selection from calibration data before loading/evaluating validation
  rows.
- It could mistake particle variation for teacher variation. The fully crossed
  common-seed design reports separate between-teacher and within-teacher scales.
- It could pass raw restoration tolerances only because Austria state scales
  are large. Normalized mean/covariance residuals are the veto quantities.
- It could pass T2 and fail recursively. Passing authorizes T3+ teacher work;
  it does not authorize or predict T20 performance.

## Execution and budget

1. Expose and test post-teacher physical affine restoration diagnostics.
2. Build four fresh 128-sample CPU-only teacher artifacts in unique paths.
3. Run one GPU/XLA campaign: calibrate the nine candidates, freeze selection,
   then run the exact baseline and selected candidate for three teachers by six
   particle seeds plus one FD check per validation teacher.
4. Write the result and inference-status tables; rerun focused tests and
   `git diff --check`.

Budget: at most 15 aggregate CPU minutes for teacher construction and 25 GPU
minutes for the campaign, with one localized unchanged-contract repair retry.
Stop instead of changing the target, partitions, grid, thresholds, sample
count, hardware class, or T2 scope after results are observed.

T20 is not part of this campaign. A T20 run requires validated Zhao-Cui teacher
objects for every time `T=1..20`; repeating the T2 teacher at later times would
compute the wrong declared program.
