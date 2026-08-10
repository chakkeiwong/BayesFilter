# Zhao-Cui GenUT Austria T2 Dual-Cap Result

Date: 2026-08-06

Status: `PASS_T2_CANDIDATE_FOR_T3_PLUS_TEACHER_EXTENSION`

Plan:
`docs/plans/bayesfilter-zhao-cui-genut-austria-t2-dual-cap-plan-2026-08-06.md`

Primary artifact:
`docs/benchmarks/artifacts/zhao_cui_genut_austria_t2_dual_cap_20260806/attempt02/result.json`

Artifact SHA-256:
`e5b947b0324c01fd0bd88d7f5e58c37c08b7a6f59f7649c77c7bee6882732d80`

## Verdict

The combined cap works for the narrow T2 numerical-validity question. Adding a
smooth coordinatewise interior cap after the final bounded-coordinate
restandardization made the previously invalid Zhao-Cui/GenUT route finite and
support-preserving. The existing radial correction-direction cap was evaluated
with and without it; it was not needed by the calibration selection rule.

The selected arm is:

```text
radial correction-direction cap: disabled
coordinatewise cap: b=0.98, p=8
diagonal strength: 0
pairwise strength: 0.02
four diagonal and four pairwise steps
```

All 18 untouched validation rows (three teachers crossed with six common
particle seeds) were finite and program-valid. All 9 validation finite-
difference coordinates passed. The teacher-to-particle standard-deviation ratio
was below the predeclared `0.5` limit for value and every score coordinate.

This is numerical viability and teacher-seed robustness evidence at `T=2`, not
evidence of score accuracy, likelihood improvement, posterior correctness,
T20 validity, HMC/NeuTra readiness, or default readiness.

## Claimed and Computed Quantities

| Item | Result |
|---|---|
| Claimed score | Total JVP of the same finite dual-cap GenUT scalar |
| Independent score authority | None for nonlinear Austria T2; finite differences test internal derivative consistency only |
| Teacher target | Independent self-normalized 128-sample Zhao-Cui bounded-coordinate moment/JVP estimator |
| Physical raw third/fourth moments | Not claimed; Lane-B defensive physical moments diverge |
| Cap classification | `extension_or_invention`; it is not a Zhao-Cui source operation |
| Scope | Austria SIR Lane-B latent-preclip, sealed observations, `T=2,N=1008`, FP32/no-TF32/GPU/XLA |

## Cap Definition

After the last diagonal/pairwise correction and restandardization, each local
bounded coordinate `x` is mapped as

\[
 f_b(x)=\frac{x}{\left(1+(x/b)^8\right)^{1/8}}, \qquad b=0.98.
\]

For finite `x`, `|f_b(x)|<b<1`. The complete tangent uses

\[
 f_b'(x)=\left(1+(x/b)^8\right)^{-9/8}.
\]

The resulting order is:

```text
diagonal correction
-> pairwise correction with optional radial direction cap
-> final local restandardization
-> coordinatewise smooth cap
-> bounded-to-unbounded inverse
-> physical affine mean/covariance restoration
```

The cap changes third/fourth moments after matching. The post-cap residuals are
therefore part of the empirical result, not a hidden exact-moment claim.

Post-run review found that the successful artifact's explanatory skew/kurtosis
and pairwise residual fields standardized the capped cloud using the pre-cap
source mean/covariance rather than the capped cloud's own mean/covariance. These
fields were not used by the selection rule, promotion gates, value/score, FD,
support, affine-restoration, or teacher-sensitivity calculations. The current
code corrects that diagnostic and adds an independent regression test. The
artifact remains valid for the stated T2 numerical and sensitivity verdict but
must not be cited for its stored post-cap moment-residual values.

## Calibration

Calibration used the existing strict calibration teacher and particle seeds
`98701,98702`. Shape strengths were frozen at the closest-support prior:
diagonal strength `0`, pairwise strength `0.02`, four steps each. Eight arms
were tested: radial cap `{off,2}` crossed with coordinate cap
`{off,0.90,0.95,0.98}`. A no-shape arm was retained as the exact baseline in
validation.

| Arm | Calibration valid | Worst maximum `|u|` | Mean cap displacement | Mean maximum inverse derivative | FD gate |
|---|---|---:|---:|---:|---|
| radial off, coordinate off | no | `1.033245` | `0` | `3.47e8` | not computable |
| radial off, coordinate `0.90` | yes | `0.868371` | `0.003355` | `8.18` | pass |
| radial off, coordinate `0.95` | yes | `0.902249` | `0.002572` | `12.41` | pass |
| radial off, coordinate `0.98` | yes | `0.920194` | `0.002175` | `16.54` | pass |
| radial `2`, coordinate off | no | `1.032220` | `0` | `3.47e8` | not computable |
| radial `2`, coordinate `0.90` | yes | `0.868156` | `0.003368` | `8.18` | pass |
| radial `2`, coordinate `0.95` | yes | `0.901946` | `0.002584` | `12.41` | pass |
| radial `2`, coordinate `0.98` | yes | `0.919832` | `0.002186` | `16.54` | pass |

The selection rule chose `radial off, coordinate 0.98` because it had the
smallest calibration mean cap displacement among valid arms. It is the least
distorting valid cap under this calibration criterion. The radial cap has a
small secondary effect but did not improve the primary selection metric.

## Validation Numerical Results

Three independent teachers were crossed with six common particle seeds, for 18
candidate rows. The no-shape baseline used the same six particle seeds.

| Diagnostic | Validation result | Gate |
|---|---:|---:|
| Finite/program-valid rows | `18/18` | all required |
| Maximum post-cap `|u|` | `0.933672` | `<1` |
| Maximum pre-cap `|u|` | `1.076091` | explanatory |
| Maximum normalized affine mean residual | `1.28e-7` | `<=2e-4` |
| Maximum normalized affine covariance residual | `7.67e-7` | `<=2e-4` |
| Maximum finite-difference absolute residual | `0.004454` | `<=0.08` |
| Maximum finite-difference normalized residual | `0.001719` | `<=0.03` |
| Maximum inverse derivative | `21.77` | explanatory; much lower than uncapped |
| Minimum coordinate-cap derivative | `0.2787` | explanatory |
| Maximum coordinate-cap active fraction | `10.65%` | explanatory |
| Maximum mean coordinate-cap displacement | `0.002270` | explanatory |

The coordinate cap is active on roughly 9--11% of coordinates, so the result
supports the stated tail-focused interpretation descriptively. It is not a
proof that only statistically irrelevant tails changed.

## Teacher Sensitivity

For each metric, teacher SD is the SD of the three teacher-specific particle
means. Particle SD is the root mean of the three within-teacher six-seed sample
variances. The ratio is therefore a teacher-versus-single-evaluation particle
variation comparison, not the MCSE of the 18-row grand mean.

| Metric | Between-teacher SD | Pooled particle SD | Ratio | Gate |
|---|---:|---:|---:|---|
| Value | `0.004038` | `0.057958` | `0.0697` | pass |
| Score 0 | `0.007632` | `0.162801` | `0.0469` | pass |
| Score 1 | `0.006567` | `0.065632` | `0.1001` | pass |
| Score 2 | `0.014548` | `0.185313` | `0.0785` | pass |

All ratios are far below `0.5`. This supports the limited statement that the
128-sample teacher randomness was not the dominant observed source of variation
in this T2 experiment.

## Values and Scores

Candidate-minus-baseline differences are descriptive only:

| Metric | Mean difference | SD | Negative | Positive |
|---|---:|---:|---:|---:|
| Value | `-0.000590` | `0.003515` | 8 | 10 |
| Score 0 | `+0.011387` | `0.021092` | 4 | 14 |
| Score 1 | `-0.017258` | `0.011805` | 18 | 0 |
| Score 2 | `+0.005545` | `0.012997` | 9 | 9 |

These differences do not establish improvement or accuracy. There is no exact
nonlinear Austria score reference in this experiment, and the cap deliberately
changes the heuristic moment-matching map.

## Per-Teacher FD Results

| Teacher | Maximum absolute FD residual | Maximum normalized FD residual |
|---:|---:|---:|
| 1 | `0.002691` | `0.000435` |
| 2 | `0.002589` | `0.000916` |
| 3 | `0.004454` | `0.001719` |

All nine parameter-coordinate checks passed.

## Run Manifest

| Field | Value |
|---|---|
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu` |
| TensorFlow | `2.19.1` |
| GPU route | TensorFlow `/GPU:0`, RTX 4080 SUPER; RTX 5080 also visible |
| Dtype / TF32 / XLA | FP32 / disabled / compiled |
| Memory policy | growth configured before logical-device initialization |
| Deterministic ops | enabled |
| Wall time | `291.54 s` |
| Artifact | `docs/benchmarks/artifacts/zhao_cui_genut_austria_t2_dual_cap_20260806/attempt02/result.json` |
| Result SHA-256 | `e5b947b0324c01fd0bd88d7f5e58c37c08b7a6f59f7649c77c7bee6882732d80` |

## Decision Table

| Decision | Criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain dual-cap route as T2 candidate | all numerical and teacher-sensitivity gates pass | none | no exact score authority; only T2 | construct/validate T3--T20 teachers | no default or HMC readiness |
| Select `b=0.98`, radial off for this scope | least cap displacement among calibration-valid arms | none | selection is scope-specific | freeze controls for next teacher-construction scope | no superiority |
| Treat values/scores as descriptive | no exact reference | no accuracy gate available | cap changes heuristic moments | use only for proposal-training diagnostics after further validation | no score correctness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | pass: all 18 validation rows and all FD checks pass |
| Statistically supported arm ranking | none; eight-arm calibration is a viability/least-distortion selection, not a superiority test |
| Descriptive differences | values, scores, cap activity, inverse derivatives, runtimes; stored artifact moment residuals are ineligible after the post-run diagnostic finding |
| Teacher-sensitivity evidence | pass for the declared `0.5` ratio screen |
| Default readiness | no |
| Next evidence needed | valid teachers for every `T=3..20`, then scope-specific T20 calibration and untouched multi-seed validation |

## Negative Result and Red Team

The uncapped route still fails its bounded-domain gate, confirming that the cap,
not a transient process difference, caused the finite-program repair. The radial
cap is not necessary for the selected T2 arm, although it slightly changes
displacement and inverse-derivative diagnostics.

The strongest alternative explanation is that the coordinate cap makes the
program finite by changing the objective enough to hide a poor higher-moment
teacher. That is why post-cap moment residuals and candidate-baseline changes
are retained as explanatory diagnostics, and why no score-accuracy claim is
made. The result would be overturned by a valid external score authority or by
failure of the same cap under T3+ or T20 scope-specific validation.
