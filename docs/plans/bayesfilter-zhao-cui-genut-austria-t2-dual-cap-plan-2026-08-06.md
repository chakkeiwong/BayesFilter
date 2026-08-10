# Zhao-Cui GenUT Austria T2 Dual-Cap Experiment Plan

Date: 2026-08-06

Status: `EXECUTED_PASS_T2_CANDIDATE`

Terminal result:
`docs/plans/bayesfilter-zhao-cui-genut-austria-t2-dual-cap-result-2026-08-06.md`

## Research Intent

| Field | Frozen decision |
|---|---|
| Main question | Does adding a smooth coordinatewise interior cap after the final bounded-coordinate restandardization make the existing Zhao-Cui bounded GenUT route finite and numerically differentiable, while changing only the tails as much as possible? |
| Candidate mechanism | Existing pairwise correction with the previously closest control (`diagonal strength=0`, `pairwise strength=0.02`, four diagonal/pairwise steps), the existing smooth per-particle radial correction-direction cap, and a new smooth coordinatewise cap applied after the final restandardization and before the bounded-to-unbounded inverse. |
| Exact baseline | The same T2 finite program with no shape correction, common Austria observations, Contract-E controls, arithmetic, and particle seeds. |
| Calibration arms | Radial cap in `{off,2}` crossed with coordinate cap `b` in `{off,0.90,0.95,0.98}`: 8 shape arms plus no-shape baseline. |
| Primary promotion criterion | A candidate must pass all numerical gates on both calibration particle seeds and all three validation teachers crossed with six untouched validation particle seeds. Among candidates that pass, nominate the smallest coordinate-cap distortion; no superiority claim is made. |
| Promotion veto | Non-finite/invalid program, non-GPU execution, corrected `|u|>=1`, normalized physical affine mean/covariance residual above `2e-4`, or same-program FD residual above absolute `0.08` or normalized `0.03`. |
| Continuation veto | The coordinate-cap JVP/support unit tests fail, no calibration arm passes, teacher artifacts fail strict reload, or the unchanged campaign budget is exhausted. |
| Explanatory diagnostics | Pre/post radial RMS, minimum radial scale, maximum and mean coordinate-cap displacement, changed-coordinate fraction, maximum post-cap `|u|`, inverse-derivative maximum, moment residuals, values, scores, and runtimes. |
| Must not be concluded | Exact physical moments, exact Austria score, score accuracy, unbiasedness, statistical arm superiority, T20 validity, posterior correctness, HMC/NeuTra readiness, or default readiness. |

The score remains the total JVP of the executed finite scalar. Same-program
finite differences test derivative consistency only; there is no independent
exact Austria T2 score authority.

## Mathematical Definition

The coordinatewise cap is applied to the final standardized bounded coordinates
`x` returned by `higher_moment_shape_jvp`:

\[
 f_b(x)=\frac{x}{\left(1+(x/b)^p\right)^{1/p}},
 \qquad p=8,\quad 0<b<1.
\]

It is odd, smooth, identity-like near zero, and satisfies `|f_b(x)|<b` for
finite `x`. Its scalar derivative is

\[
 f_b'(x)=\left(1+(x/b)^p\right)^{-1/p-1}.
\]

The tangent is multiplied coordinatewise by this derivative. `b=0` is an exact
no-cap route. This cap changes the teacher third/fourth moments; it is an
explicit bounded heuristic, not exact moment matching.

The existing radial cap remains inside the pairwise correction direction. The
order is therefore:

```text
diagonal correction
-> pairwise correction with radial direction cap
-> final restandardization
-> coordinatewise smooth cap
-> bounded-to-unbounded inverse
-> physical affine mean/covariance restoration
```

## Scope and Partitions

| Role | Frozen choice |
|---|---|
| Model | Austria SIR Lane-B latent-preclip |
| Horizon / particles | `T=2`, `N=1008` |
| Teacher | Existing strict 128-sample Zhao-Cui T1/T2 artifacts |
| Calibration teacher | `teacher-calibration-n128-seeds98541-98542` |
| Calibration particles | `98701`, `98702` |
| Validation teachers | Existing `teacher-validation01..03-n128` artifacts |
| Validation particles | `98801..98806`, common across teachers and baseline |
| Arithmetic | FP32, TF32 disabled, GPU/XLA, verified memory growth |
| Fixed shape controls | diagonal strength `0`, pairwise strength `0.02`, four steps each |
| Radial cap | `0` (disabled) or `2` |
| Coordinate cap | `0` (disabled), `0.90`, `0.95`, or `0.98`; exponent `p=8` |

## Selection Rule

First discard any arm failing all calibration numerical/FD gates. For each
remaining arm compute the calibration mean coordinate-cap displacement and the
changed-coordinate fraction. Nominate the arm with the smallest mean
coordinate-cap displacement; break ties by smaller inverse-derivative maximum,
then lower radial activity. This chooses the least intrusive valid cap, not the
best likelihood or score.

The validation stage is run only after calibration selection is frozen. It
reports the three-teacher by six-particle table and the predeclared teacher to
particle SD ratio `<=0.5` for value and each score coordinate. A passing ratio
only says teacher sampling is not the dominant observed variation; it does not
establish score accuracy.

## Default and Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| `p=8` | smooth tail-focused hypothesis | cap may still affect bulk or create derivative saturation | displacement and changed-fraction diagnostics |
| `b={.90,.95,.98}` | interior-margin hypothesis | no arm may balance support and distortion | calibration gate and inverse-derivative diagnostic |
| radial `{off,2}` | existing cap diagnostic and closest prior candidate | interaction could be non-monotone | crossed cap table |
| fixed strengths | closest-support prior arm, not a default | strength tuning may still matter | nonclaim; new strength grid requires a new plan |
| least-displacement selection | preserves tails subject to validity | may select a noisier score | FD and teacher-sensitivity gates; no accuracy claim |
| `0.5` teacher/particle ratio | predeclared user criterion | only three teacher seeds | report ratio descriptively and forbid ranking |

## Skeptical Plan Audit

Verdict: `PASS_AFTER_MATERIAL_REVISION`.

- **Wrong baseline:** no-shape uses the exact zero-step bypass and shares all
  non-candidate inputs.
- **Proxy promotion:** cap activity and moment residuals select/narrate only;
  they cannot establish score accuracy.
- **Support claim:** the coordinate map has strict range `(-b,b)`, unlike the
  additive correction. The final gate remains active as a defense.
- **Placement:** the cap is after the final restandardization; earlier placement
  would not guarantee support after a later affine restandardization.
- **Derivative:** the cap tangent is explicit and covered by FP64 autodiff and
  finite-difference tests before GPU use.
- **Fairness:** common random numbers, sealed observations, strict teachers,
  and frozen controls are shared across arms.
- **Hidden target change:** the cap changes third/fourth moments. The artifact
  must label the route heuristic and report post-cap residuals.
- **Continuation:** no validation or T20 run occurs if calibration has no valid
  arm. Existing validation teachers remain untouched until then.

## Pre-mortem

- A cap could make the scalar finite while its JVP is wrong. Prevent this with
  an independent FP64 ForwardAccumulator test and same-program GPU FD.
- A cap could guarantee support but distort nearly every coordinate. Record the
  changed fraction and displacement; selection minimizes displacement among
  valid arms.
- A cap could pass at T2 and fail after nonlinear resets. The result authorizes
  only the stated T2 teacher-sensitivity check, not T20.
- The radial and coordinate caps could interact non-monotonically. All eight
  combinations are evaluated on common calibration particles.
- A clipped defensive inverse could hide a boundary failure. The strict
  `|u|<1` gate remains a hard veto; defensive clipping cannot promote a row.

## Execution Budget

1. Add the coordinatewise cap and diagnostics without changing defaults.
2. Add focused FP64/TF32-off tests and run the affected suite CPU-hidden.
3. Run an escalated GPU/XLA calibration ladder for 8 arms plus baseline.
4. If at least one arm passes, freeze the least-distorting arm and run the
   untouched 3-teacher x 6-particle validation plus FD checks.
5. Write a result note and reset memo, rerun affected tests and `git diff --check`.

Budget: one focused implementation retry and one GPU campaign with one
unchanged-contract infrastructure retry; no new teacher generation is needed.
