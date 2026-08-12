# Zhao-Cui Austria SIR Lane-B T1 Score Plan

Date: 2026-07-31

Status: `REVIEWED_ACTIVE_EXECUTION`

Parent value result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t1-result-2026-07-31.md`.

Parameter-child mechanics result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-parameter-child-zero-slice-result-2026-07-31.md`.

## Scope And Verdict Before Execution

This phase asks only whether a compact origin tangent of the admitted T1
squared-TT parent can represent the analytical derivative of the same Austria
SIR T1 target well enough that the exact child-normalizer derivative agrees
with an independent observed-data score estimate. It does not open T2 until
T1 passes, and it does not authorize HMC.

The pre-execution audit rejected fitting `d sqrt(q_1)/d theta` without the
defensive density. The admitted finite density is `h_0^2 + tau lambda`, not
`h_0^2`. The refreshed program therefore fits the derivative of the full
unnormalized log density while freezing `tau` and `lambda`.

Audit verdict: `PASS_FOR_T1_EXECUTION`.

## Pilot-01 Objective Refresh

The first two 96-step arms exposed a material missing condition in the initial
objective. The higher-LR arm fitted the target log-score mean on validation
closely, but its exact contracted child-normalizer score remained materially
different. Pointwise fitting of `D rho/rho` to `D q/q` does not imply equality
of their expectations because the admitted parent `rho` is a finite TT
approximation to the shifted target `q`, not the target itself.

This is an objective-design failure, not a failure of the analytical local
score, manual child contraction, parent value, or tangent capacity. The
remaining uncalibrated arms are stopped. The refreshed candidate retains the
same parent, data roles, tangent shape fit, L1 grid, value, and claim gate, then
performs a training-only exact normalizer-score calibration.

Let `H=int h_0^2 dnu`, `Z=H+tau`, and let `s_child` be the exact contracted
score after shape fitting. Adding `alpha_a` times one parent core to tangent
coordinate `a` adds `alpha_a h_0` to `D_a h`. Therefore it shifts the exact
normalizer score by `2 alpha_a H/Z`. With the training Fisher estimate
`s_train`, set

\[
  \alpha_a=(s_{train,a}-s_{child,a})\frac{Z}{2H}.
\]

This is a repository-owned analytic gauge calibration, not optimizer
backpropagation and not claim-data fitting. Its coefficient, pre/post scores,
training Fisher estimate, and calibrated tangent tensors are artifact-bound.
The validation and untouched gates are unchanged. If calibration causes poor
validation point-score behavior or untouched disagreement, the candidate
fails. Refreshed skeptical audit verdict: `PASS_FOR_CALIBRATED_PILOTS`.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can three compact tangent-core fields attached to the admitted T1 parent reproduce the analytical complete-data target score and yield a T1 normalizer score consistent with independent same-target Monte Carlo? |
| Candidate | Parent-immutable external-theta linear core child, trained only at `theta=0` by the full defensive-density log-score residual. |
| Exact baseline | T1 identity `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`, selected arm `p05_r4_b5_lr3e4_l1_1e9`, value `-31.1290512231882`. This is also the parent lineage of the admitted T2 artifact. |
| Expected failure | Analytical local score is wrong, tangent capacity is inadequate, gauge directions destabilize training, the defensive term is omitted, parent cores mutate, data roles leak, score MCSE is too large, or memory exceeds budget. |
| Promotion criterion | Exact origin value/cores; manual child derivative parity; finite validation score residual; and untouched child T1 increment score within the predeclared independent Fisher-score interval coordinatewise. |
| Promotion veto | Any parent mutation, changed frame/shift/tau/defensive density, theta integration, target-score autodiff used as training authority, runtime autodiff/finite difference score, role overlap, nonfinite result, post-claim tolerance change, or memory breach. |
| Continuation veto | The analytical target score is invalid, the declared finite child derivative cannot be computed manually, or no finite tangent fit can pass after the bounded rank-preserving tuning arms. A failed arm alone is a repair trigger. |
| Repair trigger | Local XLA, optimizer, serialization, scale, or tuning failure under the unchanged parent and objective. |
| Explanatory diagnostics | Per-coordinate weighted RMS, correlation, score scale, tangent norm, gradient norm, runtime, MCSE, and allocator peak. |
| Must not be concluded | No T2/T5/T10/T20 score, exact nonlinear likelihood theorem, source-faithful parameter algorithm, HMC/posterior readiness, production readiness, or cross-method superiority. |

## Mathematical Target

For physical `z=(z_1,z_0)`, the exact declared T1 target is

\[
 q_1(z;\theta)=p_0(z_0)f_\theta(z_1\mid z_0)g_\theta(y_1\mid z_1).
\]

The initial-density parameter score is zero. The analytical complete-data
score used as the training authority is

\[
 s_q(z)=\left.\nabla_\theta\log f_\theta(z_1\mid z_0)
       +\nabla_\theta\log g_\theta(y_1\mid z_1)\right|_{\theta=0}.
\]

The admitted parent in fixed local coordinates `r` is

\[
 \rho_0(r)=h_0(r)^2+\tau\lambda(r),\qquad
 Z_0=\int\rho_0(r)d\nu(r).
\]

For parameter coordinate `a`, train one tangent-core field whose TT product
rule gives `d_a h(r)`. Parent cores are constants. Because `tau`, `lambda`, the
frame, and the shift are frozen, the full origin derivative of the
unnormalized child log density is

\[
 s_{\rho,a}(r)=\frac{2h_0(r)d_a h(r)}
                    {h_0(r)^2+\tau\lambda(r)}.
\]

The fit objective uses self-normalized T1 target weights on independent
proposal draws `z_i ~ p_0 f_0`:

\[
 \mathcal J=\sum_i \bar w_i
   \sum_{a=1}^3\frac{(s_{\rho,a}(r_i)-s_{q,a}(z_i))^2}{b_a^2}
   +\gamma_1\sum|D|+\gamma_2\sum D^2,
 \quad \bar w_i\propto g_0(y_1\mid z_{1,i}).
\]

The coordinate scales `b_a` are RMS values computed from training data only.
They are frozen before validation. L1 is tuned as required by the Zhao-Cui
training policy; zero L1 may appear only as an explicit comparator.

The child increment and exact manual score are

\[
 \ell_1^{child}(\theta)=\log Z^{child}_1(\theta)-c_1,
 \qquad
 \left.D_a\ell_1^{child}\right|_0=
 \frac{\int 2h_0d_a h\,d\nu}{Z_0}.
\]

No derivative of a fitting algorithm is claimed. The frozen trained tangent
fields define the finite child program, and the contraction above is the total
derivative of that declared T1 child scalar.

## Independent Score Comparator

On a disjoint cloud `z_i ~ p_0 f_0`, the Austria T1 observed-data score is
estimated by Fisher's identity:

\[
 \widehat s_{MC}=\frac{\sum_i w_i s_q(z_i)}{\sum_i w_i},
 \qquad w_i=g_0(y_1\mid z_{1,i}).
\]

Its delta-method standard error uses iid influence rows
`w_i(s_q(z_i)-s_hat)/mean(w)`. This Monte Carlo estimator is an independent
comparator and claim gate; it is not substituted into the child runtime.
The untouched coordinatewise gate is

\[
 |s^{child}_a-\widehat s_{MC,a}|\le 3\,MCSE_a+10^{-5}.
\]

The calibration run must show that this interval is informative before the
untouched cloud is generated. If any MCSE is too large to discriminate the
candidate, the run is underpowered and must not pass by widening the gate.

## Source And Classification Ledger

| Operation | Classification | Anchor |
|---|---|---|
| Sequential joint target and event order | `source_faithful` operation in the local extension | Zhao-Cui Eq. (15), Algorithm 2(a), paper lines 693-719; author `models/full_sol.m:72-80,132-135` |
| Squared-TT plus defensive density and state marginalization | `source_faithful` operation with local fitted values | Zhao-Cui Eq. (16), Algorithm 2(b-c), paper lines 703-722; author `@TTSIRT/marginalise.m:19-85` |
| Increment `log Z-c` | `source_faithful` operation | author `models/full_sol.m:84-124` |
| Austria RK4 transition and Gaussian observation | `source_faithful` model operation | author `sir_austria/sir_step.mlx`; `transition.mlx`; `like.mlx` |
| Frozen external theta, tangent cores, score regression, and manual child derivative | `extension_or_invention` | Project derivation above; not present in the inspected author parameter-estimation route |
| Frozen parent identity and stateless role seeds | `fixed_hmc_adaptation` | Deterministic adaptation of author randomness; no HMC authorization |

## Evidence Contract

| Field | Contract |
|---|---|
| Engineering/scientific question | Does the declared T1 parameter child compute a correct manual derivative of its own finite value and agree with an independent Austria T1 observed-data score estimate? |
| Baseline/comparator | Admitted T1 parent plus independent Fisher identity under the exact latent-preclip T1 target. No UKF, SGQF, GenUT, APF, source replica, or retained grid. |
| Primary criterion | Untouched coordinatewise score interval after exact origin and manual-derivative gates. |
| Hard vetoes | Identity/source drift, role leakage, wrong local score, changed finite scalar, parent mutation, nonfinite values, failed manual/tape diagnostic, memory cap, or underpowered comparator. |
| Explanatory only | Training loss, validation RMS/correlation, descriptive arm differences, tangent norms, runtime, and acceptance-like fit summaries. |
| Nonclaims | Passing T1 does not establish later-time recursion, broad likelihood accuracy, HMC, production, or superiority. |
| Artifact | Unique directories under `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260731/`, plus a result note and exact run manifests. |

## Data Roles, Defaults, And Assumptions

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Parent `p05` rather than descriptively selected `p06` | Required by admitted T2 lineage; baseline | training against a parent that cannot extend to T2 | exact identity assertion before any batch generation |
| Fresh train/validation/calibration/untouched seeds | New phase; reviewed default | reuse or leakage understates error | bind role, seed, point/log-weight/score hashes; require disjoint seeds |
| Rank-preserving three tangent banks | Zero-slice mechanics; hypothesis | inadequate parameter capacity | validation residual and Fisher gate; failure does not mutate parent rank |
| Training-only RMS coordinate scaling | Convenience hypothesis | hides a poorly fit coordinate | report raw per-coordinate validation metrics and comparator residuals |
| L1 grid including zero comparator | Zhao-Cui lane policy | gauge directions or over-shrinkage | tangent norms, validation residuals, origin identity; select before claim |
| FP64 CPU algebra then GPU/XLA training | Repository backend policy | backend mismatch | focused CPU parity, then exact frozen-child CPU/GPU score tie-out |
| Training `8192`, validation `16384`, calibration `32768`, untouched `65536` | Existing T1 value ladder; hypothesis | MCSE too large or compute waste | calibration MCSE and wall-time check before untouched run |
| `6 GiB` GPU allocator cap | Existing Lane-B cap | hidden sample-by-parameter or dense-grid allocation | static tensor estimate and TensorFlow allocator peak |

Training, validation, calibration, and untouched audit are pairwise disjoint.
The audit cloud is final-only and cannot repair or select a tangent arm.

## Skeptical Plan Audit

| Required challenge | Finding |
|---|---|
| Wrong baseline | Bound to the admitted T1 identity used by T2, not historical P88, APF, source replica, generic retained grid, or the descriptively selected incompatible T1 arm. |
| Proxy promotion | Fit residual and manual/tape parity cannot promote. The primary gate is an untouched same-target observed-data score interval. |
| Missing stop conditions | Mathematical/identity, data leakage, underpowered MC, nonfinite, memory, and bounded-arm exhaustion stops are explicit. |
| Unfair comparison | No cross-method ranking occurs. The independent Monte Carlo score evaluates the same T1 target. |
| Hidden assumptions | Frozen frame/shift/tau/defensive density, zero shift derivative, target-weighted fit measure, and training-only coordinate scales are explicit. |
| Stale context | T1/T2 values and zero-slice mechanics are admitted; synthetic tangents are explicitly not score evidence. |
| Environment mismatch | CPU-only checks hide CUDA before TensorFlow import. Serious training is trusted GPU/XLA with verified memory growth. |
| Non-answering artifact | Saved tangents, exact hashes, raw score metrics, independent MCSE, identity, source closure, command, environment, wall time, and allocator peak answer the declared T1 question. |

## Execution Ladder And Budget

1. Implement analytical T1 score batches, manual origin contractions, tangent
   trainer, serialization, and focused tests.
2. CPU-hidden algebra smoke: local analytical score versus diagnostic
   `GradientTape`; manual child contraction versus diagnostic tape; exact parent
   immutability and origin value.
3. Trusted GPU/XLA pilot arms with fixed parent rank and a bounded L1/LR grid.
   Selection uses validation only. Descriptive differences do not support a
   statistical ranking.
4. One disjoint calibration Fisher comparison. Freeze the selected child and
   confirm the claim interval is informative.
5. One untouched score claim and one fresh reload/tie-out. Never overwrite a
   prior attempt.
6. Only after `PASS_LANE_B_T1_VALUE_AND_TOTAL_SCORE`, review the T2 recursion.

Budget: at most four focused CPU launches, six GPU pilot arms, one calibration,
one untouched claim, one replay, 75 minutes of additional wall time, 6 GiB GPU
allocator peak, and 12 GiB CPU process peak. A localized harness failure may be
repaired within this budget under a fresh output directory.

## Pre-Mortem And Nonclaims

The run could appear to pass if validation and audit reuse the same cloud, if
the target-score regression drops `tau lambda`, if one high-variance score
coordinate makes the interval vacuous, or if manual parity proves only random
tangent algebra. The role ledger, full-density formula, calibration
informativeness check, and independent untouched comparison address those
risks.

A failed tangent arm is tuning/capacity evidence, not evidence that the
analytical target score, T1 parent value, or Zhao-Cui direction is wrong. A
failure of the analytical local-score diagnostic or same-child derivative is
an implementation veto. No HMC will run in this phase.
