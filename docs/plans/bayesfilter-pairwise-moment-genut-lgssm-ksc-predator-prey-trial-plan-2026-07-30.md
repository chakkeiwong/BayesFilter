# Pairwise-Moment GenUT Cross-Model Trial Plan

Date: 2026-07-30  
Status: `AUTHORIZED_BOUNDED_FEASIBILITY_TRIAL`

## Research Intent

Test whether the deterministic off-diagonal pairwise higher-moment correction
that repaired Austria-SIR recursive-score variance also helps the active
LGSSM and predator-prey targets, and establish its exact structural effect on
the scalar-state KSC-SV target.

The candidate matches, in whitened coordinates,

\[
E[z_i^2z_j],\quad i\ne j,
\qquad\text{and}\qquad
E[z_i^2z_j^2],\quad i<j,
\]

after the existing diagonal skewness/kurtosis correction. Every pairwise step
restandardizes the cloud, preserving the restored mean and full covariance up
to the finite-program numerical tolerance. The runtime score is the manual
recursive JVP of this same finite program.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Does pairwise matching reduce recursive-score error or variance without an unacceptable value regression? |
| Candidate | Current non-fused GenUT/Contract-E route plus ordered pairwise co-skewness and unordered pairwise co-kurtosis correction |
| Primary baseline | Same finite route and same scope-specific July 23 controls with pairwise steps fixed to zero |
| Claim targets | Exact July 23 `lgssm_T50`, `ksc_sv_T10`, and `predator_prey_T20` observations, event order, theta chart, and hashes |
| Claim particles/seeds | `N=1008`, common seeds `98201..98216` |
| Tuning | LGSSM and predator-prey only; disjoint calibration/validation trajectories and seeds `98401,98402` |
| Exact references | LGSSM exact affine Kalman value/score; KSC-SV converged sequential dense transformed-mixture value and diagnostic centered-FD score |
| Approximate comparators | Fixed SGQF and fixed-variant Zhao-Cui, target-hash checked and diagnostic only |
| Promotion criterion | Hard numerical gates pass; score variance falls; value remains stable; oracle-backed score error improves where an exact/dense reference exists |
| Promotion veto | Invalid OT/reset/covariance/JVP, nonfinite result, score-increment mismatch, value SD above `1.25x` baseline, or unsupported oracle/comparator substitution |
| Repair trigger | A nonzero arm reduces pair residual but fails value or score behavior; preserve it and narrow the strength/step tradeoff in a later plan |
| Continuation veto | Target/hash/event-order mismatch, broken reference convergence, invalid baseline, or exhausted bounded grid |
| Explanatory diagnostics | Pair and diagonal residuals, paired value/score changes, per-coordinate SDs/CIs, runtime, allocator peak |
| Nonclaims | No broad nonlinear superiority, exact predator-prey score, HMC/default readiness, NAWM result, or universal control setting |

## Model-Specific Reference Contract

| Model | State dimension | Pairwise test | Accuracy reference |
| --- | ---: | --- | --- |
| LGSSM `T=50` | 3 | Full tuning and 16-seed claim | Exact Kalman value and analytical score |
| KSC-SV `T=10` | 1 | Structural-null 16-seed parity test | Sequential dense transformed-mixture value; converged centered-FD score is diagnostic reference only |
| Predator-prey `T=20` | 2 | Full tuning and 16-seed claim | None exact; SGQF and Zhao-Cui are descriptive diagnostics |

KSC-SV has no off-diagonal coordinate pairs. Therefore this pairwise family
cannot add distributional information in that row. The nonzero-control KSC
arm tests whether the implementation is numerically inert; it is not a tuning
or improvement candidate.

SGQF is exact for the affine LGSSM row but not for nonlinear SIR,
KSC-SV, or predator-prey. It is never used as the nonlinear truth anchor in
this trial.

## Controls And Selection

Each model inherits only its own July 23 diagonal/OT controls as a frozen
baseline. Cross-model transfer of Austria's selected pairwise controls is not
allowed. For LGSSM and predator-prey, append:

```text
pairwise_steps={0,1,2,4}
pairwise_strength={0.005,0.01,0.02,0.05}
pairwise_floor=1e-5
```

The zero-step arm is the baseline. Candidate selection is oracle-free and does
not read the claim observations, claim seeds, Kalman result, dense KSC result,
SGQF, or Zhao-Cui. Selection order is:

1. finite-program and reset/OT/score-additivity gates;
2. lower validation pairwise residual than the zero-step baseline;
3. no validation score-coordinate variance above baseline;
4. validation value SD no more than `1.25x` baseline;
5. lowest maximum and geometric-mean score-variance ratio;
6. lower displacement, strength, and step count.

If no nonzero arm passes, retain the baseline and report that pairwise matching
did not survive tuning. KSC-SV executes zero-step and a fixed nonzero control
only because every pair residual is structurally zero.

## Statistical Contract

Claim intervals are Student-t 95% intervals over 16 particle seeds. Paired
candidate-minus-baseline intervals use the common seeds. A seeded paired
bootstrap reports the geometric score-variance ratio.

For LGSSM and KSC-SV, per-seed absolute score error to the reference is also
reported. A lower mean error without a paired interval excluding zero is
descriptive only. For predator-prey, lower score variance is precision evidence
only and cannot establish score accuracy.

## Skeptical Plan Audit

| Risk | Audit result |
| --- | --- |
| Wrong baseline | Pass: exact July 23 target hashes, event orders, controls, `N`, and common seeds are loaded and checked |
| Proxy promoted to truth | Pass: moment residual is a tuning prerequisite only; SGQF/Zhao-Cui remain diagnostics |
| Hidden oracle leakage | Pass: tuning uses fresh trajectories/seeds and never reads claim/reference values |
| KSC category error | Repaired: `d=1` is explicitly a structural null, not a pairwise candidate |
| Missing stop conditions | Pass: target/reference/baseline invalidity and bounded-grid exhaustion are explicit |
| Unfair comparison | Pass: candidate and baseline differ only in pairwise controls and use identical particle noise per seed |
| Environment mismatch | Pass: serious run requires TensorFlow FP32, TF32, GPU, XLA, and verified memory growth |
| Misleading success | Controlled: variance reduction is separated from score accuracy and value no-regression |

Audit decision: `PASS_AFTER_KSC_NULL_AND_REFERENCE_HIERARCHY_REPAIR`.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| July diagonal/OT controls | Scope-specific July 23 tuning artifact | frozen baseline | stale target or source drift | hash/control equality check |
| Pairwise grid | Austria feasibility grid | target-specific hypothesis grid, not default | boundary selection or missed useful scale | report full validation curve |
| Two tuning seeds | Prior bounded feasibility protocol | convenience choice | noisy variance selection | untouched 16-seed claim |
| `N=1008` | Active leaderboard scope and `2d` divisibility | comparison scope | remaining Monte Carlo error | 16-seed intervals |
| KSC dense orders `401,601` | July KSC admission reference | diagnostic reference | under-resolved grid/FD | order and FD-step gaps |

## Execution And Budget

1. Add a standalone cross-model runner without changing runtime defaults.
2. Run Python compilation and focused existing pairwise tests CPU-only.
3. Run a one-seed trusted GPU/XLA smoke for all three rows.
4. Tune LGSSM and predator-prey on the bounded 13-arm grid.
5. Run baseline/candidate claims on 16 common seeds for all rows.
6. Compute exact/dense references, paired uncertainty, and result artifacts.

Budget: one focused test phase, one smoke, at most 13 tuning arms for each of
two models, 32 claim evaluations per model, and one bounded dense KSC reference.
All outputs use fresh versioned directories under
`docs/benchmarks/artifacts/pairwise_moment_genut_cross_model_20260730/`.

