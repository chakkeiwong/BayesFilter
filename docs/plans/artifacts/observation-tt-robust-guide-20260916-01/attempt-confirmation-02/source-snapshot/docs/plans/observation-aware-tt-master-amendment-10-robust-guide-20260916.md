# A10: stable TT coordinates and physical proposal defense

Status: reviewed protocol; implementation and execution follow this amendment.
Owner instruction, 2026-09-16: document the proposed protection thoroughly in
LaTeX, in proposition/proof form, audit it with MathDevMCP, then plan, review
and execute. This explicitly authorizes the proposed extension. Start charged
interval: 2026-09-16 12:48:00 UTC. Branch surrogate-hmc; initial commit
50f93709d8a4c0b9481d49b07ec5373b73b23473. Preserve unrelated changes.
Governing master: observation-aware-tt-repair-complete-program-20260913.md.

## Research question and implementation boundary

Can an observation-adapted, positive quadrature repair, model-based TT chart,
and transition-density defense remove A09's covariance-collapse mechanism
while retaining useful exact-weight filtering? This is an owner-authorized
extension_or_invention, not a new claim of Zhao--Cui source faithfulness.
Keep the exact SV model, conditional particle weighting and resampling target.
No covariance clipping/ridge, no SGQF-vs-TT fit-loss veto, no HMC/default claim.

Separate three mathematical objects: the Gaussian approximation to the
posterior; the invertible coordinates in which TT is fitted; and the physical
proposal sampled by particles. A failed first object must not collapse the
other two. The main limitation to resolve is numerical coverage, not whether
TT approximates the analytic SGQF Gaussian more closely than SGQF itself.

## Manuscript and proof obligations

Revise the existing algorithm note at
artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex.
Preserve the full baseline and checksum under the A10 artifact root. Add one
continuous section before the appendix, retaining existing derivations. Reader:
a researcher familiar with Gaussian filtering, importance sampling and TT.
Argument: observed collapse -> failure of scale-free SPD -> stable coordinates
-> resolved guide quadrature -> physical mixture -> qualified smoothness.

Prove: (1) SPD conditioning does not control scale; (2) signed cancellation
and positive-weight concentration have different failure mechanisms; (3) the
SV Gaussian-predictive posterior has a unique mode with positive Hessian;
(4) change-of-measure positive Gauss--Hermite reweighting and its limits;
(5) model-based charts remain nonsingular independently of guide covariance;
(6) a transition mixture is normalized, importance-correct, and supplies a
finite conditional second-moment bound for this SV likelihood; (7) smooth
mixture density under explicit assumptions does not prove smooth finite-particle
resampling, L1 training, or adaptive quadrature selection. No finite ESS lower
bound, posterior-accuracy certificate, or global derivative claim is implied.
Use MathDevMCP on the actual new labeled propositions/equations; retain outputs,
repair mathematical findings and report unresolved backend limitations honestly.
Compile and visually inspect the new pages. Human prose acceptance is pending.

## Mechanisms and defaults audit

* Guide guard: compute moments in predictive standardized coordinates. Positive
  finite signed mass, a roundoff/cancellation budget and an independent scale
  margin can reject unresolved rules. Record signed cancellation, dominant
  absolute weight, covariance spectrum in predictive units, and higher moments.
  The numerical guard is a rejection check, not a posterior approximation theorem.
  Roundoff budget uses float64 epsilon, number of nodes and absolute summands;
  a 64-fold margin is a conservative arithmetic hypothesis, tested on healthy
  and collapsed fixtures. No hidden covariance floor.
* Accuracy check: compare successive accepted rules' means, covariances and
  log integrals. Budgets .02 predictive SD, .05 relative predictive covariance
  Frobenius scale and .02 log integral are explicit diagnostic hypotheses. They
  are chosen below the downstream .15 log-evidence slack per sequence, but
  do not imply a cumulative error bound. Record all discrepancies and failures.
  Large higher moments are explanatory, not a universal hard threshold: true
  non-Gaussian distributions may have arbitrarily large moments.
* Repair: if SGQF resolution fails, solve the strictly convex SV Gaussian-prior
  negative log posterior by damped Newton with analytical gradient/Hessian and
  backtracking. No ridge is needed for the mathematical Hessian; reject invalid
  numerical factorizations. Positive tensor Gauss--Hermite in the Laplace chart
  includes the exact predictive-density/proposal-density ratio. Orders 5,7,9
  (at most 9^4 nodes) compare resolution. This is deliberately d<=4 diagnostic
  work, not a scalable tensor-product production route or replacement for the
  Zhao--Cui source route. Unresolved orders record failure and use the predictive
  Gaussian as a labeled guide fallback, preserving a well-defined proposal.
* TT chart: center at the accepted/repaired guide mean but use model covariance
  R0=P0, Rt=A Rt-1 A' + Q independently of observation reweighting. With the
  stationary fixture this is P0 throughout. This deliberately sacrifices some
  concentration for scale protection; whether degree/rank suffice is empirical.
  The SGQF joint initializer and weighted regression rows must be transformed
  into these charts without changing their physical distributions.
* Defense: q=(1-epsilon)qTT+epsilon f(x|z), or prior at t0. Always evaluate both
  physical component densities at the actual sampled point. Internal TT defense
  1e-5 remains solely for polynomial-zero protection. External epsilon candidates
  .05,.2,.5 imply worst conditional second-moment inflation bounds 20,5,2 relative
  to transition importance sampling. Select the smallest satisfying calibration
  non-harm/validity screens; do not maximize primary MSE to pick epsilon. These
  are explicit cost/robustness hypotheses, not universal defaults.
* Fitting: A09 frozen scalar degree4/4096/L1=.001, d4 degree4/1024/L1=.001,
  rank3/four sweeps/128 proximal steps, zero-centered L1 are warm starts only.
  Include local L1 {1e-5,.001} calibration because the coordinate scope changes.
  Choose L1 by calibration filtering MSE subject to valid coverage; freeze it
  before confirmation. This is bounded diagnostic tuning, not final rank search.
* Precision/backend: TF float64; GPU/XLA for repeated numerical kernels;
  memory growth before initialization. CPU eigendecomposition/TT-SVD and host
  rule setup are explicit small reference/setup exceptions. No NumPy runtime,
  pfor, new packages, external release, or HMC execution.

## Evidence contract and intent ledger

| Role | Definition |
|---|---|
| Primary question | Eliminate coordinate/coverage collapse while retaining useful downstream filtering |
| Baselines | A09 standalone degree4 warm TT; repaired-guide TT with its own covariance; stable-chart TT; full stable-chart/transition-defense TT |
| Cheap adversaries | Transition (model dynamics), stationary prior (stable broad scale), SGQF marginal (cheap observation response), SGQF joint (cheap temporal dependence), each in the same exact-weight consumer |
| Situations | d1/d4; near-zero, ordinary and large observations using inherited fixed boundaries; exposed collapse and indefinite cases are debugging only |
| Primary safety acceptance | Reject the exposed collapsed guide; exact mixture density/sampling identities; finite charts/consumers; no silent fallback; healthy accepted guide moments unchanged to roundoff |
| Downstream promotion | Complete fresh reference-valid coverage; log-evidence agreement; no conditional heuristic veto; paired MSE non-harm bound before suggesting an optional repaired route |
| Promotion veto | CDF/domain/nonfinite/target mismatch; reference imprecision; incomplete coverage; log-evidence disagreement; observed conditional heuristic loss (descriptive veto only) |
| Continuation veto | Broken target/importance math, corrupted artifacts, source drift during runs, unresolved required numerical checks, exhausted budget |
| Repair trigger | Guide rejection, optimizer resolution failure, low ESS, high KKT; preserve failures and continue planned isolation/repair within caps |
| Explanatory only | TT loss/Hellinger/KKT, chart spectra, quadrature diagnostics, ESS/max weights, timing; none alone proves quality |
| Forbidden conclusion | Universal SGQF reliability, positive cubature exactness, finite-particle unbiased means, minimum ESS guarantee, global smoothness, HMC/default readiness, scalable high-d success |

The conditional heuristic screen is a falsification instrument, never a tuning
target. For non-harm of the safety change, the primary paired contrast is final
TT minus A09 TT mean normalized filtering MSE, with a predeclared 10% relative
margin on sequences where both are valid; report absolute differences too.
This margin is an explicit owner-facing cost-of-safety hypothesis, not an
equivalence theorem. Do not discard failure cases to claim overall non-harm.
With few sequences, report paired bootstrap intervals as exploratory and all
continuous differences descriptively unless intervals support the stated claim.

## Stages, seeds and bounded commands

1. Finish manuscript/proofs, skeptical plan review, MathDev audit and build.
2. Implement optional library components and endpoint wiring. CPU tests prove
   linear-Gaussian moments, collapsed-node rejection, log-mixture evaluation,
   sampled-component density at both branches, normalizer/second-moment bound,
   positive quadrature change of measure, healthy-case no-fire and call chain.
3. GPU smoke: old exposed d4-s04/s10 cases only as mechanism diagnostics; short
   healthy d1/d4. Check XLA, memory growth, saved diagnostics and CDF failures.
4. Fresh calibration: three T20 sequences per dimension, N512, four repetitions;
   compare L1 candidates and epsilon curve under above roles. Freeze controls.
5. Fresh confirmation: twelve T20 sequences per dimension, N512, four repetitions;
   reference scalar nested grids, d4 bootstrap reference 32768/65536/131072 with
   inherited .02 mean-MCSE/.10 log-MCSE and resolution screens. Failures remain.
   If a baseline guide fails, preserve it and run protected candidates; report
   coverage separately and paired metrics only on explicitly matched sets.
6. Terminal integrity/math-to-code audit, result/inference-status tables,
   manuscript numerical update, master/checkpoint/budget closeout.

New seed sequence blocks start at 1000000000 + 10000000*k; k=0..5 calibration,
24..47 confirmation, dimension groups explicitly disjoint. Within each block
fit +2000000 (validation +100000, audit +200000), particles +3000000+1000*r+t,
reference +4000000+1000000*level+1000*r+t. Seeds remain int32-safe. Common random
numbers across methods are deliberate; no sequence/repetition/time collisions.

Driver: docs/benchmarks/run_observation_tt_robust_guide.py, stages smoke,
calibration, confirmation; --output-root must be a fresh attempt directory;
confirmation requires --calibration-root. Commands/env are saved in manifests.
Numerical root: docs/benchmarks/artifacts/observation_tt_robust_guide_20260916/.
Plan/review/math/build/test artifacts: docs/plans/artifacts/observation-tt-robust-guide-20260916-01/.
Result: docs/plans/observation-aware-tt-robust-guide-20260916-result.md.

Budget: eight active hours (28800 s) from the remaining 41020.892609 s H11
balance, including writing/audits/engineering/closeout. Numerical cap 18000 s;
at most three smokes (600 s each), two calibrations (5400 s each), two
confirmations (5400 s each), subject to total cap. Local repairs may retry
without reapproval within unchanged scope. No outside paid compute or GPU
class expansion. Stop at campaign cap; record under-budgeted claims honestly.

## Skeptical plan review before implementation

Localized repair after confirmation-01 (all sources/results preserved): see
[amplitude-sign repair and review](artifacts/observation-tt-robust-guide-20260916-01/amplitude-sign-repair.md).
A negative least-squares amplitude scale was incorrectly rejected at one
stable-chart fit. Its sign is not a constraint on a squared TT density. The
repair retains finite/nonzero checks, exact positive-branch parity and all
scientific criteria. Use the remaining smoke/calibration/confirmation slots;
confirmation-02 uses fresh blocks 48--71. Original calibration blocks repeat
only for calibration and smoke-03 exposes the failed data/seed as diagnostics.
No tuning on confirmation-01 and no expansion of either campaign ceiling.

Implementation clarification before calibration, 2026-09-16: the epsilon
non-harm calibration compares the defended arm with the same stable-chart TT
without external defense. The final primary comparison remains against A09.
An L1 candidate must have all three reference-valid calibration sequences;
if none does, L1=.001 is a labeled diagnostic representative only. If no
epsilon passes, epsilon=.5 is likewise diagnostic only. Fresh confirmation
cannot erase failed calibration admission. Confirmation uses 9,999 paired
sequence bootstrap resamples, seed 109277777, with Bonferroni simultaneous
intervals across the six dimension-by-repair contrasts; other intervals are
exploratory. The initial analytic SGQF joint needs no TT fit: stable coordinates
apply to the subsequent TT fits, including their previous-state coordinate.
These are explicit reporting/selection rules, not changes to the target.

Pre-calibration review: PASS. GPU smoke-02 completed in 169.254 seconds,
including the two exposed A09 failures and healthy d1/d4 fixtures. Protected
arms had no nonfinite consumer steps or inverse-CDF bracket failures. Smoke
uses no accuracy reference and cannot support a filtering-quality claim.
Smoke-01 failed in one-repetition MCSE reporting; four repetitions repaired
that harness error. It consumes a conservative 150-second budget reservation.

PASS for this diagnostic extension after correcting the earlier informal
proposal: finite diagnostics are not an accuracy certificate; signed weights
are not probabilities; positive weights do not prevent node collapse; small
posterior covariance can be real; a transition mixture gives a conditional
second-moment bound, not guaranteed finite-sample ESS; continuous density does
not establish smooth resampling/training. Baseline coverage, seed independence,
new-chart L1 tuning, actual mixture-density wiring and fresh references are
explicit. The exposed A09 failures are debugging fixtures, not holdouts.
Main residual risks: d4 tensor-quadrature cost, rank3/degree4 on broader charts,
reference imprecision, and a bounded epsilon curve that may not identify a
useful setting. None is silently promoted to a default.
