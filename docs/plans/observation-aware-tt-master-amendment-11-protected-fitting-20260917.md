# A11: protected coordinates and better pair-TT fitting

Owner authorization: 2026-09-17 request to derive the proposed improvement,
audit with MathDevMCP, build the manuscript, review a plan, and execute it.
This is the next phase of the [master](observation-aware-tt-repair-complete-program-20260913.md).
The A10 physical-transition mixture remains the comparator and protection.
Status: **plan reviewed; mathematics and implementation in progress**.

## Question and research intent

Can observation-adapted coordinates with an independent covariance floor,
more training rows, and additional TT capacity reduce the actual filtering
error of the protected A10 method without reviving covariance or importance
weight collapse? The target is the same stochastic-volatility filter as A10,
with exact importance weights. This is an owner-authorized extension, not a
claim to reproduce Zhao-Cui's TT-cross implementation. No HMC consumer changes.

For the checked guide covariance P_t and the independent recursion
R_0=P_0, R_t=A R_(t-1) A' + Q, use
C_t(lambda)=(1-lambda)P_t+lambda R_t, lambda in (0,1].
Thus C_t >= lambda R_t. Retain q_epsilon=(1-epsilon)q_TT+epsilon f,
so q_epsilon >= epsilon f and the implemented proposal has its exact density.
The TT is fitted separately at each observation time and retains its previous
approximation. Better guide coordinates do not remove that accumulated error.

| Role | Predeclared meaning |
|---|---|
| Primary criterion | Independent-reference normalized filtering mean MSE, averaged first within a sequence, with paired uncertainty across fresh sequences |
| Promotion | Reference-complete panel; upper simultaneous paired interval for relative MSE change below zero for improvement, or <=10% for non-harm; all validity and conditional heuristic screens pass |
| Promotion veto | Invalid covariance/CDF/density, nonfinite weights, failed log-evidence screen, unresolved reference, or conditional loss to a cheap heuristic in its observed regime |
| Continuation veto | Wrong target/weight density, broken seed separation, corrupted evidence, exhausted budget, or failed mathematical assumptions |
| Repair trigger | A current candidate fails accuracy, optimization, coverage, or numerical checks; proceed to the planned ablation or localized repair |
| Explanatory diagnostics | Heldout fit error, coefficient truncation/compression error, KKT residual, ESS, maximum weight, resampling, guide fallbacks, runtime and memory |
| Must not conclude | A fit-loss improvement certifies filtering; an ESS average supplies a uniform ESS floor; better measured MSE proves total gradients, source-faithfulness, broad superiority, or default readiness |

The heuristic set is constructed from the problem: transition bootstrap
(correct dynamics), stationary prior (robust state coverage), repaired SGQF
marginal (cheap observation adaptation), repaired SGQF joint conditional
(cheap dependence adaptation). Evaluate each in the same near-zero,
ordinary, and large standardized-observation regimes used by A10, separately
by dimension. This falsification panel is not used to tune TT controls.
Its descriptive losses conservatively veto promotion without establishing a
statistical ranking. The principal comparator is A10 full protection, not
an SGQF regression-loss threshold.

## Defaults and bounded fitting hypotheses

Baseline: A10 full protection: lambda=1, physical epsilon=.05, internal
defense=1e-5, degree=4, rank=3, sweeps=4, proximal steps=128, L1=.001,
rows=4096 (d1) or 1024 (d4). Initial t=0 proposal remains analytic SGQF.
All candidates use the same repaired guide and underlying pair fitter.

Calibration profiles, each with L1 in {1e-5,.001}:

| Profile | lambda | rows d1/d4 | degree | rank | sweeps |
|---|---:|---:|---:|---:|---:|
| half | .5 | 4096/1024 | 4 | 3 | 4 |
| quarter | .25 | 4096/1024 | 4 | 3 | 4 |
| rows | .25 | 4096/4096 | 4 | 3 | 4 |
| sweeps | .25 | 4096/4096 | 4 | 3 | 8 |
| rank | .25 | 4096/4096 | 4 | 5 | 8 |
| degree | .25 | 4096/4096 | 5 | 3 | 8 |
| combined | .25 | 4096/4096 | 5 | 5 | 8 |
| broad-capacity | 1 | 4096/4096 | 4 | 5 | 8 |

Deduplicate identical d1 configurations: boundary ranks are one, so rank does
not increase scalar capacity; rows=4096 is already the scalar baseline.
Dense initialization is bounded at (5+1)^8=1,679,616 coefficients.
No adaptive rank, per-observation tuning, or audit-based selection.

The lambda grid has an interpretable minimum standard deviation of 1,
sqrt(.5), or .5 in R-whitened coordinates. It is a hypothesis, not a universal
default. Failure: broad coordinates leave structure unresolved or smaller
coordinates lose useful tail resolution. Check generalized covariance
eigenvalues and training/validation coverage first. A convex blend need not
shrink every direction, since P_t need not be below R_t.

Rows and sweeps are geometric budget increments addressing coverage and
optimization separately. Rank/degree increments address representation;
nested model classes improve only the exact unregularized optimum, not a
finite L1-regularized fit. Inspect projection errors, KKT and heldout errors
before interpreting downstream differences. L1 choices extend A10's tuned
grid to each new scope and are reselected on calibration only. Row mixture
epsilon=.2 is the inherited bounded-density sampler, with importance ratio
<=5; changing it is deferred until coverage evidence requires another plan.
The positive internal defense, fixed proximal budget, resampling threshold
.5N, float64 and N=512 are comparator settings, with observed density, KKT,
CDF and low-ESS diagnostics preserving their failure risks.

For each dimension, nominate the lowest calibration-MSE valid profile at
epsilon=.05. Break exact ties by lower degree, rank, sweeps, rows, then name.
Assess epsilon in {.05,.1,.2} for that frozen fitted path. Choose the largest
epsilon with <=10% calibration-MSE inflation relative to its .05 arm and no
increase in the fraction of steps below ESS=.05N. This is a safety non-harm
curve: a larger epsilon strengthens the density bound, not necessarily ESS.
If neither stronger value passes, retain .05. Do not tune against heuristics.
Confirmation includes the nominated fit with .05 and with selected epsilon;
deduplicate if identical. A10 baseline is always retained.

## Evidence and execution stages

1. Add propositions/proofs to the protected manuscript; MathDevMCP equation
   audit and derivation-tree review; address substantive findings and record
   coverage limits. Build PDF and inspect rendered new pages.
2. Implement an optional covariance-blend chart constructor and focused tests:
   lambda=1 parity, spectral bound, invalid lambda/input rejection,
   covariance-collapse fixture, Cholesky derivative parity, and actual filter
   wiring. Existing A10 regression tests remain required.
3. GPU/XLA smoke, d1/d4, T=4, N=512, representative maximum-capacity and
   baseline paths. Validate exact density, CDF, coordinate bounds, and a
   stronger d4 reference before the research ladder.
4. Calibration: 6 fresh sequences per dimension, T=20, four particle
   replications. Save per-step cores/diagnostics and fit once per profile;
   reuse the path for physical-mixture curves. Freeze controls and code hashes.
5. Confirmation: 12 fresh sequences per dimension, T=20, four particle
   replications; A10 baseline, up to two frozen nominees, all four heuristics.
   Do not select controls or repair fitting on confirmation evidence.
6. Terminal self-review, inference/decision tables, complete run manifests,
   manuscript results and clean build, master and checkpoint closeout.

Reference: scalar quadrature unchanged (801/1201 agreement <=1e-6).
d4 uses 8 independent bootstrap reference replicates at N=65536,131072,
262144,524288, stopping at the first consecutive-level precision pass.
Require maximum normalized mean MCSE<=.01 and logZ MCSE<=.05; retain A10's
bias-resolution tolerances .01 for means and .10 for logZ, using Student
t_7=.975 quantile 2.364624251. Save every reference replication. If the
maximum level fails, retain all filter diagnostics but exclude that sequence
from every admitted accuracy comparison and withhold full-panel ranking.
Do not relabel a point estimate as exact merely because N is large.

Paired sequence bootstrap: 9999 resamples, fixed seed 1172026; statistic
(mean MSE_candidate - mean MSE_baseline)/mean MSE_baseline. Report simultaneous
95% percentile intervals using Bonferroni over four declared contrasts
(two nominees by two dimensions), even when a duplicate arm reduces the
realized number. Missing references block full-panel inference. Conditional
regime results, ESS tails and runtime are descriptive, not ranking evidence.
LogZ validation uses .15 + t_3 * sqrt(MCSE_candidate^2+MCSE_reference^2),
t_3=3.182446305, as in A10; guide fallback is flagged, not silently accepted
as an accurate SGQF update.

Fresh stateless seed namespace: base=-1800000000+10000000*block.
Attempt-1 blocks: calibration 0..11, confirmation 24..47, smoke 80..81.
Attempt-2 blocks: calibration 12..23, confirmation 48..71, smoke 82..83.
Within each block: data +0, fit +2000000, PF +3000000+1000*r,
reference +4000000+1000000*level+10000*r. Fit train/validation/audit
offsets stay in their own million. Test nonoverlap and signed-int32 bounds.
Old A10 confirmation cases are never calibration cases here.

## Budget, commands and artifacts

Sole budget: artifacts/observation-tt-continuation-24h-20260915-01/budget.json.
A11 starts 2026-09-16T19:24:00Z; available before charge 106807.892609 seconds.
Cap: 43200 active seconds including at most 21600 numerical seconds; numerical
time is not charged twice. Prior reservations remain. At most two attempts
per stage (smoke, calibration, confirmation), fresh output directories and
seed partitions. Repairs under the same scientific contract are authorized;
budget exhaustion is a continuation stop, not permission to cut evidence.

Environment: /home/chakwong/anaconda3/envs/tftwogpu/bin/python; escalated
GPU access, CUDA_VISIBLE_DEVICES=1, TF_FORCE_GPU_ALLOW_GROWTH=true, two CPU
intra-op threads, one inter-op thread, float64, XLA, stable graph signatures.
CPU coefficient SVD is an explicit existing setup exception. Reference code
uses TensorFlow. Pure CPU unit checks hide GPU before import. No package
installation, HMC run, default change, or external publication.

Numerical driver: docs/benchmarks/run_observation_tt_protected_fitting.py.
Commands: python driver --stage STAGE --attempt 1 --output-root OUTPUT
--wall-budget-seconds LIMIT; confirmation additionally consumes
--calibration-root CALIBRATION_OUTPUT. Limits: smoke 900, calibration 10800,
confirmation 5400 seconds, with retries sharing the 21600-second total.
Full exact invoked commands, dependency checksums/source snapshots, Git HEAD,
environment, device/memory policy, seeds, timing and outputs go in manifests.

Plan/math/review artifacts: artifacts/observation-tt-protected-fitting-20260917-01/.
Numerical artifacts: ../benchmarks/artifacts/observation_tt_protected_fitting_20260917/attempt-STAGE-NN/.
Result: observation-aware-tt-protected-fitting-20260917-result.md.
Keep logs on disk and emit only bounded summaries.

## Skeptical review and pre-mortem — 2026-09-17

PASS for bounded execution after mathematical/engineering checks. Reviewed
risks: A10's broader charts may be innocent, SGQF loss is the wrong objective,
high ESS can coexist with biased or weak filtering, a covariance lower bound
does not prove guide accuracy, rank is inert in d1, finite ALS can miss a
better representable fit, and a noisy reference can reverse apparent rankings.
The ablations, exact density checks, nested-capacity caveat, independent
reference ladder, fresh partitions and downstream criterion address these.
No expected candidate failure is upgraded to a continuation veto. A cheap
heuristic can veto promotion but does not justify abandoning TT. Setup/SVD
time and fitting dominate cost and must be included; caching across epsilon
arms must not hide construction time. Fixed controls and local analytical
chart derivatives do not establish total parameter derivatives of the full
L1 fitting/filtering program. MathDevMCP is a scoped review tool, not a proof
certificate. The strongest alternative explanation to any small improvement
is Monte Carlo/reference error; paired fresh-sequence uncertainty is required.
