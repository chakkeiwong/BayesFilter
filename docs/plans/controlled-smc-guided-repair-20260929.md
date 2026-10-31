# Neuroscience comparison repair: normalized guides and original filtering targets

The owner asked to repair the known gaps, review the master and execute. This
phase uses the remaining inherited campaign budget; it creates no new compute
authorization. The prior Figure 4 experiment and its fitted policies are frozen.

## Question and evidence contract

Do the checked controlled-SMC estimates agree with a converged scalar filtering
reference, and how do their likelihood and filtering errors compare with cheap
observation-guided particle filters? A source-curve replication, an accurate
filter and a superior method are separate conclusions.

Use the same 3000 author observations, alpha .99, initial N(0,1), binomial n=50,
and all twenty process variances .01 through .20. Preserve 100 repetitions per
method/cell. Reuse, without refitting or selecting, the existing cSMC N128/I3,
its initial BPF N128 and the source BPF N5529 streams. Replay must reproduce
their saved terminal logZ to 1e-8 absolute plus 1e-12 relative tolerance. This
is extraction of additional diagnostics, not independent new replication.
Add EKF-proposal PF and three-point Gaussian-quadrature-proposal PF, both N128,
with independent streams fixed below and exact importance corrections. These
are constructed classical adversaries, not claimed parts of the author code.

Engineering pass requires analytic Gaussian proposal checks, exact zero-twist
recovery, independent fixed-input sequential parity, valid normalized proposal
corrections, stable graph signatures, and real-data cost/validity checks.
Full collection requires all prescribed cells and repetitions, finite valid
results, successful replay and converged reference checks. A wrong target,
failed oracle, corrupted output, nonfinite value or exhausted phase budget is
a continuation veto for that collection. A method losing a valid comparison
blocks promotion; it does not invalidate replication or stop the other track.

The primary temporal quantity is squared error of the estimated filtering mean
against the converged reference, averaged separately within each fixed count
regime: extreme y<=5 or y>=45; central 15<=y<=35; all remaining counts.
Report empty regimes explicitly. Compare the source terminal logZ statistic,
terminal logZ bias/variance relative to the reference, filtering-variance error,
and corrected prefix-normalizer errors too. The latter are explanatory; noisy
ratios of prefix normalizers are not exact normalized predictive densities.

Across the 100 independent repetitions use 3999 percentile bootstrap resamples
for pointwise 95% intervals, including cSMC-minus-each-comparator mean-error
contrasts. Preserve each regime and variance separately. There is no
simultaneous 20-cell or multiple-comparator ranking claim. Any observed
heuristic underperformance is a promotion veto and a headline finding;
distinguish descriptive losses from pointwise uncertainty-supported losses.
No model or guide tuning uses these comparisons.

## Mathematical repair and implementation

At time t a controlled filter represents an unnormalized prefix density
gamma_t^psi(x_0:t) = p(x_0:t,y_0:t) L_t(x_t), where
L_t(x)=M_{t+1}(psi_{t+1})(x), and L_T=1. This follows by multiplying the
twisted initial distribution, transitions and potentials: every psi factor
cancels except the next-transition integral. Consequently, given normalized
controlled weights W_t and its prefix estimate Z_t^psi,

    Zhat_t = Zhat_t^psi sum_i W_t(i)/L_t(x_t(i))
    Woriginal_t(i) proportional to W_t(i)/L_t(x_t(i)).

Implement this identity using log-sum-exp and the already checked analytic
Gaussian integral. The reference target is the ordinary filtering marginal,
even though the proposal policy was learned using the whole observation
sequence. This is an offline comparison; it cannot establish causal online
latency or turn the adapted proposal into an online algorithm.

For a Gaussian predictive component N(m,v), an EKF guide uses h(m), h'(m)
and observation variance R(m). Its Gaussian conditional has
S=h'(m)^2 v+R(m), K=v h'(m)/S, mean m+K(y-h(m)) and variance v R(m)/S.
For the binomial model h(x)=50 sigmoid(x) and R(x)=50 p(x)(1-p(x)).
The alternative guide uses nodes m+sqrt(v){-sqrt(3),0,sqrt(3)} with positive
weights {1/6,2/3,1/6}; compute E[h], Cov(x,h), and Var(h)+E[R], then condition
the joint Gaussian. These nodes integrate Gaussian polynomials through degree
five, fix the guide without a tuning search, and give the exact same conditional
as the EKF for a linear Gaussian observation. Both proposals retain the exact
weight g(y|x) N(x;m,v)/q(x|m,y). No clipping, damping or covariance floor is
silently applied. Invalid conditional variance is rejected.

A diagnostic TF/FP64 dense quadrature filter integrates the original transition
and normalized observation on uniform trapezoid grids. Freeze three levels:
1025 points on [-32,32], 2049 on [-32,32], and 2561 on [-40,40]
(after the recorded support repair below). Check maximum
prefix-logZ difference <=1e-6, mean/variance difference <=1e-7, and total
predictive mass loss <=1e-10. Check against an analytic Kalman filter first.
The tolerance is negligible beside Monte Carlo errors; agreement is numerical
convergence evidence, not a rigorous error bound. Failure triggers resolution
or support diagnosis before any reference-based interpretation.

## Defaults, failure modes and early checks

| Choice | Provenance/status | Failure risk and earliest diagnostic |
|---|---|---|
| Source data and cSMC/BPF policies/settings | Frozen completed Figure 4 experiment | Stale streams/policies; terminal replay equality for every reused replicate |
| N128 guided PF | Same cloud size as source controlled PF; cheap heuristic, not cost matching | A particle count comparison hides fitting cost; report original fit/forward cost and new diagnostic replay cost separately |
| EKF and positive three-point guide | Derived Gaussian conditioning; fixed untuned adversaries | Invalid or poor local approximation; linear Gaussian identity and extreme-count probes before real data |
| No proposal damping/floor | Positive Gaussian Schur complement mathematically; fail-closed numerics | Cancellation or saturated binomial moments; relative variance margin and observed extrema, no silent repair |
| Grid extent and spacing | Reference hypotheses, not exact-oracle defaults | Tail truncation or discretization error; independent extent/refinement checks in every cell |
| Counts define regimes | Binomial likelihood curvature and extremes, fixed before new outcomes | Sparse/empty regimes; count and report them without redefining |
| FP64/GPU/XLA, TF32 off | Existing source-parity execution scope | Compilation, memory, retracing; same-input graph/no-XLA diagnostic parity and timed source-length pilot |

## Execution and budget

New dispatcher phases: c3_guided_validation (each attempt <=300 GPU seconds,
<=600 total, at most three attempts); c3_guided_matrix (<=7200 GPU seconds
across at most two attempts). The latter requires the validation and an
estimated collection time, with factor 1.5 plus 120 seconds, below its cap.
Extend the current allocation to CPU300/GPU18000 seconds and total attempt cap
25, all drawn from the existing inherited balance of 122359.588265 GPU seconds
after attempt19. No unrelated GPU job may run concurrently through this driver.

Seed 20260929; new guide streams 5000000+grid*10000+batch*100 for EKF and
6000000+grid*10000+batch*100 for quadrature, with +1 for resampling uniforms.
Existing replay streams remain exactly as recorded in attempt15. Bootstrap
streams begin 7000000. Batch10 throughout new guides and cSMC replay; BPF5529
retains its measured scalar execution. A Python loop dispatches independent
batches/cells and writes artifacts; every repeated scientific kernel uses a
stable TensorFlow signature and native time loop. No pfor or NumPy runtime.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c3_guided_validation
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c3_guided_matrix

Preserve new unique attempt directories with manifest/source/input hashes,
validation checks, per-replica/regime metrics, grid convergence, replay errors,
trace/memory/cost diagnostics, progress, summary and uncertainty tables. Record
every failed attempt and remaining balance; never discard a bad seed.

## Skeptical review before implementation/execution

PASS for this bounded repair specification. The audit found and repaired the
original temporal-target mismatch. Baselines are normalized and untuned;
source replication is not conflated with superiority; grid accuracy and replay
are checked before interpreting outcomes; costs include adaptation and separate
replay; uncertain defaults have early diagnostics and explicit rejection rules.
The strongest remaining risk is poor importance sampling after removing a
strong future twist. Such a valid but inaccurate cloud is a candidate failure
and an informative comparison, not permission to change the target or threshold.
Implementation must still pass the executable validation before the matrix.

Lorenz/RBF ports and K full learning remain separate phases of the master.
Missing K author coefficients/schedules cannot be repaired by calling chosen
values recovered settings. A labeled reconstruction is possible; exact published
curve reproduction is unsupported until those settings are recovered.

## Validation result and matrix review

Attempt20 passed43 analytic, original-target, graph/XLA, Kalman-reference,
extreme-count and full-horizon pilot checks in10.382698 charged GPU seconds.
Guide warm batch10 times were .344--.365 seconds. Each fixed signature traced
once. TensorFlow's aggregate retracing warning counted different deliberately
constructed diagnostic engines; it did not indicate repeated tracing of one
configuration. The matrix creates one guide engine per method and passes q as
a tensor, so the twenty-cell loop reuses each guide graph.

The matrix begins with an additional independent Python six-step reference
for both binomial guides, exact ancestry included, before collecting results.
This diagnostic-only reference and the grid-convergence/cost blocks are
mandatory preflight within the matrix command. Preserved Figure4 timings
give77.225 seconds for initial N128 forwards,49.147 for final controlled
forwards and2225.883 for BPF5529. The first full reference cell supplies the
missing grid cost; the command enforces the predeclared conservative cap.

Static call-chain review PASS: the matrix calls the same normalized guide
forward kernel as the analytic checks; controlled replay calls the newly
wired original_filtering endpoint and verifies every terminal saved logZ.
The reference uses original Gaussian transitions and normalized observations.
No NumPy, pfor, per-sample Python runtime or silent numerical floor is present.
The original forward and fitting bodies were not changed. All input streams,
policies and source snapshots remain fixed. Full scientific interpretation
waits for complete reference convergence and10000 expected comparator rows.


## Attempt21 reference-support failure and localized repair

The independent binomial sequential preflight passed all10 checks, including
exact ancestry. The cost estimate was3940.572 seconds below7200. Reference
checks passed through q=.09, but at q=.10 the original [-16,16] support had
predictive-mass loss8.00003e-10 and variance difference5.86387e-7 against
[-20,20]. This exceeds the frozen1e-10 mass and1e-7 moment criteria. No
comparison rows were collected. The attempt consumed16.926206 GPU seconds.

This invalidates that reference support, not either filtering algorithm.
Repair: expand to1025/2049 points on [-32,32] and2561 on [-40,40], retaining
the original .0625/.03125 spacings, all scientific targets, thresholds, source
streams and phase caps. The broader extent addresses the observed truncation;
all twenty convergence checks remain mandatory. The first cell again measures
cost. No failed seed is omitted and no acceptance threshold is relaxed.
