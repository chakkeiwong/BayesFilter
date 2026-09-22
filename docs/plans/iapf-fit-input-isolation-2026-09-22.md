# Actual iAPF fitting inputs: objective escape versus recursive target error

Status: COMPLETE. Four actual adaptive replays, 64 isolated fits and 384
independent R objective/gradient rows pass their diagnostic checks.
[Result](artifacts/iapf-fit-input-isolation-20260922-01/result.md).
Prior: [guide geometry](artifacts/iapf-guide-geometry-20260922-01/result.md).

## Question and source-grounded mechanism

Why can a converged local density fit produce a poor future guide? Capture the
real clouds entering `execute_iapf -> make_density_recursive_fit_kernel ->
bounded_density_fit`. Preserve all calls, including the rejected d5 fit. For a
fixed cloud, its positive target is b_i=g_t(x_i) Fpsi_(t+1)(x_i), using the
newly fitted future guide and its actual floor; at T it is simply g_T(x_i).
GJL Section 5.1, Equation 15 profiles a positive scale multiplying b.

For vector p of Gaussian density evaluations, lambda=(p'b)/(b'b), so
L=||p-lambda b||²/N=(||p||²/N)[1-(p'b)²/(||p||²||b||²)].
Write E=||p||²/N and S=1-cos²(p,b). Decreasing E can decrease L even as the
shape error S increases. On any finite cloud, moving a Gaussian mean to infinity
at fixed covariance makes E and L tend to zero regardless of shape. This is an
unbounded-objective degeneracy, not proof that every bounded trajectory escapes.
Check actual bounded trajectories before attributing their failures to it.

The existing optional relative_shape objective minimizes S and changes Equation
15. It is an explicitly different objective, never a paper-faithfulness repair
by itself. Exact backward targets distinguish local fitting error from errors
propagated through fitted future messages. Inspect the saved technical paper
section and actual call chain before implementing this diagnostic.

## Intent, evidence contract and constructed comparisons

Four deliberately selected mechanism cases from the preserved initialization
study: d2/s82/cloud_moments/initial_peak (bad terminal guide); the same d2 seed
with log_quadratic/initial_peak (initialization control); d5/s82/log_quadratic/
initial_peak (iteration-4 nonconvergence); d10/s82/log_quadratic/native
(completed higher-dimensional fit). These are diagnostic selections, not a
random sample or accuracy ranking.

1. Replay the actual adaptive consumers with the identical saved configuration,
   observations and seed namespace, wrapping the existing factory without
   changing returned values. Preserve each input cloud and returned fit. Require
   the same status, coefficients, controller history, count and output (1e-9
   numerical tolerance); the d5 rejection must reproduce, not disappear.
2. On each last captured cloud/time, reconstruct the actual recursive target
   and an independent R exact backward Gaussian target. Re-run the original
   bounded fit on the actual target and require coefficient/floor agreement to
   1e-8 with the captured recursive fit. Record initial and final density loss,
   density energy, relative shape, projected gradient, clipping, bounds and KL
   to the exact message. No extra steps or relaxed tolerance.
3. Fixed-cloud factorial: actual versus exact target, density_l2 versus the
   already implemented relative_shape objective, identical original
   initialization/bounds/2000-step cap. Relative shape uses native units as
   required by its API. Preserve rejected fits; do not call them successful
   repairs. These are explanatory diagnostics, not downstream filtering runs.
4. Independently recompute density/shape/energy/analytic gradients in base R
   from exported inputs and parameters. Require scaled error <=1e-8 and
   L=E*S within 1e-10 scaled error. This tests arithmetic, not optimizer quality.

Primary diagnostic criterion: verified actual-call replay and independent
objective identities. An observed decrease in density loss with increased shape
error and reduced density energy supports objective escape on that trajectory.
A bad terminal guide localizes failure without a future-target explanation.
Differences between actual and exact-target fits diagnose recursive target
contamination; neither one alone proves the other's absence.

Constructed cheap comparators: unchanged cloud-moment Gaussian (no optimization),
strict QR initialization (existing analytic log-shape fit), exact KL-optimal
diagonal Gaussian, and full exact Gaussian. Compare by time, dimension,
initialization and actual/exact target; their objectives and KL are explanatory.
The previous bootstrap/constant-guide/one-step/Kalman filtering vetoes remain
active; fixed-cloud improvement cannot clear them. No method promotion is
possible in this phase. Mark the heuristic verdict accordingly.

Continuation vetoes: failed source/data identity, wrong replay, incorrect target
or R objective identity, nonfinite accepted output, missing capture or exhausted
budget. Poor shape, changed local minima and failed alternative fits are repair
triggers, not continuation vetoes. Never interpret a fit to exact messages as a
deployable learner or heldout likelihood result. No stochastic ranking, default,
TF32, paper replication, model-score, canonical LEDH or HMC claim.

## Defaults, pre-mortem and skeptical review

The selected cases and last-call focus are hypotheses chosen to isolate known
failure classes; selection limits prevalence claims. Capturing all earlier calls
guards against silently changing the controller. Standardization, bounds,
tolerance, floor and cap come from the original test and remain frozen for
causal comparison; they are not validated general defaults. A bounded mean or
variance can alter even QR/exact-target behavior; report clipping and boundary
activity before interpretation. The full exact covariance need not be diagonal:
its diagonal KL projection is a different objective from finite-cloud density
fitting. Exact R messages are independently checked; terminal target equality
is a cheap early check. FP64/XLA is a diagnostic exception; strict TF32 remains
vetoed. Saved positive floors stay in actual recursive targets.

Pre-mortem: a successful command could silently analyze new random clouds,
drop the positive floor, confuse an exact oracle with the fitted target, or
mistake tiny loss for a good guide. Replay checks and separate target/energy/
shape/KL fields detect these. A relative-shape failure may be a local-solver or
bound failure, not evidence that all scale-invariant fitting is impossible.

Skeptical review PASS: the initial-law gap is already isolated; this plan tests
actual fitter inputs. No proxy metric promotes a candidate. The rejected case,
known-good initialization control, exact-message comparator and independent
arithmetic prevent a one-sided optimizer narrative. No production change.

## Execution and budget

Root `artifacts/iapf-fit-input-isolation-20260922-01/`; transfer exact remaining
balance from guide-geometry budget (45.851 CPU / 47.830 GPU hours). Phase cap
1 CPU hour and 30 GPU minutes, maximum six launches, three local repairs.
Each launch <=300 seconds. Launcher:
`tftwogpu/bin/python docs/benchmarks/diagnose_iapf_fit_inputs.py --attempt <unique>
--device cpu|gpu --mode capture|isolate|verify`. Actual capture and fitting use
RTX 4080 SUPER UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b, escalated, verified
memory growth; independent R verification uses CPU with GPU hidden. Archive
commands, source/input hashes and snapshots, seeds, environments, wall times,
captured arrays, decision/inference tables and terminal review. Local harness
repairs preserve failed evidence and count against the unchanged phase budget.
