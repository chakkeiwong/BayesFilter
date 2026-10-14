# Repair continuation: finite iAPF fitter and adaptive consumer

Authorized by the owner's 28 September instruction to close known gaps, refresh,
review and execute. This is R4c under the existing master and resource ceilings.
It does not replace the missing original iAPF executable with a claim of recovery.

## Gap assessment

Known engineering closures: joint Eq.15 finite fitter, backward targets,
adaptive controller, actual endpoint wiring, then GPU/XLA parity. MDPF source
metric closure is done; actual Appendix B learning and fair normalized-density
comparators remain separate work. Exact iAPF original settings remain
unrecoverable from inspected public sources; proceed with the frozen explicit
reconstruction, while exact-author-code reproduction stays unavailable.
Full independent MDPF training repetitions are under-budgeted at measured cost.

## Evidence contract

Question: does the TensorFlow implementation preserve the frozen R finite-BFGS
reconstruction and adaptive controller within checked cases?
Comparator: R 4.1.2 optim(method=BFGS,maxit=100,reltol=sqrt(epsilon)), the existing
finite_problem, iapf_fit_initial, backward targets and iapf_iterate.
Read R's matching vmmin implementation, including Armijo step reduction,
relative-change stopping, inverse-Hessian resets and iteration counting.
A generic TFP optimizer is not interchangeable with this finite procedure.

Primary implementation checks, in order:
1. Initializer, joint objective and analytic gradient parity on Gaussian,
   nonconcave/multimodal and paper-cloud inputs, both raw and scaled recipes.
2. Finite fit endpoints, stopping codes and proposal parameters against R.
3. Backward fitted twists and all recursion outputs with paired randomness.
4. Adaptive controller iteration/particle history, stopping and fresh final
   pass; actual shared fitter/filter entrypoints must be invoked.
5. CPU FP64 diagnostic, GPU graph diagnostic, GPU/XLA on the same fixtures.

Predeclare relative tolerance 1e-8*(1+abs(reference)) for objective/gradient;
1e-5*(1+abs(reference)) for finite fit parameters and fixed-input backward
outputs, with exact validity/termination/controller classifications.
A tolerance failure is a port-repair trigger, not permission to increase
tolerance. Record maximum discrepancies and inspect the first differing step.
If backend reduction order changes finite optimizer trajectories materially,
report that finite-recipe parity failed and evaluate an explicitly proposed
repair; do not claim exact equivalence from similar final losses.

Validity/continuation vetoes: nonfinite intermediate accepted scientific values,
nonpositive fitted amplitude/covariance, altered objective/floor, missing
source or trace information, wrong comparator, budget exhaustion.
Candidate fit failure is preserved with parameters, targets, initializer,
optimizer iterate/evaluation history, failure code and last gradient.
100-iteration termination is retained as in R, not a stationarity veto.
Oracle and BPF checks are falsification controls; not training targets.
Fit loss/gradient/shape/ESS and timing are explanatory only. No statistical
ranking or full replication/default/HMC claim is possible from these checks.

## Defaults, budgets and early checks

Use frozen A_raw and B_scaled only, not a recipe search. Source-supported
finite-BFGS family; initializer/coordinates/controls/floor remain reconstruction
choices. Floor delta=1/N and scale-aware peak formula remain unchanged.
No ridge, covariance clipping, regularization or objective change is introduced.
A non-pivoted QR numerical implementation is acceptable only on demonstrated
full-rank cases matching the R initializer; fail closed on an ambiguous rank
margin. This is an explicit restricted initializer domain, not general parity
with R's rank-pivoted QR. Weighted-moment fallback for nonconcavity must match.
The earliest checks include full-rank recovery and fallback cases.

Use native TensorFlow loops, stable signatures and vectorized particle
operations; no NumPy runtime, pfor or Python particle/time fitting loops.
Python may orchestrate adaptive passes and shape changes, logging each unique
compiled particle size. A finite-size cache bounds shape specialization.
CPU-only imports hide GPU. GPU phases require escalation, nvidia-smi then
framework probe, verified memory growth and FP64/TF32-off recorded.

Numerical phase ceilings: reference export 90 CPU seconds; CPU parity 240
seconds; GPU graph 300 seconds; GPU/XLA 900 seconds; downstream adaptive
diagnostic 600 GPU seconds. These draw from the existing balance, not additional
allowance. At most 10 additional launches, preserving all prior attempts.
Do not run the downstream diagnostic before fitter/controller controls pass.
Failures consume the same budget; revise diagnosis after two same-fault retries.

Versioned outputs remain in the existing campaign's attemptNN_phase directories.
Exact dispatch commands, source snapshots and dependency identities are required
before launch. Every phase emits decision.csv, runtime/diagnostics and manifest.
The current source is frozen in each attempt, never overwritten as evidence.

## Skeptical pre-implementation review

PASS for this bounded reconstruction test, with the initializer rank restriction
explicit. Wrong-baseline risk is addressed by matching R's actual finite solver,
not only its objective. Proxy success cannot mark replication complete.
Comparisons use identical points/targets or explicit matched entropy. Source
unknowns and small sample uncertainty cannot be hidden by GPU success.
Safety caps reject and flag; they do not alter accepted arithmetic.
No complete MDPF or paper-scale claim is authorized by these checks.
Review is self-review. No independent agent was invoked.

## Execution repair
The first CPU comparison passed all 129 numerical checks, but the harness wrote
only decision.json while the dispatcher requires decision.csv. One prematurely
launched retry repeated that serialization failure. Preserve both attempts;
add the required CSV to the harness and rerun once. This is an infrastructure
failure, not evidence against the finite fitter. No criterion or method changes.

## Downstream paper-model check (frozen before launch)
After all three 129-check parity modes pass, repeat the actual adaptive consumer
on d=5, T=100, A_ij=.42^(abs(i-j)+1), initial/Q/R identity, C identity,
A_raw, N0=1000, max_iterations=12, max_particles=8000, k=5, tau=.5,
kappa=.5. Fresh data seed 9286300; saved entropy seed 9286400+filter_call.
An R export (120 CPU-second ceiling) and GPU/XLA consumer (600 GPU-second
ceiling) use identical arrays. Primary criterion is the same finite-fitter,
backward and adaptive parity, including all prefix likelihoods and clouds.
Kalman likelihood is an explanatory oracle for this one-run diagnostic; it
cannot certify likelihood bias/variance or choose a recipe. No tolerance changes
are allowed. This is the smallest paper-horizon integration diagnostic, not a
d=5 scientific replication or a favorable-seed selection. A fitting or parity
failure is preserved and triggers diagnosis. It may not be replaced by another
seed to obtain a pass. The existing 10-launch and CPU/GPU campaign caps apply.
Defaults are from paper model and frozen A_raw reconstruction; shorter 12-pass
and 8000-particle caps only bound cost and yield INCOMPLETE when reached.
Self-review PASS: source-compatible initial law and adaptive ESS retained;
no short-run metric is promoted, and exact R entropy makes implementation
mismatches distinguishable from Monte Carlo variation.

## Paper-horizon failure isolation (attempt24)
Five of six fixed-cloud backward calls pass. Backward call2 fails at times54
and53 (one-based): maximum center error 0.0002534552 and covariance error
0.0002552702, above the unchanged 1e-5 relative-plus-absolute tolerance.
No actual adaptive GPU run was launched after this veto.
Do not repeat the full run without diagnosis.

Bounded diagnostic: replay call2/time54 (first failing time in backward order)
and adjacent times55/53 using exactly saved R clouds, targets, initializers and
accepted optimizer points. Evaluate CPU graph, GPU graph and GPU/XLA in FP64.
Compare initializer, endpoint, optimizer trajectory and analytic gradients
along the R path. Reconstruct the time54 target from the failed TensorFlow
backward result to distinguish propagation of earlier errors from a local
fitting mismatch. Preserve full diagnostic fits and per-iterate errors.
The earlier tolerances remain unchanged; no close loss can substitute for
parameter parity. This diagnostic has no promotion or ranking authority.
GPU accounting includes the CPU comparator; limit180 seconds, at most one
launch initially. The existing tranche/campaign caps apply.
Self-review PASS: first differing backward location and fixed-input replay
answer the mismatch; a failed candidate is a repair trigger, not method rejection.

## Serialization and BFGS edge-case repair review
Attempt25 replay isolates a sensitive finite solve at call2/time54. At identical
saved R points, analytic gradient errors are <=4.1e-17 across all three modes,
yet late optimizer paths differ; GPU graph also fails, so XLA alone is not the
cause. Reconstructed upstream targets differ by only9.21e-12. This is evidence
of finite-optimizer sensitivity; exact causal attribution to arithmetic,
initializer or serialization remains open.

A direct round-trip control proves the fixture writer loses precision:
pi=3.1415926535897931 becomes3.14159265358979 under write.table even with
options(digits=17). Repair exports using explicit17-significant-digit strings
and assert exact numeric round-trip on every exported matrix. Re-export the
same d5T100 seed and entropy; include the eight original fitting controls in
that fixture. This repairs the comparator, not the numerical tolerance.

Source review also corrects a no-change line-search edge: R retains Fmin when
relative change is still large but the proposed point rounds to unchanged;
the port previously replaced it with the last rejected value. Record this
localized semantic repair without claiming it caused the observed maxit100
failure. No regularizer, clipping, damping or scientific objective changes.

Authorize at most three additional launch slots, raising the cap26 to28
(two more total slots, not compute); export ceiling120CPU and consumer600GPU
remain. Failed attempts24/25 stay preserved. The new v2 paper phase supersedes
the rounded-fixture admission attempt, but a new failure must be diagnosed
rather than hidden by another seed. Contract/tolerances stay fixed.
Self-review PASS: exact round-trip is the cheapest control for the discovered
artifact defect, and all eight fit cases are repeated on the updated fitter
before the paper-horizon backward/controller checks.

## R4c sensitivity and downstream-impact diagnostic (attempt28)
The exact-roundtrip fixture rerun27 still fails backward2 center/covariance/
floor agreement (max absolute .0029515/.0029782/.0012121). Eight isolated fit
controls and five other backward passes pass; the actual adaptive consumer has
not run at this paper horizon. Therefore precision loss was a real fixture
defect, but fixing it did not close the finite-solver discrepancy.

Question: how sensitive is the first failing fit, and does this prevent the
actual shared TensorFlow GPU/XLA consumer from completing? Replay exact
fixture26 call2/time54 on GPU/XLA, then apply deterministic zero-mean alternating
log-target perturbations of +/-1e-14 and +/-1e-12. Preserve initializers,
endpoints, objective, analytic gradient, finite exit code, and density changes
on the actual point cloud. These are explanatory sensitivity measurements,
not optimizer-selection or promotion criteria.

Then run the unchanged shared run_finite_iapf paper-horizon consumer using
the exact reference data and saved entropy. This is an explicit diagnostic
exception to the admission ordering; failed R-parameter parity remains FAIL.
Compare every filter prefix and cloud, controller decisions, final likelihood
and Kalman oracle, and report where discrete paths diverge. Stop if entropy
shape/count differs from the reference, inputs/outputs are invalid, the
existing controller caps fire, or the 600 GPU-second phase ceiling is reached.
No fresh seeds, fallback optimizer, tolerance change, regularizer, default
promotion, or claim of stochastic accuracy is permitted. A completed diagnostic
is not a passed filter replication. No heuristic-superiority conclusion is
sought: exact Kalman is the diagnostic comparator; BPF/fully adapted and
conditional heuristic comparisons remain required before any quality promotion.

This consumes the existing final launch slot28 and existing compute balance.
The source snapshot and attempt manifest preserve the actual numerical
consumer. Plan self-review PASS: distinguishing execution failure from
parameter sensitivity answers the next question; a parameter-parity failure
is a promotion veto, not a continuation veto for this explanatory run.
Missing source settings, single-seed uncertainty and optimizer conditioning
remain explicit limitations.
