# RBF backward fitting and full-horizon consumer validation

This continues the authorized controlled-SMC/KDM campaign. Attempts34/35
confirmed the pinned author's ancestor/component indexing error and validated
the corrected TF/XLA forward kernel (18 checks). Both full replications remain
incomplete. Source and correction evidence is in
controlled-smc-rbf-consumer-audit-20260929.md.

## Question and evidence contract

Can a native TensorFlow nonnegative least-squares (NNLS) backward fit reproduce
the inspected one-refinement RBF policy and its corrected full-horizon filter?
The target is the particle-knot branch of ToyMultimodal/cSMC_Resample.m,
lines188–238: natural-scale least squares of the max-shifted backward target,
all N particle knots, then the largest M=16 coefficients retained. This is
distinct from the quadratic log-scale fitting already validated. Published
Algorithm1/eq11 and author supplement9.1 require both component probabilities
and means to use the selected ancestor. The corrected reference changes that
one index; it must never be labeled literal author-code parity.

Reference: installed Octave lsqnonneg, whose implementation uses Lawson–Hanson
active-set least squares, running the pinned author fit. TensorFlow implements
that mathematical active-set procedure with fresh QR solves. An independent
SciPy NNLS check of selected frozen regression problems checks the objective
and KKT conditions, not unidentified coefficient equality. Source files and
source modifications are preserved with every attempt.

Primary correctness criteria, declared before execution:

- Feasible nonnegative coefficients and finite policies. Scaled NNLS dual and
  complementarity residuals must be within20 times the source gradient stopping
  tolerance, 10 eps ||G||_1 N. QR rank failure, active-set capacity exhaustion,
  or iteration exhaustion invalidates the candidate; no ridge or fallback.
- On source backward targets, fitted predictions agree within1e-7 absolute plus
  1e-6 relative; objectives agree within1e-8 plus1e-6 relative. Coefficients and
  supports are explanatory because correlated columns can yield nonunique or
  numerically unidentified coefficients. After top16 truncation, policy values
  and Gaussian integrals are checked separately.
- Full fit-to-forward replay must agree with corrected source trajectories,
  incremental weights, likelihoods and ESS within1e-6 absolute plus1e-8 relative,
  with exact resampling ancestry. Thus matching regression loss alone cannot
  promote the consumer. Fresh streams must also pass validity checks.
- Explicit stable input signatures, one trace per endpoint, GPU/XLA execution,
  memory growth and no runtime NumPy/pfor are engineering requirements.

Veto roles: nonfinite values, invalid source export, incorrect kernel identities,
or failed KKT checks veto correctness. A valid but different fitted policy is a
repair trigger for conditioning/support diagnosis, not evidence against the
paper or authority to loosen these criteria. Missing provenance, corrupted
artifacts or exhausted budget are continuation vetoes. No stochastic ranking,
Table2 reproduction, default readiness, general NNLS admission or HMC claim
follows from this stage.

## Scope and default audit

| Choice | Provenance and reason | Risk and early check | Status |
|---|---|---|---|
| N512, M16, one refinement | Author Table2 scripts | Full NNLS must precede top16 pruning; record active and retained support | Source baseline |
| R=1, q=10, prior N(0,5), beta=.3 sample-SD | Author supplied SigmaSq1 data and script; sample SD uses N-1 | Other observation variances remain untested | First source scope |
| 101 observations | Actual supplied source data; paper says100 | Disclose source/text discrepancy, never relabel horizon | Source convention |
| Seeds20261006/07 | Frozen independent validation streams | Two seeds do not establish performance; use only correctness | Convenience fixture |
| Float64, TF32 off | Isolate method/solver correctness | Does not qualify float32; compare objective and downstream filter | Reference precision |
| Gradient tolerance10 eps ||G||_1 N | Installed Octave default, scale-aware roundoff criterion | Ill-conditioned columns; retain KKT and prediction checks | Explicit source solver convention |
| QR active capacity64 | Static XLA resource bound, four times retained basis size | Capacity can be exceeded; fail closed, never truncate the NNLS problem | Engineering bound only |
| Iteration cap5N | Bounded diagnostic work; no claim of universal convergence | Exhaustion fails rather than silently accepting an iterate | Engineering bound only |
| No ridge/damping | Solves the declared unregularized convex problem with QR | Small active R diagonal triggers a numerical-rank veto; no unsafe accepted factor | Mathematical target |

This is numerical validation, so heuristic controls are exact NNLS solutions
(identity design with positive/negative targets; duplicate columns; zero target)
and the initial zero-policy bootstrap filter. A zero feasible coefficient vector
also bounds the NNLS objective from above. These controls span active/inactive,
rank-deficient and zero-signal situations and are vetoes, not tuning targets.
Conditional empirical comparisons against particle/Kalman-style heuristics
remain mandatory in the later Table2 experiment and cannot be replaced by these
numerical controls.

## Execution and stop conditions

1. c1_rbf_fit_reference: execute the corrected author source for two seeds at
   N512/H101, export bootstrap clouds, all fitted coefficients and backward
   targets, retained policies and full corrected forward results. Preserve
   the exact correction diff. Independent SciPy checks at initial, middle and
   terminal times compare KKT/objective. Stop on invalid data or reference.
2. c2_rbf_fit_validation: run the fixed-shape batch-native active-set solver,
   then backward recursion and corrected forward consumer. Record all primary
   diagnostics, fit/forward trace counts, compilation and wall time, allocator
   peak, source and fitted replay outputs and fresh-stream validity. Stop on
   the first invalid numerical result; preserve the failing time/support.
3. Review the result, refresh master/checkpoint, then plan the remaining two
   variance settings and the repeated Table2 comparison only if this passes.

Use scripts/run_controlled_smc_kdm.py run --phase PHASE with the existing approved
Python command prefix. Reference CPU hides GPUs. GPU execution is trusted,
TF_FORCE_GPU_ALLOW_GROWTH=true, XLA enabled, no environment/package mutation.
Reference cap90s/attempt,180s aggregate, at most2 attempts. GPU cap1800s/attempt,
3600s aggregate, at most2 attempts. Increase the attempt ceiling36 to40 within
the unchanged CPU300/GPU18000 allocation and inherited campaign balance.
Through35, balance is646.350976 CPU seconds and120127.635396 GPU seconds;
tranche balance is193.184041 CPU seconds and7176.335045 GPU seconds.
Every attempt gets a fresh directory under the existing campaign root, a
source/input snapshot and hash, command, environment, seed and timing manifest.

Skeptical pre-run audit PASS: this plan tests the actual NNLS objective and
fit-to-forward call chain, preserves the mandatory top16 truncation, separates
the corrected conditional kernel from literal source behavior, and includes
bounded resource/rank failures. It cannot pass solely through low training
error or a finite likelihood. Its weakest assumption is numerical support
stability under nearly dependent RBF columns; prediction, integral, and full
consumer checks explicitly expose that risk. An apparent success with only a
terminal regression or short horizon would not answer this plan's question.

## Attempt36 reference timeout and bounded repair

The source run timed out after90.004024 seconds before completing the first
seed; no fit/reference pass is claimed. Its scalar transition sampler calls
DiscreteSample512 times per observation, including repeated argument checks,
histograms and randperm(1). This is a concrete candidate cost cause, not yet
a measured attribution. The retry logs every NNLS entry/exit and uses a
vectorized inverse-CDF sampler with identical selected-parent probabilities.
An Octave RNG probe confirms randperm(1) consumes one draw; rand(2,N) preserves
the old draw order. A small actual-source endpoint must match trajectories,
ancestry and likelihoods bitwise before the full accelerated reference starts.
NNLS uses the same objective/tolerance with an explicit5N iteration cap and
rejects nonconvergence. The cap changes only failed/unfinished cases.
The corrected and unvectorized diagnostic source copies and diff are retained.
Source CPU phase aggregate cap becomes180s using existing CPU300 headroom;
all numerical criteria, target and total campaign/tranche budget are unchanged.
Skeptical repair review: source acceleration is acceptable only if its no-change
regression passes; the full run remains separately labeled as accelerated.


## Attempt37 localized NNLS failure and next discriminator

The acceleration passed exact small-fixture trajectories/ancestry/likelihoods.
At the full horizon, source NNLS at time101 converged in20 iterations and0.005s,
time100 in239 iterations and0.039s, then time99 hit2560 iterations in0.575s.
The source numerical solver is therefore not yet an authority for a completed
fit. This is a solver failure, not an XLA, GPU, Python-retracing or paper-math
failure. The capped guard exposed it promptly; the earlier default allowed
100000 iterations. Attempt37 used1.216490 CPU seconds and is preserved.

Run c1_rbf_nnls_diagnosis,45s/attempt,60s aggregate, max2, from existing CPU300
allocation. Export the failing design, target and source iterate. Compute its
singular spectrum and KKT residuals; compare SciPy NNLS and exact-least-squares
BVLS at their declared stopping criteria. Preserve coefficients, predictions,
objectives and convergence flags. A diagnosis PASS means the frozen failure
and alternatives are recorded, not that full fitting passed. A reference is
eligible only if it converged and passes the existing20-times-source-tolerance
KKT guard. No new ridge, pruning, clipping or relaxed correctness criteria.
Then validate the TensorFlow NNLS endpoint against an eligible frozen reference
before retrying the whole recursion. All later empirical phases remain held.
Skeptical repair review: isolates one actual source-generated regression and
changes solver only, retaining the stated nonnegative least-squares objective.


Attempt38 preserved the frozen regression and both independent coefficient
files, then failed summary serialization on a reference-only NumPy boolean.
Cast that boolean to a standard Python bool and retry within its remaining
phase allowance; the1.417126 seconds remain charged. No numerical change.
The focused GPU follow-up c2_rbf_nnls_validation has120s/attempt,240s aggregate,
max2 attempts, consuming the same GPU18000 tranche. Attempt ceiling42 covers
these localized repairs. It checks exact active/deletion/duplicate/zero-signal
controls and the actual frozen512-column failure against the eligible
independent reference, using the unchanged prediction/objective/KKT criteria.
The unvalidated full-horizon TensorFlow consumer remains blocked until this
localized solver test passes and a completed reference can be constructed.


Attempt39 completed the frozen diagnosis. SciPy NNLS converged in0.046s,
69 positive coefficients, dual residual1.50e-14 and RSS0.0002433004944.
BVLS reported success but failed the declared KKT guard (dual3.61e-8), so it
is ineligible. The Octave iterate had RSS0.0002433011022 and dual5.49e-10,
close to optimum but above its4.89e-11 stopping tolerance; its loose diagnostic
KKT guard passed, but the convergence flag remained a veto. The512-column
matrix has numerical rank190 at the standard eps*N cutoff and singular values
38.54 to2.32e-18. These findings support an ill-conditioned solver pathology,
not failure of the convex NNLS objective.

Pre-GPU audit caught a material capacity mismatch: the valid reference has
69 positive coefficients, exceeding the proposed64-slot bound. Increase that
engineering capacity to128 before execution; all512 candidate columns remain
in the same unregularized objective. This does not relax any error criterion
or change the top16 policy. Capacity is still a fail-closed resource bound,
not a support restriction on accepted fits. No GPU attempt is spent testing
the known-insufficient64-slot proposal. Skeptical audit PASS for the128-slot
frozen test; full-horizon reference and generality remain unproved.


Attempt40 stopped before compilation: two new NNLS guard constants were inferred
as float32 by tf.maximum against float64 tensors. Replace them with dtype-matched
zeros_like/ones_like; this changes no mathematical value or tolerance. Charge
2.719385 GPU seconds and retry the same focused validation. No numerical test
had yet run, so this failure is engineering evidence only.

## Accuracy refinement after attempt41

Attempt41 passed primitive, feasibility, rank, capacity, KKT, objective and
single-trace checks, but failed prediction agreement: maximum absolute error
4.9531e-7;34 entries exceed the unchanged1e-7 absolute plus1e-6 relative limit.
Its final dual residual4.24e-11 nearly reaches the source stopping tolerance
4.89e-11; the eligible independent reference has dual residual1.50e-14.
The current candidate failed accuracy; the research direction remains viable.

Question: does tighter active-set termination meet the existing prediction
criterion on this frozen problem? Compare factors1,0.1,0.01 multiplying
eps*||G||_1*N, against attempt41's inherited factor10. This is a stopping-error
calibration, not a change in the least-squares target, rank threshold, capacity,
top16 rule, or promotion tolerances. Solve all three as a native batch with one
trace. Preserve the source-scaled KKT acceptance bound independently from the
termination setting. Select the loosest factor passing the unchanged frozen
prediction, objective, convergence and validity criteria; if none passes,
preserve failures and investigate rather than relaxing criteria.

Defaults audit: factor10 has source provenance but no prediction-error guarantee
for a nearly singular design; the smaller factors are explicit hypotheses.
More iterations, cycling and roundoff are possible failures; expose iterations,
dual error, support size and QR margin for every arm. Zero initialization and
no ridge retain the source unregularized convex target; controls already verify
known positive, negative, duplicate and zero-target cases. A valid independent
NNLS solution and the zero vector are the relevant numerical comparators;
this phase makes no empirical filtering-performance or superiority claim.

Skeptical audit PASS: the comparator solved the same frozen objective and passed
independent KKT checks. Prediction accuracy is an existing primary criterion;
RSS alone cannot promote a candidate. Iteration/rank/capacity exhaustion and
nonfinite values veto the candidate. Source mismatch or corrupted fixtures veto
continuation. A passing frozen problem cannot establish full-recursion accuracy.

New phase c2_rbf_nnls_accuracy: GPU120s per attempt,240s total,max2 attempts;
the same GPU18000s tranche applies. Raise the administrative global attempt
ceiling42 to46 for this refinement and the already planned full-reference/fit
work; this adds no compute grant. Full-reference work remains bounded by the
remaining CPU300s tranche and full-fit work by its existing GPU3600s cap.


## Completed frozen calibration and independent full reference

Attempt42 passed all8 aggregate checks. Prediction errors for stopping factors
1,0.1,0.01 were4.55e-8,1.23e-8,1.16e-12; the predetermined loosest-passing
rule selects1 for the next validation. This is a scope-specific warm-start
hypothesis, not a general NNLS default. No acceptance criteria changed.

Octave failed the source full fit, so the next reference uses SciPy NNLS for
all202 backward problems on two source BPF clouds (N512,H101,R1,M16,seeds
20261006/07). Keep the same natural-scale objective, sample-SD bandwidth,
max-shift target and top16 retention. Preserve complete coefficients, targets,
KKT diagnostics and per-time runtime. Every fit must converge within5N steps
and satisfy the previously declared KKT bound; failed solves invalidate the
reference. NumPy/SciPy is an explicitly independent CPU-only reference.

The corrected author forward program runs both BPF and supplied learned policy.
A diagnostic injection loads the externally fitted policy before RunSMC; it
does not alter its forward equations. Restore the saved random states after
the BPF so forward generation follows the original one-refinement schedule.
Recover random replay inputs with the existing checked procedure. Preserve
pinned source, the diff and exact injected policy. This validates an alternative
solver for the same first-refinement objective; it is not literal unmodified
source-code parity. Per-time coefficient equality remains unrequired, but
retained policy, backward target and final forward criteria remain required.
Top16 truncation can amplify tiny NNLS differences; that is a real veto of
the claimed full numerical parity and must be investigated if observed.

Phase c1_rbf_scipy_reference: CPU75s/attempt,95s total,max2 attempts within the
remaining CPU300s tranche. Phase c2_rbf_fit_validation consumes this reference
and the factor selected by42; existing GPU1800s/attempt,3600s total,max2.
Skeptical audit PASS: honest alternative-solver labeling, identical objective,
source forward call-chain, all-time KKT guards, no proxy-based promotion and
bounded cost. A reference solve failure triggers localization. Invalid
reference data vetoes TF comparison; a TF candidate failure triggers repair.


Attempt43 produced10 valid backward fits, then SciPy NNLS reached its5N
iteration limit at zero-based time90,seed20261006. This rejects that solver
configuration for the full recursion. Local installed _nnls.py shows that its
deprecated atol parameter is ignored; the source-scaled tolerance has only
been an external KKT guard for that reference, not its stopping rule.
Remove the misleading argument. The second already-budgeted CPU attempt will
preserve a failing frozen matrix and try bounded BVLS on the same full matrix
at tolerances1e-14 then1e-15, each with5N iterations. Select the first converged
candidate satisfying the unchanged source-scaled KKT/feasibility criteria;
record every attempt, including rejected candidates. These tolerances are
explicit numerical-accuracy hypotheses motivated by39's insufficient BVLS
termination at1e-12, not empirical filtering tuning. There is no ridge or
column pruning. Stop if all solvers fail, preserving the frozen problem.

Skeptical audit PASS for this localized solver retry: same convex objective,
same independent optimality checks, same phase/CPU cap, honest solver identity
per time point. A valid reference does not guarantee agreement after top16
selection; full TF consumer checks remain mandatory. No repeated retries after
this phase's two-attempt cap without reassessment.


## Reference feasibility localization after44

Attempt44 again stops at time90. Both BVLS settings report convergence and
dual residual3.45e-12,complementarity6.72e-15,RSS1.15e-14, yet the combined
validity guard rejects them. Before changing criteria, isolate which guard
failed by recording minimum coefficient, finiteness and every inequality on
the saved frozen problem. Exact nonnegativity may be exposing roundoff at the
active boundary; this is a hypothesis, not yet a conclusion.

Phase c1_rbf_feasibility_diagnosis has CPU20s,max1 and consumes the same
remaining CPU300s tranche. It evaluates only this matrix and deterministic
positive/zero/negative guard controls. If coefficients lie within16*eps times
max(1,maxabs(coefficient)) below zero, evaluate projection to zero as an
explicit Class C reference-only candidate: require unchanged healthy vectors,
reject negatives beyond the roundoff band, require prediction displacement
<=1e-12*(1+maxabs(original_prediction)), and recheck original KKT/feasibility.
This follows the projection error bound ||G delta||infinity <=
||G||infinity*||delta||infinity; it does not justify clipping larger negatives
or loosening optimization accuracy. No empirical filtering metric selects it.
Record all before/after quantities and the error bound. A successful diagnosis
can authorize one full-reference retry within the existing CPU cap; otherwise
preserve the failed reference and investigate the identified guard.

Skeptical audit PASS for this local diagnostic: the missing guard detail is
an observability defect and should have been retained initially. The proposed
boundary correction is conditional on measured non-harm, exact feasibility,
and unchanged numerical acceptance limits. Raise the attempt ceiling46 to48
to accommodate this diagnostic and a conditional retry plus the pending GPU
consumer; no change to CPU/GPU allocations or inherited grant.


Attempt45 isolated the guard: BVLS returned34 negative roundoff coefficients,
minimum-1.3553e-20. Finiteness and both KKT checks already passed. Projection
of only these boundary values changes predictions by3.55e-20 (derived norm
bound4.30e-19), leaves healthy controls identical and rejects the materially
negative control. All5 checks passed. Adopt the measured reference-only
boundary projection under its fail-closed eligibility checks. Permit the
conditional third c1_rbf_scipy_reference attempt, still within95s aggregate
and the unchanged CPU300s tranche; no further retry is implied.


Attempt46 passed the repaired time90 boundary and reached time82, where
SciPy NNLS returned normally but failed complementarity (4.05e-4). The
reference correctly rejected it. The harness tried BVLS only after an exception,
although the plan required every validity failure to reject that solver and
try the next bounded solver. Repair this dispatch mismatch: validate each
solver candidate before selection, including normal-return NNLS candidates;
preserve failure matrices and all failed candidates' metrics. This changes no
solver, tolerance or acceptance criterion. One fourth reference attempt is
allowed within the same95CPU-second aggregate. Stop on the first problem for
which every declared solver fails. Skeptical audit PASS for matching the
implementation to the existing fallback contract; no numerical guard removed.


Attempt47 correctly tried all three declared solvers at time82 and all failed:
NNLS returned a coefficient vector with complementarity4.05e-4 (its reported
RSS0.264835 disagrees with recomputed0.265319); BVLS has dual residual1.30e-8
above the unchanged8.12e-10 bound. This is a valid-reference blocker, not a
TF/XLA failure. Preserve the frozen matrix and rejected coefficient vectors.

Next smallest diagnostic c1_rbf_solver_diagnosis (CPU30s,total30,max1) checks
SciPy trust-region-reflective bounded least squares at tol1e-14 and1e-15,
then column-normalized NNLS, on this one matrix. Exact column scaling is a
coordinate change: solve G D^-1 z with z>=0, then w=D^-1 z; evaluate every
criterion on original G,w,y and retain original coefficients before top16.
No columns are removed, no ridge added and no optimization objective changed.
The same iteration cap5N and external KKT/feasibility limits apply. Record all
solver outcomes, original-coordinate residuals, conditioning, wall time and
a primal-dual gap diagnostic. Select the first solver meeting all existing
checks, not the smallest descriptive residual. Failure leaves the reference
blocked. Success permits a new full-reference repair phase only after cost
assessment. No full TF run may use an invalid reference.

Skeptical audit PASS: source-scaled termination has no universal prediction
guarantee, so full prediction/policy/consumer tests remain required; a solver
success flag or an RSS from its own return tuple cannot establish correctness.
The original residual must be recomputed. Attempt ceiling52 allows these
localized repairs within unchanged inherited and tranche compute ceilings.


Attempt48 rejected both trust-region solves (no convergence,dual0.0239) and
column-scaled NNLS (complementarity3.76). Those failures do not alter any
criterion. The remaining localized hypothesis is unstable or prematurely
terminated internal least-squares updates. Test an independent Lawson-Hanson
active set with a fresh SVD of the active columns at every solve, using all
original columns as candidates and rejecting rank loss rather than silently
truncating or regularizing. TensorFlow uses QR, so this supplies a different
factorization and independent CPU implementation of the same convex target.

c1_rbf_svd_diagnosis: CPU30s,max1 on frozen time82 and the existing analytical
controls;5N total active-set iterations,termination factor0.01*eps*||G||_1*N
(the stricter setting already examined on time98). This factor remains a
hypothesis across time. Require original KKT/feasibility, analytical controls,
and independently computed primal-dual diagnostics. No rank-deficient active
solve may pass. A success nominates this reference for full-horizon validation;
it does not replace the declared TF prediction/retained-policy criteria.
Skeptical audit PASS: same unregularized problem, explicit rank/iteration
vetoes and independent numerical factorization; no new empirical claim.


Attempt49 passed all four analytical controls and the frozen time82 problem:
154 fresh-SVD solves,peak37 active columns,minimum relative singular margin
3.93e-7,dual6.42e-14,complementarity1.61e-14,RSS0.26483567659.
The dual-feasible certificate gives a conservative prediction-error norm bound
8.54e-7; this is explanatory, not a substitute for pointwise GPU/reference
agreement. Local wall time0.035s supports a bounded full-reference attempt.

New phase c1_rbf_svd_reference uses this independent fresh-SVD solver at every
backward time for both seeds, with unchanged objective/top16/KKT guards and
the corrected author forward program. CPU75s/attempt,80s aggregate,max2,
clipped to the remaining CPU300s tranche. c2_rbf_fit_validation now depends
on this completed reference plus the calibrated GPU solver; GPU limits remain
unchanged. Retire the failed c1_rbf_scipy_reference as historical evidence,
not a passing prerequisite. Every failure must preserve its frozen matrix.
Skeptical audit PASS: the scope remains independent numerical replication,
the measured cost fits the remaining allocation, and the complete filter
comparison remains the final criterion for this stage.


## Localize fit-to-forward mismatch after51

Attempt51 completed398.49GPU seconds with valid fits and fresh forward streams,
single trace for each fit/forward/NNLS function, no rank/capacity/iteration
failure. It failed full numerical parity: prediction max0.01376,backward
target0.01823,retained log-integral5.55,forward logZ0.2034 and ancestry mismatch.
These fail the original criteria; no empirical replication is claimed.

Next c1_rbf_truncation_diagnosis (CPU20s,max1) reads immutable51 tensors and
reference50. Walk backward and locate the first mismatch in beta,target,
full-basis fitted predictions,retained top16 policy and its exact Gaussian
lookahead. Reconstruct the source retained policy from all coefficients to
check the fixture itself. Compare original-objective residuals only when
targets agree; differing recursive targets are not the same NNLS problem.
Record coefficient/support differences and amplification after truncation.
This is post-run explanatory diagnosis, not an empirical performance run.
No threshold can be relaxed or solver selected from downstream likelihood
differences. If local accuracy drives the first mismatch, a frozen same-target
GPU probe must precede another full retry; if a formula/wiring mismatch is
found, repair that instead. Skeptical audit PASS: this separates parameter
nonidentifiability/truncation sensitivity from wrong recursion math, does not
treat per-fit KKT checks as full-method correctness, and fits the remaining
CPU/GPU allocation. Administrative attempt ceiling56; allocations unchanged.

Attempt52 stopped before analysis because Octave-exported dimensions are JSON
floats. Cast the shape metadata to int; allow one localized retry within the
unchanged20CPU-second aggregate. No numerical value or criterion changes.


Attempt53 locates the first divergence at time99 in both seeds. Targets match
to1.4e-15 and full predictions pass (5.62e-8/3.75e-8), but coefficients differ
by0.062 and top16 supports overlap only12/13. Truncated log-policy differences
are20.6/41.6 and true-parent Gaussian lookahead differences0.0501/0.115.
Source top16 reconstruction error is exactly zero. Thus per-fit prediction
accuracy is insufficient for this computation; the top16 operation amplifies
coefficient ambiguity before the earlier recursive targets differ.

Phase c2_rbf_policy_accuracy (GPU180s/attempt,360s total,max2) evaluates both
same-target frozen time99 problems at stopping factors1,0.1,0.01,0.001.
Use one native batched solve and one trace; retain all512 columns in the
objective and the exact top16 rule. Apply unchanged KKT, full-prediction and
objective guards, PLUS the existing retained-log-policy and Gaussian-integral
criteria (1e-7 absolute+1e-6 relative at the frozen cloud, common additive
constant zero). Record per-arm residuals, coefficients, support, and rank.
A factor is eligible only if all criteria pass for both seeds. If none does,
do not spend a second full-fit attempt or loosen criteria; record coefficient
selection/truncation sensitivity and assess a separately specified repair.
Any new regularization, basis reduction, or policy-retention rule changes
the computed scientific method and cannot silently count as source replication.
Skeptical audit PASS: evaluates the learned component as actually consumed,
preserves thresholds, bounds cost, and treats hypothesis failure as evidence.


Attempt54 rejects all4 termination factors for retained-policy replication in
both cases, while all8 solves pass the original per-fit screens. The full
retry is now blocked by a real unmet consumer criterion.

Final local sensitivity phase c1_rbf_coefficient_stability (CPU20s,max1) checks
the reference's own coefficient selection at factors0.1,0.01,0.001 and compares
full512-component versus retained16-component Gaussian integrals for the saved
GPU candidates. This is an explanatory evaluation of frozen fitted functions;
it does not change any runtime method or nominate keeping512 components as
a new default. Report singular spectrum/effective rank, coefficient changes,
full prediction error and exact Gaussian-integral differences before and after
truncation. Preserve the original criteria. A matching reference replay at
factor0.01 is a harness validity check; tolerance-dependent reference policies
would show that cross-solver disagreement is not solely a TF implementation
defect. Stop this bounded RBF repair stage with a concrete verdict after this
diagnostic; any changed objective/tie-break/pruning policy needs a new explicit
method contract before a full retry. Skeptical audit PASS: same frozen data,
no performance-based selection and no unsupported broader-paper conclusion.


### Final diagnostic review: matching the finite stopping procedure

Attempt55 exactly reproduced the factor0.01 reference coefficients for both
frozen time99 problems. Its overall FAIL records iteration exhaustion in the
two factor0.001 alternatives, not a failure of reference replay. Factor0.1
reference solves remained KKT-valid but changed the retained policies. The
reported GPU and reference factor0.1 discrepancies against factor0.01 look
similar; that observation requires a direct matched comparison before inference.

The final post-run phase c1_rbf_matched_solver reads existing coefficients only
(CPU20s,max1, attempt56, within CPU300/GPU18000). It runs no new fitting and
introduces no new policy. The question is whether GPU QR and independent SVD
agree at the same stopping factor. Baselines are the saved SVD coefficients
at factors0.1 and0.01, each on the identical frozen design and target. Primary
diagnostic is full-prediction and retained-policy/Gaussian-integral parity at
unchanged atol1e-7/rtol1e-6 and objective atol1e-8/rtol1e-6, separately in both
seeds. Reference replay, saved fit validity and complete inputs veto use of
the comparison. Coefficient differences and full-mixture tail log errors are
explanatory. No setting can be promoted, no full retry can be launched from
this result, and the failed comparison against fixed factor0.01 remains intact.

This extends the final sensitivity reporting because the review identified a
baseline distinction: sensitivity to the stopping rule is different from
disagreement between implementations of the same finite stopping rule. This
does not change the algorithm, pruning rule, prior thresholds or frozen failed
candidate. Skeptical audit PASS: paired settings answer that distinction,
the frozen inputs cannot establish holdout generality or empirical replication,
and candidate rejection is kept separate from diagnostic validity.
Artifact: attempt56_c1_rbf_matched_solver/summary.json and its run manifest.
Close this bounded diagnostic stage after that comparison; any subsequent
full fitted-policy experiment needs its own explicit paired-solver contract.


Matched-factor result56: factor0.1 passed both frozen cases, while factor0.01
failed. The bounded diagnosis is closed. Further work is explicitly specified
in controlled-smc-rbf-paired-stopping-validation-20260929.md, which tests the
nominated finite procedure on fresh particle seeds without changing thresholds,
objective or retention. The earlier fixed-reference failures remain rejected.
