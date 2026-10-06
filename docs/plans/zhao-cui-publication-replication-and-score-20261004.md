# Zhao–Cui publication replication, then local regression score

Date: 2026-10-04. Branch: sqmc-development. Owner request: explicitly replicate
published results before constructing a quadratic-regression score reference.
This is an original-author-code CPU reference campaign, not a new LEDH run.
No pre-2026-08-21 LEDH result is a baseline. The previous rank-4 smoke is not
publication replication and does not test the published method's accuracy.

## Research intent and evidence contract

Main question: do the author TT filtering and smoothing algorithms reproduce
the SIR and predator–prey findings in Zhao and Cui, JMLR 25(244), sections
6.3–6.4, Figures 15–18? Only after that, can those proposals support a reliable
local numerical likelihood gradient at our specified parameter/data points?

Stage A uses T=20, 33 polynomial degrees of freedom, five ALS sweeps, SIR ranks
10/20/40 and PP rank20 with linear and nonlinear preconditioning. We retain
joint-path importance ESS at each time, the raw weights, true and sampled
paths, unweighted posterior median/5th/95th percentiles (the paper's plotted
quantity), weighted summaries, actual ranks, fit diagnostics and elapsed time.
ESS measures proposal/posterior agreement, not likelihood or score accuracy.
The paper reports about 40% terminal ESS for nonlinear PP; SIR needs ranks above
20 for accurate results. Figures contain no tabulated exact numerical arrays.
Continuous results and their limitations must be reported, not just a pass bit.

The comparator is the published setup and figures, with a separate released-
driver profile. There are no supplied author data, saved fits or figure arrays
in the pinned examples, and MATLAB is unavailable. Octave RNG seed equality
cannot establish MATLAB dataset equality. Thus this is a numerical replication
on a documented synthetic realization, not a claim of identical published
numbers. Cross-seed uncertainty is required before generalizing a pattern.

Primary replication evidence: the full curves and trajectory summaries under
paper settings, agreement or disagreement with the stated published pattern,
and a transparent accounting of source/paper discrepancies. Nonfinite weights,
a missing callback, incorrect target, an unrecorded algorithm substitution,
corrupt output, or source mutation veto interpretation and trigger repair.
A low ESS is a failed candidate/reference, not proof the paper is wrong and
not a continuation veto for another planned rank/preconditioner. Runtime and
ALS residuals explain results; they do not establish accuracy. Missing exact
MATLAB data prevents exact-number replication claims, not execution here.

Stage B fits local log-likelihood differences at the current saved T20 PP and
SIR points. It is conditional on an adequate Stage A reference and additional
same-target checks. The published PP problem integrates six unknown parameters;
a fixed-parameter likelihood is a different conditioning problem. A joint
parameter/state smoother must not be relabeled a fixed-parameter likelihood.
Current PP and SIR data/parameters are loaded from committed dataset JSON,
validated against actual TensorFlow model callbacks and fingerprinted. SIR's
current integrator differs from textbook RK4; preserve the intended target.

## Source anchors and two explicit profiles

Paper: .localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf
(official https://jmlr.org/papers/v25/23-0743.html). Algorithms 1–4, Eq.12,
Eq.34, sections 6.3–6.4 and Figures 15–18 are the mathematical anchors.
Author snapshot: third_party/audit/zhao_cui_tensor_ssm_p10/source, upstream
DeepTransport/tensor-ssm-paper-demo commit
80034dccb99eb1d86284a1839b4a12067d13b9da.

Source operations: eg3_sir/mainscript.m:38–64; eg4_predatorprey/mainscript.m:45–87;
models/full_sol.m:16–132 (fit), :139–206 (smoothing); models/pre_sol.m:8–267
(nonlinear fit), :270–343 (smoothing). The extracted live-script callbacks carry
source hashes and generated line anchors in each extraction manifest.

Released-driver profile preserves all author model callbacks, half-step fourth
RK stage, sampled initial data state, SIR susceptible clipping, driver iteration
limits (SIR 8 then2; PP4 then2), truncated/scaled Gaussian nonlinear reference,
and driver parameter/prior choices. It is source-algorithm faithful modulo
separately recorded Octave syntax compatibility and stable log diagnostics.

Paper profile explicitly harmonizes the following discrepancies to the printed
experiment. These are paper-grounded changes to released source, not claims
that the released driver already implements the printed setup:
- Both models: standard RK4 fourth stage uses x+h*k3, rather than source x+h*k3/2.
- SIR: fixed true initial state (485+j,15-j), matching Gaussian prior mean;
  driver uses (486+j,14-j) and samples its true initial state. Gaussian state
  noise without the source susceptible clipping matches the printed transition.
- PP: physical truth (r,K,a,s,u,v)=(.6,114,25,.3,.5,.5), prior endpoints
  (.1,110,20,.1,0,0) and (1.1,130,30,1.1,1,1), fixed true x0=(50,5).
  Source chart is (r,s,u,v,(K-90)/20,(a-20)/10): use chart truth
  (.6,.3,.5,.5,1.2,.5) and prior offsets (.1,.1,0,0,1,0).
- Five ALS sweeps at every step; released drivers use fewer warm-start sweeps.
- PP nonlinear reference is standard multivariate Gaussian (paper Eq.34),
  rather than the driver's Gaussian truncated at three SD and scaled by three.

Classification: TT-cross, triangular conditioning, backward smoothing and
preconditioning are source_faithful operations. Reproducible smoothing seeds
are fixed_hmc_adaptation in the limited sense of freezing author randomness;
no HMC claim follows. Paper harmonizations and instrumented classes are explicit
extensions to released code, grounded in the cited paper; they cannot close a
claim of byte-identical author execution. Quadratic regression is a local
extension requested by the owner, not an original Zhao–Cui score algorithm.
Pinned files remain immutable; all overlays are written to fresh run directories.

## Defaults and assumptions audit

| Choice | Provenance/role | Risk and earliest check |
|---|---|---|
| T20, 33 DOF, ranks, five sweeps | paper experimental baseline | paper/source mismatch; retain both profiles |
| fit cloud 5000 SIR /10000 PP, low-rank warm start, epd4/5 | released driver, reconstruction hypothesis where paper silent | covariance/domain truncation; report weights, tails and fit residuals |
| SIR initial rank min(20,maxrank), kick5; PP init10 kick10 | source settings, cap for lower-rank paper arms | budget/rank interaction; report realized ranks |
| data seed1, fitseed2, smoothing seed3000+t | source seeds plus fixed previously unspecified smoothing seed | different MATLAB RNG; save data and state explicitly |
| Octave6.4 CPU, two BLAS threads, at most two jobs | available independent-reference environment | syntax/distribution/linear algebra mismatch; source call-chain and healthy log-PDF parity |
| Gaussian log evaluation for smoothing | algebraic stable equivalent of source PDFs | model mismatch; compare healthy PDF values before each run |
| standard-normal nonlinear reference in paper profile | paper Eq.34/section6.4 | coordinate-domain interaction; retain source profile as paired explanation |
| Regression scale/radius | hypothesis, not selected by LEDH agreement | truncation/noise tradeoff; h,h/2,h/4 and heldout directional checks |

No numerics-changing damping, clipping or ridge is silently added. A numerical
failure is retained and localized; such a protection would need its own declared
non-harm evaluation. Author linear algebra and bounded polynomial support may
fail or truncate tails; that risk is measured rather than silently repaired.

## Execution, resources and stopping

Versioned root: docs/plans/artifacts/zhao-cui-publication-replication-20261004/.
One attempt gets one directory, immutable logs, source/extraction hashes, a run
manifest (Git, command, environment, CPU status, seeds, data hash, plan/result,
wall time) and structured results. Failed runs consume budget and remain saved.
Active budget after owner extension:172,800 additional aggregate job-seconds,
counting from the recovery smokes, plus the preserved14,408.25s already spent.
At most two simultaneous full fits with two BLAS/OpenMP threads each; full-run
limit64,800s. The original8h aggregate/2h-case limits are historical. Routine Python tests and plotting are outside this fit budget.
A tiny full-call-chain smoke precedes expensive cases; it establishes mechanics
only. Stop a job for invalid target/nonfinite output or its timeout; repair
localized compatibility faults under the same budget. Stop the campaign on
budget exhaustion, source mutation, or a materially unresolvable target mismatch.
Do not stop other planned repairs merely because an earlier candidate has low ESS.

Priority: (1) paper PP nonlinear20 and SIR40, (2) paper PP linear20 and SIR10/20,
(3) released-driver primary arms to explain discrepancies, (4) independent fit/
smoothing seeds of viable primary arms, (5) current-target score construction.
The first full-size fit gives a measured runtime estimate. If this prevents all
stages fitting the budget, report the unfinished stages explicitly; no abbreviated
run substitutes for the requested publication experiment. No paid compute,
package mutation, GPU training or new scientific default is authorized here.

Commands are issued through docs/benchmarks/run_zhao_cui_publication_replication.py
with explicit model, route, profile, rank, fitseed, horizon and output directory.
Exact launched commands and time are saved automatically. No hidden shell RNG.

## Regression mathematics and prospective validation

At theta0 choose physical parameter scales D (diagonal) and points
 theta_j = theta0 + h D z_j.
Fit log likelihood differences with a full quadratic
 a + b^T z + (1/2) z^T C z,
including all cross terms. The score estimate is (h D)^(-T)b; it is an estimate
of the derivative of log likelihood, not the likelihood itself. For six
parameters there are 28 coefficients, for three there are 10. Begin with 1024
symmetric space-filling training points and 256 heldout points at each of three
radii; increase samples/replicates to address Monte Carlo noise, not polynomial
order by default. D is the positive physical theta0 scale for PP and unit scale
for SIR log parameters; start h=.04,.02,.01, labeled hypotheses. Reject invalid
parameter points instead of clipping them. These values are not tuned to LEDH.

For a validated fixed-theta0 smoothing proposal q0, retain common paths Xi and
alpha_i proportional to p_theta0(Xi,y)/q0(Xi). Then
 log[L(theta)/L(theta0)] ~= log sum_i alpha_i exp(log p_theta(Xi,y)-log p_theta0(Xi,y)).
This follows by importance integration of p_theta(X,y) under q0 and division by
the analogous theta0 estimate. A theta-independent normalization of q0 cancels.
It avoids a fresh TT fit at every nearby point but retains importance bias and
Monte Carlo error. It requires overlap over the whole neighborhood; report ESS
and weight concentration for every point. A PP joint-theta smoother is not q0.
Independent fitted proposals/path samples and local retrained spot checks must
quantify shared proposal bias. Many regression points cannot cure a biased q0.

Engineering checks: full-quadratic exact recovery with known cross terms and
non-unit scales; a Gaussian latent-variable likelihood with analytic score;
ill-conditioned-design rejection; score units/parameter transform; independent
heldout points. Numerical validity: radius stability, proposal/rank/sample
stability, reweighting ESS, independent-seed intervals and heldout residuals.
Scientific promotion: a declared uncertainty interval for each score coordinate
that includes all measured numerical variation, plus same-target likelihood
agreement with an independent reference. Until that is established use
'quadratic regression score estimate', not 'oracle'. The result note reports
continuous discrepancies and uncertainty even when a veto fires.

Cheap adversaries for the later score estimate: central differences, symmetric
linear regression, full quadratic regression, and the independently available
bootstrap likelihood/score comparator. For trajectories: observation-as-infected-
state estimate, deterministic mean dynamics, and a bootstrap particle filter
are diagnostic sanity checks if making an improvement claim. Publication
replication itself makes no superiority claim. Conditional evaluation separates
early/late times, observation excursions, each model and each parameter direction.
No score comparison with LEDH is a tuning or selection criterion.

## Skeptical pre-execution audit

Reviewed before implementation. The original idea would have been invalid if
it reused rank4 smokes, silently treated released settings as paper settings,
compared a joint PP evidence to a conditional likelihood, or called a precise
quadratic fit an oracle. This revised plan addresses each: explicit paper and
source profiles, full published ranks/horizon, separate fixed-theta target,
independent numerical checks and uncertainty. Missing author data prevents exact
figure-number equality; state this without turning it into a claim against the
paper. Finite ESS and smooth regression remain explanatory evidence, not oracle
certification. Budget and invalidity stop conditions are explicit. Audit passes
for bounded execution with these limits; strongest remaining risk is systematic
TT/support bias shared by fitted proposals.

Implementation audit addendum, before execution: the released nonlinear solver
sets its second linear transform to the identity (pre_sol.m:215–221). Keeping
its [-1,1] polynomial domain while replacing its scaled truncated reference
with a standard Gaussian would silently discard most reference mass. The paper
profile therefore uses the author's existing AlgebraicMapping(1) basis domain
for PP nonlinear fitting (the author driver already constructs this alternative
poly2). This is an explicit reconstruction choice where the paper is silent,
not a claim of exact released-driver settings. The author_driver profile keeps
the original bounded domain/reference. Its effect remains a possible replication
discrepancy. The audit rejected the bounded/standard-Gaussian combination before
any experiment.

## Historical runtime calibration and superseded budget proposal

The first PP nonlinear update took173.44s and its second took581.63s. The SIR
rank40 first update took45.01s; subsequent ALS passes are much more expensive
because they evaluate the preceding nontrivial TT density. Extrapolation from
these early updates suggests roughly3–4h for one PP nonlinear run and5–8h for
one SIR rank40 run; these are uncertain estimates, not completed-run timings.
The original7200s per-case deadline is therefore likely to interrupt each fit
beforeT20, and the eight aggregate hours cannot cover the full comparison matrix.

Proposed revision: retain the same scientific contract, two simultaneous CPU
jobs/two threads each, targets and profiles; allow28,800s (8h) per full run and
86,400 aggregate job-seconds (24h), counting all elapsed attempts. Priority is
unchanged. The new budget may still be exhausted before independent fits and
score validation, which would remain explicitly unfinished. No extra hardware,
paid services, packages or GPU allocation are proposed. An execution-controller
repair must preserve running Octave processes and logs when transferring their
supervision; otherwise record the failed attempt and retry in a fresh directory.
Do not silently extend the active7200s deadlines before owner agreement.


## Authorized recovery and budget extension, 2026-10-04

Owner instruction: “you have 48 hours more. continue”. This supersedes the
pending 24-hour proposal and original active time limits. Count 172,800 additional
aggregate job-seconds from recovery onward, at most two CPU jobs concurrently
and two BLAS threads per job. Each full case may run up to64,800 seconds (18h).
The two interrupted cases consumed14,400.27 seconds; the earlier smoke consumed
7.98 seconds. Preserve those costs separately; do not silently reset evidence.

Both initial cases ended at7200s before authorization arrived. PP completed12/20
updates and SIR4/20. SIR updates2–4 took1478,1978,2198 seconds: the earlier5–8h
projection was too optimistic. Octave cannot serialize these solver class objects
on signal, so no valid restart state exists. Restart full cases in fresh -02
directories, with unchanged scientific settings. This is a harness/budget failure,
not evidence against the Zhao–Cui method.

Recovery mechanism: invoke the existing author backward smoother after each
completed update and save numerical arrays immediately. Save and restore Octave
RNG around each smoothing call. Verify exact equality of fit ESS, path samples,
weights and raw log weights against end-only smoothing in a T2 smoke. This is an
observability overlay, not a new filtering algorithm or source-faithfulness claim.
It preserves completed estimates if a later update fails, but does not claim to
serialize or resume the live TT solver. Existing timestamps/arrays are not overwritten.

Skeptical recovery audit: completed update state contains its SIRT, transform and
weights (full_sol.m:30–42; pre_sol.m:25–29), sufficient for the existing smoothing
method. RNG restoration is necessary to prevent an unfair comparison; parity is
the prelaunch check. Use a timer handle because nested source calls use tic/toc.
Missing or nonfinite saved arrays remain validity vetoes. No rank, sample size,
ALS count, target, reference distribution or acceptance criterion is relaxed.
Budget and concurrency remain bounded. After the two primary completions, use
measured costs to allocate remaining planned comparisons and score validation;
unexecuted arms remain unfinished, never silently counted as replications.

Recovery smoke02 caught an environment mismatch before full launch: the pinned
Octave compatibility rng.m implements only seed-setting, not state query/restore.
The two failed smokes consumed3.58s and10.53s. Preserve the actual legacy
rand('seed') and randn('seed') streams instead; the same parity test must pass.
This is an infrastructure repair, with no scientific settings changed.

Recovery prelaunch check passed: validation/incremental-parity-pp-03.json and
incremental-parity-sir-03.json each report16 exact-equality checks, including
fit ESS/ranks, all sampled paths, normalized/raw weights and quantiles. Both
T2 same-seed comparisons completed with unchanged author source. Small-rank SIR
weight collapse is only smoke evidence, not publication evidence. Authorize the
planned -02 full restarts under the owner-granted budget.


## Fixed-target likelihood implementation audit (before Stage B execution)

The current_target runner profile reuses the checked diagnostic conditioning
adapter, source full_sol and TT/smoothing operations. It removes PP parameter
integration, fixes physical theta=(.6,114,25,.3,.5,.5), uses the current PP standard
RK4 and current SIR half-stage RK4, and loads the exact saved observations. It is
an explicitly labeled target adaptation, not replication of the published PP
joint-parameter posterior. Dataset fingerprints and five transition/prior/observation
probes against actual NonlinearSQMCSpec callbacks must pass before any fit. A
prepare-only mode checks this call chain without fitting. PriorN(mean,I), process
and observation covariances have no additional numerical ridge or clipping.

Numerical path likelihood values will call the existing canonical TensorFlow
model factories directly in a stable-signature XLA graph; no analytic scores or
autodiff are inputs to the regression. Flattening paths across time is valid for
these time-homogeneous transition laws. The likelihood adds the initial density,
every transition and each corresponding observation. SIR observation variance
100*exp(2*theta_3) must include its log determinant. Independent NumPy Gaussian
checks, density reconstruction from Octave's raw weights plus proposal history,
and perturbed-parameter callback comparisons precede substantive regression.
This is a CPU independent-reference exception; GPU production status is not claimed.
A single fitted proposal cannot yield between-fit uncertainty: report that gap
even when within-proposal batches appear stable. No ordinary regression standard
error is treated as likelihood/score uncertainty.


## PP nonlinear failure localization, before the next run

The paper-profile PP -02 fit failed in smoothing at t=5 after2388.54s. All four
saved path ESS fractions were .99394,.98370,.96759,.95388. Filtering step5 ESS
was9849.16/10000, which cannot validate a failed path smoother. The immediate
continuation veto applies to interpreting this run beyondt4, not to investigating
or repairing the reference. SIR -02 continues independently.

Next smallest discriminating run: PP paper nonlinear rank20, original10000 fit
and smoothing samples,5ALS, fitseed2, smoothseed3000, horizon5 (exact prefix of
the same simulated data),3600s limit. Retain failing paths, latent parameters,
log target and log proposal before the existing fail-closed guard throws. Verify
the observations and t1–t4 path arrays match -02, then count NaN/+Inf/-Inf and
locate their first density contribution. This is failure localization only;
no shortened-horizon publication claim. No path is dropped, clipped or reweighted.
Strongest alternatives: transition overflow, Gaussian-CDF tail saturation, or
TT conditional-density evaluation; the saved arrays will discriminate them.

A source audit also found proposal_history stores cumulative backward densities
(full_sol.m:171–172), not independent increments. The score harness therefore
uses its first column as the full log proposal. Summing columns would be wrong;
the pre-score equality check against raw weight plus log proposal enforces this.
Between-fit intervals require common rank/sample count and distinct fit seeds.
Mixing ranks into an IID standard-error calculation is forbidden.


## Additional source checks and Gaussian quantile diagnostic

Paper section6 introduction (p29, local text1715–1717) states that all ESS figures
show25/50/75% quantiles from40 repeated smoothing experiments. The initial plan
omitted that reporting convention. Existing single-draw curves are therefore
preliminary replication observations, not reproductions of the published error
bars. The text does not resolve which fitted/data random components are repeated;
report our repetition scope explicitly. Distinct fitted proposals remain necessary
for between-fit variation. No bootstrap of one saved sample may be called40 fresh
author experiments. Within the fixed budget, add fresh smoothing repetitions to
future feasible runs, and state any remaining repetition gap.

The pinned Octave compatibility norminv.m (lines1–5) uses sqrt(2)*erfinv(2*p-1).
For0<p<2^-55, floating-point2*p-1 can round to-1, giving-infinity even though the
Gaussian quantile is finite. From p=erfc(-x/sqrt(2))/2, the equivalent inverse is
-sqrt(2)*erfcinv(2*p) for p<=.5 and sqrt(2)*erfcinv(2*(1-p)) otherwise. No probability
clipping or target change is involved. Test both on an explicit tail/interior grid
against scipy.special.ndtri; preserve exact endpoints, NaN/domain semantics,
CDF round-trip error and interior differences. This diagnostic is independent
of the still-running t5 failure reproduction and cannot by itself establish its
root cause. Only an observed failing-path probability links the two.


## Resolved failure and bounded arithmetic repair

The exact-prefix replay matches all24 saved PP arrays at times1–4 and the five
observations. At t5 only path5383 (zero-based) has an invalid target density;
all sampled states, latent parameters and proposal log densities are finite.
The transition from(8514.423609668846,273.26053310934327) overflows standard RK4.
A100-digit replay of equation(38), paper pp40–41, and the generated
models/pp/odefun.m / predator_step.m gives a finite negative log density with
log10(-log p)=1.927837598132897e23. The Gaussian weight is unrepresentably small.
This is a transition arithmetic failure, not evidence that norminv caused this
NaN. The separate quantile diagnostic found both the old shim and Octave erfcinv
unreliable in extreme tails; neither is silently substituted now.

Next repair: an opt-in independent-reference fallback only for NaN/+Inf PP log
transition values. Replay the same20 RK4 steps, parameter chart, step size and
Gaussian density at100 and200 decimal digits using mpmath; fail closed unless
precision levels agree. Healthy double-precision results remain bit-for-bit
unchanged. A finite high-precision log density below the float64 range converts
to negative infinity, representing zero weight; retain the sample in the
normalizer's N. Do not discard samples, clip states, change the integrator,
change the probability model or replace failed proposal densities. Explicitly
allow negative-infinite log weights when some finite weight remains; still
reject NaN, positive infinity or an entirely zero-weight sample.

Classification: extension_or_invention, a disclosed arithmetic/observability
adapter for this independent reference, not an additional author algorithm or
closure of a source-faithfulness gap. Source operation: paper Eq(38), Gaussian
transition p41; models/pp/predator_step.m RK4 loop and models/pp/odefun.m.
For author_driver keep its half-stage k4; paper/current_target keep full-stage
k4. The no-harm criterion is exact equality on unchanged healthy callback
outputs plus cross-precision agreement on repaired tails. Tests must also reject
nonfinite inputs/singular denominators. Every repair logs inputs, two precisions,
outputs and source hashes. No repair of a failed TT fit is inferred from this
smoother-only arithmetic check.

Skeptical audit: replacing NaN by zero without examining the path would be wrong;
this plan instead recomputes the stated transition. A large negative log density
can be numerically valid. Treating every -Inf log weight as a veto was too strict
for Gaussian tail evaluation, while NaN/+Inf remain vetoes. Independent high-
precision replay is diagnostic/reference code, so it satisfies the NumPy/backend
boundary and leaves all production paths unchanged. The100/200-digit comparison
is a numerical check, not a formal arbitrary-input error bound. Repair and retry
remain within the existing48 job-hour budget. First replay saved paths and a tiny
no-fire case; only then restart the full PP case.


Forty-draw reporting implementation: future primary PP runs will draw40 fresh
10000-path smoothers at every completed time from the same fitted TT, seeds
3000+t+1000000*(replicate-1). This supplies conditional smoothing quartiles;
it does not measure between-fit or between-dataset uncertainty and does not
claim the paper used this exact repetition scope. Preserve every repetition's
ESS, maximal/zero weights, runtime, seed and state quantiles; preserve full paths
for the first draw at each time. Repeated draws restore both RNG streams so
future fits remain exactly equal. A T2, three-repeat check against prior
one-repeat artifacts must show unchanged fits and first-draw arrays before the
full restart. A caller may explicitly narrow repetition times, with the missing
quartiles reported rather than inferred. Measured PP smoother costs imply about
7 job-hours for40 full curves; fit+smooth timeout remains18h and total48h.


Prelaunch arithmetic/repetition checks passed: six high-precision tests;
49,999 healthy saved transition values bit-identical, one recovered tail,
all10,000 samples retained, t5 ESS9503.858361379007. Three-repeat PP and SIR
smokes each pass16 exact fit/first-draw parity checks against the original
single-repeat fixtures. No paper conclusion follows from these mechanics.
Launch PP paper nonlinear rank20, T20,5ALS,10000 fitting samples,40 independent
10000-path smoothing draws per time, existing seeds, opt-in precision adapter,
18h limit, fresh output pp-paper-nonlinear-r20-seed2-04. SIR rank40 -02 continues.
Snapshot03 records the remaining aggregate budget before launch.

## Bounded monograph addition: regression reference

Active source: docs/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex, built through docs/main.tex. The fable-rewrite tree is preserved. Protected baseline and SHA-256: documentation-01/baseline.json under the campaign root. Add one section before the HMC discussion; retain every existing equation, assumption and citation.

Reader: a state-space researcher who knows likelihoods and importance sampling but needs to reconstruct the numerical score. Prerequisites: joint path density, normalized weights, Taylor expansion and least squares. The argument proceeds from fixed-data likelihood to one frozen conditional proposal, common-path likelihood ratios, scaled quadratic coefficients, a worked Gaussian case, and uncertainty/support limitations. The reader should be able to distinguish a joint parameter smoother from a conditional proposal, recover score units, and explain why many nearby likelihood points do not establish an oracle.

Skeptical pre-edit audit: the earlier copied-core prohibition concerns differentiation of an adaptive fitted scalar; a fixed proposal importance integral is a different, explicitly defined reference. The new text must explain this distinction, require target support, preserve Gaussian normalization and parameter-dependent covariance, and distinguish conditional smoother repetitions from independent fits. No ESS, heldout residual or build success will be promoted to likelihood/score accuracy. This addition derives a requested local extension, not an author score algorithm. Existing unit/value-parity checks support only implementation mechanics. Compile the complete monograph and inspect the rendered added section; author self-review is provisional and human readability acceptance remains pending.

## Score weight validity repair audit

Observed PP reference draws legitimately contain zero Gaussian weights (negative infinite log weights), with all paths retained. The score loader currently rejects these along with NaN. This is an engineering defect: it can reject a valid importance sample. Before serious score evaluations, allow negative infinity when at least one weight is finite; reject NaN, positive infinity and all-zero samples. Evaluate each perturbed log weight directly as log joint minus the fixed finite log proposal. Compare Octave/TF baseline densities on their finite entries and require matching negative-infinity masks; use the TF baseline consistently in the reference scalar after parity passes. No density clipping, sample removal, proposal change or tail repair is introduced in this step. Test finite-case parity, zero-to-positive weight changes, all-zero rejection, and actual harness smokes. These checks establish arithmetic validity only.

## Automatic scheduling of the remaining paper comparators

Queue paper-comparator-queue-01.json contains exactly the already planned PP linear rank20 and SIR linear ranks20/10, all atT20 with the printed five-sweep/33-DOF settings. A small standard-library supervisor may launch them when a slot opens, counting the two currently active jobs toward the limit. It makes no scientific selections and performs no automatic code repair. The aggregate48-hour budget includes earlier failures; the supervisor records every command and elapsed time and stops owned processes if that budget is exhausted. It must never start a third full job, reuse an output directory, or overwrite prior results. Existing jobs are observed but not modified.

For these comparators, save one smoothing draw at each time and40 independent smoothing draws atT20. This bounds repeated-sampling cost while measuring terminal conditional quartiles. Intermediate-time quartiles and between-fit variability remain unmeasured; no full40-experiment replication claim follows. This reporting scope is explicit in CLI settings and reports. The PP nonlinear run already in progress retains40 draws at every time.

Skeptical scheduling audit: the queue uses the paper comparator ladder, not low-rank smokes. Fixed output paths are new, all models keep their own source settings, and logged nonfinite/source failures remain evidence rather than successful results. Scheduling changes no target, solver or promotion criterion. No score campaign starts before the completed publication experiments are assessed.

## Published-figure extraction and first completed comparison

Vector extraction of Figures15/17 (PDFpages39/41) records medians and quartile-bar endpoints against printed tick labels, preserving the paper PDF SHA and generated SVG. These are digitized plot values, not author raw arrays. Figure17 contains ten points at times1–10: its nonlinear terminal median is80.7%, while section6.4 separately states about40% after20 steps. No axis rescaling or extrapolation is valid. SIRrank40 at20 is69.3% (IQR67.8–72.2%).

The completed local SIRrank40 firstdraw is50.1%. The local PPlinear run completed with terminal conditional median84.4% (IQR80.2–86.4%); its time10 firstdraw is88.4%. Nonlinear PPtime10 conditional median is81.9% (IQR80.7–82.6%). The printed nonlinear advantage is not reproduced on this synthetic realization. This is a descriptive discrepancy and a trigger for the already planned released-driver comparison, not a continuation veto or evidence that the paper's algorithm is wrong. Data equality and between-fit uncertainty are absent. Same-target likelihood agreement remains mandatory before any score-oracle claim.

Skeptical audit: report both time10 figure comparisons and time20 prose comparisons. Do not substitute the nonlinear prose value for the plotted value, treat conditional smoothing quartiles as independent fitted replications, or interpret increased ESS as proof of a more accurate likelihood. Preserve full continuous values and the exact source/profile adaptations.

## Conditional path-sampling uncertainty for the regression

Before serious score execution, add a delete-group jackknife that preserves the
whole common-path likelihood calculation. Partition N=10000 independent path
draws into20 equal groups in their original draw order. For every parameter
point compute the likelihood ratio after omitting each group, then refit the
same full quadratic. If s[-g] is that fitted score, report
SE_jack=sqrt((G-1)/G sum_g(s[-g]-mean_g s[-g])^2) coordinatewise. This is a
conditional sampling diagnostic, not a total-error interval or correction of
the central estimate. It excludes fitted-TT variability, support error and
radius bias. Concentrated weights or a zero remaining weight sum make this
diagnostic unreliable/unavailable and must be explicit. No ridge, clipping,
weight trimming or numerical protection changes the accepted full estimate.

Also compute prefix path counts2500/5000/10000 from the same already evaluated
joint densities, retaining their regressions and baseline likelihood/ESS. This
is a coupled sample-size diagnostic, not three independent replications and not
a selection criterion. Full10000 paths remain the declared estimate. The
heldout parameter design remains separate from fitting. These diagnostics are
explanatory or promotion vetoes for instability; they do not halt the planned
independent-fit or rank repair merely because a candidate is poor.

Assumption audit: equal20 groups is a conventional bounded jackknife choice,
not a tuned hyperparameter; original KR draws are independent columns under
one fixed fit. Test against literal path omission and the known standard error
of a sample mean. Inspect group-mass concentration before using SE. Record
unavailable intervals when remaining support vanishes. Synthetic tests and
finite-run parity are required before serious use. This adds observability
within the existing method and48-hour budget; audit passes with these limits.

## Released-driver comparison launch audit

The paper-comparator queue is complete (SIRr40/20/10 and PPlinear finished; PPnonlinear continues its final draws). Next run the already planned author_driver profile: PPlinear r20, PPnonlinear r20 and SIRlinear r40, allT20, sourceNfit and sourceALS settings. Save one10000-path smoothing draw per time, exactly matching the released demo's single terminal-draw scope atT20; intermediate draws restore fitting RNG. These are direct released-settings diagnostics, not a40-repetition replication. Full route sources are full_sol.m:16–206, pre_sol.m:8–343, eg4_predatorprey/mainscript.m:45–87 and eg3_sir/mainscript.m:38–64. The paper comparators are Algorithms1/2/4, Eq34, sections6.3–6.4. Source/paper differences remain separately identified in manifests.

These runs test whether the published/local discrepancy persists under the released driver rather than selecting a favorable setup. Healthy values remain unchanged, PP exceptional target arithmetic uses the already validated optional replay. The original source is immutable. Total48-hour budget and max2 full jobs remain unchanged. Each run has18h cap and aggregate budget is enforced by the supervisor. Approximate additional cost is1–7h depending on source nonlinear fitting/SIR iterations; that is a runtime hypothesis, monitored through incremental outputs. Numeric/source invalidity triggers repair; lowESS alone does not stop the remaining model/rank work. No exact-paper-data match or score-oracle conclusion is authorized by this comparison. Skeptical audit passes: correct source baseline, explicit initial/model differences, no likelihood comparison across different models/data, finite diagnostics retained.


## Completed publication assessment and conditional-target launch

At 2026-10-05 00:43 UTC all paper-profile runs have completed. PP nonlinear
T20 conditional median is 75.1842% (IQR72.0064–77.1187%), while paper prose says
approximately40%; at T10 its81.9147% median closely matches Figure17's80.7064%.
The paper-profile linear T10 first draw is88.3880%, so the plotted nonlinear
advantage is not reproduced on this realization. The released-driver PP runs
instead give linear/nonlinear T20 first draws33.8426%/60.2910%, recovering the
qualitative advantage on their separately generated data and settings. SIR
paper r40/r20/r10 terminal measurements are50.0811% (one draw),36.5458% and
11.3022% (conditional40-draw medians); published medians are69.3355%,57.8447%
and15.4021%. No exact-number replication or cross-fit ranking is established.

The source algorithm now supplies finite full-horizon path proposals, with
source immutability and target-density checks; its accuracy depends materially
on the setup. This is adequate to proceed to the already planned independent
same-target validation, not to assume an oracle. Low ESS or unmatched figure
levels are promotion vetoes/repair triggers rather than continuation vetoes.
The remaining released SIR run continues separately to its terminal result.

The next queue is conditional-reference-queue-03.json, with exact commands.
Use the tftwogpu Python environment with CUDA_VISIBLE_DEVICES=-1, two BLAS/TF
threads, two full jobs total across queues, and the unchanged48-hour aggregate
budget. At this assessment about17.3h remain; source SIR is at time19. Launch
SIR current-target rank40 seed2 first (cost hypothesis8–10h), then PP rank20
seeds2/17/29 and SIR rank20 seeds2/17/29 in the available second slot (combined
cost hypothesis3–5h), with up to2h reserved for score evaluation. Each proposal
uses10,000 smoothing paths; fitting uses PP10,000 or SIR5,000 points, five ALS
sweeps,33 polynomial DOF, and source linear preconditioning. These are paper
baselines, not validated settings transferred across models. Seeds17/29 are
predeclared independent diagnostic replications, not selected for favorable
results. Terminal-only smoothing saves reporting cost and does not alter fits.

Fixed conditional PP has four two-slice state dimensions; it has no sampled
parameters. SIR uses the actual canonical half-k4 target, not the paper-profile
full-k4 target. Both load the saved current datasets and require callback
parity within1e-8 before fitting. Author operations are Algorithms1/2/4 and
Eq12, source models/full_sol.m:16–206 and models/pre_sol.m:8–343. Fixing theta
and observations adapts their conditioning; the canonical model callbacks and
quadratic regression remain explicitly local reference extensions.

Baseline absolute likelihood and path-density parity are checked before a
score is interpreted. Independent bootstrap likelihood values are comparison
references; the PP float32 parameter discrepancy must be bounded or replaced
with an exact float64 calculation. At each proposal, run a64-training/32-heldout
full-path timing pilot before the planned1024/256 three-radius score. The pilot
can veto invalid arithmetic and estimate cost but cannot establish score
accuracy. Regression uses canonical TF/XLA values only. Three independent rank20
fits measure between-fit variation; the SIR rank40 fit tests a higher rank.
PP rank40 or extra sampling are bounded follow-ups only if remaining budget
permits and the first evidence motivates them. Compare continuous errors,
jackknife errors, prefix/radius/rank sensitivity and central differences; never
select by agreement with LEDH or tune on heldout points.

Skeptical audit: the paper experiment and current target are explicitly
different, and absolute likelihoods are never compared across them. Frozen
proposal support and correlated Monte Carlo error remain unresolved scientific
risks; same-target likelihood agreement and multiple fits are necessary.
No high ESS, many design points, small heldout residual or source provenance
substitutes for those checks. Nonfinite values, wrong density normalizers,
source mutation, target mismatch and budget exhaustion are continuation vetoes;
poor proposal quality triggers the planned higher-rank/sample checks. The
cheap adversaries are central differences, linear regression on the identical
symmetric design (algebraically the same slope, not independent validation),
and independent bootstrap likelihoods. Comparisons remain descriptive unless
uncertainty supports a stronger conclusion. Audit passes for diagnostic
execution with no oracle or default promotion.

Monograph update: add the equal-group conditional jackknife derivation and its independent-path assumption, group concentration failure, shared-proposal and radius limitations. This documents the already tested diagnostic, preserves all existing mathematics, and requires a fresh full build and rendered equation inspection.


Before the score pilot, a focused full-horizon target/parity check may evaluate
one completed PP proposal at theta0 and the saved float32-rounded comparator
point. This is a bounded120-second diagnostic, not a third full campaign job.
It verifies the same-normalizer path densities and measures the tiny parameter
shift on common paths; retain a delete-group uncertainty estimate. The paired
shift is a measured importance estimate, not a global analytical bound. Compare
the TT likelihood at the rounded point directly with the saved bootstrap value.
No score fitting or candidate selection occurs in this check. All runtime is
included in the aggregate budget. The saved bootstrap reference has four
replications at524288 particles; its finite-particle bias remains unmeasured.


Full-horizon PP pilot completed in8.39s at64/32 points and10,000 paths. Finite
same-target likelihood is-97.3440563. The .02-to-.01 score change and heldout RMS
.01740-to-.001933 show material radius error; they do not invalidate the value
harness. Proceed with the predeclared1024/256 three-radius design and all three
independent fitted proposals, retain .04 results even if overlap is poor, then
use smaller radii as the planned truncation diagnostic. No parameter radius is
selected for agreement with bootstrap or LEDH. Expected full PP runtime is under
10min based on pilot; hard timeout3600s, unchanged aggregate budget. At launch
approximately9.7h remain and only the SIR rank40 fit is active. The PP rounded
parameter check shifts log likelihood by-7.3717e-7 (conditionalSE3.20e-9), far
below bootstrap MCSE. It verifies a common target at the rounded comparator
point but is not a global error bound. Baseline TF/Octave parity is1.42e-13.


## Score overlap diagnosis and radius continuation (2026-10-05 06:00 UTC)

PP main run completed in195.97s, three rank20 fits,1024/256 design. Mean
physical score changes from(-35.0373,-.554873,.0192154,4.84134,-8.57343,10.6705)
at h=.02 to(-33.7315,-.558038,.0201293,4.59012,-8.68440,10.8079) at h=.01.
Heldout RMS drops to .00144–.00159 and minimum ESS exceeds3007 at h=.01,
but radius bias is still visible. The planned truncation follow-up will use
h=.005,.0025,.00125 with the identical three proposals/design,1024/256 points,
and3600s cap. These radii are successive halvings for convergence assessment,
not selected against any comparator score. Retain every radius.

SIR full-path timing pilot completed in52.35s. At h=.02/.01 the minimum
ESS is1.001/1.172 out of10000 and heldout RMS3.085/.927 log units. Poor
off-center overlap is a promotion veto and repair trigger, not arithmetic
failure or a continuation veto. Full original-radius1024/256 evaluation on
three rank20 fits is retained as a reproducible failure diagnosis (expected
15–25min;3600s hard cap), followed by successive-halving radii as budget
permits. All evaluations use fixed canonical targets and common paths; no
score or LEDH comparison selects the radius. The higher-rank SIR proposal
continues independently. Existing48h aggregate budget and two-full-job limit
are unchanged; about9.2h remain.

Skeptical audit passes for diagnostic continuation: original wide radii are
convenience baselines, not target-tuned defaults; both weight concentration
and nonquadratic residuals are reported. Many design points cannot repair
importance-support error. Smaller radius tests local truncation/overlap,
higher rank tests proposal approximation, and independent fits/path-prefix
checks test sampling variation. None alone certifies an oracle.


## PP path-count follow-up

The three10000-path PP fits at h=.005,.0025,.00125 show successive score
changes close to a factor of four, consistent with the O(h^2) truncation
term on a symmetric design. At h=.00125 the mean score is
(-33.3040,-.558648,.0204137,4.50717,-8.71605,10.84744). Conditional jackknife
SE for its first component ranges .140–.261 across fits, and prefix scores
vary more than the last radius change .02033. Increase terminal smoothing
to100000 paths on the same three fixed fit seeds/settings, with no model,
rank, or method change. Because fitted Octave class instances were not saved,
the runner repeats the same fitting commands and changes only smooth-samples.
This is conditional sample-count evidence; the two N levels are not treated
as independent fit replications. Source full_sol.m:146–172 explicitly draws
iid uniform samples before the triangular inverse transform (Algorithm4).
Smoothing randomness is isolated from fitting by the tested RNG restoration.

Exact fitting commands are in conditional-reference-queue-04.json; each has
1800s hard limit, max2 full jobs globally, under the same48h budget. Expected
combined fit/sample time under15min follows measured PP costs, not a guarantee.
Then run1024/256 regression at h=.00125,.000625,.0003125,6000s combined cap.
Run cost is expected25–40min from linear scaling of the204s N10000 ladder.
Do not start this if live remaining budget cannot accommodate it and the
already running SIRrank40 completion.

Skeptical audit: sample count responds to measured sampling variation, not
comparator agreement; both old and new samples are retained. Higher N does
not repair TT support error. Record central likelihood, source/TF parity,
prefix/fit/radius changes, ESS, conditional jackknife and independent-fit
intervals. A source/target mismatch or budget exhaustion remains a stop;
failed precision improvement alone is descriptive evidence, not a reason to
alter the target. No oracle or filter-ranking claim follows automatically.


Reporting audit: score tables average the separately fitted finite-sample
log-likelihood derivatives across independent proposals. They do not claim to
be derivatives of a likelihood pooled across fits. Between-fit t intervals
(n=3) and conditional path jackknife SE are both shown; neither covers shared
support or radius bias. Publication plot labels were inspected after repair.
The monograph now records actual paper/source comparison values and derives
why mean log weight is not the log mean weight used for marginal likelihood.
The printed ESS experiments do not use that returned likelihood diagnostic.

SIR full-design runtime estimate was optimistic: the pilot has204 evaluations
and the three-fit full ladder11574, a factor56.7. Observed first full-radius
time343s implies roughly51min for all nine fits, within the3600s cap. Future
three-fit ladders reserve this cost rather than the earlier15–25min estimate.


## Local SIR regime and final score allocation

The SIR wide-radius run completed in3131.45s. At h=.01, its three-fit mean
score is(197.387,-113.825,5.4501), but95%t halfwidths are(238.907,109.711,
2.9922), minESS1.17–1.34 and heldoutRMS.7655–.8464. This is poor-reference
evidence, not a usable oracle. The successive shrinking pilot at h=.0025,
.000625,.00015625 completed in84.98s. The last two yield scores
(103.858,-61.2307,5.56857) and(103.418,-60.9115,5.60023), heldoutRMS
.00011357/1.733e-6 and minESS1321/2182. The smallest-radius conditional
jackknife SE remains(15.494,8.909,.1374): shrinking repairs local overlap
and truncation but does not supply missing sampling precision.

Use the full1024/256 design at h=.000625,.0003125,.00015625 on all three
rank20 proposals, then the same design/radii on rank40 when complete. These
radii are selected by local overlap and halving behavior only, not comparator
agreement. Time caps3600s for the three-fit run and1800s for rank40; a timeout
is retained as incomplete evidence, not silently retried beyond the budget.
At2026-10-05 09:24UTC4.81 aggregatehours remain, SIRrank40fit is at update16,
and the threePP100000-path fits completed in157–163s each. Budget expectation:
remaining SIRfit about2h, PPscore about.6h, SIRr20score .87h, SIRr40score .3h.
Start PPscore first; recompute budget before each subsequent launch. No new
fits beyond the existing rank40 and no new data/model are added.

Skeptical audit: the tiny pilot design cannot establish a precise slope. Full
design and all independent fits remain required. Baseline overlap may still
be inadequate at rank20; that motivates the already-running rank40 check.
Do not rank filters or declare an oracle from small heldout residuals.


The existing bounded queue supervisor now also accepts the score-reference
runner and waits for explicitly named completed proposal manifests. This
localized harness change permits the already planned SIR rank20/rank40 scores
to start as slots become available, preserving max2 concurrent full jobs and
the shared aggregate budget. Failed dependencies stop that score queue. No
method/settings/target or promotion rule changes. Queue05 contains the exact
commands; dry-run verifies the accepted command paths and remaining budget.


## Terminal execution and post-run audit (2026-10-05T11:47 UTC)

The authorized queue finished with 47.004 of 48 aggregate job-hours used and
no running jobs. The PP 100,000-path continuation completed for all three
rank-20 proposals. Its smallest-radius aggregate is the final precision
comparison in the result note. The SIR rank-20 small-radius continuation hit
the declared 3,600-second timeout after eight of nine estimates; the partial
file is preserved and cannot be used as a complete three-fit aggregate. The
SIR rank-40 small-radius check completed for its one available proposal.

The skeptical execution audit passed at the terminal state. Baselines were
not silently changed: the paper/source ESS runs, fixed-target likelihoods,
wide-radius score, and local-radius score remain separate comparators. The
primary diagnostic was the predeclared full-quadratic score with path and
between-fit uncertainty; held-out RMS and ESS were explanatory/validity
checks, not promotion criteria. The timeout is incomplete evidence, not a
reason to declare the score method or research direction invalid. The final
report explicitly separates descriptive agreement with the independent
bootstrap from oracle certification.

Post-run red-team: PP precision can still be explained by common proposal
support and finite-particle bias; SIR rank-40 agreement is based on one fit and
can still be explained by sampling or rank. An independent rank-40 replication,
higher-N common-path calculation, and exact source-data tie-out are the smallest
artifacts that could materially change the conclusion. No default, production,
LEDH or filter-ranking claim is promoted.
