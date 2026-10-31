# Expanded marginal-weight verification — 6 October 2026

Status: authorized by the owner's request to update, review, execute, and
analyze. This extends the completed October 5 master. No posterior HMC run is
included in this component experiment. The eventual research goal remains
**testing surrogate HMC**, whose likelihood and analytical score must first be
accurate enough for inference.

Terminal execution status (9 October): the source04 master stopped normally
at the authorized **48 GPU-worker-hour ceiling**, including the owner's
additional 24 hours. It completed 27/90 jobs: 17 current-source comparisons and
10 explicitly retained earlier comparisons. Nine jobs retain partial rows,
one started without completing a comparison, and 53 were not started.
The campaign is **INCOMPLETE** and no worker is running.

See the [terminal review](artifacts/surrogate-hmc-marginal-expanded-20261006-04/terminal-review-20261009.md),
[current checkpoint](artifacts/surrogate-hmc-marginal-expanded-20261006-04/checkpoint.md),
[per-cell results](artifacts/surrogate-hmc-marginal-expanded-20261006-04/results.md),
and [budget extension](surrogate-hmc-marginal-budget-extension-2026-10-07.md).
Engineering and repeated N4096 precision checks passed. Completed component
comparisons do not establish HMC readiness; inconclusive KSC cells, invalid
M13 baselines, simple-comparator promotion vetoes and missing grid coverage
remain explicit. Budgeted resumption would continue this same plan. Earlier
execution entries below remain historical records.

## Question and scope

Does replacing ancestor-conditioned weights by exact Gaussian marginal-mixture
weights preserve LGSSM and KSC value/score accuracy and reproducibly reduce
range–bearing errors on fresh data? Does the extra computation buy accuracy
relative to inexpensive filters?

The requested horizons are T = 10, 20, 50, 120, 240. LGSSM dimensions are
d = 1, 3, 10, 20, with observation dimension d. For d > 1 use the explicitly
defined extension of the existing d=3 law: A has diagonal 0.6 and adjacent
off-diagonals 0.1; F = theta[0] A, Q = exp(2 theta[1]) I, H = I, R = 0.5 I,
and the stationary initial distribution. This preserves the existing d=3
model, and its largest eigenvalue is below 0.8. The three parameter points
(1,0), (0.8,-0.5), (1.2,0.3) are stable. LG1 retains its existing scalar AR(1)
law and parameter points 0.4, 0.8, 0.9.

The existing KSC model has d=m=1 and range–bearing has d=4, m=2. All five
horizons apply to both. Applying d=1,3,10,20 literally to range–bearing would
change the statistical model, and has been raised with the owner. Pending that
answer, implement the unambiguous LGSSM grid and the original nonlinear models.
Do not describe a dimension as tested by padding states or silently replacing
the model. Any approved extension must state equations and independent
references before its execution. Report each model, dimension, horizon, and
parameter point separately; these points are evaluation regimes, not different
data-generating truths. Truth follows the existing model specification.

## Evidence contract and intent ledger

Candidate: the committed optional `marginal_mixture` policy. Baseline: the
same shared filter with `ancestor`, identical observations, initial particles,
innovations, reset designs, and frozen numerical controls. Neither changes the
default. Both carry UKF covariances, Contract E–Chol reset and dual-cap GenUT.

Primary metrics are absolute total log-likelihood error in nats and
0.1 times physical-score RMS error. Record raw scores, raw signed errors,
per-observation errors, and reference uncertainty as explanatory quantities.
The local non-inferiority margin remains 0.05 for each primary metric; it is
deliberately not multiplied by horizon or dimension. This is a stringent
component materiality hypothesis, not an established posterior tolerance.

Use 32 paired particle designs per fresh dataset and initial four datasets,
then eight and sixteen cumulatively if confidence intervals remain unresolved.
Average paired error differences within each dataset before computing
cross-dataset uncertainty. Use paired Student intervals with familywise
alpha=0.05 divided across every predeclared model/regime/N cell, two metrics,
three looks, and two tails. Show dataset-clustered bootstrap sensitivity and
leave-one-dataset-out sensitivity. Four datasets give fragile tail estimates;
agreement of these approximations does not prove their coverage. Never pool
designs as if they were independent datasets. An upper bound below 0.05 supports
local non-inferiority; below zero supports lower error for that cell. A lower
bound above 0.05 supports deterioration. All other outcomes are inconclusive.

Hard promotion vetoes: nonfinite/invalid filter outputs, failed analytical
score checks, unsettled independent reference, missing required calibration,
or underperformance against a constructed simple comparator. Candidate failure
does not stop the grid. Continue other scopes and the particle-count repair
ladder. Continuation vetoes are corrupted or mismatched data/source artifacts,
incorrect model/reference mathematics, insufficient memory even for a bounded
repair, or the overall time cap. Record unfinished work explicitly.

This is a controlled numerical-method comparison, not a canonical LEDH tuning
admission, default promotion, proof of universal improvement, or surrogate-HMC
posterior certification. Scope controls are repository-generated comparison
records, not public tuner admission artifacts. If canonical admission is later
requested, the public artifact authority and downstream HMC checks are required.

## Execution and references

1. Update the main LaTeX algorithm chapter and the standalone proposal's current
   implementation status. Preserve the historical derivation, hypotheses, and
   mathematical limitations. Build and inspect a PDF of the updated chapter.
2. Engineering: true multiple-row consumer parity against separate calls for
   both policies at distinct parameters in all three models; every parameter's
   finite differences in FP64. Repeat M13 N=4096 FP32/TF32 versus FP64 on the
   same FP32-rounded inputs, including repeats and explicit validity. Precision
   deltas are warnings when small relative to component error/materiality;
   nonfinite results or inconsistent finite-program derivatives are vetoes.
3. For every (model,d,T,N) scope, evaluate the two previously declared numerical
   control hypotheses on fresh calibration and separate validation data. Select
   one shared control set by the smaller worst-method reference error on
   calibration, among settings numerically valid on validation. Freeze before
   claim data generation. This is an ablation of weights. It does not establish
   that these are the best controls; failure triggers a recorded future tuning
   repair, not transfer of another scope's chosen settings.
4. Run the fresh-data replication grid at N=1024, then N=4096 and N=16384,
   subject to measured resource feasibility. Preserve partial progress per
   dataset, per design, and per scope. Refresh the next-phase file after every
   completed or rejected scope, including remaining budget and unfinished work.
5. LGSSM: exact covariance Kalman likelihood and reference-only differentiation.
   KSC: analytical integration score with grids 512/1024 and extent 18/24;
   increase to 2048/extent32 if the refinement tolerance fails. Range–bearing:
   quadrature at T=1 as a reference check (not a substitute for T>1), then
   replicated PF/Fisher references at 65536 and 131072 particles, 16 independent
   replicates. Escalate to 262144 when refinement or MCSE affects a verdict.
   Use independent random seeds between reference rungs and candidate designs.
   Report genealogical score variance, refinement bias risk, and likelihood
   weighting of reference scores. A shift within MCSE is not zero-bias proof.
6. Construct three simple adversaries for every model: prior trajectory
   importance sampling (no observation adaptation), bootstrap PF (resampling
   without a flow), and a Gaussian filter (exact Kalman, moment-matched KSC,
   or EKF for range–bearing). These answer whether flow complexity earns its
   cost. Report separately by regime, with missing scores explicitly missing.
   Equal-runtime comparisons use measured steady-state time, excluding tracing
   from one comparison and including it in a separate cold-start comparison.
   Average independent likelihood estimates in probability space when matching
   a runtime budget; never average log likelihoods and call that a PF estimate.

## Defaults, risks, and early checks

| Choice | Provenance and status | Risk | Early diagnostic |
|---|---|---|---|
| FP32/TF32 GPU/XLA | owner production direction | score/precision drift | matched N4096 precision and FD |
| Two control sets | October 5 warm-start hypotheses | wrong at long T/high d | per-scope calibration/validation, status fields |
| Shared controls | isolates the weight-law change | favors one arm; not best tuned comparison | report both calibration losses |
| 4→8→16 datasets, 32 designs | requested replication ladder | few clusters, uncertain tails | adjusted intervals, bootstrap/leave-one-out sensitivity |
| 0.05 and score scale 0.1 | earlier local materiality hypotheses | cannot imply posterior adequacy | raw/per-T errors and no HMC claim |
| N1024/4096/16384 | proposed resolution ladder | quadratic mixture/reset cost | timed disjoint pilot, peak memory, hard time cap |
| M13 large PF reference | independent method | finite-N bias/genealogical collapse | replicate MCSE, refinement and sensitivity |
| LG banded dimension extension | retains existing d3 law | not every high-dimensional LGSSM | stability/model-shape and Kalman checks |

The current two control families keep all correction safeguards on. Their
numerics-altering caps and ridges are hypotheses, not recommended defaults.
This campaign does not tune them on claim data or treat low likelihood error
as their safety justification. Emit all existing validity and correction
diagnostics to make cap activity and numerical margins inspectable.

## Resources, commands, and recovery

Use only the otherwise idle RTX 4080 SUPER,
`GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, with verified memory growth,
TensorFlow/TFP in `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
Initial campaign ceiling: 24 elapsed GPU-work hours, including compilation,
calibration, references, and at most two localized infrastructure repairs per
scope. This is a ceiling, not a forecast that the complete grid fits. Forecast
after the first scope and the first d20/T240 calibration. Do not silently
reduce the requested datasets, dimensions, horizons, or 32 designs to fit.
If the forecast exceeds the ceiling, save the measured estimate and exact
remaining scope while continuing useful authorized scopes within the cap.

Master: `scripts/run_surrogate_hmc_marginal_expanded.py`; engineering and scope
workers run as bounded subprocesses. Active continuation output root:
`docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03/`.
Attempts 01 and 02 remain preserved. Attempt 03 carries all elapsed worker
budget, reruns the affected LG evidence and engineering checks, and retains
only the unchanged M13 precision result as completed work. The localized
repairs and reporting review are documented below.
Every attempt gets a new subdirectory. Manifest records Git revision, source
hashes, exact command, environment, GPU/memory policy, all seeds, observations
checksum, controls, wall time, plan and result paths. Resume must verify source
and configuration; it cannot overwrite evidence or silently mix algorithms.
Exact invocation (trusted GPU execution, repository working directory):

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_marginal_expanded.py --output docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03 --budget-seconds 86400

The parent gives a scope worker at most two hours per invocation and other
workers at most one hour, always within the remaining total. A timed-out
scope preserves completed designs; it remains unfinished, and can resume.
At most three attempts per job include the original and two localized repairs.
The --stage option executes a subset but never declares the campaign complete.
Even a completed currently defined grid explicitly retains the pending nonlinear
dimension extension.

Equal-runtime replication counts are fixed from a separate timing measurement,
before drawing the likelihoods being averaged; cap at 256. Report actual cost
ratio and label a match only within 20% of the target budget. The 20% window is
a timing-report convention, not an accuracy tolerance. Missing matches cannot
pass the heuristic comparison. Report compilation costs separately.

Engineering also compares fresh T=1 range-bearing quadrature at 256/512
intervals with the larger replicated PF references. A disagreement beyond
twice the recorded three-MCSE-plus-refinement sensitivity plus 0.001 triggers
reference repair before continuing. This is not a proof of long-horizon PF
reference accuracy.

## Skeptical audit before launch

Material defects in a literal reuse of the old master: it only supports seven
old scopes; LG dimensions beyond 3 are constructed incorrectly; dataset seeds
are fixed; uncertainty is conditional on one dataset; timing lacks equal-cost
comparators; the nonlinear dimension request is undefined. Repair these before
the expanded grid. The new plan separates reference error, weight-law error,
precision, and downstream inference, and retains candidate failures. A
successful command alone cannot close any scientific question. Final audit,
focused tests, measured feasibility, and owner dimension clarification are
recorded below as they become available.

Documentation reader: a reader familiar with Gaussian state-space models and
importance sampling must reconstruct the full finite algorithm and its total
derivative from the chapter. Git f5e69d716 protects the pre-edit main chapter;
ordinary SHA-256 copies preserve the untracked proposal. Human prose feedback
is pending and does not block the requested drafting or numerical work.

### Implementation audit, 6 October

Static audit passed for bounded engineering execution. The driver preserves
the production analytical call chain and requests all correction diagnostics;
the comparison changes only the weight policy within each paired scope.
Fresh dataset IDs change simulation and design seeds independently of partition.
The cluster analysis uses datasets as units, includes reference sensitivity,
and rejects incomplete designs or invalid references. Missing heuristic results
remain incomplete. Timings are not censored; warm timings are measured explicitly.
Subset completion cannot be confused with completion of the requested campaign.
Eight focused CPU-only regressions passed (3.88 s), covering dataset independence,
all four LG dimensions and stability, exact precision-input parity, dataset-level
uncertainty and invalid/missing evidence handling; driver syntax compilation passed.
These are engineering checks, not numerical accuracy evidence. GPU derivative,
multiple-row and precision checks are the next required executable audit.
The unresolved nonlinear dimension definition and possible compute shortfall
remain explicit; neither authorizes silently reducing the requested grid.

### Localized harness repair audit

The first execution exposed two reference factory calls made inside a traced
graph. KSC mixture validation and range-bearing fixture serialization require
host tensors; symbolic execution failed before those comparisons were usable.
Move fixture construction outside tracing, pass observations as explicit stable
graph inputs, and check parity with the original bound-data API on two datasets.
The LEDH value/score implementation, dataset seeds, controls, primary criteria,
and holdout separation are unchanged. Regression checks must cover every
reference/comparator consumer, including KSC and range-bearing, before resuming.

The first-run simple comparators used FP64 while LEDH used FP32/TF32.
Their numerical values remain diagnostic comparisons, but those times cannot
establish a computational advantage. Correct the simple comparators to
FP32/TF32 on the same GPU/XLA backend. Retain FP64 for independent accuracy
authorities. Repeat the already completed scopes' comparators with the same
saved observations, preserving their LEDH results; do not tune on those data.
GPU timing is sequential on the dedicated RTX4080. FP32 comparator values are
checked against the FP64 authorities and their status/precision are recorded.
The bootstrap and prior filters return likelihood only; equal-runtime results
therefore compare likelihood accuracy under a compute budget, and do not claim
equivalent score capability or superiority for HMC.

Preserve attempt 01 as the original evidence. A new versioned root imports only
unchanged numerical evidence with original provenance, charges all spent worker
seconds and repair probes to the same 24-hour cap, and reruns the failed scopes.
No posterior admission, numerical safeguard setting, data law, or primary
promotion criterion changes. Repaired reference construction must pass dynamic
observation/legacy parity, shape/dtype, finite-value, and no-retracing checks.
This audit permits the bounded regression; continuation requires it to pass.

Repair verification: 11 CPU/XLA regressions passed (21.05 s test time); the GPU/XLA probe passed all 12 model/reference consumers on two datasets (38.09 s numerical time). Dynamic wrappers traced once per fixed signature and matched the original bound-data values/scores. FP32 simple comparators were finite. The active continuation is attempt 02; migration.json charges 4998.32 prior/repair seconds to the unchanged 86400-second budget. The original three completed LGSSM accuracy scopes remain evidence; their computational comparators are refreshed.

### Initialization diagnostic after the first high-dimensional failures

Saved d10/d20 failures report NONFINITE_INPUT at INITIAL, before any filtering
time. The stationary LGSSM adapter deliberately returns NaNs when its checked
Lyapunov solve fails. A bounded diagnostic therefore compares that adapter
across d=1,3,10,20 and all declared parameter points, in FP32 and FP64, with
an independent symmetric-eigenbasis stationary-covariance calculation.
The hypothesis is a numerical rejection in initialization, not failure of
marginal weighting. CPU-only XLA provides a control; an isolated GPU/XLA probe
with TF32 on/off discriminates device precision if needed.

Budget: at most five GPU-work minutes, charged to the existing ceiling.
No concurrent campaign GPU timing during a GPU probe. Preserve/drain the current
worker before GPU diagnosis; ordinary CPU smoke checks may proceed independently.
Record validity, stability, Lyapunov residual, precision, and covariance/tangent
errors. No numerical tolerance or production method changes are authorized by a
successful probe alone. A localized implementation repair requires a recorded
no-harm regression, fresh scope calibration, and source-versioned continuation.
The diagnostic cannot establish likelihood accuracy or posterior convergence.
Skeptical audit: a small residual alone cannot certify stationarity; the
independent eigenbasis check includes stability and matches rounded inputs.

### Stationary initialization repair audit

The isolated GPU probe reproduces all four d10/d20 rejected parameter points
with TF32 on. The same 24 dimension/parameter/precision cases pass on CPU and
on GPU with TF32 off. In the failed FP32 cases the stability certificate and
positive-definiteness checks pass; the reported equation residual is
0.000154--0.000454, larger than 64*d*FP32-epsilon. An accepted high-dimensional
case has initial-tangent relative error about 0.000217 against the analytic
sine-basis solution; GPU without TF32 reduces the maximum to 0.00000136.
This is a precision mismatch in initialization, not evidence against either
importance-weight formula. The invalid-scope outcomes are superseded for
interpretation and must be rerun after repair.

Repair only the small stationary calculation's matrix products: compute
F'F, P-FPF'-Q, and dF*P*F'+F*P*dF'+dQ in FP64 when inputs are FP32.
Keep the covariance and tangent Lyapunov solves, their returned tensors, and
the full particle filter in their original dtype. Keep TF32 enabled for the
filter. The equation-residual tolerance remains 64*d*input-epsilon; do not
loosen it. The exact target is P=FPF'+Q and
dP-F*dP*F'=dF*P*F'+F*P*dF'+dQ. These products are O(d^3) with O(d^2) matrices,
independent of the particle count; the modification does not introduce an
FP64 particle cloud or transport matrix.

Skeptical audit: merely relaxing the residual guard would hide derivative
error, so the repair improves the checked calculation instead. No damping,
ridge, model law, likelihood weight, or materiality threshold changes.
The independent oracle uses analytic sine eigenvectors for the declared
tridiagonal matrices; generic XLA eigh has its own iterative tolerance and was
replaced only in the diagnostic oracle. Preserve all prior artifacts.

Acceptance: every declared stationary case valid on GPU/TF32; covariance and
tangent agreement with the analytic solution within a conditioning-aware
FP32 rounding bound; unstable inputs still rejected. At the actual full-filter
consumer, LG1/LG3 values and scores must agree with the protected old adapter
within 0.00005 relative/absolute numerical tolerance, while KSC/M13 remain
identical. Check d10/d20 at N1024 beyond initialization and retain any new
stage failure. CPU regressions and GPU checks are diagnostic, not promotion.
Budget at most ten additional GPU-work minutes under the original 24 hours.
Then use a new source-versioned continuation, rerun affected accuracy scopes,
and use calibration/validation dataset ID 1 (claim partitions remain separate).
Do not reuse rejected initialization scopes as completed accuracy jobs.

### Final continuation audit

Repair checks: the isolated GPU/TF32 initialization passes all 24 cases.
Seventeen CPU regressions pass. Full-consumer LG1/LG3 and KSC checks pass;
largest scaled change 3.58e-6 in LG3, KSC unchanged. The tiny fresh-data
M13/N32 fixture is invalid in both old and repaired versions with identical
values, scores, and status; it cannot certify valid M13 filtering. Therefore
the new continuation must first rerun the previously valid engineering
fixtures, including M13, before any accuracy scope. The existing mandatory
engineering stage supplies this check within the campaign cap.

All high-dimensional initialization checks pass. At T2/N1024 all ancestor
rows are valid; marginal rows have remaining factorization failures at the
WEIGHTS stage (d10: two of three points; d20: one of three).
These are later failures, not recurrence of the initialization bug.
Preserve them and continue planned calibration and particle-count repairs.
No safeguard is relaxed and no failed row becomes accuracy evidence.

Final code review also found that one matched timing row could make an
incomplete equal-runtime comparison appear complete. Require every planned
row to have a finite comparator and matched budget before reporting no
observed heuristic loss, and emit usable/expected row counts. This affects
only completeness reporting, not filter or reference outputs. A focused
regression checks partial coverage. Missing evidence remains incomplete.

The source-versioned attempt 03 reruns every affected LG accuracy scope and
the feasibility pilot. It reruns the engineering stage, preserves only the
unchanged M13 precision result with an explicit provenance link, and uses
fresh calibration/validation dataset ID1. Earlier results stay preserved.
The new 24-hour budget carries all prior spent work plus 60 seconds for
localization and 540 seconds for the repair probes (startup included).
Idle scheduler hold is not GPU computation. Document its interval separately.


## Review record and current execution status — 6 October 2026

The plan survived a second skeptical audit before the expanded run. The exact
baseline is the same source implementation with the ancestor policy, the
candidate changes only the importance-weight policy, and both use the same
observations, base innovations, reset designs and frozen controls. Exact
Kalman values and analytical scores are the LGSSM authority; refined
one-dimensional integration is the KSC authority; the range--bearing reference
is explicitly finite-particle/quadrature based and carries MCSE and
genealogical-degeneracy limits. The primary metrics are error relative to
those authorities. Validation loss, acceptance and runtime cannot substitute
for the declared accuracy comparisons.

The audit found and repaired two material harness problems before this
continuation: symbolic reference construction inside traced graphs and an
FP64/FP32 mismatch in comparator timings. The stationary-initialization
precision failure was then diagnosed and repaired by using FP64 only for the
small checked matrix products. No residual tolerance, model law, weight
identity, or promotion margin was relaxed. Focused CPU and GPU checks passed,
and the actual consumer call chain was exercised. The remaining high-
dimensional mixture-factorization failures are validity failures of candidate
scope executions; they are not silently converted into accuracy results or
continuation vetoes.

Attempt 03 is the active execution of this plan. It has completed engineering,
the T=10 LGSSM/KSC/M13 scopes, and LG1 at T=20 at N=1024; LG10 and LG20 at
T=10 currently have no numerically valid shared controls and therefore no
accuracy result. The active worker is continuing the declared scope order.
Every completed scope refreshes next-phase.json; the current state and
remaining budget are in the versioned artifact root
docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03/. The
24-hour ceiling is still active and does not imply that all 90
(model, horizon, particle-count) scopes will fit. A timed or rejected scope
preserves its partial datasets and remains unfinished.

The completed evidence has the following interpretation. LGSSM d=1 at T=10
and T=20 and d=3 at T=10 met the component non-inferiority screen in all
nine parameter cells; adjusted intervals support lower score error in five
cells and lower value error in two. KSC T=10 has two non-inferior parameter
points and one inconclusive point. M13 T=10 has lower paired value and score
error at both points, but bootstrap and EKF likelihood errors are lower in
the observed rows, so the constructed-heuristic gate vetoes promotion. The M13-versus-ancestor rankings are supported under the stated approximate
cluster-interval procedure; the simple-comparator means are descriptive.
Neither establishes an HMC conclusion. The exact values and intervals remain in each
attempts/*/result.json.

The plan therefore passes as a controlled verification plan with these
explicit limits: native KSC and range--bearing dimensions remain their
specified d=1 and d=4 models; a literal d=1,3,10,20 nonlinear extension has
not been invented; the particle ladder and longer horizons remain pending;
and no posterior draws, acceptance, ESS, R-hat, or endpoint-target test has
been run. The next action is continuation of the same master, followed by a
terminal result review after the time cap or full feasible ladder.


## Bounded localization of the remaining high-dimensional failure

The repaired stationary initializer now passes, but d10/d20 calibration rejects
the marginal arm at first-observation WEIGHTS factorization. This is a repair
trigger, not evidence against the marginal identity. Before changing numerics,
capture the actual component Q and pushed covariance B Q B' from the existing
consumer. Use the first observation/innovation of the unchanged T10 calibration
dataset 1, design 0, control 0, N1024, each existing parameter point. Instrument
only a diagnostic process; no source implementation or guard is modified.

Question: does the covariance fail because TF32 product rounding breaks the
existing relative symmetry condition, because it loses positive definiteness,
or because another finite-output condition fails? Comparator: the same consumer
with TF32 disabled, and the captured B/Q product evaluated in FP64 and cast to
FP32. Record exact guard results, relative asymmetry, symmetric-part eigenvalues,
all finiteness flags, final consumer status and source hashes. The symmetric
part is used only for diagnostic eigenvalues; it is never returned to the filter.

Success means attributing the failure from actual captured inputs. It does not
promote a protection or imply accuracy. Invalid capture or nonmatching prefix
status leaves attribution unresolved. Any numerical repair requires its own
paired no-harm and derivative checks followed by fresh calibration. TF32-off
is a reference exception, not a new default. Maximum 600 GPU-worker seconds,
charged within the existing 24-hour ceiling. Run exclusively on the campaign
4080, allowing the current worker to finish before this diagnostic; automatically
resume master dispatch afterward.

Skeptical audit: the covariance is taken from the claim-bearing consumer, not
an invented matrix; T1 is a diagnostic prefix, not a new tuned scope. No
symmetry guard or margin is relaxed. A precision comparison cannot establish
posterior correctness. Instrumentation may affect compiler optimization, so
full unchanged-program confirmation is required before attributing a repair.

Command:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/probe_marginal_covariance_failure.py --output docs/plans/artifacts/surrogate-hmc-marginal-covariance-diagnostic-20261006-01`


## Runtime measurement review and diagnostic

A source audit found that current seconds includes recursive serialization
of every returned tensor. LEDH emits many correction diagnostics; bootstrap
returns a much smaller record. These are report-ready harness timings, not
comparable kernel-only timings. In addition, one-call timing predicts repeated
bootstrap cost poorly: the median actual/target ratio is 0.65 for KSC and 0.63
for M13 at T10. Only 69/1536 and 33/1024 rows meet the declared 20% window.
The repeat-count cap explains only one M13 mismatch. Accuracy observations
remain valid, but the existing cost comparison cannot establish computational
value. No failed match will be relabeled as a match.

Before changing the campaign, run a separate 180-second GPU timing diagnostic,
in the same exclusive slot as the covariance probe. On KSC/M13 T10, N1024,
existing scope-frozen controls, validation dataset 1/design 0/point 0, compare
twelve warmed calls using (i) synchronization on the XLA value output before
report serialization and (ii) the current full report-ready timing. Record
both timing components for the real LEDH and bootstrap consumers. This is
debugging evidence about timing overhead, not a speed ranking. No method,
data, controls or primary accuracy criterion changes.

The exclusive diagnostic slot is therefore bounded by 600+180 seconds,
inside the original total budget. The master resumes automatically afterward.
A later runtime comparison must calibrate repeated-call cost on disjoint
timing draws, freeze the repetition count before accuracy draws, report both
kernel and end-to-end costs, and preserve the value-only versus value/score
workload distinction. Its primary requirement is complete budget matching;
the current incomplete subset cannot satisfy it.

The saved cost-sensitivity.json explores T-linear and N-linear/quadratic
scenarios from measured report-ready timings. Even the optimistic N-linear
case for four datasets across only the four native/basic scopes is about
96 hours for the filter pairs alone; this excludes references, comparators,
compilation and dimensions 10/20. It is a scenario for this diagnostic harness,
not a calibrated forecast of kernel scaling. The initial 24-hour ceiling
cannot be presented as sufficient for the full requested grid.


## Diagnostic outcomes — 6 October 2026

The bounded covariance probe completed in 321.7 seconds on the campaign
RTX 4080 SUPER. It captured the actual first-observation inputs from the
claim-bearing T=10 consumer for LG10 and LG20, all three parameter points,
with TF32 on and off. With TF32 enabled, the failing LG10 points had relative
pushed-covariance asymmetries of \(1.5321\times10^{-5}\) and
\(1.2550\times10^{-4}\), against the d=10 guard tolerance
\(9.5367\times10^{-6}\). The failing LG20 point had asymmetry
\(1.2581\times10^{-4}\), against \(1.9073\times10^{-5}\). The positive
symmetric-part eigenvalues and finiteness checks remained valid. With TF32
disabled, every LG10/LG20 point was valid; the largest observed asymmetry was
\(6.14\times10^{-8}\). Re-evaluating the captured \(BQB^\top\) product in
FP64 and casting to FP32 was valid for every point. The untouched consumer
status matched the instrumented status.

This attributes the observed rejection to TF32 product rounding at the
existing symmetry guard. It does not establish that FP64 recomputation is a
safe default. It is a Class C numerical candidate and therefore remains
unimplemented pending a paired no-harm regression, total-derivative check,
fresh calibration, and a new source-versioned continuation. TF32-off remains
a diagnostic/reference arm, not a campaign default. The retracing warnings
emitted by the diagnostic are instrumentation overhead and do not alter the
attribution; an unchanged-program check is still required before any repair.

The synchronized runtime probe completed in 81.9 seconds after warm
compilation. For KSC T=10, N=1024, the median marginal-mixture call took
0.05497 s to synchronize the XLA value and 0.08846 s through report-ready
serialization; bootstrap took 0.00292 s and 0.00294 s, respectively. For
M13 T=10, the corresponding medians were 0.16341 s and 0.19733 s for the
marginal mixture, versus 0.00240 s and 0.00242 s for bootstrap. The
serialization component was about 0.033 s for LEDH and negligible for
bootstrap. These are timing-localization measurements, not a speed ranking:
the current report-ready harness records unequal diagnostic payloads, and no
complete equal-runtime accuracy comparison has been established.

The supervisor resumed the unchanged master after both diagnostics. The
remaining scope and particle ladder therefore continue under the original
24-hour ceiling. These findings do not add HMC posterior evidence, alter the
candidate/default policy, or authorize a numerical repair by themselves.


## Pushed-covariance precision candidate: bounded repair evaluation

The symmetry diagnosis triggers the existing numerical-repair step. Evaluate
a diagnostic-only implementation that computes B Q B' and its complete
analytical tangent in FP64, then casts both to the filter's input dtype. All
other operations, guard tolerances, observations, seeds, controls and weight
laws are identical. No ridge, damping, symmetrization, model-specific branch,
or TF32-global change is introduced. This is a Class C numerical candidate:
same mathematics but different rounding, so it must demonstrate non-harm
before source adoption.

Use T10 calibration/validation dataset 1 with N1024 and existing control 0.
Check original versus candidate full consumers for LG1/LG3/KSC/M13, all existing
parameter points, two particle designs. Require unchanged validity and, for
previously valid rows, absolute value and 0.1-scaled score differences below
0.005; report stricter scaled differences as well. This engineering screen
uses one tenth of the campaign materiality margin, is not statistical
non-inferiority, and cannot establish posterior adequacy. The reference FP64
program must be identical because the product operation already uses FP64.
Check d10/d20 at T10 across all parameter points with the same inputs; recovery
of t=0 WEIGHTS rejection is required, and every later invalidity is retained.
Compare analytical scores with symmetric finite differences in FP64 at a
bounded N32/T2 d10 fixture for all parameter directions. This is diagnostic
parity, never the claim-bearing score path.

Budget: at most 900 GPU-worker seconds, charged to the original total, run
exclusively between campaign workers on the same RTX4080. A diagnostic
monkeypatch affects this process only; the active master source remains frozen.
The supervisor always resumes the unchanged master. Output:
docs/plans/artifacts/surrogate-hmc-marginal-covariance-repair-20261006-01.
Command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/probe_marginal_covariance_repair.py --output docs/plans/artifacts/surrogate-hmc-marginal-covariance-repair-20261006-01

Skeptical audit: a valid t=0 factorization alone is insufficient; full T10
consumers and low-dimensional no-harm checks are required. Fixed diagnostic
controls are not tuning admission. Candidate failures are retained rather than
censored. FD noise is assessed in FP64, not mistaken for the TF32 finite
program's differential. Even a passing test requires fresh per-scope
calibration and a source-versioned continuation; no existing accuracy
observation can be silently relabeled as produced by the repair.

## Precision candidate result and narrower repair (12:40 UTC)

The unconditional FP64-product candidate recovered all 24 high-dimensional
T10 cases and passed analytical/FD checks (maximum relative error 7.41e-8).
LG1/KSC were unchanged and LG3 changes were small. It failed the predeclared
no-harm screen on three low-noise M13 rows: maximum value change0.02729,
scaled score change0.11323. It is rejected for broad adoption; this is not
evidence of statistical deterioration, because this diagnostic measured
changes, not reference errors. The original master resumed unchanged.

The next bounded diagnostic is part of the same numerical repair phase:
recompute BQB' and its complete tangent in FP64 only for finite components
whose FP32 product fails the existing normalized symmetry margin8*d*eps.
Use the original FP32 products otherwise. Final covariance guards remain
unchanged; no clipping, ridge, tolerance relaxation, or model-specific fork.
The precision branch is a numerical validity repair, not parameter tuning.
Its analytical derivative follows the selected branch; behavior near the
precision-switch boundary remains a numerical/HMC diagnostic obligation.

Use the same six full-consumer scopes, paired data/designs, .005 no-harm
screen, high-dimensional validity requirement, FP64 equality and FD checks.
A healthy original full trajectory must remain within the original screen;
a recovered invalid trajectory is reported separately. Budget<=900seconds,
within the original24GPU-worker hours, no new approval boundary. Run after
the current worker; supervisor resumes the unchanged master on every exit.
Output: docs/plans/artifacts/surrogate-hmc-marginal-covariance-selective-repair-20261006-01.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/probe_marginal_covariance_selective_repair.py --output docs/plans/artifacts/surrogate-hmc-marginal-covariance-selective-repair-20261006-01

Skeptical audit: a high-dimensional rescue cannot excuse changing healthy
M13 trajectories. The precision mask uses the exact existing symmetry
criterion, and the original final factorization must still pass. This
diagnostic cannot establish score accuracy or a smooth HMC target. It is
a localized implementation repair test with preserved rejected evidence.

## Equal-runtime calibration repair evaluation

Before changing the master, test synchronized value/score/valid timing with
report serialization outside the timed region. Warm both methods. Use the
median of five marginal calls and nine bootstrap calls to initialize the
bootstrap repeat count; refine using three independent blocks for at most
three timing-only attempts. Freeze the count before any evaluated estimates.
Bootstrap aggregation is an arithmetic likelihood mean, computed as logsumexp
minus log(count), and its cost is included. The repeat cap remains256.
No stopping or count selection may depend on likelihood accuracy.

The diagnostic uses KSC/M13 T10,N1024, point0, validation dataset1, designs0..7,
and the already frozen per-scope controls. Calibration and evaluated random
seeds are disjoint. Pass requires every row finite/valid and actual runtime
ratio within[.8,1.2]. A failed match is preserved, never replaced selectively.
This is calibration evidence only, not a statistical speed/accuracy ranking.
The workloads remain different: LEDH computes value and score, bootstrap value.

Budget<=300GPU-worker seconds inside the existing ceiling, exclusive after
the next short LG10T50 worker and the covariance diagnostic have finished.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/probe_marginal_runtime_budget.py --output docs/plans/artifacts/surrogate-hmc-marginal-runtime-budget-20261006-01
Skeptical audit: synchronize score/status as well as value, include arithmetic
aggregation, avoid cold compilation, and preserve all mismatches. Even a
passing two-scope test does not imply complete timing coverage on the grid.

## Bring scheduled repairs forward at the saved T50 checkpoint

The M13T50 worker has completed and saved one fresh dataset; subsequent rows
are saved by atomic replacement. Interrupt only that owned worker to run the
already planned precision/timing diagnostics now. Preserve every completed
JSON and all partial rows, classify the interruption as infrastructure
scheduling for the planned repair, and keep its full wall time in the campaign
budget. This does not reject either method or change the scientific question.
The existing supervisor runs the precision diagnostic and resumes master
dispatch; the runtime diagnostic then occupies the next short worker slot.
Skeptical audit: atomic persistence is verified in marginal_weight_campaign.save;
the exact PID and command are checked before interruption; no unrelated GPU
process is touched, and no saved result is deleted or rewritten.

## Source continuation after selective precision checks

The selective repair passed all six T10 scopes, recovered the 12 formerly
invalid high-dimensional cases, and left every previously valid tested value
and score identical. Adopt only after draining the worker and closing source
version03. The final covariance validity test stays unchanged. The precision
switch has a piecewise derivative; this test does not establish HMC behavior
near its switching boundary.

The synchronized timing experiment failed for KSC: its final warm calls were
roughly three times faster than the early calls used to set the budget. M13
matched on all eight evaluated designs. Preserve this failed experiment.
Repeat in a fresh diagnostic root with bounded adaptive timing warm-up: at
least20 calls, at most60, and two successive five-call medians within10 percent.
This is a runtime-stability diagnostic, not sampler warm-up or likelihood
tuning. Retain the [.8,1.2] matching criterion, all eight evaluated designs and
repeat cap256. Failure leaves runtime claims unresolved without discarding
accuracy evidence. Allow300 seconds from the same total budget. Output:
docs/plans/artifacts/surrogate-hmc-marginal-runtime-budget-20261006-02.

Version04 uses new calibration/validation dataset2 and untouched claim dataset
indices100 onward. Version03 comparisons retain their actual source identity.
Carry all worker and diagnostic time into the original86400-second ceiling.
Continue failed and unfinished scopes first; completed valid version03 scope
comparisons are not silently rerun or relabeled. N4096/16384 remain scheduled.

Skeptical audit: do not relax covariance validity. Check healthy-row equality,
analytical derivatives, batch wiring and source identities. Failed claims
must not become tuning data. Exclude compilation and report serialization from
warm latency; unmatched costs cannot support equal-runtime claims. A completed
job with an invalid candidate is not a successful comparison. These repairs
remain part of the existing plan and budget.

The actual source integration check compares the new public consumer against
the saved selective-repair experiment on identical calibration/validation data
and designs. Budget600 seconds, same campaign. Require the earlier no-harm
margins (.005 nats and .005 scaled score), every high-dimensional test valid,
and nonzero persisted repair counts for every recovered row. Independent
finite differences and batch parity are rerun by the master. This catches
call-chain and XLA changes introduced when preserving repair diagnostics.
Timing calibration has a60-second cap per warm-up/adjustment block; a timeout
marks timing unstable and forbids equal-runtime promotion, while accuracy
sampling can continue. No inference from an unmatched cost is authorized.

Actual integration preserved LGSSM/KSC within the .005 screen and recovered
all high-dimensional tests, but two M13 rows moved beyond that screen after
adding precision-count outputs. Do not promote this source yet. Test the
observability hypothesis on all eight original M13 rows, using the saved
score/batch source independently of the repaired mixture routine. Two bounded
variants (old batch alone; old score and batch) have a combined300-second cap.
Keep the same data and no-harm criterion. This is implementation diagnosis,
not new tuning or permission to relax the screen.

## Baseline recompilation check before interpreting the integration difference

Both observability variants still failed the original no-harm screen; removing
count outputs alone does not explain the discrepancy. Compare two independently
traced copies of the unchanged archived implementation and the integrated
candidate in one process on the same eight M13 T10 rows. Repeat each call once.
Use all three archived modules for the original, and verify their hashes.
Question: is the saved baseline reproducible to the same .005 value/score
criterion across compilations, and is a repair effect distinguishable from
that baseline variability? Do not relax the no-harm criterion. Within-graph
repeatability and between-graph differences are explanatory; a failed baseline
reproduction invalidates attribution to the repair. Budget300 seconds from the
same campaign. Output surrogate-hmc-marginal-recompile-repeatability-20261006-01.
Skeptical audit: a source snapshot must include mixture, score and batch;
forcing get_concrete_function before replacing module bindings prevents a
deferred trace from silently calling another implementation. No claim data
are used. This diagnostic changes neither controls nor the target.

The unchanged baseline recompilation result is now available: within one
process, both independent original graphs and the integrated repair agreed
exactly on all eight rows, including repeat calls. The unchanged original
nevertheless differed from its saved prior-process values by up to .0153022
nats and .198898 scaled score (three rows beyond the same .005 screen).
Thus the saved-run discrepancy cannot be attributed to the repair alone.
Do not claim process-independent FP32/TF32 repeatability.

Repeat the same actual-consumer original/candidate comparison concurrently
traced in one process for all six T10 scopes (68 paired calibration/validation
rows). The archived baseline modules and integrated modules must both be
loaded and traced explicitly before evaluation, preserving their hashes.
Keep the unchanged .005 no-harm screen, all high-dimensional validity checks,
and persisted repair counts for recovered rows. Budget900 seconds, within the
same campaign. Output surrogate-hmc-marginal-integrated-nonharm-20261006-01.
This isolates the implementation change from a saved-run comparison and
preserves the unresolved cross-process precision limitation. A pass permits
the optional component experiment to continue, not HMC-force or default
promotion. The campaign still performs same-N precision/repeat checks.
Skeptical audit: do not discard the earlier failed reproduction; report it as
a separate limitation. This is a paired implementation comparison, not an
accuracy ranking. No promotion margin or mathematical target changes.
