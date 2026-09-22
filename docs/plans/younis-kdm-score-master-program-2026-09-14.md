# Master program: systematic investigation of KDM, LEDH, and model-score estimators

Active execution, 2026-09-22: supervisor PID 4021639 is executing the reviewed
[adaptive R replication ladder](iapf-adaptive-replication-ladder-2026-09-22.md),
accumulating100,300,1000 complete learner labels per dimension under the
remaining budget. The [adaptive comparison](artifacts/iapf-adaptive-score-reference-20260922-01/result.md)
is complete:400 evaluations,729 checks, no rejected/capped learner. Score/tail8
floor distortion disappears from fit2 in all d80 runs, but score/later doubling
fails the d40 descriptive heuristic screen (RMSE.340 vs FA-APF.317). QR/later
has RMSE.120 there. Ten replicates do not support statistical ranking. Early
doubling changes N/cost. The next ladder preserves both reconstructed fitters,
uses the explicit later-doubling hypothesis, fresh data/seeds, exact controls
and uncertainty intervals. No Eq15, author-identity or full-paper replication
claim follows. It runs automatically through all stages unless a validity or
budget veto fires; ordinary candidate losses do not stop execution.
The [paper-model transfer result](artifacts/iapf-paper-score-transfer-20260922-01/result.md)
remains preserved with119 terminal checks and400 final evaluations.
The completed
[covariance/floor factorial](artifacts/iapf-covariance-floor-factorial-20260922-01/result.md)
passes 419 checks and 48 independent R recursions. At d10, removing the .01 floor
reduces diagonal-guide observed MSE from 15.230 to .081; restoring full covariance
with the floor retained gives 12.456. The full/negligible arm recovers the exact
Gaussian oracle to 4.89e-15 in coefficients and 2.85e-14 in likelihood. This
identifies the floor as the larger contributor on the replayed cases, with
covariance effects remaining. No population ranking or paper transfer follows.
The [score-regression result](artifacts/iapf-score-regression-20260922-01/result.md)
is complete: 446 checks pass, all 576 final evaluations complete, and independent
R reproduces the recursive fits. The diagonal, .01-floor candidate nevertheless
fails the conditional heuristic screen at dimensions 5 and 10. Observed mean
squared log-likelihood errors are 1.064 and 15.230, versus .031 and .069 for the
current-observation guide. Four data sets per dimension do not support a
population ranking. Numerical correctness does not imply downstream accuracy.
The completed factorial refits all four combinations. This is a diagnostic local extension,
not Eq15, original-author code, a default change, or paper replication.
The [actual-consumer fit observability](iapf-fit-observability-2026-09-22.md)
phase is complete:24 CPU tests,330 terminal checks, all original outputs exactly
replayed across four cases and14 recursive fit calls. Five appended fields agree
with independent references within4.27e-14; the same candidate rejection remains.
The [oracle-start result](artifacts/iapf-oracle-start-population-20260922-01/result.md)
is complete:57/64 fits meet the unchanged optimizer gate. A converged terminal
d5 density fit moves from the exact diagonal minimum KL0.198 to7.18; a converged
relative-shape fit also reduces its residual while worsening global geometry.
Known-Gaussian controls verify concentration at N1000 and a population
counterexample to inferring small KL from small fitting error. Initialization
alone and density normalization alone are insufficient on these cases.
The [representation diagnosis](artifacts/iapf-representation-support-20260922-01/result.md)
is complete with 63 terminal checks. Full quadratic recovery exactly reconstructs
all 16 Gaussian messages; omitted cross terms explain diagonal-QR coefficient
distortion. Some target-squared criteria give one particle 97% of the weight
and have local curvature ratios near ten million. These are fixed-case
mechanisms, not a population failure rate or a proposed paper-scale full fit.
The completed [independent optimizer comparison](artifacts/iapf-optimizer-isolation-20260922-01/result.md)
checks all 64 fixed problems: R L-BFGS-B meets the unchanged gradient criterion
in 57/64, compared with 48/64 for local projected descent. This is no stochastic
ranking. Converged fits can still have small sampled shape error and large
exact-message KL. All 16 oracle diagonal guides lie inside the original box.
The [fit-input result](artifacts/iapf-fit-input-isolation-20260922-01/result.md)
already proves density-energy escape on actual terminal fits with independently
correct objective/gradient arithmetic. The [guide-geometry result](artifacts/iapf-guide-geometry-20260922-01/result.md)
verifies the FP64 consumer with explicit finite initial-integration error.
Current budget: about45.226 CPU /47.745 GPU hours; the active phase ledger is authoritative.
No defaults changed. Filtering vetoes, TF32 and paper timing/controller/author
choice gaps remain open.

Latest continuation result, 2026-09-22: the reviewed
[adaptive consumer comparison](iapf-adaptive-consumer-parity-2026-09-22.md)
has completed: 1732 final independent checks and 22 regression tests pass.
All ten CPU and ten GPU FP64/XLA adaptive consumers match independently fitted
R guides, full clouds, count/stopping decisions and final finite-program scores.
The scalar restriction is removed for the local linear consumer; nonlinear
scalar restrictions and claim/tuning verification remain. One finite-difference
boundary crossing was resolved by label-stable steps with unchanged tolerances.
[Result and remaining gaps](artifacts/iapf-adaptive-consumer-20260922-01/result.md).
This is conformance of the local adaptation, not paper-scale replication or
clearance of the prior TF32 veto. The now completed
[density-scale diagnosis](iapf-density-scale-diagnosis-2026-09-22.md)
tested absolute-density stopping and initialization against an exactly representable
Gaussian target as dimension grows. It found eight inaccurate zero-step fits
per device at d40/d80. Fixed scaling alone was insufficient, motivating the
now completed initialization isolation. These are closed evidence phases.

Latest completed result, 2026-09-22: the owner-authorized
[renewed mechanism campaign](iapf-renewed-mechanism-campaign-2026-09-22.md)
has completed all four phases and terminal skeptical review. The wider master
program remains open. [Results, decisions and inference status](artifacts/iapf-renewed-mechanism-20260922-01/result.md)
and the [mathematical/source audit](artifacts/iapf-renewed-mechanism-20260922-01/source-and-math-audit.md)
replace the previous next-action instructions below.

The campaign found and repaired a real GPU execution defect: the shared
TF32/XLA transition could effectively reuse the first guide center at later
times. A standalone broadcast matrix product reproduces the error; disabling
Triton GEMM in one diagnostic selects cuBLAS and removes it. The retained
repair computes Kc-Km instead of K(c-m), with the matching analytical tangent.
Fifty-four focused and dependent consumer tests pass. The final FP64 and
ordinary FP32 full-filter comparisons pass 18/18 each. Strict TF32 comparisons
still fail 11/18 from rounding and changed discrete choices; that veto remains.
No numerical backend/default or canonical LEDH route changed.

Earlier GPU/XLA/TF32 evidence using the original affected fitted transition
requires revalidation before reuse. Preserve those artifacts. The finding is
specific to the tested fused route and installed custom compiler build; it
is not evidence that every scalar, FP64, R or other-GPU result was wrong.

The fresh confirmation contains 160 cells and 480 learner attempts. QR and
safeguarded unweighted fitting complete 160/160 each; weighted fitting
completes 158/160 under the frozen 12-iteration cap. Unweighted matches QR to
3.64e-12 with no ridge interventions. Weighted fitting uses 6,650 ridges and
can have weight ESS near one for a 161-coefficient d80 fit. All four estimable
adjusted accuracy intervals contain zero; d80 has no complete paired interval.
Every learned arm fails conditional heuristic screens, so no ranking or
promotion follows. The two caps complete at iterations 13 and 14 in separate
extensions with exactly reproduced original prefixes; confirmation is unchanged.

The new allocation is 48 CPU hours and 48 GPU hours, with no wall deadline.
[Budget accounting](artifacts/iapf-renewed-mechanism-20260922-01/budget.json)
records about 2.072 CPU hours and 4.95 GPU minutes consumed, leaving about
45.887 CPU hours and 47.891 GPU hours after the adaptive comparison. The active
density-scale phase budget records subsequent spending. All failed attempts are charged. No
research worker is running. The old expired deadline and token-style launch
rules below are historical; the remaining authorization does not require
another owner approval for local work under the same scientific scope.

The next master actions, in dependency order, are:

1. Preserve the repaired shared consumer and compiler reproducer. Revalidate
   any affected old TF32 result before using it; evaluate distributional
   likelihood accuracy separately from fixed-random-stream path parity.
2. DONE on tested d1/d2/d5 FP64 CPU/GPU fixtures: close the actual R/TF adaptive-consumer boundary; explicitly match the model,
   X1 versus X0 timing, retained weights/resampling, diagonal guide, objective,
   floor and stopping semantics. Test the entire multidimensional consumer
   against the independent R reference before removing the d=o=1 guard.
   Full consumer evidence now supports this removal; high-dimensional fitting
   quality and the different paper-study model remain open.
3. Keep Equation 15 reconstruction separate from the log-regression extensions.
   The unrestricted density objective has a vanishing-density degeneracy;
   the authors' initialized local solver and numerical floor remain unknown.
   Test any concrete reconstructed procedure as a declared hypothesis, with
   fresh data and the existing oracle/heuristic ladder, before a paper-scale
   replication. Do not spend the remaining budget repeating a solver-validity
   check as if it established likelihood accuracy.
4. After these reference and consumer questions are resolved, return to the
   marginal model-score comparison and then KDM/LEDH integration under their
   own source, tuning and canonical-route requirements. The present result
   establishes none of those downstream claims.

The concise [checkpoint](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md)
contains the exact resume state and evidence paths. The records below preserve
earlier phases; their old next-step text and deadlines are superseded by this
current entry and the newest repository governance.

<!-- IAPF_24H_AUTOMATIC_STATUS -->
Earlier unattended campaign status: frozen campaign completed. Five-dimension study complete: True. All-dimension reference screens: {'qr': False, 'short_qr': False}. No algorithm default changed. [Terminal report](artifacts/iapf-r-24hour-campaign-20260921-01/result.md).
<!-- END_IAPF_24H_AUTOMATIC_STATUS -->

Date: 2026-09-14

Previous result, 2026-09-22: the owner-authorized
[independent validation](iapf-r-independent-validation-2026-09-22.md) is complete.
The [results and terminal review](artifacts/iapf-r-independent-validation-20260922-01/result.md)
record 80/80 paired cells, 160/160 full learners and 880/880 additional
filter/control evaluations. All exact Gaussian controls agree with Kalman
(maximum absolute log error 3.64e-12). Both frozen learners complete the new
cases; the repaired learner's observed mean absolute error is higher in all
five dimensions, but every predeclared paired 99% interval includes zero.
No accuracy ranking is statistically supported by this bounded sample.

Fourteen conditional heuristic comparisons veto promotion of the repaired
reference. Both learners have a tested particle count satisfying the declared
relative-RMSE precision screen at every dimension; measured costs do not
establish an efficiency ranking. The 51,100 recorded repaired fits include
3,331 positive ridges, four active curvature bounds and three active-set
fallbacks; the largest KKT residual is 5.80e-7, below the frozen 1e-6 gate.
Full paper replication and original-author identity remain open. Numerical
execution used 2,209.885535 worker seconds in 84/90 launches, below the
4,100-second cap, finishing at 2026-09-21T20:03:51Z before the original
20:04:26Z deadline. A bounded scheduling repair completed the last two cells
with fresh attempts; the first index/report remain preserved. No worker is
running. Unused compute allocation does not renew the expired wall deadline.

Previous result, 2026-09-22: the owner-authorized
[targeted fitting repair plan](iapf-r-targeted-fitting-repair-2026-09-22.md)
is complete. The [results and terminal review](artifacts/iapf-r-targeted-fitting-repair-20260922-01/result.md)
record all 25 blocked warm starts reaching the optimizer under both tolerances,
while Equation 15 still gives inadequate guide shapes. The explicit
weighted-log/ridge extension completes all eight reproduced d80 failures after
one localized convex-solver repair, as well as all six fresh learning runs;
the six QR comparators also complete. Forty-three focused checks and the
existing author-choice regression suite pass.

Numerical repair is established for those checked cases; an accuracy advantage
is not. The repaired learner has larger absolute terminal error than QR in
three of six fresh comparisons, and conditional heuristic checks veto promotion.
Full paper replication remains open. This is an independent CPU R reference
extension. Execution used 494.012354 worker seconds in 29/30 launches under
the 4,800-second cap and original deadline. The remaining prior allocation is
4,513.601570 seconds, with no implicit deadline renewal. No worker is running.

Prior diagnosis, 2026-09-22: the owner's requested step back is
complete. The [source/code root-cause audit](iapf-r-root-cause-audit-2026-09-22.md)
and [mathematical findings](artifacts/iapf-r-root-cause-audit-20260922-01/result.md)
identify separate fitting failures: actual Equation15 density disappearance;
omitted Gaussian cross terms amplified by concentrated weighted regression;
and QR numerical-rank rejection. Exact saved-cloud replays and the shared
exact-guide APF/Kalman check pass. Crucially, 25/32 F2 failures occurred in
strict QR initialization before the optimizer could use an available valid
previous guide. The implemented F2 variant failed qualification, but those
results do not fairly reject warm starting itself. No worker remains active.

Previously completed, the reviewed
[author-choice hypothesis campaign](iapf-r-author-choice-hypotheses-2026-09-22.md)
is complete. [Final findings and review](artifacts/iapf-r-author-choice-hypotheses-20260922-01/terminal-review.md)
and the [revised numerical report](artifacts/iapf-r-author-choice-hypotheses-20260922-01/runs/report-v3/results/result.md)
replace the proposed execution instructions. No worker remains active.

The strict/loose Equation15 and previous-fit initialization variants did not
qualify. Target-weighted log fitting completed all 16 d5/d20 probes, but every
tested d80 run failed during the first backward sweep. Its 161-coefficient
weighted regression had only 1.0–2.4 effective weighted observations; direct
rank/concavity checks identify the failures. Alternative floors did not repair
this. Sample versus population SD left all 16 completed paired runs unchanged.
No combination passed all six calibration datasets, so untouched validation
correctly did not trigger. This rejects tested candidates, not the iAPF direction.

Final review found and repaired an actual harness defect: an explanatory
Equation15 underflow check wrongly rejected a weighted-log fit. The identical
replayed case now completes; peak/N^4 passes 4/6 cells. The overall selection
remains unchanged. All focused algebra, call-chain, failure-diagnostic and
preserved-core parity checks pass. Original outputs and source versions remain
preserved; the superseded record is identified explicitly.

The campaign produced 126 completed fixed-cloud fits and 288 final full-filter
records (189 complete, 99 failed), with eight further scheduled replications
resource-censored by three timed workers. It used 2188.624/7200 aggregate worker
seconds and 69/100 launches under the unchanged deadline. Small calibration
samples do not support method ranking or full paper replication.

Next research item: use the saved independent cases to separate learned-guide
error from the diagonal Gaussian family's approximation error. Compare fitted
guides, both analytic diagonal projections and the exact full guide at matched
times and inputs before proposing another fitting change. Concentrated weights
still reach effective sample size near one; successful optimization alone does
not resolve that information loss. Preserve both validated references and use
fresh data for any later candidate evaluation. A continuation experiment needs
a new bounded time allocation because the original wall deadline has expired.
Author settings remain necessary for implementation identity. Printed Equation
15 and the weighted-log/ridge extension remain separate: this validation does
not close original-author replication or the KDM/LEDH score questions.

Previous completed amendment, 2026-09-21: the reviewed
[Equation15 resolution amendment](iapf-r-equation15-resolution-2026-09-21.md)
is complete. [Results and terminal review](artifacts/iapf-r-equation15-resolution-20260921-01/result.md)
and [mathematical findings](artifacts/iapf-r-equation15-resolution-20260921-01/mathematical-findings.md)
replace the earlier proposed next actions. No numerical worker is running.

All four local Equation15 variants failed the d5/d20 full-filter probes,
including the prescribed alternative starts and larger-iteration retries:
32 attempts, zero completed filters. The algebra, gradient and consumer tests
pass; fixed-cloud fits show that tiny absolute density loss can accompany a
badly wrong Gaussian guide. This closes the bounded diagnostic, not the fitting
replication gap or the iAPF research direction. Untouched validation was not
triggered because no candidate met its entry condition.

The saved d80 audit reproduces all prior means and verifies 1,000 parent hashes.
One replica supplies 84% of the six-estimate arm's ordinary-prefix loss. An exact
future-guide diagnostic has terminal log error below 5.46e-12 yet potentially
large prefix variance, with the density-ratio identity checked at every time.
Existing project prefix vetoes remain recorded; they do not refute the paper's
terminal-likelihood result. No statistical ranking follows from this amendment.

Timing now separates filter/learning work from extra diagnostics and I/O.
Sixteen sequential records completed. These small runs do not establish matched-
accuracy efficiency. All workers and artifact checks completed normally.
The amendment used 98.081 of its 10,000 process-second allowance in 59 launches;
previous use is retained in the manifest. The original deadline remains
2026-09-21 20:04:26 UTC. The bounded amendment ended on its scientific decision,
not a budget, review or permission stop.

The formerly proposed reconstruction hypotheses have now been executed; their
current results and next research item are stated above. The earlier box1 and QR
findings remain preserved, and the distinction between terminal likelihood and
online filtering remains binding.

Prior evidence: [completed five-dimension campaign](artifacts/iapf-r-24hour-campaign-20260921-01/result.md)
and [remaining-gap audit](artifacts/iapf-r-24hour-campaign-20260921-01/remaining-replication-gaps.md).
Its 1,000-replica-per-dimension coverage gap is closed. Full paper replication,
original numerical settings/data, matched-cost comparison, later studies and
multidimensional R/TF parity remain open. No LEDH/KDM/HMC or production default
claim follows. Active state is in the
[master checkpoint](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).

Historical completed allocation, superseded by the new campaign: the next fitting and controller actions
were already in this program and have now been executed within their bounded
allocation. The [full-filter fitting comparison](iapf-r-full-filter-fitting-comparison-2026-09-21.md)
is complete. The first dataset of the separately reviewed
[controller-window comparison](iapf-r-controller-window-comparison-2026-09-21.md)
is complete after an automatic timeout repair. No numerical worker remains.
Full evidence and decisions: [result](artifacts/iapf-r-full-filter-fitting-20260921-01/result.md).

All 32 constrained Equation15 validation replicas completed, but the frozen box1
fit fails the practical likelihood/variability/particle screen on both fresh d20
datasets; QR passes both. Observed SDs are .501/.523 versus .120/.100 for QR.
The constrained fit also loses the observed conditional heuristic screen to FA;
paired intervals span zero, so population inferiority is not established.
Saved-guide Gaussian-component KL is about2.23 versus .19 for QR, consistent
with the previously derived diffuse-density escape and distorted guide shape.
Compact bounds restore a finite fitting problem but do not fix its objective.
This rejects the tested constrained reconstruction, not every iAPF implementation.

At d80, all eight paired controller replicas and three heuristic baselines are
complete on fresh dataset89400080. Shortening the stopping window from six to
five estimates reduces final N from2000 to1000 in every pair and saves a mean
6.675 seconds (paired95% interval [-7.627,-5.879]). Its likelihood-mean interval
[.8603,1.2194] misses the prescribed upper limit1.20. The six-estimate comparator
passes likelihood accuracy conditions but exceeds N<=1713. Both clear the
observed heuristic screen. Neither controller arm is promoted; the terminal
squared-error difference remains unresolved and the evidence covers one dataset.
All tail/fit checks pass. Default controller behavior is unchanged.

Execution: 2354.902653/2400 summed worker seconds, 15/16 launches, leaving
45.097347 seconds and one launch. Attempt14 timed out after26 complete method
pairs; attempt15 finished only the14 missing pairs with identical algorithms,
data and seed recipes. All40 pairs and diagnostic records validate. Twelve
regressions, direct identities, controller/consumer wiring, repair smoke,
source/merge hashes and git diff --check pass. CPU R independent reference only.
The earlier audit's unused549.198570 seconds remain separate and unspent.

Next exact task: complete the frozen second d80 dataset89500080, IDs2601--2608,
with QR and the five-estimate extension plus BPF/FA/SIS. The current remaining
allocation cannot cover that cell. Reserve roughly700 additional worker seconds
based on the first dataset's roughly550 seconds of method work, using resumable
method/replica units; this allocation is proposed, not silently spent. Do not
retune the controller or fitting settings on the first dataset. Then reassess
reference viability before the five-dimension1000-repeat study. Candidate
failure triggers evidence/repair work; insufficient remaining compute is the
present continuation boundary.

Original-author data/settings and literal paper variability/resampling agreement
remain open, as do later studies and multidimensional full-filter R/TF parity;
the existing TF endpoint is one-dimensional. QR uses a different fitting
objective from Equation15. None of this establishes LEDH, KDM, GPU or HMC
correctness or failure. Prior source/math findings:
[audit](artifacts/iapf-r-replication-gap-audit-20260921-01/source-and-math-audit.md).
Exact current checkpoint:
[checkpoint](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).
The bounded fitting campaign is complete; the broader research program is not.
The dated entries below preserve historical decisions. Their old next-step
instructions and budget balances are superseded by this state and checkpoint.

Completed owner-directed reconstruction, 2026-09-21: the user explicitly permits
trying plausible unknown numerical choices. Execute the
[bounded reconstruction comparison](iapf-r-plausible-reconstruction-2026-09-21.md)
across current/delayed doubling, an intermediate positive floor, and a bounded
local Eq15 fit. The previous missing-author-settings dependency now limits
claims of original implementation identity; it does not block this authorized
empirical comparison. Compare the paper's actual SD, particle and resampling
patterns, then check a frozen choice on fresh data against Kalman and three
heuristics. Transfer exactly 868.02274921973 remaining summed worker seconds;
preserve earlier evidence and execute repairs/phase transitions automatically.
No original-code, 1000-replicate, GPU, HMC or algorithm-default claim follows.
The comparison has now completed ten launches, with nine successful cells and
one active-bound rejection. Delayed doubling fixes the d5/d10/d20 particle
count in this sample (1000). Fresh d10 mean ratio is 0.999454, SD 0.074891;
fresh d80 mean ratio is 0.958843, SD 0.198255, with mean N=1937.5 versus the
paper's 1142. The d80 particle criterion fails. At d20, ordinary-innovation
MSE is descriptively worse than FA-APF; its paired difference interval includes
zero, so this is a conservative promotion veto, not established inferiority.
Only d5/d10 pass every practical-reference screen. Every tested dimension
misses the literal variability/resampling agreement screen. Original-code
identity remains unknown. See the
[full result](artifacts/iapf-r-plausible-reconstruction-20260921-01/result.md).
Accounted use 795.381162 of 868.022749 seconds; 72.641587 remain. The planned
eight-repeat d40 cell plus reporting reserve does not fit that balance.
Next: complete new-setting d40 and obtain fresh d20 evidence when enough
campaign allowance is available. A shorter controller window is nominated only
as an explicitly labeled extension: saved histories implicate the earliest
estimate in fifteen of sixteen d80 stopping windows. No worker remains running.

Owner continuation directive, 2026-09-20: execute the authorized program
across phase boundaries without waiting for another message. A phase boundary
is an internal checkpoint: save the result, classify failures, execute bounded
repairs, refresh and skeptically review the successor, then launch the next
eligible work. A failed candidate, finished batch, missing advisory reviewer,
or routine infrastructure failure is not by itself a reason to end execution.
Continue independent work when only a dependent row is blocked. Pause only
when all useful authorized work is blocked, the applicable budget is exhausted,
or the next necessary action crosses a real permission, cost or scientific
scope boundary. State the precise blocker and smallest resolution when pausing.
This makes the existing between-phase repair mechanism operational; it adds
no approval ceremony and does not permit silently changing a scientific target.

Current state, 2026-09-21: the authorized automatic continuation completed
source reconciliation, the missing same-setting d5/d10/d20 checks, and the
R–TensorFlow component comparison without a phase-approval stop. All 96 new
repetitions pass their accuracy, fit, tail and conditional heuristic screens;
all eight shared numerical comparisons and four wiring/endpoint checks pass.
The largest component discrepancy is 6.22e-15. Eleven focused tests pass.
No numerical core, scientific setting, algorithmic default or production
consumer changed. Results:
[small dimensions](artifacts/iapf-r-small-dimension-completion-20260921-01/result.md)
and [component parity](artifacts/iapf-r-tf-component-parity-20260921-01/result.md).

The next original-paper replication step has a substantive dependency:
recover or explicitly define the numerical fitting procedure behind Eq.15.
The unrestricted written optimization can lack a finite minimizer, and the
author's solver/constraints/floor/early-controller choices were not recovered.
The qualified log-quadratic R method remains a different objective. The
current TF consumer additionally has a scalar restriction and a different
model; its separate log-quadratic comparator uses full covariance. A larger
same-method experiment cannot be specified honestly by ignoring those gaps.
This is a source/method-definition blocker for that experiment, not a phase
approval requirement or evidence against every iAPF method. The independent
checks that could answer the current questions without resolving it are done.
No worker remains running. The transferred allowance has 868.022749220 worker
seconds left; it is not exhausted. Current
[checkpoint](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).

Completed source investigation: see the
[result and mathematical counterexample](artifacts/iapf-r-source-reconciliation-20260920-01/result.md).
Both recovered arXiv versions leave the original optimizer/constraints/floor
unspecified. A deterministic counterexample verifies that unrestricted Eq.15
can have infimum zero without a finite minimizer. This blocks exact numerical
paper replication, not the authorized optional R reference work. The
[frozen small-dimension completion](iapf-r-small-dimension-completion-2026-09-21.md)
then executed without a new permission stop. It transferred 479.519126 seconds remaining after the
source diagnostic plus 853.973386 unused positive-floor validation seconds,
total 1333.492513; prior manifests stay unchanged and allowances cannot be
spent twice. Its historical handoff
[checkpoint](artifacts/iapf-r-source-reconciliation-20260920-01/checkpoint.md).

Active execution update, 2026-09-20: the
[d40 confirmation plan](iapf-r-d40-confirmation-2026-09-20.md) is COMPLETE.
All 64 new repetitions pass the separate accuracy, fit, tail and conditional
heuristic screens. The same-data mean likelihood ratio is 0.967538, with
bootstrap 95% interval [0.932404, 1.002977]; fresh data give 1.022861,
[0.982292, 1.061802]. The earlier upward discrepancy did not recur; neither
interval proves unbiasedness. All 38,600 QR fits and 6,400 Gaussian-limit tail
checks pass. Numerical settings and the R core are unchanged. Eleven focused
tests pass, including a mutation test for returning the stopping estimate
instead of the fresh final estimate. The source audit derives conditional
unbiasedness for the idealized checked algorithm. See the
[result](artifacts/iapf-r-d40-confirmation-20260920-01/result.md) and
[derivation](artifacts/iapf-r-d40-confirmation-20260920-01/likelihood-audit.md).
Use: 1,020.265967 / 1,500 summed worker seconds; 479.734033 unused. Both
planned launches succeeded, no retry occurred, and no worker remains running.
Current [checkpoint](artifacts/iapf-r-d40-confirmation-20260920-01/checkpoint.md).

Previous completed stage: the [frozen positive-floor validation
plan](iapf-r-positive-floor-validation-2026-09-20.md) is COMPLETE. All96
repetitions passed the separate dataset accuracy/tail/heuristic screens:
d80-A mean1.020141, bootstrap95[.975004,1.073665]; d80-B mean.981628,
[.927681,1.036752]; d40 mean1.041181,[1.005136,1.078378]. All64100 fits
and9600 Gaussian-limit tail checks pass. The d40 interval excludes1 despite
passing the predeclared10% tolerance, so exact agreement/unbiasedness is not
established. The optional log-quadratic objective and floor8 remain frozen;
numerical defaults are unchanged. See the
[terminal result](artifacts/iapf-r-positive-floor-validation-20260920-01/result.md).
Use3146.026614/4000 worker-seconds;853.973386 remain, with no extra launch
allocated and no process running. Ten focused tests and six evidence-mutation
checks pass. One pre-launch reporting-wrapper failure was repaired; no filter
run or scientific setting was retried. Previous
[checkpoint](artifacts/iapf-r-positive-floor-validation-20260920-01/checkpoint.md).

Pending source dependency: the original-paper numerical fitting procedure is
not fully recovered. The written Eq.15 amplitude escape is now checked, and
the pinned public R comparator uses a different scaled objective with
unverified author provenance. The successful optional log-quadratic fit is
also a different objective. More repetitions cannot establish Eq.15
conformance. Resolve this identity before labeling a 1,000-repeat study exact
paper replication; continue the qualified optional reference meanwhile. The
later TensorFlow comparison must identify its actual R comparator. No
score/HMC/KDM/LEDH promotion follows from this study.

Active reference-development branch, 2026-09-20: the owner requested a complete
independent R iAPF implementation and replication before using it to debug the
TensorFlow program. The selected scope is the paper's first linear-Gaussian
study, starting with bounded runs and scaling toward published settings. The
owner's careful paper/code audit and continuation have been planned,
self-reviewed and executed. Current evidence is the
[paper/code audit](artifacts/iapf-r-paper-code-audit-20260920-01/result.md),
[log-fit repair result](artifacts/iapf-r-log-fit-repair-20260920-01/result.md) and
[mathematical explanation](../reference/iapf-independent-r-reference.md).

The optional `relative_l2` fitter removes the demonstrated density-amplitude
escape. It is explicitly different from equation (15); `paper_eq15` remains the
default. Complete 32-repeat comparisons at T100,d5/d10 pass the prespecified
likelihood-accuracy screen. Mean ratios to Kalman are 1.0011 [0.9897,1.0131] and
1.0044 [0.9886,1.0204]. Conditional bootstrap intervals support lower variance
than the fully adapted filter on those datasets, at unequal computing budgets.
Ordinary/large-innovation heuristic vetoes do not fire. There is no paper-scale,
equal-cost, nonlinear, score, HMC or canonical LEDH conclusion.

The d20 relative-loss fit remains rejected. L-BFGS-B fails a fresh pilot; explicit nlminb
passes seven saved fits but fails another fresh pilot and a 10000-iteration
replay. The first failed fit has squared-target effective size 1.272/1000 and
residual-Jacobian condition ratio 48,446; its independent gradient check passes.
The audit now confirms the off-cloud failure: exact relative errors under the
smoothing measure are .472332 and .261061, worse than the initial fits, despite
training residuals near zero. Diagonal target-moment diagnostics give .018480
and .017046. This is a fitting/coverage failure in the checked cases, not a
failure of the proposal/importance-correction identities.
Author solver/floor/early-controller choices and original data remain unknown.
The full 1000-repeat study and the TensorFlow comparison are not complete.

The separately labeled `log_quadratic` fitter completed four d20 pilot repeats
and 16 repeats on another untouched dataset. The latter mean likelihood ratio
to Kalman is .989741, with bootstrap95% interval [.951537,1.029334], passing its
conditional accuracy screen. The new validation stage added two fresh d20
32-repeat datasets: means .996404 and 1.009890, with bootstrap95% intervals
[.976202,1.016881] and [.985510,1.033173]. A d40 four-repeat pilot was
descriptive (.875362); its independent 16-repeat batch gave 1.004931
[.961774,1.051352]. All completed fits pass strict QR/rank/curvature checks and
all conditional heuristic screens pass. This optional objective differs from
equation15; the default is unchanged, and no method ranking is claimed.

The d80 pilot is a scoped failure: it hit the frozen 20-iteration controller
cap at N=8000, with last-six likelihood CV1.401898 versus the .5 stopping
threshold. All 2,000 QR fits were admissible. Exact replay reproduced status,
history, counts and fits; a full-covariance exact-twist control matched Kalman
with zero recorded log error. The failure was subsequently localized to the
chosen positive floor at observation93. Its mixture probability exceeded.99999
and all4000/8000 particles used the original transition in two bad passes.
Removing that floor only reduced log errors from-37.70/-36.05 to-1.21/-1.89.
This identifies a local floor calibration failure; it is not evidence that
the original author's unspecified floor behaves similarly.

The separately reviewed positive-floor repair keeps the same controller limits
and changes the explicit optional tail power from2 to8, selected from analytic
mixture probabilities. Healthy outputs are exactly equal;1700 Gaussian-limit
tail checks pass. Eight d20 non-harm repetitions, two d80 pilot repetitions and
four repetitions on a fresh d80 dataset complete, with9000 valid fits and no
conditional heuristic veto. Every d80 run takes eight adaptive passes and
N2000; the fresh mean ratio is1.088298, descriptive bootstrap95%[1.008317,1.162667].
This is feasibility, not established accuracy from four repetitions. Ordinary
d20 prefix MSE rises descriptively25.6-36.3%, within the declared factor2 screen.
The default objective/floor remain unchanged. See the
[floor diagnosis and repair](artifacts/iapf-r-log-fit-validation-20260920-01/floor-result.md).

The audit also repaired strict-fit nonconvergence/underflow acceptance,
incomplete saved failure context and execution of live rather than captured
sources. Final checks: 88 reference plus 33 alternative R checks, three source
mutations and snapshot regressions pass (seven pytest cases, 1.78s). Exact-twist
controls through d80 retain their limited oracle role. Repair use is
1516.719870/1550 worker-seconds across all12 launches; combined worker use is
1765.102396/1800 seconds. Mechanics are charged55/120 seconds. That bounded
campaign is complete; its remaining33.280130 seconds have no launch slots.

The owner-authorized
[frozen log-fit validation plan](iapf-r-log-fit-validation-2026-09-20.md) has
an1800 CPU-worker-second/eight-planned-launch budget. The
[validation result](artifacts/iapf-r-log-fit-validation-20260920-01/result.md)
records the completed d20/d40 runs and original d80 failure. The owner's
continuation executed the [floor diagnosis](iapf-r-d80-floor-diagnosis-2026-09-20.md)
and [positive-floor repair](iapf-r-d80-positive-floor-repair-2026-09-20.md).
Campaign use is1474.773800 seconds, including one localized logging/status
infrastructure retry (nine process launches). The repair and retry share the
original600-second allocation; no numerical settings changed on retry.
325.226200 seconds remain in that closed allocation; no repair job remains.
The owner's next execution request completed the separate
[positive-floor validation](iapf-r-positive-floor-validation-2026-09-20.md),
with frozen32-repeat validation on each of two untouched d80 datasets and
32 repeats on a fresh d40 control. All pass the declared Kalman accuracy,
conditional heuristic and numerical screens. These results test accuracy and
dataset dependence of power8 before scaling toward1000 repetitions; they do
not establish equality to the original fitting procedure.
Author objective/solver/floor/data ambiguities remain separate
paper-replication gaps. TensorFlow comparison
follows the qualified R reference; no nonlinear, score, HMC, KDM or canonical
LEDH promotion follows from this result. This reference exception does not
change the TensorFlow/GPU production direction.

Reference status, 2026-09-20: the requested
[public-iAPF paper-conformance audit](artifacts/iapf-paper-conformance-20260919-01/result.md)
is COMPLETE. Thirteen checked Gaussian/procedure identities pass; the public R
code differs in the fitting objective, positive floor, earliest stopping index,
and T=1 handling. It remains an unverified-original-author comparator for
`matched_gaussian_operations_only`, with `paper_reference_eligible=false`.
The actual comparison consumer now enforces that scope; 49 focused CPU tests
pass, including source mutations and consumer wiring. This is reference
classification evidence, not a new score-performance run or an explanation of
case 1900. The next scientific question below is unchanged.

Current result and next question, 2026-09-19: the owner directed continued
execution without unnecessary stops. All four stages of the
[fitting/precision continuation](younis-iapf-fitting-protocol-2026-09-19.md) are
COMPLETE; the [terminal result](artifacts/younis-iapf-fitting-protocol-20260919-01/result.md)
and [independent audit](artifacts/younis-iapf-fitting-protocol-20260919-01/analysis.json)
are the current evidence. Execution continued after fitting-only validation
failed, first to a fresh-data particle-precision comparison and then to a
replay of the known failures.

Fitting-only validation produced 0/4 statistically supported gains. Increasing
final N from 4096 to 16384 with each proposal frozen produced 4/4 primary
interval passes on new observations (observed MSE reductions 72--79%), and
3/3 on the failure replay. Two of the three known cases now clear the observed
heuristic screen. **1900 still fails the observed UKF screen**, retaining a poor first-step
twist and an earlier fitting-bound contact. This isolates substantial Monte
Carlo error without proving the remaining error is solely a proposal defect.
The selected large-cloud fitting protocol has not been refitted on 1900.
No equal-cost, default, HMC or LEDH promotion follows.

The [mathematical explanation](artifacts/younis-iapf-fitting-protocol-20260919-01/mathematical-explanation.md)
now separates proved defects from causal hypotheses. The active estimator is
iAPF plus a Fisher-identity score and frozen linear controls. Likelihood-optimal
twisting need not minimize score variance; the old CV threshold is provably
ineffective. For 1900, the observed MSE excess over UKF is .0011519, while the
descriptive standard error of that MSE is .0015491. A true risk ordering and a
causal explanation of that excess are not established. Saved-data arithmetic
and executed-source checks used no new filter calls, fits or GPU launches.

Next scientific question: separate inadequate cloud coverage, optimizer failure,
and Gaussian-family limitations in 1900's first backward fit. Preserve the exact
failing fit; inspect predictive coverage and fit geometry before choosing a
repair. Any accuracy confirmation must use fresh observations. A reporting-only
repair now preserves fitting-cloud SD through the shared recursive fitter and
actual adapter; 33 focused CPU fitting/adapter/scope tests pass. Historical
scientific artifacts retain their old schema and executed source snapshots.

This allocation used 4/4 launches, 28/32 adaptive fits, 7988/8000 charged calls
and 255.037560/1800 driver seconds; CPU checks are conservatively charged
160/600 seconds. No experiment is running, and the remaining twelve calls
cannot cover another research stage. This paragraph supersedes older
next-action/budget paragraphs. The research direction remains open; the next
allocation must address the specific fit question above rather than repeat
completed screens.

Historical terminal continuation, 2026-09-19: the owner-approved
[pinned/fresh-data/bound-sensitivity plan](younis-iapf-pinned-continuation-2026-09-19.md)
is COMPLETE. [Results](artifacts/younis-iapf-pinned-continuation-20260919-01/result.md)
and the [mathematical fitting diagnosis](artifacts/younis-iapf-pinned-continuation-20260919-01/fitting-diagnosis.md)
remain archived evidence. No process from that allocation is running.

The intended RTX5080 confirmation passes all four primary nonlinear comparisons
with fresh calibration/final streams, observed MSE reductions of 42--76%, and
all four nonlinear heuristic screens. Physical UUID, memory growth and actual
GPU:0 score/control outputs are verified. This closes the device-selection gap.

Fresh observations change the scientific result: datasets 1900/1901 still lose
to simple filters; only 3/4 primary new-versus-ancestor comparisons pass. Both
curved datasets clear the observed heuristic screens. The original-box fit for
1900 contacts a bound. The predeclared paired wider-box stage also passes only
3/4 primary comparisons, retains that contact, and leaves both weak cases
losing to UKF. The three interior proposal fits are exactly unchanged, so box
expansion passes interior non-harm but fails as a complete fitting repair.

A checked mathematical defect in the inherited diagnostic configuration is
now explicit: k=1 means a two-likelihood window with CV <= sqrt(2), so tau=100
always passes at the first eligible stop. This cannot establish fitting
convergence. Dataset1900 keeps only 16 fitting particles and has first-step
predictive shape residual 0.996807 (original box) / 0.888282 (wider box).
Small-cloud coverage, optimizer behavior and Gaussian-family limitations remain
separate hypotheses. An interior-fit failure on 1901 also shows that removing
bound contacts alone cannot clear the weak-case score-accuracy gap.

Budget used: 3/4 launches, 3068/8000 charged filter calls, 8/8 adaptive fits,
137.859234/1800 driver seconds and conservative 60/600 CPU test/probe seconds.
The planned stages are complete and the fitting allowance is exhausted. Remaining
calls do not authorize more fitting. Eight focused CPU tests and six analytic
checks against the actual stopping callable pass; independent arithmetic,
frozen-calibration, source snapshots, reference and GPU audits pass.

Exact next task: design and skeptically review a bounded, target-specific
larger-fitting-cloud and informative-stopping protocol using new calibration
observations, followed by untouched validation. Retain matched ancestor controls,
raw Fisher and UKF; preserve current failed finals without tuning on them.
Do this before longer horizons/dimensions or iAPF-moment/LEDH integration.
No default/HMC/LEDH promotion or population ranking; general backward smoothing
remains deferred. The whole master remains incomplete. This refresh supersedes
all older active/next-action/budget paragraphs below.

Current terminal refresh, 2026-09-19: the
[Gaussian-innovation control plan](younis-iapf-innovation-control-2026-09-18.md)
has been implemented and its bounded GPU comparison is COMPLETE. The
[result and terminal audit](artifacts/younis-iapf-resampling-control-20260918-01/innovation-result.md)
record 4/4 nonlinear primary MSE comparisons passing, observed reductions of
49--73% against a matched ancestor-only correction, and 4/4 nonlinear mean-bias
screens passing. All four nonlinear observed heuristic screens now pass
against EKF, UKF and no resampling. The prior frozen 192-calibration correction
is also retained as a comparator; its exploratory intervals favor the new
controls. Exact Kalman still dominates the affine case. No default/HMC/LEDH
promotion; this remains five fixed scalar datasets at T2 and N4096.

Execution defect: this launch omitted the previous RTX5080 UUID pin, exposing
both GPUs with RTX4080 SUPER as TensorFlow GPU:0. Actual per-output ordinals
were not saved. Matched comparisons remain evidence for the executed scope,
but intended RTX5080 replication is outstanding. The executed source/plan
are preserved; the driver now pins the reference UUID before TensorFlow
import, checks device identity and records tensor devices. Six focused CPU
repair checks pass after 22 pre-launch checks (23 distinct tests overall).
The repaired GPU launch has not run. Arithmetic, frozen coefficients,
references, single-trace checks and executed source hashes pass.

This campaign is CLOSED at 4/4 launches, 8000/8000 filter calls,
166.131425/1800 driver seconds, 5/8 adaptive fits and conservative 240/600
test/probe seconds. No run is active. Remaining wall time does not permit
another call or launch under this campaign.

Exact next task: write and skeptically review a NEW bounded campaign for a
fresh-stream confirmation on the pinned RTX5080, retaining both ancestor
baselines and the same target. Then repair/audit scope-specific iAPF fitting
bounds and validate on fresh observation datasets and independent calibration
before extending horizons/dimensions or integrating iAPF moments into LEDH.
Do not tune against any completed final streams. Gaussian controls preserve
finite-N bias and are not finite-program likelihood gradients. General backward
smoothing stays deferred; the whole master remains incomplete. This terminal
refresh supersedes every older active/next-action paragraph below.

Current execution refresh, 2026-09-18: the
[resampling-control plan](younis-iapf-resampling-control-2026-09-18.md) and its
fresh-stream conditioning follow-up are COMPLETE. See the
[comparison result](artifacts/younis-iapf-resampling-control-20260918-01/result.md)
and [safety result](artifacts/younis-iapf-resampling-control-20260918-01/conditioning-result.md).
The corrected Fisher score still loses to UKF on weak dataset 1500; curved
dataset 1511 is statistically indistinguishable from no resampling. No
default/HMC/LEDH promotion. Relative to raw Fisher at N4096, all four nonlinear
primary comparisons pass, with observed MSE reductions of 29--47% and 4/4
mean-bias screens passing. The control is exactly centered under the declared
ancestor sampling law and preserves finite-N bias; it is not a finite-program
likelihood gradient.

Affine control regression exposed an incorrect numerical-rank assumption:
FP64 fitting retained an FP32 roundoff direction although the exact rank is
at most seven. The explicit input-precision safeguard passes fresh validation:
affine rank repaired and Kalman bias screen passed; all four nonlinear final
corrections exactly unchanged. Seventeen focused CPU tests pass; all kernels
trace once, references and terminal source hashes pass. Existing defaults are
unchanged. Total use: 3/4 launches, 129.311595/1800 driver seconds, 5/8 fits,
6884/8000 filter charges, conservative 180/600 test/probe seconds. No run is
active; all planned comparisons and safety checks are complete.

Exact next task: derive Gaussian-innovation zero-mean controls or conditional
integration to address variance left after ancestor controls. Subtract any
control outside the normalized statistic, freeze coefficients on independent
calibration, explicitly declare input precision, and compare conditionally
against UKF/no resampling on fresh streams. First write and skeptically review
that evidence contract; 1116 filter charges and one launch remain in the
current budget, so a larger ladder needs an explicit new bounded budget.
Do not reuse final streams from either completed stage. General backward
smoothing, fitting-bound repair, iAPF-moment/LEDH integration and wider model
coverage remain pending. The whole master is incomplete. This refresh
supersedes older next-action instructions below.

Active execution refresh, 2026-09-18: the
[iAPF Fisher-score plan](younis-iapf-fisher-score-2026-09-18.md) is reviewed,
implemented and its bounded GPU comparison is complete. See the
[result](artifacts/younis-iapf-fisher-score-20260918-01/result.md).
The new analytical terminal-genealogy score passes the N4096 mean-bias screen
on all four fresh nonlinear datasets; the fixed-label derivative fails on all
four. However, Fisher has larger observed sampling variance, higher MSE than
the fixed-label score, and loses to UKF on both weak datasets. No candidate is
promoted. This is a physical-score diagnostic; it is not the gradient of the
reported finite likelihood and cannot silently replace an HMC force.

The optional score is returned by the same iAPF kernel invocation. Separate
FP32 compilations can change a categorical choice; that precision limitation
is documented and the failed attempt preserved. FP64 parity and independent
genealogy checks pass. KDM mixture/SGQF quadrature validity propagation is
repaired; all 29 distinct focused CPU checks pass. Campaign use: 2/4 attempts,
128.756/1800 driver seconds, 5/8 fits, 3000/3000 filter-call charges. This stage
is complete; its call budget is exhausted.

Exact next task: derive target-preserving conditional integration or an
exactly centered control variate to reduce score variance, beginning with
the tractable short-horizon identity reference. Check the expectation of the
self-normalized ratio, not merely separate expectations of numerator and
denominator. Review that derivation before a fresh bounded calibration and
confirmation campaign; do not fit controls on the diagnostic final streams.
The bounded Fisher identity comparison is an explicit exception to the
deferred inventory below, not activation of the general backward-smoothing
program. Fitting-bound repair remains open, followed by eligible
iAPF-moment/LEDH integration and wider model coverage. The whole master is
incomplete. This paragraph supersedes older next-action instructions.

Latest execution refresh: 2026-09-18. The requested
[curved iAPF diagnosis](younis-iapf-curved-diagnosis-2026-09-18.md) is reviewed
and complete; see its
[result](artifacts/younis-iapf-curved-diagnosis-20260918-01/result.md).
Same-cloud calibration fits agree under 2000/5000/10000 solver caps, while
fresh datasets still require up to 3823 steps. Floor .001 was nominated by
calibration shape only, but confirmation has offline bound activity. At
N=4096 its curved calibration score MSE is 0.009514 versus 0.005106 without
resampling; the paired conditional 99% difference interval is positive.
Finite-program derivative parity is about 1e-10, yet a mean score component
is 21 Monte Carlo standard errors from the model oracle. This separates
correct local calculus from model-score accuracy. No setting is promoted.
The UKF baseline's stale quadrature return-value handling was repaired;
12 focused checks pass. Campaign use: 3/4 attempts, 252.093/2400 driver
seconds, 37/40 adaptive fits and 3141/3500 filter-call charges.
The next task at that stage was to derive and test an analytical model-score estimator accounting for
categorical sampling, keeping the current fixed-label finite-program
derivative as a distinct target. Use the existing Fisher/complete-data
direction as a checked comparator; repair fitting bounds on fresh data
before any promotion. iAPF-moment/LEDH integration and the broader master
remain incomplete. The newer Fisher result above now governs the next action.

Execution refresh: 2026-09-18. The owner-requested
[iAPF/KDM public-reference comparison](younis-iapf-kdm-reference-comparison-2026-09-18.md)
was planned, self-reviewed and executed. Its
[result](artifacts/younis-iapf-kdm-reference-comparison-20260918-01/result.md)
records agreement for all tested shared Gaussian algebra, KDM analytical
derivatives and IWSG resampling after accounting for the author code's epsilon
terms. The exact iAPF filter example agrees to floating-point precision. One
public R default fit misses the known variance by 0.00402 despite reporting
convergence; changing only its stopping control reduces all variance errors
below 7.4e-7. The original failed comparison is retained. Across two attempts,
143/144 distinct checks pass; 19 focused consumer/score regressions pass.
The CPU reference campaign used 2/4 attempts and a conservative 100/1800
seconds; no runtime default changed. The R source uses an objective monotone
in our relative-shape objective, omits the paper's positive floor, and has an
earlier stopping boundary. It is an independent public reference, not verified
original-author code. These comparisons do not close the curved-score failure.
Next remains fresh Phase 0E fitting-control and family/bound/floor diagnosis;
do not replace checked local formulas merely to copy reference defaults.

Execution refresh: 2026-09-17. The latest
[relative-shape repair and solver diagnostic](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-relative-shape-repair-20260917-01/result-and-refresh.md)
executed from clean `e6298503`: 29 accepted fresh numerical rows, one
unconverged validation fit, two blocked weak-regime claims and no underflow
failures. The final diagnostic resolved that validation failure at 4,201
optimizer steps without relaxing the 1e-7 threshold. Curved repaired claims
still fail the boundary and EKF/UKF/local-linear screens. No candidate is
promoted and no statistical ranking is supported. The main implementation is
updated; 32 isolated checks, three oracle hooks, and 13 main-checkout checks
pass. All three permitted GPU launches are used; total consumption is 249/280
charges, 252.98/3000 GPU wall seconds and 1074.95/7200 CPU seconds.

The preceding
[fresh fit-calibration phase](younis-score-iapf-fit-calibration-fresh-2026-09-17.md)
has executed from clean commit `418e5388`: 54 completed numerical rows,
14 rejected underflow fits, and 12 selected-candidate claim rows blocked by
incomplete source studies. Both conditional heuristic tables and all twelve
frozen-control baseline claim rows are complete. The baseline has larger
observed score error than EKF and UKF on every claim dataset; no candidate is
promoted and no statistical ranking is supported. See the
[result and diagnosis](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-calibration-fresh-20260917-01/result-and-refresh.md).

Phase 0E has implemented and executed the
[relative-shape fitting repair](younis-score-iapf-relative-shape-repair-2026-09-17.md).
Its objective divides profiled residual squared norm by density squared norm;
it is an explicit Algorithm-3 adaptation, different from the published Eq. (15)
retained as comparator. The source anchors, analytical quotient derivative,
failure conditions and fresh-data experiment were written before execution.
Next is a new bounded fitting-control calibration with fresh partitions:
justify the solver budget or improve optimizer geometry, and address the
remaining curved-regime bound/floor, Gaussian-family and particle-budget
questions. The old campaign's three-launch allowance is exhausted; unused
row charges do not authorize a fourth launch.
Fitted-moment integration into LEDH, wider model coverage and
powered replication remain open. The whole master is incomplete. Current
execution details and the exact restart action are in the
[active checkpoint](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).

## Purpose

This program investigates how to estimate the observed-data model score

\[
 S(\theta;y_{1:T})=\nabla_\theta\log p_\theta(y_{1:T})
\]

when a particle filter uses LEDH, KDM, GenUT, UKF, continuous resampling, or
optimal transport. The program is deliberately organised around the target
quantity. It does not treat a differentiable finite filter, a KDM expectation,
an unnormalised likelihood derivative, and the marginal score as interchangeable
objects.

The central research question is:

> Which proposal, score identity, resampling representation, and variance-
> reduction method gives the smallest scientifically relevant error for the
> model score at a declared compute budget, while remaining valid for the
> transition-support class of the model?

The applicability record distinguishes two non-interchangeable tracks:

1. **Regular-transition track.** Transitions admit densities on a common
   reference measure. This is the setting in which model-corrected mixture
   proposals, Fisher identities, backward pair estimates, Rao--Blackwellization,
   and Younis-style KDM calculus can be tested.
2. **Degenerate-transition track.** Transitions are deterministic or singular
   conditional on an ancestor, as in the DSGE target. Ordinary ambient-space
   Gaussian-mixture density ratios and regular-transition smoothing arguments
   are not assumed valid. A support-aware score identity must be derived before
   implementation or comparison.

No result from the regular track may be presented as evidence that the
degenerate track is solved.

**Active execution scope, amended 2026-09-14.** This program now executes
KDM/IWSG filtering and estimator combinations, proposal/covariance alternatives
(UKF, KDM, SGQF), twisting/iAPF, and coupled directional finite differences.
Regular models provide tractable reference tests for these mechanisms.
Regular-transition smoothing and degenerate/DSGE score derivations are assigned
to separate programs. Phases 5 and 6 and the corresponding literature rows
remain reference material; they are not generated as executable rows here and
are not prerequisites for the active KDM filtering study. Rhee--Glynn/JLS,
Nemeth, PaRIS, and other smoothing estimators are deferred with that work.
An imported result needs target, support, and provenance checks before use.

The changes below incorporate the mathematical corrections agreed in Claude's
follow-up. Claude's bounded Phase 4C review gave a REVISE verdict and marked
other sections unchecked; it did not approve those sections. The dependency,
tuning, and SGQF amendments are additional findings from this program review.

This master includes specification repair, implementation, testing, tuning,
experimental execution, analysis, and between-phase repair. Phases 0A--0G
below build the prerequisites for the scientific comparisons. Running the
master begins with those development tasks; it does not require a separately
executed implementation program. The former E0--E7 execution plan is
superseded by the phases and dependencies in this document.

The initial phases are performed by the supervising developer or agent with
the repository's existing development tools. They build the coordinator that
later runs numerical rows. They cannot require that coordinator to exist
before starting. All phases, including implementation and terminal review,
use the repair and next-phase refresh procedure in this master.

### Program milestones and starting status

| Milestone | Work inside this master | Completion evidence |
|---|---|---|
| Begin master execution | Read Phase 0's scientific requirements and start Phase 0A, followed by 0B and 0C. | Source/specification reconciliation and implementation work have begun; no pre-existing master CLI is required. |
| Baseline study runs | Complete 0A--0C and applicable Phase 1 admission checks. | Real Gaussian baseline, oracle, scoped tuning, report, and repair/resume tests pass. An oracle-only or fake-endpoint smoke is insufficient. |
| Active method matrix runs | Complete the applicable 0D--0G implementations and Phase 1 checks for every active family. | Actual method endpoints, controls, and reports execute under the required backend; missing integrations remain recorded as incomplete work. |
| Research program completed | Execute applicable Phases 2--4C and 7--9, repairs, uncertainty analysis, and final dispositions. | Evidence supports the reported positive, negative, or unresolved result for every planned row. Software availability alone is not scientific completion. |

Execution began on 2026-09-14. The coordinator, Gaussian providers, canonical
adapter, scoped selection/consumption, and diagnostic reporting now exist in
`bayesfilter/score_study/`, with a real CLI and versioned study specifications.
Seven baseline rows have run on CPU/XLA and GPU/XLA; full six-parameter
derivatives include the initial distribution. These are mechanics fixtures,
not evidence of score improvement. The active execution checkpoint is
[checkpoint.md](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).
KDM integration, persistent covariance alternatives, fixed and locally fitted
twists, finite differences, normalization reports, and nonlinear comparison
endpoints now execute. Bounded density-objective iAPF fitting and adaptive
iteration and adaptive-N selection/reporting now pass scalar Gaussian CPU/GPU
mechanics checks. The scalar nonlinear iAPF extension now passes twenty GPU
rows and analytical derivative/consumer checks. Comprehensive control calibration
and the broader research matrix remain work inside this master.
A missing method cannot be removed to declare the full matrix implemented.
The earlier [recursive fit-control calibration](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-calibration-result-and-refresh.md)
closed partial at 64/64 charges; its invalid rows remain historical failure
evidence. The [underflow-guard repair](younis-score-iapf-underflow-guard-2026-09-16.md)
rejects zero density objective with positive shape residual; the isolated and
main fitter suites each pass 19 tests. This admission repair is complete.

The [fresh calibration result](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-calibration-fresh-20260917-01/result-and-refresh.md)
now supplies both weak and curved heuristic tables and the frozen-control
baseline on all six untouched claim datasets with two independent final streams.
It used one GPU launch and 171/280 charges. Fourteen wider-candidate source
rows fail the guard; their solver-converged flags do not make them valid.
Both selected-candidate studies remain incomplete and cannot issue tuning
artifacts. Baseline-only selection succeeds, but baseline score errors exceed
EKF and UKF on every claim dataset, and fitting boundaries are active on ten
of twelve baseline claim rows. The source/result audit passes. No candidate
is promoted and no ranking is statistically supported. This triggered the
executed relative-shape repair linked above, not rejection of twisting or a
harness failure. That repair makes objective identity explicit, has no observed
underflow in the fresh run, and shows that the former 2,000-step solver cap
was insufficient on a saved validation cloud. Fresh control calibration and
the curved score-error problem remain open before iAPF-moment integration
into LEDH. The current repair/driver checks pass in both source checkouts.
The earlier implementation slice is [nonlinear iAPF](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-iapf-result-and-refresh.md): twenty GPU rows, thirty frozen-source checks and the integrated consumers pass. Exact conditional twisting and frozen-fit derivatives are derived in manuscript Section 7.3. The observed score errors exceed EKF/UKF errors in both tested regimes; no ranking or promotion is supported. Its fitting controls now require the repair identified above. The preceding [adaptive-N selection/reporting](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-result-and-refresh.md) includes an actual 16-to-32 count change. The preceding [control diagnostics and safety screen](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/control-safety-result-and-refresh.md) completed:
108 comparison rows and eight GPU smoke rows complete after a validator-only
coordinate-cap repair. The inherited cap substantially compresses intermediate
coordinates; larger caps reduce that compression but do not rescue observed
score errors. All tested controls fail the descriptive heuristic screen in
weak and concentrated regimes. These two-dataset results select no default.
The diagnostics and repaired validator are integrated and seven shared-checkout
consumer tests pass. Complete control calibration remains open.
The preceding [bounded iAPF implementation and repair](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-result-and-refresh.md) records:
eight GPU rows and 12 directional checks pass after an explicit offline
FP64 fitting repair; final filtering and analytical scores remain FP32/TF32/XLA.
It establishes mechanics only. The preceding [1,440-row nonlinear covariance pilot](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-calibration-result-and-refresh.md)
passes numerical/reference and provenance checks, but all three covariance
candidates fail the observed conditional heuristic promotion screen. The
whole program remains incomplete; these are candidate-level results.

Concurrent edits to the shared canonical executor interrupted the first
numerical launch during provenance inspection. Subsequent runs use the isolated
checkout `/tmp/bayesfilter-younis-score-execution-20260914`, based on
`e7f2a88ecff49ced481b9615c0b1237b8cabe732` plus recorded repairs. Source closures
include package initializers; numerical evidence is tied to that checkout.
An XLA-elided assertion also required an explicit invalid-value propagation
guard at the final GenUT reset. Healthy and injected-invalid regressions pass.

## Project implementation constraints

The master program inherits the repository's active implementation contracts.
The claim-bearing LEDH baseline uses Contract E--Chol and its declared
analytical total-derivative composition. Autodiff, finite-program JVPs, and
GradientTape may be used for parity and debugging, but they are not a
claim-bearing score route. Active transport rows use the repository's exact
divisor chunk policy, and TensorFlow/TFP with the configured GPU, TF32, XLA,
and memory-growth settings is the default execution target. NumPy is confined
to independent references, tests, and post-run diagnostics. A candidate that
violates one of these contracts is either repaired or labelled diagnostic; it
does not enter the score leaderboard by changing a CLI flag.

These engineering constraints do not decide the scientific question. A route
can satisfy the implementation contract and still estimate the wrong target or
lose to a simple heuristic. Conversely, a mathematically interesting reference
route can remain diagnostic until its backend and derivative contracts are
implemented.

## Program architecture

The implementation should be a typed experiment system with one coordinator
and separate scientific components. This keeps a KDM expectation gradient from
being combined with a marginal score without detecting the target change.

### Registries

The coordinator reads five repository-owned registries:

1. **Model registry:** model equations, data generator, support class,
   parameterisation, initial law, transition representation, observation law,
   and oracle provider.
2. **Target registry:** exact marginal score, finite-program derivative, KDM
   expectation gradient, unnormalised derivative, or another explicitly named
   target. Each target declares its measure, normalisation, and valid support
   class.
3. **Proposal registry:** bootstrap, adapted, LEDH, KDM, forward/backward, OT,
   twisted PF, iAPF, SGQF-guided, exploration mixture, or coordinate proposal.
   Each proposal exposes both its sampler and its evaluable density or mass
   correction.
4. **Estimator registry:** Fisher, backward-pair, IWSG, pathwise, hybrid,
   Rao--Blackwell, ratio, or control-variate estimator. Each estimator declares
   the quantities it requires and the target it returns.
5. **Tuning registry:** scope identity, calibration partitions, candidate
   controls, selected controls, and the untouched claim partition.

Registry validation must reject an experiment before tracing or sampling when a
proposal, estimator, target, and support class are incompatible. In particular,
an ambient Gaussian KDM proposal cannot be registered for a degenerate model
under a regular-transition score identity.

Every registered row also has an execution status: active, deferred, or blocked
on a named prerequisite. The matrix generator expands active eligible rows
only. Listing a method in the registries, factorial axes, or literature table
does not authorize its execution.

### Matrix generator

The matrix generator expands a declarative study into rows over model, regime,
horizon, particle count, proposal, score estimator, variance-reduction method,
dtype/backend, and seed. It first generates the baseline ladder, then all
single-factor and pairwise interactions, and finally the full factorial of
surviving candidates. This gives exhaustive coverage without allowing an
unvalidated option to acquire promotion status merely because it appeared in a
large pooled sweep.

Each row is immutable after tuning. The claim runner consumes a repository-issued
tuning artifact and refuses a missing, stale, cross-model, or cross-horizon
artifact. Calibration and validation runs never write into the claim directory.

### Execution lanes

Use distinct lanes for:

- oracle construction;
- fixed-cloud estimator comparison;
- full-filter proposal comparison;
- long-horizon and particle-count scaling;
- externally supplied support/oracle evidence, imported read-only; and
- report assembly.

The fixed-cloud lane isolates conditional score-estimator variance. Repeat
over independent clouds to measure variation in the cloud itself; conditioning
on one cloud does not establish marginal-score accuracy. The full-filter lane
measures the combined effect of proposal, resampling, and score recursion. Their
outputs must not be merged into one score table without a lane label.

### Coordinator pseudocode

This is the numerical scheduler to implement in Phase 0B. It operates after
the initial development phases have created its services. Implementation and
mathematical repair remain supervised master tasks; their tests and resulting
capability status feed this scheduler.

```text
load model_registry, target_registry, proposal_registry,
     estimator_registry, tuning_registry
validate_registries_and_supports()

candidate_status = initialize_active_candidates_and_predeclared_repairs()
phase_state = load_current_plans_dependencies_and_budget()
while task := next_ready_task(active_tasks, candidate_status, phase_state):
    require_current_phase_plan_and_applicable_boundary_dispositions(task)
    if task.requires_development:
        complete_supervised_implementation_and_required_tests(task)
    matrix = generate_eligible_rows(task, candidate_status, declared_scopes)
    record_blocked_and_excluded_rows_with_reasons(task)
    for row in matrix:
        validate_target_measure_support(row)
        verify_named_prerequisites(row)
        if row.is_quality_claim:
            tuning = prepare_and_freeze_scope_on_allowed_partitions(row)
            validate_tuning_scope(tuning, row)
            require_repository_issued_tuning_artifact(tuning)
        for dataset in row.declared_partition:
            oracle = model_registry[row.model].oracle_provider.get(dataset, row.parameter)
            for seed in row.declared_seeds:
                result = execute_one_row_recording_success_or_failure(row, dataset, seed, oracle)
                write_result_to_new_attempt_directory(result)
    decision = aggregate_under_declared_evidence_role(task.results)
    apply_vetoes_and_heuristic_gate(decision)
    issues = classify_validity_defects_candidate_failures_and_uncertainty(decision)
    repairs = select_in_scope_repairs(issues, phase_state.remaining_budget)
    complete_available_repairs_and_impacted_regressions(repairs)
    invalidate_affected_evidence_and_recompute_decision(task)
    record_unresolved_issues_and_block_only_dependent_rows()
    update_candidate_status_from_verified_evidence()
    refresh_next_phase_plans_from_actual_results_or_terminal_disposition()
    audit_refreshed_dependencies_scopes_partitions_tests_commands_and_budget()
    write_phase_closeout_and_checkpoint()
    phase_state = reload_current_plans_dependencies_and_budget()
write_complete_program_row_accounting_including_unresolved_work()
```

The coordinator may schedule independent rows in parallel, but each row must
retain the exact command, environment, device, random seed, tuning scope, and
source revision needed to reproduce it. Parallel scheduling must not change
random-number coupling within a paired comparison.

Repair completion is a workflow obligation, not a promise that the coordinator
can invent or fix numerical methods automatically. The supervising developer
or agent performs source/mathematical repairs; the coordinator records source
changes, executes declared checks/retries, and refuses reuse of affected stale
evidence. A phase with remaining required repairs stays partial or blocked.

### Result schema

Every row writes machine-readable JSON plus a human-readable Markdown note with
at least:

```text
program_id, phase_id, phase_plan_version, row_id, attempt_id, git_commit
model_id, dataset_id, regime_id, parameter_id, horizon, particle_count
support_class, measure_id, target_id, score_kind, normalization_kind
proposal_id, resampling_id, estimator_id, variance_reduction_id
reset_id, derivative_id, tuning_scope_id, tuning_artifact_hash
dtype, backend, jit_compile, device, memory_policy, chunk_policy
seed, paired_seed_group, wall_time, peak_memory
oracle_id, oracle_status, score_error, value_error, diagnostics
hard_veto_status, heuristic_verdict, inference_status, artifact_paths
execution_status, blocked_prerequisites, repair_status, invalidated_evidence
```

The aggregation layer computes paired confidence intervals, bias/variance/MSE
decompositions, and the decision table. Diagnostic plots retain failed rows
with their status; rankings require passed vetoes, target compatibility, and
uncertainty evidence.

### Factorial axes

The active study matrix crosses the following axes, subject to compatibility
and prerequisite checks. The broader option table below retains deferred
literature coverage separately.

| Axis | Levels to enumerate |
|---|---|
| Target | exact marginal score; finite-program derivative; KDM expectation gradient; unnormalised value/derivative; declared approximation |
| Proposal | bootstrap transition; EKF; UKF; LEDH; physical transition mixture; KDM; fixed lookahead mixture; twisted PF; iAPF; SGQF-guided; defensive mixture |
| Resampling/coupling | multinomial/systematic discrete resampling with declared derivative target; continuous KDM mixture draw; OT with a derived correction or changed-target label; OT cloud plus explicit jitter; OT as coupling between replicas |
| Covariance | model (Q); UKF covariance; KDM weighted covariance; LEDH local covariance; GenUT-restored covariance; SGQF projected covariance; calibrated hybrid choices |
| Score identity | analytical finite scalar derivative; IWSG expectation gradient; analytical/IWSG hybrid; ratio of unnormalised estimates; finite-difference diagnostic |
| Variance reduction | none; conditional Rao--Blackwell; exact-mean control variate; independently calibrated estimator combination; declared coupling of finite differences |
| Support | regular common-density; bounded/constraint regular with checked support and differentiation conditions |
| Evaluation lane | fixed-cloud estimator with independent-cloud replication; full filtering recursion; directional diagnostics; long-horizon scaling |

The generator records why each combination is omitted. “Not applicable” and
“not yet derived” are distinct statuses; silently dropping an incompatible row
would make the final option inventory incomplete.

## Non-negotiable target definitions

Every implementation and result must label which of the following it computes.

### Exact model score

\[
 Z(\theta)=p_\theta(y_{1:T}),\qquad
 S(\theta)=\nabla_\theta\log Z(\theta).
\]

For a regular state-space model, Fisher's identity is

\[
 S(\theta)=\mathbb E_\theta\left[
 \nabla_\theta\log \mu_\theta(X_0)+
 \sum_{t=1}^T\{
 \nabla_\theta\log f_\theta(X_t\mid X_{t-1})+
 \nabla_\theta\log g_\theta(y_t\mid X_t)\}
 \mid y_{1:T}\right].
\]

The identity requires the stated support and differentiation conditions. For a
singular transition, its density notation is not used until a valid
ancestor/innovation base measure has been supplied.

### Finite-program derivative

\[
 L_N(\theta;\xi)=\text{the scalar produced by one declared finite program},
 \qquad
 S_N^{\mathrm{fin}}=\nabla_\theta L_N.
\]

This derivative is a valid derivative of the finite program only when all
declared dependencies are differentiated. It is not automatically an estimate
of \(S(\theta)\).

### KDM expectation gradient

For a mixture \(m_\theta\) and integrand \(F_\theta\),

\[
 \mathcal J(\theta)=\int F_\theta(z)m_\theta(z)\,dz.
\]

Assume a common support, differentiability, integrable domination permitting
differentiation under the integral, and a positive sampling density wherever
the derivative integrand contributes. With \(q_0=m_{\theta_0}\) held fixed
while differentiating,

\[
 \mathbb E_{q_0}\left[
 \nabla_\theta F_\theta(Z)|_{\theta_0}
 +F_{\theta_0}(Z)\nabla_\theta\log m_\theta(Z)|_{\theta_0}
 \right]
 =\nabla_\theta\mathcal J(\theta)|_{\theta_0}.
\]

This is the scope of the Younis IWSG identity. If \(F=\log g(y\mid z)\),
the target is an expected conditional log likelihood. In general,

\[
 \mathbb E_m[\log g(y\mid Z)]
 \neq \log\mathbb E_m[g(y\mid Z)].
\]

### Unnormalised pair and normalised score

The program must retain separately

\[
 \widehat Z_N,\qquad \widehat D_N\approx\nabla Z,
 \qquad \widehat S_N=\widehat D_N/\widehat Z_N.
\]

Unbiasedness of the first two does not imply unbiasedness of the third.

## Evidence contract

The primary comparison question is conditional model-score accuracy for a fixed
dataset and parameter, averaged over independent particle randomisations. The
exact comparator is an analytic or numerically certified score oracle. When no
oracle exists, the result is restricted to identity checks, convergence
evidence, and descriptive comparisons; it is not called unbiased or accurate
without a derivation.

The primary promotion criterion is a predeclared paired error measure against
the oracle, such as squared Euclidean score error or a fixed scale-normalised
version. Choose the metric before selection and account for oracle uncertainty
when it is nonzero. Confidence intervals respect independent datasets and the
particle replicates nested within them; the latter are not independent
datasets. Any claim of improvement needs its predeclared uncertainty criterion,
including an effect-size margin or precision target where relevant.
Required vetoes are:

- non-finite values, invalid supports, negative or zero proposal densities where
  the numerator has mass, failed derivative identities, or failed likelihood
  normalisation checks;
- a mismatch between the implemented target and the claimed target;
- missing initial-law, transition, observation, mixture, Jacobian, or total
  derivative terms;
- use of a regular-transition identity on a degenerate transition without a
  support derivation;
- data leakage from tuning or control-variate fitting into the claim run; and
- a failed heuristic-dominance check in any salient regime.

Explanatory diagnostics include bias/variance decomposition, ESS, weight tails,
ancestor entropy, mixture overlap, bandwidth sensitivity, covariance calibration,
OT marginal error, and runtime. They cannot replace the primary criterion.

Before each research run, record the main question, mechanism, expected
failure, promotion criterion, promotion veto, continuation veto, repair
trigger, explanatory diagnostics, and forbidden inference. Classify every
diagnostic by these roles. A failed candidate is not a continuation veto when
a planned repair addresses that failure. A model-score error claim and a
finite-program derivative claim require separate evidence even if they use
the same numerical estimates.

Passing this program does not establish posterior correctness for an untested
model, HMC readiness, asymptotic unbiasedness, or superiority outside the
declared scope.

## Campaign envelope and stop rules

### Initial implementation and validation allocation

The first tranche of this master is Phases 0A--0C plus Phase 0F's
deterministic mechanics. It includes implementation, tiny CPU reference
checks, and a bounded GPU integration check. The initial engineering
validation allocation is 12 CPU process-hours and 8 GPU device-hours,
including compilation and failed runs, with at most 12 GPU launches.
Reserve 2 of those GPU hours for repairs. These are planning allocations,
not measured costs or scientific precision thresholds; use the first timing
observation to reassess the remaining allocation and record exhaustion
instead of weakening a check.

Phase 0A pins the available interpreter/environment and exact commands for
the next phase. It does not guess a conda environment or install packages.
GPU checks use trusted/escalated access and verified memory growth; CPU
references explicitly set `CUDA_VISIBLE_DEVICES=-1`. Later phase refreshes
record their finite row counts, replications, compute caps, and attempt
budgets before serious launches. Establishing those values is a task inside
the master, not a request for a separate preparatory program.

### Scientific comparison allocation

“Budget and time are not a problem” permits exhaustive coverage, but it does
not remove the need for a finite, reproducible campaign definition. The initial
claiming envelope is:

- at least 100 independent observation datasets per regular model/regime;
- 64 independent particle randomisations per dataset for the first claim run,
  with a maximum of 256 if the paired interval remains too wide;
- particle counts \(N\in\{32,64,128,256,512,1024,2048\}\) and horizons
  \(T\in\{1,5,20,50,100\}\) where the model supports them;
- all baseline and identity rows, all single-factor ablations, and the complete
  factorial only for candidates that pass the preceding phase gates; and
- no replacement of a failed artifact: each repair or retry gets a new,
  versioned output directory.

These values are an explicit planning envelope, not inherited scientific
defaults. A row may stop early when a hard veto fires, when its target identity
fails, or when a repeated implementation failure has an established cause.
Stop unchanged retries and enter the phase-boundary repair process; this is
not a veto on a justified repair or an independent row. Three unsuccessful
repair trials for one cause require diagnostic reassessment and an explicit
remaining-budget disposition. A viable row may stop replication when its predeclared paired
interval is sufficiently narrow for the decision question; otherwise it is
reported as descriptive-only at the maximum replication count. A candidate
that fails a promotion criterion but has a planned repair continues to that
repair; it is not silently discarded as evidence against the whole method.

Before a serious launch, its concise run plan must fill in the finite number
of eligible rows, maximum attempts, total wall/GPU-hour budget, hardware,
uncertainty precision needed, and stop conditions. The large grid above is a
coverage proposal, not an instruction to launch its Cartesian product. A pilot
determines feasible allocation and power; its outcomes cannot select claim
data. Any optional replication increase uses a predeclared sequentially valid
interval/stopping rule or a fixed terminal sample size. Budget exhaustion
leaves an unresolved result; it does not relax a scientific criterion.

The campaign root is a unique versioned directory under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`. Every launch records
the exact matrix manifest, git revision, environment, hardware, seeds, command,
wall time, and output hashes. The campaign is closed only after each planned
row is classified as promoted, viable-but-underranked, failed-with-repair,
failed-support, or unresolved-evidence.

## Skeptical pre-mortem

Assume the program produces an apparently favourable KDM or LEDH result. It
could still be misleading for the following reasons:

- the implementation differentiates the finite cloud while the claim is about
  the marginal likelihood; the earliest check is target metadata plus the
  analytic linear-Gaussian score;
- the KDM bandwidth or OT reset changes the propagated measure and the apparent
  gain is a target change; the earliest check is a physical-numerator,
  evaluable-denominator identity test;
- the score error is dominated by a random denominator, so a lower variance
  expectation gradient is irrelevant to the reported score; the earliest check
  is the separate \(\widehat Z_N\), \(\widehat D_N\), and ratio study;
- a missing initial or transition derivative is hidden by a fixed-cloud finite
  difference; the earliest check is a parameter-dependent initialization test
  and an independent exact Gaussian derivative fixture;
- tuning or control-variate fitting leaks information from the claim set; the
  earliest check is a scope-bound manifest with disjoint data hashes;
- a complex route loses to a cheap UKF, adapted PF, or bootstrap method in a
  salient regime while its pooled mean looks favourable; the earliest check is
  the conditional heuristic table; and
- a regular-transition success is reported as a DSGE result; the earliest check
  is the support-class field and a mandatory derivation reference for every
  degenerate run.

Any one of these findings blocks promotion until the corresponding repair is
completed. A failed candidate remains useful evidence about that candidate; it
does not by itself terminate the broader program.

## Option matrix

Every active candidate requires a complete estimator and target label.
In this inventory, Fisher/backward smoothing (including Nemeth, PaRIS,
Fearnhead, JLS), iterated filtering, continuous-likelihood methods, multilevel
debiasing, variational objectives, and degenerate coordinates are deferred
literature/reference rows. Their presence does not create executable tasks.
KDM/IWSG filtering, covariance/proposal alternatives, twisting/iAPF, and FD are
active subject to their named prerequisites.

| Family | Candidate | Intended role | Main question |
|---|---|---|---|
| Baseline | Bootstrap PF with a declared value/score route | classical reference | How much error comes from ordinary particle approximation? |
| Baseline | Locally adapted proposal with exact importance correction | tuned classical reference | Does simple observation adaptation explain any gain? |
| Baseline | EKF and UKF approximate-likelihood scores | cheap heuristic adversaries | Is low-order Gaussian moment information sufficient? |
| LEDH | Canonical LEDH-PF-PF-OT with analytical recursive score | project baseline | What does the existing finite algorithm achieve? |
| KDM | KDM proposal with physical transition numerator and evaluable proposal denominator | model-corrected proposal | Does KDM improve coverage without changing the target? |
| KDM | KDM expectation/IWSG gradient | declared expectation objective | What is its variance and bias for its own objective? |
| KDM/Rao--Blackwell | Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and observed information | direct regular-model score comparator | Does the published linear-cost kernel estimator improve horizon variance under its bandwidth and mixing assumptions? |
| Covariance | UKF per-particle covariance, KDM weighted covariance, and hybrid covariance | proposal-shape study | Does a better covariance estimate improve the score through proposal quality, or only change a finite approximation? |
| Fisher | Forward particle score with complete-data terms | direct score estimator | Can transition and observation score terms be propagated stably? |
| Smoothing | Backward pair/Fisher estimator | regular-transition only | Does ancestor averaging reduce score variance or finite-\(N\) bias? |
| Smoothing | Fearnhead--Wyncoll--Tawn sequential smoother | regular-transition only | How do linear- and quadratic-cost additive-functional smoothers compare? |
| Smoothing | PaRIS with explicit backward-draw count | regular-transition only | Does \(\widetilde N\ge2\) control path-degeneracy variance at the target horizon? |
| Debiasing | Coupled conditional particle filter with Rhee--Glynn correction | regular tractable-transition only | Can unbiased smoothing expectations and Fisher scores be obtained at acceptable random cost? |
| Rao--Blackwell | Analytically integrate tractable state blocks or ancestor indices | variance reduction | Which conditional integrations are exact and worthwhile? |
| Control variate | Exact-mean control variate for unnormalised derivative | variance reduction | Does known centering reduce variance while preserving expectation? |
| Control variate | Tractable reference plus residual | variance reduction | Does a separately corrected approximate model help? |
| Hybrid | Pathwise plus importance-weight gradient | gradient variance candidate | Can the two estimators be combined without changing the target? |
| Combination | Calibrated linear combination of two score estimators | bias--variance study | When do two biased but target-matched estimators have lower MSE? |
| Multilevel | Coupled \(N\)/bandwidth or randomized telescoping estimator | bias reduction | Is there a verified expansion supporting debiasing? |
| Proposal correction | Forward/backward mixture proposal with generative correction | proposal design | Can future-data coverage help while preserving the model measure? |
| Continuous likelihood | Malik--Pitt/DeJong-style continuous particle likelihood | continuous-resampling comparator | Which discontinuity is removed, and what finite-particle measure and derivative does the method target? |
| Simulation-only sensitivity | Ionides iterated filtering | unavailable-transition-density comparator | What bias, variance, and mixing cost arise as artificial parameter noise decreases? |
| Iterated APF | Guarniero--Johansen--Lee twisted/auxiliary filter | proposal-quality comparator | Does future-data twisting reduce normalizer and downstream score variance when the analytical score recursion is held fixed? |
| Proposal variance | Whiteley--Lee twisted particle filter | proposal-only variance control | Can normalizer variance be reduced without changing the physical target correction? |
| Proposal moments | Fixed structured Gaussian quadrature filter (SGQF) moments | high-dimensional proposal-design comparator | Do sparse-grid projected moments improve proposal coverage at an admissible cost without replacing the physical target? |
| Directional validation | Three-point, five-point fourth-order, and eleven-point cubic finite differences | oracle-calibrated score diagnostic | Which fixed positive \(h\)-ladder gives the best bias--variance tradeoff for each directional derivative? |
| Discretization | Jasra--Kamatani--Law--Zhou multilevel particle filter | coupled-level variance control | Does a score-level telescope exist for the chosen discretization and coupling? |
| Degenerate | Ancestor/innovation-coordinate estimator | unresolved support-aware route | Can a valid base measure and score identity be derived for DSGE? |

## Phase 0: specification and oracle ladder

This phase states the scientific specifications for the implementation phases
that follow. Phases 0A--0C build and verify the catalogue, oracles, baselines,
and tuning services specified here. Phase 0 does not assume those services
already exist, and Gate A is issued only after the applicable implementations
and tests pass.

### 0.1 Build a model catalogue

Use a minimum ladder of:

1. scalar and multivariate linear-Gaussian state-space models with exact Kalman
   value and score;
2. nonlinear regular-transition models with high-accuracy reference scores;
3. models with multimodal or strongly curved posteriors;
4. the existing repository models used by the LEDH adapters; and
5. the degenerate DSGE model as a deferred applicability record, without
   scheduling its support derivation or experiments in this program.

Record state dimension, parameter dimension, horizon, transition rank,
observation rank, support, analytic oracle availability, and the model's
parameter-dependent initial law.

### 0.2 Construct the oracle ladder

For each model, implement in order:

1. closed-form value and score where available;
2. deterministic quadrature or Rao--Blackwellised integration;
3. high-particle, replicated reference estimates with an error certificate;
4. an independent implementation using different code and random numbers; and
5. externally supplied support-specific references when applicable, with
   provenance and applicability checked before import.

Use the highest justified reference available; an unavailable closed form
does not block construction of a valid next rung. Replication at high particle
count bounds Monte Carlo uncertainty, not approximation bias by itself.
A numerical certificate must also bound relevant discretization/particle bias;
otherwise label the reference provisional and restrict the resulting claim.

An oracle is admitted only after value, directional derivative, finite-difference,
and parameter-coordinate checks agree within predeclared tolerances.

### 0.3 Freeze notation and target metadata

Every result records `target_id`, `measure_id`, `support_class`, `score_kind`,
`normalization_kind`, `proposal_id`, `reset_id`, `derivative_id`, and whether the
proposal is detached locally. A result with missing metadata is diagnostic only.

### 0.4 Prepare baselines and tuning before quality comparisons

Build and verify the exact Gaussian oracle and the applicable cheap baselines
before the first proposal or estimator comparison. For an active regular model,
the minimum ladder is bootstrap PF, an EKF/UKF approximation, a classically
adapted proposal where evaluable, canonical LEDH, and the proposed enhancement.
The Kalman oracle measures error in Gaussian fixtures; it is not a Monte Carlo
competitor that a particle method must beat. Define the heuristic adversaries
and salient regimes from Phase 7 now, but reserve their final evaluation as a
falsification check rather than a tuning objective.

Scope-specific tuning is a reusable prerequisite of every quality claim in
Phases 2, 3, 4, 4B, 4C, and 7; it does not wait until Phase 8. Bind model,
target, horizon, particle count, dimensions, dtype/TF32, backend, chunk policy,
proposal/estimator route, and all controls, including bandwidth, covariance
protection, twisting fit, finite-difference span/directions, and combination
coefficients. A horizon or particle-count change creates a new scope.
Partition calibration, validation/selection, and untouched claim observations
and particle streams in advance. Fit or choose controls on permitted partitions,
freeze them, and require the repository-issued scope artifact in every claim
row. Mechanics and debugging fixtures need no statistical tuning artifact but
cannot be used for quality claims.

For each material numerical default, record provenance, justification, likely
failure mode, smallest diagnostic, and status as baseline, hypothesis, or
reviewed choice. In particular, imported bandwidths, ridges, covariance floors,
lookahead families, step sizes, and conditioning limits are hypotheses until
justified for this scope. Phase 8 performs final replication and cost analysis.

### Implementation defaults and early diagnostics

Apply this audit while implementing Phases 0A--0G. The later experiment plan
refresh specifies target-specific values before the corresponding run;
unexamined inherited settings cannot become defaults through a passing smoke.

| Choice | Provenance and purpose | Failure mode and earliest check | Status |
|---|---|---|---|
| Regular Gaussian first | Phase 0; exact score separates target and implementation errors. | Apparent success hides missing initial-law derivatives; use a nonconstant-initialization fixture. | Reference baseline, not cross-model evidence. |
| Canonical analytical LEDH/GenUT/Contract E | Current owner policy. | A simplified lane appears to pass; test the actual consumer and reset composition. | Required baseline identity, with correctness still to verify. |
| Tiny smoke allocation and first-tranche caps | Master implementation allocation; make the first failure cheap and reserve repair time. | A smoke is misreported as a ranking, or compilation exhausts allocation; record no-ranking status and early timing. | Convenience allocation, not a scientific threshold. |
| Existing tuning services and kernels | Repository reuse candidates. | Their API exists but cannot issue/consume the required scope; exercise the complete path. | Unverified until tested. |
| Step sizes, bandwidths, ridges, twist families, and covariance controls | Selected derivations plus per-scope calibration. | Bias/robustness or target changes are concealed by numerical success; log realized choices and earliest validity/curve checks. | Hypotheses until justified, including off/zero settings. |
| Statistical precision and oracle tolerances | Exact identities, dtype/scale analysis, reference convergence, and powered pilot. | Arbitrary cutoffs or replication-driven selection; validate error scale and freeze a valid uncertainty rule. | Must be specified in the refreshed run plan, never inferred from a successful smoke. |

## Phase 0A: reconcile once, then build

Entry: Phase 0's scientific requirements and the current checkout. This is the first implementation task of the master and needs no master CLI.

1. Read the current canonical repair result and inspect the actual callable
   interfaces. Record each capability as verified, unverified, blocked, or
   deferred, with its first executable test. Refresh stale source claims.
2. Reconcile the manuscript's smoothness requirement, deterministic versus
   stochastic FD checks, and SGQF prediction/conditioning/reset description
   with the amended master. Preserve the existing derivations and source
   boundaries. Fix the malformed `,qquad`, build, and inspect the affected pages.
3. Define the exact numerical return contract: log likelihood/value target,
   derivative target, coordinate system, initial-law dependence, denominator,
   proposal law, and oracle precision. Specify which outputs are model-score
   estimates and which are derivatives of a different quantity.
4. Inventory reuse before adding modules. Identify real production/candidate
   baseline endpoints; diagnostic reference functions remain reference lanes.
   Verify paper equations and author code for method-faithfulness claims when
   their numerical implementation is selected. Do not wait for a new broad
   literature survey to build the coordinator.

Exit: no unresolved ambiguity about the first baseline's mathematical target
or the 0B interface. Remaining method gaps have named tests and block only
their dependent rows. Refresh 0B/0C using the current repair status.

## Phase 0B: implement a small coordinator that cannot hide failures

Entry: Phase 0A's target/return contract. This phase builds the infrastructure used by later numerical rows.

Proposed new locations, to be reconciled with existing services in 0A:

- `bayesfilter/score_study/`: typed study/row definitions, registry
  validation, adapters, phase state, and report assembly;
- `scripts/run_younis_score_master.py`: a thin CLI over those services;
- `docs/plans/configs/younis_score/`: versioned baseline and later study inputs;
- `tests/highdim/test_younis_score_master_*.py`: coordinator contract and
  integration tests.

These paths are planned deliverables, not existing executable commands.
Keep Python standard-library scheduling, JSON, and provenance outside the
numerical kernels. Use TensorFlow/TFP for candidate numerical calculations,
tuning, and admission decisions. No NumPy runtime dependency is introduced.

Execution reconciliation: the coordinator lives directly under `bayesfilter`
because importing `bayesfilter.highdim` eagerly loads numerical modules.
This preserves validation without TensorFlow/GPU initialization. Numerical
adapters may import the existing high-dimensional providers when run.

The CLI needs `validate`, `dry-run`, `run`, `resume`, and `report` actions.
Validation and dry-run must enumerate both runnable rows and excluded rows
with reasons without importing a GPU-initializing kernel. A run may not
silently reduce the requested matrix. Resume checks the saved scientific
specification, source dependencies, partitions, tuning scope, and completed
results before reusing work.

Code or mathematical repair remains work for the supervising developer or
agent. The CLI records the changed revision and issue disposition, schedules
declared regression/retry commands, and enforces the updated dependencies; it
does not generate arbitrary source fixes or silently choose a new method.

Implement separate fields for numerical validity, scientific candidate
decision, and execution completion. A low-MSE failure, unavailable kernel,
insufficient precision, and a deferred method are different states. Store
paired randomness by model, dataset, particle replicate, perturbation, and
coupling group, independent of scheduling order.

Required tests include incompatible support/target rejection; stale tuning;
missing initialization terms; missing active methods; deterministic row
generation; repeated/resumed execution; bounded attempts; incomplete output
rejection; preservation of failed attempts; and a blocked branch alongside an
independent runnable branch. Test the repair/refresh transitions explicitly.
A Phase 0B fake numerical endpoint can test scheduling, but its evidence must never
be presented as a real particle-filter run.

Exit: a synthetic failing task can be repaired and resumed with the right
dependencies and artifacts. This establishes coordinator behavior only.

Implement and test the following interface in this phase. The paths and
commands describe deliverables to create, not code already implemented:

```text
python scripts/run_younis_score_master.py --study <baseline.json> --action validate
python scripts/run_younis_score_master.py --study <baseline.json> --action dry-run
python scripts/run_younis_score_master.py --study <baseline.json> --action run --output <new-run-directory>
python scripts/run_younis_score_master.py --study <baseline.json> --action resume --output <existing-run-directory>
python scripts/run_younis_score_master.py --study <baseline.json> --action report --output <existing-run-directory>
python scripts/run_younis_score_master.py --study <tuning.json> --action select --output <tuning-run-directory> --selection <new-selection.json>
python scripts/run_younis_score_master.py --study <baseline.json> --action compare --output <existing-run-directory>
```

Replace placeholders with exact versioned paths when the phase creates its
fixtures and study configuration; save actual commands in the manifest.
Resume reads validated existing results and writes new attempts to fresh
subdirectories without overwriting evidence. Phase 0C exercises this
interface with real baseline endpoints. This CLI is an output of master
execution, not a precondition for beginning Phase 0A.

## Phase 0C: deliver a real baseline study before adding enhancements

Entry: Phases 0A--0B for orchestration; baseline-specific numerical repairs are part of this phase. Independent reference fixtures may be developed while the coordinator is being built.

Use scalar and multivariate regular LGSSMs. Include parameter effects on
initial mean/covariance, transition, and observation parameters rather than
only the existing transition-direction example. Verify the independent
analytical Kalman oracle and explicit total derivatives. Same-finite-program
parity and error against the exact model score are separate tests.

Construct the initial cheap comparator set: bootstrap PF, UKF Gaussian
approximation, and the analytically adapted Gaussian proposal where its
density is available. Kalman is the error oracle. Add the canonical
LEDH/GenUT/Contract E baseline only after its actual endpoint passes the
required lifecycle, reset, determinant, and analytical-sensitivity checks.
Keep approximation and finite-stream targets visible in every comparison.
Keep prior and adapted sequential importance sampling without resampling as
separate naive arms. The bootstrap/adapted PF arms resample explicitly. For
discrete resampling, a derivative with ancestor labels locally fixed is an
almost-everywhere finite-program derivative; it does not establish an unbiased
derivative of the program's expectation. Common uniforms preserve marginal
multinomial laws but do not remove this target distinction.

Build the tuning adapter and reports now. Each claim scope binds all master
fields, including proposal/estimator controls, bandwidth, FD choices, dtype,
TF32, horizon, particle count, and backend. Check the repository's actual
tuning issuer and reject a caller-stamped identity. Tune on calibration and
validation partitions; claim data and streams remain untouched until freeze.
Mechanics smokes do not need statistical tuning and cannot support a quality
claim.

The report preserves raw dataset/replicate results, paired oracle errors,
uncertainty, conditional heuristic tables, failures, oracle limitations, and
cost. Test aggregation on independently known small examples, including
shared versus independent particle noise and datasets with unequal numbers
of successful replicates. Do not silently discard failures or treat particle
replicates as independent observation datasets.

Run CPU-only reference checks with `CUDA_VISIBLE_DEVICES=-1`. Verify candidate
kernels separately on trusted GPU/XLA with stable signatures, analytical
derivatives, verified memory growth, TF32 recorded, and canonical chunking.
Autodiff is parity evidence only. Do not recover a missing capability through
a scalar fallback, pfor, or a reduced LEDH fork.

Exit: the same study specification drives a real baseline smoke, a scoped
tuning/claim plumbing check, artifact aggregation, and an interrupted-run
resume. The tiny claim-plumbing fixture carries explicit no-ranking status.
All required baseline rows either pass or keep 0C incomplete; running only
the oracle does not close this milestone. End with repair and refreshed 0D,
0E, and 0F plans. A statistical baseline claim requires its own sufficiently
powered run after this engineering exit.

## Phase 0D: KDM/IWSG, control variates, and estimator combinations

Entry: the Phase 0C baseline and the capabilities needed by the selected KDM row. Complete the shared executor repair here when it remains open; no external implementation program is required.

First test frozen-mixture IWSG identities against analytic integration or exact
small examples. Then vary independent clouds to separate conditional sampling
variance from cloud variation. Fix trace/callback support and required
analytical tangents in the shared canonical executor before integrated or
resampling KDM runs. Compare scalar and every claimed batch/device lane through
the consumer, including parameter-dependent initialization and bandwidth.

Implement the actual KDM proposal density, physical numerator, component
covariance lifecycle, and any OT/jitter law used. Check that samples and weight
denominators describe the same distribution. Do not add an OT determinant
without a corresponding density derivation or present a changed measure as
the original filter.

For an exact control variate, identify the sampling law and either a known
center or an independent unbiased center estimate. Test the coefficient and
center dependence conditions, account for center cost/variance, and report
the unchanged mean of the baseline estimator. For two biased estimates,
implement a separate calibration path using oracle error and held-out MSE;
variance minimization alone does not determine the better combination.

Exit: conditional identities and full-filter wiring pass independently; ratio
bias and target distinctions remain explicit. A variance reduction result
alone cannot close a model-score improvement claim. If centering is unknown,
that exact-CV row remains blocked while a correctly labeled calibrated
combination can still run.

## Phase 0E: covariance alternatives and corrected proposals

Current execution note (2026-09-18): the
[fitting/score diagnosis](artifacts/younis-iapf-curved-diagnosis-20260918-01/result.md)
has completed the next diagnostic step. Fitting limitations remain, but
sampling-law contributions now require an explicit analytical score study.
Finite-program parity and physical-model score agreement remain separate
requirements. No iAPF candidate is promoted and no broader phase exit is claimed.

Execution update (2026-09-15): Phase 0D's two full KDM consumers, conditional
identities, known-center control and independent calibrated blend now execute.
See `artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0d-result-and-refresh.md`.
The 48-row CPU fixture and three GPU/XLA rows are mechanics evidence only.
The exact-center control had higher observed validation error; the small
calibrated blend was descriptively favorable. Neither result establishes a
ranking or removes the need for the later scientific comparison.

The [density-fit iAPF prerequisite](younis-score-iapf-implementation-2026-09-15.md) implements the density-scale objective and adaptive iteration controller alongside the log-quadratic comparator. Source `cf823241` passes the scalar Gaussian GPU consumer and derivative checks. The unrestricted objective need not attain its infimum; numerical bounds, local optimizer, stopping rule, offline fitting precision and realized particle count are explicit. [Adaptive-N selection/reporting](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-result-and-refresh.md) now executes at `1c12eefa`, binding the selected procedure and recording its realized count and full offline/final work. Ten GPU rows pass, including actual count growth; this is mechanics evidence, not scientific promotion. The [control-safety screen](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/control-safety-result-and-refresh.md) completes its bounded observability and domain repair; comprehensive control calibration remains open. The [nonlinear extension](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-iapf-result-and-refresh.md) at f5a4d411 now executes through shared kernels: twenty GPU rows pass, including independent final streams, actual adaptive counts, and physical/executed observation validation. The inherited fit controls fail the descriptive heuristic screen and often reach their declared bounds. Next audit and execute complete numerical-control calibration using fresh partitions; cap-only or k-only sweeps cannot close this obligation. Wider dimensional coverage and use of fitted iAPF moments inside the LEDH covariance lifecycle remain separate implementation work.

Entry: Phase 0C for the affected baseline. A KDM covariance provider also requires the relevant Phase 0D mixture-law tests; other providers need not wait for KDM.

Implement one provider interface for prior/predicted moments, observation
conditioning, persistent per-component state, reset/ancestry mapping, and
analytical sensitivities. Integrate UKF first, then KDM and SGQF. A
prediction-only replacement cannot stand in for a complete filtering
covariance lifecycle. Test the actual moments and derivatives consumed by
LEDH with linear-Gaussian exact moments and nonlinear stress examples.

SGQF requires separate checks for signed integration weights versus sampling
probabilities, projected covariance validity, sparse-grid point count,
dimension/level growth, and the real GPU/XLA call chain. Existing standalone
code does not certify that chain. Evaluate covariance safeguards explicitly;
record any ridge, clipping, or damping as a numerics-altering choice and test
its non-harm before promotion.

For twisting/iAPF, start with a positive fixed one-step lookahead whose
ancestor law, transition proposal, normalization integral, and weight
correction can be derived and checked. Test a finite discrete example and a
Gaussian example before fitted twists, including initialization, terminal
factors, telescoping, and recovery of the untwisted law when the lookahead is
constant. Inspect primary equations and author
code for source-faithfulness claims. Select hyperparameters and the fitting
procedure on calibration data. That frozen procedure may fit a proposal to each
new observation dataset using independent offline streams; freeze its fitted
coefficients and realized count before final sampling. Oracle scores, heldout
errors and final sampling noise must never enter fitting. Distinguish held-fixed learned controls from
their explicitly parameter-dependent evaluations and include the latter's
derivatives. An unknown required normalizer blocks that proposed correction.

Exit: each provider/proposal has density and total-derivative tests and a
consumer integration test. Better covariance or ESS alone is explanatory;
score-MSE improvement still requires untouched evidence. An unrepairable
within-scope proposal assumption rejects that row, not the rest of 0E.

## Phase 0F: finite differences and error attribution

Entry: Phase 0A for deterministic mechanics, implemented here on exact fixtures. Stochastic FD needs Phase 0C value endpoints; normalization and consistency studies need only the implemented estimator they assess.

Move the already checked exact three-, five-, and eleven-point examples into
maintained diagnostic tests. Include polynomial cancellation, smoothness
counterexamples, rectangular direction recovery, deficient rank, parameter
boundaries, and precision/conditioning checks. They may be developed before
KDM or SGQF integration is complete.

Implement the stochastic FD lane using verified value endpoints and an
explicit coupling of replicas. Check that each replica has the correct
marginal law; an OT coupling here is distinct from OT as a filter reset.
Measure the stencil covariance and the full bias-square term, including the
cross-term. Use a positive scale-aware step ladder and QR/SVD direction
reconstruction. Do not require a universal U-curve or a stochastic slope equal
to deterministic stencil order.

Once eligible estimators exist, implement the value/derivative/ratio study
and Phase 4B consistency diagnostics. Fit step sizes, combinations, and any
diagnostic-based decision rule on calibration/validation only. Measure oracle
error and diagnostic correlation with uncertainty on separate data. Unknown
bias attribution limits interpretation; it does not invalidate a correctly
measured held-out MSE comparison.

Exit: exact mechanics, coupling, reconstruction, normalization targets, and
partition checks pass. Approximate consistency remains evidence about the
tested relationship, not proof of unbiasedness.

## Phase 0G: implement model coverage, study configurations, and final reports

Entry: the applicable Phase 0C baseline and the specific 0D--0F capabilities
used by each study. Implement this phase in slices before each new model,
regime, dimension/level, or method interaction is measured. It is not a
requirement that every advanced method finish before any experiment begins.

1. Implement regular nonlinear model adapters, data generators, derivative
   inputs, and usable reference providers. Include the weakly nonlinear,
   strongly curved, multimodal, long-horizon, and high-dimensional regimes
   actually selected for the study. Check reference uncertainty and the full
   adapter-to-kernel call chain. A missing oracle limits the allowed conclusion
   and does not authorize inventing an accuracy certificate.
2. Implement explicit study configurations for the baseline ladder,
   single-factor changes, pairwise interactions, and the eligible larger
   matrix. Bind each row to its target, support, method implementation,
   tests, tuning scope, data/stream partitions, heuristic comparators, and
   evidence role. Include failed-candidate repairs that the later study is
   designed to discriminate; a failed single-factor arm cannot silently
   delete its predeclared repair experiment.
3. Implement capacity/timing measurement and bounded pilots before SGQF
   dimension/level expansion and factorial launches. Capture compile time,
   memory, point count, per-row costs, and failed attempts. The pilot updates
   the next run's finite budget and cannot select claim observations.
4. Implement report coverage for matched particle count and matched total
   cost, including tuning, twist fitting, center estimation, additional FD
   calls, compilation, and explicit amortization assumptions. Construct the
   conditional heuristic tables and nested/paired uncertainty calculations
   for each selected regime. Failed rows remain visible.
5. Implement final replication and audit commands and their integration
   tests before Phases 8--9. Test untouched-partition enforcement, complete
   accounting for requested methods, and invalidation after a terminal
   repair. Trace every scientific output back to its actual numerical
   endpoint and source revision.

Exit: each requested study has a validated configuration, actual model and
method endpoints, reference status, measured feasibility, and tested report
assembly. A tiny integration run establishes that this computation works;
it does not establish the subsequent scientific comparison. Missing
implementations keep their rows incomplete, with an explicit repair or
blocked disposition. Refresh the applicable Phases 2--4C and 7--9 after the
repair pass.

### Implementation coverage and initial capability inventory

The existing code below is input to Phase 0A's audit, not evidence that the
new integrations have passed. Verify actual calls and tests before reuse.

| Existing component | Implementation phase and required check |
|---|---|
| [Canonical scalar executor](../../bayesfilter/highdim/ledh_canonical_score_tf.py) and [ongoing loop repair](ledh-while-loop-regression-repair-plan-2026-09-14.md) | 0C/0D: preserve the shared canonical algorithm, repair initialization and required trace/callback capabilities, verify all affected consumer lanes. |
| [KDM primitives](../../bayesfilter/highdim/ledh_younis_kdm_tf.py), [integrated route](../../bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py), and [resampling route](../../bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py) | 0D: verify mixture laws, identities, covariances, corrections, and analytical consumer derivatives. |
| [LGSSM reference](../../bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py) | 0C: extend or select an oracle for all claimed parameter dependences; retain the bootstrap's fixed-stream finite-program target. |
| [SGQF values](../../bayesfilter/nonlinear/fixed_sgqf_tf.py) and [derivatives](../../bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py) | 0E/0G: implement the LEDH filtering lifecycle, consumer sensitivities, GPU/XLA capability, and scaling measurements. |
| [Tuning scopes](../../bayesfilter/highdim/ledh_tuning_scope.py) and [route registry](../../bayesfilter/highdim/ledh_tuning_registry.py) | 0B/0C: verify the complete issuer/consumer path; a scope type alone is not a tuner. |
| [GPU memory policy](../../bayesfilter/runtime/gpu_memory_policy.py) | 0C: configure and verify before device initialization, then record the actual backend policy. |
| [Previous Phase 4A runner](../benchmarks/run_ledh_younis_kdm_phase4a_campaign.py) | 0B/0G: audit reusable code against the amended targets, partition rules, and reports; it is not the new master merely because it is a CLI. |

Before a research row runs, its implementation record must identify the
producing phase, real callable, required capability tests and their results,
source dependencies, current tuning scope where applicable, and result/report
consumer. An absent field identifies unfinished master work. The record does
not authorize callers to self-attest canonical route identity.

## Phase 1: identity and call-chain verification

Phase 1 is admission of the executable evidence produced during Phases
0C--0G, evaluated per method and target. Its tests are implemented and run
inside those prerequisite phases and reused here at the checked revision.
Implementation does not wait for a later Phase 1 pass to create its own tests.
An admission failure returns to the responsible implementation phase's repair
step; it does not create a separate execution program.

Before measuring quality, choose the identity test from the estimator's
declared mathematical target:

| Target | Required evidence |
|---|---|
| Analytical derivative of a fixed finite scalar | Compare with that same scalar at fixed randomness using diagnostic FD/autodiff where differentiable; include initial and recursive dependencies. |
| IWSG gradient of a mixture expectation | Derive the locally fixed-proposal identity with support and differentiation/integrability assumptions; compare repeated estimates with a known expectation gradient. Samplewise equality with a different pathwise estimator is not required. |
| Unnormalised value/derivative pair | Verify the sampling law and the stated expectation/derivative identities separately; declare whether \(\widehat D\) differentiates this particular \(\widehat Z\). |
| Estimate of the model score | Verify its own derivation and implementation, then measure oracle error and uncertainty. Neither pathwise parity nor low MSE proves unbiasedness. |

All applicable initial, transition, observation, mixture-weight, covariance,
bandwidth, flow-Jacobian, OT, and GenUT dependencies must be covered.
Parameter-dependent initial means and covariances need explicit sensitivity
fixtures. First compare the same finite scalar; separately compare the
estimator statistically with Kalman. A finite particle realization need not
equal the exact model score.

Require executable consumer-to-provider tests for rank, batch, dtype,
graph/XLA/device, and analytical-sensitivity contracts. For a model-corrected
proposal, verify the actual sampler/denominator law and physical numerator.
Support and positivity checks apply to the relevant density or covariance;
signed quadrature integration weights are not sampling probabilities.

The protected execution snapshots have passed actual integrated-KDM,
resampling-IWSG, SGQF and persistent-mixture consumer tests. They also include
explicit initial-state and initial-covariance tangents, checked against the
six-parameter Gaussian oracle and fixed-program finite differences. This
resolves the original prerequisites for those snapshots. The main checkout's
native-loop integration subsequently passed its own endpoint regressions and
GPU mechanics checks, including the repaired callback interface. The versioned
integration result records those checks; passes remain tied to their actual
source revisions and do not imply scientific promotion.

## Phase 2: proposal and representation study

Execution update, 15 September: the first nonlinear settings failed the
conditional heuristic screen descriptively. The next repair is the
[nonlinear calibration and untouched pilot](younis-score-nonlinear-calibration-2026-09-15.md):
nine proposal/regime calibration scopes, frozen selections and fresh evaluation
against four constructed heuristics. Phase 0G now includes
`scripts/run_younis_score_campaign.py` and
`bayesfilter/score_study/heldout_reporting.py`, with executable tests for
selection-before-evaluation and dataset-level uncertainty. These are master
implementation tasks. The 1,440-row GPU pilot completed after these checks
passed; all nine selections preceded untouched evaluation. Its
[result and phase refresh](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-calibration-result-and-refresh.md)
records the heuristic promotion veto in all three regimes. Concentrated-regime
paired intervals include zero. Incomplete control calibration and modest
dataset count prohibit final ranking or default claims. Preserve the opened
evaluation data and continue to fresh calibration and the remaining method
prerequisites; this is not a continuation veto on the research direction.

Hold the score estimator fixed and vary only the proposal representation. The
factorial factors are proposal family, lookahead construction, covariance
source, particle count, horizon, observation regime, and compute budget:

1. bootstrap transition proposal;
2. UKF/EKF adapted proposal;
3. LEDH flow proposal;
4. KDM with explicit component covariances;
5. OT-resampled KDM cloud;
6. a fixed lookahead mixture proposal (backward-smoothing extensions deferred);
7. mixtures with a declared support-preserving exploration component; and
8. the Guarniero--Johansen--Lee iterated auxiliary particle filter (iAPF),
   where its learned \(\psi\)-twisting is used only to construct an auxiliary
   proposal.

The proposal family is crossed with the following covariance sources: physical
\(P^Q\), per-particle UKF, KDM weighted-mixture, LEDH/GenUT, SGQF projected,
and a calibrated convex hybrid. SGQF is included as a proposal-design arm
because the repository already exposes \texttt{tf\_fixed\_sgqf\_cloud} and
\texttt{tf\_fixed\_sgqf\_filter}; the current implementation is an
eager/Python-branch approximate lane, so it is not automatically a
claim-bearing high-dimensional filter. Its sparse-grid point count, memory,
moment positive-definiteness, and dimension/level scaling are recorded before
any score comparison. The SGQF moments parameterise \(q_t\), while the
physical transition and observation factors remain in the numerator and the
actual SGQF-induced proposal density remains in the denominator.

Twisted PF and iAPF are crossed as none, fixed one-step \(\psi\), and
iteratively fitted \(\psi\). The fitted lookahead is produced on a disjoint
pilot/calibration partition and frozen before validation. A twisted or iAPF
arm must report the ancestor law, state proposal, correction factor, support,
and whether the proposal density is joint or marginalized. It is not allowed
to substitute an approximate future likelihood for the physical target.

For regular transitions, compare both ordinary PF weights and full model-
corrected mixture weights. This separates proposal quality from target changes.
Measure conditional ESS, weight variance, coverage of high-posterior regions,
and score error at equal particle count and equal wall-clock cost.

The minimum proposal test set is: (i) linear-Gaussian/Kalman normalizer and
score with an exact model-correction identity; (ii) a nonlinear
regular-transition model with mode or tail stress; and (iii) a high-dimensional
SGQF scaling ladder. For every row, test proposal normalisation, support,
finite weights, target-preserving \(\widehat Z_t\), and the call chain from the
claim-bearing endpoint to the declared proposal. Proposal normalizer MSE is
the primary criterion for a normalizer-accuracy claim; variance alone suffices
only under a proved common expectation. ESS, coefficient of variation, coverage,
and weight tails explain or screen proposals under predeclared roles. A claim
of better score computation requires lower held-out oracle score MSE at the
declared budget. Proposal metrics cannot substitute for that criterion.
A paired uncertainty interval is required before a row is called better;
otherwise it is classified as viable or descriptive-only.

The focused proposal tests must cover (a) equality of the sampled proposal
density and the denominator evaluated at the sample, (b) normalization of the
twisted/iAPF correction on a finite discrete auxiliary example, (c) SGQF
cloud-weight and projected-moment invariants, (d) positive-definiteness and
conditioning of every covariance arm, and (e) a linear-Gaussian identity in
which every proposal recovers the same normalizer in expectation. These tests
precede the factorial campaign; a failed identity or support test excludes the
row rather than being averaged into its score result.

The iAPF is a proposal-quality experiment, not a replacement score identity.
Guarniero, Johansen, and Lee define a family of positive future-data twisting
functions \(\psi\), show that the corresponding \(\psi\)-auxiliary particle
filter preserves the marginal-likelihood normalizer after the change-of-measure
correction, and iteratively approximate the zero-variance \(\psi^*\) sequence
from particle output (their Sections 2--3 and Algorithms 2--4). In this
program, an iAPF arm changes the auxiliary ancestor and state proposal laws;
it may use UKF or LEDH moments within that construction. It is not merely a
replacement covariance filter. The physical target remains fixed, while the
analytical recursion must be derived for the changed sampling law and include derivatives of the
new ancestor probabilities, proposal density, twisting correction, and all
parameter-dependent normalizing integrals. Pilot-fitted controls may be frozen,
but parameter dependence in their declared evaluation is not silently detached.
The denominator and support record must describe the new sampling law.
If ancestor sampling is discrete, an expectation-gradient claim needs the
appropriate sampling-law terms; differentiating fixed ancestor labels is not
a replacement for them.
The arm is admissible only when the required transition integrals, proposal
sampling, and correction are evaluable. An ambient Gaussian implementation is
not a solution for a singular DSGE transition; a disturbance- or
manifold-coordinate version would require a separate support and derivative
derivation.

This is distinct from Ionides iterated filtering. Ionides augments the model
with an artificial parameter random walk and decreases its perturbation scale
to obtain a simulation-only likelihood-derivative approximation. It is a
separate comparator for implicit models, not a UKF replacement and not an
iAPF proposal module.

Twisting is also not synonymous with replacing the UKF. The UKF supplies
local moment geometry for a proposal (and, in canonical LEDH, the per-particle
covariance lifecycle). A twisted filter changes ancestor and transition
sampling through a positive look-ahead function \(\psi\), with a correction
that preserves the declared likelihood normalizer. The two mechanisms can be
combined: UKF/LEDH can provide local proposals while twisting supplies an
auxiliary look-ahead weight. A route that replaces the UKF with a twisted
transition must therefore be registered as a new proposal route and must
prove that its sampling law, normalizing integral, correction, and support are
evaluable. It remains a proposal-variance experiment; it does not change the
analytical score identity.

Every UKF/KDM/SGQF provider must specify the full time-indexed filtering
lifecycle: incoming component means, covariances, and weights; prediction
through the model; conditioning on the new observation; outgoing moments;
and reassignment or transformation of persistent component state at resampling
and Contract E reset. Physical process covariance, proposal covariance, and
KDM bandwidth have distinct roles. Record the actual moments consumed by LEDH
and their analytical sensitivities, not only the provider's standalone output.

SGQF's signed integration weights must not be used as categorical probabilities.
If projected moments define a Gaussian or mixture proposal, validate its density
parameters separately and declare any protection that changes those moments.
Before its scaling study, require multistep Gaussian moment fixtures and an
executable provider-to-LEDH test across batch dimensions, dtype, graph/XLA,
device, and analytical sensitivities. An existing eager standalone filter
does not establish this integrated capability.

The OT study must include exact marginal-balance error, coupling entropy,
transport cost, sensitivity to Sinkhorn iterations, and whether the transported
cloud remains compatible with the density used in the importance correction.
OT is never treated as a resampling theorem by itself.

Record where the existing LEDH flow Jacobian enters the likelihood weights.
OT moment restoration is not a proof of preservation of the filtering law;
retain its full moments/weights/transport dependence in the total derivative.
No extra OT log determinant is inserted without a derivation for the actual
sampling law. Clarifying the existing flow determinant and deriving an OT
density correction are different tasks.

OT has five separate experimental roles that must not be conflated:

1. a coupling for comparing two particle systems with common random numbers;
2. a deterministic cloud rearrangement used only as a proposal heuristic;
3. a continuous resampling law whose density can be evaluated and corrected;
4. an OT cloud followed by an explicit, evaluable jitter kernel; and
5. a model-changing reset whose score is the derivative of a new finite scalar.

Roles 1 and 2 do not by themselves define a model-corrected proposal. Role 3
requires a density and support proof. Role 4 requires the actual jitter-mixture
density. A change-of-variables Jacobian is included only when the sampling
construction requires it; jitter around fixed transported centers does not
create a new OT determinant. Role 5 is allowed only under the finite-program
target label and cannot be compared to the exact score as though the reset were
measure-preserving.

The Sen--Thiéry--Jasra coupling belongs specifically to role 1. It couples two
particle-filter replicas while preserving their marginal algorithms, usually to
estimate a difference between parameters, models, or discretization levels.
For score work this can reduce the variance of a finite-difference or
multilevel difference when the replicas are positively correlated. It does not
reduce the variance of one LEDH score by itself, and averaging two positively
coupled copies is not a variance-reduction argument. The Jacob--Lindsten--Schön
coupled conditional-particle construction is a different route: its meeting
and Rhee--Glynn correction can debias a smoothing expectation under regular,
tractable-transition assumptions. Neither coupling construction supplies a
singular-DSGE score identity.

### UKF, KDM, and the covariance question

The UKF already provides a Gaussian moment approximation, but it does not make
the KDM mixture identity true and it does not supply a model-score correction
for an arbitrary nonlinear or singular transition. The covariance study must
therefore compare three distinct objects:

1. UKF covariance used only to construct a proposal;
2. KDM weighted covariance computed from the continuous mixture; and
3. a hybrid proposal that uses UKF moments for local propagation and KDM
   moments for mixture placement.

For each object, hold the physical model numerator and score estimator fixed,
then measure proposal coverage, importance-weight variability, and score error.
This identifies whether a covariance improvement helps because it places
particles better, because it changes the target measure, or because it merely
changes the finite-program derivative. No covariance comparison is interpreted
as evidence that UKF has been replaced as the filter unless the complete
filtering recursion and its target are explicitly changed and audited.

For covariance interpretation, keep the score code and physical numerator
fixed while swapping only the proposal moments. Record eigenvalue margins,
condition numbers, weight tails, and proposal-density agreement. A covariance
arm fails closed if it is not positive definite, if a caller-supplied
covariance does not match the generated density, or if it is used to claim
replacement of the UKF without a complete filtering derivation.

## Phase 3: score-estimator study on a fixed particle law

For each proposal, compare the following estimators using the same particle
cloud whenever possible:

1. analytical derivative of the executed finite value program;
2. unnormalised likelihood derivative divided by the estimated normalizer;
3. KDM IWSG gradient of a declared mixture expectation;
4. hybrid analytical/IWSG gradients with an explicit expectation target;
5. Rao--Blackwellised conditional-integration variants; and
6. exact-mean control-variate and independently calibrated combination variants.

Fisher/backward-pair, Nemeth, Fearnhead, PaRIS, and coupled
conditional-particle/Rhee--Glynn rows are deferred to the separate smoothing
program. An available exact conditional expectation can still serve as a
small identity reference. Target-specific identity tests precede quality
comparisons; gradients of different objectives cannot be pooled.

Report vector bias, covariance, MSE, componentwise error, and correlation of
errors across coupled estimators. Do not call a lower MSE an unbiasedness result.

Control variates require a known or separately corrected centering constant.
Estimate coefficients on calibration data only, freeze them, and test whether
the empirical variance reduction survives on untouched data. A fitted baseline
whose centering expectation is unknown is labelled a heuristic variance
reduction candidate, not an unbiased control variate.

Exact centering preserves the expectation of the baseline estimator; a biased
baseline remains biased. Independent random centering also requires a proved
unbiased center and its variance contribution in the comparison. If
\(\widehat D=\nabla\widehat Z\) for the same positive finite scalar, then
\(\widehat D/\widehat Z=\nabla\log\widehat Z\) pathwise. Comparing these two
expressions tests derivative consistency, not ratio bias; compare their
expectation with \(\nabla\log Z\) to measure that bias.

Concretely, for a scalar LEDH score estimate \(B\), a KDM/IWSG statistic \(H\),
and its mean \(\mu_H=\mathbb E[H]\) under the actual joint experiment,
\(C_\beta=B-\beta(H-\mu_H)\) has \(\mathbb E[C_\beta]=\mathbb E[B]\)
for a frozen coefficient \(\beta\). Its variance is
\(\operatorname{Var}(B)+\beta^2\operatorname{Var}(H)
-2\beta\operatorname{Cov}(B,H)\). An independent unbiased estimated center
adds \(\beta^2\operatorname{Var}(\widehat\mu_H)\). A center derived conditional
on the incoming cloud must be valid for that cloud; repeat over independent
clouds to evaluate total error. The KDM mean identity, the centering mechanism,
and the correlation with LEDH error are three separate requirements.
If coefficient fitting reuses the random center, prove the required conditional
centering identity or use an independent center; unconditional unbiasedness of
the center alone does not justify multiplying it by a correlated coefficient.

### Combining two imperfect estimators

If (A) and (B) estimate the same score (S), a combined estimator

\[
 C_\alpha=\alpha A+(1-\alpha)B
\]

has error

\[
 \mathbb E[C_\alpha-S]
 =\alpha b_A+(1-\alpha)b_B,
\]

and, for a scalar score component, variance

\[
 \operatorname{Var}(C_\alpha)=
 \alpha^2V_A+(1-\alpha)^2V_B
 +2\alpha(1-\alpha)\operatorname{Cov}(A,B).
\]

The program must first establish that (A) and (B) target the same score.
Then estimate \(\alpha\) on independent calibration replications, using a
predeclared loss or an exact control-variate relation, freeze it, and evaluate
on untouched replications. With an oracle, the MSE-optimal scalar coefficient
can be estimated directly. Without an oracle, a variance-minimising coefficient
does not generally minimise MSE because the unknown bias remains. A combination
of two biased estimators is therefore a candidate for variance reduction, not
an automatic debiasing method. The study must report the two component biases,
their covariance, the selected coefficient, and the combined estimator's
untouched error.

For vector scores the cross term is
\(\alpha(1-\alpha)[\operatorname{Cov}(A,B)+\operatorname{Cov}(B,A)]\).
With a fixed positive-definite error metric \(W\), put
\(\Delta=A-B\). The oracle MSE quadratic has

\[
 \alpha^*=-\frac{\mathbb E[\Delta^\mathsf T W(B-S)]}
                   {\mathbb E[\Delta^\mathsf T W\Delta]},
\]

when the denominator is positive. If it is zero, the estimators agree in this
metric almost surely. A convex-combination restriction projects this value
onto \([0,1]\). Estimate the coefficient on calibration data, check the
denominator and oracle uncertainty, freeze it, and evaluate it on untouched
data. Without an oracle or a derived error identity, minimising variance is
a different objective and generally does not identify this MSE optimum.

The active comparison should include:

1. canonical LEDH and an eligible model-corrected KDM score estimate, each
   explicitly targeting the same model score;
2. analytical finite-program and coupled FD estimates of that same finite
   program, kept separate from model-score accuracy claims;
3. LEDH with a KDM/IWSG statistic whose centering under the actual joint
   sampling law is known or independently estimated without bias; and
4. IWSG plus an analytical or diagnostic pathwise gradient only when both have
   the same declared expectation target.

An unnormalised derivative cannot be added directly to a score without a
derived scaling and centering relation. An oracle-trained bias prediction from
Phase 4B is a heuristic residual-correction arm and must pass held-out
model-score MSE; it is not an exact-mean control variate.

If the component targets differ, the linear combination is a new target and
must be analysed as such. It cannot be described as a better estimate of the
original score merely because its numerical variance is smaller.

## Phase 4: normalisation and long-horizon study

This phase isolates the random-ratio problem.

1. Compare \(\widehat Z_N\), \(\widehat D_N\), and
   \(\widehat D_N/\widehat Z_N\) separately.
2. Vary particle count over a logarithmic ladder and retain paired random
   numbers across methods.
3. Vary horizon while holding model and dimension fixed, with a new tuning
   scope and frozen controls for each horizon.
4. Measure denominator coefficient of variation, reciprocal-weight tails,
   score bias, and variance.
5. Compare eligible analytical/model-corrected score estimators with ratio
   estimators at equal compute, without launching deferred smoothing rows.
6. Test coupled finite differences and consistency ladders after their own
   prerequisites; randomized debiasing remains outside the active scope.

The conclusion distinguishes established derivative or measure errors from
evidence about Monte Carlo and random-normalisation error. If attribution is
unresolved, say so. This limits causal explanation; it does not by itself
invalidate a correctly measured held-out MSE comparison.

## Phase 4B: approximate consistency diagnostics

When an exact score oracle is unavailable, the program may study diagnostics
that are plausibly correlated with bias, but it must keep them separate from
the score criterion. Candidate diagnostics include:

- score estimates at \(N,2N,4N\) under coupled random numbers;
- Richardson-style slopes in \(N\) or bandwidth, when a fitted expansion is
  supported by the data;
- agreement between eligible analytical, IWSG, and finite-difference estimates
  after checking that the claimed targets agree;
- conditional mixture/normalisation identities; backward/ancestor-pair
  residuals are deferred with smoothing;
- likelihood normalisation residuals and denominator stability;
- score directional identities under parameter perturbations; and
- replicate-to-replicate correlation between each diagnostic and measured score
  error on oracle-bearing models.

The diagnostic study must be calibrated on oracle-bearing models first. A
diagnostic can then be used as an explanatory warning or a predeclared
promotion veto. A continuation veto requires evidence of invalid execution or
a separately justified stop condition, not merely correlation with error.
The diagnostic cannot prove low bias on an oracle-free model merely because
its correlation is high elsewhere. The program must report the calibration
model, held-out model, correlation
uncertainty, and the strongest counterexample.

Central finite differences of a particle log-likelihood are useful for this
diagnostic, but they are not automatically a bias estimate. Writing
\(\ell(\theta)=\log Z(\theta)\) and
\(b_N(\theta)=\mathbb E[\log \widehat Z_N(\theta)]-\ell(\theta)\),

\[
\mathbb E[\widehat s_{\varepsilon,N}]-\ell'(\theta)
=
\underbrace{\frac{\ell(\theta+\varepsilon)-\ell(\theta-\varepsilon)}
 {2\varepsilon}-\ell'(\theta)}_{O(\varepsilon^2)\text{ under smoothness}}
+
\frac{b_N(\theta+\varepsilon)-b_N(\theta-\varepsilon)}{2\varepsilon}.
\]

Common random numbers or Sen-style coupled resampling can reduce the variance of
the difference, but they do not remove the log-normalisation bias, finite-
particle bias, or finite-difference truncation bias. If the same fixed random
design contains discrete resampling, the \(\varepsilon\to0\) limit may instead
differentiate a discontinuous finite program. Richardson extrapolation is
allowed only after a relevant truncation/error expansion has been justified
and tested on the quantity being extrapolated. A noisy MSE slope is not that
test; Phase 4C separates deterministic truncation from stochastic error.

A symmetric local-polynomial fit is a useful multi-point version of this
diagnostic, but the number of points is not itself an accuracy guarantee. For
points \(\theta+k h\), the derivative is the slope of the fitted polynomial at
zero. With a symmetric stencil, a linear fit has
\[
 D_h\ell=s+\frac{\ell^{(3)}(\theta)}{6}
   \frac{\sum_k k^4}{\sum_k k^2}h^2+O(h^4),
\]
for deterministic \(\ell\) with bounded fifth derivative locally. A quadratic
fit has the same leading order because the quadratic term is
even and is orthogonal to the slope on the symmetric design. To cancel the
third-derivative term one may fit a cubic or use the five-point fourth-order
stencil. These have different weights and error constants on different grids;
both give \(O(h^4)\) truncation with five bounded continuous derivatives locally.
More points can reduce observation-noise variance, but
they can also enlarge the stencil and increase the truncation constant or
amplify correlated particle noise. The implementation must therefore choose a
fixed symmetric grid, fit degree and span separately, and report both the
deterministic order in \(h\) separately from stochastic bias and variance.
The fitted derivative still
inherits the finite-particle/log-normalisation bias of the values being
regressed; regression does not turn it into an oracle.
An unconstrained normal draw for (h) is a poor default because draws near zero
divide particle noise by a small number; use a deterministic positive ladder
(or a distribution truncated away from zero), and use negative points as the
symmetric stencil rather than as independent random magnitudes.

A finite-difference statistic can be an exactly centered control variate when
its own expectation \(\mu_D\) is known (or independently estimated without
bias). An exact model score is not automatically the expectation of a finite
stencil applied to random log-likelihood estimates. When the center is valid, use
\(S_{\mathrm{cv}}=S-\beta(D-\mu_D)\), estimate \(\beta\) on calibration
replicates, and freeze it. If \(\mu_D\) is unknown and replaced by the
finite-difference estimate of the target score, the construction is a
calibrated heuristic or residual estimator, not an unbiased control variate.

## Phase 4C: fixed symmetric directional finite-difference ladder

This phase tests whether a coupled finite-difference calculation is a useful
directional score estimate or calibration diagnostic. It has two distinct
questions: does the deterministic stencil implement its stated approximation,
and how accurately does that stencil, applied to random particle log-likelihoods,
estimate the model score? Passing the first test does not answer the second.
Keep the model, proposal, target label, particle count, horizon, and coupling
fixed within a comparison; selected tuning controls stay frozen at all its
perturbed parameter points.

### 4C.1 Stencils, smoothness, and deterministic mechanics

Let \(f_v(t)=\ell(\theta+tv)\), where \(\ell=\log Z\) and \(v\) is a
declared unit direction in the chosen parameter coordinates. Test

\[
D_2(h;v)=\frac{\ell(\theta+hv)-\ell(\theta-hv)}{2h},
\]

\[
D_4(h;v)=\frac{\ell(\theta-2hv)-8\ell(\theta-hv)+8\ell(\theta+hv)-\ell(\theta+2hv)}{12h},
\]

and an unweighted eleven-point cubic least-squares fit at \(j=-5,\ldots,5\).
Write its derivative as \(D_{\rm cub}(h;v)=h^{-1}\sum_j c_j f_v(jh)\).
With \(S_r=\sum_{j=-5}^5 j^r\), its weights are

\[
 c_j=\frac{S_6 j-S_4 j^3}{S_2S_6-S_4^2}.
\]

These follow by solving the odd block of the cubic normal equations.
A fit against dimensionless \(j\) returns a linear coefficient that must be
divided by \(h\); a fit against \(jh\) already returns the derivative in
parameter units. Use QR or SVD for fitted coefficients, not explicit
normal-equation inversion. Centered even polynomial terms do not change the
unweighted slope. Linear/quadratic symmetric fits remain second order;
eleven points alone do not produce fourth order.

A sufficient local assumption is \(f_v\in C^3\) with bounded third derivative
for \(D_2-f'_v(0)=O(h^2)\), and \(f_v\in C^5\) with bounded fifth derivative
for the five-point and cubic fourth-order statements. Four continuous
derivatives alone are insufficient. Taylor substitution gives

\[
 D_4-f'_v(0)=-\frac{f_v^{(5)}(0)}{30}h^4+o(h^4),\qquad
 D_{\rm cub}-f'_v(0)=-\frac{143}{90}f_v^{(5)}(0)h^4+o(h^4).
\]

The eleven-point fit spans \(5h\) in each direction; the five-point formula
spans \(2h\). Compare node spacing and total span separately rather than
interpreting more points as an automatic accuracy improvement.

Mechanics tests use exact polynomials and noise-free functions with known
derivatives, followed by deterministic Kalman log-likelihoods. Check stencil
moments, fitted-slope units, and a function with nonzero leading truncation
coefficient. Measure order only in a range above the roundoff/evaluation-error
floor. Exact cancellation on a polynomial is a passing algebraic check, not a
failed slope test. Any numerical slope tolerance must follow from the fixture's
error budget; no universal tolerance or condition-number cutoff is assumed.

### 4C.2 Particle bias, covariance, and the actual MSE

For a fixed dataset and parameter, let
\(L_j=\log\widehat Z_N(\theta+jhv;\xi_j)\), with the correct marginal
particle law at every point and a declared coupling across the \(\xi_j\).
Assume positive likelihood estimates, finite second moments of \(L_j\), and
write \(b_N(\vartheta)=\mathbb E[L_N(\vartheta)]-\ell(\vartheta)\).
For any of the fixed stencils, set

\[
 \widehat D_h=\frac{1}{h}\sum_j c_j L_j,\quad
 T_h=\frac{1}{h}\sum_j c_j\ell(\theta+jhv)-s_v,\quad
 B_{N,h}=\frac{1}{h}\sum_j c_j b_N(\theta+jhv),
 \qquad s_v=v^\mathsf T S(\theta).
\]

Taking expectations and then using the variance of a linear combination yields

\[
 \mathbb E[\widehat D_h]-s_v=T_h+B_{N,h},\qquad
 \operatorname{Var}(\widehat D_h)=\frac{c^\mathsf T\Sigma(h)c}{h^2},
\]
\[
 \operatorname{MSE}(\widehat D_h)
 =\bigl(T_h+B_{N,h}\bigr)^2+
   \frac{c^\mathsf T\Sigma(h)c}{h^2},
 \qquad \Sigma_{jk}(h)=\operatorname{Cov}(L_j,L_k).
\]

The squared bias includes \(2T_hB_{N,h}\); truncation and finite-particle bias
can cancel. Summing their squares separately is wrong. On oracle-bearing
models, deterministic evaluations give \(T_h\); repeated particle evaluations
estimate total bias, \(B_{N,h}\), and covariance with uncertainty. Away from
such oracles these terms generally cannot be identified separately.

Estimate MSE directly from replicated squared oracle errors. A squared
replicate-mean error is not itself an unbiased estimate of squared bias;
retain its finite-replication uncertainty or apply a justified correction.

There is no universal stochastic slope or U-shaped MSE curve. If the same
additive noise enters all \(L_j\), it cancels because \(\sum_jc_j=0\). With a
mean-square differentiable coupled process, the numerator variance can be
\(O(h^2)\), so division by \(h^2\) leaves bounded variance. With independent
noise of nonvanishing variance it instead grows as \(h^{-2}\). Discrete
resampling and other couplings can behave differently. Record
\(c^\mathsf T\Sigma(h)c\), total bias, variance, and MSE; report a plateau,
boundary minimum, or irregular curve if that is observed. No noisy error
slope is required to match deterministic stencil order.

### 4C.3 Calibration, held-out comparison, and full-score reconstruction

Use a finite deterministic positive ladder, such as
\(\{h_0,h_0/2,h_0/4,h_0/8\}\), whose span, lower bound, and length are chosen
for the parameter scale, dtype, model domain, and evaluation accuracy.
Every perturbed point must be valid and distinguishable in the execution
dtype. If a log-likelihood evaluation has error bounded by \(\delta_j\), the
corresponding derivative error is bounded by
\(h^{-1}\sum_j |c_j|\delta_j\); use such bounds or measured reference errors
to justify the numerical range. No fixed universal range of \(h\) is imposed.
An unrestricted normal draw for \(h\) is excluded because it can approach zero.
A distribution truncated away from zero is an optional later comparison.

For constrained or differently scaled parameters, declare the coordinates
before choosing directions and steps. If \(\theta=g(\eta)\), the computed
likelihood score in these coordinates is
\(S_\eta=(\partial\theta/\partial\eta)^\mathsf T S_\theta\); reconstruct in
those coordinates and transform with the appropriate nonsingular Jacobian
when an original-coordinate score is required. This is likelihood
reparameterisation, not a posterior-density Jacobian term.

Calibrate each stencil and coupling on independent particle replications of
Gaussian oracle models, then nonlinear regular oracle models. Preserve each
replication's entire vector of perturbed values to estimate the full
within-stencil covariance, including correlations across directions and
stencils. Common random numbers must preserve each perturbed filter's marginal
algorithm. Independent noise is a useful covariance reference arm.

Predeclare the directional or scaled vector MSE, cost accounting, selection
rule, and uncertainty method. Use disjoint calibration, validation/selection,
and final claim data and streams; freeze the selected stencil, \(h\), directions,
and coupling before claims. The selected calibration minimum is optimistic.
Do not require validation MSE to fall inside a calibration confidence interval.
Use paired differences in squared oracle error on the untouched set, with
dataset-level or hierarchical uncertainty for particle replicates nested in
datasets. Report all predeclared comparisons or account for multiple selection;
do not select a winner on the claim set.

For \(m\) directions in \(d\) coordinates, define
\(V=[v_1,\ldots,v_m]\in\mathbb R^{d\times m}\) and let
\(b\in\mathbb R^m\) contain their derivative estimates. Then
\(b\approx V^\mathsf T s\). If \(V\) has full row rank, the least-squares
solution is

\[
 \widehat s=\arg\min_s\|V^\mathsf T s-b\|_2^2
           =(VV^\mathsf T)^{-1}Vb.
\]

Solve by QR/SVD without forming the displayed inverse. Coordinate directions
provide a simple reference; fewer than \(d\) independent directions identify
only a projection. Report singular values, reconstruction residual, and
propagated error: for \(A=(VV^\mathsf T)^{-1}V\), bias is \(A\,\operatorname{Bias}(b)\)
and covariance is \(A\,\operatorname{Cov}(b)A^\mathsf T\). Rank failure or error
amplification beyond the declared dtype/accuracy budget vetoes a full-score
claim. A small overdetermined residual alone does not establish accuracy.
Any generalized least-squares extension must freeze its independently
estimated weighting rule and account for its estimation uncertainty.

Compare equal-particle and equal-compute performance separately, charging all
nonzero-weight function evaluations, directions, pilot fitting, and replication.
Report both total campaign cost and amortized per-evaluation cost. MSE times
runtime is not a universal objective: averaging \(R\) replicas gives
bias squared plus variance divided by \(R\), leaving the bias unchanged.

### 4C.4 Pass criteria and saved evidence

Deterministic algebra/order failures, invalid support, non-finite values,
incorrect coupling marginals, missing sensitivities in a claimed derivative,
or insufficient reconstruction rank are repair triggers and relevant validity
vetoes. Noisy slopes and the absence of a U-curve are explanatory observations,
not vetoes. A valid candidate that fails held-out MSE improvement is rejected
for promotion at that scope; continue any planned repair within budget.
Without a model-score oracle or a certified reference-error bound, this is
a consistency/parity diagnostic, not a measured bias correction.

Focused tests cover exact weights and moments, cubic-fit units, invalid steps,
finite-precision point collapse, coupling-stream reuse and marginal laws,
rectangular direction reconstruction, and error propagation. Campaign artifacts
save perturbed parameter points, weights, coordinate transform, coupling,
deterministic errors, replicate vectors, covariance, MSE uncertainty,
selection partitions, reconstruction diagnostics, costs, and the criterion/
veto/explanatory role of each diagnostic. Store them under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`.

## Phase 5 (deferred): regular-transition smoothing

The ordinary Younis forward/backward smoothing study belongs to the separate
regular-transition program. KDM filtering and IWSG remain active in Phases 2--4;
this deferred phase must not be a prerequisite for those experiments and no
smoothing row is generated by this master.

The external program should, in its own plan, implement the physical-transition
mixture before positive bandwidth/OT, add the auxiliary backward density,
compute adjacent-state expectations, and retune every method in its own scope.
It must keep proposal-placement gains separate from a score identity. Positive-
bandwidth KDM is a changed filtering approximation unless a limiting argument
and model correction are supplied.

## Phase 6 (deferred): degenerate-transition support program

This is a derivation program owned by the separate DSGE/support workstream,
before it is an implementation program. This master records applicability but
does not schedule these rows or transfer regular-track results to them.

1. Inspect the manifold/constrained-state particle-filter literature and
   classify each method by its state-space reference measure and target. A
   fixed-manifold intrinsic smoother is admissible only as a special-case
   control: it must supply (or permit deriving) a transition density and
   backward kernel with respect to a common, parameter-independent manifold
   measure. Structural DSGE equilibrium and policy manifolds are generally
   parameter-dependent, so this branch cannot be promoted to the DSGE route
   without a separate moving-support derivation.
2. Write the DSGE transition in explicit ancestor/innovation coordinates.
3. Identify the base measure on which the conditional transition law is
   represented and determine whether its support depends on parameters.
4. Derive the complete-data score including the deterministic map and any
   Jacobian or constraint terms.
5. Determine whether distinct ancestors induce mutually singular supports and
   whether backward ancestor probabilities collapse.
6. Prove or disprove a valid marginalisation identity for the proposed
   coordinate representation.
7. Build a tiny exact fixture with symbolic or numerical integration.
8. Only after those checks, implement an estimator and compare it with an
   independent support-aware oracle.

Full-rank Gaussian kernels, ordinary ambient-space KDM densities, and regular-
transition PaRIS formulas may be used as stress tests, but cannot establish
validity for DSGE. If the support derivation fails, the scientific conclusion
is that the route is unresolved, not that a tuned approximation solves it.

The manifold literature supplies useful state-space machinery, including
geodesic or tangent-space sampling and implicit-constraint proposals. It does
not, by itself, supply the observed-data parameter score. A future intrinsic
Poyiadjis/PaRIS-style estimator would be a new theorem on a fixed manifold;
for DSGE, the required object is instead a moving-support or
disturbance-coordinate derivation. Manifold score-matching papers, which
estimate $\nabla_x\log p(x)$, do not answer this parameter-score question.

## Phase 7: heuristic dominance and practical comparison

For each salient regime, construct and evaluate cheap adversaries:

- linear-Gaussian: bootstrap and locally adapted PF, with Kalman as the exact
  error oracle;
- weakly nonlinear: EKF and UKF score;
- highly nonlinear: bootstrap PF and locally adapted PF;
- long horizon: larger-particle bootstrap and a fixed-coupling central
  finite-difference reference of its likelihood;
- multimodal: mixture proposal without KDM score correction;
- no degenerate row is executed here; imported support-aware results require
  an applicability and provenance check.

A complex method losing to a heuristic in a salient regime is a promotion veto,
even if it beats another complex method on average. The heuristic table is not
a tuning target.

Construct three to seven concrete cheap competitors for each evaluated regime
from these families, with a rationale and the same target/budget accounting.
Oracle distance is the primary error measure; an exact oracle is not itself
a stochastic method to beat. Use paired uncertainty for loss comparisons.
A UKF evaluated on a linear-Gaussian model coincides with the Kalman oracle
when its moment rule is exact for affine maps. Such a row verifies correctness
and calibrates approximation error; it cannot establish particle-score
superiority over the exact Gaussian solution. Failure to improve that row
does not cancel the planned nonlinear comparison. An invalid identity or
implementation, in contrast, triggers repair before that method continues.
A statistically supported loss to a heuristic vetoes promotion; inconclusive
differences block a superiority claim until the planned precision is reached
or the budget ends. Do not tune against the final heuristic evaluation.

## Phase 8: final replication and compute fairness

Tuning is already a prerequisite of each claim row under Phase 0. Each route
receives a separate tuning scope binding model, target, horizon,
particle count, dimensions, dtype, backend, chunk policy, and all tunable
controls. Calibration, validation, and untouched claim data are disjoint.

The final comparison uses at least:

1. multiple independent observation datasets;
2. multiple particle randomisations per dataset;
3. paired seeds across methods where coupling is meaningful;
4. a particle-count and an equal-compute comparison;
5. uncertainty intervals for paired score error; and
6. preserved failed candidates and repair attempts.

One seed or one short chain is descriptive only. A candidate may be called
viable after passing hard screens, but a ranking requires uncertainty evidence.

## Phase 9: implementation and production audit

This is the terminal audit of implementations built in Phases 0A--0G and
repaired at phase boundaries. It is not the first implementation phase. If
the audit finds missing functionality or invalid evidence, return to its
implementation producer, repair it, and repeat the affected evaluation before
closing the program.

Before any default or HMC-facing recommendation, audit:

1. source-faithfulness to Younis and Sudderth where claimed;
2. support class and reference measure;
3. complete call chain from consumer to implementation;
4. analytical derivative route, with autodiff restricted to parity diagnostics;
5. TensorFlow/XLA, dtype, GPU memory-growth, and chunk-policy compliance;
6. tuning-artifact scope identity;
7. reproducibility manifest; and
8. rendered scientific documentation.

The canonical LEDH route remains governed by the repository's Contract E and
per-scope tuning policies. A new estimator can be scientifically promising
without replacing the canonical route until its own gates pass.

## Phase exit gates and dependency order

The phases are sequential where a later phase depends on a mathematical or
implementation identity, and parallel only where the dependency is explicit.
Every implementation phase ends with its stated executable checks and the
repair/refresh procedure. Completion of a plan paragraph is not completion
of the implementation it specifies.

| Task | Implementation producer and minimum prerequisite |
|---|---|
| Phase 0A specification/source reconciliation | Phase 0 scientific requirements and current checkout; developer tools only. |
| Phase 0B coordinator | 0A target/return contract; implement its own state, validation, and repair/resume tests. |
| Phase 0C baseline, oracle, tuning, and report integration | 0A--0B and baseline-specific repairs performed here. |
| Phase 0D KDM/IWSG/combinations | The applicable 0C baseline; repair required shared-executor capabilities here. |
| Phase 0E covariance/SGQF/twisting | The applicable 0C baseline; KDM-provider rows additionally use 0D law tests. |
| Phase 0F deterministic FD | 0A target specification; implement exact fixtures here, independently of KDM or the coordinator. |
| Phase 0F stochastic FD/normalization/consistency implementation | 0C value endpoints and only the implemented 0D/0E estimators actually used. |
| Phase 0G model/scale/report coverage | Applicable 0C--0F capabilities for each new study slice. |
| Phase 1 admission | Target-specific tests produced during 0C--0G at the actual consumer revision; failed tests return to their implementation producer. |
| Phase 2 proposal quality | 0C or 0D/0E proposal implementation; 0G when coverage expands; Gates A/B, actual correction law, and scope tuning. |
| Phase 3 estimator comparison | 0D or other applicable verified estimator implementation; Gates A/B, fixed-cloud law, and Gate C for each new proposal used. |
| Phase 4 normalisation study | 0F implementation and Phase 3 evidence for the verified value/derivative pairs compared. |
| Phase 4B consistency calibration | 0F implementation, oracle, and eligible estimators; deterministic FD checks when used, not completion of the whole normalization or FD campaign. |
| Phase 4C deterministic mechanics | Execute the exact scalar/direction tests implemented in 0F. |
| Phase 4C stochastic/full-score study | 0F implementation, eligible value/proposal law, deterministic mechanics, valid directions/coupling, and scope tuning. |
| Phases 7--9 comparisons, replication, and final audit | 0G coverage/report implementation, applicable mechanism gates, and frozen scopes for the candidates compared. |

These are dependencies for each row, not a requirement that every candidate
finish one numbered phase before any candidate enters another. A KDM wiring
block does not stop oracle FD mechanics or an already verified UKF comparison.
Neither Phase 4B nor Phase 4C requires the separately owned Phases 5 or 6.

1. **Gate A, after the applicable Phase 0C/0G implementation:** the model and
   regime being admitted have Phase 0's target record, executable baseline
   ladder, tuning service/scope, and checked oracle status. No score-quality
   comparison proceeds without a valid comparator or an explicitly diagnostic
   label. Later models need their own records and checks; they do not block
   the first verified Gaussian study.
2. **Gate B, after Phase 1:** each target-specific identity, support,
   positivity, sensitivity, and consumer-to-provider call-chain test passes;
   an IWSG identity is not replaced by finite-program parity. Failures return
   to implementation repair.
3. **Gate C, after Phase 2:** proposal comparisons show the physical numerator,
   proposal denominator, and resampling representation separately. Missing
   correction blocks a model-corrected claim. A separately defined, valid
   finite scalar may continue as a changed-target approximation; an invalid
   density or implementation cannot be repaired by relabeling it.
4. **Gate D, after Phase 3:** estimator-level conditional error is available on
   common clouds and total error over independent clouds; Phase 4 uses these
   verified estimators for normalisation comparisons.
5. **Gate E, after Phase 4:** \(\widehat Z,\widehat D,\widehat D/\widehat Z\)
   and their uncertainty are reported; derivative, measure, Monte Carlo, and
   ratio explanations are classified where evidence permits. Unresolved
   attribution limits causal language but does not erase a valid held-out MSE.
6. **Gate F, after Phase 4B:** consistency diagnostics have oracle calibration,
   held-out correlation uncertainty, and explicit explanatory/veto roles.
   Correlation is never a proof of low bias.
7. **Gate G, after Phase 4C:** deterministic mechanics, scale-aware stochastic
   calibration, coupling marginals, and full-rank reconstruction pass. A noisy
   MSE slope or absent U-curve is not a veto.
8. **Gate H, after Phase 7--9:** only candidates with statistical uncertainty,
   complete manifests, scope-specific tuning, and implementation audit may be
   considered for default or HMC-facing status.

Phases 0, 0A--0G, 1--4, 4B, and 4C form the active implementation and research
foundation. Phase 7 evaluates active regular
candidates against conditional heuristics. Phases 5 and 6 are deferred
external branches and cannot supply active claim evidence. Phases 8 and 9 are
terminal validation and audit, not sources of new tuning data.

### Between-phase repair and refresh

These steps execute automatically under the owner's continuing authorization.
After step 5, proceed directly to the next eligible phase. A terminal chat
response is not a required phase gate. If a dependency is blocked, execute
independent work and applicable repairs before asking the owner for anything.
Read-only source recovery and small falsification checks are part of repair,
not separate projects requiring permission. Report progress while executing.

Every phase, including a failed or terminal phase, ends with an explicit
repair pass and a refresh of its successors. A scientific gate passing does
not waive unresolved defects in the computation that produced its evidence.
An issue is repairable when there is a concrete correction or discriminating
test within the current scientific scope and remaining resources. Record and
attempt those repairs; a promise that unspecified further tuning might help
is not a completed repair plan.

1. Reconcile planned, completed, failed, blocked, and omitted rows. State
   whether the finding invalidates the harness, mathematical target,
   implementation, numerical result, or evidence, or only rejects a candidate.
2. For each repairable issue, record its affected rows/callers, cause or
   smallest diagnostic, proposed correction, regression checks, scope and
   partition consequences, and remaining compute/attempt budget.
3. Perform the feasible corrections and applicable candidate repairs. Run
   focused regressions and affected consumer/integration checks, invalidate
   dependent evidence when necessary, and recompute the phase decision. An
   edit without verification does not close the issue. Do not relax the
   target, baseline, or scientific criterion to obtain a pass.
4. Give every unresolved issue a disposition. Missing validity prerequisites
   block their dependent rows; independent verified work can continue. An
   inconclusive comparison remains unresolved, and a valid negative result
   can close its experiment without making the candidate successful. Mark a
   phase partial when its required work remains unfinished.
5. Refresh each next-phase plan using the actual findings: question, eligible
   rows and repairs, prerequisites, baselines, evidence roles, controls and
   tuning scopes, data/stream partitions, tests, exact commands/environment,
   budget, and stop conditions. Record a brief skeptical audit and link the
   revised plan before the corresponding dependent work begins. For the last
   phase, refresh the final disposition and any justified follow-up instead.

Repairs informed by claim outcomes create a new candidate version with fresh
calibration, validation, and claim partitions. An infrastructure retry that
preserves the numerical program may reuse its streams in a fresh attempt
directory; it cannot select a favorable result or overwrite the failure.
Source or scope changes require checking which previous evidence and tuning
remain applicable. Unknown control-variate centers or proposal normalizers
cannot be fixed by changing a status label.

The closeout note and machine-readable issue dispositions are sufficient;
routine repairs and next-phase refreshes do not require a new external review
or renewed permission within an authorized campaign. Phase 0B implements
tests for failure followed by repair; a repair that invalidates a formerly
passing dependency; changed horizon/source with stale tuning; a shared-code
failure affecting two methods; a blocked method beside a runnable method;
exhausted repair budget; and a terminal repair requiring fresh evaluation.
The scheduler refuses dependent launches without the applicable disposition
and current next-phase plan. Phase 0C verifies this behavior with real
endpoints, and Phase 0G extends it to the final reports.

Use the phase note plus a machine-readable closeout record containing:

```text
phase_id, phase_plan_version, source_revision, source_changes
planned_rows, completed_rows, blocked_rows, omitted_rows_with_reasons
engineering_status, numerical_status, scientific_decision, inference_status
issues: [issue_id, affected_rows, classification, root_cause, repair,
         regression_evidence, scope_effect, disposition]
attempts, consumed_cpu_hours, consumed_gpu_hours, remaining_budget
invalidated_evidence, current_tuning_scopes, partition_usage
next_phase_plan, next_phase_plan_version, refreshed_dependencies
```

For developer phases before the coordinator exists, record the same
information in the phase note; Phase 0B adds the machine-readable support.
The process does not require its own unbuilt machinery to get started.

## Required artifacts

Each phase produces:

- a plan and skeptical pre-mortem;
- a model/target/measure ledger;
- an oracle or oracle-status report;
- executable identity and call-chain tests;
- a run manifest containing commit, command, environment, hardware, seeds,
  wall time, tuning scope, and artifact paths;
- raw structured results and uncertainty calculations;
- a decision table separating hard vetoes, descriptive evidence, statistical
  ranking, and default readiness;
- a repair record with verified fixes, unresolved issue dispositions, affected
  evidence, and remaining budget;
- a refreshed next-phase plan (or terminal disposition), linked to the actual
  results and checked prerequisites; and
- a red-team note stating the strongest alternative explanation, the result
  that would overturn the conclusion, and the weakest evidence.

The master result must include a matrix with one row per method, model, horizon,
particle count, and score kind. It must never combine KDM expectation gradients,
finite-program derivatives, and marginal scores in one unlabeled leaderboard.

## Decision rules

The program may promote a candidate only when:

1. the claimed target matches the computed quantity by derivation; a declared
   approximation retains its distinct target and reports its measured or
   bounded error against the model score;
2. all support, positivity, normalization, and derivative vetoes pass;
3. the candidate clears the conditional heuristic-dominance table;
4. paired uncertainty supports the stated comparison;
5. the result survives an untouched retest under its own tuning scope; and
6. no conclusion crosses from the regular-transition track to the degenerate
   track without a new support derivation.

If a candidate fails, classify the failure as implementation, tuning,
diagnostic, finite-normalisation, support, or evidence failure. Rejecting one
candidate does not reject the research direction; rejecting the direction
requires evidence that the repairable alternatives and their target identities
have also failed.

## First executable tranche

Executing this master starts with implementation. Follow this order within
this document:

1. Read Phase 0's target/oracle requirements and perform Phase 0A's source,
   capability, and manuscript reconciliation. Pin the existing environment
   and the initial validation commands.
2. Execute Phase 0B: implement the registries, CLI, dependency checks, result
   records, budget accounting, and repair/resume behavior. Develop Phase 0F's
   exact three/five/eleven-point and direction fixtures independently when
   their Phase 0A specifications are available.
3. Execute Phase 0C: implement and verify the independent Gaussian oracle,
   parameter-dependent initialization, actual bootstrap/UKF/adapted and
   canonical LEDH endpoints, tuning issuer/consumer path, and reports. Repair
   the applicable shared canonical code instead of adding a reduced lane.
4. Perform the repair/refresh step after each phase. Complete the real
   Gaussian baseline smoke, interruption/resume check, and scoped
   tuning/claim-plumbing check with explicit no-ranking status. This is the
   first executable software milestone produced by the master.
5. Execute the applicable 0D--0F branches for KDM/IWSG, centered control
   variates, biased combinations, covariance/SGQF providers, twisting, and
   stochastic FD. Finish Phase 1 admission for each method using the tests
   produced by its implementation phase. A blocked method does not stop
   independent verified branches.
6. Use Phase 0G to implement each additional study's model, configuration,
   reference, and reporting requirements before Phases 2--4C and 7 run it.
   Tune on disjoint partitions, use common clouds for conditional comparisons
   and independent clouds for total error, and preserve every replica's
   marginal law in paired full-filter comparisons.
7. Run the scientific studies under their refreshed finite plans, repair and
   update after each phase, then perform Phases 8--9 replication and final
   audit. Any outcome-informed repair uses fresh evaluation partitions.

No separate execution plan is needed before these tasks begin. The deferred
DSGE/support and smoothing programs remain separate assignments; this
tranche neither launches them nor inherits an unsupported numerical result
from them.

## Literature coverage addendum (2026-09-14)

This is a literature applicability inventory, not an expansion of the active
execution scope. Smoothing, iterated filtering, continuous-likelihood methods,
multilevel debiasing, and DSGE derivations remain deferred; their earlier
implementation requests are superseded by the active-scope amendment.

The literature gap audit in
`docs/plans/artifacts/particle-filter-score-literature-gap-audit-20260914/`
identified several named methods that must be made explicit before interpreting
the inventory as a complete score-computation study. One direct
regular-model comparator is the Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and
observed-information estimator. The existing `KDM` row is only a method label;
it does not yet specify Nemeth's kernel recursion, bandwidth, conditional
integration, or its regular-model asymptotic assumptions.

The smoothing branch must also name the PaRIS backward-draw parameter and the
Fearnhead--Wyncoll--Tawn linear-cost smoother. A generic backward-pair label
cannot expose the distinction between one backward draw, two or more draws,
mixing assumptions, path degeneracy, and O(N) versus O(N^2) computation. These
methods estimate additive complete-data functionals; their output is not
automatically the marginal score and their regular-HMM assumptions do not close
the degenerate DSGE branch.

The literature inventory also includes the coupled conditional-particle smoother of
Jacob, Lindsten, and Schön. Its Rhee--Glynn construction is a qualitatively
different answer from averaging two biased score estimates: under its meeting
and moment assumptions it debiases a finite-horizon smoothing expectation, and
the authors explicitly combine it with Fisher's identity to obtain an unbiased
score when the transition density is tractable. The estimator has random
runtime and its own variance/cost trade-off. It is a regular-model comparator,
not a singular-transition result.

The inventory records continuous-likelihood particle filters (Malik--Pitt and
DeJong et al., with Flury--Shephard as a related econometric source) and
iterated filtering (Ionides et al.) as deferred comparators. The first family
addresses discontinuous particle likelihood evaluation; it does not prove that
the derivative of a continuous finite particle program is the model score. The
second is designed for simulation-only models with unavailable transition
density and gives a vanishing-artificial-noise approximation with an explicit
bias, variance, and mixing trade-off. It is relevant only if a model's
transition can be simulated but its density or derivative cannot be evaluated
in the required base measure. When the transition density and its parameter
derivative are available, the plug-and-play motivation does not by itself
justify this extra arm. No claim of statistical dominance follows without a
comparison. If investigated elsewhere, iterated filtering must carry
a distinct target identifier from an exact fixed-parameter score.

Twisted particle filters should be an optional proposal-quality arm because
they preserve the likelihood through a change-of-measure correction while
optimising an asymptotic normalizer-variance criterion. Multilevel particle
filters remain a deferred discretization/coupling family: their telescoping result
does not by itself yield an unbiased score or an unbiased ratio derivative.
The variational particle-objective literature (FIVO/AESMC/VSMC) belongs in the
target-boundary table so that an ELBO gradient is not reported as an observed-
data score.

These additions do not change the central support split. The inspected sources
provide no generic ambient-density KDM or PaRIS theorem for deterministic or
singular DSGE transitions. Phase 6 still requires a base-measure or
disturbance-coordinate derivation and its own score estimator before any
regular-track result can be transferred.

## Open prerequisites and amendment review

This master specifies both the implementation and the experiments; it is not
evidence that either has passed. The following unfinished work is assigned to
phases in this master. It blocks only the dependent research rows, not the
start of the master or unrelated verified work.

| Obligation | Producing phase | Current evidence and required work |
|---|---|---|
| Coordinator, study inputs, and resume/report services | 0B/0C/0G | Implemented under score_study, with backend-free tests and real baseline run/resume. Research aggregation and later method coverage remain separate milestones. |
| KDM integrated/resampling endpoint | 0D, with shared baseline regression in 0C | Implemented and tested in the protected 0D snapshot; 48 CPU combination rows and three GPU consumer rows complete. Main integration preserves the concurrent native loops; both consumers pass main CPU direction checks and the focused GPU callback retry. |
| Parameter-dependent initialization | 0C and every affected 0D--0G adapter | Explicit initial-state/covariance tangents implemented; six-parameter canonical finite-difference regression passes. Extend the same contract to each later adapter. |
| UKF/KDM/SGQF moment lifecycle | 0C--0E | Shared UKF/SGQF and persistent Gaussian-mixture prediction, observation conditioning and reset carry execute with total analytical tangents. Provider snapshot 677e38a8 has CPU/GPU and independent selection/claim checks. Nonlinear snapshot a99a1c55 reaches the same shared consumers. The mixture construction is a local assumed-density candidate, not a reproduction of Younis's learned filter. |
| Twisting/iAPF | 0E and bounded Phase 3 diagnostic | The September 18 public-reference, curved-fit and Fisher comparisons are complete (results linked above). Correct fixed-label differentiation is distinct from physical-score estimation. Terminal Fisher corrects the mean-bias failure in the tested scopes but has high variance. Exactly centered ancestor controls reduce N4096 error versus raw Fisher on four fresh nonlinear datasets under the primary uncertainty criterion; UKF/no-resampling promotion vetoes remain. The optional input-precision rank safeguard passes fresh numerical non-harm validation. Next: derive remaining innovation controls/conditional integration with fresh calibration, preserving the score target. Earlier fitting-bound failures, wider dimensions and iAPF-moment/LEDH integration remain open. |
| KDM as a LEDH control variate | 0D | Known-zero-center mixture density-score control and independent calibrated biased-score blend execute. The tiny exact-control fixture has worse descriptive validation error; no ranking. Other unknown centers cannot inherit this result. |
| Coupled finite differences | 0F | Three stencils, full directional reconstruction/covariance and frozen-design consumption execute. Forty-row pilot plus independent eight-row selection and two held-out mechanics rows complete. Conditional normalization, ratio covariance and N/2N/4N consistency reports also execute; 48 diagnostic rows complete. The eight-row pooled LEDH consistency association is only r=0.058, descriptive and insufficient for bias calibration. Larger replication and calibrated combinations remain open. See phase-0f-result-and-refresh.md and phase-0g-capacity-normalization-result.md in the active artifact root. |
| Companion manuscript | 0A, then relevant phase repairs | Synchronize FD smoothness, stochastic MSE, direction convention and covariance lifecycle before using it as the revised implementation specification. |
| Additional models, row matrix, and final reports | 0G | Scalar nonlinear transition/observation adapters, a refined numerical grid reference, EKF/UKF and corrected particle baselines execute at a99a1c55. All 96 rows across three regimes complete. At these untuned settings, every LEDH covariance candidate loses descriptively to EKF in every regime; this vetoes promotion, not the research direction. GPU capacity checks at d=4 and d=12, N=64, T=3, o=2 complete for Kalman and all three shared LEDH consumers. Scientific scope tuning, wider model coverage, replication and terminal reporting remain open. |
| Serious run | Applicable phase and preceding refresh | Fill the evidence contract, defaults audit, partitions, finite attempts/compute budget, exact environment/commands, and unique output root. |

The 15 September provider/nonlinear result and required next repair are recorded
in [phase-0g-provider-nonlinear-result.md](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-provider-nonlinear-result.md).
The [density-fit audit](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-density-fit-audit.md)
adds a required Phase0E specification: an unrestricted covariance/scale search
in the paper's density-scale criterion can have a zero infimum without a
finite minimizer. The implemented arm specifies its local optimizer and
termination semantics; a small absolute fitting loss alone cannot certify its
shape or downstream score quality. The existing local fitted comparator
remains available under its declared target and method identity.
The completed normalization/capacity allocation is recorded in
[phase-0g-capacity-normalization-result.md](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-capacity-normalization-result.md).
The completed consumers have been integrated into the main checkout while
preserving the concurrent native flow-substep loop. Protected/native-loop
parity, main consumer regressions and all nine planned GPU consumer smokes
passed, including a recorded repair of two initially omitted KDM callback
modules. The [integration result and next repair](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-main-integration-result.md)
records the versioned evidence. This establishes bounded implementation
coverage; scientific tuning and the remaining master phases are still open.

The companion manuscript at
[ledh_younis_kdm_score.tex](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex)
now uses C5 smoothness, proves the finite-difference MSE decomposition with
its cross-term, separates stochastic error from deterministic order, and gives
the complete SGQF prediction/conditioning lifecycle. The PDF was rebuilt and
the amended pages inspected. Four MathDevMCP algebra checks passed; its two
equation-role audit abstentions are preserved and are not a whole-document
proof certificate. The existing ratio-bias witness remains valid, and
clarifying the flow determinant does not imply adding an OT determinant.

The [amendment review](../reviews/younis-score-master-program-amendment-review-2026-09-14.md)
records the mathematical checks, remaining implementation/documentation
obligations, and review scope. Historical reviews remain preserved. Their
procedural gates do not supersede the current repository policy, and a review
or document check is not a MathDevMCP audit result.
