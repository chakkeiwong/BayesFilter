# RBF source repair and fitting execution

Active result,30 September2026. This continues the guided/Lorenz result in
controlled-smc-kdm-repair-execution-result-20260929.md. Controlled-SMC and
KDM empirical replications both remain incomplete. The authoritative plan is
controlled-smc-kdm-replication-master-program-20260928.md; current execution
specification is controlled-smc-rbf-fitting-validation-20260929.md.
Current stage: paired full validation58 failed ten checks; diagnosis59
isolates coefficient/top16 sensitivity. Fixed-policy pilot60 completed768
runs; final precision stage61 is running under
controlled-smc-rbf-fixed-policy-sanity-20260930.md.
Full fit51 and calibration54 failed against the earlier factor0.01 reference.
Attempts55/56 isolated stopping and component-retention sensitivity; they did
not complete replication. Chronological sections below preserve rejected attempts.

## Checked source error and corrected forward consumer

The pinned ToyMultimodal/cSMC_Resample.m lines110–119 choose a mixture
component with probabilities belonging to row n, but its Gaussian mean
belongs to the resampled parent Ancestor(n). This is wrong relative to
published eq11/Algorithm1 and supplement9.1: the conditional proposal's
component probabilities and component means must use the same parent.

For a Gaussian base N(m(x),q) and policy exp(c) sum_j w_j exp[-beta(z-k_j)^2],
let D=1+2 beta q. The component variance is q/D, its mean is
(m(x)+2 beta q k_j)/D, and its probability is proportional to
w_j exp[-beta(m(x)-k_j)^2/D]. All three are computed at the selected parent.
The integral is exp(c)/sqrt(D) times the sum of those component masses.
Using probabilities from a different parent changes the proposal density;
the inspected source weight does not correct that unintended proposal.

Attempt34 executed the actual source. Instrumentation left its trajectories
and likelihoods bitwise unchanged. It found98 affected particle transitions
in the N64/H6 fixture, with maximum probability total variation0.9999971113.
Replacing the probability row by Ancestor(n) eliminated the mismatch. An
analytic two-parent case changed the conditional mean by5.33333333; identity
ancestry and constant-policy controls showed zero discrepancy.

Attempt35 passed18 GPU/XLA corrected-forward checks. Maximum trajectory error
was3.55e-15, incremental-weight error4.17e-13, likelihood error1.99e-13 and
ESS error1.28e-13; ancestry matched exactly. Both forward and mixture
functions traced once. Memory growth was verified, peak allocator use109568
bytes, float64/TF32-off/XLA-on. Fresh streams also passed validity checks.
This validates the scoped forward kernel; no fitting or Table2 claim follows.

## Full-horizon fitting failure

Attempt36 tried N512/H101/M16, R=1 and two fixed seeds. It timed out after
90.004024 CPU seconds before completing the first seed. The source has more
than50000 scalar categorical calls per forward pass. A vectorized sampler
preserving the corrected parent probabilities and Octave random draw order
was therefore evaluated, with an exact small-fixture regression before use.

Attempt37 passed that bitwise acceleration check and logged every fit.
NNLS at time101 converged in20 iterations, time100 in239; time99 exhausted
the new5N=2560 iteration guard. The source default allows100000 iterations
without the caller checking a convergence flag. Thus the short single-solve
cost probe did not justify the full source run, and source availability did
not establish numerical convergence. The timeout cannot be attributed solely
to sampler overhead; a concrete backward-fitting failure is now recorded.

Attempt38 froze the failing matrix, target and source iterate and evaluated
independent solvers, but its summary failed to serialize a NumPy reference
boolean. Its MAT and coefficient files remain preserved. The serializer was
repaired without changing numerical calculations; attempt39 completed the diagnosis.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject pinned RBF proposal as the paper's conditional kernel | Parent mismatch proven by source execution and derivation | Wrong density/weight pairing | Exact provenance of published saved tables | Use explicitly corrected kernel | Paper mathematics or published tables are disproved |
| Accept scoped corrected TF forward consumer |18 checks pass including exact ancestry | No veto in this fixture | Full-horizon and fitted-policy behavior | Validate NNLS and complete consumer | Empirical replication or default readiness |
| Hold full native NNLS fitting promotion | Full GPU51 and fixed-reference calibration54 failed; matched frozen factor0.1 comparison56 passed | Retained policy and downstream reproduction still required | Full-scope sensitivity on new particle seeds | Execute paired validation57/58 | Controlled-SMC direction is rejected |

| Inference class | Status |
|---|---|
| Hard veto screen | Source conditional kernel wrong; source solver convergence fails;51/54 fail retained-policy reproduction |
| Statistically supported ranking | None in this diagnostic stage |
| Descriptive-only differences | Timing, support sizes and numerical errors |
| Default readiness | Not established |
| Next evidence needed | Paired full-scope validation on fresh particle seeds, then untouched empirical repetitions |

Artifacts are in artifacts/controlled-smc-kdm-replication-20260928/attemptNN_PHASE.
Every attempt preserves source snapshots/diffs, command log and run manifest.
CPU references intentionally hide GPUs. Failed attempts remain charged.
The dispatcher/checkpoint carry the live resource totals.

Post-run red-team: an Octave-specific NNLS failure does not prove MATLAB's
solver or the paper's fitting objective fails. The next independent solve
discriminates this possibility. A corrected proposal at N64/H6 says nothing
about full fitting at N512/H101. The strongest current conclusion concerns
the explicit source index mismatch; attribution of published empirical results
would require the exact driver, solver and saved-table provenance.


## Frozen-problem result, attempt39

SciPy NNLS converged with69 positive coefficients in0.046s and passed the
declared KKT guard: dual residual1.50e-14, RSS0.0002433004944. The Octave
iterate was close (RSS0.0002433011022), but its dual residual5.49e-10
exceeded its4.89e-11 stopping threshold and it failed to terminate. The
512-column Gaussian design has singular values38.54 to2.32e-18 and numerical
rank190 at eps*N scaling. SciPy BVLS reported success but failed the
independent KKT check (3.61e-8), demonstrating why solver success flags alone
are insufficient. These are numerical diagnostics, not stochastic rankings.

The eligible independent reference shows the objective can be solved on the
failing input. The planned TF capacity64 was insufficient for its69 positive
coefficients; the pre-execution review raised the engineering bound to128,
preserving all512 candidate columns and the original objective. The focused
GPU validation is next. Full-horizon fitting remains unvalidated.


Attempt40 stopped on a dtype mismatch in guard constants before numerical
testing; dtype-matched constants repaired it. Attempt41 then passed9/10 checks
but missed prediction accuracy (max4.9531e-7,34 entries), while objective error
was7.67e-11. Its407 iterations reached dual4.24e-11, peak65 active columns,
171 removals and minimum QR margin1.15e-6. The failure triggers the plan's
bounded stopping-accuracy calibration; no acceptance threshold is relaxed.
GPU charges40/41:2.719385s and22.056080s. Through41, GPU10848.440419s and
CPU200.920838s consumed; inherited GPU120102.859932s remains.


Attempt42 resolved the frozen prediction discrepancy by calibrating solver
termination. All three factors passed unchanged criteria;factor1 selected by
the predeclared rule, with prediction error4.55e-8. Factor0.01 agreed to1.16e-12.
One batched XLA trace; charge33.181254GPU seconds. This supports insufficient
termination accuracy as the cause of41's failure. It does not certify the
backward recursion or the discontinuous top16 coefficient-retention step.


## Full independent reference completed, attempt50

Attempts43/44 exposed a SciPy iteration failure at zero-based time90. BVLS
returned tiny negative roundoff coefficients;45 isolated that feasibility
failure and validated a reference-only boundary correction (maximum prediction
change3.55e-20). Attempt46 then caught a normally returned SciPy NNLS solution
with complementarity4.05e-4 at time82. After repairing fallback dispatch,47
rejected both NNLS and BVLS there. Attempt48 also rejected trust-region and
column-scaled solvers. These rejected outputs are not reference authorities.

Attempt49's independent active-set implementation with fresh SVD solves passed
the frozen time82 case and four analytical controls. Attempt50 applied it
without fallback to every backward time on both seeds. All202 fits passed:
max dual1.56e-13, max complementarity8.96e-14, max active count98,
max iterations1207, minimum relative SVD margin1.19e-10. The corrected author
forward consumer and complete replay fixtures were also exported. CPU charge
34.874853s. Exact command, source snapshots, solver diagnostics and fixtures:
artifacts/controlled-smc-kdm-replication-20260928/attempt50_c1_rbf_svd_reference/.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept this independent reference |202 original-objective KKT/feasibility checks pass | No active-rank or iteration veto | Top16 sensitivity and GPU parity remain unchecked | Execute complete TF fit/forward comparison | Literal author solver parity or Table2 replication |
| Reject installed-solver candidates on observed cases | Guarded original-coordinate residuals fail | Iteration, complementarity or dual failures | Specific library/conditioning mechanism | Preserve failures; use validated fresh factorization | Convex NNLS objective or paper is wrong |

Hard numerical vetoes above are supported. The fresh-SVD reference remains
viable for numerical comparison; no stochastic ranking is involved. Default
readiness remains open. The strongest alternative explanation is an interaction
between extremely ill-conditioned columns and particular solver updates; no
general claim about SciPy or Octave on other NNLS problems is justified.
Even matching full fitted predictions would not guarantee agreement after
retaining only16 coefficients; the next test must inspect that operation.


## Full GPU candidate rejected; first divergence located

Attempt51 executed the complete native GPU/XLA backward recursion and corrected
forward consumer. Numerical fits remained valid, with maximum866 iterations,
peak90 active columns and minimum QR margin6.86e-9. Fit, forward and frozen-NNLS
kernels each traced once. Fresh filter streams were valid. Charge398.489946GPU
seconds; first full-fit call384.897108s. This run had no observed XLA failure,
Python time loop, NumPy runtime path or retracing defect.

The full comparison nevertheless failed: maximum full-prediction difference
0.01376, backward-target0.01823, retained-log-integral5.55, cumulative-logZ
0.2034, and24821 ancestry entries differ. These are hard failures relative
to the predeclared numerical replication criteria, not statistical comparisons
of method quality. Passing individual KKT screens did not establish the
correctness of the complete fitted-policy reproduction.

Attempt52's post-run reader failed on float-valued JSON shape metadata;53
fixed the integer conversion and completed the diagnosis. For both seeds,
the earliest reverse-time discrepancy occurs at zero-based time99:

| Seed | Target error | Full prediction error | Coefficient error | Shared top16 indices | Retained lookahead error |
|---|---:|---:|---:|---:|---:|
|20261006|1.39e-15|5.62e-8|0.06190|12|0.05009|
|20261007|1.17e-15|3.75e-8|0.06271|13|0.11500|

At that point the full predictions still satisfy the frozen-solve tolerance.
The source top16 operation converts different near-equivalent coefficient
representations into different policies; the next backward targets therefore
differ. Reconstructing the reference's top16 policy directly from its full
coefficients gives exactly zero discrepancy, excluding a reference-export
selection mismatch. The policy values at the current particles have maximum
log differences20.62/41.61, including regions where both policies are tiny;
the true-parent lookahead differences above show an effect on the actual
recursion rather than only a tail diagnostic.

This establishes truncation sensitivity for the observed candidate. It does
not prove no tighter solve can reproduce the reference, and does not disprove
controlled SMC mathematically. Attempt54 tests both time99 problems across
four stopping factors using the actual retained-policy criteria. A further
full retry is unjustified unless a setting passes both cases.


## Stopping-rule and component-selection diagnosis, attempts54–56

Attempt54 tested factors1,0.1,0.01,0.001 on the two frozen time99 problems.
All8 solves passed feasibility/KKT and absolute full-prediction/objective
checks, but none passed retained-policy and integral comparison against the
fixed0.01 SVD reference in both cases. Charge37.393265 GPU seconds.
Tightening termination alone did not reproduce that reference.

Attempt55 tested the independent reference's sensitivity and evaluated the
saved GPU coefficients before and after truncation. Both0.01 reference replays
were coefficient-identical to their saved outputs. Both0.1 reference fits
remained numerically valid but changed the retained policies; both0.001
reference fits exhausted the2560-step iteration cap. Therefore55's overall
FAIL records failed tight-tolerance alternatives, not invalid reference replay.
CPU charge6.123706 seconds. Effective numerical ranks of the512-column designs
were134 and140, using eps*512*sigma_max; minimum singular values were about3e-18.
These are floating-point effective ranks, not a proof of exact rank deficiency.

For the8 saved GPU coefficient vectors, full-mixture Gaussian integrals at the
actual preceding-parent means all passed the original mixed tolerance; top16
integrals all failed. At factor1, removing components amplified log-integral
discrepancies from8.27e-6 to0.05009 and from5.23e-7 to0.11500. Full-mixture
log-policy comparisons in low-density tails still failed, even when absolute
full predictions passed. Truncation is a measured amplifier; it is not the
only reason absolute fitting error is an inadequate policy diagnostic. Keeping
all512 components has not been adopted or validated as an algorithm repair.

The review then distinguished two comparisons: reproducing the old fixed0.01
reference and reproducing the same finite stopping procedure across solvers.
Attempt56 used saved coefficients only and kept the criteria unchanged:

| Particle seed | Shared stopping factor | Full-prediction error | Coefficient error | Top16 overlap | Retained-log-policy error | True-parent log-integral error | Paired policy result |
|---|---:|---:|---:|---:|---:|---:|---|
|20261006|0.1|1.11e-15|7.55e-12|16|1.06e-10|2.86e-12|PASS|
|20261007|0.1|1.33e-15|4.83e-12|16|2.32e-10|1.26e-11|PASS|
|20261006|0.01|5.83e-11|5.34e-5|16|0.00138|4.65e-5|FAIL|
|20261007|0.01|6.65e-9|0.06176|15|31.744|0.10572|FAIL|

Attempt56 PASS means the saved-data diagnostic completed; only the factor0.1
arm passed both frozen comparisons. It does not erase54's failed fixed-reference
comparison. This nominates a paired numerical procedure, not a statistically
better method or a default. CPU charge2.719592 seconds.

The executable old full phase now requires retained-policy qualification and
reads that qualification's stopping factor. A direct invocation was rejected
before numerical launch with 'Required diagnostic has not passed'; no attempt
or compute charge was created. The new paired phase is a separate named
hypothesis with a separate plan and fresh particle seeds.

## Paired finite-procedure validation, attempts57–58

Plan: controlled-smc-rbf-paired-stopping-validation-20260929.md. Factor0.1 is
frozen in both the independent reference and GPU candidate; objective,512
columns, top16 retention and all thresholds are unchanged. The observed series
is unchanged, so new seeds20261009/20261010 provide fresh particle draws, not
new observations. TF forward stream20261011 is separately frozen.

Attempt57 passed202 independent fits, max dual1.04e-12 and complementarity
2.13e-13, CPU charge20.206999 seconds. Its fixture records and validates the
seeds and stopping factor before the native comparison. Attempt58 finished
and failed; see the terminal assessment below.
The allocation remains CPU300/GPU18000 seconds; only the attempt ceiling was
reassessed from56 to58 for these two phases. No fresh GPU-hours grant was used.


## Paired validation58 and first-divergence59: terminal assessment

Attempt58 completed in446.275804 charged GPU seconds with no exception and
all fitted/forward/fresh-forward numerical-validity checks passing. Fit, NNLS
and forward each traced once; GPU memory growth was verified and allocator
peak was1,094,322,432 bytes. Max NNLS dual1.10e-12, complementarity2.15e-14,
peak active92/128, minimum checked QR margin1.59e-8, max iterations1502/2560.
Nevertheless,10/30 comparison checks failed. Maximum retained-log-policy
error83.1302, parent-log-integral error0.106316 and forward-logZ error0.0104804
reject cross-solver reproducibility at the original criteria. Valid NNLS
diagnostics do not imply a matching retained policy.

Attempt59 (3.069356 CPU seconds) reconstructs the reference top16 exactly and
locates the first divergence before target-recursion feedback:

| Particle seed | Reverse time | Target error | Full prediction error | Coefficient error | Top16 overlap | Retained-log-policy error | Parent-log-integral error |
|---|---:|---:|---:|---:|---:|---:|---:|
|20261009|90|1.77e-14|1.92e-8|0.06024|13/16|69.7011|0.0439547|
|20261010|99|1.11e-15|1.77e-8|0.08454|13/16|23.9539|0.0439461|

The target and bandwidth match at the first failure. Coefficient selection
among nearly dependent RBF columns, amplified by top16 retention, is a
checked failure mechanism. Later target and ancestry differences are downstream
effects. Stopping-factor matching did not repair it. No further factor search
on these seeds is authorized by this plan. XLA, retracing and NumPy runtime are
not the cause of this recorded failure.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
|Reject paired0.1 computational equivalence|Failed10 comparisons|No nonfinite/rank/capacity/trace failure; retained-policy comparison fails|How coefficient sensitivity affects downstream estimator quality|Frozen-policy/oracle diagnostic60|Neither paper invalidity nor successful RBF replication|
|Accept59 as localization evidence|Reference reconstruction exactly zero|Inputs intact; matched-target first divergences found|Not a unique mathematical NNLS solution or universal mechanism proof|Preserve frozen failures and stop tolerance search|No method-quality ranking|

| Inference status | Verdict |
|---|---|
|Hard veto screen|Computational-equivalence candidate rejected; forward arithmetic validity passes|
|Statistically supported ranking|None; deterministic parity is the question here|
|Descriptive-only differences|Observed errors and runtime characterize these two particle seeds|
|Default-readiness|Unsupported|
|Next evidence needed|Independent normalizer reference and fresh forward repetitions conditional on each saved policy|

Post-run red-team: the strongest alternative is that solver-specific finite
solutions differ while both resulting importance samplers remain valid. Exact
policy equality across ill-conditioned fits is not required by equation17's
normalizer construction. This possibility must be tested, not assumed and not
used to erase the failed equivalence result. The new fixed-policy diagnostic
does not relax the original criterion or promote either fitting procedure.

Through59 the ledger charges CPU283.775664/GPU11763.780688 seconds in the
CPU300/GPU18000 allocation. Inherited GPU balance119187.519663 seconds
(33.107644 hours); failed attempts are included. No job from58 remains running.


## Why fixed-policy validity is a separate question

The inspected author manuscript equations13–17 (local text415–543) makes the
distinction explicit. If L_t(x)=M_{t+1}(psi_{t+1})(x), the twisted marginal is
eta_t^psi(dx) proportional to eta_t(dx)L_t(x), while the terminal normalizer
is unchanged. Therefore the filtering-moment estimator divides current
twisted weights by L_t, normalizes them and computes the moment. The implemented
diagnostic subtracts log L_t from the log weights before softmax; this call
chain was checked in the actual forward consumer. Without that correction,
filtering accuracy would be measured against the wrong target.

Conditional on any admissible fixed positive policy, equation17 supplies the
same normalizer construction. Coefficient equality across finite NNLS
implementations is sufficient for deterministic replay, but is not a necessary
condition for both importance samplers to target that normalizer. Conversely,
small fitting residuals or high twisted-weight ESS alone establish neither
filtering accuracy nor adequate coverage after division by L_t.

Pilot60 used this separation without relaxing the failed parity contract. All
four controlled policies had higher filtering-mean MSE than both BPF baselines
in each of the three observation groups. This is the main negative finding,
currently descriptive. A filtering or default promotion is vetoed. It does not
by itself contradict the source's RBF normalizer-variance experiment.

Pilot60 completed29.149317 GPU seconds;768 rows and converged quadrature are
preserved. A post-run checkpoint reader assumed progress.json was a dictionary,
but this diagnostic wrote a list of completed arms. The numerical subprocess
and manifest completed successfully. The reader was repaired to support both
formats and checked against the existing768 rows without rerunning computation.
One intermediate indentation error was caught by host compilation before any
launch and fixed. Stage61 freezes4096 fresh repetitions per arm; no fitting,
policy selection or reuse of pilot draws.

## Final fixed-policy precision61 and deterministic coverage62

The four controlled policies fail the normalizer agreement diagnostic after
4096 independent forward runs each. Both bootstrap controls agree with the
independent quadrature. This discrepancy and the poor filtering means are the
substantive findings; successful numerical execution does not promote the method.

| Frozen arm | Mean Z/Zref | Nominal familywise99% bootstrap interval | MCSE |
|---|---:|---:|---:|
|QR, seed20261009|0.885817|[0.863728,0.945083]|0.014129|
|SVD, seed20261009|0.878959|[0.864056,0.901567]|0.006096|
|QR, seed20261010|0.920817|[0.901193,0.951169]|0.008162|
|SVD, seed20261010|0.920867|[0.900941,0.951035]|0.008144|
|BPF512|0.992342|[0.929591,1.060524]|0.021022|
|BPF6553|0.997419|[0.983089,1.012177]|0.004567|

Bootstrap intervals cannot reveal unseen importance-weight tails. Controlled
maximum ratios range18.7–57.2. The arithmetic mean discrepancy is established
for this finite sample, but mathematical estimator bias is not established.
No QR-versus-SVD ranking follows. All four controlled policies lose to both BPF
filtering-mean baselines in all three declared observation groups. For example,
QR seed20261009 MSEs are9.609,19.703,1.459 versus BPF5120.114,0.409,0.555.
These MSE differences are descriptive; no uncertainty test of that ranking was
run. They veto filtering/default promotion under the declared sanity criterion.
All four also lose descriptively to zero in the low-observation group.

Deterministic diagnosis62 finds severe undercoverage on both quadrature grids.
At time78, both solvers for seed20261009 assign about4.10e-12 ideal proposal
mass to a region containing4.051% smoothing mass and17.452% filtering mass.
At time93, both solvers for seed20261010 assign about2.69e-13 proposal mass
to a region containing2.591% smoothing mass and61.962% filtering mass.
These rows use q/p_smoothing<1e-8. Even4096*512 ideal draws would expect
fewer than0.000009 visits in either example. This is an ideal marginal
calculation, not the actual finite-particle law or a finite-N probability bound.

The independent backward and forward log normalizers agree within4.55e-13;
the two-grid smoothing means agree within7.33e-13. The analytic initial RBF
normalizer agrees with direct integration within2.85e-14. Both stable graphs
trace once at each fixed grid size. Coverage62 completed7.513119 charged GPU
seconds, peak275,615,232 bytes, verified memory growth. Precision61 completed
351.528664 charged GPU seconds, peak240,926,720 bytes,24576 saved rows.
The numerical validity checks pass in both runs.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
|Reject normalizer agreement for these four finite ensembles|All controlled intervals exclude1; BPF controls agree|No invalid arithmetic or oracle failure|Unseen importance tails; other errors may coexist|Compare full and retained fitted policies using saved coefficients|No proof of estimator bias or paper invalidity|
|Retain poor coverage as an explanation|Independent two-grid regional masses agree substantively|Initial analytic normalization passes|How much originates before top16 truncation; cumulative path mass|Deterministic stage63, no fitting or repetitions|Not an exact finite-particle coverage theorem|
|Reject filtering/default promotion|All controlled arms lose descriptively in every observation group|Heuristic sanity veto fires|MSE differences have no uncertainty intervals|Preserve negative result and diagnose mechanism|No superiority or default readiness|

| Inference status | Verdict |
|---|---|
|Hard veto screen|Fitting equivalence and downstream promotion fail; numerical/oracle validity passes|
|Statistically supported ranking|None among controlled policies; bootstrap intervals diagnose finite-sample disagreement subject to unseen-tail limitation|
|Descriptive-only differences|Filtering MSE, ESS, runtime, maxima and QR/SVD differences|
|Default-readiness|Vetoed|
|Next evidence needed|Locate undercoverage in saved full-basis fit versus top16 retention; quantify posterior path mass in the scarce regions|

Post-run red-team: poor coverage can coexist with a weighting error. The oracle,
zero-policy identity and initial normalization checks reduce specific error
hypotheses but do not prove the entire consumer correct. A full-basis policy
that covers the missing modes would implicate truncation; full-basis failure
would instead implicate fitting design, training support or backward targets.
Neither outcome licenses tuning on these diagnostic observations.

Through62: CPU283.775664/GPU12151.971788 seconds charged; inherited GPU
balance118799.328563 seconds (32.999813 hours). The allocation remains
CPU300/GPU18000. There is no new compute grant.

## Mechanism63: truncation causes the principal severe-coverage loss

The full-fit counterfactual restores most of the missing coverage. This is a
local before/after comparison of the saved coefficients, not a backward refit.

| Fitting seed | Retained16 posterior path mass in scarce regions | Full-fit path mass | Fitting particles in worst retained region |
|---|---:|---:|---:|
|20261009|10.5460%|0.006565%|330/512|
|20261010|6.7328%|0.003834%|385/512|

Rows use the4097-point grid; QR and SVD agree to the displayed precision.
The2049-point results are10.5277%/6.7235% retained and0.006350%/0.004063% full.
Path mass means posterior probability of entering at least one region with
ideal q/p_smoothing<1e-8, obtained by exact-model quadrature with killing.
This is not a finite-particle hitting probability or a proof that this mass
must equal the observed8–12% normalizer shortfall.

All saved top16 coefficients/positive knots reconstruct exactly, old coverage62
replays within the stated1e-10 criterion, and independent no-kill integration
agrees with the oracle. Missing training support cannot explain the large
retained-policy failure:330 and385 training particles occupy the badly
represented regions. Top16 retention suppresses those regions despite that
support. Residual full-policy failures are much smaller and need separate
diagnosis if they matter to a later claim.

| Decision | Primary criterion | Veto status | Uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
|Identify severe coverage loss with coefficient truncation in these saved fits|Before/after same coefficients and two-grid path masses|All replay/normalization/trace checks pass|Full backward recursion has not been refit; finite-particle effects may coexist|Full-retention fixed-policy pilot and final diagnostic under the new retention plan|No paper invalidity, exact bias proof, default or full replication|

| Inference status | Verdict |
|---|---|
|Hard veto screen|Deterministic mechanism checks pass; original downstream promotion veto persists|
|Statistically supported ranking|None; this is a deterministic counterfactual|
|Descriptive-only differences|Stochastic downstream changes still untested|
|Default-readiness|Unsupported|
|Next evidence needed|Actual forward consumer with all positive coefficients; independent normalizer and conditional heuristic checks|

Stage63 completed16.824371 GPU seconds. Both grids and full per-time quantities
are preserved. The snapshot records seed labels with a harmless trailing.0 from
the source JSON numeric type; seeds themselves are unchanged. The executable
script now formats these labels as integers for later use.

Strongest alternative: the saved full fits are internally influenced by the
retained backward recursion; this comparison does not establish what a fully
untruncated fit would do. The repair trial therefore freezes these precise
policies and makes that limitation explicit. It also tests the actual consumer,
rather than treating deterministic ideal coverage as downstream success.


## Full-retention terminal assessment, attempts64–65

Keeping every positive saved NNLS coefficient removes the previously observed
normalizer discrepancy in this fixed-policy diagnostic. Final65 used4096 fresh
forward runs for each of four policies and two BPF controls (24,576 rows).
All3094 execution checks passed. All four controlled intervals include the
quadrature normalizer; three satisfy the predeclared interval-within10% screen.
The remaining QR seed20261009 arm is compatible but under-resolved.

| Arm | Mean Z/Zref | Nominal simultaneous99% bootstrap interval | Diagnostic classification |
|---|---:|---|---|
| QR full saved fit, seed20261009 | 1.03545040 | [0.98001291,1.19427165] | Compatible, under-resolved |
| SVD full saved fit, seed20261009 | 1.00272609 | [0.98038242,1.04323817] | Resolved within10% |
| QR full saved fit, seed20261010 | 1.00030373 | [0.97801594,1.04682665] | Resolved within10% |
| SVD full saved fit, seed20261010 | 1.00022797 | [0.97796975,1.04660190] | Resolved within10% |
| BPF512 | 0.95821278 | [0.90165143,1.01582577] | Resolved within10% |
| BPF6553 | 0.99422862 | [0.98027741,1.00912425] | Resolved within10% |

The QR seed09 maximum individual ratio is151.68256, compared with30.97282 for
SVD seed09. This is rare-tail evidence, not a statistically established solver
ranking. A percentile bootstrap can miss unseen tails even with4096 estimates.
The two controls and independently converged quadrature support the comparison,
but cannot turn a finite-sample screen into a proof of unbiasedness.

Filtering-mean errors remain substantial. Conditional MSE for QR seed09 is
7.5892,19.7273,1.45977 in y<5,5<=y<15,y>=15 respectively, versus
0.11343,0.41298,0.58439 for BPF512 and0.00888,0.03162,0.04514 for BPF6553.
Every controlled arm loses descriptively to both BPF controls in each group.
This is a promotion veto for filtering/default/superiority claims. The RBF
paper experiment measures normalizer variance, so poor intermediate filtering
means do not by themselves refute that distinct claim.

The computational target was the full saved positive mixture, represented
with91 slots after deleting only exact zero coefficients. That target was
checked against the retained full coefficients. The fits themselves still used
backward targets propagated from retained16 policies. This was therefore an
exact evaluation of a named repair counterfactual, not a full backward refit,
and not reproduction of the published M16 setting. The same repository
RBFControlledSMC.forward consumer executed; no reduced kernel was substituted.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close fixed-policy branch64/65; nominate full backward refit | Normalizer intervals all include oracle; three controlled arms resolved | All3094 execution checks pass; filtering heuristic veto remains | Rare tails, two fitted seeds, original retained-policy backward targets | Fit again with every positive coefficient retained at every backward time, under a separate bounded diagnostic | Table2/source-M16 replication, filtering superiority, full replication or default promotion |

| Inference question | Status |
|---|---|
| Hard veto screen | No nonfinite output, lost positive coefficient, identity failure, excess trace or invalid oracle; observed filtering underperformance veto remains |
| Statistically supported ranking | None |
| Descriptive-only differences | Solver-arm means, extremes, conditional MSE, ESS and runtime |
| Default-readiness | Not established |
| Next evidence needed | Full backward-refit validity and downstream oracle checks, then a separately specified empirical replication matrix with learning cost and uncertainty |

GPU/XLA,FP64,TF32off,memory growth verified; policy/BPF/noise traces all1;
allocator peak257,234,944 bytes. Pilot64 charged36.257507s; final65 charged
519.198207s (child516.569670s). Total current allocation charged
CPU283.775664s/GPU12724.251872s, leaving16.224336s/5275.748128s in this
allocation. Inherited remaining GPU118227.048478s (32.840847h); no new grant.

Command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_full_retention_precision

Evidence:
[final summary](artifacts/controlled-smc-kdm-replication-20260928/attempt65_c2_rbf_full_retention_precision/summary.json),
[run manifest](artifacts/controlled-smc-kdm-replication-20260928/attempt65_c2_rbf_full_retention_precision/manifest.json).
The snapshot and rows preserve code, inputs, seeds, per-run outcomes and timings.

Post-run red-team: the strongest alternative explanation is that more forward
draws merely exposed some tails while others remain unseen. The deterministic
coverage comparison63 specifically attributes a large loss to truncation,
which supports that mechanism independently of65. A full backward refit can
still fail because retained-policy targets shaped the old fits, because NNLS
capacity/conditioning fails, or because the basis still misses posterior mass.
No more repetitions of64/65 are authorized by that branch.


## Full backward-refit terminal assessment, attempts66–68

Both fully refitted policies pass the predeclared normalizer screen. The two
mechanism controls, which keep all saved coefficients but retain the original
truncated-policy fitting targets, fail the same fresh screen. Every controlled
arm still loses descriptively to both bootstrap filters on filtering-mean MSE
in each observation group. The normalizer diagnostic therefore supports the
full-recursion repair in this narrow setting; filtering/default promotion
remains vetoed. Neither outcome completes the published M16 experiment.

Attempt66 refitted all202 backward problems with the original512 basis
functions and retained every positive coefficient throughout the recursion.
All10 checks passed. Independent recomputation from unsorted coefficients gave
maximum backward-target error2.81e-14 and maximum constant/represented-policy
error1.43e-14. Peak active set93/128, maximum iterations1613/2560,
maximum dual error1.10e-12, complementarity3.93e-14 and minimum QR
margin1.5866e-8 passed the frozen numerical guards. Fitting and audit each
used one trace; allocator peak21,373,440 bytes. Charge615.536298 GPU seconds.

Attempt67 completed128 estimates per six arms and118 execution checks.
Both full refits passed the normalizer screen; one saved-fit control excluded
the oracle. This did not select candidates or change the final design: the
pilot's engineering validity and cost were the predeclared launch conditions.
Timed arm total19.706807s projected660.617819s for the fixed4096-repeat final,
below its1500s cap. Pilot charge33.466579s.

Attempt68 completed24,576 estimates, passed all3094 execution checks, and
used92 slots, dropping only exact-zero coefficients. Each wrapper, forward
consumer and noise generator traced once. All phases used the existing
repository RBFBackwardFit.fit and RBFControlledSMC.forward call chains.

| Final arm | Mean Z / quadrature Z | Nominal simultaneous99% bootstrap interval | Predeclared classification |
|---|---:|---|---|
| Old full saved fit, seed20261009 | 0.98272235 | [0.97603804,0.99130403] | Discrepancy; control rejected under this screen |
| Full backward refit, seed20261009 | 1.00188069 | [0.99712553,1.00664658] | Resolved within10% |
| Old full saved fit, seed20261010 | 0.98351947 | [0.97583749,0.99349064] | Discrepancy; control rejected under this screen |
| Full backward refit, seed20261010 | 1.00204889 | [0.99738342,1.00666845] | Resolved within10% |
| BPF512 | 1.01187309 | [0.94577159,1.08529823] | Resolved within10% |
| BPF6553 | 1.00030845 | [0.98591872,1.01442668] | Resolved within10% |

The previous saved-fit results65 remain preserved. Their compatibility in that
batch and incompatibility here expose the fragility of finite-sample intervals
for rare-tail importance weights. The old saved-fit maximum ratio was151.68
in65 and only6.27 for the corresponding arm in68. Do not average away the
failed fresh screen or call it mathematical bias. The full-refit maxima in68
were1.4358 and1.3418, which is descriptive evidence only. Percentile bootstrap
coverage is nominal, particularly for unseen tails.

| Arm | Filtering MSE, y<5 | 5<=y<15 | y>=15 |
|---|---:|---:|---:|
| Full backward refit seed09 | 4.840822 | 19.448419 | 1.459833 |
| Full backward refit seed10 | 4.798888 | 19.493839 | 1.460086 |
| BPF512 | 0.113928 | 0.405269 | 0.559120 |
| BPF6553 | 0.008873 | 0.031160 | 0.044841 |

These are descriptive conditional comparisons, not a statistical ranking.
The filtering heuristic veto is active. The source experiment targets
normalizer variance; a policy using future observations need not yield a good
finite-particle intermediate filtering mean. Untwisting was checked in the
actual consumer; a high ESS or an accurate normalizer cannot substitute for
filtering accuracy.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close66–68; full refit passes this diagnostic | Both fresh normalizer intervals contain1 and lie within[.9,1.1] | All execution checks pass; filtering heuristic veto remains | Two fixed fitted clouds, one data series/R1, nominal bootstrap coverage | Continue the master Lorenz multiple-refinement/APF validation; specify a separate complete RBF empirical design if pursued | Source-M16/Table2 replication, unbiasedness proof, filtering superiority or default readiness |
| Reject saved-fit-only repair as reliable under this screen | Both fresh control intervals exclude1 | No engineering invalidity | Unseen rare tails versus remaining coverage loss | Preserve failed controls; no adaptive repetition under this closed branch | Mathematical bias, failure of controlled-SMC theory, or invalidation of valid full refits |

| Inference question | Status |
|---|---|
| Hard veto screen | No invalid numerical execution; filtering promotion veto remains; old saved-fit controls fail the predeclared normalizer screen |
| Statistically supported ranking | None; marginal intervals and conditional MSE do not establish comparative superiority |
| Descriptive-only differences | Means, tails, ESS, runtime and conditional filtering errors |
| Default-readiness | Not established; all-positive expansion is a named repair hypothesis |
| Next evidence needed | Independent full fit-and-forward repetitions over source noise regimes with cost and uncertainty; Lorenz repeated-refinement/source APF checks; KDM full learning protocol |

Final68 charge517.278206 GPU seconds (child514.722615s), allocator
peak252,917,504 bytes; GPU/XLA FP64, TF32off and verified memory growth.
The three new stages charged1166.281082s. Current allocation use is
CPU283.775664/GPU13890.532955s, leaving16.224336/4109.467045s.
Inherited remaining GPU117060.767396s =32.516880h; no fresh grant.
The fixed4096-repeat branch is closed; no extra repetitions are inferred.

Exact commands, input/source hashes, environment, seeds, plan identity and
timings are in each run manifest. Full-refit clouds use seeds20261009/10;
pilot/final forward seed bases are202609340/350, with the declared comparator,
identity and bootstrap offsets. Fixture57 and its hashes identify the source
data and original training clouds. All runs bind base Git commit
89065bc6354801cb368d5e163ba49fd9c4372d10 and preserve dirty source snapshots.

[Full fit summary](artifacts/controlled-smc-kdm-replication-20260928/attempt66_c2_rbf_full_refit/summary.json),
[full fit manifest](artifacts/controlled-smc-kdm-replication-20260928/attempt66_c2_rbf_full_refit/manifest.json),
[pilot summary](artifacts/controlled-smc-kdm-replication-20260928/attempt67_c2_rbf_refit_pilot/summary.json),
[pilot manifest](artifacts/controlled-smc-kdm-replication-20260928/attempt67_c2_rbf_refit_pilot/manifest.json),
[final summary](artifacts/controlled-smc-kdm-replication-20260928/attempt68_c2_rbf_refit_precision/summary.json),
[final manifest](artifacts/controlled-smc-kdm-replication-20260928/attempt68_c2_rbf_refit_precision/manifest.json).

Post-run red-team: the strongest alternative explanation is favorable behavior
on two chosen fitting clouds rather than a general repair. Untouched forward
draws validate those fixed policies only. Independent fitting/data repetitions
could overturn generalization; unseen tails could overturn the apparent
normalizer precision. The checked target is the all-positive expansion at
every recursion, not the published top16 approximation. The full method,
original-source settings, gradient behavior and publisher-supplement identity
remain separate unfinished questions.
