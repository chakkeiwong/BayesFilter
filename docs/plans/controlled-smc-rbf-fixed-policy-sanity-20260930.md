# Fixed RBF policies against an independent normalizer, 2026-09-30

This diagnostic continues the authorized controlled-SMC replication campaign.
Attempts51,54,58 remain failed computational-equivalence tests. No tolerance,
fitted coefficient, training seed, objective, truncation or default is changed.
Both controlled-SMC and KDM replications remain incomplete.

## Question and research intent

Do the already frozen QR and SVD policies from57/58 produce valid estimates of
the same terminal normalizer, despite their different retained knots? The
mechanism under test is the corrected paper-derived forward filter, conditional
on a fixed positive fitted policy. This is separate from reproducibility of the
fitting procedure and from Table2's independent fitting repetitions.

The inspected author manuscript equation17, local text lines501–543, states the
unbiased-normalizer construction. Supplement section6.1, lines2817–2846, prescribes
natural-scale nonnegative regression on all particle knots followed by top-M
retention and warns about numerical stability. Pinned ToyMultimodal source
cSMC_Resample.m203–215 implements that fitting sequence. The forward formula is
the previously checked equation11/Algorithm1 correction, with parent selection
applied to both Gaussian-mixture probabilities and means. Its fixed-policy
18-check validation is attempt35. None of these sources promises coefficient or
trajectory equality across finite NNLS solvers.

Attempt59 localizes fresh-seed differences to reverse times90 and99. Targets
agree within1.8e-14 and full predictions within2e-8; top16 overlap is13/16,
retained-log-policy errors69.70/23.95 and true-parent log-integral errors~0.04395.
Reference policy reconstruction error is zero. This rejects the paired stopping
repair, but does not establish biased normalizer estimation.

## Evidence contract and diagnostic roles

Comparator: an independently written one-dimensional trapezoidal quadrature
filter for the exact same R=1, Q=10, prior variance5, nonlinear transition,
observations and initial-observation convention in fixture57. Gaussian transition
densities are integrated directly, without importing the particle or RBF
routines. It reports logZ at each time and normalized filtering moments.

First validate quadrature on grids1025/2049 over[-60,60] and2049 over[-80,80].
Require all pairwise terminal-logZ differences<=1e-7, filtering-mean
differences<=1e-6, predictive mass error<=1e-10, boundary posterior mass<=1e-12
and finite results. These thresholds are numerical-oracle validity vetoes, not
method-performance targets. If they fail, stop before particle repetitions;
a separate resolution study is required. No unvalidated oracle may reject a
particle method.

Then use four saved policies (QR/SVD for each of the two training seeds), each
with128 fresh forward repetitions at N=512. Also run BPF512 and the source's
R=1 cost comparator BPF6553,128 repetitions each. These are particle-count
comparators; this diagnostic makes no equal-GPU-cost claim. Zero-policy RBF
specialization implements BPF; its identity is checked on fixed noise.

Primary diagnostic: Zi/Zquadrature on the natural scale, never mean(logZi) as
an unbiasedness test. Save all128 per-arm values. Form20000 ordinary bootstrap
means, seed202609309, and Bonferroni-adjusted99% familywise percentile intervals
for the six arms. An interval entirely within[0.9,1.1] resolves agreement to
10% conditional precision; containment of1 alone means compatible but may be
under-resolved. An interval excluding1 triggers investigation; it is not proof
of bias with128 draws. Bootstrap cannot detect unobserved rare tails.

Hard continuation vetoes: invalid oracle, missing or nonfinite policies,
nonpositive policy mass, invalid forward status, nonfinite outputs, retracing,
source/input mismatch or budget exhaustion. Failure of precision or heuristic
performance is a result, not permission to tune or rerun. No candidate promotion
criterion is defined; this is an explanatory downstream diagnostic only.

Constructed heuristic set: BPF512 uses the same particle count without a fitted
policy; BPF6553 is the source R=1 comparison count; zero conditional-mean estimate
is a cheap adversary motivated by the observation's sign ambiguity. The numerical
oracle supplies reference filtering means. Evaluate filtering-mean squared error
separately for y<5, 5<=y<15 and y>=15, and for each frozen fitting seed/solver.
Before computing filtering moments, remove the next-policy integral from the
twisted incremental weights, as required by the untwisted filtering target.
Report every observed loss to a heuristic prominently; continuous rankings are
descriptive unless a declared interval supports them. Do not select policies on
this table. Other observation-noise regimes are outside this diagnostic.

## Defaults, review and pre-mortem

| Choice | Provenance/status | Risk and earliest check |
|---|---|---|
| Four fixed fitted policies | Unchanged failed-validation57/58 outputs; hypotheses | Reusing their fitting data cannot establish generalization; use fresh forward noise, label conditional scope |
| factor0.1/top16/bandwidth0.3 | Frozen paired plan/paper R=1 settings | Ill-conditioned coefficient selection; no refitting or tuning in this run |
|128 repetitions/batch8|Bounded diagnostic convenience|Wide/skewed intervals or rare tails; retain all draws and do not claim unbiasedness from a pass|
|Quadrature grids/domains|Independent numerical-reference hypotheses|Discretization/truncation; compare all three before repetitions|
|10% precision|Diagnostic resolution target, not source/default threshold|May be too weak for Table2; no empirical-replication claim|
|GPU FP64, TF32 off, XLA, growth|Existing numerical comparison lane|Device, allocator, finite checks and one trace per signature|
|Six-arm bootstrap|Predeclared exploratory uncertainty analysis|Finite bootstrap and sample tails; report limitations and no default selection|

Skeptical review PASS: failed equivalence evidence is retained; no fitting
criterion is relaxed or converted into success. An invariant target and an
independent oracle answer a different necessary question. Fixed-policy reuse
cannot masquerade as100 independently fitted source repetitions. The source
ancestor defect remains disclosed. All filtering metrics use the untwisted
target. The implementation must call the existing RBF forward consumer, not
copy it. Three short quadrature calls precede the longer repetition loop.
Python controls independent batches and artifact I/O only; numerical time loops
remain stable TensorFlow/XLA graphs. No pfor or NumPy runtime.

A successful run could mislead through a bad oracle, reuse of training data,
undetected importance-weight tails or interpretation of conditional success as
complete replication. The stated oracle checks, saved raw draws and scope
limits address these risks without claiming to eliminate finite-sample tails.

## Bounded execution and preserved result

Add one named phase c2_rbf_fixed_policy_sanity, capped at900 GPU seconds and one
attempt. Attempt ceiling59->60; allocation CPU300/GPU18000 and inherited campaign
budget unchanged. At59, charged CPU283.775664/GPU11763.780688 seconds, leaving
16.224336 CPU and6236.219312 GPU seconds in this allocation. No new48h grant.

Exact command (escalated, existing approved prefix):
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_fixed_policy_sanity

Versioned output: attempt60_c2_rbf_fixed_policy_sanity under
docs/plans/artifacts/controlled-smc-kdm-replication-20260928.
Preserve run manifest, source/input hashes, command log, quadrature results,
per-repetition JSONL, bootstrap intervals, conditional heuristic table, GPU
memory growth/allocator/traces, wall time and explicit diagnostic-only verdict.
Update the master, active checkpoint and fitting result after execution.

If oracle or numerical validity fails, localize it before interpretation. If
normalizer estimates disagree, trace the corrected forward formula using the
fixed policies. If they are compatible, record that fitting reproducibility
remains unresolved and decide the next replication criterion explicitly; do
not automatically launch Table2 or declare the RBF replication complete.


## Fixed precision stage after60

Attempt60 completed768 forward runs in29.149317 charged GPU seconds. All oracle,
finite, zero-policy identity and trace checks passed. The finest quadrature grids
agree in terminal logZ to2.84e-14; logZ=-250.6230216566. Every normalizer interval
includes1, but none lies within[0.9,1.1]. Mean ratios are0.908/0.908 and0.976/0.975
for QR/SVD policies,0.829 for BPF512 and0.961 for BPF6553. This is compatible,
under-resolved evidence, not proof of unbiasedness.

The substantive negative finding is filtering-mean error: all four controlled
policies lose descriptively to both BPF comparators in all three observation
situations. For QR seed20261009, MSE is9.611/19.709/1.460 versus BPF512
0.116/0.390/0.717. There is no method/default promotion. Larger repetition counts
are for estimator diagnosis, not tuning away these losses.

One final precision stage c2_rbf_fixed_policy_precision will run exactly4096
fresh forward repetitions per arm (24576 total), with the same policies, grids,
criteria, hypotheses and six-arm multiplicity adjustment. This is not a
sequential keep-running-until-pass procedure: there is one predeclared terminal
stage and no extension on its outcome. Do not pool the128-run pilot into it.
New common512 seed202609310, BPF6553 seed202609311, identity seed202609312,
bootstrap seed202609319. All source fitting seeds are unchanged.

The repetition count is a precision hypothesis: at pilot MCSE0.029–0.084,
sqrt(128/4096) scaling projects0.0051–0.0149 MCSE, conditional on representative
tails. Unobserved tails remain a limitation. Measured six-arm work15.13 seconds
per128 predicts~484 seconds plus oracle/compilation/analysis. Allocate one
attempt capped at900 seconds. Bootstrap samples are generated in bounded
chunks of128 inside a stable TensorFlow loop, preventing a multi-gigabyte
temporary. An all-ones regression must return exact means and intervals1 and
zero MCSE before actual bootstrap interpretation. This changes analysis
implementation only, not the scientific object or percentile convention.

Skeptical audit PASS: no policy fitting or selection, thresholds unchanged,
new disjoint Monte Carlo draws, explicit finite-sample uncertainty, fixed terminal
count, no performance ranking inferred from compatibility and no empirical
replication claim. Filtering losses remain the headline if confirmed.
Attempt ceiling60->61; CPU300/GPU18000 and inherited budget unchanged.
Through60 charges are CPU283.775664/GPU11792.930005 seconds, leaving
6207.069995 GPU seconds in the allocation.

Exact command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_fixed_policy_precision

Output: attempt61_c2_rbf_fixed_policy_precision. Freeze61 as terminal evidence
and write the next replication decision; no new tolerance or repeat ladder.


## Terminal61 and deterministic coverage diagnosis62

The final24576-run stage61 completed351.528664 GPU seconds. QR/SVD normalizer
mean ratios are0.88582/0.87896 for the first fitted seed and0.92082/0.92087
for the second. All four nominal99% familywise bootstrap intervals exclude1.
BPF512 and BPF6553 ratios0.99234/0.99742 pass the declared10% precision
diagnostic. Controlled-policy maximum ratios18.7–57.2 reveal substantial tails;
ordinary bootstrap can miss unobserved rare events. This is a discrepancy
requiring diagnosis, not a proof of mathematical bias. The fixed repetition
ladder is closed. All four policies also lose descriptively to BPF filtering
means in all three observation groups.

Next ask whether the fixed RBF policies nearly exclude states with appreciable
posterior mass. Use deterministic quadrature only, no more random repetitions.
For an ideal exact previous twisted marginal, equation15 gives proposal marginal
q_t(x) proportional to p(x_t|y_0:t-1)*psi_t(x). This follows by multiplying
eta_{t-1}(u) M_t(psi_t)(u) by f_t(u,x)psi_t(x)/M_t(psi_t)(u) and integrating u.
Compare it to independently computed smoothing density
p(x_t|y_0:T) proportional to p(x_t|y_0:t-1)*p(y_t:T|x_t).
The latter backward likelihood is evaluated by direct Gaussian quadrature.
This ideal marginal calculation is explanatory, not the actual finite-particle
cloud distribution or a proof of finite-N hitting probabilities.

Use grids2049 and4097 over[-80,80]. Validate backwards initial logZ against the
forward quadrature and61 oracle within1e-7, predictive-mass/boundary checks as
before, and smoothing means between grids within1e-6. Independently integrate
the actual code's initial proposal and compare its analytic normalizer within
1e-8. These are validity checks and continuation vetoes. Compare regional masses
on both grids; boundaries can discretize differently, so preserve both outputs.

For each frozen policy and each time, report posterior mass and ideal proposal
mass where q_t/p_smoothing is below1e-4,1e-6,1e-8,1e-12 and1/(4096*512).
These thresholds describe sampling scarcity and are not tuning targets. Report
the corresponding filtering mass and ideal expected proposal hits among4096*512
draws as explanatory quantities with the ideal-marginal caveat. Also report
the predicted observation group and time of the largest suppressed posterior
mass. If the same material mass is suppressed on both grids, poor coverage is
supported; an implementation error may still coexist. If it is absent, inspect
the sampling/weight identity directly rather than concluding bias from61.

Skeptical review PASS: the reference is independent, grids are checked, the
density-ratio target is explicitly smoothing for normalizer-path coverage, while
filtering error remains separately measured. No finite-N or paper-validity
claim follows from an ideal calculation. No fitting changes, regularizer, floor,
tolerance tuning or repeated-Monte-Carlo rescue.

One GPU/XLA phase c2_rbf_policy_coverage, cap180 seconds, one attempt.
Attempt ceiling61->62; CPU300/GPU18000 and inherited balance unchanged.
Exact command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_policy_coverage

Artifacts: attempt62_c2_rbf_policy_coverage with both quadrature resolutions,
per-time regional masses, identity checks, source/input hashes and result.

## Coverage mechanism63: reviewed deterministic continuation

Question: does the severe coverage failure already exist in the saved full
NNLS fit, or is it introduced by top16 retention? How much posterior trajectory
mass enters at least one of the poorly covered regions? This repairs an
explanatory gap exposed by62; it changes no fitted policy or runtime default.

Compare four frozen full512-component policies (native QR and independent SVD,
two particle seeds) with their corresponding retained16 policies. Full policies
use the original knots and coefficients, bandwidth and log scale from exactly
the same backward step. This is a local counterfactual, not a refitted full-basis
backward recursion. Count original fitting particles in the same scarce region
to distinguish absent training support from a fit that misses represented states.

Use the frozen criterion q/p_smoothing<1e-8; do not optimize its value. At each
time, construct the region from independent quadrature and the original ideal
proposal. Integrate the original state-space model while killing trajectories
on first entry to any of those regions. Then1-Z_surviving/Z is the posterior
probability that a path enters the union. This is a deterministic posterior-path
quantity, not an estimate of the actual finite-N hitting probability. Report
both grids2049/4097 over[-80,80]; threshold-boundary discretization can differ.

Primary criterion: correct unchanged-policy reconstruction and numerical
identities; diagnosis is explanatory only, never algorithm promotion. Veto on
nonfinite masses, failed original-policy match to62, forward/backward or grid
resolution failures, wrong source inputs, or invalid normalization. Require
original-policy regional mass replay within1e-10; survival fractions in[0,1];
no-killing logZ must agree with the independent oracle within1e-7. Preserve
between-grid differences in regional and path masses rather than declaring
exact continuous-boundary convergence. Count fitting particles using grid
interpolation only as explanatory diagnostics, with both resolutions reported.

Hypotheses and interpretation:
- Full policy restores coverage: top16 deletion contributes to this failure.
- Full policy also excludes the region, with few/no fitting particles there:
  the initial particle design and extrapolation are implicated.
- Full policy also excludes it despite fitting particles there: inspect the
  natural-scale loss and propagated targets at those particles next.
- None of these results proves that a weighting bug cannot coexist.

Default audit: original512 knots and all fitted coefficients come from57/58;
no new bandwidth, regularizer, coefficient floor or stopping rule. Grid and
threshold are fixed from62; their only role is diagnosis. The previously
constructed BPF512/BPF6553/zero heuristic set and61 conditional failures remain
the downstream sanity evidence. No new learned component or tuning protocol.

Skeptical review PASS: same fitted object is compared before/after truncation;
the full-basis counterfactual does not masquerade as a refit. Original-model
killed-density integration answers the stated union-mass question. Sampling
uncertainty and bootstrap issues cannot explain deterministic grid agreement.
Potential grid-boundary errors are exposed by the two resolutions, not hidden
behind a single tolerance. Runtime is capped before any empirical expansion.

Execute c2_rbf_coverage_mechanism, one attempt,180 GPU seconds maximum.
Attempt ceiling62->63; total CPU300/GPU18000 and inherited allocation unchanged.
Exact command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_coverage_mechanism

Artifacts: attempt63_c2_rbf_coverage_mechanism, full source/input hashes, both
grid results, region/path masses, reconstruction checks and terminal manifest.
Stop this diagnostic branch after63 unless a validity failure needs a localized
repair. Further algorithm changes require a separately stated method contract;
more repetitions of the existing policies are not a justified repair.
