# Observation-conditioned proposal default and SIR accuracy comparison

Owner request, 28 October 2026: document the ESS mechanism and evidence, select
conditional proposals with observation-weighted parents as the engineering
default, and compare likelihoods and scores at T=10,20,50 across SIR parameters.
The default decision is owner-directed; it is not a claim of score superiority.

## Implementation and manuscript contract
Use one shared analytical conditional-Gaussian implementation for affine Gaussian
observations. Preserve identity and local branches and their mixture weights;
replace the global branch by the conditional mixture with predictive-likelihood
parent probabilities. Keep the original proposal as an explicit comparator.
Scalar, batched, directional and fixed-draw consumers must reach the same code;
fixed uniforms do not mean fixed observation-dependent parent indices. Unsupported
observation laws must reject this option explicitly, never silently approximate it.
No canonical LEDH or HMC admission follows: categorical selection is piecewise
constant and its boundary contribution is not a smooth finite-program derivative.

Reader: a researcher familiar with Gaussian conditioning and particle filters.
Explain the two changes separately, derive the complete mixture denominator,
show why ESS improves, retain the mixed score findings and reference uncertainty.
Preserve a source copy and checksum before editing the manuscript. Human review
of exposition remains pending; compilation and self-review check presentation.

## Research intent and evidence contract
Question: how do conditional geometry and parent allocation change log-likelihood
and all three score components for the same SIR target and observations?
Primary comparator: the adapted Zhao-Cui TT smoothing importance calculation,
using both retained rank20/fit17 and rank40/fit2 proposals for the exact same data.
It is an approximate reference, not a certified oracle. Reweight its saved full
paths at each parameter, retaining the proposal denominator. Compute analytical
complete-data scores and their normalized importance average (Fisher identity);
check against centered differences of that same finite importance likelihood.
No TT algorithm or author source is modified.

Fixed parameter cases: nominal (0,0,0), infection scale 0.9 and 1.1, recovery
scale 0.9 and 1.1, observation-noise scale 0.8 and 1.25, expressed as log scales.
These are evaluation points on one fixed dataset, not seven new data-generating
regimes. Horizons are exact T10,T20,T50 prefixes. Data seed26100611.

Baseline ladder: original three branches; conditional/uniform parents; default
conditional/weighted parents. Cheap adversaries: identity-only, conditional-only
uniform parents, fully adapted conditional-only. Evaluate each case and horizon
separately. Preserve covariance/reset checks and every failed stream. N1008,
eight paired seeds281201--281208; N2016 four paired seeds281201--281204 tests
particle sensitivity if pilot cost permits. No tuning on comparison outcomes.

Primary accuracy statistics: log-likelihood and componentwise score differences,
seed RMSE to each reference, and paired uncertainty intervals. A ranking requires
paired 95% intervals excluding zero and agreement across reference fits, with
adequate reference overlap. Multiple comparisons remain exploratory. ESS is
explanatory, not an accuracy certificate. Heuristic underperformance vetoes
scientific promotion. No default-readiness, HMC, exact-oracle or all-SIR claim.

Hard vetoes: target/data/chart mismatch, nonfinite accepted results, failed
derivative or covariance checks, corrupted artifacts. Low reference importance
ESS or fit disagreement blocks accuracy interpretation for the affected cell,
but does not stop valid measurements elsewhere. Repair localized harness errors
within budget. Do not turn an unsuccessful candidate into a direction rejection.

## Assumptions, budget and skeptical review
Flow16/reset40/epsilon1/correction0 are inherited untuned isolation controls, not
per-scope admission tuning. FP64 GPU/XLA is a precision reference exception to
TF32 production direction. CPU author-path evaluation is an independent-reference
exception. GPU memory growth and placement must be verified. N uses K=N.
The first diagnostic checks implementation, branch-preserving derivatives,
runtime and reference overlap. Parameter steps are fixed convenience scenarios;
their failure mode is path degeneracy, checked before interpreting accuracy.
Rank20 and rank40 differ also in fit seed: their difference cannot isolate rank
convergence. Conditional MCSE cannot measure TT support/truncation bias.

New campaign budget: 7200 aggregate measured subprocess seconds, at most two
localized retries per failed stage, one GPU worker, two CPU threads per reference
worker. Reserve time for summaries. Every launch uses a unique directory, timeout,
atomic per-row checkpoint, manifest, source hashes and exact command. Stop on
budget exhaustion; report unfinished cells explicitly. No paid compute or installs.
Output root: docs/plans/artifacts/observation-parent-sir-accuracy-20261028/.
Result: docs/plans/observation-parent-sir-accuracy-result-20261028.md.

Pre-run skeptical audit: the prior Zhao score is a numerical quadratic estimate,
and ESS=N is possible with an inaccurate incoming cloud. Neither is a promotion
criterion. Reusing saved proposal paths with exact importance correction changes
the evaluation parameter, not the target definition; overlap and fit agreement
must be checked at every case/horizon. This design answers fixed-data accuracy
and cannot establish performance across independently generated SIR outbreaks.
Implementation may proceed; numerical campaign awaits source/derivative checks
and a recorded final plan review before launch.

## Final pre-run review, source anchors, and executable schedule
The source comparison has been checked against Zhao--Cui (2024), equation (26)
and Algorithm 4 (p.18), and author code
third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:139--208.
The saved smoother draws reverse conditional paths and records exact-target
minus proposal log density. Our log normalizer uses log-mean-exp of these
weights; the author's mean-log-weight display is not our likelihood estimator.
The saved SIR target adaptation is retained, not relabeled source-faithful.
The new evaluator changes neither the saved proposal nor the TT fitting method.

Reference checks: complete-data analytical scores must agree with independent
canonical-callback centered differences (relative scaled error <1e-4), and
joint values within 1e-7 in FP64. Reference ESS >=1000 of 100000 is a conservative
convenience screen for gross degeneracy, not a bias certificate. Require both
fits to pass and their value/each score difference to be within three combined
conditional jackknife standard errors, evaluated separately per quantity.
Report disagreement rather than average conflicting references. Twenty equal
path blocks give delete-group jackknife errors conditional on the fitted TT.
A paired t95 interval across eight streams is exploratory, particularly with
heavy tails; report every stream and do not claim familywise superiority.

Execute reference first (maximum 1500s), and a GPU/XLA N1008,T50 one-seed pilot
(maximum 600s). Then execute the seven cases and eight paired seeds with six
arms, maximum remaining aggregate budget less 300s for summaries. Reuse the
T50 recurrence's prefix increments at T10/20: horizon only sizes the fixed noise
arrays and loop; check prefix equality on a small fixture before using this
optimization. Traverse all cases for one seed before adding replications so a
budget stop still covers the requested grid. Save each arm and prefix before
moving on. Do not run N2016 unless the complete N1008 grid and reserve fit.
Every computation stays FP64/XLA; these are comparative diagnostics, not a tuned
canonical LEDH admission. Record unavailable or failed cells explicitly.

Skeptical review PASS for this bounded diagnostic: the exact target, data and
reference denominator are fixed; naive controls and reference reliability are
measured; ESS cannot promote scores; finite-program derivatives are distinguished
from likelihood-score accuracy; no new model/default-readiness claim is made.
The owner's proposal-default decision is separate from this scientific ranking.
Remaining risks are TT bias/support, categorical boundaries, untuned reset and
flow controls, fixed-data generalization, and small-seed uncertainty. Each has
an explicit diagnostic or nonconclusion. An advisory review is requested before
launch; material findings must be repaired, procedural disagreement is recorded.

Advisory review attempt: the trusted Claude wrapper exited with “Not logged in ·
Please run /login”; no scientific review was returned. No authentication or
settings were changed. The explicit skeptical audit above is the completed local
review; independent review remains unavailable. Proceed under the repository's
review-proportionality policy. Focused CPU implementation checks passed (2 tests,
33.01s); GPU/XLA and reference derivative checks remain executable first gates.

Pilot01 infrastructure failure: an imported numerical module initialized the GPU
before the memory-growth helper. TF_FORCE_GPU_ALLOW_GROWTH was set, but the strict
helper correctly rejected late initialization. Repair: invoke and verify the
helper immediately after TensorFlow import, before numerical imports. No
scientific change. Retry pilot02 within the same budget.

Reference overlap repair: the predeclared +/-10% infection/recovery cases produce
path ESS near one at T10. They remain failed-reference cells, not evidence that
a filter is inaccurate. Add four explicitly local cases exp(theta)=0.999/1.001
for infection and recovery using the same saved proposals and evaluation math.
These do not replace or erase the original cases. Their only purpose is a
smaller fixed-data neighborhood with measurable reference overlap. No rank,
proposal, target, data or acceptance threshold is tuned. Larger-shift accuracy
will require fresh TT fits if the fixed-proposal references remain inadequate.

Pilot02 completed in226.19s: six arms valid at T50,18 prefix rows; branch-stable
score and independent-prefix checks passed. Main combines all eleven cases in
one process, five fixed-signature programs,4500s ceiling,eight paired seeds,
pilot rows reused. Diagnostic repair masks stopped-row parent indices to -1,
restoring sharded/full parity without changing healthy values or scores. Local
reference allocation<=900s, subject to aggregate7200s and300s reporting reserve.

Inference specification before aggregate analysis: for each parameter/horizon,
quantity (log likelihood or one score component), and pair of arms, form paired
seed squared-error differences against each saved reference separately. Report
the mean difference and two-sided95% Student-t interval (df=paired seeds-1),
plus an exact two-sided paired sign test as a small-sample robustness check.
A supported exploratory ranking requires the interval to exclude zero and sign
p<=0.05 with the same direction for both references, eight valid paired seeds,
reference ESS>=1000 per fit, and each of the four fit differences within three
combined conditional jackknife SEs. If any check fails, retain descriptive RMSE
only (and no accuracy RMSE at all for failed-reference cells). No multiplicity-
adjusted, population-wide or true-oracle ranking is asserted. Reference bias
outside sampled support is not included in these intervals. Heuristic dominance
is checked conditionally for every retained case, horizon and quantity; any
observed disadvantage blocks promotion, even when uncertainty prevents ranking.

Batching audit: the shared recurrence supports native parameter batches, but a
new B11 compilation/performance probe would consume budget and change the
execution configuration mid-campaign. Keep the already validated B1 execution
for this comparison. Fixed-signature program objects are cached; TensorFlow's
warning about five initial traces reflects five distinct arms, not unbounded
recompilation. No controls or metrics are tuned on the observed results.

Default-consumer audit: older future-guided runners and their diagnostic tests
must explicitly request original_three_branch, since combining that separate
future guide with the new conditional policy is not implemented. Update those
call sites rather than silently falling back or claiming the combination works.
This preserves their original experiment semantics; it does not change the
running SIR campaign. A rejection test protects the new default boundary.

Final compatibility audit found additional historical replay constructors that
load controls saved before proposal_policy existed. Such records must explicitly
resolve missing policy to original_three_branch; an explicit saved policy always
wins. This prevents a default change from silently reinterpreting old tuning or
HMC force definitions. Existing selection/scope checks remain in force and may
reject stale artifacts. No HMC tuner or sampling semantics is changed. Old
future-guide/unsupported-observation diagnostics and original call-chain tests
also declare their original policy. The current CLI exposes an explicit policy
switch while retaining the new default for supported fresh runs. Verify these
localized migrations by AST/compile checks and a small original call-chain smoke;
no additional research campaign or numerical retuning is authorized here.
