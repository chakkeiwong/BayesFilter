# Historical 6 October criteria for controlled SMC and KDM replication

**Planning-error correction, 11 October 2026:** the controlled-SMC
1% mean-likelihood and 5% `Var(log Z)` rule was **wrong planning**. Although
approved on 6 October, it set uniform pass/fail margins without first
establishing the released source's tail/moment stability or the precision
attainable within the planned budget. Q2's calculations remain valid under
that old rule, but its fixed-margin failure is not an active verdict on the
port. Those rows and interval tests below describe the historical Q2
decision only. The active controlled-SMC
rule and its limits are in the
[author-relative acceptance reset](controlled-smc-author-relative-acceptance-20261011.md).
The KDM 0.01-nat NLL and 1% RMSE criteria and the shared source-fidelity and
added-failure requirements are unchanged. Do not use this document or the Q2
decision code to launch or score a new controlled-SMC qualification.

Date: 6 October 2026. Authority:
[master](controlled-smc-kdm-replication-master-program-20260928.md).
The owner added 48 GPU hours and 48 CPU hours, then accepted these criteria and
directed continuation on 6 October 2026. The grant was recorded once; acceptance
does not add it again. These rules governed qualification under the 6 October
plan; the controlled-SMC margin rule ceased to govern on 11 October. Earlier
failed diagnostic screens retain their historical verdicts. Execution follows bounded phase
plans; adopting criteria did not establish a numerical result. The controlled-SMC
margin rows were classified as wrong planning and withdrawn on 11 October 2026.

## Target and comparator

Accept a TensorFlow implementation that preserves the released author's method
and has no material numerical degradation under matched conditions. Preserve
author objectives, densities and normalizing constants, sampling laws,
particle counts, data, architecture, optimizer semantics, training stages and
stopping rules. Source-intrinsic failures are acceptable reference outcomes.
Repairing the author's method, surpassing its performance, and Task 3/E0 are
outside this question.

The controlled-SMC comparator remains the pinned author code in the recorded
Octave/LAPACK runtime. Equivalence to the author's original MATLAB installation
has not been established and must not be claimed. KDM uses the pinned author
code and actual checkpoint loader; MSE-labelled evaluation tasks currently
resolve to NLL checkpoints. Preserve that behavior and report paper-intent
ambiguities separately. Identical integer seeds across frameworks do not
guarantee identical draws; numerical unit comparisons must replay common draws.

Source anchors for the measured quantities:
Heng et al., published Section 6.2, Figure 6 and Table 1
(.localresources/papers/heng2020-controlled-smc-published.txt:1188-1227)
report log-marginal-likelihood variability and adaptive refinements.
Preserve the published relative-variance statistic as a reproduction report,
but use variance(log Z) for the additional translation check because dividing
by mean(log Z)^2 depends on an arbitrary common likelihood constant.
Younis and Sudderth, Sections 5-6, equation (18), and Appendix D.1 equation (23)
(.localresources/papers/younis-sudderth-2023-long-range-tracking.txt:534-654,
1578-1600) define posterior NLL and tracking RMSE, with circular state treatment.
The existing source maps and circular consumer result anchor actual consumers.

## Owner-approved acceptance rule

All applicable rows must pass in the declared experiment scope. Report outcomes
per experiment and dtype. A completed diagnostic, source failure or passing
subset does not complete the full replication.

| Requirement | Accepted rule | Why this quantity |
|---|---|---|
| Algorithm fidelity | No unexplained change to the source mathematical operations, objective, randomness law, optimizer/stage dependencies or evaluation checkpoint | Good metrics cannot certify a different algorithm |
| Fixed-input correctness | Valid probabilities/covariances and matched value, gradient and optimizer-update checks with operation-specific, conditioning-aware bounds | Catches translation bugs before stochastic comparisons |
| Failures | No new failures on fixed author-valid fixtures or the untouched paired qualification set; report the one-sided confidence bound on the extra-failure probability | Zero observed failures does not prove a zero population rate |
| Controlled-SMC likelihood location | Equivalence interval for log(E[Z_port]/E[Z_author]) lies within plus/minus log(1.01) | Bounds multiplicative mean likelihood disagreement to approximately 1% without taking a percentage of an arbitrary log-likelihood |
| Controlled-SMC variability | Upper one-sided 95% confidence limit on Var(log Z_port)/Var(log Z_author) is at most 1.05 | Allows at most 5% more Monte Carlo variance, about 2.47% more standard deviation |
| KDM posterior accuracy | Upper one-sided 95% confidence limit on mean NLL_port minus mean NLL_author is at most 0.01 nat per evaluated joint state | A 0.01-nat loss corresponds to about 1% lower geometric-mean density at true states |
| KDM state estimation | Upper one-sided 95% confidence limit on RMSE_port/RMSE_author is at most 1.01, using the actual source evaluator | A 1% RMSE increase corresponds to about 2.01% more MSE |
| Full learning consumer | Source-matched checkpoint conversion, loss, gradient, optimizer state/update and complete training comparisons pass | Frozen primitive tests cannot establish end-to-end training |

The 1%, 5% and 0.01-nat margins are owner-approved practical definitions of
"no material degradation", not published tolerances or conclusions derived
from current candidate errors. They have interpretable effects on the
quantities being replicated. Literal zero degradation cannot generally be
established by a finite stochastic experiment between equivalent algorithms.
A margin is not a numerical fix; code must still compute the intended method.

The likelihood-location target is the mean likelihood estimate, not mean
log-likelihood. Compute means stably from log estimates with log-sum-exp; do not
exponentiate unshifted extreme logs. Constants and observation horizons must
match. This comparison establishes agreement with the author, not accuracy
against truth or unbiasedness. Exact small-model likelihood/posterior controls
are required correctness checks where tractable. Retain mean(log Z), original
paper relative variance, ESS and policy-fit residuals as diagnostics.

When a reference variance or RMSE is effectively zero, the ratio is undefined
or uninformative. Use an independently justified absolute numerical bound from
the exact/reference solution before running that case; do not add an arbitrary
denominator floor. Unresolved absolute bounds leave that case unqualified.
If likelihood moments are too poorly estimated for a reliable interval, the
result is inconclusive; an unstable mean estimate is not a pass.

## Numerical tolerance and the perturbation experiment

Do not require bitwise equivalence, exact ill-conditioned regression
coefficients, identical random streams from unrelated RNGs, or identical
resampling ancestors after continuous states legitimately diverge. Require the
same algorithm, finite valid outputs and correct downstream consumers.

The existing circular float32 screen remains failed: four comparisons exceeded
its frozen bound. It is a localization result, not a universal acceptance
threshold for the whole filter. Those mismatches require source-operation,
derivative and actual optimizer-consumer checks. They cannot be waived merely
because float64 passes, nor must every well-understood roundoff difference
force a bitwise port. Any proposed replacement numerical bound must have an
independent derivation/calibration and fresh verification, with the original
failure preserved.

The owner's 1e-8 random-input perturbation experiment tests sensitivity.
Use both signs and predeclared directions on source-only calibration cases.
Record the realized perturbation after conversion to the actual dtype:
1e-8 can round away in float32. Separate ordinary trajectories from resampling
or stopping-boundary changes. Compare port differences with this sensitivity as
explanatory evidence, not with the largest observed perturbation response as an
automatic acceptance allowance. Perturbing inputs does not measure error
against the true posterior or likelihood.

## Uncertainty, regimes and scope

Use fresh qualification data/seeds, separated from development and calibration.
Freeze candidate code, margins, regimes, sample sizes and interval procedure
before qualification. Calibration may estimate cost and required sample size;
it must not increase margins to accommodate candidate discrepancies.

For controlled SMC, resample independent run pairs within each fixed
parameter/data case; retain all source and port outcomes. For KDM, independently
trained models are replication units. Evaluation trajectories are nested under
training seeds and reused across source/port comparisons. Time steps from one
trajectory and 5,000 trajectories under one trained model are not 5,000
independent training replications. Use an uncertainty method that includes both
training-seed and evaluation-data variation, paired where valid.

Use a two-sided 90% interval wholly inside the likelihood equivalence margins
(the two one-sided 5% tests) and one-sided 95% upper limits for degradation.
Freeze an interval method appropriate to the data; paired/hierarchical
resampling is a candidate, not a universal guarantee with a handful of seeds.
Require adequate tail/moment diagnostics before using a variance or likelihood
mean interval. With few training seeds, report the resulting uncertainty and
any model assumptions; do not convert a nonsignificant test into equivalence.

The global claim requires every predeclared primary criterion to pass, so it is
an intersection of component tests. This does not authorize searching among
regimes or repeatedly checking conventional intervals until one passes.
For separately advertised per-regime conclusions or multiple interim looks,
predeclare appropriate simultaneous/sequential inference. Otherwise choose
the final sample size from source-only calibration and inspect once for
qualification. Failure may trigger repair, but used claim data then become
development data; a revised candidate needs untouched qualification cases.

Retain the source parameter matrix, with explicit low/central/high forcing or
noise summaries; do not hide a failing grid point in a pooled average.
For KDM report the declared normal/outlier tracking situations, angular seams
and early/late portions of long trajectories. Fix these categories from the
data-generating design before candidate scoring. Sparse training labels and
dense evaluation labels remain as in the source.

For zero extra failures in n independent, source-valid paired trials, the
one-sided 95% upper binomial bound is 1 - 0.05^(1/n). Thus 100 such trials bound
the probability only to about 2.95%; 300 bound it to about 0.994%. This follows
by solving (1-p)^n=0.05. A less-than-1% statement requires at least 299 such
trials under that independent common-probability sampling model. Shared training
models or deliberately heterogeneous grid cells must not be counted as those
independent trials. Report the scope/strata and weaker precision when necessary.
[NIST exact binomial intervals](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
supports the interval construction; the zero-failure expression is the
special-case derivation above.

Source failures are recorded, including their inputs and classification.
Do not treat unexecuted cases as failures or successes. Match known intrinsic
source failures on replay, and assess stochastic failures with stated
denominators. Infrastructure failures require repair and do not establish
algorithmic behavior. Report continuous metrics with their valid-case
denominators; silently dropping bad runs is forbidden.

No new superiority requirement against unrelated algorithms is added to
replication. Retain the original paper comparators and exact simple-model
controls. Runtime, GPU memory, compilation and trace counts establish execution
feasibility and explain cost; faster-than-author execution is not a requirement.

## Defaults, review and next action

| Choice | Provenance/status | Failure mode | Early check |
|---|---|---|---|
| Author settings and checkpoint paths | Frozen replication baseline, not a robustness default | Reproducing a label rather than the actual consumer | Existing source/loader audits plus full neural call-chain checks |
| Numerical bounds | Operation-specific checks; current circular bounds remain historical diagnostic requirements | Ill-conditioning or near-zero values make a blanket relative threshold misleading | Exact toy, directional derivatives and actual one-update comparisons |
| 1% likelihood/RMSE, 5% variance, 0.01 nat NLL | Owner-approved practical margins, not yet a claim result | Too permissive scientifically or impossible to resolve within budget | Source-only precision/cost assessment; revise scope or report inconclusive, never tune margin to port error |
| Confidence level | Owner-approved 5% test size | Optional stopping or treating steps as independent | Frozen sample size and resampling unit; audited decision code |
| GPU/TF/XLA | Repository target | CPU/reference success used as default evidence | Trusted GPU device, memory growth, placement, stable signatures and source parity |
| Training repetitions | Preserve source eleven-trial paper scope; cost before allocation | Claiming full paper replication from one fitted network | Separate pipeline validation, repeated-run translation and paper-statistic verdicts |

Skeptical review: the old raw-gradient/strict-fit screens cannot by themselves
answer numerical no-worse performance. The accepted rule uses the actual
consumers and outcomes while retaining source correctness and failure checks.
It does not infer accuracy from higher likelihood, loss reduction, one seed or
source perturbation maxima. The margins are explicit judgment calls rather
than source claims. Missing neural implementation, original MATLAB equivalence
and unresolved source specifications remain visible.

The new grant makes costed continuation possible. It does not establish that
48 GPU hours cover component pretraining plus author and TF training for all
eleven trials and both MDPF variants. Next: cost those dependencies and the
precision needed for these margins; finish the source-matched neural/optimizer
checks and localized arithmetic investigations. Then freeze an executable
qualification plan within the remaining allocation. If the budget cannot
resolve the accepted margins, report incomplete/inconclusive evidence
instead of accepting an underpowered comparison.

Stop a run for corrupt provenance, wrong target/data, invalid harness, exceeded
budget or resource failure. A valid failed candidate blocks acceptance and
triggers the next bounded repair; it does not reject the research direction.
No enhancement work is activated.

## Decision record

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Record 48 GPU + 48 CPU hours | Explicit owner grant | None | Full training cost | Add once to the live ledger | Enough compute for every paper trial |
| Adopt the rules above on owner instruction | Source fidelity plus practical equivalence/non-inferiority | Old failed screens remain failed | Margin precision and missing consumers | Source-only cost/power assessment and frozen execution plan | Either replication accepted |

| Inference status | Current conclusion |
|---|---|
| Hard veto screen | Existing float32 and Lorenz diagnostics still fail their own screens |
| Statistically supported ranking | None; no new comparison was executed |
| Descriptive differences | Existing deterministic discrepancies only |
| Default readiness | Not established |
| Next evidence | Matched full consumers and independent repeated-run qualification |
