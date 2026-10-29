# Judging surrogate HMC by the inference it delivers

The goal is to test whether an imperfect DCTR proposal force can support
reliable parameter inference through endpoint-corrected HMC, and to measure
its cost. A useful pilot, a formal equivalence result, and an efficiency claim
answer different questions. Report each explicitly instead of compressing
them into one pass/fail label.

This revision follows the owner's acceptance of LG1 after inspecting its
parameter differences, acceptance and R-hat. It governs prospective scope
design and interpretation. Archived comparisons and executable thresholds
remain unchanged. It does not authorize more LG1 sampling, promote an
uncalibrated uncertainty estimator, or change the campaign budget.

## 1. Identify the posterior before judging accuracy

| Comparison | Question | Consequence of a problem |
|---|---|---|
| B/R: DCTR force, exact model endpoint, independent model reference | Does the surrogate force support sampling of the model posterior? | Inspect the proposal, mixing, implementation or sampling precision. A finite-DCTR likelihood discrepancy does not invalidate this comparison. |
| C/D: DCTR force, finite-DCTR endpoint, independent finite-posterior reference | Does the sampler sample its declared finite posterior? | Inspect the sampler against that same target. |
| D/R: finite posterior versus model posterior | Does the likelihood approximation materially change inference? | Repair the finite likelihood if the change matters for the intended inference. This is a separate question from whether the surrogate force works. |
| C/R: finite-endpoint samples versus model reference | Is the combined inference accurate enough? | Use C/D and D/R to identify the source of a material difference. |

Retain A, exact-force/exact-endpoint HMC, as the sampling control wherever
available. If the same precision test is unresolved for A and B, investigate
the test's sample-size requirements before attributing the result to the
surrogate force. Failure of a shared implementation check still invalidates
both affected arms.

The invariance argument is in master-program section 3.1: a frozen,
deterministic position-only force can generate reversible, volume-preserving
proposals even when it differs from the endpoint gradient. Correct endpoint
and kinetic-energy acceptance then preserves the declared posterior in ideal
arithmetic. This does not establish numerical correctness or finite-run
exploration; the following checks remain necessary.

## 2. Make five separate decisions

| Decision | Evidence required | Interpretation |
|---|---|---|
| Implementation and target validity | Correct model law, prior, parameter chart/Jacobian, endpoint and per-chain kinetic energy; qualified proposal map; finite/status checks; frozen retained kernel; traceable actual call chain | A failed check invalidates the affected result. Repair it before interpreting posterior accuracy. |
| Sampling health and precision | Four dispersed chains; maximum rank-normalized split/folded R-hat at most 1.01; bulk/tail ESS at least 400 as initial screens; movement and relevant mode/tail exploration; function-specific MCSE | A mixing problem blocks posterior acceptance. Low precision limits the claim and may justify a budgeted extension. Acceptance rate is explanatory, not a convergence test. |
| Practical accuracy for this scope | Reference comparisons in physical units and posterior-SD units, uncertainty, interval/shape and relevant predictive or decision quantities, assessed against consequences specified before the run | A pilot can be accepted for moving to the next scope when the evidence meets its stated purpose. Explicit scientific judgment must give its rationale and cannot conceal an unresolved material concern. |
| Formal statistical confirmation | Qualified uncertainty intervals, including reference error and the declared multiplicity/sequential rule, wholly inside frozen margins for all primary quantities | Report equivalent, discrepant or inconclusive separately for B/R, C/D, D/R and C/R. An interval crossing a margin is inconclusive. An interval containing zero alone does not establish equivalence. |
| Computational usefulness | Total preparation, tuning, compilation and sampling cost to comparable posterior precision; scope-appropriate cheap baselines on the same data and important regimes | Correct sampling can be expensive. A supported efficiency or superiority claim requires a fair comparison and uncertainty evidence. |

The diagnostic recommendations in Vehtari et al., sections 2, 3.2 and 4,
treat R-hat and ESS thresholds as initial checks and require precision for the
actual quantities of interest. Rank bulk-ESS is not the effective sample size
of every posterior mean or event probability. Their folding, localization and
simulation examples also explain why checking only means or a single R-hat
can miss exploration problems. See the [paper](https://arxiv.org/abs/1903.08008)
and the inspected [local copy](../../.localresources/papers/vehtari-et-al-2021-rank-normalization-folding.pdf).
These recommendations do not supply universal scientific error tolerances.

## 3. Choose tolerable errors from the intended use

Each remaining scope must name its important inference quantities before
sampling: parameter location and uncertainty, structurally important
dependence or mode probabilities, and a predictive/decision quantity where
applicable. Explain what error would change the scientific conclusion. Keep
the supporting diagnostics, but declare which quantities determine the claim;
do not select that list after seeing which tests pass.

For a parameter mean, report both the difference in physical units and
`abs(mean_candidate - mean_reference) / SD_reference`. The existing 0.10-SD
margin is a starting hypothesis, not a universal fact. Likewise, the existing
0.10 centered-second-moment, 0.02 probability and 0.03 predictive-probability
margins require scope-specific justification. A two-percentage-point error
can be modest for a central probability and unacceptable for a rare event or
a decision near its threshold. A percentage change in a parameter near zero
is an unstable measure of scientific accuracy.

Use reference uncertainty as well as sampling uncertainty. If the reference
is empirical, report its refinement evidence and remaining error; do not call
a refinement difference a rigorous bound. Joint/mode discrepancies can veto
an accuracy claim even if marginal means agree. MLE agreement is useful
supporting evidence but does not determine posterior spread, tails or mass.

For a finite-likelihood approximation, judge posterior distortion rather than
the percentage difference between two log-likelihood numbers. With the same
prior and support, write `r(theta) = log L_D(theta) - log L_M(theta)`. Assuming
both posteriors are proper and their likelihood ratio is defined on the model
posterior's support,

```text
Z_D = integral L_M(theta) exp(r(theta)) prior(theta) dtheta
    = Z_M E_pi_M[exp(r(theta))],
pi_D(theta) / pi_M(theta) = exp(r(theta)) / E_pi_M[exp(r(theta))].
```

A constant log-likelihood offset therefore cancels exactly; variation over
posterior-relevant regions changes inference. For a pointwise likelihood
comparison, the percentage difference is `100 * (exp(r) - 1)`. Neither that
percentage nor MLE agreement replaces the posterior comparison. Inspect
scientifically important tails and modes as well as the central region.

## 4. Budget for the claim, then stop when that purpose is met

Declare the stage at launch: a bounded mechanism pilot or a formal
confirmation. Before a confirmation, forecast interval width for its actual
functions, reference uncertainty and permitted looks. As a planning
approximation for a mean, with function-specific effective sample size and
negligible reference error,

```text
MCSE / SD approximately 1 / sqrt(ESS_mean),
interval half-width / SD approximately c / sqrt(ESS_mean).
```

For example, a 0.05-SD half-width with `c=1.96` needs about 1,537 effective
draws. This is a single-interval normal approximation, not a power guarantee
or a substitute for measured MCSE. Multiplicity, dependence-estimation
uncertainty and reference error must enter the actual plan. The existing
[precision audit](surrogate-hmc-inference-precision-result-2026-09-22.md)
shows why transferring one short schedule to thirteen functions and four
comparisons was inadequate for the proposed 0.02 probability margin.

Stop an accepted practical pilot and proceed to the next model. Extend only
when a predeclared remaining question matters, the extension can plausibly
resolve it, and the reserved budget permits it. At the cap, report the
specific uncertainty and stop that comparison. Do not turn a limited
precision result into sampler failure or repeat chains indefinitely to obtain
a pass. A failed candidate can still motivate the next planned repair.

Uncertainty diagnostics themselves need calibration. The old absolute
batch-lag screen rejected 454/512 exact IID controls; that is a diagnostic
defect, not evidence that all those samples failed to mix. The replacement
binary/randomization method has only its tested Gaussian qualification. Use
the [calibration record](surrogate-hmc-mcse-diagnostic-result-2026-09-20.md)
and existing repair results to design a bounded prospective validation for
the actual remaining functions. Do not silently replace the runtime rule or
rescore closed chains.

## 5. LG1 under these criteria

| Evidence | Observed result | Judgment |
|---|---|---|
| B sampling health | Mean acceptance 73.33%; maximum R-hat 1.00148; bulk/tail ESS about 3,844/3,603; zero recorded divergences | Healthy pilot sampling; acceptance is supporting information. |
| B/R posterior mean | Difference 0.04215 reference SD; original interval [-0.00465, 0.08896] | Inside the original 0.10-SD margin, including its reported uncertainty. |
| B/R central probability | Median-CDF difference -0.01050; original interval [-0.03581, 0.01481] | Small observed difference; the stricter 0.02 formal equivalence question remains inconclusive. |
| D/R approximation | Median-CDF difference -0.021117, about 2.11 percentage points | Original formal discrepancy remains recorded; the owner accepts its practical size for this pilot. It does not invalidate B's exact-endpoint sampling. |
| Finite-likelihood MLE | Grid shift 0.03340, or 0.090 reference posterior SD; exact likelihood loss at that MLE 0.2793% | Supporting evidence for the owner's practical assessment; distinct from the B/R posterior mean. |
| Efficiency | Preparation and unequal kernel/compilation costs have not established a speed advantage | No efficiency advantage claimed. |

Decision: **LG1 remains an accepted practical pilot; move on.** This records
the owner's scientific judgment without retrospectively increasing the 0.02
margin or declaring formal equivalence. No extra LG1 experiment is required.
Evidence: [sampling result](surrogate-hmc-lg1-inference-result-2026-09-20.md),
[finite result](surrogate-hmc-lg1-finite-result-2026-09-20.md), and
[likelihood result](surrogate-hmc-lg1-likelihood-result-2026-09-22.md).

## 6. Implementation boundary and review

The master plan now separates practical scope decisions from statistical
evidence. Future reports should record the declared stage, practical decision
and rationale, each formal comparison, mixing/precision limitations, cost and
next action. Current executable verdicts remain historical inputs to that
report; new reporting fields and any diagnostic replacement still require
implementation and focused checks before being described as available.
No tuner, sampler or numerical default is changed here. The supported
position-field binding to `tune_hmc_kernel` remains the applicable route under
the HMC tuning interface and capability registry.

Skeptical audit: the strongest risk is making a pilot label excuse meaningful
bias or poor exploration. Prevent that by requiring direct reference evidence,
uncertainty and an explicit rationale, preserving every adverse result, and
forbidding a pilot acceptance from becoming a formal/default-readiness claim.
For new scopes, choose accuracy margins from the intended use before results;
do not raise them to fit LG1. The R-hat/ESS floors retain their literature
provenance as screens; the numerical accuracy margins remain hypotheses until
justified for the scope. The stopping rule addresses the observed precision
and diagnostic-design failures without weakening target or numerical checks.

No new experiment was run for this revision. The remaining allocation stays
26,810.767347 CPU-worker seconds and 201,480.192604 GPU-wall seconds, subject
to reconciliation before another launch. All-model practical acceptance and
all-model formal confirmation must be reported separately; neither follows
from the LG1 pilot.
