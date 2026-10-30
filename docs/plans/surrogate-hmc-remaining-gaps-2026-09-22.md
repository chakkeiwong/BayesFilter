# Remaining gaps in the surrogate-HMC campaign

Execution update, 2026-09-22: the closure master now implements the revised
practical/formal reporting, and all 70 regression tests pass. The wrapped M13
observation-law repair passed its local checks. The uncertainty candidate passed
the declared calibration on the finite KSC function table with fresh dependent
confirmation; transfer to actual HMC and continuous-reference uncertainty remains
open. AR2's canonical structural consumer is still unimplemented. The inventory
below records the roadmap that motivated this pass; use the
[closure execution result](surrogate-hmc-closure-master-result-2026-09-22.md)
for current completion evidence, limits and remaining budget. Do not rerun LG1
or the completed diagnostic pass to satisfy stale wording below.

Revised on 2026-09-22 under the
[judgment criteria](surrogate-hmc-judgement-criteria-2026-09-22.md), against the
active checkpoint, archived reports, executable scope dispatch and explanatory
audit. This revision replaces the earlier priorities in this inventory; it
does not change any archived numerical result or formal comparison verdict.
The goal remains to test whether the full canonical analytical DCTR force can
support reliable parameter inference when a separately declared posterior
potential determines acceptance. Exact agreement between that force and the
reference gradient is not a prerequisite for this experiment.

LG1 is closed as an accepted practical pilot. Its finite-likelihood discrepancy
does not require another repair or sampling run for that pilot. Force/endpoint
gradient disagreement is also not an outstanding defect. Formal B/R results
remain INCONCLUSIVE, including LG1; practical acceptance and formal confirmation
are separate judgments.

The shared proposal/acceptance implementation has focused mechanics and
consumer-wiring checks. Retained experiments exist for four model scopes and
one Gaussian control. The other three model pilots have healthy recorded
numerical diagnostics and small observed posterior differences:

| Reference-endpoint B arm | Largest absolute parameter-mean difference, in reference posterior SD | Largest absolute checked marginal-CDF difference, percentage points | Maximum parameter R-hat | Revised interpretation |
|---|---:|---:|---:|---|
| SV20 | 0.04912 | 1.7875 | 1.001690 | Viable pilot; unresolved function-specific uncertainty qualification. Finite-endpoint comparisons remain untested. |
| KSC T2 | 0.00344 | 0.9625 | 1.000464 | Viable pilot; unresolved function-specific uncertainty qualification. Independent D/R finite-posterior comparison passed its declared test with empirical reference allowances. |
| M10 | 0.02708 | 1.2875 | 1.000766 | Viable pilot; first look lacked precision and the final look failed one uncertainty screen. Finite-endpoint comparisons remain untested. |

These are descriptive summaries of the already archived final looks, not new
posterior looks, revised equivalence tests or a ranking. The mean columns use
the first two standardized-parameter functions; the CDF column uses the six
prespecified marginal probabilities. All three retained B arms passed their
recorded numerical-health checks. None has a qualified discrepancy result.
Small point estimates do not settle the uncertainty issue or automatically
give these scopes the owner's LG1 acceptance. Their immediate issue is how
strongly the existing evidence supports inference, not a demonstrated failure
of their proposal/acceptance implementation. No automatic rerun follows.

## Shared gaps and their completion criteria

| Gap | Checked finding | What closes the gap |
|---|---|---|
| 1. Reliable uncertainty and useful reporting | The absolute batch-lag screen rejected 454/512 IID controls. Some dependent controls also exposed underestimated uncertainty, so deleting the check is insufficient. The replacement has only a limited Gaussian calibration. The new practical/formal distinction is in the plan, not executable reporting. | Qualify uncertainty for the functions and comparisons actually needed by the next declared stage; assess false vetoes and false equivalence. Implement separate practical judgment, formal evidence and next-action reporting without overwriting old verdicts. This is a shared test/reporting repair. |
| 2. Fit the experiment to its intended claim | A transferred 2,000-draw schedule cannot justify tight joint precision for thirteen functions/four comparisons. SV20 imposed one leapfrog step; LG1/KSC/M10 allowed at most two. | Budget for the next scope's practical or formal question before launch. Longer-trajectory experiments are needed for claims about their usefulness, not to invalidate or reopen accepted one-step pilots. Freeze any scope-specific trajectory/step/mass selection before retention. |
| 3. Faithful model-to-DCTR adapters | AR2's valid singular process covariance is rejected by the current full-rank consumer. M09's nonadditive/shared-noise law lacks demonstrated canonical support. M13's nearest-alias angle density does not represent the general wrapped simulator law. | Wire structural/generative laws through the actual DCTR value, flow, weights and analytical score; qualify references for the same law. AR2 must follow Chapter 18b. M13 needs a normalized wrapped likelihood with controlled alias truncation over its declared parameter range. Each changed scope needs its own checks and tuning. |
| 4. Untested finite-likelihood accuracy | LG1's measured D/R difference is accepted for its practical pilot and is closed. KSC T2 D/R passed its declared test with empirical numerical allowances. SV20 and M10 lack finite-target reference/comparison work; LG2 has no posterior comparison. | For a new finite-endpoint claim, qualify D and compare sampling error separately from model-posterior distortion. Repair a material approximation error if found. These missing tests do not block an independently valid B/R force test. More samples from D cannot remove D/R approximation error. |
| 5. Complete executable coverage | The register has 14 model families plus three control families, but the CLI admits only LG1, LG2, SV20, KSC, M10 and C01. Source discovery does not yet prove exhaustive coverage of repository model instances and variants. | Reconcile coverage; define concrete data, parameters, priors, transformations, horizons and references; implement and qualify missing consumers, tuning and retained stages. Preserve explicit unresolved scopes rather than counting a listed family as tested. |
| 6. Computational usefulness | No supported efficiency advantage or complete conditional heuristic comparison has been established. Good acceptance and small posterior differences do not answer the cost question. | When seeking an efficiency claim, compare reference-gradient HMC and constructed cheap alternatives: Gaussian random-walk Metropolis, a fitted frozen Gaussian independence proposal, and the one-step drift proposal. Evaluate important central/tail situations and total cost at comparable precision with replication. This is separate from practical accuracy of a bounded pilot. |
| 7. Replication, stress and scale | Current retained results are bounded pilots in restricted scopes. Wider parameterizations, longer horizons, higher dimensions and most independent data/bank confirmations remain open. | After viable small-scope results, use fresh data, random-design banks and chain groups, plus declared stress regimes. Retune and validate each changed scope. A small successful scope does not establish its larger family variant. |
| 8. Affordable execution and final synthesis | The existing retained schedule is not a qualified precision design. The LG2 pilot forecast exceeds its current family allocation. The master remains an incomplete implementation. | Produce scope-specific forecasts for the revised design, reconcile allocations within the campaign cap, execute ready work and preserve failures. Finish the coverage, posterior-accuracy, heuristic and cost tables with uncertainty. Review remains deferred until substantive evidence is ready. |

Gap 1 and a stage-appropriate precision/cost design must be addressed before a
new expensive run is interpreted as resolving posterior equivalence. This
does not require a trajectory sweep before every pilot. Adapter and reference
work can advance independently. A candidate failure is not an automatic veto
on another qualified scope or the next planned repair.

## Current coverage

Here B/R tests the DCTR proposal force with reference acceptance. C/D compares
finite-endpoint samples with the independently integrated finite posterior;
D/R measures finite-likelihood posterior error; C/R measures their combined
practical effect. If C uses the exact gradient of its own endpoint potential,
it is ordinary HMC for that approximate posterior. This classification does
not change any archived numerical result.

| Registered scope | Current evidence | Remaining work |
|---|---|---|
| M01: LG1 | Accepted practical pilot; original B/R, C/D, C/R INCONCLUSIVE and D/R DISCREPANCY preserved | None for this pilot. No further LG1-specific repair, retained extension or review is required. Wider LG scopes remain separate obligations. |
| M01: LG2 | Reference, consumer, controls and inference endpoint qualified; posterior comparisons NOT_RUN | Affordable precision design, public HMC tuning and retained experiments; finite-target comparisons. |
| M01: LG4/LG20/diagonal/full triangular parameters | Scopes unresolved | Concrete scope, adapter, reference and complete execution. |
| M02: actual SV20 | Healthy viable B pilot with small observed differences; formal B/R INCONCLUSIVE; C/D, D/R, C/R NOT_RUN | Resolve the scope's practical interpretation and shared uncertainty limitations before deciding whether more B sampling matters. Finite-target tests and SV40/real-data/other synthetic scopes remain open. |
| M03: KSC T2 | Healthy viable B/C pilots; formal B/R, C/D, C/R INCONCLUSIVE; D/R EQUIVALENT under empirical reference refinement | Resolve practical interpretation and uncertainty limitations; no automatic repeat of closed chains. Longer horizons, the distinct exact-transformed SV task and any efficiency/confirmation claim require their own evidence. |
| M04: generalized SV | Scopes unresolved | Both prior-mean projection and native two-state tasks require defined, qualified inference scopes. |
| M05: predator-prey | Scopes unresolved | Restricted PP20 and six-parameter targets, adapters, references and execution. |
| M06: Austria SIR | Scopes unresolved | Keep physical-clipped and preclip laws distinct; qualify their targets and references before inference. |
| M07: coupled nonlinear | Scope unresolved | Concrete target, adapter, reference, tuning and inference. |
| M08: structural | Canonical AR2 consumer fails on singular Q. Three existing TF structural reference tests passed; wrapper remains value-only. | Structural DCTR integration and analytical score, then reference/tuning/inference. Affine, nonlinear five-parameter and quadratic-completion variants remain open. |
| M09: accumulation/growth | Model B/C scopes unresolved; current canonical support for the required nonadditive/shared-innovation law is missing | General generative adapter, same-law references and full scoped execution. |
| M10: nonlinear score-study | Healthy viable B pilot with small observed differences; formal B/R INCONCLUSIVE; C/D, D/R, C/R NOT_RUN | Resolve practical interpretation and shared uncertainty limitations before another B run; finite-target reference and comparisons remain separate unfinished work. |
| M11: SSL-LSTM | Fixed and complexity-ladder scopes unresolved | Freeze concrete target/model/data versions and develop qualifying adapters/references before scaling. |
| M12: hard-bound macrofinancial | Scope unresolved | Define the concrete posterior, boundaries and qualifying reference; implement and test inference. |
| M13: range-bearing | Wrapped-model reference unqualified | Repair angle likelihood/reference across the declared noise-scale range, then tune and sample. |
| M14: BGS | Concrete upstream instance unresolved | Identify the actual model instance before defining an inference task. |
| C01: Gaussian controls | Anisotropic exact/half/rotational comparisons INCONCLUSIVE; zero field TUNING_REJECTED | Qualified precision experiment and conditional controls; d100 remains unresolved. These controls are not model-level DCTR evidence. |
| C02: non-Gaussian controls | Funnel/banana/mixture/varying-Hessian tasks unresolved | Implement and execute declared geometry/mode/tail controls with independent references. |
| C03: German-credit posterior | Data version and scope unresolved | Freeze data/target and qualify its independent reference and experiment. |

The existing INCONCLUSIVE outcomes remain closed. A new calibrated procedure
requires prospective evaluation; it must not be chosen to make old chains pass.

## Known repair paths and remaining empirical questions

We know concrete next steps for the identified implementation defects. We do
not yet know that every model will meet its accuracy and cost goals. Closing a
test gap means obtaining a trustworthy answer; that answer can be a failed
candidate with an identified limitation.

The reporting change is straightforward: expose implementation validity,
sampling health, practical judgment, formal comparison and next action as
separate fields. Preserve archived formal verdicts and the owner's LG1 pilot
acceptance. Regression checks should ensure that an unresolved confidence
interval is not reported as a numerical failure or as proof of equivalence.
The uncertainty repair has a concrete candidate: the revised diagnostic passed
all eight declared Gaussian calibration screens in 2,048 fresh replications.
That supports a starting point, not automatic transfer. The next assessment
must cover the actual function types, dependence, reference error, margins and
planned looks, including false vetoes and false equivalence. No existing chain
needs to be rerun merely to implement this reporting distinction.

For AR2, Chapter 18b gives the mathematical representation and
`TFStructuralStateSpace` already separates innovations from completed states.
Integrate over the previous-state/innovation law and enforce the lag relation
pointwise. Carry this representation through the canonical proposal, weighting,
reset and analytical score; changing only the failed Cholesky is insufficient.
Use deterministic-residual checks, Kalman value comparisons and independent
derivative checks before tuning and a small posterior pilot. Numerical
derivatives are diagnostic authorities only. M09 needs the same general
generative capability for its own nonadditive/shared-innovation law, followed
by a model-specific reference. The direction is known; a complete canonical
derivation and working implementation for these laws remain outstanding.

For M13, replace the reference observation density with the sum over wrapped
Gaussian aliases, evaluated stably with controlled truncation across the
declared noise scales. Check unit mass, periodicity, boundary behavior and
derivatives. The existing audit supplies narrow- and broad-noise controls.
This fixes the observation-law mismatch; it does not by itself supply the
latent-state marginal likelihood or a qualified parameter-posterior reference.

For the remaining finite-target comparisons, first construct and refine the
independent finite and model posteriors where feasible. Their difference can
identify material likelihood error without another HMC run. A material error
would trigger fresh scope-specific calibration of particle count and numerical
controls, followed by untouched validation. Increasing particles is a
candidate repair, not a convergence guarantee for the entire finite program.
For broader coverage, bind each registered family to concrete model, data,
parameter and horizon instances, then qualify its actual consumer and
independent reference. References for the larger and more complex scopes are
not yet established; this portion is more than filling in missing CLI options.

Efficiency is a separate empirical question. Measure component costs and
compare total cost at matched posterior precision with an appropriate simple
sampler and, where available, exact-gradient HMC. This determines whether
batching, reuse, trajectory selection or other measured bottlenecks merit
work. No speed advantage is established, and an accurate but slower method is
a possible completed finding. The LG2 allocation mismatch can be addressed by
a cheaper justified design or explicit reconciliation within the existing
campaign cap; it does not show that more total budget is already necessary.

The recommended sequence is shared reporting and scoped uncertainty checks,
AR2 structural integration, then M13/M09 and the next ready model scopes.
Small law and consumer checks precede expensive chains. Structural work can
advance while uncertainty calibration is prepared. LG1 remains closed.

## Budget and immediate priorities

The recorded remaining campaign balance is 26,810.767347 CPU-worker seconds
(7.447 hours) and 201,480.192604 GPU-wall seconds (55.967 hours). These are
accounting limits, not evidence that the entire outstanding campaign fits.
The current LG2 2,000-draw B/R pilot forecast alone is 15,327.988770 CPU seconds
and 16,120.324203 GPU-wall seconds. Its family balance is only 1,023.995559 CPU
seconds and 6,665.035729 GPU-wall seconds. Even funding that pilot would not
supply the missing precision justification.

Immediate priorities are to make the master report the revised distinctions,
qualify uncertainty for the next scope's actual question, and implement the
known model-law gaps. For AR2, wire the Chapter 18b structural measure and
analytical score through the canonical DCTR consumer, using the passing
structural reference fixtures as controls. Then choose the next retained run
from ready scopes using a stage-specific accuracy/precision/cost design. These
tasks do not require reopening LG1. Longer trajectories, broad replication and
conditional efficiency comparisons answer subsequent claims; they are not
blanket prerequisites for all implementation or pilot progress.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Keep LG1 closed | Owner accepts practical inference accuracy | No unresolved pilot implementation/mixing concern | Tighter formal comparisons remain unresolved or discrepant | Move to remaining scopes | Retrospective equivalence or a speed advantage |
| Retain SV20/KSC/M10 as viable pilots | Small archived mean/CDF differences; practical acceptance not yet settled | Recorded numerical health passes; functional uncertainty qualification unresolved | Precision and diagnostic calibration | Repair shared assessment and specify the next material question | Formal equivalence, automatic practical acceptance or a ranking |
| Repair AR2/M09/M13 and finish coverage | Required law support or concrete model tests missing | Actual implementation/reference blockers on affected scopes | Scope-specific behavior and cost | Structural/generative/wrapped-law integration and qualification | Rejection of surrogate HMC across models |

| Inference status | Finding |
|---|---|
| Hard veto screen | AR2 unsupported singular transition, M09 unsupported required noise law and M13 reference-law mismatch remain concrete blockers. Completed pilots have uncertainty limitations, not newly detected numerical failures. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Archived point differences, R-hat/ESS and cost summaries do not establish a ranking. |
| Default-readiness | No runtime default or formal admission changed. |
| Next evidence needed | Qualified uncertainty for each intended claim, missing model/finite-target comparisons and fair cost evidence if seeking an efficiency claim. |

Post-review check: the main risk is treating small point estimates as proof
despite uncertainty defects. Preserve the numerical summaries as descriptive
and retain their original formal outcomes. Conversely, a flawed uncertainty
screen cannot diagnose a defective sampler by itself. This revision reads
existing reports only, changes no samples or margins, and launches no experiment.

Evidence: [active checkpoint](artifacts/surrogate-hmc-all-models-20260919-review-tests/checkpoint.md),
[model register](artifacts/surrogate-hmc-all-models-20260922-master-reporting-03/model-register.json),
[scoped decisions](artifacts/surrogate-hmc-all-models-20260922-master-reporting-03/decision.json),
[code/literature audit](surrogate-hmc-algorithm-code-literature-audit-2026-09-22.md),
[SV20 archived report](artifacts/surrogate-hmc-all-models-20260922-sv20-inference-01/report.json),
[KSC archived report](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-inference-01/report.json),
[M10 archived report](artifacts/surrogate-hmc-all-models-20260922-m10-inference-02/P3-M10-inference/report.json),
and [master stages and completion criteria](surrogate-hmc-all-models-master-program-2026-09-18.md).
Executable scope choices were checked in
`bayesfilter/inference/surrogate_hmc_program.py::main`; inherited per-family
missing-field text is superseded by the checked scoped results where it is
stale. This inventory records no new numerical experiment or scientific ranking.
