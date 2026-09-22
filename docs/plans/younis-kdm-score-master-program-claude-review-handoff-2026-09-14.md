# Claude handoff: review of the KDM, proposal, and score research program

Date: 2026-09-14. Status: **PREPARED — REVIEW NOT RUN**.

## Assignment and research question

Please give an independent, skeptical review of the mathematics, motivation,
literature support, implementation claims, phase dependencies, and test coverage
of this research program. The question is whether the proposed experiments can
establish a more accurate marginal-likelihood score for LEDH/GenUT filtering,
and identify why a candidate succeeds or fails. Internal agreement between two
implementations is insufficient when both compute the wrong target.

Review the complete active program, including the earlier KDM/IWSG and
control-variate arguments on which the additions depend. Give concrete repairs
and the smallest discriminating checks. Distinguish mathematical error,
implementation error, incomplete implementation, insufficient evidence, and a
scientifically reasonable hypothesis that has not yet been tested.

Claude is a **read-only reviewer**. Do not edit files, execute commands, launch
agents, run experiments, or change the program's scope. Return findings to
Codex, which remains responsible for repairs, checks, and saving the review.
This handoff authorizes no experiment or default-policy change.

## Current scope and the user's unresolved questions

The active work has three connected branches:

1. KDM/IWSG: identify exactly which expectation is differentiated, investigate
   whether replacing proposal moment calculations is useful, and evaluate
   properly defined control variates or calibrated combinations of imperfect
   scores.
2. Proposal quality: compare UKF, KDM filtering moments, LEDH/GenUT moments,
   structured Gaussian quadrature filtering (SGQF), twisting, and iterated
   auxiliary particle filtering (iAPF). Keep their different roles explicit.
3. Directional validation and calibration: three-point central differences,
   five-point fourth-order differences, and an eleven-point cubic regression,
   using a fixed positive step-size ladder, coupled likelihood evaluations,
   independent directions, and oracle-based bias–variance measurements.

The user wants to know how filtering means and covariances evolve with the
observations; how KDM differs from the UKF already used in LEDH; where OT enters;
whether better covariance improves the score; and whether noisy directional
checks can support useful bias indicators or score combinations. The document
must answer these questions, rather than merely name additional methods.

Regular-transition smoothing/Fisher-score work and the degenerate-transition
DSGE program belong to other agents. Rhee–Glynn/Jacob–Lindsten–Schön is outside
the present active investigation. Review their appearances in the master plan
for scope and dependency errors, but do not reopen those research programs.
In particular, do not make a blanket claim about all smoothing on singular
models: identify the transition-density assumptions of the particular method
where a boundary statement is needed.

## Checkout, sources, and evidence status

Repository: `/home/chakwong/BayesFilter`; branch: `surrogate-hmc`;
HEAD when inspected: `14a292098f35b6de450ffca29131ea34f4a4d8d7`.
The working tree contains unrelated work. The master plan and this handoff are
untracked; the manuscript directory is ignored by Git. A clean diff check
therefore does not validate their contents. Do not stage, revert, or commit.

The [input manifest](artifacts/younis-score-review-handoff-20260914-01/input-manifest.json)
records the inspected source hashes and preserves the existing build logs.
The [previous handoff](artifacts/younis-score-review-handoff-20260914-01/previous-handoff.md)
is historical context only; its wider scope and old line references are
superseded by this memo. Hashes establish which files were inspected, not their
correctness. A subsequent source change requires checking the affected review
conclusion again, not restarting every unrelated review.

| Item | Actual state at handoff |
| --- | --- |
| [Master program](younis-kdm-score-master-program-2026-09-14.md) | 1,095 lines; proposal study and Phases 4B/4C are documented. The new harnesses and experiments are not established by this document. |
| [Main LaTeX source](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex) | 3,483 lines; new proposal and directional-difference sections need substantive review. |
| [Rendered manuscript](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf) | Existing 50-page PDF; the additions appear around pages 20–23. Compilation is not evidence of mathematical or typographic correctness. |
| [Bibliography](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.bib) | Local references exist; correctness and claim-level support still need checking. |
| SGQF implementation | A standalone approximate filtering implementation exists. Its presence does not prove that an LEDH proposal consumer uses its moments. |
| New proposal/FD studies | Proposed. No completed stochastic comparison, bias–variance calibration campaign, or score-improvement finding is supplied here. |
| Independent review | No Claude review or MathDevMCP audit of these additions has been run in this handoff task. Earlier audits cannot certify new text. |

## How to conduct a thorough review without another stall

This memo is an index for successive bounded reviews, not an instruction to load
all cited files at once. The repository's `AGENTS.md`, under “Claude Review
Prompt Shape,” requires a first review of one exact path and one question.
Start with the following question because it tests an assumption that could
invalidate the proposed experiment:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
/home/chakwong/BayesFilter/docs/plans/younis-kdm-score-master-program-2026-09-14.md
Read only “Phase 4C: fixed symmetric directional finite-difference ladder”
(lines 775–840 in the recorded snapshot).
Do not edit, run commands, launch agents, or review the whole repo.
Question: Does this phase correctly distinguish deterministic stencil accuracy
from finite-particle marginal-score accuracy when defining its pass/fail tests?
Give precise findings and the smallest necessary repair. If information is
missing, request the next exact path and section; do not infer it.
End with VERDICT: AGREE or VERDICT: REVISE.
```

Continue through the stages below, one substantive question and one primary
source range at a time. Read a cited dependency only when needed to answer the
question. Section names and symbols are the stable anchors; line numbers below
refer to the manifest snapshot and may move. Do not treat an unread dependency
as checked. If a review stalls, use the bounded recovery process in
`/home/chakwong/.codex/skills/claude-readonly-review-probe/SKILL.md`;
Codex supervises any health probe or relaunch. The applicable one-path first
prompt rule takes precedence over older broad packet instructions.

Reviewer unavailability is a recorded limitation, not a new approval ceremony.
A material mathematical or numerical defect remains a real blocker for the
claim or experiment that depends on it.

## Findings already identified: verify independently

These are reasons to scrutinize the draft, not a substitute for your review.
Do not assume that repairing them exhausts the problem.

| Priority and anchor | Concern to resolve |
| --- | --- |
| Mathematical error: LaTeX `sec:fd-ladder`, around line 1550 | “Four continuous derivatives” does not generally imply fourth-order error for the cubic derivative estimate. Derive a sufficient smoothness condition and a remainder bound. Check a symmetric function such as `sign(x) * abs(x)^(4+alpha)`, `0<alpha<1`, at zero as a counterexample to the stated condition. |
| Experimental validity: master Phase 4C | Error against the exact Kalman score from a noisy finite-particle log likelihood need not have slope two or four. Particle/log bias, randomness, and roundoff can obscure truncation order. Separate deterministic stencil tests from stochastic score-MSE experiments before using observed slopes as a veto. |
| Target clarity: LaTeX `eq:fd-three-point` through `eq:fd-cubic-fit` | Exact `ell`, a fixed-randomness particle log likelihood, and its expectation are mixed. Each displayed derivative and error order needs an explicit target. |
| Incomplete derivation: LaTeX `eq:sgqf-moments`, lines 1475–1503 | Projecting a quadrature cloud is not a complete filtering update. The displayed quantities omit explicit process/measurement noise, observation conditioning, and the moment lifecycle. The weights with tildes are undefined. Compare with the actual SGQF implementation. |
| Rendering defect: LaTeX line 1483 | The source contains `x_ell^+,qquad` without the command's backslash, so the equation prints “qquad.” Fixing this is necessary even though LaTeX compiles. |
| Unsupported implementation implication: same SGQF subsection | An eager standalone filter has not been shown to be an integrated LEDH moment provider. Trace the proposed consumer, covariance state, and derivative path before calling that integration implemented. |
| Statistical target: master Phase 4B and control-variate discussion | An exact model score is generally not the mean of a finite-N noisy finite-difference control. Specify the control's actual mean or the bias introduced by approximate centering. Also assess independently unbiased random centering instead of insisting that a deterministic known mean is the only possibility. |
| Proposal derivation: LaTeX proposal-quality section | A conditional importance-weight identity does not establish the full-horizon likelihood properties of a filter with deterministic OT reset. State which particle law and reset support each conclusion. |
| Derivative convention: twisting/iAPF discussion | Holding fitted twist coefficients fixed is different from freezing the twist's dependence on its state argument. “Same score estimator” must not conceal omitted derivatives of new proposal moments, weights, or state recursions. |
| Research alignment: master Phase 2 and proposal metrics | ESS, covariance accuracy, and normalizer variance can screen or explain; they cannot silently replace marginal-score MSE as the final improvement criterion. |
| Dependencies and scope: master phase/gate structure | Phase 8 tuning appears after quality comparisons; Phases 4B/4C are absent from parts of the gate graph; a universal finite-program derivative gate may be inappropriate for an expectation-gradient estimator. Deferred smoothing/DSGE work must not become a prerequisite for the active branch. |
| Statistical qualification: CRN and multipoint regression prose | Common random numbers do not guarantee positive covariance or variance reduction. More points and higher order do not guarantee lower MSE or cheaper full-score estimation. Measure the relevant covariance and cost. |

## Stage 1 — motivation, estimands, and the KDM/IWSG argument

Primary ranges: master “Exact model score,” “Finite-program derivative,”
“KDM expectation gradient,” and “Unnormalised pair/ratio” (approximately
lines 196–266); then the LaTeX IWSG discussion (lines 402–578) and
control-variate subsection around line 2573 when needed.

Check that the document distinguishes at least:

- `s(theta) = grad log Z(theta)`, the marginal-likelihood score;
- `grad log Zhat_N(theta; xi)`, a derivative of a specified finite value
  program, where that derivative exists;
- `grad E[log Zhat_N(theta; Xi)]`, including the dependence of the simulation
  law when it depends on the differentiated parameter;
- `grad E_m_phi[F(z, phi)]`, the expectation-gradient target of IWSG;
- an unnormalised derivative estimate, a ratio estimate, a conditional
  one-step likelihood, and a full-horizon likelihood.

Reproduce the user's IWSG identity fairly. Check differentiation under the
integral, domination/integrability, support, the derivative of `F`, and the
mixture score. Explain why a detached proposal at the current parameter can
produce an unbiased gradient of the stated expectation. Trace all earlier
random laws when extending the argument through time. The limitation is not
adequately expressed as “conditional unbiased, marginal biased”: an unbiased
expectation gradient can concern a whole finite-horizon simulation objective,
while that objective still differs from `log Z`.

Check the bandwidth-dependent KDM target and distinguish filtering from
smoothing. The fact that a cited paper is about a smoother does not make every
KDM filtering operation a smoothing score method. Conversely, continuous
resampling and unbiased IWSG do not by themselves prove unbiased state
estimates, an unbiased original-model likelihood, or an unbiased model score.

For the propositions, require explicit random variables, conditioning,
normalization, parameter dependence, and sufficient assumptions. Identify the
exact assertion proved by each proof. Check whether positivity, interchange of
limits/derivatives/expectations, and integrability are justified, rather than
inferred from numerical finiteness.

Review proposed combinations of two imperfect scores under squared-error
loss, including bias, variance, and cross-covariance. For a construction
`S - B(D - mu_D)`, derive the mean and MSE under the actual centering scheme.
Separate a genuine centered control variate from empirically calibrated bias
correction or shrinkage. An independent unbiased estimate of `mu_D` can preserve
centering in expectation but adds noise; reused or fitted coefficients can
introduce dependence. Require sample splitting/cross-fitting where needed.
Distinguish averaging log likelihoods from taking the log of an average of
independent likelihood estimates. Preserve the useful empirical question of
whether an approximate diagnostic predicts score error on heldout oracle
cases, without claiming an oracle-free bias guarantee.

Expected result: a corrected target map and proposition-by-proposition verdict,
plus an explanation of how the proposed control is generated, centered, and
used in the actual LEDH score calculation.

## Stage 2 — proposal preservation, twisting, and iAPF

Primary range: LaTeX “Proposal-quality branch: twisting, iAPF, covariance, and
SGQF,” lines 1378–1474; master Phase 2, especially the twisting/iAPF paragraphs.
Use the earlier physical-mixture proposal section, lines 1175–1377, only for
specific dependencies.

Re-derive the model-corrected marginal weight and the joint ancestor/state
weight under their respective sampling laws. Check factors of `N`, normalized
versus unnormalized weights, ancestor probabilities, mixture denominators,
flow Jacobians, support, and normalizing constants. Verify that an evaluable
proposal density is actually available; a list of moments alone is insufficient.
State precisely which one-step and multi-step expectation identities follow.

For positive twists `psi_t`, distinguish an auxiliary lookahead weight, a
normalized twisted transition, the complete twisted Feynman–Kac construction,
and an approximate learned future-likelihood function. Derive the cancellation
of the relevant normalizers and show why the original target is preserved
when the correction is correct. Do not label an arbitrary lookahead heuristic
as an implementation of a paper's theorem.

For iAPF, specify pilot simulations, fitted function family, recursion and
terminal condition, fitting loss, iteration schedule, stopping rule, and the
final corrected filtering pass. Explain what can depend on the observed
series and what is held out to prevent performance-selection leakage. Fitting
a proposal conditional on the target series is different from tuning on the
untouched evaluation result; “disjoint pilot data” must not conflate these.

Make the relationship to UKF precise. A covariance provider, an ancestor
selection rule, and a future-observation twist act at different points in the
filter. Replacing UKF is not a complete description of twisting or iAPF.
Freeze offline choices consistently for the finite objective being scored,
while retaining derivatives through state arguments and declared model
parameters. Identify any missing analytical derivative recursions.

For OT, name its exact role in each candidate: transporting the filtering
cloud, resetting retained covariance state, coupling perturbed runs, or
constructing a proposal. These roles are not interchangeable. Determine which
canonical reset and weight identities survive each change.

Expected result: an explicit sampling/weight/derivative specification for each
proposal arm, with unsupported likelihood or source-faithfulness claims marked.

## Stage 3 — filtering moment lifecycle and SGQF integration

Primary range: LaTeX lines 1456–1524; then
`bayesfilter/nonlinear/fixed_sgqf_tf.py`, beginning with the specific symbols
`tf_fixed_sgqf_cloud` and `tf_fixed_sgqf_filter`.

Require the full time-indexed recursion for each provider's state: incoming
mean/covariance or mixture components and probabilities; transition prediction;
observation prediction and conditioning; filtered mean/covariance; and the
state handed to the next time step. Show where the actual particle, previous
filtering weights, model parameters, and observation enter. Explain how
per-particle covariance differs from a global filtering covariance.

Distinguish process covariance, predicted state covariance, filtered covariance,
local flow covariance, and proposal covariance. For a mixture, include both
within-component and between-component covariance. Define how component
probabilities and covariance states are updated or transported after resampling.
A recursion without the appropriate observation dependence must not be called
a filtering recursion.

Useful inspected SGQF anchors are:

| Symbol or operation | Snapshot anchor in `fixed_sgqf_tf.py` |
| --- | --- |
| Runtime classification | `FIXED_SGQF_RUNTIME_MODE`, line 17 |
| Sparse quadrature cloud | `tf_fixed_sgqf_cloud`, around line 713 |
| Weighted moment helpers | Around lines 873–881 |
| Prediction and addition of process covariance | Around lines 1074–1079 |
| Observation moments, measurement covariance, cross-covariance | Around lines 1113–1127 |
| Innovation solve and filtering update | Around lines 1159–1173 |

Check signed quadrature weights separately from nonnegative mixture
probabilities. Signed weights cannot automatically define a sampling mixture
and can produce invalid covariance estimates. Require scale-aware validity
checks and an explicit policy for any numerics-altering regularization.

Trace every claim-bearing consumer to the actual provider. Check return shapes,
batch dimensions, dtype, device, persistent component state, and analytical
sensitivity support. The current eager/Python-branch SGQF route is not evidence
of a GPU/XLA LEDH integration. Request a wiring or parity test of that endpoint;
without one, mark integration **not checked** or **not implemented**, as the
source warrants. Do not accept a standalone function-existence argument.

Assess high-dimensional feasibility through actual quadrature point counts,
level/dimension growth, memory, and cost. Require a compatible-factor matrix
before crossing every covariance with every proposal. Explain why improved
moment accuracy might help weights but still fail to improve the score.

Expected result: a self-contained moment recursion and a concrete integration
gap list, with proposal screening separated from downstream score evidence.

## Stage 4 — finite differences, regression, coupling, and controls

Primary ranges: LaTeX `sec:fd-ladder`, lines 1525–1574; master Phases 4B/4C,
lines 699–840. Review the proposed experiment after the first bounded finding.

Derive each estimator as `D_h = sum_k c_k L(theta + k h v) / h`. Verify its
moment cancellations, leading error term, and sufficient smoothness. Distinguish
a five-point fourth-order stencil from a five-point cubic least-squares fit;
the latter is not automatically the same coefficient vector. Explain why a
symmetric linear or quadratic fit retains second-order bias. For the eleven
points, derive the cubic-fit derivative weights and their error constant.

Define `b_N(theta) = E[L_N(theta)] - ell(theta)`. Where the needed derivatives
and integrability exist, decompose expected directional error into truncation
of `ell`, the stencil applied to `b_N`, and any remaining target mismatch.
Do not imply that `h -> 0` removes finite-particle log bias. Check the additional
assumptions needed for a fixed-randomness resampling program to be smooth.

Use the full covariance formula
`Var(D_h) = c^T Sigma(h) c / h^2`, not just independent-noise intuition.
For two points, report plus/minus covariance; for regression, report the
relevant joint covariance or an empirical replicate variance. Coupling can
reduce variance only when its covariance structure helps the chosen contrast.
An antithetic parameter perturbation is different from antithetic random noise.
Explain the risk from random step sizes near zero conditional on how their
likelihood noise scales; do not assert universal variance divergence under
perfectly smooth common randomness.

Require the following design checks:

- Deterministic analytic-function and exact-Kalman-likelihood checks establish
  stencil algebra/order over a pre-roundoff range. Stochastic particle runs
  separately estimate score bias, variance, and MSE across `h`, `N`, and horizon.
- Fit a scaled coordinate such as `k` with QR/SVD or a justified stable solver,
  then convert the slope to parameter units. Small `h` must not silently make
  raw polynomial normal equations ill-conditioned.
- Record parameter coordinates, scaling, domain boundaries, support validity,
  and representably distinct perturbations. A unit direction in arbitrary
  mixed units is not a complete step-size policy.
- Separate calibration, validation, and untouched evaluation. Derive or
  calibrate `h0`; the displayed dyadic ladder is a hypothesis, not an admitted
  default. Freeze all selected controls before final comparisons.
- Compare equal compute and disclose perturbation span: at common spacing the
  central, five-point, and eleven-point designs reach `h`, `2h`, and `5h`.
  Count actual likelihood calls, shared points, pilot costs, and cached values.
  A symmetric center can have zero derivative weight; do not charge or reuse
  it inconsistently across methods.
- For `V^T s = d`, require full row rank of the direction matrix `V`, sufficient
  independent directions, and conditioning diagnostics. Assess correlated
  directional error and whether weighted least squares is justified. Fewer
  directions provide directional evidence, not a reconstructed full score.
- Test whether a directional discrepancy predicts actual score error on
  heldout oracle cases. Evaluate the resulting corrected/combined score by its
  heldout MSE, including coefficient-estimation cost and uncertainty. Failure
  to obtain a certified exact control does not invalidate this empirical arm.

Expected result: corrected propositions and an executable experimental design
whose success or failure answers the score question rather than a noise-floor
or step-scaling accident.

## Stage 5 — prerequisites, test coverage, and experiment fairness

Primary source: the master plan's phase definitions and gate structure.
Audit the entire active dependency graph, including Phases 4B/4C. Check
architecture, registries, target labels, artifacts, and stopping rules against
the mathematical conclusions above. The following dependencies are mandatory
questions for the review, not claims that the corresponding work exists:

| Work item | Required before a meaningful downstream comparison |
| --- | --- |
| Target and oracle registry | Explicit estimator targets; exact scalar/multivariate Kalman values and scores; at least one justified nonlinear reference; prepared data and parameter-coordinate definitions. |
| Coupled likelihood evaluator | Correct marginals at every perturbation; deterministic replay policy; support/finite-value guards; actual covariance and cost reporting. |
| Proposal adapters | Evaluable normalized proposal or exact joint correction; filtering moment state; ancestor law; flow/reset identity; negative tests for omitted factors. |
| KDM/IWSG estimator | Mixture score and direct derivative terms; support/bandwidth assumptions; expectation-gradient tests distinct from pathwise tests. |
| FD/regression estimator | Stencil algebra, smoothness/order checks, stable regression, valid perturbations, rank and conditioning checks. |
| Tuning and selection | Scope-specific calibration before quality claims; fixed controls; disjoint validation and untouched evaluation; complete tuning-cost accounting. |
| Score combination | Centering target and coefficient rule; dependence control; heldout prediction of error; downstream MSE evaluation. |
| Proposal/score study | Validated endpoints and applicable analytical score recursion; compatible factors; replicate policy; paired uncertainty; total compute/attempt cap and stop conditions. |

For every proposed test, identify the consumer endpoint, property asserted,
reference/oracle, failure cases, artifact, and downstream claim it can support.
Use explicit statuses: **existing and inspected**, **exists but not inspected**,
**planned**, **missing**, or **not applicable**. Do not count a test title as
coverage or invent test executions.

At minimum, cover exact Gaussian limits; non-Gaussian/multimodal cases;
informative and weak observations; covariance conditioning; signed quadrature
weights; proposal support failures; zero/near-zero bandwidth or invalid steps;
resampling boundaries; insufficient direction rank; and long-horizon or
high-dimensional cost where those claims are made. Define expected failure
behavior as well as healthy-case parity. Check that the model-specific
capabilities needed by an arm are explicit and that deferred singular-model
work is not silently imported into the active test requirement.

Build a small practitioner baseline ladder: the unadjusted current analytical
score, central differences, an available Gaussian/UKF approximation, bootstrap
or simple adapted proposals where applicable, and the exact oracle as reference.
Use compatible targets and identical observations; disclose particle versus
compute matching. Evaluate conditional regimes separately. A complex method
losing to a cheap relevant baseline is a promotion veto, not a footnote. These
baselines are falsification checks; do not tune on the untouched comparisons.

The primary improvement criterion must be declared score error with uncertainty
against an appropriate oracle. Likelihood variance, ESS, moment error, weight
tails, derivative parity, and speed have explicitly assigned screening,
explanatory, repair, or veto roles. They must not change roles after results
arrive. Distinguish candidate rejection from failure of the harness or research
direction; proceed to a planned repair unless a true continuation veto fires.

Audit material defaults: bandwidth, covariance regularization, twist family,
lookahead length, pilot iterations, quadrature level, OT controls, particle
counts, horizon, stencil span, step scales, direction design, seeds, precision,
and stopping thresholds. Each needs provenance, rationale, failure mode, and
an early diagnostic. Unexamined inherited values are hypotheses, not defaults.

The master contains large per-cell replicate and particle ladders but does not
yet establish a summed campaign resource budget. Before a serious launch,
require a pilot cost estimate, staged cells, total wall-time/compute and attempt
caps, versioned outputs, and explicit stop conditions. Do not substitute
unbounded full-factorial execution for systematic investigation.

## Stage 6 — implementation and source-faithfulness checks

Request only the exact implementation or test needed by a disputed claim.
Potential starting points, **not instructions for a bulk read**, are:

- `bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py` and its corresponding
  `tests/highdim/test_ledh_younis_kdm_integrated_tf.py` for integrated KDM claims;
- `bayesfilter/highdim/ledh_younis_kdm_tf.py` and
  `tests/highdim/test_ledh_younis_kdm_tf.py` for local mixture-gradient claims;
- `bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py` and
  `tests/highdim/test_ledh_younis_kdm_resampling_tf.py` for resampling claims;
- `bayesfilter/nonlinear/fixed_sgqf_tf.py` and a specific test from
  `tests/test_fixed_sgqf_tf.py`, `tests/test_fixed_sgqf_integration_tf.py`, or
  `tests/test_fixed_sgqf_scores_tf.py` for SGQF claims.

These paths exist; their mere enumeration is not a call-chain audit or a test
result. No new FD-ladder endpoint or result has been established by this
handoff. Ask Codex to locate a runnable endpoint if the manuscript implies one.

Respect the active repository policies. Pre-2026-08-21 LEDH results cannot
supply new baselines or tuning evidence. Canonical LEDH requires the documented
Li Algorithm 1 lifecycle, Contract E–Chol reset, GenUT/trust-region correction,
and analytical recursive score. Autodiff and finite differences are diagnostic
comparators. Any research alternative must have an explicit identity and must
not quietly acquire canonical/default/HMC status. Check source-cloud moment
and weight derivatives through reset, not only transported-cloud derivatives.

GPU/TFP, batching, XLA, memory growth, exact-divisor transport chunks, and
per-scope offline tuning must be satisfied by the actual claimed consumer.
A standalone eager reference does not satisfy those requirements. CPU or NumPy
checks may be independent diagnostics only. No HMC consumer changes are in
scope; if a proposed conclusion depends on one, identify the missing interface
review rather than advising a new HMC route here.

For paper-dependent claims, read technical methods, the relevant theory and
appendix, and original-author code when it exists. Local tractable paper copies
are available under `.localresources/papers/`, using these stems:

- `younis-sudderth-2024-learning-to-be-smooth` — especially Section 2.3;
- `younis-sudderth-2023-long-range-tracking` — the IWSG derivation;
- `whiteley2014-twisted-particle-filters` — exact twisted construction;
- `guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter` — iAPF;
- `sen-thiery-jasra-2017-coupling-particle-filter-trajectories` — if coupling
  claims rely on that paper rather than a locally derived mechanism.

Each has a local `.txt`/`.pdf` copy. Request the needed section separately;
do not infer source-faithfulness from a bibliography entry. Author-code anchors
have not been verified by this handoff. Separate a source theorem, a derivation
for this filter, and an empirical extension. Check that citations support the
actual formula/claim and that no result for a different random law is imported.

## Stage 7 — evidence, manuscript readability, and mathematical audit readiness

The preceding drafting session produced a successful PDF build using
`pdflatex`/BibTeX after `latexmk` was unavailable. Preserved logs are in
[the evidence directory](artifacts/younis-score-review-handoff-20260914-01/).
The build was not rerun in this handoff task. It establishes buildability of
that snapshot, not mathematical correctness. The surviving “qquad” defect
shows that the earlier visual check was incomplete.

The previous session also reported an ad hoc NumPy diagnostic using
`f(x)=exp(0.3*x)+0.2*x^5` at zero with a halving ladder. Reported slopes were
about 2.17, 4.00, and 4.00 for central, five-point, and cubic-fit estimates.
No saved executable or structured numerical result for that check was
identified. Treat those numbers as conversational diagnostic history only.
They do not establish Kalman-score, particle-filter, stochastic-order,
variance-reduction, or full-score performance. Require reproducible artifacts
for any corresponding claim. No experiments were run to create this handoff.

Read the rendered additions alongside the source. Require enough explanation
for the user to reconstruct one filtering time step and one full directional
score evaluation, including what is sampled, conditioned on, updated, and
held fixed. Define symbols before use; distinguish component probabilities,
quadrature weights, filtering weights, and regression coefficients. Correct
stale lists, unexplained equations, malformed Markdown/LaTeX, and inconsistent
notation. Keep operational review instructions out of reader-facing theory.

For MathDevMCP readiness, return a proposition ledger: statement, assumptions,
proof dependencies, exact target, proof verdict, and a counterexample or repair
for failures. Identify any missing proposition required by the algorithm.
Do not claim a MathDevMCP pass without an actual audit record bound to the
reviewed version. A proposed proof, Codex check, Claude verdict, and MathDevMCP
result are different evidence sources. Codex can arrange the mathematical
audit separately after incorporating findings.

## Required review output

Return the review as text for Codex to save under
`docs/reviews/younis-score-proposal-fd-review-2026-09-14.md`; do not write it
yourself. Lead with the most serious scientific defects. Include:

1. Findings with severity, exact source anchor, claimed target, actual quantity
   or behavior, mathematical/source reason, consequence, concrete repair, and
   the check that would resolve it. Label unchecked claims plainly.
2. A target/proposition ledger, a corrected active dependency graph, and a
   coverage table separating existing tests from planned and missing tests.
3. A decision table with decision, primary criterion status, veto status, main
   uncertainty, next justified action, and what is not established. Separate
   readiness to implement, readiness to run a bounded experiment, and evidence
   of score improvement.
4. An inference-status table covering hard vetoes, statistically supported
   ranking, descriptive-only differences, default-readiness, and next evidence
   needed. With the currently supplied evidence, no stochastic ranking has
   been established; assess any stronger claim against actual artifacts.
5. The strongest alternative explanation for any apparent success or failure,
   and a prioritized repair sequence that preserves the user's research
   question. List deferred/out-of-scope items separately from blockers.

Use `BLOCK` severity only for a material defect that invalidates a specified
claim or dependent action, not missing ceremonial paperwork. A whole-program
`AGREE` requires the active mathematics, dependencies, and evidence statements
to withstand review; an unread component must remain explicitly unchecked.
A stage-level agreement covers only that stage. End each bounded review, and
the final synthesis, with exactly one of:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```
