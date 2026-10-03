# HMC burn-in assessment and Monte Carlo precision repair

Baseline: `5139f151237e9764ebf34b4e8cf6ceee1e74a38d`, 2026-09-15.
Status: source review and proposed repair; numerical defaults and runtime code
are unchanged. This extends the
[whole-procedure review](bayesfilter-hmc-tuning-followup-review-2026-09-15.md).

The user asks whether BayesFilter determines sufficient burn-in, whether it
estimates Monte Carlo standard errors (MCSE), and how to bring the procedure
into line with established Bayesian practice. The repair must preserve the
earlier decision: R-hat and posterior precision do not qualify or remove an
HMC tuning candidate.

## Review contract and skeptical audit

Inspect the actual public candidate execution, posterior bridge, both sequential
controllers, diagnostic implementations, guide, and relevant tests. Compare
mathematical quantities with primary sources and official implementations,
not just similarly named functions or existing tests. Record separate findings
for adaptation, equilibration, and estimation precision.

One bounded diagnostic may compare the existing R-hat implementations with an
independent evaluation of the published formula on fixed arrays. The pass
criterion is agreement with that formula, including ranks, folding, splitting,
and the square root. Arrays are arithmetic counterexamples, not posterior
samples or evidence about sampler performance. Use the existing `tfgpu`
interpreter with GPUs intentionally hidden and a convenience ceiling of
120 CPU wall seconds for this diagnostic. No sampler, training, default-policy
experiment, package installation, or GPU run is part of this review.
Preserve the script, result, source hashes, environment, and timing under
`docs/plans/artifacts/hmc-warmup-precision-review-2026-09-15/`.

Skeptical audit before execution: fixed transition counts do not establish
equilibration; a small R-hat cannot certify undiscovered modes; MCSE requires
estimand-specific moment/mixing assumptions; tests copying a wrong formula
cannot establish correctness. Adjacent epochs of the same chain are dependent.
Repeated diagnostic looks do not automatically inherit fixed-time coverage.
Local CPU arithmetic can test an estimator but cannot qualify the GPU sampler
or establish a target-specific burn-in count. The diagnostic observes these
boundaries, uses no stochastic ranking, and can proceed.

## What already exists

| Part | Actual implementation | Limitation |
| --- | --- | --- |
| Ordinary geometry preparation | `hmc_preparation.py:73` invokes operational windowed adaptation. The standard preparation allocation is dimension based (`hmc_kernel_tuning.py:16815`), with distinct smoke/diagnostic presets. | An adaptation allocation is not a posterior burn-in estimate. |
| Candidate measurement and verification | `HMCCandidateExecutionConfig.num_warmup_steps` declares discarded steps; their numerical health is checked. | Counts are fixed. R-hat is reporting-only, correctly for the current tuning contract. |
| Shared posterior warmup | `neutra_hmc.py:171` and `:643`: default minimum 2,000 transitions/chain, latest 1,000-transition window, threshold 1.05, cap 10,000. The first passing healthy check ends warmup. | An inherited operational heuristic, not a target-specific estimate or calibrated error bound. |
| Warmup reporting | `neutra_hmc.py:856`: counts, checks, cap, health, archived warmup and exclusion from posterior summaries. | Reporting is useful but only reaches consumers using that controller; it relies on the diagnostic implementation below. |
| Retained stopping | `neutra_hmc.py:764` defaults to cumulative R-hat; a callback can replace the diagnostic. The archived controller has optional ESS checks (`:2934`). | No uniform, mandatory estimand-specific precision assessment. Callback replacement can omit even the built-in R-hat check. |
| Mean MCSE | `hmc_posterior_diagnostics.py:307` computes original-scale mean ESS and SD/sqrt(ESS), pooled and per chain. | MCSE exists; it is not lugsail and is not integrated into all posterior routes. Its ESS is not identical to Stan's current estimator. |
| Other diagnostics | The same module has initialization-memory and adjacent-epoch drift diagnostics; `predictive_equivalence.py:1477` has batch-means long-run covariance for a separate predictive procedure. | Useful components, not a shared burn-in/precision protocol. Adjacent-epoch MCSE arithmetic omits covariance between the epochs. |
| Lugsail and precision stopping | No lugsail implementation or general fixed-width posterior stopping procedure was found in the inspected runtime/test tree. | Add these through a shared diagnostic layer, not a third posterior controller. |

There are two sequential configurations behind the same public facade
(`neutra_hmc.py:1767`), with different defaults and output contracts. The prior
review also demonstrated that the candidate-to-sequential bridge changes target
shape/telemetry requirements and does not validate all derived seeds. Those
defects must be repaired before extending this route.

## Diagnostic correctness findings

1. `hmc_posterior_diagnostics.py:152` uses `(rank - 3/8)/(S - 1/4)`.
   Vehtari et al. equation (14) and Stan posterior's `backtransform_ranks`
   use `(rank - 3/8)/(S + 1/4)`. Its test at
   `tests/test_hmc_posterior_diagnostics.py:48` repeats the wrong denominator.
   `hmc_convergence.py:389` already has the correct denominator.
2. `hmc_convergence.py:399` passes rank-normalized split chains directly to
   TFP's `potential_scale_reduction`, and reports its return as modern R-hat.
   The installed TFP implementation (`diagnostic.py:567`) returns
   `((m+1)/m) * variance_plus/W - (n-1)/(m*n)`, where `m` is the split-chain
   count and `n` the draws per split chain. The published modern diagnostic is
   `sqrt(variance_plus/W)`. These are different quantities; simply applying a
   square root to TFP's return would still retain its extra correction.
   The posterior module implements the square-root formula, but has finding 1.
3. Both local ESS helpers reproduce TFP-style positive-pair truncation and lag
   weighting. Stan posterior additionally implements its initial monotone
   sequence, antithetic refinement and stability bound. The existing code
   should not be described as identical to Stan. This difference alone does
   not prove the TFP estimator invalid; its finite-sample behavior and the
   consistency between public reports need explicit validation.
4. `epoch_drift_statistics` uses `sqrt(MCSE_current^2 + MCSE_previous^2)`
   for consecutive epochs. Consecutive epochs are not independent merely
   because they do not overlap. This is a descriptive drift statistic, not a
   calibrated independent-window z test. Do not promote its existing numeric
   cutoff to a universal burn-in rule.

The independent fixed-array check completed in 3.009 seconds on CPU with
Python 3.13.13, TensorFlow 2.20.0 and TFP 0.25.0. The arrays have 64 draws,
four chains and two columns; they are deterministic trigonometric/modular
fixtures, including ties. They are not draws from an HMC run.

| Column | Published modern R-hat | Shared controller report | Posterior diagnostics report |
| --- | --- | --- | --- |
| Continuous ranks | 1.126974588 | 1.307736936 | 1.128368714 |
| Tied ranks | 1.021793878 | 1.053476821 | 1.022244607 |

The shared rank transform agrees with an independent SciPy evaluation to
2.23e-16, but its reported statistic agrees with the distinct TFP ratio.
The posterior module has the right square-root structure but the wrong
rank transform. This confirms both mismatches without estimating their impact
on any target's warmup length. Artifacts:
[script](artifacts/hmc-warmup-precision-review-2026-09-15/diagnostic_formula_check.py),
[result](artifacts/hmc-warmup-precision-review-2026-09-15/diagnostic-formula-result.json),
[manifest](artifacts/hmc-warmup-precision-review-2026-09-15/run-manifest.json).

## Source basis

Local PDFs, extracted text, and exact downloaded official code are retained in
`.localresources/papers/hmc_warmup_precision_20260915/`; the review artifact
manifest records checksums. ResearchAssistant's source fetch failed first on
sandbox DNS, then with HTTP 406 under trusted execution. PDFs were obtained
from arXiv and parsed locally; official code and Stan documentation were
obtained from GitHub after the Stan website failed DNS. No metadata-only result
is used as support for a method claim.
The ResearchAssistant fetch was invoked using the existing Python 3.13
interpreter, outside that tool's stated Python 3.11 runtime contract; its failed
fetch is recorded only as retrieval history. The downloaded PDFs, metadata
headers, inspected text and official source files provide the evidence here.

* Stan reference manual, **Automatic parameter tuning**, and
  `stan/mcmc/windowed_adaptation.hpp`: configured warmup with fast/slow/fast
  adaptation and growing metric windows. The supplied total warmup count
  controls this schedule. The manual's **Finite numbers of states** discussion
  explicitly says discarding a finite warmup cannot guarantee sufficiency.
  Official sources: <https://mc-stan.org/docs/reference-manual/mcmc.html> and
  <https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp>.
* Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021),
  *Rank-normalization, folding, and localization*, arXiv:1903.08008v5.
  Inspected sections 3.2, 4.1–4.4 and Appendix A. Equation (14) specifies
  ranks; sections 3.2 and 4.1 distinguish mean precision from rank-based ESS;
  section 4.4 gives quantile error intervals. The recommendation of four
  chains, R-hat about 1.01 and pooled ESS 400 is a practical diagnostic
  recommendation, not an accuracy requirement for every estimand.
  <https://arxiv.org/abs/1903.08008>.
* Official Stan posterior `R/convergence.R`: inspected `z_scale`,
  `backtransform_ranks`, `.rhat`, `.ess`, `mcse_mean` and quantile routines.
  Mean MCSE uses unranked mean ESS. This is a direct implementation comparator;
  versions/source checksums must accompany parity fixtures.
  <https://github.com/stan-dev/posterior/blob/master/R/convergence.R>.
* Vats and Flegal, *Lugsail lag windows for estimating time-average covariance
  matrices*, arXiv:1809.04541v3. Inspected sections 2–5, the VAR/MCMC examples,
  and relevant appendix arguments for the batch-means construction.
  Equation (7), assumptions 1–2, and theorems 3–6 give the construction and
  its conditions. Section 5 explicitly rejects one universal lugsail setting;
  finite estimates can have negative eigenvalues. The paper warns that
  settings designed for positive correlation can be counterproductive for
  antithetic chains. <https://arxiv.org/abs/1809.04541>.
* Official `dvats/mcmcse` sources `R/mcse_multi.R`, `R/batchSize.R` and
  `R/mcmcse.R`: inspected the lugsail combination, r=3/c=0.5 setting, batch
  selection and fallback behavior. Its fallback/regularization must not be
  copied silently into a BayesFilter precision decision.
  <https://github.com/dvats/mcmcse>.
* Flegal and Gong, *Relative fixed-width stopping rules for Markov chain
  Monte Carlo simulations*, arXiv:1303.0238. Inspected section 2, section 3.1
  and Appendix A's proof. The stopping theory needs a functional CLT and
  strongly consistent variance estimators; nominal coverage is asymptotic as
  tolerance shrinks, not a finite-sample anytime guarantee.
  <https://arxiv.org/abs/1303.0238>.

## Proposed common procedure

Use one posterior controller for every eligible frozen exact transition, with
target-specific preparation remaining in the existing public tuners. A generic
public name such as `run_hmc_posterior` is proposed; it does not yet exist.
The NeuTra names become compatibility wrappers with explicit policy versions.
Conditional proposal-field mechanics do not gain posterior authority merely
because their output passes diagnostics.

### 1. Adapt geometry, qualify candidates, and freeze the selected member

Keep the broad L search, candidate-specific epsilon repair, fresh verification,
and retention of every verified member. Adaptation draws, tuning measurements
and verification draws remain excluded from posterior estimates. Preparation
should report the actual metric/step-size adaptation history and budget status,
without claiming that adaptation completion establishes stationarity.

The posterior call names a member explicitly and preserves its transition,
coordinate transform, scalar/batched topology, target-health policy, and XLA
qualification. Changing mass, transport, epsilon or L creates a new kernel
requiring the existing qualification procedure. Do not silently pool draws
from different members, independently tuned kernels, or geometry revisions.

### 2. Assess discarded equilibration using the frozen kernel

Run multiple independently driven chains from documented, meaningfully
dispersed valid starts. A clustered tuning endpoint bank is useful continuity
evidence but cannot establish coverage of alternative modes. Report its start
coverage; any additional starts need target/health validation and fresh seeds.
Provide an explicit new-start path rather than mutating a verified endpoint.

Archive all discarded chunks and evaluate predeclared recent windows after the
last adaptation or kernel change. Use the corrected maximum of rank-normalized
split and folded R-hat, bulk/tail ESS as information checks, chain health, and
the named model quantities that matter to the application. Report trace/rank
plots, per-chain location/scale, autocorrelation, evolution of diagnostics, and
cross-window drift. A model can add log-density, mode indicators and important
nonlinear functionals; checking parameter marginals alone cannot cover them.

Use multiple predeclared windows and a confirmation window as a candidate
improvement over a single lucky passing look. Number, size, growth rule and
required persistence need calibration before becoming defaults. Avoid choosing
the best-looking suffix retrospectively. Growing windows can provide more
information when fixed windows yield unstable diagnostics. Drift comparisons
remain descriptive unless their serial dependence and multiplicity are handled.

Passing means **the declared equilibration checks passed**, not that burn-in is
proved sufficient. A cap gives `equilibration_inconclusive_at_cap` with the
failed checks. It does not remove the member from the tuning candidate set.
Do not increase the existing owner-declared 10,000-transition NeuTra cap as
an incidental change; larger campaigns need their own authorized budget.

### 3. Estimate precision for the actual requested posterior quantities

Provide one TensorFlow diagnostic module with one chain-axis convention and
explicit adapters. It computes corrected R-hat, bulk/tail ESS, original-scale
mean MCSE, per-chain results and combined results. Rank-based bulk ESS is not
substituted into an original-scale mean MCSE formula. Additional estimands
include supplied expectations, event probabilities and requested quantiles;
each needs its own error estimate.

Add ordinary batch means and lugsail batch means as named long-run variance
estimators. For a scalar series `Y_t = g(theta_t)`, let `v_b` be its batch-means
estimate using batch length `b`. The lugsail estimate is

\[
\widehat v_{L} =
\frac{\widehat v_b-c\widehat v_{\lfloor b/r\rfloor}}{1-c}.
\]

For `m` independent chains of equal length `n`, each satisfying the relevant
stationary/CLT approximation,

\[
\widehat{\operatorname{Var}}(\bar g)
=\frac{1}{m^2}\sum_{j=1}^{m}\frac{\widehat v_{L,j}}{n},
\qquad \operatorname{MCSE}(\bar g)
=\sqrt{\widehat{\operatorname{Var}}(\bar g)}.
\]

The combination follows by taking the variance of the average of independent
chain means. Do not concatenate separate chains as one time series. Failed
between-chain diagnostics block precision-based completion even if these
within-chain variance estimates look small. Preserve contiguous batches across
storage chunks; storage boundaries are not statistical independence boundaries.
Specify how incomplete terminal batches are handled and record effective counts.

Report the estimator, batch length/count, r, c, original and adjusted estimates,
and validity reason. Too few batches, nonfinite/nonpositive estimates, or absent
required moments yield unavailable precision, not zero MCSE. No clipping of a
negative variance to zero, silent fallback, or selecting the smallest MCSE
among estimators. Finite samples cannot establish finite population moments;
the estimand contract records those assumptions. A constant or unobserved event
requires separate degeneracy handling, not an automatic precision pass.
Define one batch-means estimator exactly, including centering and incomplete
batch treatment. For the divisible case `n=a*b`, use
`v_b = b/(a-1) * sum_k (batch_mean_k - chain_mean)^2`.
Require `r >= 1`, `0 <= c < 1` and positive integer batch lengths with enough
complete batches at both scales. Unequal chain lengths must either use an
explicit weighted formula for the reported pooled estimate or fail validation;
the equal-length formula above cannot be applied unchanged. Do not lower
moment requirements just because the rank-normalized diagnostics are finite.

Keep a source-checked Geyer/Stan comparator. Lugsail is an optional estimator
until parity and calibration support a default; its name does not make it
uniformly better for HMC. Batch size must grow with sample size under the
estimator's assumptions. Square-root batch length is a literature baseline;
AR plug-in selection is another method requiring its own source audit and
validation, not a new unexplained constant.

Quantile MCSE must use a quantile-specific method (for example the checked
Vehtari section 4.4 indicator-ESS/order-statistic construction). MCSE for a
mean does not quantify error in a tail quantile. A multivariate covariance
extension should operate on a declared estimand vector with explicit rank/PSD
checks, not allocate dense matrices for every model parameter by default.

### 4. Continue retained sampling until the declared precision is met

After the warmup boundary, accumulate all retained draws of the frozen kernel.
Check numerical health, corrected cumulative R-hat and information diagnostics,
and estimand-specific accuracy. Custom callbacks may add model checks but must
not silently replace mandatory core checks. Without requested precision, return
diagnostics with `precision_not_requested`; do not claim that a sufficient
number of samples has been established.

The user declares either an absolute MCSE tolerance in the estimand's units,
an MCSE-to-posterior-SD tolerance, or an interval-width target and confidence
level. Treat these as distinct options. For an MCSE-to-SD target `delta`, the
identity `ESS_mean = SD^2/MCSE^2` implies `ESS_mean >= delta^(-2)`.
Thus an illustrative 1% target implies 10,000 effective observations; that
number is derived from the example tolerance, not a proposed global default.

For an interval half-width target `h`, test `z * MCSE <= h`; Flegal–Gong's
full-width convention is `2*z*MCSE`. State which convention is used. If formal
relative fixed-width stopping is implemented, preserve the minimum effort,
penalty, growing batch and regularity conditions of the source. A capped,
periodically inspected implementation is initially an operational accuracy
screen; claiming the paper's asymptotic result requires a checked mapping of
its assumptions and inspection schedule. Neither grants finite-sample anytime
coverage. Simultaneous interval claims need a declared multiplicity treatment.

If the retained run develops failed convergence checks, freeze promotion and
continue only under the declared repair/continuation policy. Do not silently
discard more of the retained history until a favorable suffix passes. Any
restart records a new boundary and keeps the old segment as failed evidence.
Low ESS or unmet precision generally motivates more fixed-kernel sampling
within budget; numerical corruption requires a different response.

### 5. Return an actionable report and restart state

The report separates `adaptation_complete`, `equilibration_status`,
`retained_diagnostic_status`, `precision_status`, and `stop_reason`.
For each monitored quantity, include R-hat, bulk/tail/mean ESS as applicable,
MCSE and its validity, requested tolerance, counts, diagnostic history and
failing reasons. Always show the discarded boundary and kernel identity.
These status names are proposed API design, not current fields.

Resume must preserve the warmup boundary, precision policy, estimand definitions,
chunks, incomplete batches, transition state, full seed inventory and budget
accounting. Fix the prior review's bridge and seed defects first. An extension
cannot mix tuning draws back into retained estimates or reset diagnostic history.

## Default and assumption audit

| Choice | Provenance and status | Failure mode | Earliest useful check |
| --- | --- | --- | --- |
| Existing 2,000/1,000/10,000 and 1.05 warmup settings | Owner policy dated 2026-07-15; inherited baseline, not target-specific evidence | Premature passing look or budget exhaustion | Report diagnostic histories and start coverage; calibrate known transient fixtures |
| Four chains, modern R-hat about 1.01, pooled bulk/tail ESS 400 | Vehtari et al. recommendations; candidate information screen, not universal precision | Shared missed mode; inaccurate nonlinear/tail estimates | Dispersed/multimodal starts and named estimands; original-scale MCSE |
| Multiple windows/confirmation | Proposed heuristic, not a literature theorem or selected default | Repeated looks and correlated windows can still pass prematurely | Replicated transient/slow-mixing false-pass calibration |
| Lugsail r=3, c=0.5 | Vats–Flegal section 5 and mcmcse; baseline for persistent positive dependence | Noisy/negative estimate, poor antithetic behavior | Positive/negative AR fixtures, batch sensitivity, raw variance validity |
| Batch length proportional to sqrt(n) | Paper simulation baseline, not automatic target-optimal length | Batches too short for long dependence or too few for stability | Autocorrelation and batch-count diagnostics, coverage calibration |
| MCSE tolerance/confidence level | Application requirement, not chosen by the library in this review | Overstating accuracy or unnecessary cost | Show expected effort and exact estimand/units before the run |
| Independent chains, finite moments, applicable CLT | Mathematical assumptions, not established by a diagnostic pass | Incorrect MCSE or nominal interval interpretation | Seed audit, target analysis, heavy-tail and missed-mode counterexamples |
| CPU/non-XLA review diagnostic | Explicit arithmetic debugging exception | Mistaking CPU parity for GPU runtime qualification | Preserve device label and forbid sampler/default conclusions |

## Implementation order and acceptance checks

1. Repair the diagnostic formulas and consolidate numerical primitives. Tests
   must use checked equations and fixed official-source fixtures, not one local
   implementation as the oracle for another. Cover ties, folding, odd lengths,
   constant chains, nonfinite values, shifted/scaled chains and antithetic ESS.
   Preserve old diagnostic identity; corrected quantities require new versioned
   report semantics rather than silently changing old results on reload.
   Trace downstream readers and recalculation tools so prior diagnostic
   summaries can be identified and recomputed from preserved draws. A report
   produced by either affected routine needs corrected evaluation before
   relying on its modern-R-hat threshold claim; the finding alone does not
   establish that its draws are invalid or that every prior decision changes.
2. Repair the exact-member posterior bridge, derived seeds, and diagnostic
   callback behavior. Reuse the member's transition and health evaluator;
   consolidate the existing sequential loops into one controller without
   changing the transition. Test actual scalar-only and no-telemetry targets,
   transformed coordinates, verified repaired members, interruptions and resume.
3. Implement estimand-specific MCSE, plain/lugsail batch means, explicit
   validity states and source-checked quantile diagnostics. Test finite-array
   arithmetic against independent references, chain combination, truncated
   batches, chunk boundaries, and offline versus resumed equality. Formula
   parity alone does not establish interval calibration.
   Pure numerical kernels use stable TensorFlow signatures and bounded
   shape/compilation policies; host-side reports and source checks stay outside
   XLA. Qualify the new kernels on the project's GPU/XLA route under a separate
   bounded numerical-equivalence and resource check. CPU formula tests alone
   cannot establish that qualification.
4. Add explicit discarded-equilibration and retained-precision policies and
   reports. Test that low R-hat cannot imply precision, callbacks cannot remove
   core checks, a cap yields incomplete evidence, fresh starts are validated,
   and posterior failure never changes tuning membership.
5. Before promoting new automatic defaults, run a separate bounded calibration
   plan: independent and AR(1) sequences with analytically known long-run
   variance, including negative and near-unit positive correlation; transient
   starts; constant/rare-event examples; heavy tails; Gaussian and known
   multimodal sampler fixtures. Compare current diagnostics, corrected
   Geyer-style diagnostics, ordinary BM and proposed lugsail variants.
   Use replicated coverage, premature-stop frequency, MCSE calibration and
   uncertainty intervals, not which estimator reports the smallest MCSE.
   Exact replication budgets and tolerances belong in that campaign plan; this
   review launches no calibration campaign. An all-chains-in-one-mode fixture documents the detection
   limit, not an impossible requirement that diagnostics infer an unseen mode.
6. Rewrite the guide's connected account of preparation, qualification,
   equilibration and precision; update `hmc-tuning-interface.md`, the relevant
   parts of chapters 21b, 22 and 26b, public API inventory and executable
   examples. Explain defaults, failures, precision units and estimator validity.
   Run examples through the actual public posterior call, build the book and
   inspect its rendered explanation. Retain historical phase readers as such.

The scope of this extension is posterior assessment and the diagnostic
primitives it shares with tuning reports. Findings R2, R4, R5, R6 and R7 in
the preceding review remain open; this plan does not silently close shared
invalidity, target-exception, budget-deferral, acceptance-documentation or
history-scaling work. Bridge R1 and seed R3 are explicit dependencies above.

Final skeptical review: adding lugsail alone would leave a wrong R-hat
definition, optional precision checks and a broken consumer bridge in place.
Automatically adopting multiple windows, 400 ESS or a lugsail setting without
calibration would replace one inherited heuristic with another. The revised
order fixes formulas and execution invariants first, exposes precision
requirements, and keeps proposed automatic defaults provisional. Sources and
the arithmetic check support the identified defects; no stochastic ranking or
target-specific sufficiency claim follows. Runtime and guide behavior remain
unchanged at the end of this review.

## Decision and limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Correct the current description | Existing source shows warmup logic and MCSE | No need to invent either from scratch | Consumer coverage is incomplete | Consolidate and expose them | Current warmup is sufficient |
| Repair numerical diagnostics before policy expansion | Two source-confirmed formula mismatches reproduced on fixed arrays | Blocks a modern-R-hat identity claim | Target-scale impact is unmeasured | Preserve the arithmetic regressions and implement shared formulas | Prior posterior results are numerically re-audited |
| Add explicit precision-controlled retained sampling | R-hat alone cannot answer estimator accuracy | Missing requested precision blocks completion | User-specific tolerance and regularity conditions | Implement typed precision policy and reporting | Tuning candidate acceptance proves posterior quality |
| Add lugsail as a checked option | Published estimator and official implementation available | Invalid variance cannot grant precision | Finite-sample/antithetic performance | Formula tests then separate calibration | Lugsail universally outperforms Geyer/Stan |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Source mismatches and the prior bridge/seed defects require repair. |
| Statistically supported ranking | None; this is a source/arithmetic audit. |
| Descriptive-only differences | Diagnostic values from fixed arrays are counterexamples, not stochastic performance comparisons. |
| Default-readiness | No new warmup, lugsail or precision default is promoted. |
| Next evidence needed | Correct formulas and controller integration, followed by a predeclared replicated calibration campaign. |

Post-review challenge: even perfect implementation can report small R-hat and
MCSE when all chains explore the same wrong region. Target correctness,
initialization coverage and model-specific validation remain separate. More
diagnostic machinery cannot replace them. The present recommendation is a
repair of engineering and reporting, with explicit limits on the scientific
conclusions it can support.
