# Overall HMC tuning, burn-in and precision repair

Baseline: `5139f151237e9764ebf34b4e8cf6ceee1e74a38d`. Authorized by the user's
2026-09-15 instruction to combine the reviews, review a repair plan and execute it.

This plan combines the [whole-procedure review](bayesfilter-hmc-tuning-followup-review-2026-09-15.md)
and [burn-in/precision review](bayesfilter-hmc-warmup-precision-repair-plan-2026-09-15.md).
Keep their counterexamples and literature records. Their findings establish
engineering defects and missing assessments, not that HMC fails on the user's
models or that a particular sampling budget is sufficient.

## Overall recommendation

Keep one candidate-set tuning procedure with target-specific preparation,
candidate-specific epsilon, fresh verification and retention of every verified
member. Connect an explicitly selected member to one posterior assessment
policy that preserves its actual transition and health requirements. Distinguish
geometry adaptation, exact-kernel qualification, discarded equilibration and
retained estimator precision in both the API and the book. R-hat and MCSE never
become tuning qualification or candidate-ranking criteria.

Correct the diagnostic mathematics before extending automatic decisions. Add
lugsail batch means as an explicitly chosen MCSE method, alongside ordinary
batch means and a checked autocorrelation estimator. A lugsail estimate is not
a burn-in estimator or a guarantee of convergence. Do not invent new universal
sample counts or claim that repeated diagnostic checks prove stationarity.

## Execution scope and dependencies

| Work | Findings addressed | Concrete deliverable |
| --- | --- | --- |
| P1: candidate lifecycle | R2, R4, R5 | Valid invalidation history; durable typed candidate-domain failures; reconsider affordable deferred work after resources change. |
| P2: diagnostic mathematics | rank denominator, TFP ratio/R-hat confusion, duplicated formulas | One source-checked rank/split/fold calculation; explicit estimator identities; independent formula regressions; dependent-window drift labeled accurately. |
| P3: posterior execution and assessment | R1, R3, optional precision and callback replacement | Checked member transitions feed the shared sequential loop; complete bounded seed validation; core health/R-hat checks cannot be replaced by callbacks; explicit burn-in evidence and retained precision policy. |
| P4: precision estimators | missing lugsail and estimand-specific precision | TensorFlow batch-means/lugsail mean MCSE, quantile-specific MCSE, validity states and requested tolerances; cumulative assessment with honest cap results. |
| P5: persistence | R7 | Immutable evidence persisted once, cached numerical validation, compact shared member exports with explicit portable export, iterative predecessor checks. |
| P6: guide and API | R6, fragmented explanations and historical configuration | Working public examples, exact acceptance predicate, correct replacement links, retired-option handling, unified tuning-to-posterior account and build. |
| P7: verification and terminal audit | cross-feature regressions | Focused regressions, bounded estimator calibration, public-route smoke, rendered guide inspection and result memo. |

P1/P2 precede posterior extension. P3 and P4 share one assessment implementation;
legacy archive formats may keep I/O wrappers but must use that implementation
for new assessments. P5 must preserve checksum, source, scope, ancestry and
shared-invalidity checks. Refactoring is confined to these responsibilities.
Wholesale movement of the 30,000-line historical tuner is deferred: its active
preparation wrapper is already separate, and moving unrelated numerical
primitives would expand the change without answering a reproduced defect.
Document this remaining structural debt rather than claiming the whole module
has been modernized.

## Evidence contract, defaults and budgets

The engineering comparator is the current documented candidate-set contract,
the two preserved counterexample collections and checked published formulas.
Pass requires regression tests for each reproduced defect, round-trip/resume
invariants, correct estimator arithmetic and a guide matching executed behavior.
No descriptive acceptance, R-hat, ESS, MCSE or runtime ranking selects a winner.
Numerical corruption, changed kernel/target identity, invalid seed schedules,
missing required evidence and incorrect formulas veto the affected operation.
Low ESS, unmet accuracy or a failed candidate are repair/continuation signals
within the declared budget, not automatic rejection of the research direction.

Retain the owner-policy warmup minimum/window/cap (2,000/1,000/10,000) and
1.05 screening threshold as inherited operational defaults, with explicit
provenance and limitations. Correcting the statistic is a bug fix requiring a
new diagnostic identity, not evidence that these thresholds are calibrated.
Expose optional minimum ESS, persistence/confirmation and precision settings;
do not silently promote uncalibrated multi-window settings. Accuracy tolerances
must be supplied for claims of sufficient retained precision. No requested
accuracy means `precision_not_requested`.

Lugsail `r=3,c=0.5` and square-root batch length are explicit literature
baselines (Vats–Flegal sections 4–5 and mcmcse), not universal recommendations.
Preserve antithetic and nonpositive-estimate failure cases. Ordinary means,
event probabilities and quantiles require different quantities; rank bulk ESS
cannot substitute for original-scale mean ESS. The full source/assumption audit
is in the linked burn-in plan. Runtime implementation uses TF/TFP; NumPy is
permitted only in independent reference tests and calibration analysis.

Routine unit/import/build checks use the existing environment. Explicit CPU
diagnostics hide GPUs before imports. The total convenience ceiling for test
and diagnostic processes is 45 CPU wall minutes, with at most 5 GPU wall minutes
for trusted GPU/XLA numerical parity/smoke if available. No model-scale HMC or
training campaign, package changes or threshold/default promotion is authorized
by this implementation plan. Stop a check that exceeds its allocation and
record the uncovered boundary; do not substitute an easier scientific claim.

For optional-MCSE calibration, use 200 independent replications of 4 chains
with 2,048 retained draws each for stationary unit-variance AR(1) coefficients
`0, 0.8, -0.5`. These are convenience diagnostic fixtures spanning independence,
positive persistence and antithetic correlation, not model defaults. Their
known long-run variance is `(1+rho)/(1-rho)`. Compare ordinary BM, lugsail BM
and the autocorrelation estimator against this truth and realized replicate
mean errors. Report coverage with binomial uncertainty and estimated versus
empirical MCSE; no universal superiority claim or automatic default follows.
Keep this within 5 of the 45 CPU minutes. Rare-event/constant/transient and
missed-mode fixtures test invalidity and detection limits separately. Formal
sequential confidence coverage and a new burn-in stopping default remain
unproved; repeated-look precision is an operational accuracy assessment.

Artifacts live under `docs/plans/artifacts/hmc-overall-repair-2026-09-15/` in
fresh subdirectories. Record commands, Python/TF/TFP, intentional CPU/GPU mode,
source hashes, seeds, elapsed time, test outcomes and limits. Preserve unrelated
dirty files using the existing hash ledger. No commit or push is needed to
complete this request.

## Skeptical plan review before execution

The reviews reveal why simply adding lugsail is insufficient: wrong R-hat
arithmetic, broken target/seed boundaries and optional core diagnostics could
still produce misleading decisions. The ordering above repairs these first.
An independent source equation is the oracle, not a test that copies the same
mistake. Existing successful tests do not override the new counterexamples.

Key constraints: do not fabricate acceptance feedback from a failed native call;
do not erase a previously verified fact when current eligibility is revoked;
do not cache mutable evidence without detecting mutation; do not combine chains
as one serial process; do not treat adjacent windows as independent; do not
choose a favorable retained suffix; and do not let a custom callback skip core
checks. Exact-target and conditional-mechanics authority stay distinct.

Potential flaws resolved in this plan: unlimited refactoring is replaced by
bounded extraction/caching; multi-window and lugsail settings remain explicit
options; calibration has known truth and uncertainty, and does not grant
default-readiness; source/version checks accompany changed diagnostic semantics.
The plan passes this review for engineering execution. A final audit will
report unresolved limitations explicitly, with no blanket claim of universal
tuning success or sufficient burn-in for arbitrary targets.

## Execution record

Completed on 2026-09-15. The [terminal result and audit](bayesfilter-hmc-overall-repair-result-2026-09-15.md)
records the implemented changes, tests, calibration, GPU check, guide build and
remaining limitations. The [run manifest](artifacts/hmc-overall-repair-2026-09-15/run-manifest.json)
preserves commands, environment, source hashes and the evidence inventory.

P1–P7 were executed within the bounded scope above. The broad integration run
passed 277 tests; after the last reporting-compatibility adjustment and two
additional regressions, the affected-route run passed 114 tests. These runs
overlap and must not be added as distinct test cases. The public Gaussian
example passed on CPU and GPU/XLA, the GPU lugsail arithmetic agreed with an
independent CPU formula to 1.05e-17, and the full guide PDF built successfully.
The GPU example predates only the final posterior-summary compatibility and
decision-wording change; its execution-time source closure is preserved.

The terminal review resolved the following scope and interpretation questions:

* The shared runtime ESS remains the identified TFP positive-pairs estimator.
  The source-translated Stan initial-monotone implementation is an independent
  calibration comparator. Its preliminary covariance normalization was wrong;
  those results are preserved as invalid and the final calibration was rerun
  after checking the official implementation body.
* Optional ESS floors, consecutive warmup checks, named functionals and mean/
  quantile accuracy targets are implemented. Inherited warmup counts are not
  recalibrated. Growing windows, automatic selection of a new start bank,
  automatic batch-size optimization, joint confidence regions and formal
  fixed-width/anytime stopping are broader research extensions, not completed
  features of this repair. No new plotting interface was needed for the
  implemented diagnostic histories and archived draws.
* Persistence now avoids repeated immutable-file parsing/writing and cached
  numerical recalculation, uses shared compact exports, and walks predecessors
  iteratively. Live content hashing, metadata validation and cumulative sample
  processing still depend on history length. R7's identified duplication is
  reduced; target-scale persistence performance remains unqualified.
* The legacy archived API retains its documented strict `<` R-hat comparison
  and extra ESS/coordinate checks. Shared and exact-transition APIs retain
  `<=`. All use the same corrected assessment arithmetic. The historical
  Phase 29 warmup screen still applies configured drift thresholds as heuristic
  rejection criteria; it is separate from the common posterior assessment.
  Its statistics remain descriptive because their normalization omits
  covariance between adjacent epochs; they are not calibrated z tests.
* The guide's suggested movement-override defect was partly stale:
  `require_all_chain_movement=False` was already rejected by its config.
  The active fixed-transport facade now also rejects retired R-hat overrides.

Terminal verdict: the scoped engineering repair is complete. Its finite tests
and calibration support the stated arithmetic and execution invariants; they
do not establish globally sufficient burn-in, universal search success,
lugsail superiority, or model-specific posterior validity.
