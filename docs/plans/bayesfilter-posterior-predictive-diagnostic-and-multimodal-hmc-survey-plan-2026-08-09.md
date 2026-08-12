# Posterior-predictive diagnostic and multimodal-HMC survey plan (2026-08-09)

## Objectives

1. Replace fixed posterior-summary plug-in assessment with a reusable diagnostic
   that draws one parameter independently from an empirical posterior for every
   simulated path, then compares the resulting posterior-predictive path bank
   with an independent true-parameter path bank.
2. Validate the complete diagnostic mechanics on analytically tractable
   unimodal and multimodal distributions, including alternatives that must be
   detected and a deliberately collapsed multimodal approximation.
3. Produce a standalone, buildable LaTeX survey of methods for applying HMC to
   multimodal targets, based on inspected primary literature and locally stored
   source copies.

The present seed-B retained archive is not eligible for a claim-bearing
multimodal posterior-predictive run because all 4,000 retained states have
positive observation weight while a known near-equal-density stationary mode
has negative observation weight. This task implements and validates the
correct procedure and a fail-closed SSL-LSTM runner for a future valid posterior
artifact; it does not silently reuse seed B as a multimodal posterior authority.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main diagnostic question | Given an empirical posterior draw archive and a conditional simulator, is the induced empirical posterior-predictive complete-path law distinguished from the true-parameter simulator law? |
| Quantity under test | `P_hat(Y in A) = M^{-1} sum_m P(Y in A | theta_m)`, implemented by an independently sampled archive index and independent simulator noise for every output path. |
| Exact comparator | Independent complete paths from the same simulator at the known synthetic true parameter. |
| Primary diagnostic decision | Whole-path energy permutation test at each separately declared horizon; reject equality only when the Monte Carlo permutation p-value is below the declared per-test alpha. |
| Required validity cases | Unimodal Gaussian posterior with analytically known Gaussian posterior predictive; bimodal Gaussian-mixture posterior with analytically known mixture predictive; shifted/wrong comparators; collapsed single-mode approximation to a bimodal posterior. |
| Expected failure mode | A programmer draws one posterior parameter and generates many conditional paths, or samples modes with the wrong weights, creating a plug-in or incorrectly weighted mixture while the energy-test code itself remains correct. |
| Promotion criterion | Mechanics and analytic validation tests pass; the SSL-LSTM runner records one posterior index per path, `replication_count=1`, independent seed domains, posterior artifact identity, and no fixed-summary fallback. |
| Promotion veto | One parameter reused for multiple test paths, wrong or missing mode weights, invalid/nonfinite posterior rows, simulator output not one path per selected row, shared random seeds across arms, or test cases unable to detect a strongly shifted/collapsed alternative. |
| Continuation veto | Test or build failure, source/citation mismatch, unavailable primary source for a material literature claim, or an attempted real SSL-LSTM run without a posterior artifact that passes its declared sampler/mode-weight authority checks. |
| Repair trigger | A null fixture rejects under fixed seeds, a strong alternative is not detected, or source inspection contradicts the survey classification/recommendation. |
| Must not conclude | Equality after non-rejection, HMC correctness from predictive agreement, posterior correctness without independent authority, correct multimodal weights from mode visitation alone, or applicability of one tempering method to every target. |

## Evidence contract

### Diagnostic implementation

1. Add an explicitly diagnostic TensorFlow module under `bayesfilter/testing/`.
   NumPy is not used. Posterior-row selection uses stateless TensorFlow random
   sampling with replacement and is replayable from a two-word seed.
2. Every requested output path receives one independently sampled posterior
   archive index. The simulator is called once on the complete selected batch.
   It must return exactly one rank-two complete path per selected parameter row.
3. Posterior-index randomness, candidate simulator randomness, truth simulator
   randomness, and permutation randomness use disjoint seed domains.
4. The SSL-LSTM wrapper passes `n` selected physical parameter rows to
   `forecast_complexity_conditional_moments` with `replication_count=1` and
   reshapes `[n,1,T]` to `[n,T]`. No posterior mean, median, MAP, repeated
   parameter row, or scalar/sample loop is an allowed fallback.
5. A future posterior artifact must bind its physical draw tensor, parameter
   names, target signature, sampler result, retained/warm-up status, and
   multimodal-weight authority. The runner fails closed when the artifact says
   multimodal weights are unresolved. This is artifact validation, not proof
   that the upstream sampler claim is true.
6. The energy test remains the existing TensorFlow/XLA whole-path permutation
   implementation. Tests must cover the composition of posterior sampling,
   conditional simulation, and equality testing rather than only retesting the
   energy kernel.

### Analytic validation

1. Unimodal null: `theta ~ N(mu,tau^2)` and `Y|theta ~ N(theta,sigma^2)`, so
   `Y ~ N(mu,tau^2+sigma^2)`. Compare an empirical posterior-predictive sample
   with an independent analytic reference sample. Record descriptive moment
   agreement and require the fixed-seed equality test not to reject at 1%.
2. Unimodal alternative: compare the same posterior predictive with a strongly
   shifted reference and require rejection.
3. Multimodal null: `theta` follows a two-component Gaussian mixture with
   declared unequal weights and `Y|theta ~ N(theta,sigma^2)`. Compare against
   an independent analytic mixture sample with the same weights and convolved
   component variances; require non-rejection under fixed seeds and verify both
   modes are selected.
4. Multimodal alternatives: use an incorrect mixture weight and a collapsed
   single-mode archive; require the diagnostic to reject both under fixed
   seeds. These tests target the additional posterior-sampling layer.
5. Fixed-seed p-values are regression fixtures, not a calibration study or
   proof of Type-I error control. The analytic distribution identities and
   direct moment/mixture checks provide independent mechanics evidence.

### Literature survey

1. Store a local PDF and extracted text for every primary paper materially used
   in the survey under `.localresources/papers/multimodal_hmc/`.
2. Inspect technical method sections, invariant-target or detailed-balance
   arguments, transition/acceptance equations, experiments, limitations, and
   relevant appendices. Abstract-only inspection is insufficient.
3. Cover at least: energy barriers in ordinary HMC; discrete parallel
   tempering/replica exchange with HMC within temperatures; tempered
   transitions/Hamiltonian tempered transitions; continuously tempered HMC;
   wormhole or mode-informed geometry; transport/reparameterization methods;
   and evidence/mode-weight estimation through AIS or SMC where relevant.
4. Separate exact invariant algorithms from heuristics, mode-discovery
   assumptions, and evidence-estimation machinery. State whether a method
   addresses mode discovery, cross-mode transitions, relative mode weights, or
   only within-mode geometry.
5. Include equations for tempered targets and swap/transition corrections,
   operational diagnostics, failure modes, a comparison table, and a bounded
   recommendation for BayesFilter. Do not recommend TFP NUTS; the repository
   policy uses fixed HMC kernels.
6. Build the standalone document with `latexmk -pdf`; unresolved citations,
   missing bibliography entries, or LaTeX errors are completion vetoes.

## Numeric provenance and default audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| One posterior row per path | Derived from the posterior-predictive integral | Monte Carlo sample from the empirical mixture | Reusing a row creates a plug-in conditional law | Alignment and unique-index tests; explicit `replication_count=1` source guard |
| Sampling with replacement | Standard empirical-distribution Monte Carlo; reviewed default | Produces iid draws conditional on a fixed archive | Without replacement changes dependence and fails when `n>M` | Test `n>M`, replay, and index bounds |
| Per-test `alpha=0.01` | User-fixed inherited diagnostic level | Direct comparability with previous path tests | Non-rejection misread as equivalence | Strict classification and explicit nonclaims |
| Analytic fixture sizes/permutations | Test-only convenience choices, to be frozen after a deterministic power/replay canary | Keep CI bounded while detecting large declared alternatives | Flaky null or underpowered alternative | Fixed seeds, moment checks, and strong-effect canary before freezing |
| Survey corpus | Primary-source selection based on method families, not an exhaustive systematic review | Sufficient to compare the principal HMC-compatible mechanisms | Publication bias or omitted recent variant | Search backward/forward citations and record scope/non-exhaustiveness |
| BayesFilter recommendation | Hypothesis pending source inspection | Must fit fixed-HMC, TensorFlow/TFP, XLA, and multimodal-weight needs | Attractive method may visit modes without estimating weights | Separate transition and weight criteria in comparison table |

## Skeptical pre-execution audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | The comparator is the same conditional simulator at the known synthetic truth. Analytic tests compare against closed-form predictive laws, not plug-in parameters. |
| Proxy promoted? | Energy p-values answer finite-horizon distributional distinguishability only. R-hat, mode visits, moment checks, and literature benchmarks remain diagnostics. |
| Missing stop conditions? | Invalid archive/shape/seed/finite/alignment/weight authority, failed analytic cases, missing primary sources, and LaTeX build failures all stop completion. |
| Unfair comparison? | Candidate and truth use equal path counts, common output coordinates, the same simulator family, and independent random streams. Analytic fixtures use exact matched or deliberately changed distributions. |
| Hidden assumptions? | Exposed: empirical posterior approximation, sampling with replacement, one posterior draw per path, correct upstream mode weights, iid conditional simulation, finite horizons, energy-test geometry, and survey non-exhaustiveness. |
| Stale context? | The seed-B archive is explicitly ineligible as a multimodal authority. Historical fixed-summary results are retained as evidence motivating the change, not reused as the new result. |
| Environment mismatch? | Implementation and fixtures use TensorFlow/TFP and XLA-capable kernels. Tests are CPU diagnostics with GPUs hidden. The survey is backend independent but recommendations are filtered through BayesFilter policy. |
| Could success mislead? | Yes. Predictive agreement does not prove posterior or HMC correctness, especially for non-identifiable models. The diagnostic and survey state this explicitly. |
| Could failure be non-scientific? | Yes. Archive contract, simulator status, source access, build errors, or an underpowered/flaky fixture are separately classified as engineering/test-design failures. |
| Will artifacts answer the request? | The module and tests directly validate the added posterior-mixture step in unimodal and multimodal settings; the standalone LaTeX source/PDF and local primary papers preserve the survey. |

Audit verdict: **PASS WITH TWO EXECUTION GUARDS**. First, no real SSL-LSTM
posterior-predictive claim is run until a multimodal posterior artifact with
resolved relative weights exists. Second, literature recommendations are not
written until the cited primary method/equation sections are locally inspected.

## Execution

1. Implement the reusable empirical posterior-predictive path-bank API and its
   fail-closed SSL-LSTM diagnostic runner.
2. Add deterministic contract, analytic unimodal, analytic multimodal,
   wrong-weight, collapsed-mode, and shifted-alternative tests; run focused
   TensorFlow/XLA checks.
3. Fetch and store the bounded primary literature corpus; extract and inspect
   relevant technical sections and original-code pointers when available.
4. Write the standalone LaTeX survey and bibliography, build the PDF, and audit
   citations and recommendation language against the inspected sources.
5. Write a result/reset note recording implementation status, test evidence,
   literature scope, build artifact, unresolved real-run dependency, and next
   justified action.

Planned implementation paths:

- `bayesfilter/testing/posterior_predictive_tf.py`
- `docs/benchmarks/run_ssl_lstm_q20_posterior_predictive_energy_diagnostic_2026_08_09.py`
- `tests/test_posterior_predictive_tf.py`
- `tests/test_ssl_lstm_q20_posterior_predictive_energy_diagnostic.py`

Planned survey paths:

- `.localresources/papers/multimodal_hmc/`
- `docs/surveys/multimodal_hmc_survey.tex`
- `docs/surveys/multimodal_hmc_survey.bib`
- `docs/surveys/multimodal_hmc_survey.pdf`

Compute budget: focused tests under 10 minutes total; no real q=20 predictive
campaign in this task; literature downloads limited to primary papers in the
declared method families; LaTeX build under 5 minutes. One repair attempt per
failed test fixture or source URL before revising the plan/source choice.
