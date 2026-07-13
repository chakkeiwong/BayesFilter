# BayesFilter SSL-LSTM Completion Roadmap

Date: 2026-07-11

Status: `CODEX_SUBSTITUTE_REVIEW_CONVERGED_A0_ONLY_RUNTIME_REQUIRES_A0_GATE`

## Purpose

This roadmap defines what remains to finish the BayesFilter state-space LSTM
(SSL-LSTM) as an end-to-end, testable model. It incorporates the new
posterior-predictive validation method without treating sampler agreement as
the whole definition of model completion.

The governing predictive-equivalence design is:

- `docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md`.

The model and statistical rationale are documented in:

- `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`.

This lane is an `extension_or_invention`. It is not a Zhao-Cui
source-faithfulness lane.

## What "Finished" Means

Completion has four milestones. They must not be collapsed into one claim.

| Milestone | Deliverable | Required claim boundary |
| --- | --- | --- |
| M1. Engineering-complete scalar core | The four-parameter scalar SSL-LSTM can be fitted with valid ordinary HMC, filtered, forecast, serialized, and checked on repeated synthetic held-out futures. | Internal/experimental scalar implementation only; no independent computational replication, posterior-correctness claim, or validated release claim. |
| M2. Computationally replicated scalar vertical slice | A valid exact-corrected NeuTra-HMC arm passes frozen predictive-equivalence confirmation and fresh-seed audit against valid ordinary HMC, and the scalar synthetic generative-calibration gate passes. | Validated only for the fixed scalar target, simulated data regimes, four-parameter mask, SVD-UKF likelihood, and frozen predictive design. |
| M3. Extensible trainable SSL-LSTM | A production TensorFlow target accepts reviewed trainable-parameter masks and passes blockwise synthetic identifiability, sampler-validity, and predictive-equivalence gates beyond the four-parameter fixture. | Only parameter blocks and data regimes that pass their own gates are supported; full 24-parameter estimation is not assumed. |
| M4. Application-ready SSL-LSTM | A data adapter, train/validation/audit split, rolling forecasts, proper-score evaluation, serialization, examples, and bounded public API have passed GPU/XLA integration tests. | Application readiness is data set and configuration specific; it is not scientific superiority or a new default policy. |

M2 is the first validated scalar release candidate. M1 may support continued
internal development when NeuTra fails, but it must remain labeled
engineering-complete rather than validated. M3 implementation work may start
after M1, but no expanded mask is promoted before M2 closes. M4 empirical
claims require both M2 and the selected M3 parameter mask.

## Current State Inventory

### Reusable Components Already Present

| Component | Current source | Evidence boundary |
| --- | --- | --- |
| Static SSL-LSTM dimensions and 24-entry scalar parameter chart | `bayesfilter/nonlinear/ssl_lstm_protocol.py` | Contract and shape behavior are implemented. |
| TensorFlow parameter unpacking, softplus scale transforms, transition, observation, and analytic derivatives | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py` | Unit-tested model mechanics exist. |
| Structural SVD-UKF model with stochastic latent coordinates and deterministic hidden/cell completion | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py` | Value/score adapter exists; the SVD-UKF remains an approximate nonlinear likelihood. |
| Typed filtered means and covariances | `bayesfilter/results_tf.py` and `tf_svd_sigma_point_filter(..., return_filtered=True)` | Enough information exists to construct a terminal Gaussian forecast state without parsing derivative traces. |
| Scalar four-free-parameter target and MAP-local geometry artifacts | Scalar filtering geometry and Phase 2S artifacts dated 2026-07-08/09 | Diagnostic CPU-hidden evidence only; benchmark-local target construction is not a production API. |
| Ordinary HMC, tuning, retained-sample, and telemetry infrastructure | `bayesfilter/inference/` | Reusable mechanics exist, but the scalar SSL-LSTM lacks a fresh four-chain GPU/XLA confirmation run. |
| Frozen affine/dense-IAF artifact loading and exact transformed-HMC binding | `bayesfilter/inference/neutra_artifacts.py` and fixed-transport HMC modules | Artifact consumption and correction exist. |
| Affine NeuTra training fixtures and bounded GPU training preflight | `bayesfilter/testing/` | These are narrow fixtures, not a generic dense-IAF production trainer. |

### Missing Completion Components

1. A reusable TensorFlow SSL-LSTM posterior target with a typed parameter mask;
   current scalar target construction is embedded in benchmark scripts.
2. A typed terminal-state and multi-step forecast API with stable provenance.
3. A predictive-statistics and chain-aware uncertainty engine.
4. Predeclared equivalence-margin, bandwidth, bootstrap, and power calibration.
5. Fresh four-chain ordinary-HMC GPU/XLA baseline evidence with native
   divergence telemetry.
6. A BayesFilter-owned GPU/XLA dense-IAF trainer that emits the existing frozen
   artifact schema.
7. Exact-corrected dense-IAF NeuTra-HMC confirmation and audit replication.
8. Repeated synthetic held-out forecast calibration for the scalar model.
9. Blockwise expansion beyond four trainable parameters with identifiability
   and data-length gates.
10. Real/application data plumbing, rolling-origin evaluation, classical
   forecast baselines, and model-adequacy evidence.
11. A bounded fit/filter/forecast API, serialization contract, examples, and
    release tests.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main engineering question | Can BayesFilter expose one deterministic, GPU/XLA-first SSL-LSTM target that supports fitting, filtering, posterior forecasting, and reproducible artifacts? |
| Main scalar validation question | Do valid ordinary HMC and exact-corrected NeuTra-HMC runs induce practically equivalent 1-to-10-step predictive laws for the same four-parameter scalar target? |
| Expansion question | Which additional parameter blocks are identifiable and computationally viable under predeclared synthetic data regimes? |
| Application question | Does the validated SSL-LSTM produce calibrated held-out forecasts competitive with naive and tuned classical state-space baselines? |
| Expected failure modes | Shared target or forecast bug, unidentifiable parameter block, insufficient data, invalid terminal state, HMC divergence/mixing failure, reverse-KL mode or tail undercoverage, transport/Jacobian defect, underpowered equivalence test, data leakage, or weak out-of-sample model adequacy. |
| Promotion criterion | Each milestone passes its applicable engineering, numerical/sampler, predictive-equivalence, synthetic-calibration, and empirical model-adequacy gates in that order. |
| Promotion veto | Nonfinite or mismatched target/score, failed oracle, failed signature, invalid filtered covariance, positive native divergences, failed R-hat/ESS/MCSE gates, transform defect, changed frozen design, data leakage, or missing artifact. |
| Continuation veto | Broken model equations/chart, no typed terminal-state route, failed exact forecast oracle, unavailable required GPU/XLA execution, unrecoverable artifact mismatch, or an expanded block that remains unidentifiable across its reviewed repair budget. Exhausting a bounded NeuTra candidate/compute repair budget closes the current M2 candidate branch and forces an M1-only handoff; it does not invalidate the ordinary-HMC model path or the research direction. |
| Repair trigger | Local target/forecast defect, sampler-specific tuning failure, dense-IAF training failure, inadequate equivalence power, or one failed expansion rung while earlier rungs remain valid. |
| What must not be concluded | Parameter-posterior correctness from predictive equivalence, exact nonlinear likelihood, full-parameter identifiability, empirical model adequacy from same-data equivalence, NeuTra superiority, broad dimensional scalability, default readiness, or Zhao-Cui source faithfulness. |

## Global Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can one typed TensorFlow/XLA SSL-LSTM target reproduce the locked scalar fixture and support filtering and posterior forecasting without benchmark-script imports? |
| Scalar scientific question | For the locked four-parameter scalar SVD-UKF target, do separately valid ordinary HMC and exact-corrected NeuTra-HMC arms induce practically equivalent 1-to-10-step posterior-predictive laws? |
| Exact computational baseline | A fresh, independently tuned, four-chain ordinary MAP-local HMC run on the locked target under trusted GPU/XLA execution. The old Phase 2V CPU-hidden chain is context only. |
| Comparator | A separately trained frozen dense-IAF transport followed by independently tuned, exact-Jacobian-corrected four-chain NeuTra-HMC on the identical target. |
| Primary M1 criterion | A0-A5 engineering and ordinary-HMC gates pass; A9 repeated synthetic held-out calibration passes; applicable A10 product tests pass. |
| Primary M2 criterion | M1 passes; A6-A8 produce valid NeuTra-HMC confirmation and fresh-seed predictive-equivalence audit; A9 passes; A10 closes the bounded scalar path. |
| Hard vetoes | Target/data/mask/signature mismatch; failed model, derivative, terminal-state, or forecast oracle; nonfinite values; invalid covariance; missing native divergence telemetry; positive native divergences; failed R-hat/ESS/MCSE or movement gates; transport roundtrip/Jacobian failure; invalid uncertainty procedure; changed frozen confirmation design; data leakage; corrupt or missing required artifact. |
| Explanatory only | Training/validation loss, acceptance inside a non-veto range, runtime, parameter summaries, higher predictive moments, extreme quantiles, local geometry residuals, and descriptive differences without uncertainty support. |
| What will not be concluded | Predictive equivalence does not prove parameter-posterior equality or correctness, identification, exact nonlinear likelihood, model adequacy on real data, sampler superiority, dimensional scalability, public/default readiness, or Zhao-Cui source faithfulness. |
| Preservation artifacts | Reviewed plans/results under `docs/plans/`, compact review bundles under `docs/reviews/`, generated review logs under `.claude_reviews/`, and dated structured run artifacts under `docs/plans/artifacts/`. |

Every phase must restate the applicable slice of this contract before any
implementation or runtime. A phase may strengthen these gates but must not
silently weaken or replace them.

## Five Separate Evidence Ledgers

| Ledger | Question | Primary evidence |
| --- | --- | --- |
| Engineering correctness | Does the implementation evaluate the declared model and forecast law? | Equation-to-code tests, finite-difference derivative checks, static shapes, seed replay, scalar/batch parity, eager/XLA parity, analytic LGSSM oracle, and artifact/signature validation. |
| Sampler validity | Does each HMC arm plausibly explore its declared target? | Native divergences, finite telemetry, movement, acceptance screen, four-chain R-hat, bulk/tail ESS, MCSE, independent initialization, and retained chain identity. |
| Computational equivalence | Do the valid ordinary and NeuTra arms induce the same useful predictive law? | Simultaneous mean/log-variance equivalence and independent-bank joint-path MMD equivalence under a frozen calibrated design. |
| Synthetic generative calibration | On repeated in-class simulated data, does the fitted scalar model give useful held-out predictive uncertainty? | Frozen repeated-data coverage, PIT/rank diagnostics, proper scores, and comparison with naive and true-parameter approximate-filter controls. |
| Empirical model adequacy | Does that predictive law describe unseen application observations usefully? | Rolling held-out log score, CRPS, interval coverage/calibration, and comparison with naive and tuned classical baselines. |

A pass in one ledger cannot substitute for a missing pass in another.

## Baseline Ladders

### Engineering And Sampler Ladder

1. Exact scalar linear-Gaussian state-space forecast oracle.
2. Fixed-parameter SSL-LSTM deterministic/noise-zero forecast fixture.
3. Ordinary MAP-local HMC, freshly tuned in its own coordinate.
4. Frozen affine transport HMC as a change-of-variables plumbing control.
5. Plain dense-IAF exact-corrected NeuTra-HMC, independently trained and tuned.

### Application Forecast Ladder

1. Naive persistence or application-appropriate seasonal persistence with a
   calibrated predictive variance.
2. Best tuned classical linear-Gaussian/AR state-space baseline under the same
   train/validation/audit split.
3. SSL-LSTM with ordinary HMC.
4. SSL-LSTM with NeuTra-HMC as a computational-equivalence arm, not a different
   scientific model.

A deterministic LSTM may be added only with a probabilistic forecast head and
the same proper-score protocol. It must not replace the classical baseline.

## Phase Index And Required Records

Only A0 is drafted initially. Later subplans are created just in time after the
preceding phase result is accepted.

| Phase | Name | Planned subplan | Required result |
| --- | --- | --- | --- |
| A0 | Governance, target, and artifact lock | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md` |
| A1 | Reusable masked posterior target | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md` |
| A2 | Typed terminal-state and forecast API | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md` |
| A3 | Forecast oracle and predictive statistics | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md` |
| A4 | Calibration and confirmation-design freeze | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-result-2026-07-11.md` |
| A5 | Fresh ordinary-HMC baseline | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a5-ordinary-hmc-baseline-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a5-ordinary-hmc-baseline-result-2026-07-11.md` |
| A6 | Dense-IAF training | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a6-dense-iaf-training-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a6-dense-iaf-training-result-2026-07-11.md` |
| A7 | Exact-corrected NeuTra-HMC | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a7-exact-neutra-hmc-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a7-exact-neutra-hmc-result-2026-07-11.md` |
| A8 | Blinded predictive confirmation and audit | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a8-predictive-confirmation-audit-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a8-predictive-confirmation-audit-result-2026-07-11.md` |
| A9 | Repeated synthetic generative calibration | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a9-synthetic-generative-calibration-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a9-synthetic-generative-calibration-result-2026-07-11.md` |
| A10 | Scalar productization and closeout | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a10-scalar-productization-closeout-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a10-scalar-productization-closeout-result-2026-07-11.md` |

Track B parameter expansion, dimensional lift, and Track C application work
require separately reviewed child programs after their stated entry gates.
Their detailed subplans must not be prewritten from scalar pre-execution
assumptions.

## Critical Path

### Track A: Finish The Scalar Vertical Slice

Track A executes and narrows Phases 0-10 of the predictive-equivalence master
program. The concrete work order is below.

#### A0. Governance, Target, And Artifact Lock

Deliverables:

- a focused reviewed Phase 0 subplan;
- stable hashes for observations, full parameter chart, four-parameter mask,
  prior, SVD-UKF numerical settings, Phase 2S map, and forecast semantics;
- separate semantic target, implementation/execution, sampler-geometry, and
  forecast-design signatures so a tuning artifact cannot redefine the target;
- exact code anchors for every reused target and HMC component;
- a decision on which existing untracked historical artifacts are inputs,
  archival context, or excluded;
- a dirty-worktree isolation strategy before implementation.

Pass:

- one target identity is unambiguous and replayable;
- the old one-chain CPU Phase 2V result remains diagnostic only;
- the failed independent-reference branch is not a prerequisite.

The locked prior is the historical unnormalized Gaussian log kernel
`-0.5 * sum((free - truth_free)^2 / 4.0^2)`. Adding the parameter-independent
normalizing constant would not change HMC, but it would change replayed target
values, so A1 must preserve the historical value convention. The Phase 2S
center, scale, and covariance are sampler-initialization/tuning context only;
they are not members of the target-semantic signature.

#### A1. Extract A Reusable Masked SSL-LSTM Posterior Target

Add a production-owned internal module, provisionally:

- `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py`.

Required types and functions:

- `SSLLSTMParameterMask`, storing names and full-chart indices;
- `SSLLSTMPosteriorConfig`, binding prior, observations, filter backend, dtype,
  JIT, and numerical policy;
- free-to-full embedding and full-to-free extraction;
- compiled `value`, `score`, and `value_and_score` surfaces;
- stable target and parameter-mask signatures;
- finite-reject behavior for invalid parameters.

Do not import benchmark scripts into production modules. Reuse their fixture
values through a narrow testing fixture or serialized manifest.

Tests:

- exact four-parameter embedding parity with the historical scalar target;
- parameter-order and mask failure cases;
- analytic score versus central finite differences at MAP, prior, and shell
  points;
- eager versus CPU-XLA parity as a reference check;
- GPU/XLA value/score canary under trusted execution;
- target replay at the Phase 2S center.

#### A2. Implement Typed Terminal-State And Forecast APIs

Add, provisionally:

- `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py`.

Required types:

- `SSLLSTMTerminalState`;
- `SSLLSTMForecastConfig`;
- `SSLLSTMInnovationBank`;
- `SSLLSTMForecastPaths`;
- `SSLLSTMForecastProvenance`.

Required behavior:

1. Rerun the SVD-UKF value filter for each parameter draw with
   `return_filtered=True`.
2. Extract the final filtered Gaussian mean/covariance through typed fields.
3. Draw terminal states and propagate the structural SSL-LSTM transition.
4. Add process noise only to the stochastic latent coordinate.
5. Complete hidden and cell coordinates deterministically.
6. Add observation noise and retain complete forecast paths.
7. Support shared and independent stateless innovation banks without confusing
   their inferential roles.
8. Default the TensorFlow execution path to `jit_compile=True`.

Tests:

- shape, dtype, horizon, and batch contracts;
- deterministic/noise-zero direct-recursion parity;
- fixed-seed exact replay and changed-seed sensitivity;
- process-noise placement invariant;
- scalar versus batched posterior-draw parity;
- terminal-state log-likelihood parity with the target;
- invalid or nonfinite covariance fail-closed tests;
- eager debug versus XLA parity.

#### A3. Build The Forecast Oracle And Predictive-Statistics Engine

Add, provisionally:

- `bayesfilter/inference/predictive_equivalence.py`;
- focused analytic LGSSM forecast fixtures under `bayesfilter/testing/`.

Required statistics:

- horizon means, variances, log variances, central moments, quantiles, and
  cross-horizon covariance;
- standardized path features;
- fixed multi-bandwidth RBF MMD U-statistic and a separately labeled biased
  V-statistic diagnostic;
- chain-aware batch means and hierarchical block/bootstrap resampling;
- simultaneous Bonferroni/studentized and max-statistic intervals;
- explicit `PASS`, `MATERIAL_DIFFERENCE`, `INCONCLUSIVE_UNDERPOWERED`, and
  `INVALID_HARD_VETO` states.

Oracle and statistical tests:

- analytic 1-to-10-step LGSSM mean and full covariance;
- identical-law null pairs;
- controlled mean, variance, skewness, and cross-horizon dependence changes;
- signed finite-sample MMD U-statistic behavior;
- bootstrap reproducibility, chain/block/forecast cluster semantics, coverage,
  and power;
- singular covariance and ridge fail-closed behavior;
- proof by test that failure to reject equality cannot emit an equivalence
  pass.

No SSL-LSTM sampler comparison proceeds if the LGSSM oracle fails.

#### A4. Calibrate And Freeze The Confirmatory Design

Use a reviewed calibration-only run to freeze:

- horizons `1..10` and equal horizon weights;
- mean scales and log-variance margins;
- forecast replication count;
- MMD bandwidths, mixture weights, and tolerance;
- block length, bootstrap type/count/seed, and confidence level;
- covariance ridge and condition-number cap;
- minimum chains, retained draws, ESS, and MCSE;
- all confirmation sampler and forecast seeds.

Calibration inputs:

1. exact LGSSM null and perturbed pairs;
2. identical SSL-LSTM posterior draws with independent forecast banks;
3. split-half calibration-only ordinary-HMC draws;
4. controlled forecast-relevant parameter and terminal-state perturbations;
5. prospective power curves over draw and forecast counts.

Stop if the design cannot distinguish simulation noise from at least one
predeclared materially different forecast law at the available budget. Do not
widen margins after seeing NeuTra confirmation results.

#### A5. Run A Fresh Ordinary-HMC Baseline

Run a fresh four-chain MAP-local baseline under trusted GPU/XLA execution.

Required gates:

- target and coordinate hashes match A0/A1;
- separate tuning and retained-sampling artifacts;
- four dispersed independent chains;
- native divergence telemetry present and zero positive divergences;
- all values, gradients, states, and log-accept telemetry finite;
- reviewed acceptance, movement, R-hat, bulk/tail ESS, and MCSE gates pass;
- chain-shaped samples and all seed identities are retained.

Failure is a sampler/tuning repair trigger. It is not evidence against the
forecast validation idea unless the target itself is invalidated.

#### A6. Implement And Validate Dense-IAF Training

Before drafting the A6 subplan, repeat the NeuTra governance audit against the
then-current BayesFilter policy and the exact `dsge_hmc` Gate-1 closure
artifacts. Historical `dsge_hmc` results are design/source context only and do
not validate a BayesFilter trainer or this SSL-LSTM target. A6 is restricted to
the plain canonical reverse-KL baseline; any enhanced objective or architecture
claim requires a separately reviewed gate. If current governance requires an
unmet predecessor gate or human override, A6 stops without training.

Add a minimal BayesFilter-owned trainer, provisionally:

- `bayesfilter/inference/neutra_training_tf.py`.

Requirements:

- TensorFlow/TFP only, `float64` for target parity, GPU execution, and XLA JIT
  enabled by default;
- canonical reverse-KL plain NeuTra objective;
- dense autoregressive IAF components plus fixed mixing/permutation layers;
- independent stateless training and validation base-noise banks;
- checkpoint/resume with exact optimizer and RNG state;
- output finalized through
  `finalize_dense_iaf_neutra_artifact_payload`;
- target, topology, tensor, training-state, and data hashes;
- finite loss, gradient, parameter, inverse, roundtrip, and log-Jacobian gates.

Baseline tests:

- standard Gaussian identity recovery;
- correlated Gaussian affine and dense-IAF controls;
- bimodal or heavy-tail negative control showing reverse-KL undercoverage can
  be detected rather than hidden;
- serialized artifact reload and forward/inverse/log-Jacobian parity;
- deterministic fixed-seed replay;
- CPU-only tiny smoke clearly labeled reference-only;
- trusted GPU/XLA training canary.

Training loss nominates a frozen transport. It is never posterior or predictive
evidence.

The A6 subplan must predeclare a finite candidate-family and compute/repair
budget before training. If no transport passes nomination within that budget,
write a terminal current-candidate rejection result, keep M2 blocked, skip A7
and A8, and continue only through the ordinary-HMC A9/A10 path toward a possible
M1 closeout. Any later M2 attempt requires a new reviewed NeuTra repair branch;
the current phase may not widen its architecture, objective, or budget after
seeing failed candidates.

#### A7. Run Exact-Corrected NeuTra-HMC

For the frozen transport:

- verify affine-control transformed-target parity first;
- bind the exact original target plus log Jacobian;
- tune the NeuTra chain independently in transport coordinates;
- run four independent GPU/XLA chains;
- transform retained draws back to free and full parameter charts;
- apply the same sampler-validity gates as A5;
- retain transport, tuning, target, and chain hashes.

A failure rejects or repairs the current transport/tuning candidate. It does
not reject ordinary HMC or the SSL-LSTM target.

The A7 subplan must likewise freeze a finite tuning/repair budget and terminal
disposition before HMC runs. Exhausting that budget rejects the current
exact-corrected NeuTra-HMC candidate, keeps M2 blocked, skips A8, and hands the
valid ordinary-HMC path to A9/A10 for M1-only closeout. It is not a continuation
veto for the SSL-LSTM implementation. A transform-only defect vetoes the
NeuTra/M2 branch. It affects the M1 path only if investigation exposes a defect
in the shared target or forecast harness, or if the execution environment
required by the ordinary-HMC path is unavailable.

#### A8. Blinded Predictive Confirmation And Audit

Decision order:

1. engineering and forecast vetoes;
2. ordinary- and NeuTra-HMC validity vetoes;
3. simultaneous horizon mean/log-variance equivalence;
4. independent-bank joint-path MMD equivalence;
5. robustness of paired moment conclusions under independent arm-specific
   forecast banks;
6. explanatory higher moments, quantiles, covariance, parameters, and runtime.

After a confirmation pass, repeat once with fresh sampler and forecast seeds
without changing the frozen design. Disagreement between confirmation and audit
blocks M2; do not average the two runs post hoc.

#### A9. Repeated Synthetic Generative Calibration

Sampler-to-sampler agreement is not enough to call the model validated. Before
an M1 or M2 closeout, run a reviewed repeated-data synthetic calibration with
independent simulation seeds that were not used for target, transport,
sampler, margin, or bandwidth tuning.

A9 planning and synthetic-data/oracle preparation may begin after the A5
ordinary-HMC baseline and all forecast/statistical oracles pass. Evidence-
bearing A9 execution waits until the NeuTra/M2 branch has either passed A8 or
reached a recorded unavailable/blocked/rejected state, so the representative-
sampler rule is frozen before A9 outputs are opened. A9 does not require a
successful NeuTra candidate and can support an M1 engineering closeout while
A6-A8 are unavailable, blocked, rejected, or fail their M2-only gates.

Freeze before execution:

- the generating parameter distribution or fixed generating configurations;
- the number of data-set replications, observed-history length, and held-out
  future length;
- the exhaustive representative-sampler rule: use ordinary HMC whenever that
  arm remains valid and the NeuTra/M2 branch is unavailable, governance-blocked,
  nomination/tuning-rejected, invalid, materially different, underpowered, or
  fails confirmation/audit replication; only after A8 equivalence may a
  predeclared representative-arm rule or a rule for evaluating both arms be
  used without double-counting them;
- predictive interval levels, PIT/rank bins, proper scores, and simultaneous
  uncertainty method;
- naive persistence, true-parameter SVD-UKF, and true-parameter/true-terminal-
  state generative forecast controls;
- pass, fail, and inconclusive thresholds.

Required evidence:

- empirical coverage of held-out future observations with uncertainty across
  independent simulated data sets;
- PIT or predictive-rank calibration appropriate to continuous forecasts;
- log predictive score and CRPS relative to the naive control;
- comparison with the true-parameter SVD-UKF forecast as an approximation-gap
  diagnostic, not an exact nonlinear oracle;
- comparison with the true-parameter, simulated-true-terminal-state forecast
  as an exact conditional generative simulator control, while recognizing that
  conditioning on the true latent state is easier than conditioning only on
  observed history;
- failure localization separating sampler, finite-data, prior, SVD-UKF
  approximation, and model-simulation explanations.

This gate assesses in-class predictive calibration. It still does not prove
the parameter posterior is correct or that the model is adequate for real
data. If too few independent data sets are affordable for uncertainty-aware
coverage, emit `INCONCLUSIVE_SYNTHETIC_CALIBRATION`, not a pass.

#### A10. Scalar Productization And Closeout

For an M1 closeout, require A0-A5, A9, and the applicable product tests below.
For an M2 closeout, additionally require A6-A8. Then:

- expose internal `fit`, `filter`, and `forecast` orchestration without changing
  package defaults;
- serialize target, mask, sampler, terminal-state, and forecast provenance;
- add a complete simulated-data example;
- add checkpoint/restart and corrupt-artifact tests;
- add one bounded end-to-end GPU/XLA integration test;
- write a result note, decision table, inference-status table, run manifests,
  and reset memo.

If NeuTra fails but ordinary HMC, the model/forecast APIs, and A9 synthetic
calibration pass, close M1 as engineering-complete only. Internal orchestration
and examples may be preserved, but M2 remains blocked and no validated scalar
release claim is made. This rejects the current acceleration/replication
candidate without rejecting the SSL-LSTM model implementation.

## Track B: Expand The Trainable Model Safely

Track B implementation may begin after M1 closes, but expanded-mask promotion
waits for M2. It must use the masked target from A1 rather than create a new
scalar implementation.

### Proposed Scalar Parameter-Block Ladder

The full scalar chart has 24 entries. Use the following candidate ladder, with
the exact data-length and prior design frozen in a reviewed subplan:

| Rung | Trainable blocks | Dimension | Purpose |
| --- | --- | ---: | --- |
| B0 | Latent and observation mean maps `(A,d,C,e)` | 4 | Validated M2 baseline. |
| B1 | B0 plus process and observation log standard deviations | 6 | Learn forecast location and scale. |
| B2 | B1 plus initial mean and initial log standard deviations | 12 | Test initial-state uncertainty without neural gate freedom. |
| B3 | B2 plus four LSTM gate biases | 16 | Admit nonlinear gate levels. |
| B4 | B3 plus four input and four recurrent gate weights | 24 | Full scalar chart. |

Candidate synthetic sequence lengths are `T=30`, `100`, and `300`, but they are
calibration rungs, not automatic promotion settings. A rung advances only when
the data regime has prospective power for that block.

### Per-Rung Gates

For each rung:

1. regenerate simulation, calibration, confirmation, and audit data from
   independent stateless seeds;
2. verify target/score and forecast oracles for the new mask;
3. evaluate local information/Hessian rank and condition after transforms;
4. use prior-to-posterior contraction and simulation-based rank/coverage
   diagnostics to identify weak blocks;
5. run ordinary HMC for engineering/identifiability exploration; if the rung is
   to be promoted as M3-supported, train a separately governed target-specific
   NeuTra arm and require both arms to pass their own validity gates;
6. for every promoted rung, freeze and pass ordinary-HMC versus NeuTra-HMC
   predictive equivalence at the rung's recalibrated budget;
7. evaluate recovery of identifiable functionals, not raw neural weights alone;
8. repeat across multiple simulated data sets before promotion.

Promotion as an M3-supported mask requires both sampler arms' validity,
ordinary-HMC versus NeuTra-HMC computational equivalence under the rung's
frozen design, calibrated synthetic predictive behavior, and no identifiability
veto for the newly admitted block. An ordinary-HMC-only rung may remain an
engineering/identifiability candidate but is not M3-promoted. A failed
B2/B3/B4 rung or its NeuTra arm does not revoke earlier rungs. Freeze,
regularize, reparameterize, or exclude the failed block and preserve the
highest fully supported mask.

### Identifiability Repairs

Allowed repairs under a reviewed branch:

- stronger scientifically justified priors;
- non-centered or whitened parameterizations;
- longer or replicated sequences;
- fixing weak initial-state blocks;
- blockwise or low-rank gate parameterizations;
- explicit symmetry constraints when derived in the project notation.

Not allowed:

- declaring all 24 parameters supported because forecasts look similar;
- selecting a favorable parameter mask after opening audit results;
- using NeuTra training loss as identifiability evidence;
- treating a failed rung as evidence that the scalar model or prior rungs are
  invalid.

### Dimension-Lift Ladder

Completing all 24 scalar parameters is not evidence that the implementation is
usable beyond scalar dimensions. After at least one expanded scalar mask
passes, use a separate static-dimension ladder:

| Rung | `(latent_dim, hidden_dim, observation_dim)` | Full-chart size | Initial role |
| --- | --- | ---: | --- |
| D0 | `(1,1,1)` | 24 | Scalar reference. |
| D1 | `(1,2,1)` | 49 | Hidden-memory lift while retaining scalar latent/output laws. |
| D2 | `(2,2,1)` | 64 | Multivariate latent lift with scalar observation. |
| D3 | `(2,4,2)` | 152 | Small multivariate stress candidate, not an automatic HMC target. |

The full-chart size for dimensions `(k,h,d)` is
`5*k*h + 4*h*h + 8*h + 4*k + d*k + 2*d`. The formula and parameter-name order
must be unit-tested rather than inferred from a serialized artifact.

At each rung:

- first pass full-chart shape, transition, observation, derivative, filter,
  terminal-state, forecast, batch, and GPU/XLA tests;
- choose a trainable mask through a reviewed identifiability/data-power plan;
- never infer that the entire 49-, 64-, or 152-entry chart is estimable merely
  because the full chart compiles;
- recalibrate predictive mean/log-variance features for every observation
  component and horizon;
- include cross-output and cross-horizon dependence through the standardized
  joint-path MMD and explanatory covariance blocks;
- recalibrate margins, MMD bandwidths, bootstrap hierarchy, sampler budget,
  and forecast replication count for the new dimension;
- record peak memory, compilation time, gradient time, and forecast throughput
  before serious sampling.

D3 is a stress candidate only. If its prospective HMC or uncertainty budget is
not viable, stop at D2 or select an application-derived dimension/mask under a
new plan. Dimensional-lift failure does not revoke scalar correctness.

## Track C: Establish Application Readiness

Track C may begin data-interface work after A2, but no empirical claim is made
until M2 and the selected Track-B mask pass.

### C0. Data Contract

Implement a typed data adapter that records:

- timestamps, frequency, feature/observation names, and missingness;
- train-only normalization and inverse transformation;
- train, tuning/validation, confirmation, and final audit windows;
- data version/hash and preprocessing configuration;
- exogenous-input policy, if later added under a separate model extension.

No audit observation may tune priors, architecture, forecast horizon, margins,
or sampler settings.

### C1. Rolling-Origin Forecast Harness

For each frozen forecast origin:

- refit or update according to a predeclared policy;
- produce 1-to-H posterior forecast paths;
- retain origin, chain, draw, and forecast-cluster identities;
- compute log predictive score, CRPS, interval coverage, calibration curves,
  and application-specific point error;
- aggregate uncertainty across forecast origins rather than treating paths from
  one origin as independent replications.

### C2. Fair Model-Adequacy Comparison

Use the application forecast baseline ladder. Predeclare:

- the primary proper score;
- practical difference or non-inferiority margin;
- paired origin-level uncertainty method;
- hard calibration/coverage vetoes;
- compute and tuning budgets for every model;
- what constitutes no decision because the audit set is too small.

NeuTra and ordinary HMC are two computations for the same SSL-LSTM model. They
must not be counted as two scientific competitors after predictive equivalence
is established.

### C3. API And Release Boundary

Only after C2:

- finalize a bounded configuration object and result schema;
- document `fit`, `filter`, `forecast`, checkpoint, and reload behavior;
- add shape/dtype/device/JIT compatibility tests across supported dimensions;
- add missing-data and invalid-input tests;
- add performance and memory envelopes on the default GPU/XLA route;
- add an end-to-end tutorial and reproducible manifest;
- decide separately whether the path remains optional or can be proposed as a
  default.

A default-policy proposal requires a new evidence review. This roadmap can
finish an optional application-ready feature without changing defaults.

## Test Matrix

| Layer | Minimum tests before promotion |
| --- | --- |
| Model equations | Transition/observation direct fixtures, analytic Jacobians and parameter derivatives, covariance transforms, static shapes. |
| Target | Mask embedding, value/score finite differences, prior inclusion, invalid-region finite reject, eager/XLA and scalar/batch parity. |
| Filter/terminal state | Log-likelihood parity, final filtered mean/covariance extraction, PSD/finite checks, deterministic completion. |
| Forecast | Analytic LGSSM oracle, deterministic SSL-LSTM recursion, seed replay, process/observation noise placement, batched draw parity. |
| Statistics | Known moments, MMD fixtures, dependence-aware bootstrap coverage/power, equivalence status logic, singular-covariance failure. |
| Ordinary HMC | Tuning separation, native divergence telemetry, movement, chain shape, R-hat/ESS/MCSE, retained archive integrity. |
| NeuTra training | Gaussian controls, finite gradients, checkpoint/restart, inverse/roundtrip/Jacobian, artifact hashes, GPU/XLA placement. |
| Exact NeuTra-HMC | Exact transformed-target parity, independent tuning, sampler gates, back-transform and forecast invariance. |
| Synthetic calibration | Independent simulated data sets, held-out coverage/PIT, proper scores, naive and true-parameter approximate-filter controls, replication-level uncertainty. |
| Parameter expansion | Per-mask score/forecast parity, information/coverage diagnostics, multiple simulated data sets, rung-specific predictive gate. |
| Application | No-leak preprocessing, rolling-origin splits, proper-score calculations, baseline parity, paired uncertainty, audit immutability. |
| Product | Serialization, corrupted artifact rejection, restart determinism, bounded end-to-end GPU/XLA test, docs build. |

## Execution And Artifact Policy

Every evidence-bearing phase must have a reviewed subplan before runtime and
must write JSON plus Markdown under a dated artifact directory. Serious run
manifests must include:

- git commit and dirty status;
- exact command, environment, TensorFlow/TFP versions, and conda environment;
- CPU/GPU status and trust basis;
- dtype, device, JIT/XLA, and TF32 settings;
- data, target, mask, forecast, transport, sampler, and design hashes;
- all simulation, training, sampler, forecast, and bootstrap seeds;
- chain/draw/forecast/origin counts and wall time;
- input/output artifact paths, plan, result, and explicit nonclaims.

NeuTra training is GPU/XLA work. External replay/training sample generation is
a separate multicore CPU lane with worker counts, seeds, and hashes. CPU-only
training may be used only for a tiny labeled smoke and cannot support quality,
HMC, or scientific claims.

## Numeric Default Provenance

| Choice | Value/status | Provenance | Role and boundary |
| --- | --- | --- | --- |
| Scalar observed history | `T=30` | Inherited from the locked scalar fixture and committed predictive-equivalence program | Target identity, not evidence that 30 points identify expanded masks. |
| Scalar free mask | `(A,d,C,e)`, dimension 4 | Inherited from the Phase 2S target and committed program | M1/M2 estimand only. |
| Prior standard deviation | `4.0` per free coordinate | Inherited from the locked scalar target | Must be hashed in A0; changing it changes the target. |
| Forecast horizon | `H=10` | Owner-selected initial design in the predictive-equivalence program | Frozen only after A4 calibration supports adequate power. |
| Horizon weights | `0.1` each | Derived from equal weights over `H=10` | Scientific weighting hypothesis; cannot be tuned to hide a horizon failure. |
| Serious HMC chains | 4 | Inherited reviewed design for chain-aware diagnostics | Minimum confirmation design, subject to A4 power/budget review. |
| Native divergences | zero positive divergences | Inherited hard veto from the predictive-equivalence program | Sampler validity veto, not a superiority metric. |
| Tensor dtype | `float64` | Target-parity requirement inherited from the scalar score path | Default for this target unless a separate parity phase admits another dtype. |
| XLA | `jit_compile=True` | Repository owner directive | Default for BayesFilter algorithmic execution; non-JIT is debug/reference only. |
| MMD bandwidth factors | `(0.5,1.0,2.0)` times calibration median | Committed predictive-equivalence design | Candidate calibration rule; numerical bandwidths freeze in A4. |
| Claude material-review rounds | maximum 5 per same blocker | Owner instruction | Review-loop cap, not permission to bypass unresolved findings. |
| Initial Claude probe timeout | 90 seconds | Local review-gate guide recommendation | Operational reviewed default; may be increased once after a recorded probe timeout. |
| Expanded sequence lengths | `30,100,300` | Roadmap convenience ladder | Hypotheses only until a Track B subplan derives or calibrates them. |
| Dimension-lift rungs | D0-D3 | Roadmap engineering ladder | Shape/stress hypotheses, not estimability or production commitments. |

All other material thresholds, timeouts, sample counts, learning rates,
architectures, bootstrap counts, equivalence margins, and runtime budgets remain
`UNSET_REQUIRES_PHASE_SPECIFIC_PROVENANCE` until their reviewed subplans define
them.

## Required Phase Result Records

Every material phase result must include:

1. a decision table with decision, primary criterion status, veto status, main
   uncertainty, next justified action, and forbidden conclusions;
2. an inference-status table covering hard vetoes, statistically supported
   rankings, descriptive-only differences, default readiness, and next evidence;
3. separate engineering, sampler/numerical, computational-equivalence,
   synthetic-calibration, and empirical-adequacy ledgers as applicable;
4. the serious-run manifest fields listed above, using `N/A` only when a field
   genuinely does not apply;
5. candidate failure versus research-direction status;
6. a post-run red-team note naming the strongest alternative explanation, what
   evidence would overturn the decision, and the weakest evidence component;
7. exact result, log, review, and next-subplan paths.

## Decision And Stop Rules

| Observed result | Interpretation | Next action |
| --- | --- | --- |
| Target/forecast oracle failure | Engineering comparison invalid. | Stop downstream work and repair the implementation. |
| Ordinary HMC hard veto | No valid baseline. | Repair target/tuning/sampler; do not train NeuTra for confirmation. |
| NeuTra training screen failure | Current transport not nominated. | Repair or reject current architecture; ordinary HMC path may remain viable. |
| NeuTra-HMC hard veto | Current transformed sampler invalid. | Repair tuning/transport or reject NeuTra candidate; do not interpret predictive differences. |
| Valid samplers, predictive material difference | Computational equivalence failed. | Localize horizons/features, then repair the failing sampler/candidate under a new plan. |
| Wide intervals outside margins | Underpowered, not equivalent. | Increase predeclared budget only through a new calibration plan. |
| Ordinary HMC and synthetic calibration pass, NeuTra fails | M1 engineering core is viable but not computationally replicated. | Preserve internal scalar path; repair or reject NeuTra before any M2 claim. |
| Scalar equivalence passes and audit replicates | M2 computational-equivalence gate passes, but model calibration and product closeout are not yet established. | Run A9 repeated synthetic generative calibration, then A10; M2 remains pending. |
| A8 and A9 pass | All M2 validation-evidence gates have passed, but M2 is not closed until A10 passes. | Run A10 productization and closeout without weakening the claim boundary. |
| A8, A9, and applicable A10 gates pass | M2 scalar vertical slice is validated within its stated boundary. | Close M2; only then promote blockwise expansion results beyond engineering-candidate status. |
| Synthetic calibration fails | The computations may agree on a biased or poorly calibrated approximate model. | Localize prior/filter/data/sampler causes; do not productize as validated. |
| Expanded block unidentifiable | That mask/rung is unsupported. | Freeze/reparameterize/exclude block; retain earlier valid rung. |
| Predictive equivalence passes but held-out scores fail | Computation agrees on an inadequate model. | Reject or revise the application/model specification, not the sampler implementation. |
| Held-out difference is descriptive only | No supported model ranking. | Collect more forecast origins or report inconclusive evidence. |

## Skeptical Plan Audit

| Risk | Audit finding and control |
| --- | --- |
| Wrong baseline | Repaired by separate engineering/sampler and application ladders. Phase 2V is not the HMC baseline; naive and tuned classical forecasts are required for adequacy. |
| Proxy promoted | Training loss, smoke tests, local geometry, same-data predictive equivalence, and parameter similarity cannot establish model adequacy or full correctness. |
| Missing stop conditions | Oracle, target, sampler, equivalence-power, identifiability, data-leakage, and artifact continuation vetoes are explicit. |
| Unfair comparison | HMC arms are independently tuned but share target/data/forecast/design; application models share splits, proper scores, and tuning budgets. |
| Hidden assumptions | The scalar SVD-UKF is an approximate-likelihood, four-parameter, simulated-data milestone. Full 24-parameter support is evidence gated. |
| Stale context | The roadmap supersedes the 2026-07-04 launch-smoke boundary and does not revive the failed 2026-07-09 reference branch. |
| Environment mismatch | Serious training and sampling are GPU/XLA; CPU exceptions are restricted and labeled. |
| Artifact mismatch | Every serious phase binds data, target, parameter mask, forecast, transport, sampler, and design hashes. |
| Two wrong samplers agree | Shared implementation is protected by analytic LGSSM forecasts, direct SSL-LSTM recursion, derivative checks, separate sampler gates, and repeated synthetic held-out calibration. |
| Predictive agreement hides nonidentifiability | Track B has independent blockwise information, coverage, and contraction gates; predictive equivalence cannot approve a parameter block. |
| Same-data agreement mistaken for usefulness | Track C reserves rolling held-out model-adequacy evidence and classical baselines. |
| Full model forced onto insufficient data | Parameter masks and data-length/power ladders can stop at the highest supported rung. |

Audit status:
`PASSED_FOR_ROADMAP_PROPOSAL_AND_PHASE_A0_PLANNING_ONLY`.

## Pre-Mortem

The program could pass misleadingly if both samplers share a broken target,
equivalence margins are too wide, common random numbers hide sensitivity, an
in-class synthetic check uses too few independent data sets, all 24 parameters
are declared trainable despite weak information, or same-data forecast
agreement is presented as real-data skill. The cheapest discriminators are the
analytic LGSSM oracle, direct deterministic SSL-LSTM recursion, controlled
alternative power curves, independent innovation banks, repeated synthetic
held-out calibration, blockwise information/coverage diagnostics, and locked
rolling audit windows.

The program could fail for engineering or tuning reasons if terminal-state
extraction binds the wrong filter, XLA cannot compile the nested forecast path,
native divergence telemetry is missing, dense-IAF inversion is unstable, or
the hierarchical bootstrap is underdetermined. Tiny target/forecast canaries,
affine transport controls, runtime/memory pilots, and fail-closed covariance
diagnostics should localize those failures before serious runs.

## Immediate Next Three Changes

1. **A0 documentation-only target lock:** write the reviewed Phase A0 subplan,
   inventory exact source/artifact hashes, and define the scalar target
   signature. No runtime.
2. **A1 target extraction:** implement the reusable typed parameter mask and
   posterior target, with exact replay against the historical four-parameter
   fixture and focused CPU/XLA tests.
3. **A2 forecast API:** implement typed terminal-state extraction and stateless
   multi-horizon path simulation, then pass deterministic, batch, and LGSSM
   oracle prerequisites before starting any new HMC or NeuTra run.

After those changes, implement A3 and run calibration A4. Do not start serious
ordinary HMC, dense-IAF training, or NeuTra-HMC before the confirmatory design
and all upstream oracles are frozen.
