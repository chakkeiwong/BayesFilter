# Scalar SSL-LSTM Predictive-Equivalence Master Program

Date: 2026-07-11

Status: `DRAFT_REVIEW_REQUIRED_NO_RUNTIME_AUTHORIZED`

## Scope And Classification

This program replaces the blocked requirement for a valid independent
four-parameter posterior reference with a narrower functional question:

> Do independently parameterized ordinary HMC and frozen-transport,
> exact-corrected NeuTra-HMC induce practically equivalent posterior-predictive
> laws for the same scalar SSL-LSTM SVD-UKF filtering target?

This is an `extension_or_invention` lane. It is not Zhao-Cui source-faithfulness
work. It does not declare parameter-posterior agreement, posterior correctness,
structural identification, SVD-UKF exactness, empirical model adequacy, HMC
readiness, NeuTra superiority, GPU/default readiness, or a statistically
supported sampler ranking.

The mathematical design is documented in:

- `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`.

The prior parameter-reference branch remains closed as recorded in:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-reset-memo-2026-07-10.md`.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | For a fixed scalar SSL-LSTM SVD-UKF target and fixed data, are the joint 1-to-10-step posterior-predictive laws from ordinary MAP-local HMC and exact-corrected NeuTra-HMC practically equivalent under predeclared interpretable and omnibus criteria? |
| Candidate/mechanism | A target-specific dense-IAF NeuTra transport trained by GPU/XLA reverse KL, frozen before sampling, and corrected by the exact transformed target and Metropolis HMC. |
| Exact baseline | Fresh four-chain ordinary MAP-local HMC on the Phase 2S coordinate and target, retuned and rerun under the production GPU/XLA/native-telemetry route. The old Phase 2V one-chain CPU-hidden result is diagnostic context only. |
| Expected failure mode | Shared forecast bug; terminal-state replay mismatch; inadequate ordinary-HMC mixing; reverse-KL tail or mode undercoverage; transport roundtrip/log-Jacobian defect; NeuTra retuning failure; predictive mean/variance mismatch; MMD mismatch; insufficient equivalence power; post-hoc changes to horizons, features, weights, margins, or seeds. |
| Primary promotion criterion | Both sampler arms pass their own hard validity gates, then all simultaneous standardized predictive-mean and log-variance equivalence intervals for horizons 1 through 10 lie inside frozen margins, and the upper confidence bound for the frozen joint-path MMD lies below its calibrated tolerance on an independent confirmation artifact. |
| Promotion veto | Any target/signature mismatch, nonfinite target/score/sample/forecast, transform roundtrip or log-Jacobian failure, missing native divergence telemetry, positive native divergences, R-hat/ESS/MCSE failure, terminal filtered-state mismatch, changed confirmation design after opening arm results, invalid covariance/bootstrap estimate, or shared forecast oracle failure. |
| Continuation veto | Broken model equations or parameter chart; inability to expose a typed terminal filtered-state law; failed LGSSM forecast oracle; unavailable GPU/XLA route for NeuTra training or sampler confirmation; missing structured artifact; confirmatory design not frozen; or a hard sampler/target veto that invalidates predictive interpretation. |
| Repair trigger | Calibration shows inadequate power, singular feature covariance, unstable high-order moments, insufficient transport coverage, or a sampler-specific tuning failure while target/forecast oracles remain valid. |
| Explanatory diagnostics | Parameter summaries, third/fourth central moments, predictive quantiles, cross-horizon covariance, training/validation loss, raw MMD components, runtime, acceptance, ESS/R-hat/MCSE beyond veto thresholds, and descriptive differences between viable arms. |
| What must not be concluded | Parameter-posterior equality/correctness, global identification, exact nonlinear-model likelihood, model adequacy on new data, general HMC or NeuTra correctness, sampler superiority, default readiness, broad GPU/XLA readiness, or Zhao-Cui source faithfulness. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific/engineering question | Can the two exact-corrected HMC parameterizations yield the same practically relevant posterior-predictive law without relying on a failed independent parameter reference? |
| Baseline/comparator | Ordinary four-chain MAP-local HMC versus frozen dense-IAF exact-corrected NeuTra-HMC; same target, observations, prior, parameter chart, forecast implementation, horizon, and confirmation design. |
| Primary pass/fail criterion | Simultaneous mean/log-variance equivalence plus joint-path MMD equivalence, only after both sampler arms pass hard validity screens. |
| Veto diagnostics | Engineering, sampler, transform, forecast, artifact, and design-freeze vetoes listed in the research-intent ledger. |
| Explanatory only | High-order moments, quantiles, covariance differences, loss curves, runtime, and continuous sampler diagnostics after hard screens pass. |
| What will not be concluded | The nonclaims in the research-intent ledger remain binding even if the predictive gate passes. |
| Preservation artifact | Per-phase JSON/Markdown under `docs/plans/artifacts/scalar-ssl-lstm-predictive-equivalence-2026-07-11/`, plus reviewed subplans/results under `docs/plans/`. |

## Baseline Ladder

The method ladder is fixed before implementation:

| Rung | Role | Required evidence |
| --- | --- | --- |
| Linear-Gaussian scalar SSM | Forecast implementation oracle | Analytic 1-to-10-step mean/covariance and simulated-path checks. |
| SSL-LSTM at a fixed parameter and terminal Gaussian state | Nonlinear forecast mechanics | Deterministic/noise-zero recursion, seed replay, shape, finite, and scalar/batch parity. |
| Ordinary MAP-local HMC | Best tuned classical baseline | Fresh four-chain GPU/XLA run with native divergences, finite telemetry, movement, R-hat, ESS, and MCSE. |
| Affine frozen transport HMC | Transport plumbing control | Exact transformed-target parity, roundtrip/Jacobian checks, and prediction invariance under an analytically tractable transform. |
| Plain dense-IAF NeuTra-HMC | Proposed method | GPU/XLA training, frozen signed artifact, independently tuned exact-corrected four-chain HMC, then predictive equivalence. |

No enhanced NeuTra loss, replay-tail loss, sampler-aware fine tuning, or
sequential NeuTra-SMC arm is in the initial promotion ladder. Such work requires
a later reviewed branch after the plain candidate is decided.

## Locked Target And Forecast Semantics

### Model Target

- Observed history: the exact 30-step stateless simulated path bound to the
  scalar validation target.
- Filtering likelihood: the same historical SVD-UKF analytic-score target used
  by Phase 2S/2U. Moving to the promoted principal-square-root target would
  change the estimand and therefore requires a separate migration/tie-out phase.
- Free parameters:
  - `latent_mean_weight.0.0`;
  - `latent_mean_bias.0`;
  - `observation_weight.0.0`;
  - `observation_bias.0`.
- Fixed parameters: the remaining 20 scalar SSL-LSTM fixture entries.
- Prior: Gaussian center at the fixed simulation free-parameter vector with
  standard deviation `4.0` per free coordinate.
- Forecast horizon: `H=10` for the initial program.

### Terminal-State Law

For every posterior parameter draw, rerun the fixed SVD-UKF value filter with
`return_filtered=True` and take the final filtered mean/covariance as the
Gaussian terminal-state approximation. This is a posterior predictive law
relative to the same SVD-UKF approximation, not the exact nonlinear filtering
distribution.

Do not parse `TFFilterDerivativeResult.trace` as the production forecast API.
Introduce a typed predictive-state adapter whose target signature binds:

- observations and data hash;
- full parameter chart and four-free-parameter embedding;
- filter backend and numerical settings;
- terminal-state mean/covariance convention;
- forecast horizon and model transition/observation identity.

### Forecast Paths

For each posterior draw and each forecast replication:

1. draw `x_T` from the terminal Gaussian approximation;
2. propagate the structural SSL-LSTM transition for horizons `1..10`;
3. add process noise only to the stochastic latent coordinate;
4. deterministically complete hidden/cell states;
5. draw observation noise and emit `y_{T+h}`;
6. retain the entire length-10 path as one clustered observation.

Use one frozen stateless innovation bank for both arms in the paired
mean/log-variance comparison. Also retain frozen, independently generated
arm-specific innovation banks. The latter are the primary input for the MMD
gate and a robustness gate for the interpretable features, because a
common-random-number coupling is not part of either marginal predictive law.

## Predictive Features And Weights

### Interpretable Co-Primary Features

For each horizon `h=1..10`:

- predictive mean;
- predictive log variance, not raw variance, so the equivalence difference is
  dimensionless and symmetric for multiplicative scale errors.

Means are standardized by a frozen predictive standard-deviation scale. The
scale comes from calibration-only baseline/oracle artifacts and is bounded
below by a documented numerical floor. It must not be recomputed after opening
the confirmatory NeuTra arm.

### Omnibus Feature

Use the full standardized 10-step forecast path. The initial omnibus statistic
is the diagonal-excluded squared-MMD U-statistic computed from a frozen mixture
of Gaussian RBF kernels. It is unbiased for squared MMD under independent
sampling, but it can be negative in finite samples and is not generally
unbiased under MCMC dependence; the chain/block uncertainty procedure must
respect that dependence. Candidate bandwidths are the pooled calibration-only
path-distance median multiplied by fixed factors `(0.5, 1.0, 2.0)`. Freeze the
numerical bandwidths before confirmation. If the median is zero or nonfinite,
fire a calibration veto rather than silently substitute a post-hoc value.

The equivalent finite characteristic-feature representation may be stored for
diagnostics. The MMD and characteristic-function forms must not become two
independent promotion tests chosen after results are seen.

### Horizon And Precision Weights

- Scientific horizon weights: equal, `lambda_h=0.1`, for `h=1..10`.
- Component scaling: frozen calibration-only predictive scales.
- Precision weighting: chain-aware long-run covariance of the fixed feature
  difference, regularized as `S_lambda = S + lambda I`.
- Ridge rule: choose `lambda` by a calibration-only condition-number ladder and
  freeze the smallest candidate that keeps the covariance positive definite and
  below a reviewed condition-number cap.

Inverse-covariance weighting is explanatory/omnibus support. It does not
override a failed horizon-specific practical-equivalence margin.

### Explanatory Features

- third and fourth central moments;
- predictive quantiles `(0.05, 0.25, 0.50, 0.75, 0.95)`;
- the full cross-horizon covariance matrix;
- proper scores on separately reserved held-out/rolling data, if later added.

These remain explanatory until a later plan predeclares stable uncertainty and
power gates. Maxima and extreme-tail differences cannot rank viable methods.

## Statistical Decision Design

### Equivalence, Not Non-Rejection

For each mean and log-variance feature, form a chain-aware simultaneous
confidence interval for the arm difference. Pass requires every interval to lie
strictly inside its frozen practical-equivalence margin. A wide interval that
contains zero but extends outside the margin is `INCONCLUSIVE_UNDERPOWERED`, not
a pass.

Use family-wise simultaneous inference across 20 co-primary features. The
initial implementation should support both:

- a conservative Bonferroni/studentized interval reference implementation;
- a chain/block bootstrap max-statistic implementation.

The bootstrap becomes the confirmatory method only after calibration establishes
coverage and power on synthetic null and perturbed alternatives.

### MMD Equivalence

Estimate the joint-path MMD and an upper confidence bound using a hierarchy that
resamples independent chains first, MCMC blocks second, and forecast-replication
clusters third. The confirmatory MMD uses independently generated arm-specific
innovation banks. If a common-random-number MMD diagnostic is retained, it must
use a separately calibrated paired estimator that excludes or corrects matched
cross-arm terms and resamples matched innovation identifiers jointly; it cannot
replace the independent-bank promotion statistic. Pass requires the
independent-bank upper bound to lie below a frozen tolerance. Permutation
equality tests are diagnostic only; failure to reject equality is not
equivalence evidence.

### Margin And Tolerance Calibration

No numerical equivalence margin is promoted directly from judgment in this
document. Freeze margins through Phase 4 calibration using:

1. exact LGSSM null pairs to estimate false-failure behavior;
2. identical SSL-LSTM posterior draws with independent forecast banks to
   estimate simulation noise;
3. split-half ordinary-HMC draws to estimate within-method Monte Carlo noise;
4. controlled parameter/terminal-state perturbations representing scientifically
   material forecast changes;
5. prospective power curves over chain length and forecast replication count.

The selected margins must separate null/simulation noise from at least one
predeclared material perturbation with reviewed power. If no separation exists,
stop with `BLOCK_PREDICTIVE_EQUIVALENCE_NOT_IDENTIFIABLE_AT_AVAILABLE_BUDGET`.

## Validation Data Separation

| Split | May be used for | Forbidden use |
| --- | --- | --- |
| Unit/oracle fixtures | Formula, shape, deterministic recursion, LGSSM analytic checks | Scientific equivalence claims |
| Calibration pilot | Choose forecast count, feature scales, bandwidths, ridge, margins, bootstrap blocks, and run budget | Confirmatory pass/fail |
| Training validation | Select NeuTra training checkpoint by predeclared loss/coverage screens | Predictive-equivalence promotion |
| Confirmation | One-time ordinary-HMC versus frozen-NeuTra-HMC decision on fresh seeds | Any tuning, margin, bandwidth, weight, block-length, or horizon change |
| Audit replication | Fresh sampler and forecast seeds after a confirmation pass | Repairing a failed confirmation without a new plan |

## Phase Program

### Phase 0: Governance And Source/Target Lock

Deliverables:

- reviewed subplan and evidence contract;
- hashes/signatures for target, observations, parameter chart, Phase 2S map,
  forecast semantics, and old artifacts used only as context;
- explicit NeuTra gate audit against BayesFilter and `dsge_hmc` governance;
- math/code anchors for the chapter equations;
- classification `extension_or_invention`.

Pass:

- all identities resolve and the intended target is unambiguous;
- no old CPU-hidden one-chain or failed-reference artifact is promoted into the
  confirmatory baseline.

Veto:

- target/filter migration hidden inside the forecast work;
- missing data or parameter signature;
- plan attempts to claim parameter-posterior correctness.

### Phase 1: Typed Terminal-State And Forecast Contract

Planned code:

- add a narrowly scoped predictive module, provisionally
  `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py`;
- define frozen dataclasses for terminal-state, forecast configuration, forecast
  paths, moments, and provenance;
- expose scalar and batch TensorFlow functions with `jit_compile=True` default;
- reuse `tf_svd_sigma_point_filter(..., return_filtered=True)` rather than
  derivative trace dictionaries;
- add stable hashes for observations, parameter embedding, model/filter
  settings, horizon, and seed bank.

Tests:

- exact shape/dtype/static-horizon tests;
- full/free parameter embedding tests;
- zero-noise deterministic recursion against direct repeated
  `ssl_lstm_transition`/`ssl_lstm_observation` calls;
- fixed-seed replay and changed-seed sensitivity;
- scalar versus batch parity;
- compiled versus eager debug-reference parity;
- process noise affects only stochastic latent coordinates;
- nonfinite/invalid covariance failure tests;
- terminal value-filter log-likelihood parity with the HMC target value at
  fixed parameters.

Pass:

- all forecast-contract tests pass and structured provenance is complete.

### Phase 2: Linear-Gaussian Forecast Oracle

Build a scalar linear-Gaussian specialization with analytic 1-to-10-step
predictive mean and covariance. Compare:

- deterministic conditional moments;
- simulated path moments with uncertainty;
- cross-horizon covariance;
- characteristic features/MMD under identical and perturbed laws.

Pass:

- analytic and implementation moments agree within predeclared numerical and
  Monte Carlo tolerances;
- identical-law MMD behaves as calibrated;
- material perturbations are detected at the planned rate.

This phase is a continuation veto: no nonlinear predictive comparison proceeds
if the forecast/statistics harness fails its exact oracle.

### Phase 3: Predictive Statistics And Uncertainty Engine

Planned code, provisionally under `bayesfilter/inference/predictive_equivalence.py`:

- horizon means, variances, log variances, central moments, quantiles, and
  cross-horizon covariance;
- fixed-bandwidth unbiased MMD;
- characteristic-feature vectors;
- chain-aware batch means and block bootstrap;
- simultaneous equivalence intervals;
- regularized inverse-covariance diagnostic;
- explicit statuses `PASS_EQUIVALENT`, `FAIL_MATERIALLY_DIFFERENT`,
  `INCONCLUSIVE_UNDERPOWERED`, and `INVALID_HARD_VETO`.

Tests:

- exact identical-array zero differences;
- known Gaussian moment and characteristic-function fixtures;
- symmetry and permutation invariance of MMD;
- signed finite-sample behavior of the diagonal-excluded MMD U-statistic, plus
  non-negativity and exact identical-array zero for a separately reported
  biased V-statistic diagnostic;
- detection of mean, variance, skew, and cross-horizon-dependence alternatives;
- autocorrelated-chain coverage simulations;
- bootstrap reproducibility and cluster/block semantics;
- covariance singularity and ridge fail-closed behavior;
- family-wise interval coverage and power simulations;
- explicit proof that failure to reject equality cannot emit `PASS_EQUIVALENT`.

### Phase 4: Calibration And Design Freeze

Run a reviewed calibration program, not a confirmation:

- LGSSM exact null and alternatives;
- SSL-LSTM identical-draw/different-forecast-bank null;
- ordinary-HMC split-half null using calibration-only draws;
- controlled material perturbations;
- ladders for posterior draw count, forecast replications, block length,
  bandwidth, ridge, and margins.

Artifact must freeze:

- horizons and horizon weights;
- mean/log-variance scaling;
- all equivalence margins;
- MMD bandwidths, weights, and tolerance;
- covariance ridge and condition cap;
- bootstrap type, block length, replication count, confidence level, and seed;
- minimum chains/draws/ESS/forecast replications;
- confirmatory sampler and forecast seeds.

Pass only if prospective null coverage and material-alternative power meet
predeclared calibration criteria. Descriptive calibration results cannot be
used to claim one sampler is better.

### Phase 5: Ordinary-HMC Confirmatory Baseline

Run fresh ordinary MAP-local HMC under trusted GPU/XLA execution:

- four independent chains;
- BayesFilter-owned production target/score path;
- separate reviewed tuning and retained sampling;
- native divergence telemetry required;
- chain-shaped samples retained for uncertainty analysis;
- sampler seeds frozen in Phase 4.

Hard pass gates:

- zero native divergences;
- no nonfinite sample, log target, score, or log accept;
- acceptance inside the reviewed envelope;
- per-parameter R-hat, bulk/tail ESS, and MCSE gates pass;
- movement/zero-jump gates pass;
- target and Phase 2S coordinate signatures match.

The old Phase 2V chain is not the baseline because it is one-chain,
CPU-hidden, non-XLA, and lacks native divergence telemetry.

### Phase 6: Plain Dense-IAF NeuTra Training

Implementation prerequisite:

- BayesFilter currently loads frozen dense-IAF artifacts but does not own a
  generic dense-IAF training surface. Implement the minimal TensorFlow trainer
  or import/adapt the reviewed `dsge_hmc` architecture under a separate
  code-review subphase. Do not import NumPy algorithmic training paths.

Training contract:

- GPU workload with trusted/managed-session GPU provenance;
- TensorFlow/TFP, the target-consistent `float64` dtype unless a separate
  reviewed parity phase admits `float32`, and XLA JIT on by default;
- target-specific four-dimensional MAP-local coordinate;
- canonical reverse-KL plain NeuTra baseline;
- fixed training/validation base-noise splits;
- dense autoregressive IAF plus mixing layers, topology frozen by reviewed
  subplan;
- structured checkpoint/training-state artifacts with target signature;
- no raw flow samples treated as posterior evidence.

Training nomination screens:

- finite loss/gradients/parameters/log Jacobian;
- forward/inverse roundtrip and Jacobian checks;
- held-out reverse-KL loss and fixed shell/anchor coverage diagnostics;
- no target-signature or checkpoint mismatch.

These screens nominate a frozen transport only. They do not establish posterior
quality or predictive equivalence.

### Phase 7: Exact-Corrected NeuTra-HMC Confirmation Arm

- load the frozen artifact through the BayesFilter loader;
- bind the fixed transport to the exact transformed target including log
  Jacobian;
- retune step size/leapfrog count in `z` space; do not reuse ordinary-HMC
  tuning;
- run four independent GPU/XLA chains with native divergence telemetry;
- transform retained `z` draws back to the four free parameters;
- require the same sampler hard gates as Phase 5;
- preserve transport, target, tuning, and chain hashes.

Failure of this arm rejects the current NeuTra candidate or triggers a reviewed
training/tuning repair. It does not reject the scalar target or the broader
predictive-equivalence research question unless a true continuation veto fires.

### Phase 8: Blinded Predictive Confirmation

Before opening arm labels:

- verify Phase 4 design hash;
- verify Phase 5/7 sampler hard gates;
- generate both path sets with the frozen common and independent forecast banks;
- store chain/draw/forecast cluster identities;
- compute statistics from anonymized arm labels where practical.

Decision order:

1. engineering/forecast hard vetoes;
2. sampler hard vetoes;
3. simultaneous mean/log-variance equivalence;
4. independent-bank joint-path MMD equivalence;
5. paired-feature robustness under the independent arm-specific forecast banks;
6. explanatory higher moments/quantiles/covariance;
7. parameter summaries for interpretation only.

Possible decisions:

- `PASS_PREDICTIVE_FUNCTIONAL_EQUIVALENCE`;
- `FAIL_PREDICTIVE_FUNCTIONAL_EQUIVALENCE_MATERIAL_DIFFERENCE`;
- `INCONCLUSIVE_PREDICTIVE_EQUIVALENCE_UNDERPOWERED`;
- `INVALID_PREDICTIVE_COMPARISON_HARD_VETO`.

### Phase 9: Independent Audit Replication

After a Phase 8 pass only, repeat with fresh sampler and forecast seeds without
changing the frozen design. Passing both runs supports a limited replicated
predictive-functional equivalence claim for this target. A disagreement between
confirmation and audit blocks promotion and triggers a replication/uncertainty
diagnostic, not post-hoc averaging.

### Phase 10: Closeout And Optional Model-Adequacy Branch

Close with:

- decision table;
- inference-status table;
- run manifests for every serious run;
- three ledgers: engineering, sampler, predictive interpretation;
- candidate rejection versus research-direction status;
- post-run red-team note;
- reset memo and visible handoff.

Held-out predictive accuracy, CRPS, log score, interval coverage, rolling-origin
calibration, parameter identification, and broader dimensions are separate
future branches. They cannot be smuggled into the equivalence closeout.

## Planned File Surface

Exact names may be narrowed by phase review, but the initial ownership map is:

| Purpose | Planned path |
| --- | --- |
| SSL-LSTM typed forecast operator | `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py` |
| Predictive equivalence statistics | `bayesfilter/inference/predictive_equivalence.py` |
| Dense-IAF training surface | `bayesfilter/inference/neutra_training_tf.py` or a narrower reviewed module |
| Forecast contract tests | `tests/test_ssl_lstm_predictive_tf.py` |
| Statistical engine tests | `tests/test_predictive_equivalence.py` |
| NeuTra training tests | `tests/test_scalar_ssl_lstm_neutra_training_tf.py` |
| Exact transformed-HMC tests | extend focused fixed-transport tests or add `tests/test_scalar_ssl_lstm_neutra_hmc.py` |
| Calibration/confirmation runners | dated scripts under `docs/benchmarks/` |
| Durable artifacts | `docs/plans/artifacts/scalar-ssl-lstm-predictive-equivalence-2026-07-11/` |

Do not add new public package exports until the internal APIs and evidence gates
pass. An optional internal feature is not a new default.

## Skeptical Plan Audit

| Risk | Audit result |
| --- | --- |
| Wrong baseline | Repaired: the baseline is a fresh four-chain GPU/XLA ordinary-HMC run, not Phase 2V or a local Gaussian reference. |
| Proxy promoted | Repaired: training loss, validation loss, raw flow samples, acceptance, short chains, and parameter similarity cannot pass predictive equivalence. |
| Missing stop conditions | Explicit phase continuation vetoes and hard sampler/forecast vetoes are defined. |
| Unfair comparison | Each sampler is independently tuned; target, data, forecast operator, horizon, innovations, and confirmation budget are matched. |
| Hidden assumptions | Terminal state is explicitly the SVD-UKF Gaussian approximation; predictive equivalence is not parameter equality or model adequacy. |
| Stale context | The plan binds the 2026-07-10 reset memo and requires fresh target/artifact hashes in Phase 0. |
| Environment mismatch | Serious NeuTra training and confirmation sampling are GPU/XLA; CPU-only work is limited to tests/reference/calibration exceptions and labeled accordingly. |
| Artifact mismatch | Every phase requires structured JSON/Markdown, signatures, commands, seeds, environment, and plan/result links. |
| Weak statistical claim | Equivalence requires confidence bounds inside margins, not failure to reject equality; design calibration and independent audit are separate. |
| Two wrong methods agree | Shared target/forecast implementation is protected by equation-to-code tests, an LGSSM oracle, deterministic recursion checks, and separate sampler hard gates. |
| MGF instability | MGF is explanatory motivation only; bounded characteristic features/MMD are the initial omnibus statistic. |

Audit status:
`PASSED_FOR_DOCUMENTATION_AND_REVIEWED_PHASE_0_PLANNING_ONLY`.

## Pre-Mortem

### How the program could pass while misleading us

- Both samplers share the same broken target or forecast operator.
- Equivalence margins are too wide or MMD bandwidths too coarse.
- Common random numbers hide instability that appears under independent banks.
- Reverse-KL NeuTra misses a region that ordinary HMC also fails to visit.
- A small four-parameter fixture is generalized to larger neural models.

Cheap discriminators:

- exact LGSSM forecast oracle;
- deterministic/noise-zero direct recursion;
- controlled material alternatives and prospective power curves;
- independent forecast bank;
- dispersed initialization and fresh sampler seeds;
- explicit restriction of the final claim to this target.

### How it could fail for engineering or tuning reasons

- terminal filtered-state extraction uses the wrong filter/backend;
- NeuTra artifact cannot be inverted or bound to the target;
- ordinary and transformed HMC use unfair tuning;
- predictive simulation is too expensive at the planned nested sample count;
- bootstrap covariance becomes singular because paths are over-clustered.

Cheap discriminators:

- affine transform and Gaussian target controls;
- tiny shape/roundtrip/XLA canaries;
- runtime budget pilot before confirmation;
- covariance/ridge calibration with fail-closed condition diagnostics;
- separate candidate failure from research-direction failure.

## Required Serious-Run Manifest

Every Phase 4-9 evidence-bearing run must record:

- git commit and dirty status;
- exact command and environment/conda env;
- CPU/GPU status and trust basis;
- TensorFlow/TFP versions, dtype, XLA/JIT, TF32, and device placement;
- data and target signatures;
- sampler, transport, forecast, and design hashes;
- all training/sampler/forecast/bootstrap seeds;
- chain, draw, forecast-replication, and block counts;
- wall time and output paths;
- plan and result paths;
- explicit nonclaims.

## Required Result Tables

Serious results must include:

1. Decision table: decision, primary criterion, veto status, uncertainty, next
   action, and nonclaims.
2. Inference-status table: hard veto screen, statistically supported ranking,
   descriptive-only differences, default readiness, and next evidence.
3. Predictive-equivalence table by horizon: standardized mean difference and
   simultaneous interval, log-variance difference and interval, margins, and
   status.
4. Omnibus table: bandwidths/features, MMD/CF statistic, uncertainty bound,
   tolerance, common/independent bank status.
5. Three-ledger table: engineering correctness, sampler validity, predictive
   interpretation.
6. Post-run red-team note: strongest alternative explanation, overturning
   evidence, and weakest evidence component.

## Next Action

Do not implement or run experiments from this master program yet. Draft a
focused Phase 0 governance-and-target-lock subplan, review it, and only then
begin source/target binding. Phase 3 GPU/XLA from the old parameter-reference
program remains blocked; this program creates a separate future GPU/XLA lane
whose claim is predictive-functional equivalence.
