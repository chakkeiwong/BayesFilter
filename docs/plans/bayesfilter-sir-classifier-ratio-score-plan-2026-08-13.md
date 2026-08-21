# Filter-Independent SIR Classifier-Ratio Score Plan

Date: 2026-08-13  
Status: `EXECUTED_EXACT_ORACLE_FAILED_SIR_CONTINUATION_VETO`

This plan supersedes the off-target Fisher-importance plan. The method below is
frozen. Replacing it with a latent-variable expectation, particle method,
finite-program gradient, analytical complete-data score, synthetic likelihood,
or summary-statistic likelihood is a continuation veto, not an allowed repair.

## Frozen Research Question And Identity

For the fixed Austria-SIR observed path and each parameter coordinate `j`, can
simulator-only balanced classification estimate

`s_j(theta,y_obs) = d/d theta_j log p_theta(y_obs)`

without a filter or analytical score implementation?

For every `epsilon` independently generate observation paths

`Y+ ~ p_{theta + epsilon e_j}` and `Y- ~ p_{theta - epsilon e_j}`.

With balanced class probability, a calibrated classifier satisfies

`logit D_{j,epsilon}(y) = log p_{theta+epsilon e_j}(y) - log p_{theta-epsilon e_j}(y)`.

The only eligible raw score estimate is

`score_hat_{j,epsilon}(y_obs) = calibrated_logit_{j,epsilon}(y_obs)/(2 epsilon)`.

The `epsilon -> 0` estimate is the intercept of a weighted regression of the
raw score estimates on `epsilon^2`. This is central-difference extrapolation,
not a new target.

## Research Intent Ledger

| Field | Frozen definition |
|---|---|
| Main question | Can observation-only likelihood-ratio classification provide a usable independent score reference at SIR `T=20/40/50`? |
| Candidate mechanism | Balanced probabilistic classification of full observation paths simulated at `theta +/- epsilon e_j`. |
| Exact calibration baseline | A simulator-only multivariate Gaussian location/log-scale family with a closed-form observed-data score; no filter. |
| Primary criterion | On the exact family, extrapolated score error passes the frozen tolerance; on SIR, each reported coordinate has at least three admitted epsilon estimates with calibrated held-out behavior and stable extrapolation. |
| Promotion veto | Exact-oracle failure; data leakage; unbalanced classes; failed probability calibration; near-chance or near-perfect separation under the frozen rules; observed path outside the held-out logit support diagnostic; unstable epsilon extrapolation. |
| Continuation veto | Any import/call to filter, particle, ancestry, resampling, smoother, Fisher-importance, local analytical-score, or likelihood-evaluation code; simulator mismatch; corrupted split identity; GPU/XLA or batch-native violation; budget exhaustion. |
| Repair trigger | Classifier under/over-capacity, optimizer instability, or one epsilon failing a diagnostic triggers only the predeclared architecture/epsilon ladder. It cannot authorize a different estimator. |
| Explanatory only | AUC, accuracy, raw training loss, runtime, and raw score ordering before calibration/extrapolation. |
| Must not be concluded | Passing does not make the estimate exact, prove a particle algorithm correct, rank algorithms statistically, establish HMC/default readiness, or establish source-faithful Zhao-Cui inference. |

## Anti-Drift Contract

The implementation and tests must establish all of the following:

1. Classifier inputs are only tensors with shape `[paths,T,9]` containing simulated observations, plus task metadata `j` and `epsilon` outside the observation tensor.
2. Labels are exactly balanced and mean `1` for `theta+epsilon e_j`, `0` for `theta-epsilon e_j`.
3. The fixed evaluated path is `base_model.simulate(final_time=50, seed=81120)[1][1:T+1]`.
4. The score field is computed only as calibrated logit divided by `2*epsilon`; a source test rejects another expression.
5. The classifier module and runner must not import modules or contain tokens matching `filter`, `particle`, `resampl`, `smooth`, `ancestr`, `complete_data_score`, `fisher_identity_simulation_score`, `transition_log_density_parameter_score`, or `observation_log_density_parameter_score`, except inside the dependency-test deny list and documentation strings.
6. A runtime import audit records loaded `bayesfilter` modules and vetoes forbidden execution dependencies.
7. The prior Fisher artifact is never read by training, selection, calibration, extrapolation, or interpretation.

## Data, Splits, And Seeds

- Parameter center: `theta=[0,0,0]` in the declared log-scale coordinates.
- Coordinates: `log_kappa_scale`, `log_nu_scale`, `log_obs_noise_scale`.
- Perturbations: `epsilon in {0.01,0.02,0.04,0.08}`. `0.08` is a separation diagnostic and may be rejected by the frozen gate; it is not automatically used for extrapolation.
- Horizons: paired prefixes `T=20,40,50` of the same observation path.
- Full paths: no hand-selected or learned low-dimensional summary replaces `[T,9]`.
- Per class, task, and independent replicate: training `2048`, architecture-selection validation `512`, Platt calibration `512`, untouched test `1024`.
- Independent classifier/simulation replicates: `3`.
- Generate `T=50` paths once per `(role,j,epsilon,sign,replicate)` and slice prefixes for paired horizon analysis.
- Root seed: `89300`; role, coordinate, epsilon, sign, replicate, and batch domains are disjoint and recorded.

The observed path is never used for training, architecture selection, Platt
calibration, early stopping, or epsilon admission. It is evaluated only after a
classifier and calibrator are frozen.

## Classifier And Training Ladder

All routes are TensorFlow, batch-native, GPU, and XLA-compiled. Training uses
the complete balanced training split as one batch (`4096` rows); this removes
mini-batch noise and reduces compilation/optimizer overhead without changing
the objective. No batch-size-one update, NumPy numerical path, row mapping, or
scalar simulator loop is allowed.

Inputs are standardized coordinatewise using the balanced training mixture.
Two architectures are predeclared:

1. `linear_full_path`: flatten the standardized complete path and emit one logit.
2. `mlp_full_path_quadratic`: concatenate standardized path entries with their centered squares, flatten, then dense widths `(128,64)` with `tanh`, followed by one logit.

The linear model is the naive density-ratio baseline. The MLP is the enhanced
candidate. Both retain every observation entry; the pointwise square adds a
basis but does not summarize or discard the path.

Training uses balanced binary cross entropy, Adam learning rate `3e-4`, at most
`160` full-batch epochs, minimum `20`, and validation early stopping patience `12`.
Regularization candidates are `0` and `1e-5`; the smaller validation binary
cross entropy selects, with the linear architecture preferred within one
validation standard error. Selection is performed separately for each horizon
using the aggregate of its three coordinates at `epsilon=0.04`, replicate zero.
The selected horizon-specific architecture/control is frozen before final
replicates or untouched tests are consumed.

Platt calibration fits `a*raw_logit+b` on the calibration split only, using
full-batch binary cross entropy. It is accepted only if finite and it does not
worsen calibration-split log loss by more than `1e-4`.

## Diagnostic And Admission Rules

For every frozen classifier head:

- held-out test class balance must be exact;
- all outputs must be finite;
- test binary cross entropy must improve on `log(2)` by more than twice its
  per-example Monte Carlo standard error (`insufficient_signal` otherwise);
- test AUC must lie in `[0.505,0.995]` (`chance` or `separation_too_large` otherwise);
- ten-bin equal-width expected calibration error must be at most `0.03`;
- Platt slope must lie in `[0.5,2.0]`;
- the observed-path calibrated logit must lie inside the held-out calibrated
  logit range expanded by `10%` (`observed_path_extrapolation` otherwise).

At least three epsilon values must pass for a coordinate/horizon. Weighted
least squares fits `score_hat(epsilon)=s+c*epsilon^2`, using across-replicate
variance with a `1e-6` floor. Admission additionally requires:

- finite intercept and standard error;
- leave-one-epsilon-out intercept range at most `max(1.0,2*SE)`;
- the smallest two admitted-epsilon means differ by at most `max(1.0,3*combined_SE)`.

Failure reports `no_classifier_ratio_reference`; it does not select a filter
score by proximity.

## Exact-Oracle Calibration Gate

Before SIR final execution, run the entire simulator/classifier/calibration/
extrapolation procedure on a multivariate Gaussian observation simulator in
which two coordinates move the mean in fixed orthogonal full-path directions
and the third moves log scale. The exact observed-data score is computed from
the Gaussian density, not a filter.

For every horizon and coordinate:

- at least three epsilons pass classifier admission;
- absolute intercept error is at most `max(0.5,3*estimated_SE)`;
- the exact score lies within `intercept +/- max(0.5,3*estimated_SE)`.

Failure is a continuation veto. Training controls may be repaired only through
the frozen architecture/regularization/epoch ladder and within the budget.

## Default And Assumption Audit

| Choice | Provenance/status | Why reasonable | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Balanced classes | Density-ratio identity; reviewed default | Removes prior-odds offset | Wrong log-ratio intercept | Exact balance assertion |
| Four epsilons | Central-difference hypothesis | Exposes signal/bias tradeoff | All chance or large-epsilon bias | Per-epsilon AUC and extrapolation |
| Full path | Required target | Preserves observed-data likelihood target | High classifier sample complexity | Exact oracle and capacity ladder |
| MLP widths | Convenience hypothesis | Modest capacity for 180-450 inputs | Under/overfit | Linear baseline, held-out loss/ECE |
| Adam `3e-4` | Warm-start hypothesis | Stable common TensorFlow setting | Nonconvergence | Loss/gradient finite checks |
| Platt scaling | Calibration hypothesis | Corrects scalar probability miscalibration | Distorts local ratio | Independent calibration/test splits |
| Three replicates | Bounded descriptive uncertainty | Captures simulation/init variability | Wide uncertainty | Across-replicate SE and nonclaim |
| Float64 simulator, float32 classifier | Target-faithful simulation plus GPU training convenience | Avoids changing simulator law while keeping training feasible | Cast loses weak signal | Exact oracle and perturbation ladder |

No inherited classifier setting is promoted outside this campaign.

## Skeptical Pre-Mortem And Plan Audit

- **Could the command succeed but answer the wrong question?** Yes, if summaries,
  latent states, or analytical scores enter. The source/runtime deny gates and
  fixed score expression prevent this.
- **Could a good classifier still give a wrong pointwise ratio?** Yes. Held-out
  calibration, observed-path support, independent replicates, multiple epsilon
  values, and the exact oracle address but do not eliminate this risk.
- **Could accuracy become the target?** No. Accuracy/AUC only nominate or veto;
  the target is calibrated logit at the fixed observed path.
- **Could perturbations be too small or too large?** Yes. The four-epsilon ladder
  and frozen chance/separation gates distinguish these cases.
- **Could architecture selection leak the observed path?** No. Selection uses
  simulated validation data only; `y_obs` is evaluated after freezing.
- **Could the exact oracle be too easy?** Yes. It includes both mean and log-scale
  coordinates, requiring linear and quadratic log-ratio recovery at all three
  horizons. Passing still does not prove SIR correctness.
- **Could shared simulation prefixes create false replication?** Horizons are
  paired by design, but inference is within horizon; independent replicate roots
  remain disjoint.
- **Environment mismatch?** Direct interpreter is
  `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; GPU probes and execution
  require trusted/escalated access, memory growth before initialization, and XLA.

Audit verdict: `READY_FOR_INDEPENDENT_REVIEW`. No experiment may start until a
bounded read-only review confirms that the plan executes the classifier-ratio
identity and contains no filter-dependent substitute.

## Compute And Attempt Budget

- Focused mechanics and exact-oracle smokes: at most `20` GPU minutes.
- Architecture selection and exact-oracle calibration: at most `40` GPU minutes.
- SIR final campaign: at most `90` GPU minutes.
- At most two infrastructure retries per stage and one scientific repair through
  the frozen ladder. Every launch uses a fresh versioned directory.
- Artifact root:
  `docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/`.

The result memo must state hard vetoes, viable coordinates, whether any score
reference is admitted, uncertainty, differences that are descriptive only,
and what additional evidence would be needed for a defensible algorithm
comparison.
