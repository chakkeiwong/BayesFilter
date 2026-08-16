# Filter-Independent SIR Joint-k Classifier-Ratio Score V3 Plan

Date: 2026-08-14  
Status: `EXECUTED_EXACT_ORACLE_FAILED_SIR_CONTINUATION_VETO`

## Frozen Research Question And Estimator

Can one conditional classifier trained jointly across several symmetric
parameter perturbations estimate the fixed-path observed-data score more
stably than independent per-epsilon classifiers, while remaining completely
independent of every state-estimation algorithm?

For coordinate `j`, let `epsilon0=0.01`, `k in {1,2,3,4}`, and
`delta_k=k*epsilon0`. Conditional on `delta_k`, generate balanced classes

`Y+ ~ p_{theta+delta_k e_j}` and `Y- ~ p_{theta-delta_k e_j}`.

The distribution over `k` is identical in both classes. Therefore the
Bayes-optimal conditional classifier obeys

`logit D_j(y,delta) = log p_{theta+delta e_j}(y) - log p_{theta-delta e_j}(y)`.

The logit is constrained to the odd local form

`z(y,delta) = c1(y) r + c3(y) r^3 + c5(y) r^5`,

where `r=delta/delta_scale` and `delta_scale=0.04`. Consequently

`score_hat_j(y_obs) = c1(y_obs)/(2*delta_scale)`.

This is the only eligible V3 score expression. It estimates the derivative
from the jointly learned raw-logit slope before any division by a small
perturbation. It must not be replaced by independent-epsilon scores, a latent
expectation, a filter, a complete-data score, a simulator derivative, an
analytical likelihood, or a summary-statistic likelihood.

## Research Intent Ledger

| Field | Frozen V3 definition |
|---|---|
| Main question | Does sharing a conditional ratio model across `k=1..4` yield an independently calibrated fixed-path score at SIR `T=20/40/50`? |
| Candidate mechanism | Balanced observation-only conditional classification with an odd degree-five perturbation basis |
| V2 failure addressed | weak small-epsilon signal, saturated large-epsilon heads, noisy division by `2*epsilon`, and separately fitted representations |
| Exact baseline | full-path Gaussian location/log-scale simulator with closed-form marginal score used only after fitting |
| Primary criterion | all nine Gaussian horizon/coordinate cells pass exact-score accuracy and precision gates before SIR |
| Promotion veto | pooled or per-k probability diagnostics fail, fixed path is outside held-out logit support, replicate score is unstable, or exact-score gate fails |
| Continuation veto | any exact cell fails; forbidden dependency loads; conditional balance or delta identity fails; GPU/XLA/memory policy fails; artifact invalidity; budget exhaustion |
| Repair trigger | no scientific repair after the full exact claim run; a failure returns to planning |
| Explanatory only | architecture choice, raw AUC, individual odd coefficients, runtime, contribution of cubic/quintic terms |
| Must not be concluded | passing does not make the SIR estimate exact or establish correctness/ranking of a filter, HMC readiness, or a repository default |

## Data, Splits, Pairing, And Seeds

- center: `theta=[0,0,0]`;
- horizons: paired prefixes `T=20,40,50` generated from `T=50` paths;
- coordinates: log-kappa scale, log-nu scale, log-observation-noise scale;
- perturbations: `delta={0.01,0.02,0.03,0.04}`;
- each conditional class is exactly balanced at every delta;
- plus and minus simulations are independent, preserving the V1/V2 sampling
  contract and ordinary per-example diagnostics;
- per `delta`, class, task, and replicate: train `2048`, architecture-selection
  validation `512`, temperature calibration `512`, untouched test `1024`;
- three independent final classifier/simulation replicates;
- selection domain `30`, final domain `40`; V1/V2 domains are not reused;
- the fixed evaluated observation path is excluded from training, selection,
  calibration, early stopping, and all gates that tune a model.

The joint training set therefore has `4*2*2048=16384` full paths. This matches
the aggregate V2 training simulation count across four perturbations while
fitting one shared classifier rather than four separate classifiers.

## Classifier And Optimization Ladder

All candidates consume the complete standardized `[T,9]` path and the declared
perturbation only through `(r,r^3,r^5)`. They emit three coefficient functions
`(c1,c3,c5)` and take their odd-basis inner product. There is no free even term
or delta-independent intercept, so `z(y,0)=0` by construction.

Candidates:

1. `joint_linear_quadratic_odd5`: features `[z,z^2-1]`, zero-initialized dense
   coefficient head of width three.
2. `joint_mlp_quadratic_odd5`: the same full features, tanh widths `(128,64)`,
   then a zero-initialized three-coefficient head.

Regularization candidates are `0` and `1e-5`. Adam learning rate is `1e-3`,
mini-batch size `2048`, maximum `80` epochs, minimum `15`, and validation
patience `10`. Each update is batch-native and XLA-compiled. The smaller model
is preferred within one validation standard error. Selection is separate for
every `(stage,horizon,coordinate)` using one selection-domain dataset pooled
over all four deltas. Exact-oracle controls cannot be copied into SIR: SIR must
select its own controls from SIR simulations before final-domain generation.

V2 full-batch linear heads often hit their 160-step ceiling. V3 has up to 640
mini-batch updates, so optimization failure can be distinguished from the
ratio-estimation hypothesis without changing the total simulated training
paths.

## Calibration And Diagnostics

Calibration fits one positive global temperature multiplying the entire odd
logit, using the pooled calibration split. No calibration intercept is allowed,
because an intercept would violate `z(y,0)=0`. The calibrated coefficients are
the raw coefficients multiplied by that temperature.

Frozen gates for every final replicate:

1. exact conditional class balance at every delta and finite tensors/logits;
2. pooled untouched-test log loss improves on `log(2)` by more than `2*SE`;
3. calibration loss is not worse than raw calibration loss by more than
   `1e-4`;
4. fitted temperature is finite and strictly positive; it is recorded but has
   no arbitrary interval veto;
5. per-delta ECE is at most `0.04`;
6. per-delta AUC is at least `0.48`, at least two deltas have AUC above `0.52`,
   and AUC may not decrease by more than `0.03` as delta increases;
7. maximum-delta AUC is at most `0.995`, excluding effectively separated
   extrapolation;
8. at every delta, the fixed-path calibrated logit lies in the held-out logit
   range expanded by `10%` of `max(range,1)`;
9. the final training best epoch is below the maximum, or the final validation
   improvement over the last ten epochs is below `1e-4`; this prevents silent
   optimizer truncation.

The V2 Platt-slope interval is removed because it rejected calibrated heads
based on the magnitude of the correction rather than on calibrated held-out
behavior. This is a predeclared V3 methodological change, not a post-result
relaxation. Exact-score error and precision are the authoritative oracle gates.

## Score Aggregation And Admission

Each final replicate produces

`s_r = calibrated_c1_r(y_obs)/(2*delta_scale)`.

For each horizon/coordinate report the three replicate estimates, mean,
sample standard error, range, and cubic/quintic logit contributions at every
delta. Admission requires:

- all three replicates pass all classifier/calibration/support gates;
- finite mean and standard error;
- replicate range at most `max(2.0,4*SE)`;
- score SE at most `max(1.0,0.25*abs(mean_score))`.

For the exact Gaussian oracle, additionally require

`abs(mean_score-exact_score) <= max(0.5,3*SE)`

and the exact score to lie in the same interval. All nine exact cells must pass.

For SIR, passing yields an admitted approximate classifier-ratio reference with
the reported replicate uncertainty. It does not yield an oracle or authorize
algorithm ranking without a separate uncertainty-aware comparison plan.

## Mathematical And Source Audit

For equal conditional class priors and the same delta distribution in each
class,

`P(+|y,delta)/P(-|y,delta)=p_{theta+delta e_j}(y)/p_{theta-delta e_j}(y)`.

If `ell(delta)=log p_{theta+delta e_j}(y)`, then

`ell(delta)-ell(-delta)=2 ell'(0) delta + ell'''(0) delta^3/3 + ell^(5)(0) delta^5/60 + O(delta^7)`.

With `r=delta/delta_scale`, the derivative of the fitted logit at zero is
`c1/delta_scale`, hence the score is `c1/(2*delta_scale)`. The classifier is not
given `ell`, its derivatives, an exact ratio, latent states, or a filter output.

The generic identity is a project derivation for this plan. The exact Gaussian
score is a calibration authority only and must never enter training or
selection. A later literature survey may improve the method, but no unchecked
literature claim is required for this bounded test.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| `k=1..4`, `epsilon0=.01` | V3 hypothesis informed by V2 | spans weak through informative ratios without V2's `.08` saturation | all deltas weak or `.04` still saturated | per-k AUC and exact oracle |
| odd degree five | Taylor-derived model class | captures central-ratio curvature while identifying the slope | insufficient higher-order fit or coefficient collinearity | exact log-scale cells and contribution table |
| shared representation | user-proposed hypothesis | pools statistical strength across perturbations | negative transfer across k | per-k held-out diagnostics |
| temperature-only calibration | structural hypothesis | preserves oddness and zero logit at delta zero | k-dependent miscalibration | per-k ECE |
| quadratic full-path features | V2 repaired baseline | exact Gaussian scale ratio lies in this span | high-dimensional variance | exact scale cells and replicate SE |
| MLP widths `(128,64)` | inherited hypothesis, not default | provides nonlinear SIR capacity | overfit or unstable pointwise coefficient | untouched exact/SIR tests and support |
| Adam `1e-3`, 80 epochs | optimization hypothesis | 640 updates repairs V2 truncation | still truncated or unstable | explicit optimizer-completion gate |
| three replicates | bounded campaign choice | exposes simulation/initialization variation | weak uncertainty estimate | precision and range gates; no superiority claim |

## Skeptical Pre-Mortem

- **Wrong target despite a successful command:** forbidden source/runtime
  dependency audits, observation-only tensor contracts, and the sole score
  expression prevent estimator substitution.
- **Pooling hides a bad perturbation:** every delta has separate untouched ECE,
  AUC, monotonicity, support, and saturation diagnostics.
- **Odd polynomial manufactures a derivative:** it restricts only delta
  dependence; every coefficient remains learned from labeled observation paths.
- **The fixed observation influences training:** all fitting and selection occur
  before the fixed path is evaluated.
- **Large uncertainty makes the exact tolerance vacuous:** an independent
  precision gate caps SE even if `3*SE` would cover the exact value.
- **Optimization masquerades as method failure:** update count is increased and
  truncation is a hard gate.
- **Gaussian success is too easy:** location and log-scale coordinates at three
  dimensions test linear, quadratic, curvature, calibration, and pointwise
  uncertainty; success remains necessary but not sufficient for SIR.
- **SIR failure is interpreted as filter evidence:** no filter is run, and a
  failed classifier yields `no_joint_k_ratio_reference`, not proximity-based
  selection of an algorithm score.

## Evidence Contract, Commands, And Budget

Question: can the joint-k conditional ratio estimator clear an exact marginal
score oracle and then provide approximate SIR references? Comparator: exact
Gaussian marginal score. Primary pass criterion: all nine exact cells.
Classifier diagnostics are vetoes; architecture/runtime are explanatory. No
claim about filter correctness, ranking, HMC, or default readiness is allowed.

Execution order:

1. focused CPU-only mechanics tests with `CUDA_VISIBLE_DEVICES=-1`;
2. trusted GPU/XLA smoke for shape, oddness, memory growth, and compilation;
3. one fresh full exact-oracle run;
4. only after exact `PASSED`, one fresh SIR run.

Budgets: 20 GPU minutes for smoke, 45 GPU minutes for the exact oracle, and 90
GPU minutes for SIR. At most two infrastructure retries per stage. No scientific
repair after the full exact run. Every attempt uses a fresh output directory.

Artifact root:
`docs/benchmarks/artifacts/sir_joint_k_classifier_ratio_score_20260814/`.

Audit verdict before implementation: `READY_FOR_THOROUGH_REVIEW`.
