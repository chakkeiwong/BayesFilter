# Filter-Independent SIR Anchored-Orthogonal Ratio Score V4 Plan

Date: 2026-08-14  
Status: `PRE_EXECUTION_REVIEW`

## Target

Estimate the fixed-path observed-data score using only balanced simulated
observation paths, with no filter, latent-state score, likelihood evaluator, or
Fisher identity.

For each coordinate `j`, condition on positive perturbation magnitude

`delta in {0.005,0.010,0.015,0.020,0.030,0.040}`

and classify paths from `theta-delta*e_j` versus `theta+delta*e_j`. The class
prior is balanced at every delta.

## Anchored Basis

Set `r=delta/0.04`, with the six observed values. Define

`phi0(r)=r`,

`phi1(r)=r^3-alpha*r^5`,

where

`alpha=sum(r_i^4)/sum(r_i^6)`

over the actual six-point design. Thus the discrete design inner product obeys
`sum phi0(r_i)*phi1(r_i)=0`; also `phi0'(0)=1` and `phi1'(0)=0`.

Fit one conditional odd logit

`z(y,delta)=c0(y)*phi0(r)+c1(y)*phi1(r)`.

The only score conversion is

`score_hat(y_obs)=calibrated_c0(y_obs)/(2*0.04)`.

The second basis coefficient models curvature but cannot change the derivative
at zero. No third basis mode is used: six points do not justify a higher-order
derivative expansion under the available finite-sample noise.

## Research Ledger And Evidence Contract

| Field | Frozen definition |
|---|---|
| Question | Does design-orthogonal anchoring stabilize the pointwise derivative coefficient? |
| Comparator | Exact Gaussian full-path marginal score, evaluated only after fitting |
| Primary gate | All nine Gaussian horizon/coordinate cells pass score-error and precision gates |
| Head vetoes | finite values, pooled signal, calibration loss, per-delta ECE, monotone/non-inverted AUC, support, optimizer completion |
| Continuation veto | Any exact cell fails; dependency contamination; balance failure; artifact invalidity; GPU/XLA/memory-policy failure |
| Nonclaims | no SIR exactness, filter correctness, ranking, HMC readiness, or default promotion |

The exact Gaussian score is never supplied as a training label, feature, or
selection criterion. It is used only for terminal oracle comparison.

## Data And Training

- `theta=[0,0,0]`;
- horizons `T=20,40,50`, generated as paired prefixes of `T=50` paths;
- per delta and class: train `2048`, validation `512`, calibration `512`, test
  `1024`;
- three independent final replicates;
- selection seed domain `50`, final seed domain `60`;
- full training set: six deltas x two classes x 2048 = 24576 paths;
- candidates: anchored linear-quadratic coefficient head and anchored MLP
  coefficient head `(128,64)`;
- coefficient output width: two; zero initialization for coefficient heads;
- optimizer: Adam `1e-3`, batch `2048`, maximum `80` epochs, minimum `15`,
  patience `10`, XLA enabled;
- calibration: one positive temperature, no intercept, fitted on calibration
  data to preserve the derivative anchor.

Controls are selected independently for each `(stage,horizon,coordinate)` on
the selection domain, then frozen before final-domain generation. V3 controls,
data, and artifacts are not reused.

## Admission

For each final replicate:

- exact balance at every delta;
- pooled test log loss beats `log(2)` by `2*SE`;
- calibration loss does not worsen by more than `1e-4`;
- per-delta ECE at most `0.04`;
- at least two informative deltas, maximum-delta AUC below `0.995`, and no
  AUC inversion larger than `0.03` between adjacent deltas;
- observed-path logit inside expanded held-out support at every delta;
- optimizer completion recorded and required.

For each horizon/coordinate, all three final replicates must pass. Their score
range must be at most `max(2,4*SE)` and `SE <= max(1,0.25*abs(mean))`. The exact
oracle additionally requires absolute score error at most `max(0.5,3*SE)` and
the exact score inside the same interval. All nine cells must pass before SIR.

## Assumption Audit And Pre-Mortem

| Choice | Status | Failure mode | Diagnostic |
|---|---|---|---|
| six perturbations | hypothesis | weak signal or saturation remains | per-delta AUC/ECE |
| two anchored modes | reviewed repair | curvature underfit | exact log-scale error and residual pattern |
| design orthogonality | mathematical construction | finite-sample weighting differs from design weights | coefficient covariance and oracle error |
| one temperature | hypothesis | delta-dependent calibration | per-delta ECE |
| three replicates | bounded diagnostic evidence | wide pointwise uncertainty | range and SE gates |
| quadratic path features | inherited comparator | high-dimensional variance | linear-vs-MLP selection and oracle |

The main pre-mortem concern is that design orthogonality does not imply
statistical orthogonality after the classifier learns `c0(y),c1(y)`. The exact
oracle therefore records coefficient covariance, score spread, and curvature
residuals; a successful AUC screen cannot promote the score.

## Execution And Artifacts

1. CPU-only focused tests.
2. Trusted GPU/XLA smoke.
3. One full exact Gaussian oracle in a fresh directory.
4. SIR only if exact status is `PASSED`.

Budgets: 20 GPU minutes smoke, 50 exact, 90 SIR. No scientific repair after
the full exact run. Artifact root:
`docs/benchmarks/artifacts/sir_anchored_orthogonal_ratio_score_20260814/`.

Audit status before implementation: `READY_FOR_REVIEW`.
