# Filter-Independent SIR Classifier-Ratio Score V2 Plan

Date: 2026-08-13  
Status: `EXECUTED_EXACT_ORACLE_FAILED_SIR_CONTINUATION_VETO`

## Research Question And Non-Negotiable Estimator

Can balanced classification of observation paths simulated at
`theta + epsilon e_j` versus `theta - epsilon e_j` estimate the marginal
observed-data SIR score at the fixed observation path for `T=20,40,50`?

The only eligible estimate remains

`calibrated_logit(y_obs)/(2*epsilon)`.

No latent path, state posterior, likelihood evaluator, complete-data score,
Fisher identity, state-estimation algorithm, or prior simulation-score artifact
may enter training, calibration, selection, extrapolation, or interpretation.
The lightweight-package source and runtime dependency vetoes remain mandatory.

## Why V2 Is Needed

V1 failed the exact Gaussian oracle after one implementation-mismatch repair.
The repaired run admitted only one of nine horizon/coordinate cells. V1 selected
one architecture per horizon by averaging validation loss across two Gaussian
location coordinates and one log-scale coordinate. This is a wrong tuning
scope: the exact location density ratio is linear in the full path, while the
exact log-scale ratio is quadratic. The hidden-layer MLP selected by the average
then overfit weak location ratios and produced unstable pointwise logits.

V2 changes only classifier candidates and selection scope. It does not change
the density-ratio identity, observation simulator, score formula, observed
path, data splits, epsilon ladder, admission diagnostics, extrapolation, or
exact-oracle continuation veto.

## Research Intent Ledger

| Field | Frozen V2 definition |
|---|---|
| Main question | Can an observation-only calibrated classifier estimate the fixed-path SIR score at `T=20/40/50`? |
| Mechanism | Balanced probabilistic classification of independently simulated full `[T,9]` observation paths |
| Expected V1 failure repaired | Shared tuning across incompatible ratio geometry and poorly initialized convex heads |
| Primary criterion | All nine exact Gaussian horizon/coordinate cells pass before SIR; each SIR cell then needs at least three admitted epsilons and stable extrapolation |
| Promotion veto | Any existing head-level calibration/signal/support gate or extrapolation gate fails |
| Continuation veto | Exact oracle fails any cell; forbidden dependency loads; split leakage; simulator parity failure; GPU/XLA/memory-policy failure; budget exhaustion |
| Repair trigger | No repair after the V2 exact claim run; failure returns to planning |
| Explanatory only | Architecture selected, AUC, validation loss, raw score variation, runtime |
| Must not be concluded | Passing is not exactness, algorithm correctness/ranking, HMC readiness, or default readiness |

## Data And Splits

Unchanged from V1:

- `theta=[0,0,0]`, horizons `20,40,50`, coordinates `0,1,2`;
- epsilons `0.01,0.02,0.04,0.08`;
- per class: train `2048`, validation `512`, Platt calibration `512`, untouched
  test `1024`;
- three independent final classifier/simulation replicates;
- selection domain `10`, final domain `20`;
- all paths generated to `T=50` and sliced into paired prefixes;
- fixed evaluated path excluded from every fit, selection, and calibration step.

## V2 Classifier Ladder

Every candidate consumes every standardized observation entry. No summary
statistic is supplied.

1. `linear_full_path`: flatten `z`; zero-initialized logistic head.
2. `linear_full_path_quadratic`: flatten `[z,z**2-1]`; zero-initialized logistic
   head. This is a convex quadratic-feature classifier, not an analytical
   density ratio.
3. `mlp_full_path_quadratic`: flatten `[z,z**2-1]`; tanh widths `(128,64)` and
   logistic output, as in repaired V1.

Zero initialization is used only for the convex one-layer heads. It starts at
the balanced null classifier and removes the V1 Glorot logit variance of order
path dimension. The MLP retains ordinary Glorot initialization because
zero-initializing all hidden layers would prevent symmetry breaking.

Regularization candidates remain `0` and `1e-5`. Adam learning rate `3e-4`,
maximum `160` full-batch epochs, minimum `20`, patience `12`, XLA, and held-out
Platt calibration are unchanged.

Controls are selected separately for each `(stage,horizon,coordinate)` on the
simulated selection domain at `epsilon=0.04`. Lowest mean validation log loss
wins, with the simpler candidate preferred when its loss is within one
selection standard error. The selected architecture and regularization are
frozen before final-domain generation and untouched tests. Exact-oracle
selections do not stamp SIR selections; SIR repeats selection using only SIR
simulations for its own coordinate/horizon scope.

## Unchanged Admission And Extrapolation

Each head must pass all V1 gates: finite output, test log loss better than
`log(2)` by `2*SE`, AUC in `[0.505,0.995]`, ECE at most `0.03`, Platt slope in
`[0.5,2]`, calibration loss not worsened by more than `1e-4`, and observed logit
inside expanded held-out support.

An epsilon is admitted only when all three final replicates pass. At least three
epsilons are required. Weighted regression of score on `epsilon**2` produces
the intercept. Leave-one-epsilon-out stability and smallest-two-epsilon
agreement use the V1 frozen rules. Exact-oracle error must be at most
`max(0.5,3*intercept_SE)` for all nine cells.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| Coordinate-specific selection | Repair hypothesis from V1 | Ratio geometry differs by parameter coordinate | Over-specialized controls fail to transfer to SIR | SIR performs its own simulated-data selection |
| Zero convex-head initialization | Optimization repair | Balanced logistic null is exactly zero | Slower movement for weak signal | selection validation loss and best epoch |
| Quadratic logistic candidate | Exact-oracle coverage hypothesis | Represents variance ratios without hidden-layer optimization | Too many coefficients and high pointwise variance | exact scale-coordinate cells |
| MLP retained | Existing enhanced candidate | May represent nonlinear SIR ratios | Overfit and unstable pointwise logit | untouched test, support, replicate spread |
| `epsilon=0.04` selection | inherited hypothesis | Middle of signal/bias ladder | wrong controls for smallest epsilon | per-epsilon final gates; no retuning |
| Three replicates | inherited bounded evidence | exposes initialization/simulation variation | wide SE and all-replicate admission failures | exact oracle; no ranking claim |
| Counts and optimizer unchanged | controlled repair | isolates classifier-form/tuning-scope defect | still underpowered | exact oracle is a hard stop |

## Skeptical Pre-Mortem

- A successful command could still answer the wrong question if it loads a
  state-estimation module: source and runtime audits fail closed.
- Exact-feature candidates could be accused of using the Gaussian oracle
  formula. They do not receive directions, sums, exact scores, or density
  coefficients; they learn unrestricted coefficients from labeled paths. The
  same generic candidates are used for SIR.
- Coordinate-specific selection could leak final data. It uses only domain `10`;
  all final fits use domain `20` and the fixed path is evaluated last.
- Adding candidates can overfit selection. The untouched exact final domain and
  three independent fits are the gate; architecture identity is explanatory.
- Passing Gaussian ratios may not establish SIR ratio accuracy. It is necessary,
  not sufficient; SIR results retain calibration, support, replication, and
  epsilon-stability gates and remain approximate.
- Small epsilons may remain too weak. If fewer than three epsilons pass, the
  procedure fails; no post hoc epsilon or threshold change is allowed.

## Evidence Contract And Budget

Baseline ladder: zero-initialized linear, zero-initialized quadratic logistic,
and repaired quadratic-feature MLP. The primary comparator is exact Gaussian
score, not a state-estimation algorithm. Head diagnostics are vetoes; runtime
and architecture choice are explanatory.

Run one fresh full exact-oracle attempt, budget at most 45 GPU minutes. If and
only if every exact cell passes, run one SIR campaign, budget at most 90 GPU
minutes. Use fresh versioned output roots. No scientific repair is authorized
after the V2 exact run. Preserve all failures.

Planned artifacts:

- exact: `docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/v2_exact_full_attempt01/`;
- SIR, only after exact pass:
  `docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/v2_sir_full_attempt01/`;
- terminal result/reset memo updated with both executed and vetoed stages.

Audit status: `READY_FOR_SKEPTICAL_REVIEW`.
