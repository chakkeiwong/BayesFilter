# BayesFilter SSL-LSTM Completion Phase A3 Forecast Oracle And Statistics Subplan

Date: 2026-07-11

Status: `HISTORICAL_SUPERSEDED_BY_ACADEMIC_TIER2_LIVE_PLAN`

## 2026-07-13 Proportional-Governance Reset

The owner-approved academic risk-tier policy in `AGENTS.md` supersedes this
document's frozen-`HEAD`, signed hash-chain, syscall-trace, finite-write-set,
phase-closure, and repeated-review requirements. Those records remain useful
historical evidence, but they are not active execution gates.

The mathematical definitions, statistical roles, failure modes, and nonclaims
below remain reference material. Active A3 execution is governed by
`docs/plans/bayesfilter-ssl-lstm-predictive-validation-live-plan-2026-07-13.md`.

Do not reactivate this document's historical governance machinery unless the
owner explicitly requests it after the 2026-07-13 policy reset.

## Phase Objective

Build and independently validate the engineering machinery needed to compare
joint 1-to-10-step forecast laws without running or interpreting an SSL-LSTM
sampler comparison. Phase A3 will provide:

1. an analytic scalar linear-Gaussian state-space model (LGSSM) forecast oracle;
2. TensorFlow `float64` forecast-summary and standardized-path features;
3. fixed-multi-bandwidth RBF diagonal-excluded MMD U-form and biased V-form
   statistics with distinct labels and inferential roles;
4. a separate cross-chain linear MMD estimator for inference near equality;
5. chain-aware batch means and chain-stratified hierarchical
   block/forecast-cluster resampling;
6. Bonferroni/studentized and bootstrap max-statistic simultaneous intervals
   with a joint feature/MMD alpha budget;
7. fail-closed evidence statuses that cannot turn failure to reject equality
   into evidence of practical equivalence; and
8. structured CPU-reference and trusted GPU/XLA oracle artifacts.

The phase is engineering and statistical-infrastructure validation only. It
does not calibrate the final confirmatory design, compare sampler arms, or
decide predictive equivalence.

## Governing Artifacts And Entry Conditions

A3 drafting inherits the accepted program semantics from:

- `docs/plans/bayesfilter-ssl-lstm-completion-roadmap-2026-07-11.md`;
- `docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md`; and
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md`.

The following A2 conditions authorize this draft but not implementation:

| Entry condition | Required state |
| --- | --- |
| A2 engineering result | `PASSED_FOR_A3_PLANNING_ONLY` |
| A2 executor ledger | `A2_EXECUTOR_WRITE_LEDGER_VALID` |
| A2 final checkpoint | `A2_PRE_RESULT_CHECKPOINT_PASSED` |
| A2 result review | Hash-bound bounded `VERDICT: AGREE`; Codex substitute, weaker than Claude |
| Protected A1 target SHA-256 | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |
| A2 predictive module SHA-256 | `0dad54c239de11f105f541527447d167114073ab046c796a813b5c1e867452ed` |
| A2 CPU artifact SHA-256 | `8bd1ed508e90674521774f73332e73e2a2f198a057879448dcddc0e30ed35df2` |
| A2 GPU artifact SHA-256 | `0294b06527620336e970bf6a57fd2e0f1a8466502bf47f9595a533d10ca23521` |
| A2 focused tests SHA-256 | `1812b338ff90633d2fa627642af8ba65425bdaf1c11211f8944d7207ecbded2c` |
| A2 independent verifier SHA-256 | `d0195063a1686a5332b6788bd1171ffc998370bd3578ceeb64edea240a2511ee` |
| A2 implementation/trace review SHA-256 | `1210e2fcced29448cbcdba7a4ce1dcee93326e3f317e27ec65d45c30364f23fb` |
| Repository `HEAD` | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |

A3 implementation is authorized only after all of these additional conditions
hold in order:

1. this exact subplan passes local/static consistency checks;
2. this exact subplan receives a bounded one-path review `VERDICT: AGREE`;
3. the execution, approval, runbook, and stop-handoff records are refreshed;
4. the A2 post-result write ledger and post-result closure are generated;
5. a fresh-process verifier returns `A2_POST_RESULT_CLOSURE_VERIFIED`;
6. the exact A2 terminal read-only trace audit returns
   `A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED`; and
7. the closure-verification trace SHA-256 and audit status are recorded before
   the first A3 source edit.

Any bound A1/A2 source or artifact drift before A3 boundary freeze is an entry
veto. Concurrent HMC/Kalman work is outside this lane and must be preserved,
not edited, staged, reverted, or interpreted by A3.

## Exact A2 Interfaces Consumed

A3 consumes the following accepted A2 public types and fields from
`bayesfilter/nonlinear/ssl_lstm_predictive_tf.py`:

```python
@dataclass(frozen=True)
class SSLLSTMForecastPaths:
    terminal_states: tf.Tensor
    states: tf.Tensor
    deterministic_transition_means: tf.Tensor
    process_innovations: tf.Tensor
    observation_means: tf.Tensor
    observation_innovations: tf.Tensor
    observations: tf.Tensor
    terminal: SSLLSTMTerminalState
    provenance: SSLLSTMForecastProvenance
```

The A3 statistics entry point consumes only `observations`, with canonical
static shape `[chain, draw, forecast_replication, horizon]` after a caller
adds the chain axis to the accepted A2 per-batch layout. A helper may transform
the A2 observation layout `[draw, forecast_replication, 10, 1]` to
`[1, draw, forecast_replication, 10]` by squeezing only the final singleton
observation coordinate. It must fail on any other observation dimension,
unknown static axis, non-`float64` dtype, nonfinite value, or horizon other than
10.

Every A3 SSL-LSTM-derived feature artifact must bind these A2 provenance fields:

- `a2_contract_signature`;
- `a1_adapter_signature`;
- `target_semantic_sha256`;
- `forecast_config_signature`;
- `innovation_bank_signature` and materialized innovation tensor hashes;
- `innovation_role` and `innovation_arm_id`;
- `draw_count`, `replication_count`, and `forecast_horizon`;
- `horizon_convention` and `cluster_unit`;
- `dtype`, `jit_compile`, `execution_role`, device rows, and `trust_basis`; and
- the terminal covariance statuses and approximation qualification.

A3 must not parse an internal filter trace, reconstruct terminal states, alter
the A2 forecast equations, regenerate an innovation bank from seed metadata, or
treat a root seed as cross-backend bitwise replay authority. Materialized tensor
hashes remain the innovation replay authority.

## Proposed A3 Engineering Interfaces

Names are provisional until implementation review, but semantics are binding.
New algorithmic code belongs in `bayesfilter/inference/predictive_equivalence.py`
and uses TensorFlow operations. The package root need not eagerly import the
module; if exports are added, they must be narrow and covered by import tests.

### Typed Configuration And Results

```python
DecisionStatus = Literal[
    "PASS",
    "MATERIAL_DIFFERENCE",
    "INCONCLUSIVE_UNDERPOWERED",
    "INVALID_HARD_VETO",
]

@dataclass(frozen=True)
class PredictiveStatisticsConfig:
    horizon: int = 10
    quantile_probabilities: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    central_moment_orders: tuple[int, ...] = (3, 4)
    jit_compile: bool = True

@dataclass(frozen=True)
class PredictiveSummary:
    means: tf.Tensor
    variances: tf.Tensor
    log_variances: tf.Tensor
    central_moments: tf.Tensor
    quantiles: tf.Tensor
    cross_horizon_covariance: tf.Tensor
    path_count: tf.Tensor
    status: tf.Tensor

@dataclass(frozen=True)
class MMDStatistics:
    squared_mmd_u: tf.Tensor
    squared_mmd_v_biased: tf.Tensor
    per_bandwidth_u: tf.Tensor
    per_bandwidth_v_biased: tf.Tensor
    bandwidths: tf.Tensor
    mixture_weights: tf.Tensor
    sampling_contract: Literal["iid_oracle_fixture", "dependent_descriptive_only"]
    inference_admissible: bool
    status: tf.Tensor

@dataclass(frozen=True)
class CrossChainLinearMMD:
    squared_mmd_linear: tf.Tensor
    kernel_contrast_sequence: tf.Tensor
    chain_pair_schedule: tf.Tensor
    independent_arm_banks_verified: bool
    stationarity_required: bool
    inference_admissible: bool
    status: tf.Tensor

@dataclass(frozen=True)
class SimultaneousIntervals:
    estimate: tf.Tensor
    lower: tf.Tensor
    upper: tf.Tensor
    standard_error: tf.Tensor
    method: Literal["bonferroni_studentized", "bootstrap_max_statistic"]
    status: tf.Tensor

@dataclass(frozen=True)
class PredictiveDecision:
    status: DecisionStatus
    primary_interval_status: DecisionStatus
    mmd_upper_bound_status: DecisionStatus
    hard_veto_codes: tuple[str, ...]
    explanatory_diagnostics: dict[str, tf.Tensor]
```

Production functions must have explicit keyword-only configuration and no
ambient RNG. At minimum, implement and test surfaces equivalent to:

```python
def summarize_forecast_paths(
    paths: tf.Tensor,
    config: PredictiveStatisticsConfig = PredictiveStatisticsConfig(),
) -> PredictiveSummary: ...

def standardize_forecast_paths(
    paths: tf.Tensor,
    center: tf.Tensor,
    scale: tf.Tensor,
    *,
    scale_floor: tf.Tensor,
) -> tf.Tensor: ...

def fixed_rbf_mmd(
    left_paths: tf.Tensor,
    right_paths: tf.Tensor,
    *,
    bandwidths: tf.Tensor,
    mixture_weights: tf.Tensor,
    sampling_contract: Literal[
        "iid_oracle_fixture", "dependent_descriptive_only"
    ],
) -> MMDStatistics: ...

def cross_chain_linear_mmd(
    left_paths: tf.Tensor,
    right_paths: tf.Tensor,
    *,
    bandwidths: tf.Tensor,
    mixture_weights: tf.Tensor,
    chain_pair_schedule: tf.Tensor,
    independent_arm_banks_verified: bool,
) -> CrossChainLinearMMD: ...

def chain_batch_means(
    values: tf.Tensor,
    *,
    block_length: int,
) -> tf.Tensor: ...

def hierarchical_resample_indices(
    *,
    chain_count: int,
    draw_count: int,
    forecast_replication_count: int,
    block_length: int,
    bootstrap_count: int,
    seed: tf.Tensor,
    chain_mode: Literal["stratified_fixed_chains"],
) -> HierarchicalBootstrapIndices: ...

def simultaneous_feature_intervals(..., *, feature_alpha: tf.Tensor) -> SimultaneousIntervals: ...

def cross_chain_mmd_upper_interval(..., *, mmd_alpha: tf.Tensor) -> MMDInterval: ...

def classify_predictive_evidence(
    ...,
    *,
    total_alpha: tf.Tensor,
    feature_alpha: tf.Tensor,
    mmd_alpha: tf.Tensor,
) -> PredictiveDecision: ...
```

The default `jit_compile=True` applies to algorithmic TensorFlow paths. A
non-JIT eager surface is a debug/reference exception and must be labeled as
such. Host-side artifact serialization and string status classification need
not be XLA compiled. A3 must not add NumPy to an algorithmic or
gradient-bearing path; an independent NumPy-free closed-form scalar reference
may use Python `math`/`decimal` only in the test fixture if needed.

## Analytic Scalar LGSSM Oracle

The independent fixture belongs provisionally at
`bayesfilter/testing/scalar_lgssm_forecast_oracle.py`. It defines the scalar
model

\[
x_{t+1} = a x_t + b + \eta_{t+1},\qquad
y_{t+1} = c x_{t+1} + d + \epsilon_{t+1},
\]

with independent \(\eta_t\sim N(0,q)\), \(\epsilon_t\sim N(0,r)\), and terminal
law \(x_T\mid y_{1:T}\sim N(m_T,P_T)\). For horizons \(h,k\in\{1,\ldots,10\}\),
the fixture derives, rather than estimates,

\[
E[x_{T+h}] = a^h m_T + b\sum_{j=0}^{h-1}a^j,
\]

\[
\operatorname{Cov}(x_{T+h},x_{T+k}) =
a^{h+k}P_T + q\sum_{j=1}^{\min(h,k)}a^{h-j}a^{k-j},
\]

and

\[
E[y_{T+h}] = cE[x_{T+h}]+d,
\]

\[
\operatorname{Cov}(y_{T+h},y_{T+k}) =
c^2\operatorname{Cov}(x_{T+h},x_{T+k}) + r\,\mathbf{1}[h=k].
\]

The implementation must handle `a=1` without a division-by-zero geometric-sum
shortcut, validate `P_T >= 0`, `q >= 0`, and `r >= 0`, and fail on nonfinite
inputs or a materially non-positive-semidefinite analytic covariance. The
primary derivation uses direct finite sums so no special stationary assumption
is hidden.

The fixture produces analytic mean, marginal variance/log variance, zero third
central moment, fourth central moment `3 * variance**2`, Gaussian quantiles,
and the full 10-by-10 covariance. Quantile reference values use a reviewed
TensorFlow Probability or TensorFlow Gaussian quantile operation when available;
if TFP is unavailable in the bound environment, the plan must be amended and
rereviewed before substituting an implementation.

The independent simulation route materializes terminal, process, and
observation standard-normal banks and constructs paths directly from the model
equations. It must not call the statistics implementation to construct expected
values and must not reuse the implementation formula as its only oracle.

## Statistical Semantics

### Observation And Cluster Units

One complete length-10 forecast path is the indivisible forecast cluster. The
canonical input axes are:

```text
[chain, retained_posterior_draw, forecast_replication, horizon]
```

Posterior dependence runs along the draw axis within each chain. Forecast
replications sharing a posterior draw are clustered and must never be flattened
into independent draws for uncertainty estimation. Chains are independent only
when their upstream sampler contract says so; the A3 LGSSM fixture creates
explicit independent chain banks.

Point summaries may reduce all admitted path clusters, but uncertainty code must
retain the hierarchy. A batch is a contiguous block of posterior draws within a
single chain and contains all forecast replications for those draws. Remainder
draws must follow an explicit policy, defaulting to fail closed in A3 rather
than silently dropping observations.

### Summary Features

Confirmatory-capable surfaces are implemented for horizon means and log
variances. Raw variances are retained to construct and audit log variances.
Variance uses an explicit denominator recorded in metadata: descriptive pooled
summaries use the sample denominator `N-1`, while analytic oracle variance is
the law variance. Zero or nonfinite sample variance makes log variance invalid
and fires a hard veto; no hidden epsilon may convert it to a finite value.

Third/fourth central moments, quantiles `(0.05, 0.25, 0.50, 0.75, 0.95)`, and
cross-horizon covariance are explanatory in A3 and remain explanatory in A4
unless a later reviewed plan explicitly promotes them. Quantile interpolation
semantics must be named and tested. Extreme quantiles and fourth moments cannot
rank candidates under A3 evidence.

### Standardized Paths

Standardization is componentwise by frozen inputs `center[h]` and `scale[h]`:

\[
z_{i,h}=(y_{i,h}-\text{center}_h)/\max(\text{scale}_h,\text{scale_floor}).
\]

For A3 oracle tests, the analytic LGSSM mean and standard deviation are allowed
as fixture values. This does not select the SSL-LSTM scales or the A4 scale
floor. Nonfinite center/scale, non-positive scale, or use of the floor by an
evidentiary row must be reported; an invalid scale is a hard veto, not silently
repaired.

### Fixed Multi-Bandwidth RBF MMD Descriptive Forms

For independent path samples `X={x_i}_{i=1}^m` and `Y={y_j}_{j=1}^n`, implement
the diagonal-excluded estimator

\[
\widehat{\operatorname{MMD}}_u^2 =
\frac{1}{m(m-1)}\sum_{i\ne i'}k(x_i,x_{i'}) +
\frac{1}{n(n-1)}\sum_{j\ne j'}k(y_j,y_{j'}) -
\frac{2}{mn}\sum_{i,j}k(x_i,y_j).
\]

The fixed mixture kernel is

\[
k(x,y)=\sum_{\ell=1}^{L}w_\ell
\exp\{-\lVert x-y\rVert^2/(2\sigma_\ell^2)\},
\quad w_\ell\ge0,\quad\sum_\ell w_\ell=1.
\]

For IID oracle samples this is the usual unbiased squared-MMD U-statistic. For
the canonical MCMC forecast hierarchy it is only a diagonal-excluded quadratic
U-form: within-arm off-diagonal pairs can be serially dependent and
forecast-cluster dependent, so finite-sample unbiasedness is not claimed. The
API must take a `sampling_contract` and may set `inference_admissible=True` only
for a verified IID oracle fixture. On dependent SSL-LSTM/MCMC paths the U-form
is descriptive only and can never supply the MMD confidence bound used by
`PASS`.

The signed diagonal-excluded value may be negative in finite samples and must
never be clipped to zero. The separately named `squared_mmd_v_biased` includes
the within-sample diagonals and is also explanatory only. Tests must distinguish
the forms and include signed-U behavior. `m < 2`, `n < 2`, duplicate or
non-positive bandwidths, invalid weights, nonfinite paths, inconsistent feature
dimensions, or a false IID declaration is a hard veto.

Common-random-number arm pairs are excluded from MMD inference. A shared-bank
U- or V-form may be emitted only as `paired_diagnostic_shared`, with matched
identifiers preserved and an explicit `inference_admissible=False`. It cannot
be pooled with, replace, or tune the independent-arm statistic. Both the
descriptive quadratic forms and the inferential route below must consume
independently generated arm-specific banks for any cross-arm law comparison.

A3 fixture bandwidths and weights are literal test inputs. A3 may demonstrate
multiple fixed bands but must not estimate, nominate, or freeze the A4 median
heuristic values, tolerance, or confirmation mixture.

### Cross-Chain Linear MMD For Inference

The MMD upper confidence bound used by decision logic is based on a separate
linear-time kernel-contrast estimator, not an ordinary bootstrap of the
degenerate quadratic U-statistic. For two distinct independent chains from
each arm and matched within-chain draw/block positions, define contrasts such
as

\[
g_t = k(X_{1,t},X_{2,t}) + k(Y_{1,t},Y_{2,t})
      - k(X_{1,t},Y_{2,t}) - k(X_{2,t},Y_{1,t}).
\]

Here each `X_{c,t}` or `Y_{c,t}` is the complete forecast-replication cluster
for posterior draw `t`. A kernel between two such clusters is the arithmetic
mean of `k(path_a, path_b)` over all cross-replication pairs from the two
distinct chains/arms. It never forms a within-posterior-draw within-arm kernel
pair. This integrates the forecast-replication dimension without pretending
replications sharing one posterior draw are independent posterior samples. The
nested forecast-cluster bootstrap resamples replication identifiers inside each
draw before recomputing these cluster-averaged kernels.

Under the declared contract that chains are independent across chain IDs, arms
use independent forecast banks, each chain is stationary after warmup, and the
kernel-contrast sequence satisfies the stated weak-dependence/mixing condition,
`E[g_t]` is squared MMD. Serial dependence along `t` is retained and handled by
moving/circular block resampling of the already constructed `g_t` sequence,
with the chain-pair sequence as the fixed stratum. This ordinary block-mean
route is admitted only because the linear contrast is nondegenerate and the
declared stationarity/mixing and block-growth assumptions hold; it is not
claimed valid for the degenerate quadratic U-form. Chain pairing schedules must be
fixed and materialized before outcomes are inspected; paths from the same chain
must never occupy both positions of a within-arm kernel term. With more than two
chains, prespecified disjoint pair schedules may be averaged or cross-fitted,
with dependence from reused chains accounted for by the resampling unit. The
implementation must not claim the quadratic U-form and this linear estimator
are numerically identical.

Inference-admissible A3 mode requires at least four valid independent chains per
arm so the fixed schedule can form at least two disjoint chain pairs without
reusing a chain, equal retained-draw/block geometry under the initial interface,
verified independent arm-specific innovation banks, at least two complete
blocks per chain-pair sequence, finite nonzero long-run uncertainty, and a
reviewed stationarity/mixing admission supplied by the caller. A3 synthetic
fixtures construct this contract. A4 must determine whether real sampler
artifacts meet it and must freeze the final disjoint-pair schedule and block
design. A schedule that reuses chains requires a separate covariance derivation
and reviewed extension; it is not part of A3. Failure of any prerequisite yields
`INVALID_HARD_VETO`; a two-chain mechanics fixture sets
`inference_admissible=False` and cannot emit `PASS`.

The null is nondegenerate for the linear contrast in general because individual
`g_t` terms retain variance even when their expectation is zero. A3 must still
test exact/near equality, zero-variance edge cases, and boundary alternatives.
If a fixture produces degenerate long-run variance, the ordinary interval route
is invalid and must fail closed; it may not substitute the biased V-form.

### Chain-Aware Batch Means

Batch means preserve a leading chain axis and partition only the retained-draw
axis into contiguous non-overlapping blocks. All forecast replications within a
draw/block remain attached. The implementation must expose batch count and
block length and reject:

- fewer than four independent chains per arm for inference-admissible
  cross-chain linear MMD
  estimator;
- fewer than two complete batches per chain;
- block length outside `[1, draw_count]`;
- unequal chain lengths unless explicitly represented with an admitted mask;
- silent remainder truncation; and
- nonfinite inputs or nonfinite covariance/standard-error outputs.

A3 tests may exercise small literal block lengths and two-chain examples only
as mechanics fixtures. Mechanics-only status is machine readable, cannot emit
`PASS`, and does not select the A4 block length or establish adequate effective
sample size for SSL-LSTM confirmation.

### Hierarchical Bootstrap

For A3 and the four-chain downstream design, the default bootstrap keeps chain
IDs fixed and stratifies resampling within every chain. Resampling the empirical
distribution of only two or four chain IDs is forbidden as an inferential
default because it cannot reliably estimate between-chain uncertainty. The
chain axis remains an explicit independent replication/stratum and uncertainty
is combined across admitted chains or fixed chain-pair contrast sequences.

The within-chain hierarchy is binding:

1. retain each admitted chain or prespecified distinct-chain contrast pair as a
   fixed stratum;
2. within each chain/sequence, resample circular or moving contiguous draw
   blocks according to an explicit mode;
3. within each selected posterior draw, resample forecast-replication cluster
   identifiers while retaining all ten horizons jointly; and
4. recompute the entire fixed feature vector or MMD statistic for that replicate.

The stateless `int32[2]` seed and all generated integer index tensors are
materialized and hash-bound. Same seed/input/config must replay exactly;
changed seed must change at least one index in a nondegenerate fixture. The
bootstrap must not resample horizons independently, pool chains before block
resampling, bootstrap a tiny empirical distribution of chain IDs, put the same
chain on both sides of a cross-chain kernel contrast, or treat forecast
replications as independent posterior draws.

The hierarchical bootstrap design, mode, count, seed, and block length used in
A3 are oracle-test fixtures only. A4 alone may calibrate and freeze their
confirmatory values.

### Simultaneous Intervals, Joint Alpha, And Decision Logic

The Bonferroni/studentized reference uses a family-wise confidence level and
the number of co-primary components explicitly. It must not use a pointwise
critical value while labeling the result simultaneous. The max-statistic route
uses the maximum absolute studentized centered bootstrap deviation across the
fixed feature family. The quadratic MMD U-form is not inserted into this
ordinary centered/studentized bootstrap. Its decision-bound MMD input comes
only from the admitted cross-chain linear contrast and its chain-stratified
block interval.

The caller supplies `total_alpha`, `feature_alpha`, and `mmd_alpha`, with strict
admission requiring positive finite values and
`feature_alpha + mmd_alpha <= total_alpha`. The feature procedure controls its
entire family at `feature_alpha`; the MMD upper interval uses `mmd_alpha`.
Bonferroni across these two co-primary families therefore provides the declared
overall confidence lower bound. A3 tests the allocation mechanism over literal
fixture allocations, but A4 alone freezes the final numerical allocation and
confidence level. A single joint max construction may be added later only under
a reviewed derivation and coverage tests; it is not assumed here.

Zero/nonfinite standard errors, too few valid bootstrap replicates, singular
required covariance without a supplied test-fixture ridge, invalid alpha
allocation, nonfinite critical values, or non-admissible MMD inference fire
`INVALID_HARD_VETO`.

Classification is fail closed:

| Condition | Status |
| --- | --- |
| Any input, covariance, resampling, interval, provenance, or artifact hard veto | `INVALID_HARD_VETO` |
| At least one valid simultaneous interval lies wholly above `+margin` or wholly below `-margin`, or a valid cross-chain linear-MMD lower confidence bound lies strictly above its supplied test-fixture tolerance | `MATERIAL_DIFFERENCE` |
| Every valid simultaneous equivalence interval lies strictly inside its supplied margins, a valid cross-chain linear-MMD upper bound lies strictly below its supplied tolerance, the joint alpha allocation is valid, and both evidence paths are inference-admissible | `PASS` |
| Neither direction is established, including any interval that overlaps/crosses a practical margin or an MMD interval that overlaps its tolerance, regardless of whether equality is rejected | `INCONCLUSIVE_UNDERPOWERED` |

For A3, margins and MMD tolerances are synthetic test arguments chosen to make
decision branches testable. A `PASS` means only that the implementation emitted
the correct branch for a controlled oracle fixture. It is not an SSL-LSTM
predictive-equivalence pass and does not authorize a sampler run. A two-sample
equality-test p-value, if retained, is explanatory only and can never by itself
produce `PASS`.

## Research Intent Ledger

| Field | A3 contract |
| --- | --- |
| Main question | Does the forecast-statistics engine reproduce an independently derived scalar LGSSM 1-to-10-step law and respond correctly to controlled law perturbations while respecting chain/draw/forecast clustering? |
| Candidate/mechanism | TensorFlow `float64`, XLA-default predictive summaries, descriptive quadratic MMD U/V forms, cross-chain linear MMD inference, chain batch means, chain-stratified block/forecast-cluster bootstrap, joint-alpha simultaneous intervals, and fail-closed classification. |
| Expected failure mode | Shared formula lineage; flattened clusters; a dependent quadratic U-form mislabeled unbiased; common-random-number MMD used inferentially; diagonal leakage or negative clipping; degenerate-null bootstrap misuse; tiny-chain resampling; missing joint alpha control; pointwise intervals mislabeled simultaneous; invalid variance/ridge silently repaired; equality non-rejection mislabeled equivalence; low fixture power; stale A2 bindings; CPU/GPU environment mismatch. |
| Promotion criterion | Exact analytic formula checks pass; Monte Carlo paths agree with the analytic law under predeclared uncertainty-aware fixture checks; IID/dependent and descriptive/inferential MMD roles remain separated; cross-chain linear-MMD null/boundary coverage and cluster semantics pass; joint alpha is enforced; controlled perturbations exercise the intended branches; CPU reference and trusted GPU/XLA agree within reviewed numerical tolerances; artifacts and reviews validate. |
| Promotion veto | Any analytic formula mismatch, nonfinite or invalid covariance, incorrect cluster/index semantics, false IID/unbiased claim, U/V/linear-estimator role confusion, non-admissible MMD bound, invalid alpha allocation, small-chain mechanics emitting `PASS`, failure-to-reject-to-pass bug, compiled-device/JIT contract failure, artifact/provenance mismatch, or review nonconvergence. |
| Continuation veto | Oracle, harness, implementation, artifact, or environment invalidity that prevents trustworthy A3 evidence, including inability to derive/verify the analytic LGSSM covariance or inability to preserve the resampling hierarchy. |
| Repair trigger | A controlled alternative is not detected at the provisional fixture budget while analytic formulas and harness validity remain intact; Monte Carlo error is too large; a provisional ridge/fixture scale is ill-conditioned; or focused review finds a local correctable defect. |
| Explanatory diagnostics | High central moments, quantiles, covariance entries, V-form, dependent-sample quadratic U-form, coverage/power point estimates, interval widths, bootstrap distributions, runtime, CPU/GPU residuals, and equality-test p-values. |
| What must not be concluded | SSL-LSTM predictive equivalence, posterior correctness, parameter agreement, HMC/NeuTra validity or readiness, calibrated margins/bandwidths/blocks/seeds, adequate confirmatory power, sampler ranking, model adequacy, production/default/public API readiness, or scientific validity. |

Before any stop, the result must state whether the target, data, analytic math,
harness, implementation, or artifact was invalid, or whether only a provisional
candidate/fixture lacked power. A valid weak-power result triggers the smallest
repair or A4 calibration planning; it does not reject the predictive-validation
research direction.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can a TensorFlow/XLA statistics engine reproduce an independent scalar LGSSM forecast law and preserve the dependence structure required by later predictive comparison? |
| Exact baseline/comparator | Closed-form scalar LGSSM mean/full covariance and direct equation-level simulation, not the SSL-LSTM implementation and not a sampler output. |
| Primary pass/fail criterion | Conjunctive analytic formula, controlled fixture, MMD role/estimand, cross-chain null/boundary coverage, joint-alpha, hierarchy/replay, decision-logic, CPU-reference, trusted GPU/XLA, artifact, and review gates. |
| Promotion vetoes | The hard vetoes in the research-intent ledger; any one prevents A3 engineering promotion. |
| Continuation vetoes | Evidence invalidity, broken analytic assumptions, missing required artifacts, or environment inability that prevents the oracle question from being answered. |
| Repair triggers | Underpowered but valid fixture, local numerical instability, or a correctable implementation/review finding. |
| Explanatory only | High moments, quantiles, V-form, dependent quadratic U-form, descriptive coverage/power, runtime, raw tails, and continuous residuals after hard screens. |
| What will not be concluded | All nonclaims remain binding even if every A3 gate passes. |
| Preservation artifact | Structured JSON/log/trace files under `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/` and a reviewed A3 result under `docs/plans/`. |

## Baseline Ladder

| Rung | Role | Required A3 evidence |
| --- | --- | --- |
| Closed-form scalar LGSSM | Independent analytic baseline | Exact finite-sum 1-to-10-step mean and full covariance, Gaussian moment/quantile identities, PSD checks. |
| Direct LGSSM equation simulation | Independent stochastic reference | Materialized innovations, exact replay, empirical-to-analytic uncertainty-aware agreement. |
| Identical-law independent-bank pair | Null mechanics | Valid intervals/MMD, no hard veto, and proof that equality non-rejection alone cannot produce `PASS`. |
| Controlled mean perturbation | Sensitivity fixture | Correct affected horizons/features and valid material-difference branch at the declared fixture budget. |
| Controlled variance perturbation | Sensitivity fixture | Log-variance response and MMD response without a false mean-only explanation. |
| Controlled skew perturbation | Omnibus/explanatory sensitivity fixture | MMD/third-moment response; third moment remains explanatory. |
| Controlled dependence perturbation | Joint-path sensitivity fixture | Cross-horizon covariance/MMD response with matched marginal means/variances where feasible. |
| TensorFlow CPU/XLA reference | Backend reference exception | Same algorithms, GPU hidden, JIT on, `float64`, structured provenance. |
| Trusted GPU/XLA route | Default execution target | GPU placement, one stable concrete trace per static program, finite output, and CPU agreement within reviewed tolerance. |

No ordinary-HMC, affine-HMC, NeuTra-HMC, or SSL-LSTM sampler rung is executed
in A3. Those master-program rungs remain downstream.

## Evidence Roles

| Evidence | Role |
| --- | --- |
| Analytic LGSSM formulas and exact deterministic checks | Promotion criterion and hard veto |
| Direct-simulation empirical agreement with uncertainty-aware thresholds | Promotion criterion; hard veto only when the predeclared check fails validly |
| IID U-statistic definition, signed behavior, U/V separation, and dependent U-form labeling | Promotion criterion and hard veto |
| Cross-chain linear-MMD estimand, independent-bank admission, distinct-chain schedule, null/boundary coverage | Promotion criterion and hard veto |
| Chain-stratified block/forecast-cluster index semantics and seed replay | Promotion criterion and hard veto |
| Joint feature/MMD alpha allocation | Promotion criterion and hard veto |
| Failure-to-reject cannot emit `PASS` test | Promotion criterion and hard veto |
| Controlled perturbation branch tests | Promotion criterion; valid low power is a repair trigger unless a predeclared minimum mechanics test fails |
| Third/fourth moments, quantiles, covariance differences | Explanatory diagnostic |
| Biased MMD V-form and dependent-sample quadratic U-form | Explanatory diagnostic only |
| Raw runtime and CPU/GPU continuous residuals | Explanatory after placement/JIT/finite vetoes pass |
| A3 fixture coverage/power frequencies | Descriptive only unless uncertainty intervals and replication design are predeclared in the artifact |
| Local smoke/import/static checks | Engineering prerequisites, never scientific evidence |

No proxy may silently become a promotion criterion. In particular, a small
validation loss, a visually close moment table, an equality p-value, a biased
MMD V-form, a dependent quadratic U-form, or a one-seed power point estimate
cannot promote A3.

## Required Artifacts And Schemas

All JSON is canonical, duplicate-key rejecting, nonfinite-rejecting, and includes
an `evidence_signature` computed after removing only explicitly volatile time and
wall-time fields. Every serious run includes a manifest with commit, dirty state,
exact command, interpreter/conda environment, Python/TensorFlow/TFP versions,
CPU/GPU visibility, XLA/TF32/dtype, trust basis, data/fixture signature, seeds,
wall time, source hashes, plan/result paths, and output paths.

Required Phase A3 directory:

`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/`

Required artifacts:

| Artifact | Minimum content/status |
| --- | --- |
| `pre-run-boundary.json` | Literal A3 write set, inherited A1/A2 bindings, outside-write inventory, `A3_SCOPED_BOUNDARY_FROZEN` |
| `fixture-contract.json` | Exact LGSSM inputs in hexadecimal form, analytic formulas/conventions, controlled perturbation definitions, test-only constants, fixture signature, `A3_FIXTURE_CONTRACT_FROZEN` |
| `oracle-cpu-reference.json` | Analytic and direct-simulation summaries, quadratic MMD U/V role labels, cross-chain linear-MMD contrasts/intervals, joint alpha allocation, decision rows, replay/index hashes, uncertainty method, source hashes, run manifest, `A3_CPU_ORACLE_PASSED` |
| `oracle-cpu-reference.log` | Human-readable check/status lines only |
| `oracle-gpu-xla-canary.json` | Same fixed fixture, device/HLO/trace provenance, CPU crosslink and residuals, `A3_GPU_XLA_ORACLE_PASSED` |
| `oracle-gpu-xla-canary.log` | Human-readable check/status lines only |
| `*-generation-write-trace.log` | File-mutation trace for each evidentiary generation command |
| `*-verification-write-trace.log` | File-mutation trace for each fresh-process verifier |
| `*-verify.log` | Fresh-process verifier status |
| `executor-write-ledger.json` | Direct and subprocess mutations against the frozen A3 boundary |
| `final-checkpoint.json` | Hash-bound sources, tests, fixtures, artifacts, reviews, ledgers, and no-cache state |
| A3 result Markdown | Decision/inference tables, manifests, repair record, red team, A4 handoff, nonclaims |
| A3 implementation review record | Exact bounded review status and findings; Claude preferred, substitution labeled weaker if required |
| A3 result review record | Exact bounded result review status and findings |
| `post-result-write-ledger.json` and `post-result-closure.json` | Post-result governance closure and bound A4 draft/review |

The CPU and GPU artifacts must contain separate ledgers for:

1. engineering correctness;
2. numerical/statistical validity; and
3. scientific interpretation/nonclaims.

The inference-status table in the A3 result must include at least: hard-veto
screen, statistically supported ranking, descriptive-only differences,
default-readiness, and next evidence needed. The statistically supported ranking
row must say that no sampler or method ranking was attempted.

## Exact A3 Write Set

Before execution, the boundary verifier must freeze a literal write set no
broader than:

### Production And Test Code

- `bayesfilter/inference/predictive_equivalence.py`;
- `bayesfilter/testing/__init__.py` only if required for a narrow lazy export;
- `bayesfilter/testing/scalar_lgssm_forecast_oracle.py`;
- `tests/test_predictive_equivalence.py`;
- `tests/test_scalar_lgssm_forecast_oracle.py`;
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py`;
- `docs/benchmarks/verify_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py`.

`bayesfilter/inference/__init__.py` is deliberately excluded because it is
currently modified by the concurrent HMC lane. A3 imports the new submodule
directly and does not edit that shared package initializer.

### A3 Plans, Results, Reviews, And Governance

- this A3 subplan;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md`;
- A3-only bounded review records under `docs/reviews/`;
- the four existing SSL-LSTM completion governance files: execution ledger,
  approval ledger, gated runbook, and stop handoff; and
- the A4 calibration subplan only after an A3 result is written and before A3
  post-result closure.

### Structured And Temporary Outputs

- the exact Phase A3 artifact directory listed above; and
- `/tmp/bayesfilter-a3-pycache`, `/tmp/bayesfilter-a3-tmp`, and their declared
  TensorFlow/XLA/CUDA subpaths.

No existing A1/A2 source, test, runtime artifact, result, or review may be
modified. No HMC/Kalman path, repository cache, model file, package metadata,
public docs chapter, `.git` state, or path outside the literal boundary may be
written. Any new required path stops execution for a visible subplan amendment
and rereview.

## Implementation Sequence

### A3.0 Boundary And Fixture Freeze

1. Verify A2 post-result closure and terminal trace audit.
2. Hash the accepted A1/A2 inputs and record concurrent outside-write paths.
3. Freeze the literal A3 write boundary.
4. Write an exact scalar-LGSSM fixture contract with hexadecimal numeric inputs,
   direct finite-sum formula conventions, seeds, and controlled alternatives.
5. Label every margin, bandwidth, block length, bootstrap count, critical level,
   and perturbation magnitude in this artifact `A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN`.

### A3.1 Independent Oracle

1. Implement scalar analytic mean and full covariance from direct finite sums.
2. Implement direct equation-level path generation from materialized innovations.
3. Test `a=1`, stable and negative `a`, zero process/observation variance edges,
   invalid variance inputs, symmetry, and PSD behavior.
4. Cross-check selected horizons by manual recurrence independent of the vector
   routine.

### A3.2 Forecast Summaries And Standardization

1. Implement strict shape/dtype/finite admission.
2. Implement means, variance/log variance, central moments, named quantiles, and
   cross-horizon covariance.
3. Implement explicit standardization inputs and scale-floor audit behavior.
4. Verify analytic Gaussian identities and direct-simulation agreement.

### A3.3 MMD And Dependence-Aware Uncertainty

1. Implement fixed mixture RBF kernels without adaptive bandwidth selection.
2. Implement diagonal-excluded signed quadratic U-form and separately labeled
   biased V-form values with mandatory IID/dependent sampling contracts.
3. Implement the independent-bank cross-chain linear MMD contrast, fixed
   distinct-chain schedule, and inference-admission status.
4. Implement chain batch means with explicit remainder policy.
5. Implement materialized chain-stratified block/forecast-cluster indices; do
   not bootstrap the tiny empirical distribution of chain identifiers.
6. Implement Bonferroni/studentized and feature max-statistic intervals plus a
   separate cross-chain linear-MMD upper interval.
7. Implement and validate the supplied joint feature/MMD alpha allocation.
8. Implement fail-closed decision classification.

### A3.4 Controlled Alternatives

Exercise independent pairs for:

- identical Gaussian laws;
- a known horizon-mean shift;
- a known scale/log-variance change;
- a centered skew perturbation with finite moments;
- a dependence perturbation preserving marginal means and variances; and
- singular/degenerate covariance and underpowered fixtures.

The identical and near-boundary law fixtures must exercise repeated independent
replications of the cross-chain linear-MMD interval and record empirical
coverage with a predeclared exact binomial uncertainty interval. The fixture
contract must predeclare enough replications and a test-only coverage slack; A3
promotion requires the lower confidence bound for coverage to be no less than
`nominal_coverage - fixture_coverage_slack`. Neither a point estimate nor a very
wide interval that merely contains nominal coverage can pass. Exact-degenerate
contrast fixtures must fail closed rather than borrow the quadratic U- or biased
V-form. These replication counts and slack are A3 validation fixtures, not A4
confirmatory settings.

For mean, variance, and covariance alternatives, analytic changed-law targets
must be recorded. For the skew alternative, construction and expected moment
direction must be explicit; the fixture must not claim a closed-form law metric
that was not derived. A valid failure to detect a deliberately small alternative
is `INCONCLUSIVE_UNDERPOWERED`, not an implementation veto, unless the exact
predeclared mechanics assertion itself fails.

### A3.5 Focused Verification And Runtime Artifacts

1. Run in-memory compilation and static backend/RNG/import scans.
2. Run focused CPU-hidden tests with repository caches disabled and writes traced.
3. Obtain bounded implementation review before evidentiary runtime generation.
4. Generate the CPU-hidden XLA reference artifact.
5. Fresh-process verify the CPU artifact by replaying formulas, simulation,
   index hashes, evidence signature, and source bindings.
6. Generate the trusted managed-session GPU/XLA artifact from the same
   materialized fixture.
7. Fresh-process verify GPU placement, HLO/trace counts, CPU crosslink, finite
   outputs, source bindings, and evidence signature.
8. Audit every trace against the literal boundary and confirm no A3-named
   repository caches.

### A3.6 Result, A4 Draft, And Closure

1. Write the A3 result with the required decision and inference-status tables,
   manifests, uncertainty qualifications, repair record, and red team.
2. Draft the A4 calibration-and-freeze subplan from actual A3 interfaces and
   evidence, without executing calibration.
3. Review the A3 result and A4 subplan as separate bounded exact-path reviews.
4. Repair material findings within the allowed loop and rerun focused checks.
5. Generate post-result ledger/closure artifacts, fresh-process verify closure,
   and terminally audit the read-only verification trace.

## Required Local Checks And Tests

The exact commands are frozen in the fixture/boundary artifact before execution.
They must use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, disable repository
bytecode/pytest caches, and direct temporary/XLA/CUDA writes to the reviewed
`/tmp/bayesfilter-a3-*` roots.

Minimum static checks:

- in-memory `compile()` for every new Python file;
- `git diff --check` restricted to A3-owned paths;
- no NumPy, PyTorch, or JAX import in production;
- no stateful TensorFlow RNG in production or fixture paths;
- no import from `docs.benchmarks` in production or tests;
- no A3 edit to `bayesfilter/inference/__init__.py`;
- no HMC/NeuTra invocation in the A3 harness;
- no repository `__pycache__`, `.pyc`, or `.pytest_cache` whose name binds an A3
  module/harness; and
- source/hash inventory against the frozen boundary.

Minimum focused test matrix:

| Test family | Required cases |
| --- | --- |
| Input contract | Static four-axis shape, singleton observation adapter, `float64`, finite values, horizon 10, invalid axes/dtypes/nonfinite rejection |
| Analytic LGSSM | Horizons 1..10, direct recurrence, full covariance symmetry/PSD, `a=1`, negative `a`, zero/noise edges, invalid variance fail-closed |
| Gaussian summaries | Mean, variance/log variance, central moments 3/4, named quantiles, cross-horizon covariance |
| Standardization | Known center/scale, non-positive/nonfinite scale rejection, floor-use audit |
| MMD descriptive forms | Identical arrays, verified IID identical-law samples, dependent-sample label, common-random-number diagnostic exclusion, unequal sample counts, multiple fixed bands, invalid weights/bands, signed U-form, U/V distinction |
| Cross-chain linear MMD | Distinct-chain schedule, same-chain rejection, independent-bank admission, exact/near equality, boundary alternatives, stationary dependent sequences, zero long-run-variance veto, repeated coverage with binomial uncertainty |
| Batch means | Per-chain contiguous blocks, all replications retained, insufficient batches, remainder veto, invalid block length |
| Hierarchical bootstrap | Exact seed replay, changed-seed sensitivity, fixed-chain stratification, no tiny-chain resampling, contiguous draw blocks, forecast-cluster resampling, horizons retained jointly |
| Intervals | Bonferroni feature-family correction, feature max-statistic construction, cross-chain linear-MMD upper interval, joint alpha allocation, zero/nonfinite SE veto, too-few bootstrap replicates, supplied-margin strictness |
| Decision logic | All four statuses, hard-veto precedence, equality non-rejection cannot emit `PASS`, mechanics-only cannot emit `PASS`, non-admissible MMD cannot emit `PASS`, wide-zero-containing interval is inconclusive |
| Alternatives | Controlled mean, variance, skewness, and dependence changes plus explicitly underpowered cases |
| XLA | Default JIT true, eager debug parity, stable concrete trace count, finite outputs |
| Devices | CPU-hidden reference label and trusted GPU output placement |
| Artifacts | Strict schemas, duplicate-key/nonfinite rejection, signature replay, source/config/fixture/CPU-GPU crosslinks |

Monte Carlo-to-analytic checks must predeclare uncertainty-aware acceptance
rules based on analytic standard errors, simultaneous confidence bounds, or a
replicated interval. A raw absolute tolerance chosen after seeing a run is
forbidden. Exact deterministic formula checks may use scale-aware floating-point
tolerances derived from machine epsilon and operation count.

## Review And Repair Loop

Claude is a read-only reviewer and Codex remains supervisor/executor. Review
prompts start with one exact path and one question:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md.
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does this A3-only plan correctly and feasibly validate the independent LGSSM
oracle, predictive statistics, dependence-aware uncertainty, evidence roles,
boundaries, and A4 handoff without authorizing sampler comparison or claiming
equivalence? Report material findings first with path/line references. End with
VERDICT: AGREE or VERDICT: REVISE.
```

Use the trusted Claude review gate. If the probe fails, record transport/policy
status and follow the already authorized bounded Codex-substitute policy. A
Codex substitute must be fresh, read-only, explicitly weaker than Claude, and
must not be described as Claude convergence.

For a fixable `REVISE`:

1. write the finding to the visible ledger/review record;
2. patch the same exact artifact visibly within the allowed write set;
3. rerun focused static or numerical checks affected by the patch;
4. rehash the exact reviewed artifact; and
5. request another bounded review.

Stop after five substantive rounds for the same unresolved blocker unless the
owner explicitly grants more. A timeout, no-verdict response, or stalled worker
is not a substantive agreement. Review cannot authorize HMC, NeuTra, model-file,
funding, product, release, scientific-claim, public API, or default-policy
boundaries.

## Skeptical Plan Audit

| Required challenge | Audit result before execution |
| --- | --- |
| Wrong baseline | Avoided: the primary oracle is a derived scalar LGSSM and direct equation simulation, not A2/SSL-LSTM output or a sampler proxy. |
| Proxy promoted | Avoided: equality p-values, V-form, dependent quadratic U-form, high moments, quantiles, runtime, validation-style closeness, and one-seed power remain explanatory. |
| Missing stop rules | Explicit entry, hard-veto, continuation-veto, repair, review-cap, write-boundary, and next-phase conditions are present. |
| Unfair comparison | CPU/GPU use identical materialized fixtures; controlled law pairs record exact intended changes; independent-bank MMD semantics are preserved. |
| Hidden assumptions | Scalar equations, terminal law, horizon timing, covariance formula, shape hierarchy, denominators, IID versus dependent MMD roles, cross-chain independence/stationarity/mixing admission, distinct-chain schedule, kernel definition, joint alpha, interval family, and decision meaning are explicit. |
| Stale context | A1/A2 hashes and closure are reverified immediately before boundary freeze; later drift invalidates A3 artifacts. |
| Environment mismatch | CPU is deliberately hidden/reference-only; trusted GPU/XLA is the serious default route; dtype/JIT/device/TF32 are recorded. |
| Commands answer the question | Formula, direct simulation, controlled alternatives, cluster-index audits, and fresh artifact replay directly test the A3 engineering question. No sampler command is included. |
| Artifact sufficiency | Structured artifacts preserve formulas, fixtures, seeds/indices, outputs, uncertainty rules, source hashes, runtime provenance, reviews, and decision/nonclaim ledgers. |

Audit decision: `PASS_FOR_A3_SUBPLAN_REVIEW_ONLY`. Implementation remains
forbidden until bounded review convergence and A2 post-result closure.

## Pre-Mortem

How A3 could pass while misleading us:

- analytic and TensorFlow paths could share the same indexing error;
- a flattened forecast-replication hierarchy could make intervals too narrow;
- the quadratic MMD U-form could accidentally include diagonals or clip negatives;
- dependent quadratic MMD pairs could be mislabeled IID/unbiased;
- common-random-number arm pairs could leak into an inferential MMD bound;
- a cross-chain schedule could reuse the same chain within a kernel contrast;
- two/four chain IDs could be bootstrapped as though they estimated a chain
  population;
- separately valid feature and MMD intervals could omit joint error control;
- a low-power fixture could make identical and perturbed pairs both look benign;
- GPU/CPU agreement could preserve a common formula error; or
- synthetic margins could be mistaken for calibrated SSL-LSTM margins.

Cheap discriminators required before long artifact runs:

- hand-expanded two- and three-horizon covariance examples;
- integer-index inspection on a tiny chain/draw/replication tensor;
- a deliberately negative finite-sample U-statistic fixture;
- a dependent-sample U-form row forced to `inference_admissible=False`;
- exact/near-null cross-chain linear-MMD coverage with binomial uncertainty;
- same-chain and shared-bank MMD admission failures;
- invalid and valid joint-alpha allocation fixtures;
- a mechanics-only small-chain row that cannot emit `PASS`;
- decision-table unit tests that inject intervals directly;
- controlled dependence changes with unchanged univariate marginals; and
- machine-readable `A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN` labels on every
  provisional number.

How A3 could fail for engineering or tuning reasons rather than invalidate the
idea:

- too few paths/replicates can cause descriptive Monte Carlo disagreement;
- a provisional block length can leave too few batches;
- a zero empirical variance fixture can invalidate log variance;
- a bootstrap count can be too small for a stable max critical value; or
- XLA may reject an otherwise valid dynamic-shape implementation.

These trigger the smallest discriminating fixture or static-shape repair while
the analytic target remains valid. Only broken oracle assumptions, invalid
math/harness/artifacts, or inability to preserve the hierarchy is a research
continuation veto.

## Required A3 Result And Post-Run Red Team

The A3 result must report:

- exact commands actually run and complete run manifests;
- deterministic analytic residuals and their predeclared tolerances;
- stochastic checks with replication counts and uncertainty intervals;
- hard vetoes supported or cleared;
- which fixtures remain viable;
- whether any ranking is statistically supported (`none` is expected);
- which differences are descriptive only;
- what additional evidence A4 needs before a defensible calibration;
- separate candidate rejection versus research-direction status; and
- a decision table with criterion status, veto status, uncertainty, next action,
  and nonconclusions.

The post-run red team must identify:

1. the strongest alternative explanation for all observed agreement;
2. the strongest alternative explanation for any controlled-alternative miss;
3. what result would overturn the A3 engineering conclusion;
4. the weakest evidence item; and
5. whether A4 can use A3 only as validated machinery or also as a calibration
   input, with the latter requiring explicit justification and separation.

## Forbidden Claims And Actions

- Do not run ordinary HMC, affine HMC, NeuTra-HMC, NeuTra training, or any
  sampler comparison.
- Do not use or alter HMC/Kalman files owned by the concurrent lane.
- Do not claim predictive equivalence, posterior correctness/equality,
  identification, model adequacy, convergence, HMC/NeuTra readiness, sampler
  superiority, or scientific validity.
- Do not calibrate, nominate, freeze, or reuse as final the A4 margins, MMD
  bandwidths/weights/tolerance, scale floor, ridge, block/bootstrap design,
  confidence level, sample counts, or confirmation seeds.
- Do not promote third/fourth moments, quantiles, covariance entries, V-form,
  equality tests, runtime, or pointwise power estimates into confirmatory gates.
- Do not treat failure to reject equality as equivalence evidence.
- Do not clip a negative MMD U-form to zero, label the V-form unbiased, label a
  dependent quadratic U-form finite-sample unbiased, or use either quadratic
  form for the dependent-sample MMD confidence bound.
- Do not use a common-random-number MMD row, a same-chain kernel contrast, a
  mechanics-only small-chain row, or an invalid joint-alpha allocation to emit
  `PASS`.
- Do not flatten chains, posterior draws, forecast replications, or horizons in
  a way that changes the declared resampling hierarchy.
- Do not silently ridge, floor, drop remainders, discard invalid replicates, or
  repair nonfinite/indefinite inputs.
- Do not add NumPy/PyTorch/JAX algorithmic implementation, disable XLA by
  default, describe CPU as the production target, or omit GPU provenance.
- Do not stage, commit, push, reset, restore, clean, install packages, fetch
  network resources, edit model files, change public/default policy, or write
  outside the reviewed boundary.
- Do not call a Codex substitute review Claude review or treat reviewer approval
  as authorization to cross a human/runtime/scientific boundary.

## Stop Conditions

Stop A3 immediately and write a blocker result if:

- A2 closure or terminal trace audit fails;
- a bound A1/A2 source/artifact hash drifts;
- the analytic LGSSM derivation or direct recurrence cannot be reconciled;
- required TFP quantile functionality is absent and no reviewed replacement is
  authorized;
- the chain/block/forecast-cluster hierarchy cannot be represented without a
  semantic change;
- any command writes outside the frozen A3 and `/tmp` boundaries;
- focused tests, fresh artifact replay, GPU/XLA placement, or evidence signature
  remain invalid after the bounded repair loop;
- review fails to converge within the authorized rounds;
- A3 would require an HMC, NeuTra, model-file, package-install, network, public
  API/default, product, or scientific-claim boundary; or
- a new ambiguity would materially change the estimand, decision semantics, or
  A4 calibration design.

Do not stop merely because a controlled candidate is underpowered when the
oracle and harness are valid. Record `INCONCLUSIVE_UNDERPOWERED`, identify the
repair trigger, and continue to the next discriminating A3 fixture or A4
calibration planning unless a true continuation veto fired.

## Exact A4 Handoff Conditions

A4 calibration subplan drafting is eligible only after:

1. the A3 analytic oracle, direct simulation, statistics, MMD, resampling,
   interval, and decision-logic focused tests pass;
2. the exact failure-to-reject-to-`PASS`, non-admissible-MMD-to-`PASS`,
   mechanics-only-to-`PASS`, and invalid-joint-alpha-to-`PASS` negative tests
   pass;
3. CPU-hidden reference and trusted GPU/XLA artifacts pass fresh-process replay
   and trace-boundary audits;
4. no hard veto or continuation veto remains open;
5. the A3 implementation receives bounded `VERDICT: AGREE`;
6. the A3 result contains required decision/inference tables and receives
   bounded `VERDICT: AGREE`;
7. the A4 draft binds the actual A3 interfaces and preserves validation,
   calibration, and confirmation separation;
8. the A4 draft explicitly owns all final margin, bandwidth, ridge, block,
   bootstrap, confidence, count, and seed choices and receives bounded
   `VERDICT: AGREE`; and
9. the A3 post-result closure and terminal read-only trace audit pass.

These conditions authorize A4 calibration planning only. A4 runtime begins
only under its own reviewed subplan and evidence contract. No A3 artifact,
fixture constant, or review verdict authorizes an SSL-LSTM sampler comparison,
predictive-equivalence claim, HMC/NeuTra run, product/default change, or
scientific conclusion.
