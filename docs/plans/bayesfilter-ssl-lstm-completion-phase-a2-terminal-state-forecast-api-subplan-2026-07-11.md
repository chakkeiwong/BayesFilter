# Phase A2 Subplan: Typed Terminal-State And Forecast API

Date: 2026-07-12

Status: `CONDITIONAL_EXECUTION_GATE_REQUIRES_HASH_BOUND_VERDICT_AGREE`

This file is executable only when a bounded review record names this exact path,
matches its then-current SHA-256, and ends `VERDICT: AGREE`. A `REVISE`, missing
record, or hash mismatch leaves A2 implementation unauthorized. This conditional
status avoids changing the reviewed bytes after agreement.

## Phase Objective

Implement a narrow TensorFlow `float64` API that maps one or a statically
batched set of accepted A1 free-parameter draws to:

1. a target-preserving predictive rerun of the A0-locked historical SVD-UKF;
2. the final filtered Gaussian over the complete augmented SSL-LSTM state;
3. deterministic, stateless ten-step posterior-predictive paths driven by an
   explicit standard-normal innovation bank; and
4. typed provenance sufficient to replay and audit the computation.

Phase A2 is an engineering-contract phase. It does not implement forecast
moments, MMD, bootstrap inference, HMC, NeuTra, calibration, equivalence
decisions, or a scientific comparison.

## Classification And Research Intent

The lower-level SVD-UKF rerun is classified as
`target_preserving_predictive_extraction`: it uses the identical full parameter
embedding, observations, model equations, SVD-UKF value route, and numerical
settings as the accepted A1 target, with `return_filtered=True` solely to expose
the already-computed filtering states. It is not a filter migration, new
target, exact nonlinear filter, or derivative-trace shortcut.

| Field | A2 contract |
| --- | --- |
| Main question | Can the accepted A1 scalar target produce replayable, typed, GPU/XLA-first ten-step predictive paths without changing its estimand or stochastic semantics? |
| Mechanism under test | Separate SVD-UKF value rerun, audited terminal PSD Gaussian, stateless terminal/process/observation innovations, and structural SSL-LSTM recursion |
| Expected failure mode | Target/value rerun mismatch; invalid terminal covariance; wrong process-noise placement; hidden/cell noise injection; observation-noise omission; scalar/batch drift; seed replay drift; eager/XLA or CPU/GPU drift; incomplete provenance |
| Promotion criterion | All reviewed A2 contract tests, CPU reference checks, and trusted GPU/XLA canary pass with complete structured provenance |
| Promotion veto | Target/filter/total-value parity failure; nonfinite or materially indefinite covariance; factor reconstruction failure; recursion/noise-placement failure; replay/batch/XLA/GPU failure; artifact/signature mismatch |
| Continuation veto | Broken A1 entry artifact; changed model/filter/target semantics; unavailable required trusted GPU/XLA route; invalid test oracle; corruption or unexplained mutation of an A2-owned artifact |
| Repair trigger | A localized implementation, test, assertion, serialization, or XLA-compatibility defect inside the reviewed A2 write set |
| Explanatory only | Runtime, trace counts, exact residuals beneath their gates, eigenvalue spectra after validity is established, and changed-seed path deltas |
| Must not be concluded | Posterior correctness, exact nonlinear filtering, parameter agreement, HMC/NeuTra readiness, predictive equivalence, model adequacy, calibration, statistical superiority, or product/default/release readiness |

Candidate rejection is separate from research-direction rejection. A failed A2
implementation is repaired inside its bounded write set unless the result
invalidates the accepted A1 target, model equations, required runtime, or
artifact chain. It cannot reject the forecast-moment validation idea by itself.

## Entry Conditions Inherited From A1

A2 planning may proceed only from all of these accepted facts:

- `HEAD` anchor: `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`;
- A1 result:
  `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md`;
- A1 result SHA-256:
  `78f269a53fb0536017d32bd12c2b36967cd013a85dcb1102936ed79ae95e34b5`;
- A1 result review:
  `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md` with `VERDICT: AGREE`;
- accepted production target SHA-256:
  `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667`;
- accepted A1 contract signature:
  `004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556`;
- target semantic signature:
  `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`;
- parameter-mask signature:
  `9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043`;
- accepted CPU signature:
  `c208b513e2fbf74d654b3b349695a7fcb811b2a6c36f5c2fa76a30dd5e9c922d`;
- accepted GPU/XLA signature:
  `077abbd5d5d8dc1068d99aba90fc8b6dd5b74001cda1dd1fe4428d13a0b4631c`;
- the complete reviewed A1 suite passed `75/75`; and
- Claude is policy-unavailable, so any bounded native Codex substitute review
  is explicitly weaker and cannot be represented as Claude review.

Before implementation, a verifier must recompute the A1 result, result-review,
production-target, lazy-export, and accepted structured-artifact hashes. A
protected mismatch is a stop, not a warning. The concurrent HMC/Kalman lane is
outside A2 ownership and must be preserved.

## Locked Predictive Law

| Element | Binding A2 definition |
| --- | --- |
| Historical data | Accepted A1 observations, shape `[30, 1]`, no missingness |
| Free parameter chart | Accepted four-coordinate A1 order |
| Full embedding | `SSLLSTMParameterMask.embed`; no alternate fixture or transform |
| Historical filter | `tf_svd_sigma_point_filter(..., backend="tf_svd_ukf", return_filtered=True)` |
| Filter construction | `make_ssl_lstm_svd_ukf_components` with accepted A1 model configuration |
| Numerical settings | `std_floor=1e-4`, UKF `alpha=1`, `beta=2`, `kappa=0`, placement floor `0`, innovation floor `1e-12`, rank tolerance `1e-12`, jitter `0` |
| Terminal law | Final historical filtered Gaussian for each parameter draw |
| Terminal state | Full augmented vector ordered `[z, a, c]`, scalar dimensions `[1, 1, 1]` |
| Forecast horizon | Exactly `10`; static, not runtime-variable |
| Transition | `ssl_lstm_transition` evaluated on the previous full state |
| Process noise | Add `process_std * epsilon_process` only to `z`; `a` and `c` remain deterministic completions |
| Observation | `ssl_lstm_observation(next_state) + observation_std * epsilon_observation` |
| Cluster unit | One complete ten-step path per `(draw, replication)` |
| Approximation statement | Predictive law conditional on the approximate historical SVD-UKF filter, not the exact nonlinear filtering law |
| Backend/default | TensorFlow `float64`; XLA JIT enabled by default; eager is debug/reference only |

The ordinary A1 `SSLLSTMPosteriorConfig.return_filtered=False` contract remains
unchanged. A2 must not mutate that field, read derivative trace dictionaries,
or replace the historical filter with the principal-square-root UKF, SGQF,
particle filtering, or another backend.

## Frozen A2 Numerical Design

Let `eps = 2^-52`. For finite scalar or tensor comparators `a` and `b`, define

`scale(a,b) = max(1, max(abs(a)), max(abs(b)))`

and `tol(m,a,b) = m * eps * scale(a,b)`. Every maximum is over all elements.
The exact A2 engineering tolerances are:

| Gate | Binding threshold |
| --- | --- |
| Rerun filter likelihood versus direct A1 filter likelihood | absolute residual `<= tol(64,a,b)` |
| Rerun likelihood-plus-prior versus accepted A1 target value | absolute residual `<= tol(64,a,b)` |
| Zero-bank direct recursion, process placement, observation timing | maximum absolute residual `<= tol(128,a,b)` |
| Scalar versus one-row/multirow batch | maximum absolute residual `<= tol(128,a,b)` |
| Eager debug versus CPU/GPU XLA | maximum absolute residual `<= tol(512,a,b)` |
| Persisted-bank CPU-XLA versus trusted GPU-XLA | maximum absolute residual `<= tol(4096,a,b)` |
| Same-runtime fixed-bank replay | raw bank hashes equal and output tensors bitwise equal |

These are conservative `float64` engineering roundoff guards, not fitted
scientific thresholds, equivalence margins, or statistical claims. A threshold
cannot be changed after its comparator has run without amending and rereviewing
this subplan.

The target-parity matrix is exactly the following row order. Values are the
accepted A1 `float64` constants; the shell step is exactly
`0x1.6666666666666p-4`.

| Row | Four free coordinates in hexadecimal `float64` |
| --- | --- |
| `truth_free` | `[0x1.6666666666666p-2, -0x1.47ae147ae147bp-4, 0x1.4cccccccccccdp-1, 0x1.999999999999ap-5]` |
| `phase2s_center` | `[0x1.2410a2e2543f1p-1, -0x1.fcd3132f8ba11p-4, 0x1.52631979a086cp-1, 0x1.1557ab4d560a3p-3]` |
| `shell_0_minus` | `[0x1.ee87ac2b0ee48p-2, -0x1.fcd3132f8ba11p-4, 0x1.52631979a086cp-1, 0x1.1557ab4d560a3p-3]` |
| `shell_0_plus` | `[0x1.50dd6faf210bep-1, -0x1.fcd3132f8ba11p-4, 0x1.52631979a086cp-1, 0x1.1557ab4d560a3p-3]` |
| `shell_1_minus` | `[0x1.2410a2e2543f1p-1, -0x1.b19cbccaf903cp-3, 0x1.52631979a086cp-1, 0x1.1557ab4d560a3p-3]` |
| `shell_1_plus` | `[0x1.2410a2e2543f1p-1, -0x1.2cd959924a756p-5, 0x1.52631979a086cp-1, 0x1.1557ab4d560a3p-3]` |
| `shell_2_minus` | `[0x1.2410a2e2543f1p-1, -0x1.fcd3132f8ba11p-4, 0x1.25964cacd3b9fp-1, 0x1.1557ab4d560a3p-3]` |
| `shell_2_plus` | `[0x1.2410a2e2543f1p-1, -0x1.fcd3132f8ba11p-4, 0x1.7f2fe6466d539p-1, 0x1.1557ab4d560a3p-3]` |
| `shell_3_minus` | `[0x1.2410a2e2543f1p-1, -0x1.fcd3132f8ba11p-4, 0x1.52631979a086cp-1, 0x1.8891e0688b5c0p-5]` |
| `shell_3_plus` | `[0x1.2410a2e2543f1p-1, -0x1.fcd3132f8ba11p-4, 0x1.52631979a086cp-1, 0x1.c88ade80893d6p-3]` |

The row-major little-endian `float64` byte hash of this `[10,4]` matrix is
`d6ba48e5a64897f87caeece4de776c139d8fc62d00fc118d89b4d88da468829a`.
All ten rows receive terminal extraction and both target-parity checks. Exact
forecast-path artifact coverage uses only rows `truth_free` and
`phase2s_center`, in that order, with exactly two replications. The count is a
convenience engineering fixture, not a statistical sample size.

The A2 contract signature is SHA-256 over canonical JSON of exactly this
payload:

```json
{"a1_adapter_signature":"004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556","classification":"target_preserving_predictive_extraction","covariance":{"factor":"symmetric_principal_eigen_square_root","factor_multiplier":16,"material_negative_policy":"reject_below_negative_tau","projection_multiplier":8,"roundoff_multiplier":64},"dimensions":{"free":4,"full":24,"hidden":1,"latent":1,"observation":1,"state":3},"filter":{"alpha_hex":"0x1.0000000000000p+0","backend":"tf_svd_ukf","beta_hex":"0x1.0000000000000p+1","innovation_floor_hex":"0x1.19799812dea11p-40","jitter_hex":"0x0.0p+0","kappa_hex":"0x0.0p+0","placement_floor_hex":"0x0.0p+0","rank_tolerance_hex":"0x1.19799812dea11p-40","return_filtered":true,"std_floor_hex":"0x1.a36e2eb1c432dp-14"},"forecast_horizon":10,"innovation":{"algorithm":"philox","evidence_root_seed":[20260712,1202],"family_codes":{"observation":1003,"process":1002,"terminal":1001},"observation_shape_suffix":[10,1],"process_shape_suffix":[10,1],"role_codes":{"independent_arm":211,"paired_diagnostic_shared":101},"terminal_shape_suffix":[3]},"observation_raw_sha256":"aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98","parameter_mask_sha256":"9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043","schema_version":"bayesfilter.ssl_lstm_completion.phase_a2_contract.v1","target_semantic_sha256":"549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e","tolerance_multipliers":{"batch":128,"cpu_gpu":4096,"eager_xla":512,"filter":64,"recursion":128,"total":64}}
```

Canonicalization is the artifact rule below. The expected signature is
`8719aa65943dcc9e4b0499debfff8ec13a96d4cec12dc48d70a8922920058804`.

The exact two-replication XLA evidence `SSLLSTMForecastConfig` signature is
SHA-256 over canonical JSON of exactly:

```json
{"a1_posterior_config_signature":"004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556","a2_contract_signature":"8719aa65943dcc9e4b0499debfff8ec13a96d4cec12dc48d70a8922920058804","allowed_innovation_roles":["paired_diagnostic_shared","independent_arm"],"backend":"tensorflow","covariance_roundoff_multiplier":64,"dtype":"float64","execution_role":"default_xla","forecast_horizon":10,"jit_compile":true,"latent_dim":1,"observation_dim":1,"replication_count":2,"schema_version":"bayesfilter.ssl_lstm_completion.phase_a2_forecast_config.v1","state_dim":3}
```

Its expected signature is
`ecb5a2cedac5f059da3bd3feee51a1065eb66aeff5aeb8dc0dd3b4e3a6926150`.
Other positive static replication counts are allowed for API tests and derive
their own canonical config signature, but cannot be serialized as A2 runtime
evidence.

## Proposed API Contract

Add `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py` with these frozen types:

### `SSLLSTMForecastConfig`

Required fields and fixed/default semantics:

- accepted `SSLLSTMPosteriorConfig` instance;
- `forecast_horizon=10`, immutable for A2;
- positive static `replication_count` supplied by the caller;
- `jit_compile=True` by default;
- execution role `default_xla` when compiled and
  `eager_debug_reference` only when noncompiled;
- covariance absolute tolerance `64 * eps(float64)` and relative tolerance
  `64 * eps(float64)`, classified as derived roundoff guards rather than
  scientific thresholds; and
- schema/signature fields binding the accepted target, parameter mask,
  observations, filter settings, horizon, dimensions, backend, dtype, and
  innovation semantics.

The covariance tolerance for a symmetrized matrix `C` is
`tau = 64 * eps * max(1, ||C||_F)`. The constant `64` is a conservative
engineering roundoff multiplier and must be recorded in provenance. It is not
a fitted or scientific tolerance.

### `SSLLSTMTerminalState`

Required tensor fields for one draw:

- `mean`: `[3]`;
- `raw_covariance`: `[3, 3]` exactly as emitted by the value filter;
- `symmetrized_covariance`: `[3, 3]` after explicit symmetry validation;
- `implemented_covariance`: `[3, 3]` after permitted roundoff clipping only;
- `factor`: `[3, 3]`, deterministic symmetric principal eigen-square-root;
- `raw_eigenvalues`: `[3]`;
- `clipped_eigenvalues`: `[3]`;
- scalar `minimum_eigenvalue`, `psd_tolerance`, `symmetry_residual`,
  `projection_residual`, and `factor_reconstruction_residual`;
- scalar rerun `filter_log_likelihood`, A1 `target_value`, reconstructed
  `total_value`, and both parity residuals; and
- the full embedded parameter vector `[24]`.

Batch extraction stacks every tensor with leading `[draw]`; it must not hide
draw-specific invalidity. The compiled numerical core must return per-draw
integer status codes plus finite diagnostic tensors for every target-parity and
covariance gate. The host-side evidence boundary must reject any nonzero status
or failed diagnostic before it constructs production provenance or serializes
a passing artifact. XLA-side debug assertions may aid localization, but they
are not the sole fail-closed authority.

### `SSLLSTMInnovationBank`

The bank contains standard-normal `float64` tensors, never already-scaled
model noise:

- `terminal_standard_normal`: `[draw, replication, 3]`;
- `process_standard_normal`: `[draw, replication, 10, 1]`;
- `observation_standard_normal`: `[draw, replication, 10, 1]`;
- a stateless TensorFlow seed of shape `[2]`, generation algorithm identity,
  role, shape metadata, and stable content signature.

Generation must use TensorFlow stateless Philox explicitly. The API includes an
integer `arm_id`. Starting from an `int32[2]` root seed, derivation is exactly:

1. `role_seed = stateless_fold_in(root_seed, role_code, alg="philox")`;
2. `arm_seed = stateless_fold_in(role_seed, arm_id, alg="philox")`;
3. `family_seed = stateless_fold_in(arm_seed, family_code, alg="philox")`; and
4. `tf.random.stateless_normal(..., seed=family_seed, alg="philox")`.

The frozen integer codes are:

| Meaning | Code |
| --- | ---: |
| `paired_diagnostic_shared` | `101` |
| `independent_arm` | `211` |
| terminal family | `1001` |
| process family | `1002` |
| observation family | `1003` |

`paired_diagnostic_shared` requires `arm_id=0`. `independent_arm` requires a
strictly positive `int32` arm id; A2 tests ids `1` and `2` under the same root
seed and requires distinct family hashes. The persisted A2 CPU/GPU evidence
bank uses root seed `[20260712,1202]`, paired role, and arm id `0`.

The bank-generation record binds TensorFlow version, algorithm, root seed,
role/arm/family codes, derived family seeds, shapes, and raw row-major
little-endian `float64` tensor content hashes. Forecast execution accepts
already materialized bank tensors and must not regenerate them inside the
compiled recursion. The CPU producer persists the exact bank; the GPU consumer
loads it, and both verify the same content hashes before comparing paths.

Allowed roles are:

- `paired_diagnostic_shared`: the same bank may be replayed for two arms to
  reduce Monte Carlo noise in explanatory paired mean/log-variance checks; and
- `independent_arm`: separately seeded arm-specific banks required by the
  later primary MMD and robustness design.

A2 tests generation and replay for both roles but makes no inferential choice.
A4, not A2, freezes confirmatory seeds, replication counts, feature scales,
moment margins, MMD settings, bootstrap settings, and the exact shared versus
independent usage. Shared innovations must not silently become the primary MMD
evidence.

### `SSLLSTMForecastPaths`

Required stacked tensors:

- `terminal_states`: `[draw, replication, 3]`;
- `states`: `[draw, replication, 10, 3]`;
- `deterministic_transition_means`: `[draw, replication, 10, 3]`;
- `process_innovations`: `[draw, replication, 10, 1]` after model scaling;
- `observation_means`: `[draw, replication, 10, 1]`;
- `observation_innovations`: `[draw, replication, 10, 1]` after model scaling;
- `observations`: `[draw, replication, 10, 1]`; and
- matching `SSLLSTMForecastProvenance`.

`states[..., t, :]` is the state after the transition and process innovation at
forecast horizon `t + 1`; the observation at that index is emitted from that
same state. Terminal historical states are stored separately and are not
duplicated as horizon zero.

### `SSLLSTMForecastProvenance`

Required fields bind:

- schema version and A2 contract signature;
- A0/A1 target, adapter, mask, observation, fixture, prior, and accepted-result
  signatures;
- forecast config and innovation-bank signatures;
- parameter-draw raw-content hash and embedded full-parameter hash;
- filter backend and exact numerical settings;
- terminal covariance diagnostics for every draw;
- TensorFlow version, dtype, device type/name, visible GPU count, XLA/JIT and
  TF32 settings, execution role, and managed-session GPU trust basis when used;
- innovation role, stateless seed, static shapes, and horizon convention;
- approximate-filter qualification and all A2 nonclaims.

Production evidence must reject testing-only A1 targets and incomplete or
self-inconsistent provenance. Provenance construction and signature hashing are
host-side reporting operations after compiled tensors and status codes return;
they are not inserted into the XLA numerical graph.

### Callable Surfaces

The module must expose narrow scalar and batch surfaces, provisionally:

- `make_ssl_lstm_innovation_bank(config, draw_count, seed, role, arm_id)`;
- `extract_ssl_lstm_terminal_state(free_draw, config)`;
- `extract_ssl_lstm_terminal_states(free_draws, config)`;
- `forecast_ssl_lstm_paths(free_draws, innovation_bank, config)`; and
- eager debug-reference counterparts where needed for parity tests.

The canonical forecast input is rank two `[draw, 4]`, including a one-row
batch. A scalar convenience surface must produce exactly the squeezed form of
the one-row batch result. Draw and replication dimensions must be statically
known and positive at tracing time. Wrong dtype, rank, static dimension,
horizon, role, seed shape, or bank shape must fail loudly before evidence can be
serialized. Host-visible validation must establish that every compiled status
code is valid; merely returning a tensor is not evidence admission.

## Terminal Covariance Policy

The historical SVD-UKF terminal covariance is permitted to be positive
semidefinite and need not be strictly positive definite. A2 must not apply an
ordinary Cholesky requirement or silently floor an invalid covariance.

For each draw:

1. require all mean and covariance entries finite;
2. compute and record the maximum absolute antisymmetric residual before
   symmetrizing;
3. reject if the symmetry residual exceeds `tau`;
4. run `tf.linalg.eigh` on the symmetrized covariance;
5. reject if any eigenvalue is below `-tau`;
6. clip only eigenvalues in `[-tau, 0)` to exactly zero;
7. define `C_impl = Q diag(clipped_eigenvalues) Q^T` independently;
8. form the symmetric principal eigen-square-root
   `F = Q diag(sqrt(clipped_eigenvalues)) Q^T`;
9. record the projection residual
   `||C_impl - C_sym||_F`;
10. require projection residual to be finite and no greater than `8 * tau`;
11. require the independently defined factor reconstruction residual
   `||F F^T - C_impl||_F` to be finite and no greater than `16 * tau`; and
12. use `mean + terminal_standard_normal @ F^T` for terminal draws.

The symmetric factor is invariant to eigenvector sign changes and to rotations
inside an exactly degenerate eigenspace. This avoids making fixed-bank replay
depend on an arbitrary eigenvector orientation. Cross-runtime equality remains
subject to recorded `float64` tolerance because eigensolvers and arithmetic may
differ slightly. A materially negative eigenvalue, nonfinite diagnostic,
excessive asymmetry, or factor reconstruction failure is a hard veto. No larger
floor, nearest-SPD repair, jitter, or Cholesky fallback is allowed in A2.

## Target-Parity Admission Gate

For every free draw, before exposing a terminal state:

1. embed it with the accepted A1 mask;
2. construct the historical components with
   `make_ssl_lstm_svd_ukf_components` and accepted `std_floor`;
3. call `tf_svd_sigma_point_filter` with `backend="tf_svd_ukf"`,
   `return_filtered=True`, placement floor `0`, innovation floor `1e-12`, rank
   tolerance `1e-12`, and jitter `0`;
4. independently evaluate the accepted A1 analytic SVD-UKF score route at the
   same full draw and retain its direct `log_likelihood` as the filter-level
   comparator;
5. independently evaluate `SSLLSTMPosteriorTarget.value` at the same free draw,
   reconstruct the exact A1 unnormalized prior kernel, and add that prior to
   the rerun filter likelihood;
6. require finite filter output, filtered history shapes `[30, 3]` and
   `[30, 3, 3]`, and final-state diagnostics; and
7. require both the rerun filter likelihood parity and total-value parity under
   the frozen `tol(64,a,b)` formula.

The filter-likelihood comparator is the direct `log_likelihood` returned by
`tf_ssl_lstm_svd_ukf_score` under the exact accepted A1 arguments. The
total-value comparator is `SSLLSTMPosteriorTarget.value`. These checks are
separate: the implementation must not derive the filter comparator by
subtracting its locally reconstructed prior from the total comparator. A
mismatch in either check blocks terminal-state admission. The repeated A1
evaluation is deliberate correctness evidence at A2 scale, not a performance
design or a license to change A1.

## Forecast Recursion Contract

For each draw and replication:

1. sample the full terminal state from its accepted principal eigen-factor;
2. evaluate `ssl_lstm_transition(parameters, previous_state)`;
3. scale the one-dimensional process standard normal by the constrained
   `process_std` and add it only to the first, latent coordinate;
4. copy hidden/cell coordinates exactly from the deterministic transition;
5. evaluate `ssl_lstm_observation(parameters, next_state)`;
6. scale observation standard normal by constrained `observation_std` and add
   it to the emission mean; and
7. retain all intermediate tensors without recomputing noise from a hidden RNG.

All randomness is externalized in the bank. Forecast execution must not call a
stateful random-number generator. Zero banks must give direct deterministic
recursion with maximum absolute residual no greater than the frozen
`tol(128,a,b)` threshold.

## Required Artifacts And Literal Write Set

Only the following literal paths may be created or modified in A2:

### Source, Test, Harness, And Verifier

- `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py`;
- `bayesfilter/nonlinear/__init__.py`;
- `tests/test_ssl_lstm_predictive_tf.py`;
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py`;
- `docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py`.

The `bayesfilter/nonlinear/__init__.py` exception permits only additive
`SSLLSTMTerminalState`, `SSLLSTMForecastConfig`, `SSLLSTMInnovationBank`,
`SSLLSTMForecastPaths`, and `SSLLSTMForecastProvenance` entries in `__all__`
and `_EXPORT_MODULES`, all pointing to
`bayesfilter.nonlinear.ssl_lstm_predictive_tf`. Every pre-A2 byte and ordering
outside those insertions must remain unchanged. This is the sole permitted
change to an existing A1-owned source file and resolves the apparent conflict
with the general A1 preservation rule.

### Structured Artifacts And Logs

- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/pre-run-boundary.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/boundary-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/executor-write-ledger.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/executor-ledger-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-write-ledger.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-ledger-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/focused-tests-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-verification-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference-verify.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-verification-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary-verify.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/final-checkpoint.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/final-checkpoint-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/closure-generation-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-closure.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/closure-verification-write-trace.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-closure-verify.log`.

### Plan, Result, Reviews, And Handoff

- this subplan;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-round1-2026-07-12.md` through
  `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-round6-2026-07-12.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-implementation-codex-substitute-review-2026-07-12.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-result-codex-substitute-review-2026-07-12.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a3-subplan-codex-substitute-review-2026-07-12.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md`; and
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md` only after A2 result acceptance.

No wildcard, sibling file, or whole-directory allowance is implied. Existing A1
source, tests, harnesses, and structured artifacts remain immutable except for
the exact additive lazy-export exception above. No HMC/Kalman-lane file may be
edited, staged, reset, restored, or cleaned. No commit or push is authorized.

After source/test/harness/verifier edits but before the first TensorFlow or
pytest runtime, `pre-run-boundary.json` records strict sorted rows for:

- every literal A2 path already present, with existence, tracked/untracked
  state, and SHA-256;
- every accepted A1 protected path/hash and accepted A1 artifact/hash; and
- all other current dirty paths as explanatory outside-write-set inventory.

Every A2 subprocess except the one frozen terminal read-only audit command
records opening and closing scoped snapshots and is wrapped by
`/usr/bin/strace -f -qq -e trace=%file -o <literal-write-trace>`. The terminal
audit exemption is limited to the exact command under **Boundary, Ledger, And
Checkpoint Commands**: it opens existing inputs read-only, does not initialize
TensorFlow, writes canonical JSON to stdout only, and creates no repository or
filesystem artifact. Any argument, mode, path, or behavior change removes the
exemption and requires plan amendment plus rereview. The
independent verifier parses each complete trace. Any successful or attempted
mutating file operation by an A2 process against a repository path outside the
literal A2 set is a veto; mutating operations include an `open/openat/creat`
with write, create, append, or truncate flags plus rename, link, symlink,
unlink, mkdir, rmdir, chmod, chown, truncate, or utime families. Repository-
relative paths are resolved against the traced process working directory.
Writes below `/tmp/bayesfilter-a2-pycache` or `/tmp/bayesfilter-a2-tmp` are
allowed and recorded; no other filesystem location is an A2 write target.

Direct file edits use `apply_patch` only. `executor-write-ledger.json` lists
every such edit through the draft A2 result as exact
`{sequence:int,path:str,before_sha256:str_or_null,after_sha256:str,reason:str}`
rows and becomes immutable before the pre-result checkpoint.
`post-result-write-ledger.json` separately records later direct edits to the A2
result/review records, A3 subplan/review, and governance handoff files. Each
ledger is checked against the literal set and the applicable final filesystem
hashes. The executor may not classify its own traced or ledgered write as
concurrent.

Boundary, ledger, and checkpoint files are generated by the independent
verifier subprocess modes below, never by direct edit. Each generated object
excludes its own creation and its still-open generation trace from its internal
rows. The next immutable stage binds both the generated object and the now-
closed generation trace. Thus:

1. executor ledger binds boundary JSON plus closed boundary-generation trace;
2. final checkpoint binds executor ledger plus closed ledger-generation trace;
3. post-result closure binds final checkpoint plus closed checkpoint-generation
   trace and post-result ledger plus closed post-result-ledger-generation trace;
4. closure verifier parses the closed closure-generation trace; and
5. the supervisor runs one final read-only terminal audit of the closed
   closure-verification trace.

The terminal audit emits canonical JSON to stdout only, performs no artifact or
repository write, and is the finite root of the write-trace chain. It must emit
`{"status":"A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED","trace_sha256":"<sha256>"}`.

A path outside the literal A2 and protected A1 sets that appears, disappears,
or changes but is absent from every A2 strace and executor-ledger row is
classified as explanatory concurrent-lane drift and is not a veto, whether or
not it existed at the opening snapshot. This exception follows the user's
explicit concurrent-lane instruction; it does not authorize A2 to edit that
path. Mutation of a protected A1 path, an unexplained write to a literal A2
path, or a repo-local cache/temp file attributable to A2 is a veto.

Every Python command uses
`PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache`, and
`TMPDIR=/tmp/bayesfilter-a2-tmp`. Every TensorFlow command additionally uses
`CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache` and
`XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla"`.
The harness/verifier records these exact values before TensorFlow import and
rejects `$HOME/.nv/ComputeCache` or any other non-`/tmp` CUDA/XLA cache write in
the strace. Every pytest command also uses
`-p no:cacheprovider`. Static compilation uses Python's in-memory `compile`
builtin, not `py_compile`. Before and after each command, scans reject
`__pycache__`, `*.pyc`, and `.pytest_cache` attributable to A2 anywhere in the
repository. The final checkpoint repeats the temporal classification and
verifies the additive-export diff structurally.

## Artifact Schemas And Verification

The strict schemas are fixed as:

| Path role | Schema |
| --- | --- |
| Pre-run boundary | `bayesfilter.ssl_lstm_completion.phase_a2_scoped_boundary.v1` |
| Executor write ledger | `bayesfilter.ssl_lstm_completion.phase_a2_executor_write_ledger.v1` |
| Post-result write ledger | `bayesfilter.ssl_lstm_completion.phase_a2_post_result_write_ledger.v1` |
| Innovation bank | `bayesfilter.ssl_lstm_completion.phase_a2_innovation_bank.v1` |
| CPU reference | `bayesfilter.ssl_lstm_completion.phase_a2_cpu_reference.v1` |
| GPU/XLA canary | `bayesfilter.ssl_lstm_completion.phase_a2_gpu_xla_canary.v1` |
| Final checkpoint | `bayesfilter.ssl_lstm_completion.phase_a2_final_checkpoint.v1` |
| Post-result closure | `bayesfilter.ssl_lstm_completion.phase_a2_post_result_closure.v1` |

The following exact schema notation is binding: `str`, `bool`, `int`, and
`finite_float` are JSON scalar types; `null` is JSON null; `X_or_null` is either
the stated type or null; `[T]` is an ordered JSON array; `{...}` is an object
with exactly the listed keys and no extras. Unannotated field names in the
top-level payload lists receive their types from the immediately following
sentences and shared rows; no additional field is permitted. All path arrays
are lexicographically sorted. All SHA-256 values are lowercase 64-character
hex.

Across every payload, `schema_version`, `status`, `created_at_utc`,
`git_commit`, `artifact_role`, `tensorflow_version`, `algorithm`, `role`, and
`trust_basis` are strings when present; `evidence_signature` is a SHA-256
string; `nonclaims` is the exact `[str]` array below. RFC 3339 UTC timestamps
ending `+00:00` are required. A `file_row.sha256` is null iff `exists=false`;
otherwise it is a SHA-256 string. A `binding_row.evidence_signature` is null
only for a non-JSON artifact with no signed payload.

### Shared Exact Rows

`file_row` is exactly
`{path:str, exists:bool, tracked:bool, sha256:str_or_null, role:str}`.
`binding_row` is exactly `{path:str, file_sha256:str,
evidence_signature:str_or_null, role:str}`. `device_row` is exactly
`{name:str, device_type:str}`. `tensor_row` is exactly
`{name:str, dtype:"float64", shape:[int], values_hex:[str],
raw_little_endian_sha256:str}`; `values_hex` is flattened row-major and every
entry is a Python-compatible finite hexadecimal float string. `check_row` is
exactly `{name:str, role:str, passed:bool, residual:finite_float_or_null,
threshold:finite_float_or_null}`.

`run_manifest` is exactly
`{git_commit:str, git_dirty:bool, command:str, cwd:str, interpreter:str,
conda_env:str, python_version:str, packages:{tensorflow:str,
tensorflow_probability:str}, environment:{CUDA_VISIBLE_DEVICES:str_or_null,
PYTHONDONTWRITEBYTECODE:"1", PYTHONPYCACHEPREFIX:str, TMPDIR:str,
CUDA_CACHE_PATH:str,XLA_FLAGS:str},
physical_devices:[device_row], logical_devices:[device_row],
tf32_enabled:bool, jit_compile:true, dtype:"float64", random_seeds:[int],
started_at_utc:str, completed_at_utc:str, wall_time_seconds:finite_float,
output_paths:[str], plan_path:str, result_path:str, trust_basis:str}`.

`compiler_row` is exactly
`{callable_name:str, static_input_shapes:[[int]], hlo_sha256:str,
hlo_byte_count:int, hlo_text:str, hlo_entry_present:bool, concrete_trace_count:int,
output_devices:[str]}`. Every byte count is positive, every entry flag is true,
every trace count is exactly one, and device arrays are nonempty. `hlo_text` is
the exact nonempty compiler-IR string; its UTF-8 byte count and SHA-256 must
equal the sibling fields and it must contain an `ENTRY` computation.

`covariance_row` is exactly
`{name:str,minimum_eigenvalue_hex:str,psd_tolerance_hex:str,
symmetry_residual:finite_float,projection_residual:finite_float,
factor_reconstruction_residual:finite_float,status:int}`.
`provenance` is exactly
`{schema_version:"bayesfilter.ssl_lstm_completion.phase_a2_provenance.v1",
a0_target_semantic_sha256:str,a1_adapter_signature:str,
a1_parameter_mask_sha256:str,a1_observation_raw_sha256:str,
a1_full_fixture_raw_sha256:str,a1_prior_center_raw_sha256:str,
a1_result_file_sha256:str,a2_subplan_file_sha256:str,
a2_contract_signature:str,forecast_config_signature:str,
innovation_bank_file_sha256:str,innovation_bank_evidence_signature:str,
free_draw_matrix_raw_sha256:str,embedded_full_parameter_matrix_raw_sha256:str,
filter:{backend:"tf_svd_ukf",std_floor_hex:str,alpha_hex:str,beta_hex:str,
kappa_hex:str,placement_floor_hex:str,innovation_floor_hex:str,
rank_tolerance_hex:str,jitter_hex:str,return_filtered:true},
terminal_covariances:[covariance_row],runtime:{tensorflow_version:str,
tensorflow_probability_version:str,dtype:"float64",jit_compile:true,
tf32_enabled:bool,execution_role:str,physical_devices:[device_row],
logical_devices:[device_row],trust_basis:str,compiler_evidence:[compiler_row]},
innovations:{algorithm:"philox",role:str,role_code:int,arm_id:int,
root_seed:[int],family_codes:{terminal:1001,process:1002,observation:1003},
tensor_hashes:{terminal:str,process:str,observation:str},horizon:10,
draw_count:2,replication_count:2},
horizon_convention:"state_and_observation_after_transition_t_plus_1",
cluster_unit:"complete_ten_step_path_per_draw_replication",
approximation_qualification:"conditional_on_approximate_historical_svd_ukf_not_exact_nonlinear_filter",
nonclaims:[str]}`. The provenance nonclaim array equals the exact phase
nonclaim array below. All hashes and hexadecimal filter values must match the
accepted/frozen constants and current signed files.

For CPU runtime artifacts, provenance/runtime `execution_role` is exactly
`cpu_hidden_xla_reference` and both manifest/provenance trust basis are exactly
`cpu_hidden_reference_exception_not_gpu_evidence`. For GPU runtime artifacts,
`execution_role` is exactly `trusted_gpu_xla_canary` and both trust basis values
are exactly `owner_designated_managed_session_visible_gpu_trusted`.

The nonclaim array is exactly, in this order:

1. `A2 terminal-state and forecast engineering evidence only`;
2. `predictive law is conditional on the approximate historical SVD-UKF`;
3. `not posterior correctness or exact nonlinear filtering evidence`;
4. `not HMC or NeuTra readiness evidence`;
5. `not predictive equivalence, calibration, or model adequacy evidence`;
6. `not performance, product, public API, default, or release evidence`; and
7. `not a sampler ranking or scientific claim`.

### Boundary Payload

`pre-run-boundary.json` has exactly
`{schema_version, status, created_at_utc, git_commit, accepted_a1_bindings,
literal_a2_rows, outside_dirty_rows, cache_scan, evidence_signature,
nonclaims}`. Status is exactly `A2_SCOPED_BOUNDARY_FROZEN`.
`accepted_a1_bindings`, `literal_a2_rows`, and `outside_dirty_rows` are
`[file_row]`. `cache_scan` is exactly
`{opening_cache_paths:[str], closing_cache_paths:[str],
a2_named_cache_paths:[str], suppression_environment_verified:bool,
passed:bool}`. Pre-existing caches are preserved and may appear in both opening
and closing arrays; they are not deleted or claimed by A2. Cache paths outside
A2 that appear, disappear, or change concurrently are classified under the
temporal outside-lane rule. `a2_named_cache_paths` must be empty,
`suppression_environment_verified=true`, and `passed=true`. An A2-named cache
is any repository path containing `ssl_lstm_predictive_tf`,
`test_ssl_lstm_predictive_tf`,
`benchmark_ssl_lstm_completion_phase_a2_terminal_forecast`, or
`verify_ssl_lstm_completion_phase_a2_terminal_forecast` and ending `.pyc` or
located below `__pycache__`/`.pytest_cache`.

`executor-write-ledger.json` has exactly
`{schema_version,status,created_at_utc,rows,strace_bindings,
evidence_signature,nonclaims}`. Status is exactly
`A2_EXECUTOR_WRITE_LEDGER_VALID`; `rows` is the ordered direct-edit row array
defined in the write-attribution section through the draft A2 result;
`strace_bindings` is `[binding_row]` for exactly the closed boundary-generation,
focused-test, CPU generation/verification, and GPU generation/verification
traces ordered by execution. The executor-ledger generation trace is excluded
because it is still open and is bound by the final checkpoint. Every row/path
is inside the literal A2 set.

`post-result-write-ledger.json` has exactly
`{schema_version,status,created_at_utc,rows,evidence_signature,nonclaims}`.
Status is exactly `A2_POST_RESULT_WRITE_LEDGER_VALID`; `rows` is the ordered
direct-edit row array after the pre-result checkpoint and before closure
generation. It must include the final A2 result/review, A3 subplan/review, and
each refreshed governance file, with no path outside the literal set. Its own
generation trace is excluded while open and is bound by the post-result
closure.

### Innovation-Bank Payload

`innovation-bank.json` has exactly
`{schema_version, status, created_at_utc, tensorflow_version, algorithm,
root_seed, role, role_code, arm_id, family_codes, derived_seeds, draw_count,
replication_count, horizon, tensors, evidence_signature, nonclaims}`. Status is
exactly `A2_INNOVATION_BANK_FROZEN`; algorithm is exactly `philox`;
`root_seed` and every value in `derived_seeds` are exactly two signed JSON
integers in `int32` range. `family_codes` is exactly
`{terminal:1001, process:1002, observation:1003}`. `derived_seeds` is exactly
`{role:[int,int], arm:[int,int], terminal:[int,int], process:[int,int],
observation:[int,int]}`. `tensors` is exactly three `tensor_row` entries ordered
terminal, process, observation, with the frozen shapes. The remaining numeric
and role fields must equal the frozen A2 design.

### CPU And GPU Payloads

Both runtime artifacts have exactly
`{schema_version, artifact_role, status, created_at_utc, run_manifest,
entry_bindings, source_files, frozen_design, bank_binding,
cpu_reference_binding, terminal_results, forecast_tensors, compiler_evidence,
provenance, contract_checks, evidence_signature, nonclaims}`. `provenance` has
the exact shared schema above and must agree field-for-field with the other
artifact sections and current source/signature bindings.

CPU `artifact_role`/`status` are exactly
`phase_a2_cpu_hidden_reference`/`CPU_REFERENCE_CONTRACT_PASSED`; GPU values are
exactly `phase_a2_trusted_gpu_xla_canary`/`GPU_XLA_CANARY_PASSED`.
`entry_bindings` is `[binding_row]` for the accepted A1 result/review, target,
lazy exports, accepted CPU/GPU artifacts, subplan, and agreed subplan review.
`source_files` is `[file_row]` for the production module, lazy exports, focused
test, harness, and verifier. `frozen_design` is exactly
`{point_names:[str], points:tensor_row, point_matrix_sha256:str,
forecast_point_names:[str], replication_count:2, horizon:10,
tolerance_multipliers:{filter:64,total:64,recursion:128,batch:128,
eager_xla:512,cpu_gpu:4096}}`.

`bank_binding` is exactly
`{path:str,file_sha256:str,evidence_signature:str,role:str,
tensor_hashes:{terminal:str,process:str,observation:str}}`. CPU
`cpu_reference_binding` is JSON null. GPU `cpu_reference_binding` is a
`binding_row` for the exact accepted CPU artifact.

Each `terminal_results` row is exactly
`{name:str,status:int,filter_log_likelihood_hex:str,
a1_filter_log_likelihood_hex:str,filter_residual:finite_float,
filter_threshold:finite_float,total_value_hex:str,a1_target_value_hex:str,
total_residual:finite_float,total_threshold:finite_float,
minimum_eigenvalue_hex:str,psd_tolerance_hex:str,
symmetry_residual:finite_float,projection_residual:finite_float,
factor_reconstruction_residual:finite_float,passed:bool}`. There are exactly
ten rows in frozen order, every status is zero, and every `passed` is true.
The corresponding provenance covariance rows contain the same diagnostic
values and statuses in the same order.

`forecast_tensors` is exactly seven `tensor_row` entries ordered terminal
states, states, deterministic transition means, process innovations,
observation means, observation innovations, and observations.
`compiler_evidence` is a nonempty `[compiler_row]` ordered by callable name.
`contract_checks` is `[check_row]`, lexicographically ordered by name, and must
contain exactly these names:
`a1_entry_hashes`, `bank_hashes`, `batch_parity`, `compiler_hlo`,
`covariance_validity`, `device_placement`, `eager_xla_parity`,
`filter_parity`, `forecast_replay`, `no_cache_writes`,
`observation_timing`, `process_noise_placement`, `status_admission`,
`total_target_parity`, and `write_boundary`. GPU adds exactly
`cpu_gpu_parity` and `cpu_reference_crosslink`. Every check passes.

### Checkpoint And Closure Payloads

`final-checkpoint.json` has exactly
`{schema_version, status, created_at_utc, git_commit, member_rows,
accepted_a1_bindings, outside_dirty_rows, cache_scan, evidence_signature,
nonclaims}`. Status is exactly `A2_PRE_RESULT_CHECKPOINT_PASSED`.
`member_rows` is `[binding_row]` for exactly these existing paths, never the
checkpoint itself or a future file: subplan; subplan reviews R1 through R6;
production module; lazy exports; focused test; harness; verifier; pre-run
boundary; closed boundary-generation trace; executor write ledger; closed
executor-ledger-generation trace; focused-test/CPU/GPU write-trace logs;
innovation-bank JSON/log; CPU JSON/log/verify log; GPU JSON/log/verify log;
implementation review; and the draft A2 result. Mutable
execution/approval/runbook/stop-handoff documents are intentionally excluded
because they are refreshed after result review and are bound only by the
post-result closure. Round 6 must agree; otherwise this exact membership is
amended and rereviewed before any later exceptional round or checkpoint.

`post-result-closure.json` has exactly
`{schema_version, status, created_at_utc, git_commit, member_rows,
final_checkpoint_binding, accepted_a1_bindings, outside_dirty_rows, cache_scan,
evidence_signature, nonclaims}`. Status is exactly
`A2_POST_RESULT_CLOSURE_PASSED`. `member_rows` is `[binding_row]` for exactly:
the final checkpoint; closed final-checkpoint-generation trace; final A2 result;
A2 result review; A3 subplan; A3 subplan review; post-result write ledger;
closed post-result-ledger-generation trace; execution ledger; approval ledger;
runbook; and stop handoff.
`final_checkpoint_binding` is the exact `binding_row` for the unchanged
checkpoint. The closure never hashes itself or its later verify log.

Every JSON parser rejects duplicate keys and nonfinite constants. Canonical
JSON uses UTF-8, sorted keys, separators `(',', ':')`, `ensure_ascii=true`, and
`allow_nan=false`. Each bank/artifact has an `evidence_signature`: SHA-256 of
its complete canonical payload after deleting only `evidence_signature`,
`created_at_utc`, and the run-manifest timing fields `started_at_utc`,
`completed_at_utc`, and `wall_time_seconds`. Paths, commands, devices, source
hashes, bank hashes, statuses, residuals, thresholds, and nonclaims remain in
the signed projection.

The independent verifier script must not import the generator harness. In a
fresh process it strictly validates schemas, signatures, source/protected
hashes, bank tensor hashes, CPU/GPU crosslinks, statuses, residual formulas,
threshold decisions, provenance, and output tensors by calling only production
A1/A2 APIs on the frozen inputs and loaded bank. The GPU verifier must execute
in the same trusted GPU context and independently replay the canary.

Concrete XLA evidence is not `jit_compile=true` metadata alone. For both CPU
and GPU compiled callables the artifact must record:

- a nonempty HLO string obtained from the concrete `jit_compile=True`
  function via TensorFlow compiler IR;
- the exact HLO text, SHA-256, and byte count, with an `ENTRY` computation
  present;
- exactly one cached concrete trace for each exercised static shape; and
- actual output tensor device names.

The CPU artifact requires CPU output placement and no visible GPU. The trusted
GPU artifact requires every canonical numerical output on a logical GPU and at
least one physical and logical GPU in the manifest. The verifier independently
re-obtains nonempty HLO and device placement; stored booleans cannot
self-certify either property.

### Boundary, Ledger, And Checkpoint Commands

All use the exact cache/environment assignments required above. These
non-runtime administrative modes hide GPU before any optional import and do not
initialize TensorFlow.

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/boundary-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --freeze-boundary --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/pre-run-boundary.json
```

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/executor-ledger-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --write-executor-ledger --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/executor-write-ledger.json
```

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/final-checkpoint-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --write-final-checkpoint --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/final-checkpoint.json
```

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-ledger-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --write-post-result-ledger --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-write-ledger.json
```

After closure verification exits and its trace closes, the finite terminal
read-only audit command is:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py --audit-terminal-trace docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/closure-verification-write-trace.log
```

The terminal mode opens files read-only, writes canonical status JSON to stdout
only, and does not initialize TensorFlow. Its returned trace hash/status are
recorded in the conversation and final handoff text; no further repository
artifact is created.

## Evidence Contract Before Runtime

| Field | Binding contract |
| --- | --- |
| Engineering/scientific question | Does the A2 implementation faithfully and replayably realize the locked approximate-filter predictive law? |
| Exact baseline/comparator | Accepted A1 target and a direct TensorFlow recursion using the same constrained parameters and explicit zero/fixed innovations |
| Primary pass criterion | Complete A2 focused suite plus structured CPU and trusted GPU/XLA canaries pass conjunctively |
| Promotion vetoes | Entry/hash drift; filter or total-target mismatch; invalid covariance; recursion/noise/replay/batch/XLA/GPU failure; invalid artifact |
| Explanatory diagnostics | Runtime, trace count, sub-tolerance parity residuals, covariance spectrum/projection residuals, changed-seed deltas |
| Not concluded even on pass | Posterior/sampler correctness, predictive equivalence, calibration, model adequacy, superiority, public/default/product readiness, or scientific validity |
| Preservation artifact | A2 JSON/log, A2 Markdown result, exact commands/run manifests, hashes, reviews, ledger, and A3 handoff |

This contract must be copied into the execution ledger before the first A2
runtime command. Smoke/debug checks may localize a defect but cannot replace the
full primary gate.

## Skeptical Plan Audit

| Audit risk | Finding and repair |
| --- | --- |
| Wrong baseline | Repaired: A2 binds the accepted A1 target and historical `tf_svd_ukf` value route, not old Phase 2V chains or another filter |
| Proxy promotion | Repaired: replay, hashes, runtime, changed-seed sensitivity, and local residuals support engineering checks only; no predictive-equivalence decision occurs |
| Missing stop conditions | Repaired: entry drift, parity, covariance, recursion, XLA/GPU, artifact, review, and boundary vetoes are explicit |
| Unfair comparison | Not yet applicable to sampler arms; shared and independent innovation roles are separated so A2 cannot preselect a favorable A8 comparison |
| Hidden assumptions | Repaired: approximate terminal Gaussian, complete-state draw, process-noise placement, observation timing, horizon, dtype, cluster unit, and filter qualification are explicit |
| Stale context | Repaired: the final accepted A1 result/review and recomputed CPU/GPU signatures replace the earlier blocker state |
| Environment mismatch | Repaired: CPU runs are labeled debug/reference and hide GPU before TensorFlow import; serious evidence includes a trusted managed-session GPU/XLA canary with device/JIT/TF32 provenance |
| Artifact cannot answer question | Repaired: artifacts contain path tensors, target-parity residuals, covariance diagnostics, replay hashes, device provenance, and exact run manifests, while excluding future inferential claims |
| PSD silently treated as SPD | Repaired: scale-aware negative-eigenvalue rejection and zero-only roundoff clipping replace Cholesky/flooring |
| Randomness ambiguity | Repaired: terminal, process, and observation standard normals have fixed shapes, roles, stateless seeds, and content signatures |
| Write-set overlap | Repaired: only additive predictive files and narrow lazy exports are allowed; existing A1 and concurrent-lane changes are preserved |

Audit disposition: `PASS_FOR_BOUNDED_SUBPLAN_REVIEW_ONLY`. It does not authorize
implementation until the exact subplan receives `VERDICT: AGREE`.

## Required Checks And Tests

### Static And Contract Checks

- compile the new module, test, and harness;
- reject NumPy/PyTorch/JAX algorithmic implementation and stateful RNG;
- confirm no benchmark module is imported by production source;
- verify lazy exports and import behavior;
- verify exact write-set and protected A1 hashes;
- run whitespace and forbidden-claim scans; and
- validate JSON schema, finite values, stable signatures, and result bindings;
- prove that nonzero compiled status codes cannot be serialized as passing
  evidence, including under XLA where debug assertions are not treated as the
  only enforcement mechanism.

### Focused TensorFlow Tests

- config/type validation, dtype, static draw/replication/horizon, and invalid
  rank/shape/seed/role contracts;
- exact parameter embedding and constrained-noise scale checks;
- terminal rerun filter likelihood and total-target parity at all ten rows of
  the frozen `[10,4]` matrix with raw hash
  `d6ba48e5a64897f87caeece4de776c139d8fc62d00fc118d89b4d88da468829a`;
- complete filtered-history and terminal tensor shapes;
- finite/symmetric PSD acceptance, exact singular PSD acceptance, roundoff-only
  clipping, materially negative eigenvalue rejection, asymmetry rejection, and
  factor reconstruction checks;
- deterministic zero-bank recursion against direct repeated
  `ssl_lstm_transition` and `ssl_lstm_observation` calls;
- process noise changes only the latent coordinate relative to the deterministic
  transition and never directly perturbs hidden/cell coordinates;
- observation noise is added after the same-horizon state transition;
- fixed-seed exact bank/path replay and changed-seed sensitivity for all three
  innovation families;
- explicit Philox family-seed separation and raw bank content hashes;
- supplied-bank replay without hidden RNG;
- scalar convenience versus one-row batch parity;
- multi-draw batch versus scalar-loop parity and draw-order preservation;
- paired-diagnostic shared-bank and independent-arm role preservation;
- eager debug-reference versus XLA parity;
- compiled trace-cache/static-shape behavior;
- provenance, signatures, nonclaims, and testing-only evidence rejection; and
- A1 regression tests sufficient to prove the target remains unchanged.

The exact focused command is:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/focused-tests-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -p no:cacheprovider -q tests/test_ssl_lstm_predictive_tf.py tests/test_ssl_lstm_posterior_tf.py
```

This is a CPU-hidden focused engineering/regression suite. It is not GPU
evidence and does not replace the trusted GPU/XLA canary.

### CPU Reference Gate

Run with `CUDA_VISIBLE_DEVICES=-1` set before TensorFlow import. It is an
explicit debug/reference exception, not the default execution route. The
structured artifact must include:

- exact command, commit, conda environment, TensorFlow/TFP versions, CPU/GPU
  visibility, intentional GPU-hiding statement, XLA/JIT state, seed, wall time,
  input and source hashes, output paths, plan/result paths, and trust role;
- exactly the frozen ten-row parity matrix and the frozen two-draw/two-
  replication forecast subset;
- terminal parity and covariance diagnostics;
- zero-bank direct-recursion, fixed-bank replay, scalar/batch, and eager/XLA
  residuals; and
- `CPU_REFERENCE_CONTRACT_PASSED` only if all hard gates pass.

The harness first writes the exact bank to `innovation-bank.json`, then loads
that persisted JSON for the CPU computation. The bank JSON stores every tensor
as row-major hexadecimal `float64` strings plus raw byte hashes; decimal JSON
roundtripping is not allowed. The CPU artifact records both bank file SHA-256
and bank evidence signature.

The exact generation commands, using the active `tfgpu` interpreter, are shown
below. Each command also runs with
`PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache`, and
`TMPDIR=/tmp/bayesfilter-a2-tmp`; these three environment assignments are
prepended literally even where wrapped below for readability. The exact
`CUDA_CACHE_PATH` and `XLA_FLAGS` assignments above are also prepended to every
TensorFlow command.

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py \
  --mode cpu-reference \
  --bank docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.json \
  --bank-log docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.log \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json \
  --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.log
```

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-verification-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py \
  --artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json \
  --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference-verify.log
```

### Trusted GPU/XLA Integration Canary

Run the same bounded contract through the repository TensorFlow GPU/XLA route.
The artifact must state trust basis
`owner_designated_managed_session_visible_gpu_trusted` and record visible
devices, selected device, TF32 state, `jit_compile=true`, XLA evidence, exact
command/environment, seed, wall time, source/input hashes, and output paths.

The canary must exercise terminal extraction on the frozen ten-row matrix and
all ten forecast steps on the same persisted two-draw/two-replication bank.
These are engineering coverage counts, not scientific sample sizes. It must
verify bank file SHA-256/evidence signature/raw tensor hashes before execution,
crosslink the exact CPU artifact file SHA-256/evidence signature, and apply the
frozen `4096 * eps` CPU/GPU threshold. A nontrusted GPU failure is sandbox
evidence only and must be rerun in a trusted context before diagnosing the
machine or framework.

The exact trusted commands are:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py \
  --mode gpu-xla-canary \
  --bank docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/innovation-bank.json \
  --cpu-reference docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/cpu-reference.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.json \
  --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.log
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-verification-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py \
  --artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary.json \
  --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/gpu-xla-canary-verify.log
```

The final checkpoint is written only after both fresh-process verifiers and the
implementation review pass. It stores exactly the members fixed by the strict
checkpoint schema, accepted protected A1 bindings, current `HEAD`, cache scan,
and classified outside-write-set inventory. Immediately before A2 result
review, rerun both artifact verifiers and regenerate this checkpoint. Any
relevant mutation invalidates the affected artifact and every downstream
crosslink.

After the final A2 result review and the separately reviewed A3 subplan exist,
run the independent verifier in `--close-phase` mode to write
`post-result-closure.json`, then in `--verify-closure` mode in a fresh process.
The exact commands are:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/closure-generation-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py \
  --close-phase \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-closure.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache TMPDIR=/tmp/bayesfilter-a2-tmp CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' /usr/bin/strace -f -qq -e trace=%file -o docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/closure-verification-write-trace.log /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py \
  --verify-closure docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-closure.json \
  --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/post-result-closure-verify.log
```

Closure verification recomputes all member hashes/signatures, validates the
unchanged final-checkpoint crosslink, result and A3 review verdicts, ledger
statuses, cache scan, and temporal outside-lane classification. Only this
verified post-result closure can satisfy A3 implementation eligibility.

## Repair And Review Loop

1. Obtain a fresh bounded one-path read-only review of this exact subplan.
2. If review returns `REVISE`, patch this same file visibly, rerun focused
   document checks, and request a fresh review.
3. The ordinary five-round cap was reached with Round 5 identifying only the
   terminal-audit exemption and handoff-binding defects. The user's explicit
   additional five-round authorization plus the current instruction to fix and
   continue authorizes up to five exceptional A2 subplan rounds for these
   remaining fixable defects. Exceptional Round 6 consumes one; it does not
   authorize runtime, scientific, product, commit, or broader-boundary changes.
4. After subplan agreement, implement only the reviewed write set.
5. Run focused checks first, then CPU reference, then trusted GPU/XLA canary.
6. Repair fixable A2 defects inside the write set and rerun every invalidated
   check/artifact.
7. Obtain bounded read-only implementation and result reviews. Codex substitute
   reviews must be labeled weaker than Claude.
8. Any semantic repair to target, model, covariance policy, innovation roles,
   or evidence criteria requires subplan amendment and rereview before runtime.

Claude remains policy-unavailable under the recorded trusted-execution ruling;
no liveness probe or indirect disclosure route may be used. This subplan review
therefore uses a fresh bounded native Codex substitute reviewer and makes no
Claude claim.

## Phase-End Result Contract

Write
`docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md`
with:

- outcome and exact status;
- evidence-role, decision, inference-status, and separate-ledger tables;
- implementation and test inventory with hashes;
- terminal parity and covariance diagnostics;
- CPU and GPU/XLA manifests;
- hard veto status, viable scope, and explicit statement that no stochastic
  ranking was attempted or supported;
- descriptive-only differences and uncertainty limitations;
- candidate-versus-direction interpretation;
- strongest alternative explanation and weakest evidence;
- exact commands, seeds, wall times, environment/device data, artifact paths,
  plan/result bindings, and Git state;
- forbidden claims/nonclaims; and
- exact A3 handoff conditions.

The result may emit only one of:

- `PASSED_FOR_A3_PLANNING_ONLY`;
- `FAILED_A2_REPAIRABLE`;
- `BLOCKED_A2_CONTINUATION_VETO`; or
- `INVALID_A2_ARTIFACT`.

## Exact A3 Handoff Conditions

### A3 Drafting And Review Gate

A3 subplan drafting may begin only when all are true:

1. this exact A2 subplan has an agreed hash-pinned bounded review;
2. entry and protected A1 hashes pass;
3. all required focused tests pass;
4. CPU reference and trusted GPU/XLA artifacts pass and validate;
5. target/filter parity, total-value parity, terminal covariance, deterministic
   recursion, process-noise placement, replay, scalar/batch, and eager/XLA gates
   pass;
6. implementation review finds no material defect;
7. the pre-result final checkpoint passes;
8. A2 result records `PASSED_FOR_A3_PLANNING_ONLY` and receives its own agreed
   hash-pinned review; and
9. ledgers/runbook/stop-handoff are refreshed to the A3-planning-only boundary.

Passing this gate authorizes only drafting, skeptically auditing, and obtaining
a bounded review of the A3 subplan from actual A2 interfaces and signatures. It
does not authorize A3 implementation.

### A3 Implementation-Eligibility Gate

A3 implementation becomes eligible only after both are additionally true:

1. the exact A3 subplan receives its own hash-bound `VERDICT: AGREE`; and
2. `post-result-closure.json` plus its fresh-process verify log pass the exact
   closure schema and bind the accepted A2 result/review, A3 subplan/review,
   final checkpoint, and refreshed governance artifacts; and
3. the exact terminal read-only audit command emits status
   `A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED` with the SHA-256 of the closed
   `closure-verification-write-trace.log`, and the supervisor records both the
   status and matching trace hash in the final stop handoff/conversation.

Only then may the exact A3 subplan govern A3 implementation. No broader
authority is implied.

## Stop Conditions

Stop immediately and write a blocker/invalid result if any of these occurs:

- accepted A1 result, review, target, contract, mask, observation, CPU, GPU, or
  protected dependency hash mismatch;
- an unexplained overlapping edit in an A2-owned file;
- a needed change to target, filter, model equations, prior, parameter chart,
  horizon, covariance policy, or innovation semantics;
- filter likelihood or total-target parity failure not explained by a localized
  A2 defect;
- nonfinite, materially asymmetric, or materially indefinite terminal
  covariance;
- direct-recursion, process-noise placement, observation timing, replay,
  scalar/batch, XLA, or trusted GPU hard-gate failure after bounded repair;
- missing/corrupt artifact, invalid provenance, or inability to preserve exact
  run evidence;
- any request to run HMC, NeuTra, calibration, sweep, benchmark ranking,
  predictive equivalence, or product/default change in A2;
- any required action outside the reviewed write set or current human
  authority;
- five substantive review rounds fail to resolve the same blocker; or
- a true continuation veto invalidates the harness, implementation target,
  data, math, runtime, or artifact rather than merely rejecting the current
  candidate.

Do not stop merely because a smoke test or candidate implementation fails when
the failure is the exact localized defect the A2 repair loop is designed to
fix. Record what failed, repair it within scope, and rerun invalidated gates.

## Forbidden Actions And Claims

A2 forbids:

- HMC or NeuTra execution, tuning, training, comparison, or readiness claims;
- forecast moments, MMD, bootstrap, calibration, equivalence margins, or
  scientific inference;
- parameter-posterior correctness/equality or model-adequacy claims;
- exact nonlinear-filter, sampler-superiority, performance-superiority,
  product, release, public-API, or default-readiness claims;
- treating shared innovation replay as independent primary evidence;
- changing A1 `return_filtered`, accepted source, tests, artifacts, model files,
  prior, parameter mask, or target semantics;
- NumPy/PyTorch/JAX algorithmic implementation, stateful forecast RNG, silent
  covariance flooring, nearest-SPD repair, or hidden Cholesky fallback;
- broad external disclosure, Claude substitution claims, detached execution,
  package installation, network fetch, commit, stage, push, reset, restore, or
  cleanup of concurrent work.
