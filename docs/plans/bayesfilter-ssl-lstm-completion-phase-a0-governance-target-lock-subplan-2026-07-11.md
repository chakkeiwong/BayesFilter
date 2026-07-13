# Phase A0 Subplan: Governance, Target, And Artifact Lock

Date: 2026-07-11

Status: `ACTIVE_ATTEMPT_02_IMMEDIATE_VERIFIER_PASSED_FINAL_HANDOFF_PENDING`

## Phase Objective

Freeze a replayable identity for the historical four-parameter scalar
SSL-LSTM SVD-UKF posterior target, classify every historical input by evidence
role, and separate target semantics from execution, sampler geometry, and
forecast design before production code is extracted.

This is a documentation, inventory, and deterministic CPU-hidden reference
replay phase. It does not implement the new target, run HMC, train NeuTra,
probe a GPU, calibrate predictive-equivalence thresholds, or make a scientific
claim.

## Inherited Entry Conditions

- The active roadmap is
  `docs/plans/bayesfilter-ssl-lstm-completion-roadmap-2026-07-11.md`.
- The governing scalar design is
  `docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md`.
- The 2026-07-10 reset memo remains binding: the old independent-parameter-
  reference branch and its Phase 3 handoff remain blocked.
- The model/method description is
  `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`.
- The repository is dirty. Existing changes and untracked historical scalar
  artifacts belong to the user and must not be reset, reformatted, staged, or
  committed.
- Codex in the current conversation is supervisor and executor. Claude Opus at
  max effort is advisory, read-only reviewer only.
- No runtime result from Phase 2V, no local quadratic reference, and no
  `dsge_hmc` artifact is admitted as the new ordinary-HMC baseline.

## Phase Write Set

Only these new SSL-LSTM completion artifacts may be written in A0:

- this subplan and its result;
- the completion roadmap, only for visible review repairs;
- the visible runbook, execution ledger, approval ledger, and stop handoff;
- compact review bundles under `docs/reviews/`;
- generated Claude logs under `.claude_reviews/`;
- the Phase A0 target-lock JSON under the declared artifact root;
- the A0 dependency-discovery manifest and logs under the declared artifact
  root;
- the new A0-only target-lock harness at
  `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`;
- the Phase A1 subplan and its compact review bundle.

No `bayesfilter/`, `tests/`, existing benchmark, historical result, LaTeX, or
other unrelated file is in the A0 write set.

## Required Artifacts

| Artifact | Required path | Role |
| --- | --- | --- |
| Roadmap | `docs/plans/bayesfilter-ssl-lstm-completion-roadmap-2026-07-11.md` | Umbrella completion contract |
| A0 subplan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md` | Binding A0 execution contract |
| A0 structured lock | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json` | Machine-readable identities, hashes, roles, and provenance |
| Dependency manifest | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json` | Non-evidentiary preflight closure used to establish the immutable attempt |
| A0 lock harness | `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py` | A0-only generator and strict independent verifier; not production code |
| A0 result | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md` | Phase close record |
| Visible runbook | `docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md` | Supervisor state machine |
| Execution ledger | `docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md` | Append-only visible state |
| Approval ledger | `docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md` | Authority and escalation boundaries |
| Stop handoff | `docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md` | Recoverable stop/resume state |
| A1 subplan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md` | Exact next-phase handoff |

The result must cite the exact reached review bundles and `.claude_reviews/`
status JSON paths. Bundle paths are created just in time and must each direct
Claude to one exact review target and one narrow question.

## Identity Model

The target lock must contain four separate signatures. No aggregate hash may
erase their different meanings.

### 1. Target-Semantic Signature

The semantic target is the historical scalar posterior log kernel in the free
coordinate, with all of the following bound:

- static dimensions: observed history `T=30`, latent `k=1`, hidden `h=1`,
  observation `d=1`, augmented state dimension `3`, diagonal covariance mode;
- the ordered 24-entry scalar parameter chart;
- the full unconstrained fixture vector from `minimal_ssl_lstm_theta()`;
- free names, in order:
  `latent_mean_weight.0.0`, `latent_mean_bias.0`,
  `observation_weight.0.0`, `observation_bias.0`;
- free full-chart indices `(12,13,14,15)` and the fixed remaining entries;
- the exact `float64`, shape `[30,1]` observation tensor produced by the
  historical stateless generator with seed `(20260708,2301)` and simulation
  noise scale `1.0`;
- the SVD-UKF analytic filtering log likelihood from
  `tf_ssl_lstm_svd_ukf_score` with `std_floor=1e-4`, `alpha=1`, `beta=2`,
  `kappa=0`, placement floor `0`, innovation floor `1e-12`, rank tolerance
  `1e-12`, spectral-gap tolerance `1e-10`, fixed-null tolerance `1e-10`,
  jitter `0`, and `allow_fixed_null_support=False`;
- the unnormalized Gaussian prior log kernel
  `-0.5 * sum((free - truth_free)^2 / 4.0^2)`, centered at the four fixture
  values, with no parameter-independent normalizing constant;
- numerical dtype `float64`.

The target-semantic signature excludes JIT/device choices, Phase 2S geometry,
HMC tuning, forecast horizon, forecast innovations, and equivalence margins.

The lock must also preserve target evaluation anchors at exactly these two
ordered free-coordinate probes:

1. `truth_free`: `(0.35,-0.08,0.65,0.05)`;
2. `phase2s_center`: the four values in the Phase 2S
   `map_local_handoff.center_free_parameter_values` field of exactly
   `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json`.

At each probe, record as `float64` scalar/vector tensor descriptors:

- SVD-UKF filtering log likelihood;
- the likelihood score gathered in the ordered free coordinates;
- unnormalized prior log kernel and prior score;
- total posterior log kernel and total score;
- the maximum absolute residuals of
  `total_value - (likelihood_value + prior_value)` and
  `total_score - (likelihood_score + prior_score)`.

For each probe, define the value-decomposition tolerance exactly as
`8 * eps64 * max(1, abs(total_value), abs(likelihood_value) + abs(prior_value))`
and the score-decomposition tolerance exactly as
`8 * eps64 * max(1, norm_inf(total_score), norm_inf(likelihood_score) + norm_inf(prior_score))`,
where `eps64 = 2**-52`. Require the absolute value residual and score infinity-
norm residual to be no larger than their respective tolerances. A
nonfinite or inconsistent required probe is a continuation veto after harness
correctness is established. These anchors are A1 replay criteria, not posterior
correctness evidence.

Historical JSON stores decimal renderings of the original `float64` anchors.
The strict verifier therefore compares a freshly evaluated scalar value
`value_current` with the corresponding historical value `value_historical`
using exactly
`8 * eps64 * max(1, abs(value_current), abs(value_historical))`. It compares
the current and historical score vectors using exactly
`8 * eps64 * max(1, norm_inf(score_current), norm_inf(score_historical))`,
where vector `norm_inf` is the maximum absolute coordinate. Require the
absolute scalar difference and score-vector infinity-norm difference to be no
larger than those respective tolerances. These tolerances apply only to the two
historical JSON anchor comparisons. Fresh-process replay against the newly
generated lock, tensor byte hashes, component/signature hashes, dependency
closure, and immutable-fingerprint checks remain exact.

### 2. Implementation/Execution Signature

Record byte hashes and Git status for the exact critical roots that currently
define the parameter chart, fixture, model equations, filter, result, and
analytic SVD-UKF score:

- `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py`;
- `bayesfilter/nonlinear/ssl_lstm_protocol.py`;
- `bayesfilter/nonlinear/ssl_lstm_zhaocui_hmc_minimal.py`;
- `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py`;
- `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`;
- `bayesfilter/nonlinear/sigma_points_tf.py`;
- `bayesfilter/structural.py`;
- `bayesfilter/structural_tf.py`;
- `bayesfilter/results_tf.py`.

Use the two-pass procedure below to discover, then pre-lock, every loaded local
Python module whose resolved file lies under the repository root. Record each
relative path and exact byte SHA-256. The preflight-discovered runtime closure,
its manifest hash, and the critical-root list are members of the execution
signature. Record the
historical non-XLA setting separately from the A1 required default
`jit_compile=True`. A1 must prove value/score parity; changing execution mode is
not permission to change target semantics.

### 3. Sampler-Geometry Signature

The authoritative Phase 2S source is exactly
`docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py`.
The authoritative Phase 2S input artifact is exactly
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json`.
The executable geometry representation uses these exact JSON mappings:

- `coordinate_names` <-
  `precondition.coordinate_contract.free_parameter_names`;
- `center_free` <- `map_local_handoff.center_free_parameter_values`;
- `scale` <- `map_local_handoff.scale`;
- `factor_z` <- `map_local_handoff.factor_z`;
- `covariance_z` <- `map_local_handoff.covariance_z`;
- `precision_z` <- `map_local_handoff.precision_z`;
- `covariance_theta` <- `map_local_handoff.covariance_theta`;
- `precision_theta` <- `map_local_handoff.precision_theta`;
- regularization fields <- `settings.mass.dense`, `settings.mass.jitter`,
  `settings.mass.eigenvalue_floor`, and
  `settings.mass.max_condition_number`.

The authoritative ordered free-coordinate tuple is
`(center_free, scale, factor_z)`, with formula
`free = center_free + scale * (factor_z @ u)`. Bind free-coordinate names/order,
`float64` dtype, shapes `[4]`, `[4]`, and `[4,4]`, and the convention that
`factor_z` is lower triangular with positive diagonal and satisfies
`factor_z @ factor_z.T = covariance_z`.

Also record `covariance_z`, `precision_z`, `covariance_theta`, and
`precision_theta` as consistency metadata. The Phase 2S source first transforms
the fitted z-coordinate precision to raw theta coordinates, then calls
`covariance_from_precision`, which symmetrizes, adds the declared `1e-9`
precision jitter, applies the eigenvalue-floor/condition-cap rule, and inverts
the resulting regularized theta precision. Therefore the source-derived checks
must reproduce that regularization path; raw cross-coordinate equality with the
stored regularized theta matrices is not an identity. Record those two raw-to-
stored discrepancies as explanatory diagnostics rather than vetoes. Bind the
originating Phase 2S JSON
byte hash plus its geometry/mass metadata: dense mass, jitter `1e-9`,
eigenvalue floor `1e-4`, and maximum condition number `1e5`. Record the role
`historical_sampler_initialization_and_tuning_context_only`. The optimizer
output is not a certified global MAP, all non-authoritative matrices are
checked metadata rather than an alternate target, and none is part of the
target-semantic signature.

Geometry checks convert all inputs to little-endian C-order `float64`, use
left-to-right `numpy.matmul`, and define `norm_inf(M)` as the maximum absolute
row sum. With `eps64=2**-52`, require:

- `max(abs(triu(factor_z,1))) == 0` and every diagonal entry strictly positive;
- `R_factor = factor_z @ factor_z.T - covariance_z`,
  `tol_factor = 64*eps64*max(1, norm_inf(factor_z)*norm_inf(factor_z.T),
  norm_inf(covariance_z))`, and `norm_inf(R_factor) <= tol_factor`;
- with `D=diag(scale)`, `Dinv=diag(1/scale)`, and
  `P_theta_raw = sym((Dinv @ precision_z) @ Dinv)`, require
  `(D @ covariance_z) @ D` to equal `sym(inv(P_theta_raw))` under the same
  `64*eps64*max(1, norm_inf(lhs), norm_inf(rhs))` tolerance;
- reproduce `mass_matrix.regularization_report` exactly as the cited
  `regularize_precision` source does: symmetrize `P_theta_raw`, add
  `jitter*eye(4)`, run `numpy.linalg.eigh`, set the effective floor to the
  maximum of the requested floor and largest jittered eigenvalue divided by
  the condition cap, clip eigenvalues to that floor, rebuild and symmetrize.
  Require its clipped-eigenvalue count and effective floor to match the report;
- require the rebuilt regularized precision to equal `precision_theta` and
  `sym(inv(rebuilt_regularized_precision))` to equal `covariance_theta`, each
  under `64*eps64*max(1, norm_inf(lhs), norm_inf(rhs))`;
- `R_z_inv = precision_z @ covariance_z - eye(4)`,
  `tol_z_inv = 64*eps64*max(1,
  norm_inf(precision_z)*norm_inf(covariance_z),1)`, and the residual norm no
  larger than tolerance;
- `R_theta_inv = precision_theta @ covariance_theta - eye(4)`,
  `tol_theta_inv = 64*eps64*max(1,
  norm_inf(precision_theta)*norm_inf(covariance_theta),1)`, and the residual
  norm no larger than tolerance;
- record, without pass/fail use, exactly
  `raw_to_stored_theta_covariance_residual_inf =
  norm_inf((D @ covariance_z) @ D - covariance_theta)` and
  `raw_to_stored_theta_precision_residual_inf =
  norm_inf(P_theta_raw - precision_theta)`. These quantify the expected
  difference between raw transformed geometry and the separately regularized
  theta-coordinate mass handoff.

### 4. Forecast-Design Signature

Record the proposed `H=10` terminal-state and path semantics from the governing
program: final SVD-UKF Gaussian filtered state, structural transition, process
noise only in the stochastic latent coordinate, deterministic hidden/cell
completion, observation noise, whole-path clustering, and separate shared and
independent stateless innovation banks. This signature is a prospective design
identity. A4, not A0, freezes calibrated margins, bandwidths, sample counts,
bootstrap settings, and confirmatory seeds.

## Versioned Schema And Canonical Hash Contract

The exact schema is
`bayesfilter.ssl_lstm_completion.phase_a0_target_lock.v1`. Every object rejects
extra or missing keys. Arrays whose order matters are stated explicitly below.
The exact top-level key set is:

- `schema_version`, `created_at_utc`, `artifact_role`, `classification`,
  `run_manifest`, `immutable_attempt_fingerprint`, `source_provenance`,
  `target_semantics`, `implementation_execution`, `sampler_geometry`,
  `forecast_design`, `historical_artifact_disposition`, `probe_results`,
  `signatures`, `nonclaims`.

Exact nested key sets:

| Object/path | Exact keys |
| --- | --- |
| `run_manifest` | `git_commit`, `git_dirty`, `cwd`, `command`, `interpreter`, `python_version`, `packages`, `environment`, `execution_mode`, `cpu_gpu_status`, `trust_basis`, `started_at_utc`, `completed_at_utc`, `wall_time_seconds`, `output_path`, `log_path`, `plan_path`, `result_path` |
| `run_manifest.packages` | `tensorflow`, `tensorflow_probability_distribution`, `numpy` |
| `run_manifest.environment` | `CUDA_VISIBLE_DEVICES`, `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, `TF_ENABLE_ONEDNN_OPTS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`, `OMP_NUM_THREADS`, `TF_CPP_MIN_LOG_LEVEL` |
| `immutable_attempt_fingerprint` | `checkpoint_stage`, `dependency_manifest_path`, `dependency_manifest_file_sha256`, `dependency_manifest_aggregate_sha256`, `members`, `aggregate_sha256` |
| each ordered fingerprint member | `path`, `sha256`, `git_status`, `role` |
| `source_provenance` | `critical_roots`, `runtime_loaded_local_dependencies`, `historical_inputs`, `governance_inputs`, `dsge_hmc` |
| each ordered source descriptor | `path`, `sha256`, `git_status`, `role` |
| `source_provenance.dsge_hmc` | `repository_path`, `git_commit`, `git_dirty`, `governance_files`, `gate1_context_files`, `role` |
| each external `dsge_hmc` file descriptor | `absolute_path`, `sha256`, `role` |
| `target_semantics` | `static_config`, `parameter_chart`, `full_fixture`, `free_mask`, `fixed_parameters`, `observations`, `likelihood`, `prior`, `dtype` |
| `target_semantics.static_config` | `horizon`, `latent_dim`, `hidden_dim`, `observation_dim`, `augmented_state_dim`, `covariance_mode`, `full_parameter_dim` |
| `target_semantics.free_mask` | `names`, `indices`, `truth_free` |
| `target_semantics.likelihood` | `name`, `score_helper`, `std_floor`, `alpha`, `beta`, `kappa`, `placement_floor`, `innovation_floor`, `rank_tolerance`, `spectral_gap_tolerance`, `fixed_null_tolerance`, `jitter`, `allow_fixed_null_support` |
| `target_semantics.prior` | `family`, `center`, `standard_deviation`, `normalized`, `log_kernel_formula` |
| `implementation_execution` | `interpreter`, `python_version`, `packages`, `environment`, `critical_roots`, `runtime_loaded_local_dependencies`, `dependency_manifest_file_sha256`, `dependency_manifest_aggregate_sha256`, `harness`, `historical_execution_mode` |
| `implementation_execution.harness` | `path`, `sha256` |
| `implementation_execution.historical_execution_mode` | `device`, `cpu_hidden`, `jit_compile`, `xla`, `dtype`, `role` |
| `sampler_geometry` | `role`, `source_path`, `source_sha256`, `source_script_path`, `source_script_sha256`, `coordinate_names`, `coordinate_formula`, `center_free`, `scale`, `factor_z`, `covariance_z`, `precision_z`, `covariance_theta`, `precision_theta`, `reconstruction_tolerance_formula`, `regularization`, `checks`, `nonclaims` |
| `sampler_geometry.regularization` | `dense_mass`, `jitter`, `eigenvalue_floor`, `max_condition_number` |
| `sampler_geometry.checks` | `factor_lower_triangular`, `factor_positive_diagonal`, `factor_covariance_residual_inf`, `factor_covariance_tolerance`, `raw_theta_covariance_residual_inf`, `raw_theta_covariance_tolerance`, `regularized_theta_precision_residual_inf`, `regularized_theta_precision_tolerance`, `regularized_theta_covariance_residual_inf`, `regularized_theta_covariance_tolerance`, `raw_to_stored_theta_covariance_residual_inf`, `raw_to_stored_theta_precision_residual_inf`, `z_inverse_residual_inf`, `z_inverse_tolerance`, `theta_inverse_residual_inf`, `theta_inverse_tolerance`, `regularization_effective_eigenvalue_floor`, `regularization_clipped_eigenvalue_count`, `passed` |
| `forecast_design` | `status`, `horizon`, `terminal_state`, `transition`, `process_noise`, `hidden_cell_completion`, `observation_noise`, `path_clustering`, `innovation_banks`, `unfrozen_fields` |
| `probe_results` | `truth_free`, `phase2s_center` |
| each `probe_results` value | `name`, `free_position`, `likelihood_value`, `likelihood_score`, `prior_value`, `prior_score`, `total_value`, `total_score`, `value_residual`, `score_residual_inf`, `value_tolerance`, `score_tolerance`, `passed` |
| each `historical_artifact_disposition` ordered row | `path`, `sha256`, `git_status`, `role`, `disposition`, `promoting` |
| `signatures` | `target_semantic_sha256`, `implementation_execution_sha256`, `sampler_geometry_sha256`, `forecast_design_sha256`, `aggregate_sha256` |

`parameter_chart` is an ordered array of 24 strings; `names`, `indices`,
`coordinate_names`, dependency/source/member arrays, probe order
`(truth_free, phase2s_center)`, disposition rows, and `nonclaims` are ordered
arrays. `fixed_parameters` is an ordered array of objects with exact keys
`name`, `index`, `value`. Every numeric scalar/vector/matrix stored as evidence
uses a tensor descriptor with exact keys `name`, `dtype`, `shape`, `order`,
`byte_order`, `values`, `raw_sha256`; its `dtype` is `float64`, `order` is `C`,
and `byte_order` is `little`. `values` is uniquely shape-nested: rank zero is a
single finite JSON number with `shape=[]`; rank one is one list of exactly
`shape[0]` finite numbers; rank two is a list of exactly `shape[0]` row lists,
each containing exactly `shape[1]` finite numbers. Higher rank is forbidden in
v1. Raw bytes are obtained by recursively materializing that nesting as NumPy
dtype `<f8`, requiring the declared shape, then applying
`numpy.ascontiguousarray(...).tobytes(order='C')`.

The immutable fingerprint `checkpoint_stage` has the sole accepted value
`opening_warmup_complete_closing_and_handoff_rehash_match`. Its `members` are the lexicographically path-sorted
unique union of the exact critical roots, dependency-manifest runtime closure,
closed historical inputs listed below, A0 harness, and dependency manifest.
When one path has multiple roles, `role` is the lexicographically sorted unique
role strings joined by `+`. Assign exact special roles
`a0_lock_harness` and `dependency_manifest_exact_bytes`; all other role strings
come from their closed-set descriptors. `git_status` is the exact two-character
XY code from `git status --porcelain=v1 --untracked-files=all -- <path>`; use
two spaces (`'  '`) when no row is returned, otherwise require exactly one row,
take its first two characters, and require its path suffix to resolve to the
same repository-relative member. Allowed XY values are the Git porcelain-v1
codes plus `??`; any other or multiple row is rejected.

`dependency_manifest_file_sha256` is SHA-256 of the exact manifest file bytes.
`dependency_manifest_aggregate_sha256` is the manifest's internally verified
semantic `aggregate_sha256`; they are distinct and both required. The immutable
aggregate projection is exactly
`{checkpoint_stage, dependency_manifest_path,
dependency_manifest_file_sha256, dependency_manifest_aggregate_sha256,
members}` and excludes `aggregate_sha256`. Opening, closing, first fresh-process
verification, and final pre-handoff rehash projections must be byte-identical
under canonical JSON before the stored aggregate is accepted.

JSON type/value contract:

- paths, hashes, formulas, roles, status fields, package versions, interpreter,
  environment values, dtype/device/execution values, and all forecast semantics
  are strings; booleans are JSON booleans; counts/indices/dimensions are JSON
  integers; finite settings, residuals, tolerances, and wall time are JSON
  numbers;
- only `target_semantics.full_fixture`, `target_semantics.free_mask.truth_free`,
  `target_semantics.observations`, `target_semantics.prior.center`, every
  `probe_results.*` numeric value/position/score, and the seven named
  `sampler_geometry` vectors/matrices use tensor descriptors; configuration
  scalars and `fixed_parameters[].value` are plain finite JSON numbers;
- `implementation_execution.packages` and `.environment` use the exact key sets
  and string values from `run_manifest`; their canonical objects must be equal;
- `forecast_design` values are exactly:
  `status='prospective_not_frozen_until_a4'`, `horizon=10`,
  `terminal_state='final_svd_ukf_filtered_gaussian_per_parameter_draw'`,
  `transition='structural_ssl_lstm_transition'`,
  `process_noise='stochastic_latent_coordinate_only'`,
  `hidden_cell_completion='deterministic'`,
  `observation_noise='ssl_lstm_observation_law'`,
  `path_clustering='entire_length_10_path_is_one_clustered_observation'`,
  `innovation_banks=['shared_for_paired_mean_log_variance',
  'independent_arm_specific_for_primary_mmd_and_robustness']`, and
  `unfrozen_fields=['equivalence_margins','feature_scales',
  'forecast_replication_count','mmd_bandwidths','mmd_mixture_weights',
  'mmd_tolerance','bootstrap_type','bootstrap_count','bootstrap_seed',
  'block_length','confidence_level','covariance_ridge',
  'condition_number_cap','sampler_seeds','forecast_seeds']` in that order.
- `sampler_geometry.reconstruction_tolerance_formula` is exactly the string
  `eps64=2**-52;norm_inf=max_abs_row_sum;matmul=left_to_right_numpy_float64;source_regularize_precision=symmetrize_plus_jitter_then_eigh_floor_cap;tol=64*eps64*max(1,lhs_norm,rhs_norm)`;
  every residual/tolerance field in `sampler_geometry.checks` is a plain finite
  JSON number, and its three pass fields are JSON booleans.

The four component projections are exact objects, not field-selection logic:

```text
target projection = {schema_version, target_semantics, probe_results}
implementation projection = {schema_version, implementation_execution}
sampler projection = {schema_version, sampler_geometry}
forecast projection = {schema_version, forecast_design}
```

The aggregate projection is exactly
`{schema_version, target_semantic_sha256, implementation_execution_sha256,
sampler_geometry_sha256, forecast_design_sha256}`. It excludes its own
`aggregate_sha256`. Timestamps, run paths, Git dirty preview, reviews,
historical dispositions, and nonclaims are excluded from these projections.

The dependency manifest uses schema
`bayesfilter.ssl_lstm_completion.phase_a0_dependency_manifest.v1` and exact
top-level keys `schema_version`, `created_at_utc`, `harness`, `critical_roots`,
`runtime_loaded_local_dependencies`, `historical_inputs`, `environment`,
`probe_names`, `discovery_only`, `aggregate_sha256`. `harness` has exact keys
`path`, `sha256`; every ordered path descriptor in the three path arrays has
exact keys `path`, `sha256`, `git_status`, `role`; `environment` uses the exact
key set declared for `run_manifest.environment`; `probe_names` is exactly the
ordered array `['truth_free','phase2s_center']`; and `discovery_only` must be
true. Its aggregate projection is the exact object
`{schema_version, harness, critical_roots,
runtime_loaded_local_dependencies, historical_inputs, environment,
probe_names, discovery_only}` and excludes timestamps and `aggregate_sha256`.
The same strict loader, exact-key, ordering, finiteness, path-within-repository,
file-existence, current-byte-hash, and aggregate-hash checks apply before it can
establish an immutable attempt.

## Closed Provenance Path Sets

`source_provenance.governance_inputs` is the lexicographically sorted descriptor
array for exactly:

- `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`;
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-reset-memo-2026-07-10.md`;
- `docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-roadmap-2026-07-11.md`.

`source_provenance.historical_inputs`, dependency-manifest
`historical_inputs`, the immutable historical-member set, and
`historical_artifact_disposition` use exactly these lexicographically sorted
BayesFilter paths:

- `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py`;
- `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json`;
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json`.

`source_provenance.dsge_hmc` is exactly:

- `repository_path='/home/ubuntu/python/dsge_hmc'`;
- `governance_files` is the ordered descriptor array for
  `/home/ubuntu/python/dsge_hmc/AGENTS.md` with role
  `external_neutra_governance` and
  `/home/ubuntu/python/dsge_hmc/CLAUDE.md` with the same role;
- `gate1_context_files` is the ordered descriptor array for
  `/home/ubuntu/python/dsge_hmc/docs/plans/neutra-gate1-real-linux-gpu-german-budget-result-2026-05-06.md`
  with role `external_gate1_closure_context` and
  `/home/ubuntu/python/dsge_hmc/docs/plans/neutra-gate3-surrogate-hmc-reset-memo-clean-2026-05-16.md`
  with role `external_neutra_current_context`;
- current Git commit/dirty boolean and role
  `external_design_provenance_only_not_bayesfilter_evidence`.

External `dsge_hmc` paths and mutable governance inputs are hash-recorded
provenance but are excluded from the immutable attempt and all four component
signatures. Their change blocks A0 handoff pending a visible provenance refresh;
it does not silently change the already locked target semantics.

- Files use SHA-256 over exact bytes.
- Numeric tensors record logical name, dtype, shape, C-order little-endian raw
  byte SHA-256, and a JSON value representation for human audit.
- Semantic objects use strict UTF-8 JSON with sorted keys, separators `,` and
  `:`, `allow_nan=false`, and SHA-256 over those canonical bytes.
- The lock records both component hashes and the four top-level signatures.
- Missing values, nonfinite values, duplicated parameter names, wrong order,
  shape drift, dtype drift, or ambiguous canonicalization are hard vetoes.
- The harness must implement a strict loader using `object_pairs_hook` to reject
  duplicate keys and `parse_constant` to reject `NaN`, `Infinity`, and
  `-Infinity`; `python -m json.tool` is syntax-only and cannot pass the gate.
- Strict verification requires the exact schema/key sets; reconstructs every
  numeric tensor from its JSON values in the declared little-endian dtype and
  C order; checks shape and finiteness; recomputes raw-byte hashes, every
  component projection, and the aggregate projection; and rejects extra or
  missing signature members.

## Locked Runtime And Exact Commands

The A0 runtime baseline is:

- interpreter: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`;
- expected Python `3.13.13`, TensorFlow distribution `2.20.0`, `tfp-nightly`
  distribution `0.25.0`, and NumPy `2.1.3`;
- `CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`,
  `TF_DETERMINISTIC_OPS=1`, `TF_ENABLE_ONEDNN_OPTS=0`,
  `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`, and
  `OMP_NUM_THREADS=1`, all set before Python/TensorFlow import;
- existing historical target setting `jit_compile=False`, explicitly labeled
  CPU-hidden non-XLA reference replay only.

Any interpreter/package mismatch blocks execution pending a visible subplan
refresh; do not install or silently substitute an environment. After the A0
harness itself passes local syntax/read-only review, run the non-evidentiary
dependency-discovery pass exactly:

```bash
mkdir -p docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0
CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py \
--discover-dependencies \
docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json \
--log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-discovery.log
```

This pass repeatedly performs one complete warm-up cycle (construct target
twice, evaluate both probes, load Phase 2S, build geometry checks) until two
consecutive cycles have identical sorted local-module path sets, with at most
five cycles. The `5` is an operational convenience bound, not evidence. Failure
to stabilize is a harness/dependency repair trigger. Only the final stable set
is written. This pass is not A0 pass evidence. The harness then strictly verifies
the dependency manifest. This discovery/preflight stage does not establish an
attempt opening. Without editing any prospective immutable member, run the
evidentiary pass exactly:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py \
--dependency-manifest \
docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json \
--output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json \
--log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.log
```

The sole canonical opening occurs inside the evidentiary process. Before that
opening, the harness must run one full non-evidentiary warm-up cycle identical
to discovery's cycle, require the resulting sorted
loaded local-module path set to equal the manifest exactly, and require every
manifest/current hash to match. Only then does it record the opening
fingerprint, runs a
fresh pair of evidentiary target constructions/probes without clearing imports,
and rechecks the same set/hashes at closing. A module loaded only after the
opening comparison is a closure-drift failure: close the attempt, rerun
discovery, rereview any required harness repair, and establish a new checkpoint.

Then verify in a fresh process with the same environment prefix and:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py \
--verify docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json
```

`--verify` performs strict JSON/schema/raw-tensor/component-projection/
signature-aggregate/dependency-manifest checks and rehashes every immutable
filesystem member. Run it first immediately after generation. After drafting
and reviewing the A0 result and A1 subplan, run the identical command again as
the final pre-handoff rehash. The lock's sole opening projection, closing
projection, first verifier projection, and final pre-handoff projection must all
be byte-identical and reproduce the stored immutable aggregate. Mutable result,
governance, review, ledger, and A1 files may change between verifier calls
because they are not immutable members.

The harness callable and arguments are therefore fixed before evidence is
generated; no inline `python -c` or alternate historical script is allowed.

## Historical Artifact Disposition

| Input | A0 disposition |
| --- | --- |
| Committed 2026-07-08 geometry source and JSON | Target-construction source/context; must be replayed and hashed, but not promoted as current production code |
| Untracked Phase 2S source and JSON | Hash-pinned sampler-geometry context only; not durable authority until explicitly committed by a human |
| Phase 2V one-chain CPU-hidden result | Diagnostic context only; forbidden as confirmatory baseline |
| Phase 2W-2Z reference/proposal artifacts | Failed or exploratory historical context only; forbidden as posterior correctness evidence |
| `dsge_hmc` Gate-1 closure artifacts | Read-only design/provenance context only; not BayesFilter trainer, target, HMC, or predictive evidence |
| Current BayesFilter fixed-transport loaders/tuners | Existing mechanics inventory; no claim that a dense-IAF trainer exists or is validated |

## Required Checks, Tests, And Reviews

### Local Document And Source Checks

1. After all planning reviews/repairs converge and the A0 harness passes its
   own read-only review, run dependency discovery and validate the prospective
   member inventory. The sole opening immutable-attempt checkpoint is recorded
   by the evidentiary harness only after its in-process warm-up and manifest
   equality check. Members are the exact critical roots, stable runtime-loaded
   local dependency closure, exact historical inputs, A0 harness, and dependency
   manifest. Generated target-lock JSON/logs, governance plans, ledgers, review
   records, result, and A1 subplan are mutable provenance and excluded; the
   generated dependency manifest is the sole explicit generated-JSON exception
   and is immutable by exact bytes once the attempt opens.
2. Record SHA-256 for the roadmap, governing program, reset memo, LaTeX chapter,
   parameter protocol, model adapter, scalar fixture, historical geometry source
   and JSON, and Phase 2S source and JSON.
3. Verify all required headings, artifact paths, evidence roles, vetoes,
   handoff conditions, stop conditions, and numeric-provenance rows with focused
   `rg` checks.
4. Run `py_compile` on the A0 harness and scoped whitespace checks for every new
   A0 file. Do not let unrelated
   dirty files become an A0 failure.
5. Parse `target-lock.json` with `python -m json.tool` only as a syntax check,
   then run the fresh-process strict `--verify` command to reject duplicate
   keys/nonfinite constants and independently recompute tensor, component, and
   raw tensor hashes, four component-projection hashes, overall signature-
   aggregate hash, dependency-manifest file/semantic hashes, every immutable
   member hash, and the immutable-fingerprint aggregate. Repeat this strict
   verifier after result/A1 review immediately before handoff.

### Deterministic CPU-Hidden Reference Replay

Run exactly the locked small reference harness with `CUDA_VISIBLE_DEVICES=-1`
before importing TensorFlow. It instantiates the existing historical target,
evaluates the two prescribed probes, and writes the versioned lock. It must not
fit geometry, run HMC, train a transport, forecast, benchmark performance, or
initialize/probe CUDA. Record the exact command and bounded output in the A0
result; the declared log preserves full output.

The replay must establish:

- dimensions, names, indices, base vector, truth-free vector, and observation
  tensor match the historical target metadata;
- two independent constructions in the same process produce byte-identical
  observations;
- both prescribed probes decompose consistently into likelihood, prior, and
  total value/score components;
- all target-defining numbers are finite;
- the successful attempt's immutable members remain unchanged from its sole
  post-warm-up opening through closing, first fresh-process verification, and
  final pre-handoff rehash after result/A1 review.

If a fixable harness/verifier defect is found, close that attempt as failed,
repair visibly, rereview the harness, and establish a new opening checkpoint;
never overwrite the prior attempt's recorded fingerprint or describe it as the
successful attempt.

### NeuTra Governance Audit

Read the current exact `dsge_hmc` governance and Gate-1 closure artifacts.
Record their repository commit and dirty status. The audit decides only which
design provenance must be revisited before A6. It cannot import `dsge_hmc`
results as BayesFilter validation or authorize A6 runtime.

### Review Protocol

- Material roadmap, A0 subplan, runbook, A0 result, and A1 subplan reviews use
  `/home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh` with
  `--model opus --effort max --probe-effort low --probe-timeout 90`.
- The first prompt for each review names one exact path, asks one precise
  question, forbids edits/commands/agents/repo-wide review, and requires exactly
  `VERDICT: AGREE` or `VERDICT: REVISE`.
- A successful tiny probe followed by no material response means the prompt is
  too broad or malformed; shrink or redesign it and retry.
- Only a failed probe/transport status permits a fresh Codex read-only
  substitute. Label it `CODEX_SUBSTITUTE_REVIEW`; never call it Claude
  convergence.
- If trusted execution forbids external review even after explicit informed
  user approval, do not retry or circumvent the policy. Use a fresh native
  Codex read-only reviewer as the materially safer substitute, record that
  Claude liveness was not tested, and retain the same weaker-label rule.
- Maximum five substantive verdict rounds for this material subplan. One round
  is one completed `AGREE` or `REVISE` verdict, regardless of edits made before
  the next verdict. A stalled/no-verdict attempt consumes no substantive round.
  A material unresolved fifth-round finding is a stop condition.
- Claude is advisory. A review verdict cannot waive a local check, evidence
  veto, human boundary, runtime boundary, or scientific claim boundary.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is one historical four-parameter scalar SSL-LSTM posterior target unambiguously specified and replayable enough to extract into production-owned TensorFlow code without changing its estimand? |
| Exact baseline | The committed 2026-07-08 target construction and byte-pinned untracked Phase 2S context at current `HEAD`; not Phase 2V and not the failed reference branch |
| Primary pass criterion | All target components resolve; deterministic replay matches metadata; semantic/execution/geometry/forecast identities are separate; structured lock validates; local checks and material reviews have no unresolved finding |
| Promotion vetoes pending repair | Schema/key/projection defect, verifier defect, incomplete result, harness bug, recoverable metadata mismatch, scoped-check failure, or unresolved review finding; no handoff while active |
| Immediate continuation vetoes | Ambiguous/missing target component; internally inconsistent source/chart; nondeterministic observations after verified clean replay; nonfinite or inconsistent required target probe after harness validation; immutable source drift not caused by an authorized repair; environment mismatch requiring install/substitution; required work outside A0 authority |
| Repair triggers | Documentation omission, canonicalization ambiguity, stale path, missing numeric provenance, review-scope problem, or harness/verifier defect while target source remains coherent |
| Repair exhaustion | Maximum five completed substantive verdict rounds for this A0 subplan; edits between verdicts do not separately consume rounds and stalled/no-verdict attempts do not count. A material fifth-round finding is a terminal A0 blocker and never becomes a pass. |
| Explanatory only | Phase 2S center/covariance, old HMC summaries, source commit age, and `dsge_hmc` provenance; prescribed target probes are engineering replay criteria, not explanatory-only |
| Preserved artifact | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json` plus the A0 result and review status files |
| Not concluded | No production implementation, posterior correctness, HMC validity/readiness/convergence, NeuTra validity, predictive equivalence, calibration, GPU/XLA readiness, sampler ranking, model adequacy, release/default readiness, or Zhao-Cui source faithfulness |

## Diagnostic Role Classification

| Diagnostic | Role |
| --- | --- |
| Target component existence/order/dtype/shape | Promotion criterion; immediate continuation veto if source ambiguity/inconsistency, otherwise promotion veto pending harness repair |
| Deterministic observation and required-probe replay | Promotion criterion; immediate continuation veto after verified clean replay failure |
| Strict tensor/component/aggregate hash recomputation | Promotion criterion; promotion veto pending verifier repair, terminal on repair exhaustion |
| Successful-attempt immutable fingerprint stability | Immediate continuation veto for unexpected target-input drift; authorized harness repair closes the old attempt and starts a new checkpoint |
| Phase 2S center/covariance | Explanatory diagnostic and future tuning input only |
| Prescribed target values/scores | A0 engineering replay criterion and A1 replay anchor; not posterior-correctness or scientific evidence |
| Phase 2V/2W-2Z sampler/reference summaries | Explanatory historical context only |
| Claude or Codex substitute review | Planning-quality veto; never scientific promotion evidence |

## Numeric Provenance

| Number | Provenance | A0 role |
| --- | --- | --- |
| `T=30`, seed `(20260708,2301)`, simulation scale `1.0` | Historical committed scalar geometry fixture | Target/data identity |
| Free indices `(12,13,14,15)` and dimension `4` | Derived from the ordered scalar chart and named free mask | Target identity; recompute, do not trust by position alone |
| Prior standard deviation `4.0` | Historical target source and governing program | Target identity |
| `std_floor=1e-4`, UKF `(alpha,beta,kappa)=(1,2,0)`, placement floor `0`, innovation floor `1e-12`, rank tolerance `1e-12`, spectral-gap tolerance `1e-10`, fixed-null tolerance `1e-10`, jitter `0`, fixed-null support false | Historical target implementation and wrapper defaults/override | Target numerical identity; harness must record all values even where the wrapper inherits a helper default |
| `float64` | Historical target and target-parity requirement | Target identity |
| Forecast `H=10` and equal weights `0.1` | Governing predictive-equivalence proposal | Prospective design hypothesis until A4 freeze |
| Phase 2S center/scale/covariance | Historical diagnostic output | Tuning context only |
| Float64 reconstruction tolerances | Source-aware factor, raw-coordinate, regularized-precision/covariance, and inverse formulas using `norm_inf`, fixed multiplication order, and `64*eps64`; the two raw-to-stored discrepancies are explanatory only | Serialization/source-route consistency only; not a scientific or sampler threshold |
| Dependency warm-up cycles | Maximum 5, stop at first two consecutive identical local-module path sets | Operational convenience bound; failure triggers harness/dependency repair |
| Probe decomposition tolerances | Fixed formulas `8 * eps64 * max(1, ...)` stated in the target-probe contract | Engineering identity check only; not a statistical threshold |
| Claude probe `90s`, material timeout `180s`, max one retry, max five substantive rounds | Local gate guide, operational convenience, and owner instruction | Review operations only |

Any unlisted threshold, training budget, HMC budget, architecture, equivalence
margin, MMD tolerance, bootstrap count, or sample count remains
`UNSET_REQUIRES_LATER_REVIEWED_SUBPLAN`.

## Skeptical Pre-Execution Audit

| Risk | A0 control |
| --- | --- |
| Wrong baseline | Locks the historical target construction only; Phase 2V and failed references remain context |
| Proxy promotion | Hash/replay/review can pass A0 only; they cannot pass implementation, HMC, NeuTra, predictive, or model-adequacy gates |
| Missing stop condition | Target ambiguity, verified-clean nondeterminism/probe failure, unexpected immutable drift, or repair exhaustion stops A0; harness/schema/hash defects block promotion pending bounded repair |
| Unfair comparison | No method comparison occurs in A0 |
| Hidden assumption | Prior normalization, filter settings, dtype, generator, and Phase 2S role are explicit |
| Stale context | Opening/closing source hashes and current reset memo are required |
| Environment mismatch | TensorFlow extraction is explicitly CPU-hidden; GPU/XLA is neither probed nor claimed |
| Artifact does not answer question | Structured lock contains semantic components and independent hash validation, rather than only file names |

Audit status before review: `PASSED_FOR_A0_REVIEW_AND_REFERENCE_INVENTORY_ONLY`.

## Forbidden Claims And Actions

- Do not edit algorithmic source or tests.
- Do not run HMC, MCMC, filtering benchmarks, geometry fitting, forecasting,
  NeuTra training, or predictive comparisons.
- Do not probe or initialize GPU/CUDA.
- Do not migrate to the principal-square-root target or alter any filter/math
  setting.
- Do not normalize the historical prior value convention.
- Do not treat Phase 2S as a certified MAP or posterior covariance.
- Do not treat a hash match, deterministic replay, or Claude agreement as
  correctness, convergence, validity, readiness, or scientific evidence.
- Do not stage, commit, push, reset, restore, clean, or overwrite unrelated
  dirty work.

## Required Phase Result

The A0 close record must include:

- decision table;
- inference-status table, including hard veto, supported ranking, descriptive-
  only differences, default readiness, and next evidence;
- separate engineering, sampler/numerical, computational-equivalence,
  synthetic-calibration, and empirical-adequacy ledgers;
- a run manifest with commit, dirty status, command, environment, deliberate
  CPU hiding, seeds, wall time, artifacts, plan, and result;
- full target-component and signature table;
- historical artifact disposition table;
- review rounds, statuses, findings, and repairs;
- candidate failure versus research-direction status;
- post-run red-team note;
- explicit nonclaims and exact A1 handoff.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

1. Roadmap, A0 subplan, and visible runbook pass scoped local checks and
   material review with no unresolved finding.
2. The A0 target-lock JSON satisfies the exact v1 schema, parses with the strict
   duplicate/nonfinite-rejecting loader, and all raw tensor, component-
   projection, overall signature-aggregate, dependency-manifest file/semantic,
   immutable-member, and immutable-fingerprint hashes recompute.
3. The deterministic CPU-hidden replay matches the locked chart, fixture,
   observations, and metadata exactly.
4. The successful attempt's sole post-warm-up opening, closing, first verifier,
   and final pre-handoff immutable projections match exactly; mutable
   governance/result/A1 artifacts are recorded separately.
5. The A0 result records every required table, manifest, review, limitation,
   and nonclaim and is accepted by material read-only review.
6. The A1 subplan is drafted from the actual A0 lock, contains exact write set,
   parity tests, CPU/GPU/XLA boundaries, evidence contract, repair/stop rules,
   and receives its own material read-only review.
7. No target, runtime, model-file, product, default-policy, or scientific-claim
   boundary remains implicit.

Only then may A1 implementation begin.

## Stop Conditions

- Any target-defining component cannot be resolved uniquely.
- Observation or required-probe replay remains nondeterministic, nonfinite, or
  inconsistent after the harness/verifier and dependency closure pass their
  own clean checks.
- An immutable target source/input changes unexpectedly during a successful
  attempt. An authorized harness-only repair instead closes the failed attempt
  and requires a fresh checkpoint.
- The untracked Phase 2S artifact disappears or changes before its role is
  recorded.
- A required lock/result/harness remains invalid or incomplete after the
  bounded fixable-repair allowance.
- A material review finding does not converge after five rounds.
- Continuing would require editing outside the A0 write set, installing a
  package, fetching a network resource, probing GPU, running HMC/NeuTra, making
  a model/default/product decision, or overriding a scientific gate.

## Mandatory Phase-End Sequence

1. Run the pre-replay local checks and the one bounded reference replay, which
   writes the structured lock.
2. Run the immediate syntax check and fresh-process strict verification of the
   structured lock.
3. Write the A0 result/close record and draft the A1 subplan from the verified
   locked facts.
4. Review the A0 result and A1 subplan independently.
5. Repair the same artifact visibly and rerun focused checks after each
   fixable finding, then rerun the fresh-process strict verifier and immutable
   rehash immediately before handoff.
6. Advance only if every handoff condition passes; otherwise update the stop
   handoff with the exact blocker.
