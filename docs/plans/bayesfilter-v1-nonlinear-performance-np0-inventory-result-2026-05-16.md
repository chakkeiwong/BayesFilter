# BayesFilter V1 Nonlinear Performance NP0 Inventory Result

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP0 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-baseline-plan-2026-05-15.md`

## Phase purpose

Record the current nonlinear surface inventory and baseline evidence before any
NP1-NP7 benchmark extension, optimization, XLA gate, CPU/GPU ladder, or
default-policy claim.

## Scope executed

This NP0 execution inspected local source, tests, benchmark harnesses, and
existing benchmark/result artifacts only. It did not edit production code, run
benchmarks, run tests, probe GPU/CUDA, or make new performance claims.

## Skeptical plan audit

Audit target: whether NP0 can answer the stated inventory question without
silently promoting prior benchmark rows into support claims.

Findings:

- Wrong-baseline risk: avoided by treating repository code and existing
  artifacts as the only baseline, not any implied future benchmark schema.
- Proxy-metric risk: dense one-step projection rows for Models B-C appear in
  existing artifacts, so NP0 must keep them labeled as explanatory-only rather
  than exact nonlinear likelihood evidence.
- Hidden-support risk: current code marks `compiled_status="eager_tf"` or
  `"eager_numpy"`; therefore graph/XLA/GPU cells cannot be inferred from result
  metadata alone and must instead be classified from direct code structure and
  existing targeted tests/artifacts.
- Unknown-cell risk: helper exports from `bayesfilter/nonlinear/__init__.py`
  could otherwise be omitted; NP0 explicitly enumerates every public export and
  assigns each a non-filter or filter role.
- Branch-overclaim risk: score paths have smooth and structural-fixed-support
  branches, so per-model/per-backend classification must be explicit rather than
  backend-global.

Audit outcome: pass. The plan can answer the inventory question if all support
cells are bounded by inspected code/tests/artifacts and unsupported cells are
stated as non-claims, not as missing information.

## Evidence contract

Question:

- What nonlinear filter surfaces exist now, and which are production
  TensorFlow candidates versus testing helpers or NumPy references?

Baseline:

- Repository state inspected on 2026-05-16.
- Existing nonlinear benchmark/result artifacts under `docs/benchmarks`.

Primary criterion:

- Every public export from `bayesfilter/nonlinear/__init__.py` is classified as
  production TensorFlow filter surface, testing helper surface, or non-filter
  helper export, with no unknown support cells.

Veto diagnostics:

- NumPy reference paths labeled XLA/GPU-ready.
- Dense one-step Model B-C projection diagnostics treated as exact nonlinear
  likelihood evidence.
- Existing tiny or single-shape XLA/GPU rows generalized into broad support or
  speedup claims.
- Score paths described without explicit branch requirements.

Explanatory diagnostics only:

- Existing CPU-only nonlinear timing artifacts.
- Existing single-shape CPU/GPU/XLA diagnostic artifacts.
- Existing dense one-step projection errors for Models B-C.

What this NP0 result does not conclude:

- any new timing result;
- any new XLA certification beyond current targeted evidence boundaries;
- any broad GPU speedup claim;
- any default backend change;
- any exact nonlinear likelihood correctness result for Models B-C;
- any HMC, Hessian, or client-switch-over readiness claim.

## Sources inspected

### Implementation surfaces

- `bayesfilter/nonlinear/__init__.py`
- `bayesfilter/nonlinear/sigma_points_tf.py`
- `bayesfilter/nonlinear/svd_cut_tf.py`
- `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`
- `bayesfilter/nonlinear/cut_tf.py`
- `bayesfilter/filters/sigma_points.py`
- `bayesfilter/filters/particles.py`
- `bayesfilter/testing/nonlinear_diagnostics_tf.py`

### Tests

- `tests/test_nonlinear_sigma_point_values_tf.py`
- `tests/test_nonlinear_sigma_point_scores_tf.py`
- `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`
- `tests/test_svd_cut_filter_tf.py`
- `tests/test_svd_cut_branch_diagnostics_tf.py`
- `tests/test_cut_rule_tf.py`
- `tests/test_structural_svd_sigma_point_tf.py`
- `tests/test_nonlinear_reference_oracles.py`
- `tests/test_nonlinear_benchmark_models_tf.py`
- `tests/test_hmc_nonlinear_model_b_readiness_tf.py`

### Benchmark harnesses and artifacts

- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`
- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py`
- `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.md`
- `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json`
- `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.md`
- `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json`
- `docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.md`
- `docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.json`

## Surface classification rules used in this result

- **Production TF filter surface**: public callable intended to return filter
  values or analytic scores for TensorFlow structural models.
- **Testing helper surface**: helper callable or dataclass used by tests,
  benchmarking, rule construction, sigma-point placement, or branch diagnostics,
  but not itself the promoted production filter API.
- **NumPy reference**: eager NumPy path used for semantics/reference only; not a
  compiled TensorFlow candidate under the current code.
- **Public non-filter helper export**: exported symbol in
  `bayesfilter/nonlinear/__init__.py` that is public but is not a value or score
  filter candidate.

## Current-surface matrix

| Surface | Kind | Public export | Classification rationale | Impl file | Eager | Graph | XLA | GPU | Static shape requirements | `return_filtered` implications | Branch requirements | Per-model/per-backend score branch classification | Result container | Diagnostics emitted | Current tests | Current benchmark coverage | Artifact references | Optimization candidates | Authoritative non-claims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tf_svd_sigma_point_filter` | production TF value | yes | High-level public TensorFlow value wrapper for cubature and UKF backends. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported | supported for fixed observation shape via `@tf.function(reduce_retracing=True)` parity/reuse tests | not currently certified; underlying loop uses Python `range` over static `T`, so only fixed-shape XLA is a candidate pending explicit NP4 gate | not claimed by code alone; diagnostic artifact exists only for fixed-shape Model B via testing helper harness, not this wrapper directly | requires static observation length because `_static_num_timesteps` raises if `observations.shape[0] is None`; model state/innovation/observation dims and sigma-rule dim are also effectively static | accepts `return_filtered`; when true it appends means/covariances to Python lists then stacks, so compiled-mode candidate exists only for fixed `T`; benchmark harness times `return_filtered=True`, but no current XLA claim for this path | value-only; no analytic derivative branch, regularization derivative target is `blocked` | Model A/B/C: not applicable for score branch because this surface is value-only | `TFFilterValueResult` | backend/rule, augmented dim, point count, polynomial degree, max integration rank, floor counts, PSD projection residuals, support residual, deterministic residual, min eigen gaps, implemented innovation covariance | `tests/test_nonlinear_sigma_point_values_tf.py`, `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`, plus CUT-adjacent coverage distinguishes rule-specific behavior elsewhere | CPU-only nonlinear benchmark harness uses testing helper wrapper over same backend; no direct dedicated GPU/XLA harness row for this exact wrapper | CPU benchmark rows for cubature/UKF in `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.{md,json}` | tensor-only fast path without rich container assembly; precomputed sigma rules; compiled-friendly filtered-state storage for `return_filtered=True`; avoid repeated augmented block assembly and full innovation precision materialization | not a certified XLA surface; not a GPU-ready claim; not an exact Model B/C likelihood oracle; not a score or Hessian path |
| `tf_svd_sigma_point_log_likelihood` | production TF value | yes | Public low-level TensorFlow value surface returning tuple rather than result object. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported | supported for fixed observation shape by same static-loop structure; exercised indirectly by graph parity and direct tests | not currently certified; candidate only for fixed static shapes under future NP4 proof | not claimed; no direct authoritative GPU row for this exact callable | static observation length required; backend rule chosen internally from static state+innovation dim | supports `return_filtered`; same Python-list accumulation implication as wrapper | value-only | not applicable | tuple `(log_likelihood, filtered_means, filtered_covariances, diagnostics)` | same raw diagnostics dictionary as wrapper, including implemented innovation covariance | `tests/test_nonlinear_sigma_point_values_tf.py` covers eager and graph parity on cubature path | indirectly covered by CPU benchmark through testing helper wrapper | indirect through same CPU benchmark artifact and by CUT-adjacent artifact for comparison | same as wrapper, plus fast-path option to skip dict assembly if future NP2 approves | not a certified XLA/GPU path; not broad compiled support; not a default-policy result |
| `tf_svd_sigma_point_log_likelihood_with_rule` | production TF value | yes | Public low-level TensorFlow value surface for fixed externally supplied sigma rule; still a filter surface, but more specialized. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported | supported for fixed observation shape by same structure | not currently certified; fixed-rule input makes it a plausible future compile target but no present claim | not claimed | static observation length; supplied `sigma_rule.dim` must equal `state_dim + innovation_dim`; effective point count static for compilation | supports `return_filtered`; same list/stack implication | value-only | not applicable | tuple `(log_likelihood, filtered_means, filtered_covariances, diagnostics)` | raw value diagnostics as above | covered indirectly through CUT4 wrapper and direct tests using CUT4/log-likelihood path | indirectly covered via CUT4 benchmark path because CUT4 wrapper calls this function with CUT4 rule | CUT4 benchmark and diagnostic artifacts cover the rule-driven implementation indirectly | precompute/supply reusable rule outside timed region; optional tensor-only fast path; compiled-friendly filtered-state storage | not a certified XLA/GPU surface; not broad rule-generic speed evidence |
| `tf_svd_cut4_filter` | production TF value | yes | High-level public TensorFlow CUT4 value wrapper. | `bayesfilter/nonlinear/svd_cut_tf.py` | supported | supported for fixed shape; explicit static-shape graph reuse test exists | fixed-shape XLA has diagnostic evidence only through testing-helper GPU/XLA harness on Model B and CPU control artifact; wrapper itself not yet NP4-certified | limited diagnostic evidence only: existing single-shape Model B helper-harness artifact reports GPU rows for CUT4; this is not yet a general GPU support claim for the production wrapper | static observation length via underlying implementation; CUT4 additionally requires `augmented_dim >= 3` because the rule constructor rejects smaller dims | supports `return_filtered`; same Python-list accumulation implication from underlying implementation | value-only | not applicable | `TFFilterValueResult` | same value diagnostics as sigma-point wrapper, with rule `CUT4-G`, point count `2*dim + 2**dim`, derivative target `blocked` | `tests/test_svd_cut_filter_tf.py`, `tests/test_nonlinear_sigma_point_values_tf.py`, `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py` | CPU benchmark has direct CUT4 rows; GPU/XLA diagnostic harness currently benchmarks CUT4 on fixed Model B shape; CPU XLA control also CUT4-only | `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.{md,json}`, `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.{md,json}`, `docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.{md,json}` | same as other value paths, but CUT4 also needs bounded dimension ladders because point count is exponential in augmented dim | current artifact does not certify broad CUT4 XLA or GPU support; no broad speedup claim; no score or Hessian claim |
| `tf_svd_cut4_log_likelihood` | production TF value | yes | Public low-level CUT4 value tuple-returning surface. | `bayesfilter/nonlinear/svd_cut_tf.py` | supported | supported for fixed shape; explicit graph reuse test compiles this callable | fixed-shape XLA plausible but not directly certified for this exact callable; current XLA evidence is via helper harness | not claimed directly | static observation length; CUT4 rule requires `augmented_dim >= 3` | supports `return_filtered`; same list/stack implication | value-only | not applicable | tuple `(log_likelihood, filtered_means, filtered_covariances, diagnostics)` | raw value diagnostics | `tests/test_svd_cut_filter_tf.py`, `tests/test_nonlinear_sigma_point_scores_tf.py` use this for finite-difference value oracle | indirectly benchmarked via wrapper/helper harness | same CUT4 artifacts as wrapper, indirectly | same as CUT4 wrapper, especially potential rule precomputation and container-skipping fast path | not a certified XLA/GPU path by itself; not exact Model B/C evidence |
| `tf_svd_cubature_score` | production TF score | yes | Public high-level analytic score surface for cubature rule. | `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py` | supported | supported for fixed shape; explicit graph parity test exists | not currently certified; no NP4 matrix or dedicated XLA parity test for score path yet | not claimed | requires static observation length and static parameter/state/innovation/observation dimensions; derivative tensors must match exact static shapes | not applicable; score result has no `return_filtered` | analytic branch requires no active floor and sufficient spectral separation on smooth branch; structural fixed-support branch additionally requires fixed null support conditions when enabled | Model A cubature: smooth branch passes in tests; Model B cubature: smooth branch passes in tests; Model C cubature with default zero phase variance: smooth branch blocked by active floor; Model C cubature with `allow_fixed_null_support=True`: structural fixed-support branch passes with one structural null | `TFFilterDerivativeResult` | value diagnostics plus score-branch diagnostics: factor derivative reconstruction residual, fixed-null derivative residual, structural null covariance residual/count, derivative branch, derivative method, derivative provider, hessian status deferred | `tests/test_nonlinear_sigma_point_scores_tf.py`, `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py` | CPU benchmark records branch summaries and score branch status, but not steady-state score timings as a separate promoted metric; no GPU/XLA score harness in NP0 scope | CPU benchmark artifact rows and branch summaries in `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.{md,json}` | vectorize parameter-axis `d_kalman_gain` loop; reuse eigensystem solves; separate branch-precheck artifact from steady-state timing; score-only fast path | not Hessian-ready; not XLA-certified; not GPU-certified; no broad score timing claim |
| `tf_svd_ukf_score` | production TF score | yes | Public high-level analytic score surface for unscented rule. | `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py` | supported | supported for fixed shape; indirect graph parity expectation through common engine, though explicit graph test is on cubature | not currently certified | not claimed | same static observation and derivative shape requirements as cubature score; rule parameters `alpha`, `beta`, `kappa` are Python/host inputs | not applicable | same branch requirements as cubature score | Model A UKF: smooth branch passes; Model B UKF: smooth branch passes; Model C UKF default zero phase variance: smooth branch blocked by active floor; Model C UKF with `allow_fixed_null_support=True`: structural fixed-support branch passes | `TFFilterDerivativeResult` | same analytic score diagnostics as cubature score | `tests/test_nonlinear_sigma_point_scores_tf.py`, `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py` | CPU benchmark records branch summaries and score status only | CPU benchmark artifact rows and branch summaries in `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.{md,json}` | same score optimizations as cubature path; possibly predeclare/freeze rule params for compile friendliness | not Hessian-ready; not XLA-certified; not GPU-certified; no broad score timing claim |
| `tf_svd_cut4_score` | production TF score | yes | Public high-level analytic score surface for CUT4 rule. | `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py` | supported | supported for fixed shape through common engine; direct eager coverage and branch tests exist | not currently certified; no score XLA gate yet | not claimed | static observation length and static derivative shapes; CUT4 additionally requires `augmented_dim >= 3` | not applicable | same smooth vs structural-fixed-support branch logic as other score paths, plus CUT4 point-count explosion with augmented dimension | Model A CUT4: smooth branch passes and finite-difference/autodiff oracle checks pass; Model B CUT4: smooth branch passes; Model C CUT4 default zero phase variance: smooth branch blocked by active floor; Model C CUT4 with `allow_fixed_null_support=True`: structural fixed-support branch passes; repeated-spectrum affine diagnostic can trigger weak spectral gap blocker | `TFFilterDerivativeResult` | same analytic score diagnostics as other score paths | `tests/test_nonlinear_sigma_point_scores_tf.py`, `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`, `tests/test_svd_cut_branch_diagnostics_tf.py` | CPU benchmark records value timing plus score branch summary; no promoted CUT4 score timing artifact in current NP0 scope | CPU benchmark artifact plus opt-in extended branch diagnostic tests; no NP4 score XLA artifact yet | same score-path optimizations as others, but dimension-bounded ladders are mandatory because CUT4 point count grows as `2*dim + 2**dim` | not Hessian-ready; not XLA-certified; not GPU-certified; no broad CUT4 score speed claim |
| `tf_svd_sigma_point_score_with_rule` | production TF score | yes | Public fixed-rule analytic score surface; specialized but still a score filter surface. | `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py` | supported | supported for fixed shape through same static-loop engine | not currently certified | not claimed | static observation length, static derivative shapes, supplied `sigma_rule.dim == state_dim + innovation_dim` | not applicable | smooth branch by default; structural fixed-support branch only when `allow_fixed_null_support=True` and null-support invariants hold | same backend/model logic as its callers, but branch class depends on supplied rule and model null structure rather than backend string alone | `TFFilterDerivativeResult` | same analytic score diagnostics as callers | indirectly covered through cubature/UKF/CUT4 score tests that call this engine | no direct benchmark row; behavior appears indirectly through caller backends only | indirect through CPU benchmark score branch summaries | rule reuse/precomputation; vectorized parameter-axis solves; optional branch-precheck separation | not a standalone certified XLA/GPU surface; not Hessian-ready |
| `TFSigmaPointRule` | public non-filter helper export | yes | Public dataclass defining offsets and weights; used to build rule objects, not itself a filter candidate. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported as eager helper object | graph support not applicable as a promoted runtime surface; usable as captured constant-like object in fixed-shape calls | XLA support not applicable as standalone export | GPU support not applicable as standalone export | `dim`, offsets rank, and weight lengths must be statically consistent at object creation | not applicable | not applicable | not applicable | dataclass | point-count property only; no filter diagnostics | indirectly exercised by all rule-based tests | not benchmarked separately | none | precompute rule once outside timed/compiled function | not a benchmark/default-policy/filter surface |
| `TFSigmaPointDiagnostics` | public non-filter helper export | yes | Public dataclass carrying placement diagnostics; helper-only. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported as returned helper payload | not a standalone graph surface | not applicable | not applicable | tensor shapes inherit from placement inputs | not applicable | not applicable | not applicable | dataclass | rank, floor_count, PSD projection residual, support residual, implemented covariance, eigenvalues | indirectly exercised via placement/value tests | not benchmarked separately | none | none for NP1-NP7 beyond possible avoidance of rich diagnostics on fast paths | not a filter or performance-candidate surface |
| `TFSigmaPointValueBackend` | public non-filter helper export | yes | Literal type alias naming cubature/UKF backends; exported for typing/API clarity, not a runtime surface. | `bayesfilter/nonlinear/sigma_points_tf.py` | not applicable | not applicable | not applicable | not applicable | static string literal choices only | not applicable | not applicable | not applicable | type alias | none | indirectly covered by backend dispatch tests | not benchmarked | none | none | not a callable/runtime surface |
| `TFStructuralFirstDerivatives` | public non-filter helper export | yes | Public dataclass packaging analytic derivative inputs; required by score surfaces but not itself a filter candidate. | `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py` | supported | graph/XLA as standalone object not applicable; contents must be shape-static for score paths | not applicable as standalone export | not applicable | parameter/state/innovation/observation dims must be statically known and shape-consistent | not applicable | encodes derivative availability precondition for score branches | not applicable | dataclass | shape validation, parameter/state/innovation/observation dimension properties | directly tested via score tests | not benchmarked separately | none | potential caching/reuse of derivative bundles in future benchmark harnesses | not a filter/performance surface |
| `TFSmoothEighFactorFirstDerivatives` | public non-filter helper export | yes | Public dataclass for internal/diagnostic derivative factorization outputs; helper-only. | `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py` | supported as helper payload | not a standalone promoted compiled surface | not applicable | not applicable | shapes depend on covariance and derivative tensors | not applicable | smooth or structural-fixed-support derivative branch only | not applicable | dataclass | eigenvalues, factor, derivative residuals, structural-null diagnostics | indirectly exercised by score tests | not benchmarked separately | none | none for NP0; possible internal reuse only | not a filter/performance surface |
| `tf_svd_sigma_point_placement` | public non-filter helper export | yes | Public sigma-point placement helper used by value/score implementations; exported, but not a top-level filter candidate. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported | compile candidate only as nested helper inside fixed-shape functions; no standalone graph/XLA claim | not certified | not claimed | covariance/rule dim must match; rank/support calculations assume static last-dimension structure | not applicable | placement-only; no value or score branch semantics | not applicable | tuple `(points, TFSigmaPointDiagnostics)` | placement diagnostics only | indirectly exercised through value and score tests | not benchmarked separately | none | possible rule/factor reuse and placement fast path | not a full filter or performance row |
| `tf_unit_sigma_point_rule` | public non-filter helper export | yes | Public rule constructor for cubature/unscented standardized Gaussian rules. | `bayesfilter/nonlinear/sigma_points_tf.py` | supported | usable in graph-captured setup but not a promoted filter runtime surface | not certified as standalone support claim | not applicable as standalone surface | `dim > 0`; unscented requires `alpha**2 * (dim + kappa) > 0` | not applicable | not applicable | not applicable | returns `TFSigmaPointRule` | rule metadata only | indirectly exercised by all cubature/UKF tests | not benchmarked separately | none | precompute outside compiled/timed path | not a filter/performance surface |
| `tf_cut4g_sigma_point_rule` | public non-filter helper export | yes | Public CUT4-G rule constructor; helper export, not filter surface. | `bayesfilter/nonlinear/cut_tf.py` | supported | usable as setup helper only | not certified as standalone support claim | not applicable as standalone surface | `dim >= 3`; point count fixed as `2*dim + 2**dim` | not applicable | not applicable | not applicable | returns `TFSigmaPointRule` | rule metadata only | `tests/test_cut_rule_tf.py` | not benchmarked separately, but point count drives CUT4 benchmark rows | indirect via all CUT4 artifacts | precompute once; dimension-bounded ladders because point count is exponential | not a filter or broad support claim |
| `StructuralSVDSigmaPointFilter.filter` | NumPy reference value | no (class is not in `bayesfilter/nonlinear/__init__.py`) | Eager NumPy reference backend used for semantics/reference only. Included because NP0 scope explicitly covers reference paths. | `bayesfilter/filters/sigma_points.py` | supported in eager NumPy | unsupported for TF graph | unsupported for XLA | unsupported for GPU | observation array shape checked eagerly; no TensorFlow static-shape compilation path | supports `return_filtered`; simply appends NumPy arrays and stacks in eager mode | value-only reference approximation; no analytic score branch | not applicable | `SigmaPointResult` | rule, augmented_dim, min prediction/update eigenvalue, eager_numpy metadata | `tests/test_structural_sigma_points.py`, `tests/test_nonlinear_reference_oracles.py` | no nonlinear TF benchmark coverage; reference-only | none in current nonlinear benchmark artifacts | no NP2-NP5 optimization candidate unless NP6 justifies a TensorFlow rewrite | not TF/XLA/GPU-ready; not a production TF backend; not exact nonlinear oracle beyond its own approximation |
| `particle_filter_log_likelihood` | NumPy reference value | no | Eager NumPy bootstrap particle reference path, fail-closed for audit semantics and Monte Carlo diagnostics. Included because NP0 scope explicitly covers reference paths. | `bayesfilter/filters/particles.py` | supported in eager NumPy | unsupported for TF graph | unsupported for XLA | unsupported for GPU | eager NumPy arrays only; runtime depends on particle count and observation array shape, not static compile constraints | supports `return_particles` rather than `return_filtered`; returns final particles optionally | Monte Carlo value-only; no analytic score branch | not applicable | `ParticleFilterResult` | num_particles, resample_count, min ESS, max identity residual, proposal_correction, eager_numpy metadata | `tests/test_nonlinear_reference_oracles.py` and particle-reference tests in repo scope | no current nonlinear TF benchmark coverage in NP0 artifact set | none in current nonlinear benchmark artifacts | no NP2-NP5 candidate unless NP6 identifies a concrete downstream TensorFlow need | not TF/XLA/GPU-ready; not deterministic timing baseline; not analytic score/Hessian path |

## Public export reconciliation from `bayesfilter/nonlinear/__init__.py`

The current public exports are completely classified as follows:

### Production TensorFlow filter surfaces

- `tf_svd_sigma_point_filter`
- `tf_svd_sigma_point_log_likelihood`
- `tf_svd_sigma_point_log_likelihood_with_rule`
- `tf_svd_cut4_filter`
- `tf_svd_cut4_log_likelihood`
- `tf_svd_cubature_score`
- `tf_svd_ukf_score`
- `tf_svd_cut4_score`
- `tf_svd_sigma_point_score_with_rule`

### Public non-filter helper exports

- `TFSigmaPointDiagnostics`
- `TFSigmaPointRule`
- `TFSigmaPointValueBackend`
- `tf_svd_sigma_point_placement`
- `tf_unit_sigma_point_rule`
- `tf_cut4g_sigma_point_rule`
- `TFStructuralFirstDerivatives`
- `TFSmoothEighFactorFirstDerivatives`

No exported symbol remains unclassified.

## Execution-mode support summary without unknown cells

### Production TensorFlow value paths

| Surface group | Eager | Graph | XLA | GPU | Support boundary |
| --- | --- | --- | --- | --- | --- |
| Cubature/UKF value (`tf_svd_sigma_point_filter`, `tf_svd_sigma_point_log_likelihood`, `tf_svd_sigma_point_log_likelihood_with_rule`) | supported | supported for fixed static observation shape; current evidence is eager-vs-graph parity for Model B cubature and shared static-loop code | unsupported as a certified claim today; candidate only for fixed static shapes pending NP4 | unsupported as a certified claim today | Python `range` over static `T`, raw metadata says `compiled_status="eager_tf"`, and no NP4 gate exists yet |
| CUT4 value (`tf_svd_cut4_filter`, `tf_svd_cut4_log_likelihood`) | supported | supported for fixed static observation shape; explicit static graph reuse test exists | diagnostic fixed-shape evidence exists via helper harness artifacts, but certified support is still unsupported pending NP4 | diagnostic fixed-shape evidence exists via helper harness artifacts, but certified support is still unsupported pending NP5/NP4 gates | CUT4 artifacts are single-shape diagnostics, not authoritative support matrix entries |

### Production TensorFlow score paths

| Surface group | Eager | Graph | XLA | GPU | Support boundary |
| --- | --- | --- | --- | --- | --- |
| Cubature/UKF/CUT4 score (`tf_svd_cubature_score`, `tf_svd_ukf_score`, `tf_svd_cut4_score`, `tf_svd_sigma_point_score_with_rule`) | supported | supported for fixed static observation and parameter shapes; direct graph parity shown for cubature and common engine shared by all | unsupported as a certified claim today | unsupported as a certified claim today | no current NP4 score XLA parity artifact or helper-harness matrix certifies score XLA/GPU support |

### Helper and reference paths

- All helper exports are marked **not applicable** for standalone benchmark support
  claims.
- All NumPy reference paths are **unsupported** for graph, XLA, and GPU because
  they are eager NumPy implementations, not TensorFlow backends.

## Static-shape findings

1. `bayesfilter/nonlinear/sigma_points_tf.py:113-117` requires static observation
   length for value paths.
2. `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py:132-140` and
   `:185-201` require static observation length and static parameter/state/
   innovation/observation dimensions for score paths.
3. Current TF implementations loop over time with Python `for t in range(T)` in
   value and score engines (`sigma_points_tf.py:304`,
   `svd_sigma_point_derivatives_tf.py:420`), so current compile-friendly usage
   is fixed-shape rather than dynamic-horizon.
4. `return_filtered=True` currently stores per-step states in Python lists and
   stacks them afterward in value paths (`sigma_points_tf.py:301-303`,
   `:422-427`), which is acceptable for eager/fixed-graph use but is exactly the
   kind of compiled-mode pressure point NP2 should inventory.
5. CUT4 rule creation requires `augmented_dim >= 3`
   (`bayesfilter/nonlinear/cut_tf.py:33-35`).

## Branch classification ledger for score paths

### Smooth branch

Condition:

- no active floor at placement or innovation stage;
- sufficient spectral separation above `spectral_gap_tolerance`;
- finite value and score.

Evidence:

- Model A affine score tests pass finite-difference parity for cubature, UKF,
  and CUT4.
- Model B nonlinear accumulation score tests pass finite-difference parity for
  cubature, UKF, and CUT4.
- Model C with positive `initial_phase_variance` passes finite-difference parity
  for cubature, UKF, and CUT4.

### Structural fixed-support branch

Condition:

- `allow_fixed_null_support=True`;
- structural null support has zero covariance residual;
- null derivative residual is within `fixed_null_tolerance`;
- no active floor on active coordinates.

Evidence:

- Model C with default zero phase variance passes for cubature, UKF, and CUT4
  when `allow_fixed_null_support=True`, with `structural_null_count == 1` in
  tests.

### Blocked branch

Current blocked modes evidenced by tests:

- `blocked_active_floor` when an active floor is triggered.
- `blocked_weak_spectral_gap` when spectral separation is insufficient.
- default Model C smooth-score attempt without fixed-null support is blocked by
  active floor.

## Tests inventory by surface family

### Value paths

- Exact affine Model A parity against linear-Gaussian Kalman reference:
  `tests/test_nonlinear_sigma_point_values_tf.py:20-65`
- Finite Model B/Model C value results and deterministic residual checks:
  `tests/test_nonlinear_sigma_point_values_tf.py:67-102`
- Dense one-step projection comparison for Model B CUT4 first step:
  `tests/test_nonlinear_sigma_point_values_tf.py:104-132`
- Graph parity for Model B cubature value path:
  `tests/test_nonlinear_sigma_point_values_tf.py:135-154`
- CUT4 affine value parity and static graph reuse:
  `tests/test_svd_cut_filter_tf.py:33-109`
- Shared value diagnostic vocabulary and branch summaries:
  `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py:202-294`

### Score paths

- Affine smooth-branch finite-difference parity for cubature/UKF and CUT4 plus
  CUT4 autodiff oracle score check:
  `tests/test_nonlinear_sigma_point_scores_tf.py:243-348`
- Model B finite-difference parity across all backends:
  `tests/test_nonlinear_sigma_point_scores_tf.py:351-386`
- Model C smooth-phase parity across all backends:
  `tests/test_nonlinear_sigma_point_scores_tf.py:388-429`
- Model C default smooth-branch blocker and structural fixed-support success:
  `tests/test_nonlinear_sigma_point_scores_tf.py:432-511`
- Structural-fixed-support blocker cases and diagnostics:
  `tests/test_nonlinear_sigma_point_scores_tf.py:514+`
- Shared score diagnostic vocabulary and branch summaries:
  `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py:229-444`
- Opt-in extended CUT branch-frequency diagnostics:
  `tests/test_svd_cut_branch_diagnostics_tf.py:11-106`

### Helper/reference paths

- CUT4 rule identities and no-NumPy TF implementation checks:
  `tests/test_cut_rule_tf.py:16-54`
- NumPy reference sigma-point and particle reference oracles:
  `tests/test_nonlinear_reference_oracles.py`
- Structural sigma-point eager NumPy reference coverage:
  `tests/test_structural_sigma_points.py`

## Benchmark coverage inventory

### Existing CPU-only nonlinear benchmark harness

Harness:

- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`

What it covers now:

- value timing for all three production TF value backends on Models A, B, and C;
- branch summaries for value and score paths over small parameter grids;
- exact affine Model A reference errors;
- dense one-step Model B/C explanatory-only projection errors;
- `return_filtered=True` value timing path;
- point count, polynomial degree, integration rank, and branch summary metadata.

What it does **not** cover now:

- direct steady-state score timing as a promoted row family;
- shape ladders beyond the fixed tiny/small cases embedded in the script;
- explicit `return_filtered=False` versus `True` split in artifact rows;
- authoritative XLA support claims;
- GPU claims.

### Existing GPU/XLA diagnostic harness

Harness:

- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py`

What it covers now:

- fixed-shape Model B value timing diagnostics for selected backend/device/mode
  combinations;
- current committed markdown artifact shows CUT4 rows only, with CPU and GPU,
  eager/graph/XLA;
- branch metadata paired with each row.

What it does **not** cover now:

- all value backends in committed markdown artifact;
- any score paths;
- `return_filtered=True` path;
- authoritative NP4 support matrix semantics;
- broad CPU/GPU conclusions.

## Artifact ledger relevant to NP1 and NP4 seeding

| Artifact | What it evidences | What it cannot evidence |
| --- | --- | --- |
| `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.{md,json}` | CPU-only fixed-shape value timings; affine exact reference rows; branch-summary metadata for current backend/model set | broad performance scaling, GPU support, XLA certification, exact Model B/C likelihood correctness |
| `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.{md,json}` | single-shape diagnostic CPU/GPU eager/graph/XLA rows for current helper-harness workload, with visible GPU in artifact environment | general GPU speedup, all backends unless shown in authoritative JSON, score-path support, production wrapper certification |
| `docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.{md,json}` | CPU-only control showing fixed-shape graph/XLA diagnostic behavior for current helper-harness workload | general XLA support matrix, production default decisions |

## Optimization-candidate ledger carried forward to later phases

### NP1 benchmark-harness needs surfaced by NP0

- Add explicit row metadata for `return_filtered=False` versus `True` on value
  paths.
- Separate value timing rows from score timing rows rather than only recording
  score branch status.
- Record authoritative support cells for fixed-shape graph/XLA/GPU instead of
  relying on mixed benchmark markdown interpretation.
- Expand shape ladder metadata beyond current fixed small cases.

### NP2 value-path candidates surfaced by NP0

- Tensor-only fast path that bypasses result containers and rich diagnostics for
  timed steady-state runs.
- Precompute sigma rules outside the compiled inner call.
- Replace Python-list filtered-state accumulation with compiled-friendly storage
  when `return_filtered=True`.
- Reuse block covariance structure and reduce repeated matrix assembly.
- Avoid full innovation precision materialization when only solve/vector and
  gain products are needed, subject to derivation gate.

### NP3 score-path candidates surfaced by NP0

- Vectorize `d_kalman_gain` parameter-axis loop
  (`svd_sigma_point_derivatives_tf.py:641-655`).
- Reuse eigensystem solves across derivative terms where algebraically valid.
- Add score-only fast path without nonessential diagnostics.
- Separate branch-precheck artifact from steady-state timing mode without
  removing branch assertions.

### NP4 support-gate needs surfaced by NP0

- Formalize fixed static-shape XLA support per backend and per path.
- Distinguish value support from score support.
- Distinguish helper-harness diagnostic rows from production-API claim cells.
- State whether `return_filtered=True` is supported separately from
  `return_filtered=False`.

### NP5 CPU/GPU ladder needs surfaced by NP0

- Repeat CPU/GPU comparisons with trusted/escalated GPU execution labels and
  identical shape/dtype/model metadata.
- Avoid extrapolating current single-shape Model B diagnostic rows to other
  models, backends, or score paths.

## Non-claims ledger by family

### Production TF value family

Current code and artifacts do **not** establish:

- dynamic-horizon compiled support;
- broad XLA support for cubature, UKF, or CUT4 wrappers;
- broad GPU speedups;
- exact nonlinear likelihood quality for Models B-C;
- any default backend winner.

### Production TF score family

Current code and artifacts do **not** establish:

- Hessian readiness;
- broad XLA or GPU support;
- broad score-runtime ranking across shapes;
- validity of smooth-branch claims on structural-null Model C without the fixed-
  support branch toggle.

### Helper family

Current helper exports do **not** imply benchmark candidacy or user-facing
performance-policy surfaces simply because they are public.

### NumPy reference family

Current reference paths do **not** imply:

- TensorFlow parity;
- compiled behavior;
- GPU readiness;
- production default candidacy without an NP6 justification and separate rewrite
  plan.

## Requests for follow-up edits not performed in NP0

No edit was made outside the allowed NP0 result path. If the supervisor wants
source-map indexing updated for this new NP0 result, `docs/source_map.yml` needs
an explicit follow-up edit by an authorized phase or supervisor, because NP0 was
write-bounded to this result file only.

A reset memo update may also be appropriate after NP0 completion if the program
expects an indexed phase-state checkpoint, but NP0 did not modify any reset memo
because that path was outside the allowed write scope.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| NP0 inventory complete | passed: every public nonlinear export is classified and reference paths are separated from production TF paths | passed: no NumPy path labeled XLA/GPU-ready; dense Model B/C projection rows kept explanatory-only | current XLA/GPU evidence remains diagnostic and single-shape in scope | start NP1 benchmark harness upgrade using this matrix as the current-surface baseline | no performance promotion, no default-policy change, no new correctness claim |

## Continuation decision

Continuation label: `NP0_COMPLETE_READY_FOR_NP1`

NP1 may proceed because NP0 has no unknown public-surface classification cells
and explicitly separates production TensorFlow paths, public helper exports, and
NumPy reference paths.

## Phase exit label

`NP0_COMPLETE_READY_FOR_NP1`
