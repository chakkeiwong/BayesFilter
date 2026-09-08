# Direct-Factor SR-UKF Model Coverage and Survey Update Plan

Date: 2026-08-17  
Status: `EXECUTED_WITH_EXPLICIT_CLASSIFICATION_BOUNDARIES`  
Owner: BayesFilter  
Related implementation result: `docs/plans/bayesfilter_direct_rectangular_block_qr_srukf_execution_result_2026_08_17.md`  
Related survey: `docs/plans/bayesfilter_square_root_sigma_point_filter_literature_survey_2026_08_17.tex`

## 1. Current answer

The direct block-QR implementation has **not** been executed as an SR-UKF on
every model in the repository because several repository rows are different
contracts or lack certified adapters. The completed campaign classifies every
canonical row and executes six eligible rows. The current evidence includes
the focused primitive/model suite and the versioned campaign under
`docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817/`, covering
the QR/block algebra, rectangular singular primitives, a small factor-SR-UKF
fixture/parity suite, route guards, and eager/XLA behavior. It is not evidence
for repository-wide model coverage.

The LaTeX survey now contains generated inventory, parity, singular robustness,
limitations, and provenance sections. It compiles in two passes; layout
overfull/underfull warnings remain non-fatal documentation quality debt.

## 2. Scope and classification

“All models” means every repository-owned model for which an SR-UKF value or
value-and-score route is mathematically applicable, plus an explicit
classification row for every other model family. It does not mean forcing an
SR-UKF onto a different algorithmic contract.

The campaign must not silently treat DPF/LEDH, SGQF, principal-root NeuTra, or
third-party demo models as direct-factor SR-UKF evidence. Every inventory row
must be assigned exactly one status:

| Status | Meaning | Required action |
|---|---|---|
| `eligible_score` | Additive Gaussian model exposes the `TFFactorSRUKFModel` and derivative contract | Run direct QR values and scores, dense/reference parity, and robustness arms |
| `eligible_value_only` | Singular/rectangular support is meaningful but smooth score is not admitted | Run `TFRectangularSRUKFModel`, support/rank tests, and value-only comparisons |
| `adapter_required` | Model is SR-UKF-applicable but has no certified factor adapter | Build and test an adapter before execution evidence |
| `not_applicable_contract` | DPF, LEDH, SGQF, NeuTra-only, deterministic/non-Gaussian, or otherwise different filter contract | Record the source route and a non-applicability reason; do not claim SR-UKF coverage |
| `owner_excluded` | Explicit owner policy excludes the model | Preserve the exclusion and do not execute it |
| `historical_only` | Existing artifact is from the old principal-root/SVD or other historical route | Use only as provenance; never as current evidence |
| `blocked` | A route is potentially applicable but currently lacks required data, derivatives, or numerical admission | Record blocker and re-entry test |

The previously stated SSL-LSTM exclusion remains active. SSL-LSTM rows are
`owner_excluded` for this campaign and must not be used to claim incomplete
SR-UKF coverage.

## 3. Frozen model inventory

The first campaign step creates a repository-owned JSON/Markdown inventory
from the canonical registries rather than hand-copying names.

### 3.1 Common V2 suite

The six rows in
`experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py::EXPECTED_V2_MODEL_IDS`
must be classified:

1. `lgssm_2d_h25_rich`
2. `sv_1d_h18_rich`
3. `range_bearing_4d_h20_rich`
4. `structural_ar1_quadratic_h16`
5. `spatial_sir_j3_rk4`
6. `predator_prey_rk4`

These are a broad repository model inventory, not proof that all six already
implement the factor SR-UKF contract. The inventory must record whether each
one receives a newly certified adapter, a value-only rectangular route, or a
non-applicable classification.

### 3.2 Existing local direct-factor parity fixtures

The existing model-level fixtures in
`tests/test_factor_srukf_model_parity.py` are the first score-bearing
certification set:

- model A affine Gaussian structural oracle;
- model B nonlinear accumulation; and
- model C autonomous nonlinear growth.

These remain mandatory regression rows after the broader campaign.

### 3.3 Active NeuTra registry models

The executable cells in
`bayesfilter/testing/neutra_model_registry_tf.py` must be classified, but the
NeuTra route itself is not the SR-UKF implementation:

- `LGSSM-EXACT`;
- `PP-UKF`;
- `PP-SGQF`;
- `SIR-SGQF`; and
- `STR-UKF`.

`PP-UKF` and `STR-UKF` are candidates for direct-factor adapter certification.
`LGSSM-EXACT` is a linear/reference parity row. `PP-SGQF` and `SIR-SGQF`
must be recorded as `not_applicable_contract` unless a separate SR-UKF model
contract is deliberately added. Existing blocked cells (`SVX-SGQF`,
`KSC-UKF`, `PP-ZC`, `STR-ZC`, `SIR-ZC`, `SVX-ZC`) retain their registry
status and are not silently promoted by this plan. `SIR-UKF` remains
`owner_excluded` under the existing owner determination.

### 3.4 Other repository models

The inventory audit must search repository-owned Python model/adapters and
classify additional models, including macrofinance/DSGE and actual-SV routes,
without importing third-party or historical source as active evidence. A row
is eligible only when its transition, observation, process loading, observation
loading, and required derivative functions can be bound to the factor contract
with fixed finite data and declared support semantics.

## 4. Scientific questions and success criteria

For every `eligible_score` row, answer:

1. Does direct block-QR produce finite values, scores, filtered means, and
   factors over the declared horizon?
2. Do direct QR and the independent reference route agree on values, scores,
   filtered means, and covariance identities within predeclared float64
   tolerances?
3. Does the result remain stable under eager and XLA execution?
4. Does the direct route avoid covariance materialization in the admitted
   runtime path?
5. Do ill-conditioned and near-singular fixtures either pass with valid
   residual/solve diagnostics or fail closed at the declared pivot/support
   boundary?

For every `eligible_value_only` row, answer:

1. Are rank, support residual, pseudodeterminant likelihood, and off-support
   behavior finite and correctly classified?
2. Does the rectangular factor reproduce the direct stack Gram identity at the
   test boundary without forming that Gram matrix in the runtime path?
3. Are repeated singular values, cutoff crossings, and support changes
   explicitly marked value-only rather than reported as analytical scores?

Promotion requires no unexplained NaN/Inf, no hidden fallback, no route guard
violation, no stale baseline comparison, and complete artifact rows. A model
with a valid blocker is not counted as a failed numerical result; it is counted
as classified but not admitted.

## 5. Baselines and parity contract

Each eligible row receives a frozen baseline bundle containing:

- model and observation checksums;
- parameter values and parameter-coordinate convention;
- initial mean/factor, process factor, and observation factor;
- horizon and dtype/backend/JIT settings;
- old-route values/scores where the old route is mathematically comparable;
- an independent dense covariance-form Kalman/UKF reference for small fixtures;
- an independent finite-difference score reference on the same finite value
  program; and
- declared tolerances for value, score, mean, factor Gram identity, solve
  residual, and branch metadata.

“Same or very similar” is not an admissible result field. The artifact must
report absolute and relative errors, norms, maximum per-time error, score
relative error, and whether the comparison is exact-linear, nonlinear
factor-gauge-sensitive, or value-only.

For nonlinear models, orthogonally equivalent factors can produce different
sigma-point approximations. The plan therefore records both numerical parity
and the factor-gauge/point-rule identity; it does not force a false bitwise
match.

## 6. Execution phases

### Phase 0: Inventory and adapter gate

1. Build `docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817/model_inventory.json` from the Common V2 and NeuTra registries plus the repository model audit.
2. Require unique model IDs, source file/function anchors, contract status,
   data checksum, parameter dimension, state/observation dimension, horizon,
   and exclusion/blocker reason where applicable.
3. Add a unit test that fails if an inventory registry row is missing,
   duplicated, or unclassified.
4. Do not begin expensive runs until all rows are classified.

### Phase 1: Adapter certification

For each candidate adapter:

1. Bind the model to `TFFactorSRUKFModel` or `TFRectangularSRUKFModel`.
2. Validate static shapes, finite factors, process-noise convention, observation
   convention, parameter ordering, and derivative orientation.
3. Validate the augmented-DZ5 process-noise rule exactly once. Do not append a
   process loading that is already represented in propagated points.
4. Run the existing primitive and model contract tests before any long run.
5. Record adapter signatures and source checksums in the inventory artifact.

### Phase 2: Benign full-rank model matrix

Run every `eligible_score` model at its frozen nominal parameter/data row in:

- direct block-QR eager;
- direct block-QR XLA;
- historical/reference route where comparable;
- independent dense reference on reduced dimensions/horizons; and
- centered finite-difference score checks.

Required recorded outputs per model:

- total and per-time log likelihood;
- total and per-time score;
- filtered means and factor diagonals;
- minimum QR and conditional pivots;
- maximum stack and derivative reconstruction residuals;
- eager/XLA deltas; and
- direct-versus-reference value and score errors.

### Phase 3: Ill-conditioned and singular matrix robustness

For every eligible factor adapter, generate deterministic stress variants with
factor singular scales approximately
`1`, `1e-4`, `1e-8`, `1e-12`, `1e-14`, and `1e-15`, plus exact rank-one and
rank-zero cases where dimensions permit.

Run:

- direct QR with pivot telemetry;
- declared pivot floors (`0`, `1e-14`, and a model-appropriate stricter
  threshold);
- rectangular value-only SVD/support route;
- on-support and off-support innovations;
- repeated singular values; and
- perturbations on both sides of the rank cutoff.

The expected result is either a finite, independently validated result or an
explicit fail-closed branch. A finite output with a changed rank/support/chart
must not be admitted as a score result.

### Phase 4: Cross-model integration and regression

1. Run the complete focused suite plus all adapter/model tests.
2. Run one deterministic short horizon for every eligible model and one longer
   horizon for models that pass the short run.
3. Compare against frozen pre-block-QR artifacts where they exist. Preserve
   historical artifacts and label nonlinear factor-gauge differences.
4. Verify no model route imports SVD/eigen/Cholesky/covariance-to-factor code in
   the admitted direct-factor path.
5. Verify TensorFlow memory-growth, GPU/XLA metadata, seeds, command lines,
   hardware, and artifact paths for serious GPU runs. CPU is allowed for small
   reference/diagnostic runs only.

### Phase 5: Independent review and closeout

Use one bounded review for the generated result summary and one bounded review
for the LaTeX diff. The review questions are:

- Is every model classified exactly once and are exclusions justified?
- Do the value/score and singular-support claims match the artifacts?
- Are baseline comparisons mathematically comparable and tolerances explicit?
- Does the report avoid claiming SR-UKF evidence for SGQF/DPF/NeuTra-only or
  owner-excluded models?
- Does the LaTeX table reproduce the artifact values and statuses exactly?

## 7. Artifact contract

Use a unique output root:

```text
docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817/
```

Required files:

- `model_inventory.json`;
- `campaign_manifest.json`;
- one JSON result per model and execution arm;
- `coverage_matrix.csv` or equivalent JSON table;
- `coverage_report.md`;
- `baseline_comparison.json`;
- `singular_robustness_report.json`;
- `latex_table_payload.json`; and
- `commands_and_environment.md`.

Every result row must include model ID, status, route, source anchors,
parameter/data checksums, seed, dtype, JIT/device, values, scores if admitted,
branch metadata, pivot/support diagnostics, comparison metrics, failure class,
and nonclaims.

No artifact may overwrite an earlier run. A failed model run is retained as
evidence and classified with a repair or blocker reason.

## 8. LaTeX update

After the campaign result is frozen, update
`docs/plans/bayesfilter_square_root_sigma_point_filter_literature_survey_2026_08_17.tex`
with a new empirical section, not hand-entered claims:

1. “Repository model inventory and applicability” table: all inventory rows,
   route status, source contract, and exclusion/blocker reason.
2. “Direct block-QR parity results” table: eligible score models, horizon,
   value error, score error, mean/factor residual, minimum pivot, and eager/XLA
   status.
3. “Singular/ill-conditioned robustness” table: scale/rank case, route,
   support status, cutoff, residual, and fail-closed result.
4. A short limitations subsection stating that this is numerical/model-suite
   evidence, not proof of exact nonlinear Bayesian inference or HMC readiness.
5. An artifact provenance paragraph naming the result root, campaign manifest,
   source revision, environment, and generation command.

Compile the document twice with `pdflatex` (or the repository LaTeX command),
check that all references/citations resolve, and store the PDF/log/checksum in
the artifact root. The LaTeX tables must be generated from the frozen JSON
payload or checked mechanically against it.

## 9. Commands and budgets

Initial bounded checks:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_stack_qr_tf.py \
  tests/test_block_qr_conditional_tf.py \
  tests/test_rectangular_factor_tf.py \
  tests/test_rectangular_srukf_tf.py \
  tests/test_factor_srukf_tf.py \
  tests/test_factor_srukf_model_parity.py \
  tests/test_factor_srukf_route_guard.py \
  tests/test_srukf_backend_policy.py
```

Inventory/model campaign commands must be added to the generated manifest,
not assumed from this plan. The campaign budget is bounded by one nominal,
one eager/XLA parity, and one robustness matrix per eligible model, with a
small-horizon CPU reference arm and GPU/XLA serious arm where available.
Stop a row on nonfinite output, changed branch identity, missing derivative
contract, unsupported likelihood measure, or an unexplained baseline delta.

## 10. Completion criteria

The campaign is complete only when:

1. every canonical inventory row is classified exactly once;
2. every `eligible_score` row has nominal, parity, derivative, and robustness
   artifacts;
3. every `eligible_value_only` row has support/rank artifacts and no score
   claim;
4. every non-applicable, blocked, historical, and owner-excluded row has an
   explicit reason and source anchor;
5. all required tests and route guards pass;
6. the LaTeX survey compiles with the generated empirical tables; and
7. a final report states exactly which models were tested and which were not.

Until these conditions hold, the correct statement remains: the direct
block-QR SR-UKF is validated on focused fixtures, not on all repository
models.

## 11. Execution result

Executed command:

```text
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache CUDA_VISIBLE_DEVICES=-1 python scripts/run_direct_factor_srukf_model_coverage_20260817.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_direct_factor_srukf_model_coverage_inventory.py tests/test_block_qr_conditional_tf.py tests/test_rectangular_factor_tf.py tests/test_rectangular_srukf_tf.py tests/test_factor_srukf_tf.py tests/test_factor_srukf_model_parity.py tests/test_factor_srukf_route_guard.py tests/test_srukf_backend_policy.py
```

Result summary:

- 24 inventory rows, all uniquely classified;
- 5 `eligible_score` rows executed: `model_a_affine`,
  `model_b_nonlinear_accumulation`, `model_c_nonlinear_growth`, `PP-UKF`, and
  `STR-UKF`;
- 1 `eligible_value_only` row executed:
  `structural_ar1_quadratic_h16`;
- 4 `adapter_required` rows retained without execution;
- 5 `not_applicable_contract` rows, 6 blocked rows, 1 historical-only row,
  and 2 owner-excluded rows retained without promotion;
- 31 focused/inventory tests passed with the pre-existing HDF5 warning;
- QR stress scales from `1` through `1e-15` were finite with zero direct-stack
  reconstruction residual; exact rank-zero, rank-one, and repeated-singular
  value-only diagnostics were finite with zero Gram reconstruction residual;
- all six executed rows had eager/XLA parity attempts passing on the CPU
  diagnostic lane; and
- the survey was compiled twice and archived with PDF, log, and SHA-256
  checksum artifacts.

The campaign does **not** establish repository-wide direct-factor SR-UKF
coverage in the sense of executing every model. The remaining adapter-required
rows need explicit source-contract adapters before they can become eligible.
