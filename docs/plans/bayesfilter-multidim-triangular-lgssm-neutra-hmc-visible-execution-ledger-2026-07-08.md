# BayesFilter Multidimensional Triangular LGSSM NeuTra-HMC Visible Execution Ledger

Date: 2026-07-08

## Status

`DRAFT_LAUNCH_REVIEW_PENDING`

### 2026-07-08 - Launch Draft - IN_PROGRESS

Evidence contract:

- Question: can BayesFilter design and execute a serious multidimensional
  triangular LGSSM NeuTra-HMC estimation benchmark without overclaiming
  stationarity, identifiability, or HMC readiness?
- Baseline/comparator: MARSS constrained structure, stationary VAR
  parameterization context, local stationary/Lyapunov code, synthetic truth,
  and later reference/HMC diagnostics.
- Primary criterion: each phase writes pass/blocker artifacts and advances only
  after review.
- Veto diagnostics: unsupported identifiability claim, missing stationary
  initial law, runtime autodiff in admitted path, `jit_compile=false`, GPU
  sample generation, nonstationarity, malformed artifacts, or unsupported
  product/default/scientific claims.
- Non-claims: no broad LGSSM, product/default, nonlinear SSM, DSGE/c603, or
  scientific readiness.

Skeptical audit:

- Wrong baseline risk: source/design inventory must precede implementation.
- Proxy-promotion risk: moment checks, compile success, training loss, and
  acceptance cannot promote serious HMC readiness.
- Hidden-assumption risk: stationarity and coordinate identification are
  separate from posterior uniqueness and must be documented separately.
- Environment risk: GPU training and CPU sampling require separate gates.

Actions:

- Drafted master program, all phase subplans, visible runbook, and launch
  review bundle.

Review:

- Claude review gate was attempted through
  `~/python/claudecodex/scripts/claude_review_gate.sh`.
- The sandbox escalation reviewer denied the Claude call as external-disclosure
  risk. No workaround was attempted.
- Same-foreground Codex substitute review:
  `docs/reviews/bayesfilter-multidim-triangular-lgssm-neutra-hmc-launch-codex-substitute-review-2026-07-08.md`.
- Substitute review verdict: `VERDICT: AGREE`.

Gate status:

- `LAUNCH_REVIEW_CODEX_SUBSTITUTE_AGREE_PHASE0_READY`

Next action:

- Begin Phase 0 source and identifiability inventory only.

### 2026-07-08 - Phase 0 - PASSED

Evidence contract:

- Question: which source-anchored constrained multidimensional LGSSM target
  should BayesFilter implement first?
- Baseline/comparator: constrained MARSS form, stationary VAR literature and
  implementation context, local stationary/Lyapunov utilities.
- Primary criterion: a clear recommendation with supported and unsupported
  claims separated.
- Veto diagnostics: unsupported identifiability claim, no stationary initial
  law, hidden dense-`A` assumption, source mismatch, or treating stationarity
  as full identifiability.
- Non-claims: no implementation correctness, HMC readiness, global
  identifiability, product/default readiness, or scientific validity.

Actions:

- Read local stationary/Lyapunov implementation and tests.
- Performed bounded source/context inventory.
- Wrote Phase 0 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-result-2026-07-08.md`.

Artifacts:

- Launch substitute review:
  `docs/reviews/bayesfilter-multidim-triangular-lgssm-neutra-hmc-launch-codex-substitute-review-2026-07-08.md`.
- Phase 0 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-result-2026-07-08.md`.

Gate status:

- `PASS_PHASE0_LOWER_TRIANGULAR_FIRST`

Next action:

- Review and, if needed, patch Phase 1 model-contract subplan before executing
  Phase 1.

### 2026-07-08 - Phase 1 Subplan Review - REPAIRED

Actions:

- Same-foreground Codex substitute review of Phase 1 subplan returned
  `VERDICT: REVISE`.
- Finding: forbidden actions should explicitly state no implementation/code
  edits and no runtime/model execution.
- Patched
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-subplan-2026-07-08.md`
  to add those boundary lines.

Gate status:

- `PHASE1_SUBPLAN_REPAIRED_READY_FOR_DOCS_EXECUTION`

Next action:

- Execute Phase 1 as docs/model-contract work only.

### 2026-07-08 - Phase 1 - PASSED

Evidence contract:

- Question: does the model contract make stationarity and coordinate
  identification explicit enough for implementation?
- Baseline/comparator: Phase 0 `lower_triangular_first` result.
- Primary criterion: define lower-triangular `A`, `H=I`, diagonal `Q/R`,
  stationary `P_inf`, parameter names, transforms, priors, seeds, and
  nonclaims.
- Veto diagnostics: missing `P_inf`, free `H`, dense unconstrained latent
  similarity, unordered coordinates, ambiguous transforms, or unsupported
  identifiability claim.
- Non-claims: no implementation correctness, XLA readiness, NeuTra usefulness,
  HMC convergence, global identifiability, or product/default readiness.

Actions:

- Wrote Phase 1 result/model contract.
- Wrote compact machine-readable contract JSON.
- No code implementation, runtime/model command, data generation, NeuTra
  training, or HMC command was run.

Artifacts:

- `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-result-2026-07-08.md`
- `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_contract_v1.json`

Gate status:

- `PASS_PHASE1_MODEL_CONTRACT_LOWER_TRIANGULAR_V1`

Next action:

- Review and refresh Phase 2 synthetic-data subplan against the Phase 1
  contract before any data generation.

### 2026-07-08 - Phase 2 - PASSED

Evidence contract:

- Question: does the synthetic fixture instantiate the Phase 1 target with
  recoverability diagnostics and no hidden nonstationarity?
- Baseline/comparator: Phase 1 contract JSON/result and stationary Lyapunov
  residual.
- Primary criterion: valid data artifact with fixed truth, stationary initial
  law, hashes, and moment sanity checks.
- Veto diagnostics: contract mismatch, nonstationary `A`, invalid covariance,
  missing seed/truth/hash, weak or degenerate signal, malformed JSON.
- Non-claims: no posterior correctness, HMC readiness, NeuTra usefulness,
  global identifiability, product/default readiness, or scientific validity.

Actions:

- Patched Phase 2 subplan to explicitly check `H=I` and diagonal `Q/R`.
- Generated deterministic seed-`20260708` data and manifest JSON artifacts.
- Repaired hash field names to distinguish canonical payload hashes from file
  byte hashes.
- Validated generated JSON artifacts with `python -m json.tool`.

Artifacts:

- `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_synthetic_data_v1_seed20260708.json`
- `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_synthetic_data_v1_manifest_seed20260708.json`
- `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase2-synthetic-data-result-2026-07-08.md`

Gate status:

- `PASS_PHASE2_SYNTHETIC_FIXTURE_VALID`

Next action:

- Review and refresh Phase 3 stationary/Lyapunov implementation subplan before
  code work.

### 2026-07-08 - Phase 3 - PASSED

Evidence contract:

- Question: can BayesFilter materialize the Phase 1/2 lower-triangular LGSSM
  model and stationary covariance without hidden nonstationarity or runtime
  autodiff?
- Baseline/comparator: Phase 1 contract, Phase 2 fixture, Lyapunov equation
  residual, and existing local stationary utilities.
- Primary criterion: transform/shape/stationary residual tests pass under
  CPU-hidden local checks.
- Veto diagnostics: runtime autodiff, non-XLA-compatible operations in admitted
  route, nonfinite covariance, contract mismatch, or derivative mismatch if
  derivatives are implemented.
- Non-claims: no posterior correctness, HMC readiness, NeuTra usefulness, score
  correctness, global identifiability, product/default readiness, or
  scientific validity.

Actions:

- Patched Phase 3 subplan to inherit concrete Phase 1/2 artifacts.
- Same-foreground Codex substitute review returned `VERDICT: AGREE`.
- Added focused testing-lane helper and tests:
  - `bayesfilter/testing/multidim_triangular_lgssm_tf.py`;
  - `tests/test_multidim_triangular_lgssm_tf.py`.
- Ran CPU-hidden py_compile and focused pytest.
- Ran runtime helper source scan for forbidden autodiff markers.

Artifacts:

- `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-result-2026-07-08.md`

Gate status:

- `PASS_PHASE3_STATIONARY_MATERIALIZATION`

Next action:

- Refresh and review Phase 4 target-score/XLA compile subplan before any score
  adapter implementation.

### 2026-07-08 - Phase 4 - PASSED

Evidence contract:

- Question: does the target adapter compute the declared lower-triangular
  LGSSM log posterior and score in an XLA-compatible path?
- Baseline/comparator: Phase 1-3 contract/artifacts, finite-difference tests,
  exact Kalman likelihood.
- Primary criterion: finite value/score and compile diagnostic with
  `jit_compile=True`; score residuals within tolerance.
- Veto diagnostics: runtime `GradientTape`, `jacobian`, `batch_jacobian`,
  `jit_compile=false`, score mismatch, nonfinite values, target/signature
  mismatch.
- Non-claims: no posterior correctness, HMC readiness, NeuTra usefulness,
  global identifiability, product/default readiness, or scientific validity.

Actions:

- Patched Phase 4 subplan to require manual/analytic score authority,
  `jit_compile=True` only, and no GPU/training/HMC.
- Same-foreground Codex substitute review returned `VERDICT: AGREE`.
- Added manual first-derivative and log posterior value/score helpers to
  `bayesfilter/testing/multidim_triangular_lgssm_tf.py`.
- Extended focused tests in `tests/test_multidim_triangular_lgssm_tf.py`.
- Ran CPU-hidden py_compile and focused pytest.
- Ran CPU-hidden `jit_compile=True` compile diagnostic.

Artifacts:

- `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_phase4_value_score_compile_diagnostic_seed20260708.json`
- `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-result-2026-07-08.md`

Gate status:

- `PASS_PHASE4_VALUE_SCORE_COMPILE_DIAGNOSTIC`

Next action:

- Refresh and review Phase 5 reference-posterior subplan before any posterior
  reference or sampling work.
