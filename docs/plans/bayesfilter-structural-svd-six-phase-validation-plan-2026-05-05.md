# Plan: structural SVD six-phase validation closure

## Date

2026-05-05

## Purpose

This plan turns the untracked handoff
`docs/plans/bayesfilter-structural-svd-code-implementation-handoff-plan-2026-05-05.md`
into an executable six-phase BayesFilter validation pass.

The handoff is directionally correct: BayesFilter should own reusable
structural filtering machinery, exact LGSSM filtering must remain separate
from nonlinear sigma-point approximation, and HMC promotion must wait for
value, derivative, JIT, and sampler diagnostics.  It is also partially stale:
BayesFilter already contains `StatePartition`, `StructuralFilterConfig`,
`FilterRunMetadata`, `StructuralSVDSigmaPointFilter`, exact degenerate Kalman
tests, structural particle semantics, factor-backend gates, and spectral
derivative policy gates.

Therefore this pass is not a broad rewrite.  It is a validation and
claim-reconciliation pass whose final state should say exactly what is closed,
what is adapter-ready, and what remains blocked.

## Final goals

1. Validate the BayesFilter structural core for toy structural models.
2. Validate that exact collapsed LGSSM/Kalman remains separately labeled from
   nonlinear structural sigma-point approximation.
3. Validate DSGE metadata integration using the current `/home/chakwong/python`
   structural metadata commit.
4. Reconcile blocker labels so completed gates are not left marked as missing.
5. Preserve conservative downstream blockers for model-specific Rotemberg/SGU
   residual evidence, EZ timing, SVD/eigen derivatives, JIT, and HMC.
6. Commit only scoped BayesFilter docs and any small BayesFilter-local fixes
   required by validation.

## Dependency path

```text
V0 hygiene and handoff reconciliation
  -> V1 BayesFilter structural core validation
    -> V2 DSGE metadata adapter validation
      -> V3 blocker/register reconciliation
        -> V4 final docs/tests/source-map validation
          -> V5 commit and next-hypothesis handoff
```

Do not run DSGE structural particle or HMC promotion phases until V2 and V3
pass.  Do not run HMC until derivative/JIT gates pass.

## Phase V0: hygiene and handoff reconciliation

### Motivation

The repo contains untracked inputs: the structural SVD handoff itself, a Julier
PDF, and `docs/plans/templates/`.  A validation pass must not accidentally
stage unrelated assets or treat the handoff's stale status notes as current
truth.

### Implementation instructions

1. Record `git status --short --branch`.
2. Read the handoff plan and classify each request as:
   - already implemented and needing validation;
   - still open and BayesFilter-local;
   - still open but client-owned;
   - stale or unsupported.
3. Do not stage the Julier PDF or templates unless explicitly requested.
4. Record the classification in the reset memo.

### Tests

- `git status --short --branch`
- `git log -5 --oneline --decorate`

### Audit

- Stop if tracked BayesFilter code files are dirty before validation.
- Continue if only untracked docs/assets are present and intentionally left
  unstaged.

### Exit gate

- V1 is justified only after the validation scope is clear and no unrelated
  tracked code file is in the write set.

## Phase V1: BayesFilter structural core validation

### Motivation

The handoff asks for structural SVD backend work, but much of that code already
exists.  The right action is to validate current behavior first and patch only
if tests expose a real gap.

### Implementation instructions

1. Validate structural partition and configuration gates.
2. Validate structural sigma-point likelihood, deterministic completion, and
   metadata.
3. Validate structural AR(p) fixtures.
4. Validate particle metadata only as already-scoped structural evidence; do
   not promote differentiable PF/PMCMC.

### Tests

```bash
pytest -q \
  tests/test_structural_partition.py \
  tests/test_structural_sigma_points.py \
  tests/test_structural_ar_p.py \
  tests/test_filter_metadata.py \
  tests/test_structural_particles.py
```

### Audit

- If structural sigma-point residual or metadata tests fail, fix
  BayesFilter-local code before proceeding.
- If tests pass, do not rewrite the backend.

### Exit gate

- V2 is justified only if BayesFilter toy structural filters pass value,
  residual, approximation-label, and metadata checks.

## Phase V2: exact Kalman and DSGE metadata validation

### Motivation

Exact collapsed LGSSM filtering and nonlinear structural sigma-point filtering
must remain separate.  DSGE metadata integration also needs verification
against the client repo before model-specific promotion work can be planned.

### Implementation instructions

1. Run exact/degenerate Kalman and derivative-smoke tests.
2. Run the BayesFilter DSGE gate tests.
3. Run the focused `/home/chakwong/python` structural metadata contract test
   with BayesFilter on `PYTHONPATH`.
4. Run a read-only gate probe that classifies SmallNK, Rotemberg, SGU, and EZ.

### Tests

```bash
pytest -q tests/test_degenerate_kalman.py tests/test_derivative_validation_smoke.py tests/test_dsge_adapter_gate.py
cd /home/chakwong/python && PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q tests/contracts/test_structural_dsge_partition.py
```

Expected read-only gate classifications:

- SmallNK: adapter-ready, `metadata_regime="all_stochastic"`;
- Rotemberg: adapter-ready, `metadata_regime="mixed_structural"`;
- SGU: adapter-ready, `metadata_regime="mixed_structural"`;
- EZ: fail-closed until timing audit.

### Audit

- Passing metadata gates are adapter-readiness evidence only.
- Rotemberg and SGU nonlinear structural filtering remain blocked until
  model-specific residual evidence exists.
- SVD/eigen derivative and HMC claims remain blocked.

### Exit gate

- V3 is justified only if exact Kalman tests and DSGE metadata tests pass, or
  if any failures are clearly client-owned and recorded as blockers.

## Phase V3: blocker/register and source-map reconciliation

### Motivation

The blocker register currently predates the DSGE metadata commit and may still
say metadata is missing.  The final state must distinguish adapter-ready from
structurally promoted.

### Implementation instructions

1. Update `docs/plans/bayesfilter-original-plan-remaining-gap-blocker-register-2026-05-05.md`.
2. Move the DSGE adapter row from `blocked_pending_client_metadata` to
   `client_metadata_ready_for_structural_tests` if V2 passes.
3. Keep explicit blockers for:
   - Rotemberg second-order/pruned identity evidence;
   - SGU residuals for deterministic states;
   - EZ timing/partition audit;
   - SVD/eigen derivative/JIT/HMC promotion;
   - MacroFinance final-provider evidence.
4. Register this plan and audit in `docs/source_map.yml`.
5. Update reset memo with V0-V3 results.

### Tests

- Source-map YAML parse.
- `git diff --check`.
- Stale-claim search:

```bash
rg -n "converged|production-ready|sampler-usable|blocked_pending_client_metadata|client_metadata_ready_for_structural_tests|not_claimed" docs/plans docs/chapters bayesfilter tests
```

### Audit

- The register must not imply Rotemberg/SGU structural nonlinear correctness
  from first-order bridge tests.
- The register must not imply HMC convergence from target or diagnostic gates.

### Exit gate

- V4 is justified only if the docs and labels agree with the validation
  evidence.

## Phase V4: final validation

### Motivation

The pass should leave the repo in a state where code tests, docs, source map,
and reset memo agree.

### Implementation instructions

1. Run focused tests from V1 and V2 as a single command.
2. Run full `pytest -q` if focused tests pass.
3. Run source-map YAML parse and `git diff --check`.
4. Run LaTeX no-op/full build if available.

### Tests

```bash
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Audit

- Treat TensorFlow Probability deprecation warnings as warnings only if tests
  pass.
- Treat client-repo environment warnings as non-blocking only when adapter
  assertions pass.

### Exit gate

- V5 is justified only if validation passes or blockers are explicitly recorded
  and no dependent phase requires them to be resolved.

## Phase V5: commit and next hypotheses

### Motivation

The final artifact must be an auditable commit plus a clear next-phase
hypothesis list.

### Implementation instructions

1. Stage only scoped docs/code hunks.
2. Do not stage:
   - Julier PDF;
   - `docs/plans/templates/`;
   - unrelated generated files.
3. Commit with a message describing validation and reconciliation.
4. Final summary must include:
   - tests run;
   - results;
   - what is closed;
   - what remains blocked;
   - explicit next hypotheses.

### Tests

- `git diff --cached --stat`
- `git status --short --branch`

### Audit

- The commit must not include unrelated untracked docs/assets.

### Exit gate

- The phase is complete when the scoped commit exists and the reset memo records
  final results.
