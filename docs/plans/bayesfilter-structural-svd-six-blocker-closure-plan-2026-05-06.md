# Plan: structural SVD six-blocker closure

## Date

2026-05-06

## Purpose

This plan closes the six blockers recorded at the end of
`docs/plans/bayesfilter-structural-svd-final-execution-result-2026-05-06.md`.
It is narrower than the twelve-phase roadmap: it only describes the work needed
to move from the current value/adapter-ready state to model-specific promotion
and HMC-ready evidence.

## Final goal

Promote only those structural SVD filtering targets that pass the full chain:

```text
model residuals
  -> derivative/Hessian safety
  -> compiled static-shape parity
  -> HMC diagnostics
  -> provenance cleanup
```

BayesFilter should continue to own generic filtering machinery.  DSGE and
MacroFinance repositories should continue to own economic semantics and
model-specific equations.

## Current baseline

Already closed:

- BayesFilter structural metadata and toy structural filters pass.
- Exact covariance Kalman remains separately labeled and value-exact.
- MacroFinance adapter classification gates pass as value/readiness gates.
- DSGE adapter metadata gates pass for SmallNK, Rotemberg, and SGU.
- EZ remains fail-closed.
- Derivative/JIT/HMC guardrails exist and block unsupported promotion.

Remaining blockers:

1. Rotemberg second-order/pruned deterministic identity evidence.
2. SGU deterministic residual evidence for `d,k,r,riskprem`.
3. EZ timing/partition audit.
4. SVD/eigen derivative and Hessian certification.
5. Compiled static-shape backend parity.
6. HMC diagnostics and convergence evidence.

## Dependency order

```text
0. preflight and evidence freeze
1. Rotemberg residual gate
2. SGU residual gate
3. EZ timing/partition gate
4. derivative and Hessian certification
5. compiled static-shape parity
6. HMC diagnostics ladder
7. provenance and claim cleanup
```

Rotemberg, SGU, and EZ can be investigated in parallel by separate agents, but
derivative certification must wait for a target definition that has passed the
relevant model residual gate.  HMC must wait for derivative and compiled parity
for each model/backend pair.

## Execution protocol

Every blocker follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

After each blocker, update the reset memo with:

- exact files changed;
- exact commands run;
- numerical tolerances;
- pass/fail interpretation;
- whether the next blocker remains justified;
- claim label allowed after the result.

Stop and ask for direction if a proposed step would:

- change DSGE or MacroFinance economic semantics;
- infer structural roles solely from zero rows of an impact matrix;
- add artificial process noise to deterministic coordinates;
- promote first-order evidence to second-order/pruned correctness;
- certify SVD/eigen derivatives without spectral-gap stress tests;
- run HMC before value, residual, derivative, and compiled-target gates pass;
- stage unrelated PDFs, templates, generated files, or another agent's work.

## Blocker 0: preflight and evidence freeze

### Motivation

The repo often contains unrelated monograph, bibliography, PDF, and template
work.  The closure pass should isolate structural filtering evidence from
documentation cleanup.

### Implementation instructions

1. In `/home/chakwong/BayesFilter`, record:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

2. In `/home/chakwong/python`, record:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

3. Record current validated baseline:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py

cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

### Exit gate

Proceed only when the write set for the next blocker is explicit and unrelated
dirty files are excluded.

## Blocker 1: Rotemberg residual gate

### Hypothesis

Rotemberg metadata is adapter-ready, but structural nonlinear promotion is safe
only if the intended order of the model preserves
`dy_next = y_next - y_current` pointwise under the structural completion map.

### Motivation

Rotemberg is the smallest mixed DSGE target with a narrow, testable
deterministic identity.  It should be closed before SGU or EZ because it is the
lowest-dimensional model-specific residual gate.

### Implementation instructions

Work in `/home/chakwong/python`.

1. Create a focused test file, for example:

```text
tests/contracts/test_rotemberg_structural_completion_residuals.py
```

2. Build deterministic sigma-point grids over:
   - previous state;
   - current innovation;
   - representative parameter points;
   - near-boundary but valid parameter points.
3. Evaluate the model's declared `bayesfilter_deterministic_completion`.
4. Assert the identity:

```text
dy_next - (y_next - y_current) = 0
```

within a predeclared tolerance.
5. Separate first-order evidence from second-order/pruned evidence.
6. If the pruned state has multiple components, test the component that
produces `dy_next` under the same timing convention used by the filter.
7. Add a fail-closed test proving the model is not promoted when the residual
map is absent or the identity fails.

### Tests

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_rotemberg_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

BayesFilter adapter guard:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

### Audit

- Passing first-order bridge tests allow only `first_order_residual_passed`.
- Second-order/pruned promotion requires second-order/pruned residual tests.
- Failure keeps Rotemberg at `adapter_ready_only`.

### Exit gate

Rotemberg may enter derivative/JIT work only after the exact intended model
order passes residual tests.

## Blocker 2: SGU residual gate

### Hypothesis

SGU can be structurally promoted only if deterministic coordinates
`d,k,r,riskprem` satisfy model-specific residual equations pointwise under the
completion map.

### Motivation

SGU has more deterministic structure than Rotemberg.  It should follow
Rotemberg because it needs a broader residual harness and more boundary tests.

### Implementation instructions

Work in `/home/chakwong/python`.

1. Create:

```text
tests/contracts/test_sgu_structural_completion_residuals.py
```

2. Write a residual function for each deterministic coordinate:
   - `d`;
   - `k`;
   - `r`;
   - `riskprem`.
3. Generate deterministic grids over prior states, innovations, and valid
   parameters.
4. Include boundary-adjacent parameter points that remain inside determinacy
   and finite-transition regions.
5. Assert:
   - finite transition outputs;
   - finite completion outputs;
   - each residual norm below tolerance;
   - completion failure is explicit when metadata is missing.
6. Record whether residuals are first-order, second-order, or pruned-order
   evidence.

### Tests

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

### Audit

- Do not infer roles solely from `eta`.
- If any coordinate residual fails, SGU remains `adapter_ready_only`.
- If all pass, SGU may proceed to derivative/JIT gates for that exact model
  order.

### Exit gate

SGU has either `model_residuals_passed` or a named coordinate blocker.

## Blocker 3: EZ timing and partition gate

### Hypothesis

EZ can be made adapter-ready with minimal BayesFilter impact only after its
state timing and deterministic/stochastic partition are explicit.

### Motivation

EZ is currently fail-closed.  It should not inherit Rotemberg or SGU metadata
patterns without a timing audit.

### Implementation instructions

Work in `/home/chakwong/python` first.

1. Create a timing audit note:

```text
docs/plans/dsge-ez-bayesfilter-timing-partition-audit-YYYY-MM-DD.md
```

2. Inventory:
   - state names;
   - state ordering;
   - current and lagged states;
   - innovation inputs;
   - predetermined states;
   - measurement timing.
3. Classify each coordinate as stochastic, deterministic, auxiliary, or
   external under BayesFilter metadata.
4. If classification is clear, expose metadata in the EZ model and add a
   fail-closed contract test.
5. If classification is not clear, keep EZ blocked and name the missing timing
   equation or implementation hook.

### Tests

If metadata is exposed:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

BayesFilter guard:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

### Audit

- A timing audit can close `metadata_missing`.
- It does not close nonlinear residual evidence unless residual tests are also
  written.

### Exit gate

EZ is either `metadata_ready_for_residual_tests` or remains `fail_closed` with
a named timing blocker.

## Blocker 4: SVD/eigen derivative and Hessian certification

### Hypothesis

SVD/eigen derivative certification will fail near small spectral gaps unless
BayesFilter records validity regions, uses custom derivatives, or switches
factorization policy.

### Motivation

HMC needs gradients.  Value tests and "autodiff runs" do not certify SVD/eigen
gradients or Hessians, especially around repeated eigenvalues.

### Implementation instructions

Work in `/home/chakwong/BayesFilter`.

1. Create:

```text
docs/plans/bayesfilter-structural-svd-derivative-certification-plan-YYYY-MM-DD.md
tests/test_structural_svd_derivative_certification.py
```

2. Split derivative obligations:
   - parameter transform;
   - transition/completion map;
   - sigma-point construction;
   - covariance prediction;
   - observation prediction;
   - innovation covariance factorization;
   - log-likelihood update.
3. Use MathDevMCP to route labeled derivation obligations where local TeX has
   labels.
4. Use ResearchAssistant or reviewed source artifacts for Matrix Backprop/SVD
   derivative assumptions before writing literature-backed claims.
5. Add tests for:
   - finite-difference gradient agreement;
   - JVP/VJP parity;
   - Hessian symmetry;
   - small spectral-gap stress;
   - explicit failure when derivative evidence is missing.
6. Decide the backend policy:
   - allow SVD/eigen gradients only with minimum-gap telemetry;
   - use custom derivatives;
   - switch to Cholesky/QR/other factorization;
   - keep the backend value-only.

### Tests

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
pytest -q tests/test_structural_svd_derivative_certification.py
```

### Audit

- Record spectral-gap thresholds and tolerances.
- Do not certify a backend outside its tested validity region.
- Keep `finite_difference_smoke_only` unless the new tests pass.

### Exit gate

Each backend/model pair has an explicit derivative status:

- `derivative_certified`;
- `custom_derivative_required`;
- `factor_backend_switch_required`;
- `value_only_blocked`.

## Blocker 5: compiled static-shape parity

### Hypothesis

A compiled static-shape backend can match the eager BayesFilter value path on
toy structural fixtures, but this must be shown before HMC.

### Motivation

Compilation is a separate gate from mathematical correctness.  A slow or
Python-callback target is not production-ready, and compiled code can diverge
from eager reference behavior.

### Implementation instructions

Work in `/home/chakwong/BayesFilter`, with optional client tests in
`/home/chakwong/python` after generic gates pass.

1. Create:

```text
docs/plans/bayesfilter-structural-svd-compiled-parity-plan-YYYY-MM-DD.md
tests/test_compiled_structural_svd_parity.py
```

2. Choose backend policy:
   - JAX if custom JVP/VJP and static shape control are primary;
   - TensorFlow if DSGE client compatibility dominates;
   - eager-only if compiled parity is not yet economical.
3. Declare static dimensions:
   - state dimension;
   - innovation dimension;
   - observation dimension;
   - time dimension;
   - mask convention;
   - parameter dimension.
4. Compare eager and compiled:
   - log likelihood;
   - gradient;
   - Hessian-vector products or selected Hessian entries;
   - failure behavior;
   - metadata labels;
   - compile time;
   - steady-state runtime.
5. Keep `compiled_status="eager"` or `"eager_numpy"` for any uncompiled path.

### Tests

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_compiled_structural_svd_parity.py tests/test_backend_readiness.py
pytest -q
```

Optional DSGE client parity:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_bayesfilter_compiled_parity.py
```

### Audit

- Passing toy parity does not certify Rotemberg, SGU, EZ, or NAWM.
- Each promoted model/backend pair needs its own compiled parity evidence.

### Exit gate

The target model/backend pair has `compiled_value_gradient_parity_passed`, or
HMC remains blocked.

## Blocker 6: HMC diagnostics ladder

### Hypothesis

HMC diagnostics are meaningful only after value, residual, derivative, and
compiled-target gates pass; finite smoke tests should remain labeled as finite
smoke only.

### Motivation

Sampler symptoms are a late-stage diagnostic, not a target-definition tool.
Running HMC before structural and derivative gates close risks confusing model
bugs with sampler behavior.

### Implementation instructions

Run targets in this order:

1. exact LGSSM recovery;
2. nonlinear toy structural fixture;
3. structural AR/lag fixture;
4. SmallNK;
5. Rotemberg after Blockers 1, 4, and 5 pass;
6. SGU after Blockers 2, 4, and 5 pass;
7. EZ after Blockers 3, residual tests, 4, and 5 pass;
8. NAWM-scale only after smaller DSGE models pass.

For each rung, create a result note:

```text
docs/plans/bayesfilter-structural-svd-hmc-<target>-result-YYYY-MM-DD.md
```

Record:

- target and gradient finite checks;
- backend and compiled status;
- seed;
- chain count;
- warmup and sample count;
- step size and trajectory/adaptation settings;
- acceptance rate;
- divergences;
- split R-hat;
- ESS;
- posterior recovery metric;
- runtime and compile time.

### Tests and command shape

Use dedicated scripts/tests, for example:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter python \
  scripts/run_structural_svd_hmc_ladder.py --target structural-ar2 --seed 123
```

BayesFilter should also include a small deterministic test that refuses HMC
promotion when diagnostics are incomplete.

### Audit

Allowed labels:

- `finite_smoke`;
- `gradient_validated`;
- `compiled_parity_passed`;
- `hmc_diagnostics_passed`;
- `posterior_recovery_passed`.

Do not use `converged` unless multi-chain diagnostics and posterior recovery
support it.

### Exit gate

Each rung advances only after diagnostics pass.  A failed rung becomes the next
blocker with the exact failed diagnostic named.

## Provenance cleanup after all blockers

### Motivation

Release claims should match evidence.  Cleanup comes after blockers close, not
before.

### Implementation instructions

1. Update reset memos after every blocker.
2. Update `docs/source_map.yml` only for committed artifacts.
3. Update monograph text only for claims supported by tests, derivations, or
   reviewed sources.
4. Search for unsupported strong labels:

```bash
rg -n "converged|production-ready|HMC-ready|certified|structurally fixed" docs bayesfilter tests
```

5. Run final validation:

```bash
cd /home/chakwong/BayesFilter
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
```

If docs changed:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Exit gate

The claim labels in docs, source map, reset memos, tests, and results agree.

## Suggested commit order

1. Rotemberg residual harness and result.
2. SGU residual harness and result.
3. EZ timing/partition audit.
4. SVD/eigen derivative certification tests and policy.
5. Compiled static-shape parity.
6. HMC ladder rung results.
7. Final provenance and release cleanup.

## Immediate next action

Start with Blocker 1 in `/home/chakwong/python`: Rotemberg residual tests.
This is the shortest path to a real model-specific promotion decision and has
the least blast radius on BayesFilter.
