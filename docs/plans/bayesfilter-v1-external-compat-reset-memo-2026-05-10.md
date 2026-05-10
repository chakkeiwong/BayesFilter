# Reset Memo: BayesFilter v1 External-compatibility Lane

## Date

2026-05-10

## Purpose

This reset memo is for the BayesFilter v1 external-compatibility lane.  It is
separate from the shared monograph reset memo and from the structural SVD
execution reset memo so that concurrent agents can work without overwriting or
mixing handoff state.

## Lane Definition

This lane owns:

- BayesFilter v1 API stabilization for filtering backends;
- external compatibility certification against MacroFinance and the DSGE
  client;
- BayesFilter-local tests, fixtures, adapters, benchmark harnesses, and
  planning artifacts;
- documentation that treats MacroFinance and DSGE as external projects until a
  later v1 integration decision.

This lane does not own:

- MacroFinance production default changes;
- MacroFinance source edits, unless explicitly authorized in a separate client
  worktree pass;
- DSGE source edits, SGU solver changes, or neural-SGU diagnostics;
- shared monograph-wide reset memo entries written by other agents;
- unrelated Windows `Zone.Identifier` sidecar files or local images.

## Current Pivot

The previous plan language emphasized a MacroFinance switch-over pilot.  The
new decision is to avoid early coupling:

- do not switch MacroFinance over to BayesFilter yet;
- treat MacroFinance as an external compatibility target;
- keep live MacroFinance checks optional and local;
- keep stable synthetic or copied fixtures inside BayesFilter for CI;
- defer the actual MacroFinance adapter/switch-over until BayesFilter v1 is
  stable.

This is the preferred direction because it lets BayesFilter stabilize its API
without forcing MacroFinance to absorb package churn, while still preserving
evidence that BayesFilter matches important MacroFinance filtering behavior.

## Baseline Status

Current BayesFilter repository state when this memo was created:

```text
main...origin/main [ahead 1]
```

The local ahead commit is:

```text
a86a51f Plan filtering gap closure boundary
```

Important working-tree observation:

- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md` has uncommitted
  changes that appear to belong to another agent's MacroFinance/DSGE execution
  notes.
- This lane-specific memo is created to avoid editing or staging those shared
  memo changes.

Unrelated untracked files to leave alone:

```text
docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md
docs/plans/templates/*:Zone.Identifier
singularity_test.png
```

## Current Planning Artifacts

Relevant existing plans and audits:

```text
docs/plans/bayesfilter-goals-status-gaps-closure-plan-2026-05-10.md
docs/plans/bayesfilter-goals-status-gaps-closure-plan-audit-2026-05-10.md
docs/plans/macrofinance-bayesfilter-switch-over-audit-2026-05-10.md
docs/plans/bayesfilter-client-switch-over-boundary-2026-05-09.md
```

These artifacts are still useful, but the next plan should revise their
emphasis:

- replace "switch-over pilot" with "external compatibility certification";
- avoid MacroFinance writes in the BayesFilter v1 lane;
- make benchmark and fixture hardening the main BayesFilter-local work;
- reserve actual client adapters for a later v1 integration lane.

## Recommended Next Plan

Create:

```text
docs/plans/bayesfilter-v1-external-compatibility-gap-closure-plan-2026-05-10.md
```

The plan should close the remaining gaps in this order:

1. define the v1 public filtering API freeze criteria;
2. convert MacroFinance live checks into external compatibility certification;
3. add stable BayesFilter-local fixtures for dense QR, masked QR, QR
   score/Hessian, and SVD/eigen value behavior;
4. keep live MacroFinance checks optional and clearly labeled;
5. inventory DSGE nonlinear targets read-only and select non-SGU compatibility
   fixtures;
6. build CPU benchmark harnesses for representative BayesFilter-local shapes;
7. run escalated GPU/XLA-GPU benchmarks only after CPU shapes are fixed;
8. gate HMC claims on exact target-model derivative branch diagnostics;
9. defer linear SVD/eigen derivatives unless external compatibility evidence
   proves a real need;
10. write a future v1 integration plan for MacroFinance and DSGE adapters.

## Stop Rules

Stop and ask for direction if:

- the next action requires editing MacroFinance or DSGE source;
- another agent has modified the same BayesFilter plan or reset memo;
- a proposed change would add MacroFinance or DSGE as a BayesFilter production
  dependency;
- benchmark or GPU claims would require escalated commands and no permission is
  available;
- a compatibility test can pass only by changing the external project.

## Handoff Summary

Use this file, not the shared monograph reset memo, for the BayesFilter v1
external-compatibility lane going forward.

The immediate next action is to write the pivoted external-compatibility
gap-closure plan under `docs/plans`, register it in `docs/source_map.yml`, and
leave the shared reset memo and other agents' uncommitted work untouched.

## 2026-05-10 execution: pivoted external-compatibility gap closure

User request:
- create a plan given the pivot away from immediate MacroFinance switch-over;
- audit that plan as another developer;
- execute each phase using
  `plan -> execute -> test -> audit -> tidy -> update reset memo`;
- continue automatically only if primary criteria and veto diagnostics allow;
- commit scoped modified files after the plan finishes;
- summarize results and next hypotheses.

Plan under execution:

```text
docs/plans/bayesfilter-v1-external-compatibility-gap-closure-plan-2026-05-10.md
```

Audit artifact:

```text
docs/plans/bayesfilter-v1-external-compatibility-gap-closure-plan-audit-2026-05-10.md
```

Initial status:
- Current HEAD at start: `a86a51f`.
- Local branch was ahead of `origin/main` by one commit and behind by zero.
- Shared monograph reset memo had unrelated uncommitted changes and was not
  touched by this lane.

### Phase 1: lane isolation and pivot record

Plan:
- keep this lane in its own reset memo;
- record that MacroFinance switch-over is deferred until BayesFilter v1;
- register the lane in `docs/source_map.yml`;
- avoid staging shared monograph memo changes.

Execute:
- Created this reset memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

- Updated `docs/source_map.yml` to register it.

Tests:
- Documentation-only phase.  No code tests required.

Audit:
- Phase 1 passes.
- The shared monograph reset memo remains outside this lane.
- No MacroFinance or DSGE files were edited.

Tidy:
- Out-of-lane files remain unstaged.

Next phase justified?
- Yes.  Proceed to Phase 2: v1 API freeze criteria.

### Phase 2: v1 API freeze criteria

Plan:
- document stable public v1 candidates and internal/testing-only surfaces;
- gate stable symbols on tests, diagnostics, dtype behavior, and independence
  from client projects.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-api-freeze-criteria-2026-05-10.md
```

Tests:
- Documentation-only phase.  Public API import testing is recommended as a
  future implementation but not required for this planning pass.

Audit:
- Phase 2 passes.
- The artifact does not promote SVD/eigen derivatives, GPU readiness, HMC
  readiness, or client-specific economics.

Tidy:
- No production code changed.

Next phase justified?
- Yes.  Proceed to Phase 3: MacroFinance external compatibility matrix.

### Phase 3: MacroFinance external compatibility matrix

Plan:
- replace switch-over language with external compatibility certification;
- classify local CI, optional live external checks, deferred v1 integration,
  and blocked claims.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-external-compatibility-matrix-2026-05-10.md
```

Tests:
- Documentation-only phase.  No code tests required.

Audit:
- Phase 3 passes.
- The matrix does not imply MacroFinance default changes.
- Optional live MacroFinance checks remain optional.

Tidy:
- No MacroFinance files were edited.

Next phase justified?
- Yes.  Proceed to Phase 4: stable local fixture gap audit.

### Phase 4: stable local fixture gap audit

Plan:
- inspect existing BayesFilter-local tests and optional live external tests;
- determine whether missing local fixtures block v1 planning.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-local-fixture-gap-audit-2026-05-10.md
```

- Inspected existing tests including:
  - `tests/test_linear_kalman_qr_tf.py`;
  - `tests/test_linear_kalman_qr_derivatives_tf.py`;
  - `tests/test_linear_kalman_svd_tf.py`;
  - `tests/test_compiled_filter_parity_tf.py`;
  - `tests/test_macrofinance_linear_compat_tf.py`.

Tests:
- Focused BayesFilter-local tests are run after Phase 10 as the consolidated
  test gate for this documentation pass.

Audit:
- Phase 4 passes.
- Local fixture coverage is sufficient for v1 planning.
- Future improvement recommended: add `tests/test_v1_public_api.py`.

Tidy:
- No test files changed.

Next phase justified?
- Yes.  Proceed to Phase 5: optional live external test policy.

### Phase 5: optional live external test policy

Plan:
- define local CI, optional live external checks, and deferred integration
  tests;
- make skip semantics explicit.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-optional-external-test-policy-2026-05-10.md
```

Tests:
- Documentation-only phase.  Optional live external checks are not required in
  this pass.

Audit:
- Phase 5 passes.
- Missing MacroFinance checkout is documented as an optional-test skip, not a
  BayesFilter failure.

Tidy:
- No external project edited.

Next phase justified?
- Yes.  Proceed to Phase 6: DSGE read-only target inventory plan.

### Phase 6: DSGE read-only target inventory plan

Plan:
- define future read-only DSGE target inventory;
- keep SGU production filtering blocked until causal locality passes;
- avoid editing `/home/chakwong/python`.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-plan-2026-05-10.md
```

Tests:
- Documentation-only phase.  No DSGE tests run.

Audit:
- Phase 6 passes.
- DSGE economics remain external.
- SGU remains blocked as a production filtering target.

Tidy:
- `/home/chakwong/python` was not edited.

Next phase justified?
- Yes.  Proceed to Phase 7: CPU benchmark harness plan.

### Phase 7: CPU benchmark harness plan

Plan:
- define benchmark metadata, backend groups, and CPU shape ladder;
- avoid claiming benchmark results before running benchmarks.

Execute:
- Added benchmark and GPU gate artifact:

```text
docs/plans/bayesfilter-v1-benchmark-and-gpu-gates-2026-05-10.md
```

Tests:
- Documentation-only phase.  Benchmarks not run.

Audit:
- Phase 7 passes.
- Compile time, point count, and shape metadata are required by the plan.

Tidy:
- No benchmark scripts changed in this pass.

Next phase justified?
- Yes, as documentation only.  Proceed to Phase 8: GPU/XLA-GPU gate plan.

### Phase 8: escalated GPU/XLA-GPU gate plan

Plan:
- document escalated GPU probe requirements;
- prevent non-escalated sandbox failures from being treated as driver evidence.

Execute:
- Included GPU/XLA-GPU gate in:

```text
docs/plans/bayesfilter-v1-benchmark-and-gpu-gates-2026-05-10.md
```

Tests:
- No GPU commands run in this pass.

Audit:
- Phase 8 passes as policy documentation.
- It matches the repository GPU sandbox policy.

Tidy:
- No device state changed.

Next phase justified?
- Yes, as documentation only.  Proceed to Phase 9: HMC readiness gate plan.

### Phase 9: HMC readiness gate plan

Plan:
- define target-specific HMC evidence requirements;
- keep generic derivative tests separate from sampler readiness.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-hmc-and-integration-gates-2026-05-10.md
```

Tests:
- No HMC tests run in this pass.

Audit:
- Phase 9 passes.
- HMC readiness remains blocked until exact model/backend evidence exists.

Tidy:
- No sampler code changed.

Next phase justified?
- Yes.  Proceed to Phase 10: future v1 integration decision plan.

### Phase 10: future v1 integration decision plan

Plan:
- define future MacroFinance/DSGE integration checklist;
- keep actual adapter work out of this lane.

Execute:
- Included future integration checklist in:

```text
docs/plans/bayesfilter-v1-hmc-and-integration-gates-2026-05-10.md
```

Tests:
- No client integration tests run in this pass.

Audit:
- Phase 10 passes.
- The checklist separates compatibility certification from client switch-over.

Tidy:
- MacroFinance and DSGE source trees were not edited.

Next phase justified?
- Yes, but only as future implementation:
  - add a small v1 public API import test;
  - add CPU benchmark scripts/artifacts;
  - optionally run live MacroFinance compatibility as a separate evidence
    check;
  - do not switch MacroFinance or DSGE until v1 integration criteria pass.

Consolidated local test gate:
- Ran deliberate CPU-only BayesFilter-local v1 compatibility subset:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

- Observed:

```text
29 passed, 2 warnings in 86.01s
```

- The warnings were the known TensorFlow Probability `distutils`
  deprecation warnings.

Final interpretation:
- The pivoted plan has been executed as a BayesFilter-local documentation and
  certification pass.
- BayesFilter-local QR, masked QR, QR derivative, SVD/eigen value, and CPU
  compiled parity tests pass for the selected v1 compatibility subset.
- No MacroFinance switch-over was performed.
- No DSGE work was performed.
- No GPU or HMC claims were made.

Next justified work:
- add `tests/test_v1_public_api.py` to import the declared public v1 API
  surface;
- build CPU benchmark scripts and benchmark artifacts;
- optionally run live MacroFinance compatibility as a separate evidence check;
- create future v1 integration readiness plans only after API and benchmark
  gates pass.
