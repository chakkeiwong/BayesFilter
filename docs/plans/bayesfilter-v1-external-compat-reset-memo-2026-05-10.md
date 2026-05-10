# Reset Memo: BayesFilter v1 External-compatibility Lane

## Date

2026-05-10

## Purpose

This reset memo is for the BayesFilter v1 external-compatibility lane.  It is
separate from the shared monograph reset memo and from the structural SVD
execution reset memo so that concurrent agents can work without overwriting or
mixing handoff state.

## Two-agent Coordination Boundary

There are two active agents in this repository.  This memo is the reset memo
for only this agent's lane:

```text
BayesFilter v1 external-compatibility lane
```

This lane must keep its handoff state here:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Do not use these files as this lane's reset memo:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md
docs/plans/bayesfilter-structural-sgu-goals-gaps-next-plan-2026-05-08.md
docs/chapters/ch18b_structural_deterministic_dynamics.tex
```

Those files are either shared monograph state, structural SVD/SGU lane state,
or chapter content that another agent may be using.

Files currently owned by this lane are limited to:

```text
docs/plans/bayesfilter-v1-*.md
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.*
tests/test_v1_public_api.py
docs/source_map.yml entries whose keys begin with bayesfilter_v1_
```

Staging rule:
- stage only the owned files above for this lane;
- do not stage shared reset memo changes unless the user explicitly asks this
  lane to take ownership of that memo;
- do not stage unrelated untracked sidecars such as `Zone.Identifier` files or
  local images;
- do not stage MacroFinance or DSGE worktree changes from this lane.

If future work needs to edit structural SVD/SGU plans, Chapter 18/18b, the
shared monograph reset memo, MacroFinance source, or DSGE source, stop and ask
for a lane-boundary decision before proceeding.

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

## 2026-05-10 update: DSGE claim audit and inventory result

Trigger:
- the user asked to double-check another agent's claim that the DSGE work was
  done and to resume BayesFilter work.

Plan:
- verify the DSGE claim as a read-only external inventory;
- distinguish a narrow SGU timing fix from production SGU filtering readiness;
- decide whether Rotemberg or EZ now justifies BayesFilter adapter
  implementation.

Execute:
- Added the result artifact:

```text
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-2026-05-10.md
```

- Updated the external compatibility matrix and the DSGE inventory plan to
  point at the result.

Tests:
- Ran deliberate CPU-only focused DSGE contracts:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_marginal_utility_timing.py \
  tests/contracts/test_sgu_causal_control_anchor_gate.py \
  tests/contracts/test_sgu_current_control_derivation_gate.py \
  tests/contracts/test_structural_dsge_partition.py::test_rotemberg_exposes_mixed_metadata_and_completion \
  tests/contracts/test_structural_dsge_partition.py::test_ez_exposes_all_stochastic_metadata_after_timing_audit \
  tests/contracts/test_structural_dsge_partition.py::test_ez_metadata_records_stability_policy_without_bk_claim \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_rotemberg_second_order_dy_completion_closes_identity_residual \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_rotemberg_second_order_dy_completion_fails_closed_when_singular \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_second_order_filter_target_is_blocked_by_foc_residual_order \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_state_identity_gate_does_not_make_quadratic_policy_better_than_linear \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_quadratic_gate_failure_is_volatility_correction_driven \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_joint_state_control_projection_is_new_target_not_gate_b \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_ez_timing_audit_exposes_all_stochastic_metadata
```

Observed:

```text
16 passed, 3 warnings in 14.08s
```

Audit:
- The other agent's SGU claim is valid only for:

```text
sgu_marginal_utility_timing_contract_passed
```

- SGU still does not earn:

```text
sgu_causal_filtering_target_passed
sgu_second_order_perturbation_filter_target_passed
```

- Rotemberg has a DSGE-owned second-order `dy` completion and is the best
  future optional live compatibility candidate.
- EZ has all-stochastic metadata and local analytical stability evidence, but
  no BK/QZ determinacy certificate and no HMC readiness.

Tidy:
- No DSGE source files were edited.
- No MacroFinance files were edited in this audit update.
- The shared monograph reset memo remains out of scope for this v1 lane.

Next phase justified?
- Yes, but as BayesFilter-local v1 hardening, not as DSGE adapter
  implementation:
  - add the v1 public API import test;
  - build CPU benchmark artifacts;
  - keep optional live Rotemberg/EZ bridge work test-only and separate;
  - keep SGU blocked until DSGE supplies a causal local target.

## 2026-05-10 update: v1 public API import gate

Plan:
- add the smallest BayesFilter-local test that checks the declared v1 freeze
  symbols are top-level importable;
- ensure importing `bayesfilter` does not import external client packages.

Execute:
- Added:

```text
tests/test_v1_public_api.py
```

Expected test:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Observed:

```text
2 passed, 2 warnings in 3.78s
```

Interpretation:
- Passing this gate means the v1 freeze-note candidates are importable from the
  top-level package.
- It does not freeze argument signatures, benchmark performance, GPU behavior,
  client switch-over, or HMC readiness.

Consolidated local compatibility subset:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Observed:

```text
31 passed, 2 warnings in 83.38s (0:01:23)
```

Next phase justified?
- Yes.  The next BayesFilter-local hardening phase should create CPU benchmark
  scripts/artifacts for the v1 matrix shapes.  MacroFinance/DSGE integration,
  GPU/XLA-GPU claims, and HMC readiness remain separate gated phases.

## 2026-05-10 update: v1 CPU benchmark harness and smoke artifact

Plan:
- add a BayesFilter-local CPU benchmark harness for the v1 filtering candidates;
- record backend names, dtype, shapes, point counts, first-call timing, and
  steady-call timing;
- keep the artifact explicitly outside client-readiness, GPU, and HMC claims.

Execute:
- Added:

```text
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.md
```

Tests/benchmarks:
- First smoke run exposed a benchmark-harness issue: graph-mode runners cannot
  return BayesFilter result dataclasses from `tf.function`.
- Fixed the harness so graph-mode runners return scalar log-likelihood tensors.
- Recorded a deliberate CPU-only artifact with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 2 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2 \
  --output docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json
```

Observed:
- all benchmark rows completed with `status = ok`;
- covered `linear_qr_value`, `linear_qr_score_hessian`,
  `linear_svd_value`, `svd_cubature_value`, `svd_ukf_value`, and
  `svd_cut4_value`;
- eager and graph modes were both recorded;
- point counts were recorded for cubature, UKF, and CUT4.

Audit:
- The artifact records `CUDA_VISIBLE_DEVICES=-1` and CPU logical device only.
- TensorFlow emitted CUDA plugin-registration and `cuInit` messages before
  settling on CPU.  Because this was an intentional CPU-hidden run, those
  messages are not GPU evidence.
- The benchmark remains smoke-scale and does not justify client-scale
  performance claims.

Next phase justified?
- Yes.  Add medium-shape benchmark artifacts and memory metadata before any
  performance claim.  GPU/XLA-GPU benchmarks still require escalated device
  probes and matching shapes.

## 2026-05-10 update: phase-completion goals, gaps, and hypotheses plan

Trigger:
- the user asked for a summary of this phase's goals, remaining gaps,
  hypotheses for those gaps, and a plan to complete the phase.

Plan artifact:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-2026-05-10.md
```

Execute:
- Created/updated the phase-completion plan with:
  - an executive summary of the BayesFilter-local v1 evidence bundle;
  - explicit phase goals under the external-compatibility pivot;
  - ten remaining gaps with closure targets;
  - a gap-hypothesis matrix mapping each gap to a test or evidence artifact;
  - an eight-phase execution path:
    `lane audit -> API gate -> local regression gate -> benchmark hardening ->
    optional external decision -> DSGE containment -> GPU/HMC/SVD blocker
    review -> commit and handoff`.
- Registered the plan in `docs/source_map.yml` under a `bayesfilter_v1_` key.

Interpretation:
- The phase remains BayesFilter-local and does not switch MacroFinance or DSGE
  over to BayesFilter.
- MacroFinance and DSGE remain external compatibility targets.
- GPU/XLA-GPU, HMC, and linear SVD/eigen derivative claims remain blocked until
  separate evidence exists.
- The next execution pass should start with the lane boundary audit and should
  stage only lane-owned files.

Next phase justified?
- Yes.  The next pass can execute the plan if requested.  It should not touch
  the shared monograph reset memo or client repositories unless the user opens a
  separate lane for that work.

## 2026-05-11 execution: v1 phase-completion plan

User request:
- update the lane-specific reset memo;
- audit the plan as another developer;
- execute every phase using
  `plan -> execute -> test -> audit -> tidy -> update reset memo`;
- continue automatically only if primary criteria and veto diagnostics allow;
- commit lane-owned files after the plan finishes;
- provide detailed results and next hypotheses.

Plan executed:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-2026-05-10.md
```

Independent audit:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-audit-2026-05-11.md
```

Execution result:

```text
docs/plans/bayesfilter-v1-phase-completion-result-2026-05-11.md
```

### Phase A: lane boundary and plan audit

Plan:
- confirm the lane boundary;
- review the phase-completion plan as another developer;
- proceed only if no lane-boundary veto appears.

Execute:
- Confirmed this lane owns `docs/plans/bayesfilter-v1-*.md`, v1 benchmark
  artifacts, `tests/test_v1_public_api.py`, and `bayesfilter_v1_` source-map
  entries.
- Created the independent audit artifact above.

Audit:
- Plan approved for scoped execution.
- Required strengthening: Phase D must add benchmark memory metadata before
  closing the benchmark gap.

Tidy:
- Shared monograph reset memo remains out of scope and must not be staged by
  this lane.

Next phase justified?
- Yes.  No MacroFinance/DSGE edit or shared-memo edit was required.

### Phase B: public API import gate

Plan:
- run the top-level v1 public API import test under CPU-only settings.

Observed:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_v1_public_api.py -p no:cacheprovider
2 passed, 2 warnings in 3.52s
```

Interpretation:
- Declared v1 public symbols are top-level importable.
- Importing `bayesfilter` does not import MacroFinance or DSGE modules.
- Warnings are TensorFlow Probability `distutils` deprecation warnings.

Next phase justified?
- Yes.  API gate passed.

### Phase C: local compatibility regression gate

Plan:
- run focused BayesFilter-local QR/SVD/compiled-parity tests under CPU-only
  settings.

Observed:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
31 passed, 2 warnings in 84.50s (0:01:24)
```

Interpretation:
- Local v1 API, QR value, QR score/Hessian, SVD value, and compiled parity
  evidence passes without external project checkouts.

Next phase justified?
- Yes.  Benchmark hardening is justified.

### Phase D: benchmark harness hardening

Plan:
- add process-memory metadata;
- regenerate the small CPU smoke artifact;
- add one medium-shape CPU artifact if runtime remains reasonable.

Execute:
- Updated:

```text
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
```

- Regenerated:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.md
```

- Added:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.md
```

Observed:
- smoke benchmark: all rows `status = ok`;
- medium benchmark: all rows `status = ok`;
- all rows record timing, shape, point-count, RSS, and high-water RSS metadata.

Key interpretation:
- Medium QR score/Hessian first eager call took about 57.6 seconds and raised
  high-water RSS by about 3335.8 MB.
- Medium QR score/Hessian graph first call took about 11.1 seconds and raised
  high-water RSS by about 486.3 MB.
- Steady calls were much faster, but first-call tracing/compile and memory
  costs are material.

Audit:
- Benchmark artifacts are CPU-only.  TensorFlow emitted CUDA initialization
  messages despite `CUDA_VISIBLE_DEVICES=-1`; logical devices were CPU-only, so
  these are not GPU evidence.
- Memory metadata is process-level diagnostic metadata, not isolated allocation
  profiling.

Next phase justified?
- Yes.  Benchmark metadata and medium-shape evidence now exist.  GPU remains a
  separate escalated-device phase.

### Phase E: optional external check decision

Plan:
- decide whether to run optional live MacroFinance in this pass.

Decision:

```text
optional_live_macrofinance_status = not_run_by_policy
```

Interpretation:
- This lane intentionally avoids client coupling.
- `tests/test_macrofinance_linear_compat_tf.py` remains available as optional
  live evidence but was not run in this completion pass.

Next phase justified?
- Yes.  Optional external status is explicit.

### Phase F: DSGE candidate containment

Plan:
- keep DSGE evidence read-only;
- do not implement Rotemberg/EZ bridges;
- keep SGU blocked.

Evidence:

```text
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-2026-05-10.md
```

Interpretation:
- SGU remains blocked for production filtering.
- Rotemberg and EZ remain future optional live fixtures.
- No BayesFilter production DSGE adapter is justified in this phase.

Next phase justified?
- Yes.  DSGE candidates remain contained.

### Phase G: GPU/HMC/SVD-derivative blocker review

Decision:
- GPU/XLA-GPU remains blocked pending escalated probes and matching benchmark
  artifacts.
- HMC remains blocked pending target-specific value/score/Hessian, branch,
  compiled-parity, and sampler evidence.
- Linear SVD/eigen derivatives remain deferred pending real client need and
  spectral-gap/floor diagnostics.

Next phase justified?
- Yes.  All blocked claims remain blocked with explicit evidence requirements.

### Phase H: commit and handoff

Pre-commit tidy requirements:
- stage only v1-lane files;
- do not stage the shared monograph reset memo;
- do not stage unrelated DSGE request notes, `Zone.Identifier` files, or local
  images;
- run `git diff --cached --check`;
- commit the scoped v1 phase-completion artifacts.

Post-execution status before commit:
- Phases A through G passed their primary criteria.
- No veto diagnostics fired.
- The scoped commit should include only v1-lane files:
  - `docs/plans/bayesfilter-v1-*.md`;
  - v1 benchmark script and artifacts under `docs/benchmarks`;
  - `tests/test_v1_public_api.py`;
  - `docs/source_map.yml` v1 provenance entries.
- The following files remain out of lane and must not be staged by this lane:
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
  - `docs/plans/templates/*:Zone.Identifier`;
  - `singularity_test.png`.

Final interpretation:
- The v1 external-compatibility phase now has local API and regression
  evidence, CPU benchmark smoke and medium artifacts with memory metadata,
  explicit optional MacroFinance deferral, DSGE containment, and blocked
  GPU/HMC/SVD-derivative claims.

## 2026-05-11 update: active phase-completion goals, gaps, and hypotheses plan

Trigger:
- the user asked to summarize the goals for this phase, remaining gaps,
  hypotheses for the gaps, and a plan to complete the phase, test the
  hypotheses, and close the gaps.

Active plan artifact:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-2026-05-11.md
```

Audit artifact:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-audit-2026-05-11.md
```

Execute:
- Created the May 11 active plan rather than overwriting the May 10 draft.
- Aligned the May 11 audit so it reviews the May 11 active plan.
- The plan keeps this work inside the BayesFilter v1 external-compatibility
  lane and treats the structural SVD/SGU files open in the IDE as out of lane.
- The plan lists ten remaining gaps:
  - uncommitted lane artifacts;
  - current-date active handoff;
  - missing benchmark memory metadata;
  - missing medium-shape CPU benchmark artifact;
  - optional live MacroFinance status not final;
  - DSGE optional bridges not implemented;
  - missing GPU/XLA-GPU evidence;
  - missing HMC readiness evidence;
  - unproven need for linear SVD/eigen derivatives;
  - final commit or handoff not done.
- The plan maps those gaps to ten hypotheses and eight execution phases:
  `lane boundary audit -> plan/source-map finalization -> API/local regression
  gate -> benchmark metadata hardening -> medium-shape benchmark -> optional
  external status decision -> blocker review -> commit or handoff`.

Interpretation:
- The phase goal is a coherent BayesFilter-local v1 evidence bundle, not
  MacroFinance/DSGE switch-over.
- The next executable work should start with lane-boundary audit and benchmark
  metadata hardening.
- GPU/XLA-GPU and HMC remain blocked behind separate target-specific evidence.

## 2026-05-11 update: post-completion gap-closure plan

Trigger:
- the user asked to summarize the goals for this phase, remaining gaps,
  hypotheses for the gaps, and a plan to complete this phase, test the
  hypotheses, and close the gaps.

Starting point:
- The previous v1 external-compatibility phase is already committed:

```text
b83d4af Complete v1 external compatibility phase
```

New plan artifact:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-plan-2026-05-11.md
```

Execute:
- Created a follow-on plan rather than reopening the completed phase.
- Registered the plan in `docs/source_map.yml`.
- Kept structural SVD/SGU files, Chapter 18/18b files, MacroFinance, DSGE, and
  the shared monograph reset memo out of this lane.

Goals summarized in the plan:
- diagnose QR score/Hessian first-call tracing and memory cost;
- build a CPU benchmark shape ladder;
- obtain escalated GPU/XLA-GPU evidence or keep GPU blocked by evidence;
- run optional live MacroFinance read-only compatibility when appropriate;
- design test-only DSGE fixtures for Rotemberg and EZ while keeping SGU
  blocked;
- select the first HMC readiness target conservatively;
- quantify SVD-CUT branch diagnostics before derivative/HMC promotion;
- define CI/runtime tiers;
- commit only v1-lane artifacts.

Interpretation:
- The next phase should begin with CPU-local benchmark diagnosis before GPU or
  optional external checks, because the medium CPU artifact already exposed a
  concrete QR score/Hessian first-call cost.
- GPU checks require escalated sandbox permissions.
- MacroFinance and DSGE remain external compatibility targets, not production
  dependencies.

## 2026-05-11 execution: post-completion gap-closure pass

User request:
- reread and tighten the post-completion plan;
- audit as another developer;
- execute each phase with
  `plan -> execute -> test -> audit -> tidy -> update reset memo`;
- continue automatically only if primary criteria and veto diagnostics allow;
- commit modified files when complete;
- summarize results and next hypotheses.

Plan under execution:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-plan-2026-05-11.md
```

Audit artifact:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-plan-audit-2026-05-11.md
```

### Phase A/B: lane audit, plan tightening, and harness extension

Plan:
- keep execution in the v1 external-compatibility lane;
- reread the plan for missing gates;
- audit as another developer;
- tighten the benchmark harness so it can test the QR derivative cost
  hypotheses.

Execute:
- Confirmed out-of-lane files remain present and must not be staged:
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
  - `docs/plans/templates/*:Zone.Identifier`;
  - `singularity_test.png`.
- Confirmed the plan belongs only to the v1 lane.
- Audited the plan and approved scoped execution with conditional gates.
- Tightened the benchmark harness with:
  - `--benchmark-selector`;
  - `--modes`;
  - `--graph-warmup-calls`;
  - `--device-scope`;
  - named one-axis ladders:
    `v1_time_ladder`, `v1_parameter_ladder`,
    `v1_state_observation_ladder`;
  - the prior mixed `v1_cpu_diagnostic` ladder.
- Tightened the plan so Phase C no longer claims one-axis attribution from a
  mixed ladder.

Interpretation:
- The plan is safe to execute if only v1-lane files are staged.
- GPU/XLA-GPU remains conditional on escalated probes.
- MacroFinance and DSGE checks remain read-only/optional.

Next phase justified?
- Yes.  Run CPU-local smoke and one-axis benchmark ladders.

### Phase C: CPU QR derivative diagnostic ladders

Plan:
- verify the tightened benchmark harness still runs the smoke fixture;
- run one-axis CPU ladders for QR score/Hessian graph mode;
- stop before larger ladders if memory evidence is already strong enough or
  runtime/memory risk becomes disproportionate.

Execute:
- Regenerated the smoke artifact:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-11-smoke.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-11-smoke.md
docs/benchmarks/bayesfilter-v1-filter-benchmark-ladder-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-ladder-2026-05-11.md
```

- Added the time ladder:

```text
docs/benchmarks/bayesfilter-v1-qr-score-hessian-time-ladder-2026-05-11.json
docs/benchmarks/bayesfilter-v1-qr-score-hessian-time-ladder-2026-05-11.md
```

- Added the parameter ladder:

```text
docs/benchmarks/bayesfilter-v1-qr-score-hessian-parameter-ladder-2026-05-11.json
docs/benchmarks/bayesfilter-v1-qr-score-hessian-parameter-ladder-2026-05-11.md
```

Observed:
- smoke benchmark: all rows `status = ok`;
- mixed CPU diagnostic ladder: all rows `status = ok`;
- time ladder: all rows `status = ok`;
- parameter ladder: all rows `status = ok`.

Interpretation:
- H1 is supported: QR score/Hessian cost is dominated by graph warmup/tracing
  and materialization, not steady recurrence cost.  After one graph warmup,
  measured calls were millisecond scale.
- H2 is partially supported and sharper than before: parameter dimension is a
  strong cost driver.  At fixed `timesteps=8`, `state_dim=2`,
  `observation_dim=2`, high-water RSS delta rose from about 1236.8 MB
  (`parameter_dim=2`) to about 3319.8 MB (`parameter_dim=4`), and warmup time
  rose from about 20.4 seconds to about 64.3 seconds.
- Time dimension also matters.  At fixed `state_dim=2`, `observation_dim=2`,
  `parameter_dim=2`, high-water RSS delta rose from about 638.3 MB
  (`timesteps=4`) to about 1702.1 MB (`timesteps=16`), and warmup time rose
  from about 11.3 seconds to about 35.4 seconds.

Audit:
- The state/observation ladder was not run because the parameter ladder already
  drove high-water RSS above 7 GB.  Continuing automatically into larger
  ladders would add machine pressure without changing the main decision.
- The key conclusion is diagnostic, not an optimization claim.

Next phase justified?
- Yes, but execution should move to policy/evidence phases rather than more
  local CPU ladders.  Future optimization should target graph construction,
  Hessian materialization, and parameter-dimension scaling.

### Phase D: GPU/XLA-GPU probe and matching smoke

Plan:
- run GPU probes with escalated permissions;
- run a small matching-shape GPU-visible benchmark only if TensorFlow sees GPU;
- keep the result scoped to device availability and small-shape behavior.

Observed probes:

```text
nvidia-smi: NVIDIA GeForce RTX 4080 SUPER visible, 16376 MiB total memory.
TensorFlow 2.19.1 physical devices: CPU and GPU visible.
```

Execute:
- Added GPU-visible smoke artifacts:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-gpu-visible-smoke-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-gpu-visible-smoke-2026-05-11.md
docs/benchmarks/bayesfilter-v1-filter-benchmark-xla-visible-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-xla-visible-2026-05-11.md
```

Observed:
- all GPU-visible smoke rows `status = ok`;
- TensorFlow logical devices include CPU and GPU;
- XLA-visible linear value rows `status = ok`;
- small-shape GPU-visible steady timings were slower than CPU-only timings for
  this tiny fixture.

Interpretation:
- GPU availability is proven under escalation.
- Small-shape GPU-visible execution works.
- XLA-visible linear value rows complete in a GPU-visible process.
- No broad GPU speedup, confirmed GPU placement for every XLA op, QR
  derivative XLA, or HMC claim is justified.

Next phase justified?
- Yes.  Optional live MacroFinance can run read-only.

### Phase E: optional live MacroFinance read-only compatibility

Plan:
- run the optional MacroFinance test read-only;
- record external checkout commit;
- do not edit MacroFinance.

Observed:

```text
/home/chakwong/MacroFinance commit:
0e81988957ef1f8b520014929bea32ffee3881f4

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_macrofinance_linear_compat_tf.py \
  -p no:cacheprovider
4 passed, 2 warnings in 74.54s (0:01:14)
```

Interpretation:
- Optional live MacroFinance compatibility passes on the observed external
  checkout.
- The MacroFinance checkout was dirty with unrelated local edits, so this is
  live compatibility evidence on an observed checkout, not clean release
  certification.
- This is compatibility evidence only; it does not authorize MacroFinance
  default switch-over.
- No MacroFinance files were edited.

Next phase justified?
- Yes.  Move to DSGE design-only, HMC target selection, SVD-CUT branch gate,
  and CI tier policy.

### Phase F-I: DSGE design, HMC target, SVD-CUT gate, CI tiers

Plan:
- close the remaining design/policy phases without editing external projects;
- keep SGU blocked;
- select a conservative first HMC target;
- turn SVD-CUT derivative cautions into measurable branch gates;
- define CI/runtime tiers.

Execute:
- Added:

```text
docs/plans/bayesfilter-v1-dsge-test-only-fixture-design-2026-05-11.md
docs/plans/bayesfilter-v1-hmc-first-target-selection-2026-05-11.md
docs/plans/bayesfilter-v1-svd-cut-branch-diagnostic-gate-2026-05-11.md
docs/plans/bayesfilter-v1-ci-runtime-tier-policy-2026-05-11.md
docs/plans/bayesfilter-v1-post-completion-gap-closure-result-2026-05-11.md
```

Interpretation:
- Rotemberg is a future optional live structural fixture candidate.
- EZ is a future metadata-only optional fixture candidate.
- SGU remains blocked by causal locality.
- The first HMC target should be:

```text
linear_qr_score_hessian_static_lgssm
```

- SVD-CUT derivative/HMC promotion remains blocked unless smooth separated
  spectrum, inactive floors, and parity evidence dominate a target region.
- CI evidence is tiered into fast local CI, focused local regression, extended
  CPU diagnostics, optional live external, escalated GPU/XLA-GPU, and HMC
  readiness.

Next phase justified?
- Yes.  Run final validation, stage only v1-lane files, and commit.

### Phase J: final validation and commit boundary

Plan:
- validate local v1 API/regression/SVD-CUT branch behavior;
- validate the benchmark CLI still runs after harness changes;
- check benchmark JSON syntax and diff whitespace;
- stage only v1-lane files.

Observed:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  -p no:cacheprovider
35 passed, 2 warnings in 92.35s (0:01:32)
```

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 1 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2 \
  --benchmark-selector linear_value \
  --modes graph
all rows status = ok
```

Additional checks:
- benchmark JSON syntax checks passed for the CPU smoke, GPU-visible smoke,
  and XLA-visible value artifacts;
- `git diff --check` passed.

Interpretation:
- The post-completion gap-closure pass is ready for a scoped v1-lane commit.
- The shared monograph reset memo, unrelated DSGE request note,
  `Zone.Identifier` sidecars, and `singularity_test.png` remain out of lane and
  must not be staged.

### Final validation and commit boundary

Validation:
- `git diff --check`: passed;
- `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q
  tests/test_v1_public_api.py -p no:cacheprovider`: `2 passed,
  2 warnings`;
- `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py --repeats 1
  --benchmark-selector linear_value --modes graph --graph-warmup-calls 1
  --timesteps 4 --state-dim 2 --observation-dim 2 --parameter-dim 2
  --output /tmp/bayesfilter-v1-final-smoke.json --markdown-output
  /tmp/bayesfilter-v1-final-smoke.md`: all rows `status = ok`.

Commit boundary:
- stage only v1 external-compatibility files:
  - `docs/benchmarks/benchmark_bayesfilter_v1_filters.py`;
  - `docs/benchmarks/bayesfilter-v1-*.json`;
  - `docs/benchmarks/bayesfilter-v1-*.md`;
  - `docs/plans/bayesfilter-v1-*.md`;
  - `docs/source_map.yml`.
- Leave out-of-lane files untouched and unstaged:
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
  - `docs/plans/templates/*:Zone.Identifier`;
  - `singularity_test.png`.

Final interpretation:
- The post-completion gap-closure phase is ready for a scoped v1-lane commit.
- The next executable development lane should start from QR derivative memory
  reduction or the first HMC target, not MacroFinance switch-over.
