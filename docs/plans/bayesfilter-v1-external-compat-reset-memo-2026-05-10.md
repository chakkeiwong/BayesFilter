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

## 2026-05-11 update: next-phase HMC and branch-diagnostic plan

Trigger:
- the user asked to summarize the goals for this phase, remaining gaps,
  hypotheses for the gaps, and create a plan to complete the phase, test the
  hypotheses, and close the gaps.

Plan artifact:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-2026-05-11.md
```

Goals:
- keep the v1 lane isolated from structural SVD/SGU and shared monograph work;
- turn QR score/Hessian memory findings into a targeted implementation
  experiment;
- build the first target-specific HMC readiness artifact for
  `linear_qr_score_hessian_static_lgssm`;
- add SVD-CUT branch-frequency diagnostics before any SVD-CUT HMC promotion;
- decide whether QR derivative XLA/GPU follow-up is justified;
- keep MacroFinance and DSGE external/read-only.

Remaining gaps:
- QR derivative memory driver is measured but not reduced;
- selected HMC target has no target contract or sampler smoke;
- SVD-CUT branch-frequency evidence is missing;
- QR derivative XLA/GPU is untested;
- MacroFinance pass is optional evidence on a dirty checkout, not release
  certification;
- DSGE bridges remain design-only;
- CI tier policy is documented but not enforced by tooling.

Hypotheses:
- H1/H2: Hessian materialization and parameter-pair contractions dominate QR
  derivative memory, while score-only derivatives may be cheap enough for HMC;
- H3/H4: the first HMC smoke should use value/score and keep Hessian as
  diagnostics;
- H5/H6: SVD-CUT HMC remains blocked unless branch-frequency evidence proves
  smooth separated-spectrum inactive-floor dominance;
- H7: QR derivative XLA/GPU should wait until CPU HMC and memory diagnostics
  identify a worthwhile shape;
- H8/H9: optional MacroFinance and DSGE evidence are not required for the first
  local HMC target;
- H10: CI tier documentation is enough unless tests become accidental defaults.

Recommended execution:
- start with lane audit, QR derivative memory-reduction design, and focused CPU
  diagnostic artifacts;
- only then implement the first HMC target contract and tiny sampler smoke;
- keep SVD-CUT branch-frequency diagnostics in the extended CPU tier;
- update this reset memo after each phase if/when execution begins.

## 2026-05-11 execution: HMC and branch-diagnostic closure

Trigger:
- the user asked to read the plan carefully, tighten ambiguities, audit it as
  another developer, then execute each phase with
  `plan -> execute -> test -> audit -> tidy -> update reset memo`, continuing
  automatically only while primary criteria and veto diagnostics allow.

Active plan:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-2026-05-11.md
```

Audit artifact:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-audit-2026-05-11.md
```

### Phase A: lane and worktree audit

Plan:
- classify the dirty worktree by lane;
- confirm that execution can proceed without editing MacroFinance, DSGE, the
  structural SVD/SGU lane, or the shared monograph reset memo;
- tighten plan ambiguity before implementation.

Observed:
- current branch: `main...origin/main [ahead 4]`;
- v1-lane files currently dirty or untracked:
  - `docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md`;
  - `docs/source_map.yml`;
  - `docs/plans/bayesfilter-v1-hmc-branch-closure-plan-2026-05-11.md`;
  - `docs/plans/bayesfilter-v1-hmc-branch-closure-plan-audit-2026-05-11.md`;
- out-of-lane dirty or untracked files remain present and must not be staged:
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - `docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-2026-05-11.md`
    supersedes the earlier `hmc-branch-closure` draft and is the governing
    plan for the final execution record;
  - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
  - `docs/plans/templates/*:Zone.Identifier`;
  - `singularity_test.png`.

Tightening applied:
- no public QR score-only API should be added in this pass;
- first-order QR score helpers may be added as private diagnostics to isolate
  score-only cost from Hessian materialization;
- the first HMC sampler path should use QR value plus TensorFlow autodiff
  score;
- the full analytic QR score/Hessian path should remain parity and curvature
  diagnostics;
- benchmark rows must materialize score/Hessian when claiming full derivative
  cost;
- SVD-CUT branch-frequency evidence remains diagnostic-only;
- GPU/XLA remains optional and escalated.

Tests:
- documentation/audit phase only; no code tests required.

Audit:
- Phase A passes.
- Execution can proceed with only BayesFilter v1-lane files.
- No veto diagnostic is active.

Next phase justified?
- Yes.  Proceed to Phase B: QR derivative memory-reduction design with
  score-only diagnostics kept private and public API expansion blocked.

### Phase B: QR derivative memory-reduction design

Plan:
- inspect `tf_qr_linear_gaussian_score_hessian`;
- decide whether score-only can be safely promoted or should remain
  diagnostic/private;
- run focused QR derivative tests before proceeding to benchmark artifacts.

Execute:
- Confirmed that the existing full QR analytic derivative path propagates
  first- and second-order tensors together through prediction, innovation,
  Kalman gain, Joseph update, and QR/Cholesky factor derivatives.
- Added private first-order QR diagnostic helpers:
  `_tf_qr_sqrt_kalman_score` and `_tf_qr_linear_gaussian_score`.
- Removed the earlier partial public score-only export from the top-level and
  linear public API surfaces.
- The first HMC target uses QR value plus TensorFlow autodiff score; full
  analytic score/Hessian remains parity and curvature diagnostic evidence.
- Added first-order QR factor helper utilities used by the private diagnostic
  path.

Tests:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  -p no:cacheprovider
12 passed, 2 warnings in 84.95s
```

Interpretation:
- Phase B closes the design ambiguity without expanding the public API:
  first-order score helpers remain private diagnostics.
- H1/H2 remain live for benchmarking: the first-order path is correct on the
  tiny fixture, but its memory/runtime advantage must be measured in Phase C.
- H4 is supported as a design rule: Hessian is diagnostic, not required in the
  sampler log-prob path.

Audit:
- Public API test confirms no score-only symbol was promoted.
- QR derivative correctness tests remain green.
- No MacroFinance, DSGE, structural SVD/SGU, or shared monograph edits are
  required by this phase.

Next phase justified?
- Yes.  Proceed to Phase C: QR score/Hessian diagnostic artifacts.

### Phase C: QR score/Hessian diagnostic artifact

Plan:
- run focused CPU-only graph rows for the private first-order score path and
  the full analytic QR score/Hessian path;
- materialize score and Hessian tensors in the rows that claim derivative cost;
- record JSON and Markdown artifacts under `docs/benchmarks`.

Execute:
- Updated the benchmark harness so derivative rows materialize the tensors they
  claim to measure:
  - `linear_qr_score` materializes log likelihood and score through the private
    first-order helper;
  - `linear_qr_score_hessian` materializes log likelihood, score, and Hessian.
- Ran:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 2 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2 \
  --benchmark-selector derivatives \
  --modes graph \
  --graph-warmup-calls 1 \
  --output docs/benchmarks/bayesfilter-v1-qr-derivative-materialization-diagnostic-2026-05-11.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-qr-derivative-materialization-diagnostic-2026-05-11.md
```

Observed:
- device scope: CPU-only with `CUDA_VISIBLE_DEVICES=-1`;
- all rows `status = ok`;
- `linear_qr_score`: warmup `2.4352s`, mean steady `0.0025s`,
  max-RSS delta `123.0 MB`;
- `linear_qr_score_hessian`: warmup `11.6858s`, mean steady `0.0048s`,
  max-RSS delta `636.0 MB`.

Interpretation:
- H1 is supported for the first fixed target shape: second-order/Hessian
  materialization is the dominant warmup and process-memory driver.
- H2 is supported for the fixed diagnostic shape: the private first-order score
  path is much cheaper than the full score/Hessian path.
- The evidence justifies keeping Hessian materialization out of the first HMC
  sampler path.
- Public score-only API promotion remains a later API-freeze review item.

Audit:
- The benchmark artifacts are CPU-only diagnostics, not client switch-over,
  GPU/XLA, or HMC convergence claims.
- Non-escalated CUDA initialization warnings were ignored because the run hid
  GPU devices deliberately.
- No veto diagnostic is active.

Next phase justified?
- Yes.  Proceed to Phases D-E: first QR HMC target contract and tiny sampler
  smoke.

### Phases D-E: first QR HMC target contract and smoke

Plan:
- define the fixed target `linear_qr_score_hessian_static_lgssm`;
- use QR value plus TensorFlow autodiff score in the sampler path;
- use full analytic QR score/Hessian for parity and curvature diagnostics;
- keep the sampler smoke tiny, fixed-seed, CPU-only, and opt-in.

Execute:
- Added a BayesFilter-local testing fixture:
  `bayesfilter/testing/tf_hmc_readiness.py`.
- Added opt-in HMC readiness tests:
  `tests/test_hmc_linear_qr_readiness_tf.py`.
- Added `pytest.ini` marker declarations for opt-in `extended` and `hmc`
  diagnostics.
- Wrote artifacts:
  - `docs/benchmarks/bayesfilter-v1-linear-qr-hmc-readiness-smoke-2026-05-11.json`;
  - `docs/benchmarks/bayesfilter-v1-linear-qr-hmc-readiness-smoke-2026-05-11.md`.

Tests:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
3 passed, 2 warnings in 39.00s
```

Observed artifact values:
- initial target log-prob: `-1.3568111688046285`;
- initial target gradient: `[-0.40893740245060484, -1.0379645721254034]`;
- analytic-vs-autodiff value residual: `0.0`;
- score residual: `4.44e-16`;
- Hessian residual: `6.66e-16`;
- target Hessian symmetry residual: `0.0`;
- negative-Hessian eigenvalues: `[2.3467420404247332, 2.998234138312041]`;
- HMC smoke: `12` finite samples, `0` nonfinite samples, acceptance rate `1.0`,
  max absolute log-accept ratio `0.0004737934694076795`.

Interpretation:
- H3 is supported for this fixed target: the tiny CPU-only fixed-seed HMC smoke
  finishes with finite samples and finite target/gradient diagnostics.
- H4 is supported: Hessian is useful for curvature and parity, but the sampler
  path does not need Hessian materialization.
- This remains a target-specific smoke, not a convergence claim, not a default
  sampler recommendation, and not a MacroFinance/DSGE/SVD-CUT/GPU/XLA claim.

Audit:
- Tests are opt-in through `BAYESFILTER_RUN_HMC_READINESS=1` and marked `hmc`.
- The artifact explicitly limits claim scope.
- No veto diagnostic is active.

Next phase justified?
- Yes.  Proceed to Phase F: SVD-CUT branch-frequency diagnostics.

### Phase F: SVD-CUT branch-frequency diagnostics

Plan:
- add a small SVD-CUT branch-frequency diagnostic over a tiny smooth parameter
  box;
- include a weak spectral-gap control case;
- keep the test opt-in and keep SVD-CUT HMC blocked.

Execute:
- Added:
  - `bayesfilter/testing/tf_svd_cut_branch_diagnostics.py`;
  - `tests/test_svd_cut_branch_diagnostics_tf.py`;
  - `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-diagnostic-2026-05-11.json`;
  - `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-diagnostic-2026-05-11.md`.

Tests:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
2 passed, 2 warnings in 11.31s
```

Observed artifact values:
- smooth box grid: `[[0.27, 0.23], [0.31, 0.27], [0.35, 0.31]]`;
- smooth box total count: `3`;
- smooth count: `3`;
- active floor count: `0`;
- weak spectral-gap count: `0`;
- nonfinite count: `0`;
- minimum placement eigen gap: `0.02189900512583248`;
- maximum support residual: `0.0`;
- maximum deterministic residual: `0.0`;
- weak-gap control total count: `2`;
- weak-gap control weak spectral-gap count: `2`.

Interpretation:
- H5 is not supported inside this deliberately tiny smooth box, but the weak-gap
  control proves the diagnostic can count branch blockers.
- H6 governs the decision: an all-smooth tiny box is not enough to promote
  SVD-CUT HMC in this phase.
- SVD-CUT HMC remains blocked pending a separate target-specific SVD-CUT HMC
  plan with wider branch evidence and sampler diagnostics.

Audit:
- The test is opt-in through `BAYESFILTER_RUN_EXTENDED_CPU=1` and marked
  `extended`.
- The artifact explicitly states diagnostic-only scope.
- No veto diagnostic is active.

Next phase justified?
- Yes.  Proceed to Phases G-H: GPU/XLA and CI/external-status review.

### Phases G-H: GPU/XLA and CI/external-status review

Plan:
- decide whether QR derivative GPU/XLA work is justified now;
- verify that new HMC and SVD-CUT diagnostics are not accidental fast-CI tests;
- keep MacroFinance and DSGE external/read-only and out of this phase.

Execute:
- Reviewed the CPU QR derivative materialization artifact and the first QR HMC
  smoke artifact.
- Decided not to run escalated QR derivative GPU/XLA in this phase.
- Updated `docs/plans/bayesfilter-v1-ci-runtime-tier-policy-2026-05-11.md`
  with current opt-in HMC and SVD-CUT branch diagnostic commands and claim
  scopes.
- Added/kept `pytest.ini` marker declarations for:
  - `extended`;
  - `hmc`;
  - existing optional `external` and `gpu`.

Interpretation:
- H7 closes by deferral: CPU evidence identifies Hessian materialization as the
  urgent cost driver, and the first QR HMC target already runs on CPU.  A
  matching-shape escalated QR derivative GPU/XLA pass is useful follow-up, but
  not required to complete this phase.
- H8 remains closed: dirty optional MacroFinance evidence does not block local
  BayesFilter HMC work and is not promoted to release certification.
- H9 remains closed: DSGE optional bridges are not required for the first local
  HMC target.
- H10 is strengthened: tests under `tests/` that are extended/HMC now require
  opt-in environment variables.

Audit:
- No GPU command was run in this phase, so no new GPU/XLA claim is made.
- New HMC readiness tests require `BAYESFILTER_RUN_HMC_READINESS=1`.
- New SVD-CUT branch diagnostics require `BAYESFILTER_RUN_EXTENDED_CPU=1`.
- No external client code was edited.

Next phase justified?
- Yes.  Proceed to Phase I: result artifact, source-map update, validation, and
  scoped commit.

### Phase I: result, source map, validation, and commit boundary

Plan:
- write the result artifact;
- register the plan/result/benchmark artifacts in `docs/source_map.yml`;
- run final validation;
- stage only v1-lane files and commit.

Execute:
- Added:
  - `docs/plans/bayesfilter-v1-hmc-branch-closure-result-2026-05-11.md`;
  - source-map entries for the plan/result and benchmark artifacts.

Observed validation before commit:

```text
git diff --check
passed

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
16 passed, 2 warnings in 91.70s

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
5 skipped, 2 warnings in 3.47s

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
3 passed, 2 warnings in 38.58s

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
2 passed, 2 warnings in 11.09s
```

Commit boundary:
- stage only v1-lane code, tests, benchmark artifacts, v1 plan artifacts,
  `pytest.ini`, and `docs/source_map.yml`;
- do not stage:
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
  - `docs/plans/templates/*:Zone.Identifier`;
  - `singularity_test.png`.

## 2026-05-11 final execution: v1 HMC branch closure

The governing plan for the final execution pass is:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-2026-05-11.md
```

Independent audit:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-audit-2026-05-11.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-result-2026-05-11.md
```

Final interpretation:
- no public QR score-only API was promoted;
- private dense first-order QR score helpers were added for diagnostics;
- the first HMC sampler path uses QR value plus TensorFlow autodiff score;
- full analytic QR score/Hessian remains parity and curvature evidence;
- SVD-CUT HMC remains blocked despite a smooth tiny-box diagnostic;
- GPU/XLA QR derivative work remains deferred;
- MacroFinance and DSGE stayed read-only and out of this lane.

Artifacts:
- `docs/benchmarks/bayesfilter-v1-qr-derivative-materialization-diagnostic-2026-05-11.*`;
- `docs/benchmarks/bayesfilter-v1-linear-qr-hmc-readiness-smoke-2026-05-11.*`;
- `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-diagnostic-2026-05-11.*`.

Next hypotheses:
- H11: public QR score-only should be considered only if dense and masked
  semantics are designed and tested together;
- H12: the first QR HMC target should be stress-tested with longer multi-chain
  CPU diagnostics;
- H13: QR derivative GPU/XLA should be tested only at a CPU-selected medium
  shape and only with escalated GPU visibility;
- H14: SVD-CUT branch smoothness should be tested on wider target regions
  before any SVD-CUT HMC plan.

## 2026-05-11 addendum: readiness plan and extended artifacts

After commit `7526b72` closed the v1 HMC branch diagnostics, an addendum plan
was retained to make the readiness narrative and remaining hypotheses explicit:

```text
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-2026-05-11.md
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-audit-2026-05-11.md
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-result-2026-05-11.md
```

This addendum does not change the implementation decision from `7526b72`:
- first-order QR score helpers remain private diagnostics;
- the first HMC sampler path uses QR value plus TensorFlow autodiff score;
- full analytic QR score/Hessian remains parity and curvature evidence;
- SVD-CUT HMC remains blocked;
- GPU/XLA QR derivative work remains deferred.

Additional artifacts recorded for the addendum:
- `docs/benchmarks/bayesfilter-v1-qr-derivative-cost-decomposition-2026-05-11.*`;
- `docs/benchmarks/bayesfilter-v1-qr-score-state-observation-ladder-2026-05-11.*`;
- `docs/benchmarks/bayesfilter-v1-qr-hmc-smoke-2026-05-11.*`;
- `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-2026-05-11.*`;
- `docs/benchmarks/benchmark_bayesfilter_v1_hmc_smoke.py`;
- `docs/benchmarks/benchmark_bayesfilter_v1_svd_cut_branch_frequency.py`.

Interpretation:
- H3 remains supported: score-only diagnostics are materially cheaper than full
  score/Hessian at the first target shape.
- H4/H5 remain supported only inside the small score-envelope ladder.
- HMC evidence remains a target-specific finite smoke, not convergence.
- SVD-CUT evidence remains diagnostic-only, even when the tiny box is smooth.

Next phase justified?
- Yes, but as a separate v1 follow-up: decide whether to promote a public
  score-only QR API, design masked score-only semantics, run longer multi-chain
  QR HMC diagnostics, and widen SVD-CUT branch sweeps before any SVD-CUT HMC
  plan.

## 2026-05-11 next plan: score API, HMC, and SVD-CUT gap closure

The next v1 follow-up plan is:

```text
docs/plans/bayesfilter-v1-score-api-hmc-svdcut-gap-closure-plan-2026-05-11.md
```

Goals:
- decide whether private first-order QR score diagnostics should become a
  public score-only API;
- define dense and masked score-only semantics before any public promotion;
- extend the first QR HMC smoke to longer target-specific multi-chain
  diagnostics;
- widen SVD-CUT branch-frequency sweeps before any SVD-CUT HMC plan;
- decide whether one QR derivative GPU/XLA medium-shape gate is justified;
- keep MacroFinance and DSGE read-only until BayesFilter v1 stabilizes.

Primary remaining gaps:
- public score-only QR API contract is undecided;
- masked score-only behavior is undefined;
- HMC evidence remains smoke-level;
- SVD-CUT branch evidence is too narrow;
- QR derivative GPU/XLA has no matching-shape evidence;
- external compatibility is not a client switch-over contract.

Hypotheses to test:
- public QR score-only is safe only if dense and masked semantics are explicit;
- masked score-only can reuse the static dummy-row convention if parity tests
  pass;
- longer QR HMC diagnostics remain finite but should not be labeled convergence
  without stronger evidence;
- SVD-CUT smoothness is local until a wider sweep proves otherwise;
- GPU/XLA derivative testing is useful only after selecting a CPU-stable medium
  shape.

Lane reminder:
- keep this follow-up in `bayesfilter`, `tests`, `docs/benchmarks`,
  `docs/plans/bayesfilter-v1-*`, `docs/source_map.yml`, and `pytest.ini`;
- do not stage the shared monograph memo, structural SVD/SGU plans,
  Chapter 18/18b, MacroFinance, DSGE, sidecars, or local images.

## 2026-05-11 execution-ready closure plan

The broad score/API/HMC/SVD-CUT follow-up has been tightened into an
execution-ready phase plan:

```text
docs/plans/bayesfilter-v1-score-hmc-svdcut-execution-closure-plan-2026-05-11.md
```

Purpose:
- summarize this phase's goals, current status, gaps, and hypotheses in one
  runnable artifact;
- make each gap falsifiable through a test, benchmark artifact, or explicit
  blocker;
- preserve the v1 lane boundary while other agents work on structural and
  monograph files.

Interpretation:
- the plan does not execute code changes yet;
- it keeps QR score-only API promotion conditional on dense and masked
  semantics;
- it keeps HMC and SVD-CUT claims diagnostic until stronger artifacts exist;
- it keeps MacroFinance and DSGE read-only.

Next phase justified?
- Yes.  The next execution pass can follow the plan's phase order:
  lane recovery and baseline tests, score-only API decision, score-only
  implementation or blocker result, longer QR HMC diagnostic, wider SVD-CUT
  branch sweep, optional escalated GPU/XLA derivative gate, and reset/source-map
  cleanup.

## 2026-05-11 nonlinear filtering master testing program

Created the nonlinear filtering master testing plan:

```text
docs/plans/bayesfilter-v1-nonlinear-filtering-master-testing-program-2026-05-11.md
```

Purpose:
- turn the Chapter 18 SVD sigma-point and SVD-CUT derivative derivations into
  implementation gates;
- remove raw `GradientTape` SVD-CUT4 derivatives from the production derivative
  path while preserving autodiff as a testing oracle;
- decide score-first versus Hessian implementation order;
- search and document benchmark nonlinear SSMs before adding them to tests;
- build a BayesFilter-local nonlinear model suite for SVD cubature, SVD-UKF,
  and SVD-CUT4.

Verification note:
- MathDevMCP found the relevant Chapter 18 labels and provenance, but its
  automated audit status remains unverified because matrix assumptions and
  proof obligations require manual formalization.  The plan therefore treats
  Chapter 18 as the source derivation and requires traceability, finite
  difference, oracle, and branch diagnostics before derivative promotion.

Next phase justified?
- Yes.  The first execution pass should be documentation and testing
  infrastructure only: derivation-to-code traceability, production/export
  cleanup for raw `GradientTape`, accepted benchmark model documentation in
  Chapter 28, and a small nonlinear TF fixture suite.

## 2026-05-12 nonlinear filtering subplans

Created three subplans under the nonlinear filtering master testing program:

```text
docs/plans/bayesfilter-v1-nonlinear-model-suite-documentation-and-testing-tools-plan-2026-05-12.md
docs/plans/bayesfilter-v1-svd-filter-analytic-gradient-audit-implementation-plan-2026-05-12.md
docs/plans/bayesfilter-v1-nonlinear-filtering-remaining-master-program-plan-2026-05-12.md
```

Interpretation:
- the model-suite subplan makes Models A-C the first execution rung and defers
  bearings, radar, and stochastic volatility until residual/noise contracts
  are explicit;
- the gradient subplan keeps the current `GradientTape` SVD-CUT4
  score/Hessian path as a testing oracle target, not a production analytic
  derivative path, and makes analytic score implementation the first priority;
- the remaining-program subplan covers value-filter consolidation, branch
  diagnostics, approximation benchmarks, CI/runtime tiers, optional escalated
  GPU/XLA evidence, and target-specific HMC readiness;
- MacroFinance, DSGE, structural plans, Chapter 18b, and the shared monograph
  reset memo remain out of this lane.

Next phase justified?
- Yes.  Execute the model-suite documentation/testing-tools subplan first,
  because analytic score tests and effectiveness benchmarks need reusable
  model fixtures and documented oracles.

### 2026-05-12 model-suite execution M0-M1

Plan and audit:

```text
docs/plans/bayesfilter-v1-nonlinear-model-suite-documentation-and-testing-tools-plan-2026-05-12.md
docs/plans/bayesfilter-v1-nonlinear-model-suite-documentation-and-testing-tools-plan-audit-2026-05-12.md
```

Source/result artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-model-suite-source-notes-and-result-2026-05-12.md
```

M0 result:
- CPU-only focused nonlinear/SVD baseline passed: `20 passed, 2 warnings`;
- warnings were TensorFlow Probability `distutils` deprecation warnings;
- out-of-lane dirty files remain untouched.

M1 result:
- local research-assistant had no matching paper summaries for the benchmark
  sources;
- DOI/web metadata was used for first-pass source notes;
- Model A and Model B are BayesFilter-local synthetic oracles;
- Model C is documented as the standard nonlinear growth benchmark associated
  with Kitagawa (1996), implemented for BayesFilter V1 as an autonomous
  phase-state testing embedding because the current structural TF transition
  does not accept an explicit time index;
- `docs/references.bib` received minimal entries for Kitagawa (1996) and
  Arulampalam et al. (2002); Gordon et al. (1993) and Chopin and
  Papaspiliopoulos (2020) already existed.

Next phase justified?
- Yes.  Proceed to M2 Chapter 28 documentation, preserving the caveat that
  dense quadrature is a one-step Gaussian moment-projection oracle, not an
  exact nonlinear likelihood.

### 2026-05-12 model-suite execution M2

M2 result:
- updated `docs/chapters/ch28_nonlinear_ssm_validation.tex` with Models A-C,
  default parameters, oracle status, and deferred Models D-F;
- updated `docs/references.bib` with minimal accepted benchmark citations;
- kept Chapter 18, Chapter 18b, structural plans, MacroFinance, DSGE, and the
  shared monograph reset memo out of this phase.

Interpretation:
- Chapter 28 now gives tests a documented law and oracle contract;
- Model C is clearly an autonomous phase-state testing fixture, not a
  production time-inhomogeneous transition API change.

Next phase justified?
- Yes.  Proceed to M3 testing tools.

### 2026-05-12 model-suite execution M3

M3 result:
- added `bayesfilter/testing/nonlinear_models_tf.py` with Models A-C, fixed
  observations, and dense one-step Gaussian moment-projection reference tools;
- exported these testing helpers from `bayesfilter.testing`;
- added nonlinear constructor, oracle, and value-filter tests under
  `tests/test_nonlinear_*`.

Interpretation:
- the implementation remains testing-only and does not change production
  filter APIs;
- Model C uses the documented autonomous phase-state embedding;
- dense quadrature remains a tiny-dimensional test oracle, not a production
  nonlinear likelihood.

Next phase justified?
- Yes, if M4 value/oracle tests pass.

### 2026-05-12 model-suite execution M4-M5

M4 result:
- new nonlinear model-suite tests passed: `11 passed, 2 warnings`;
- focused nonlinear/SVD regression passed: `31 passed, 2 warnings`;
- `py_compile` passed for the new fixture and tests;
- `git diff --check` passed for touched files;
- production NumPy scan found no NumPy imports in `bayesfilter/nonlinear`;
- NumPy use is confined to tests and the testing-only dense quadrature oracle.

M5 result:
- registered execution provenance in `docs/source_map.yml`;
- updated only this V1 lane reset memo;
- did not stage or edit the shared monograph reset memo, structural plans,
  Chapter 18b, MacroFinance, DSGE, sidecar files, or local images.
- committed the first model-suite pass as
  `14eb751 Add nonlinear model suite fixtures`.
- final documentation sanity check:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `docs/` produced `main.pdf`; the new Chapter 28 citations resolved, while
  remaining warnings are pre-existing unresolved DPF citations/references
  outside this subplan.

Interpretation:
- first-rung nonlinear model-suite infrastructure is now in place;
- Model A supplies exact linear recovery;
- Models B-C supply nonlinear structural value checks and one-step dense
  Gaussian projection oracles;
- this phase deliberately makes no analytic-gradient, Hessian, HMC, GPU/XLA,
  SMC, or client switch-over claim.
- the first subplan is closed against its done definition.

Next phase justified?
- Yes.  The next phase should execute the SVD filter analytic-gradient audit
  and implementation subplan.  Its first hypotheses are that the Chapter 18
  score equations can be mapped to one shared fixed-rule derivative core, and
  that the current raw `GradientTape` SVD-CUT4 score/Hessian path can move to a
  testing oracle without losing branch diagnostics.

## 2026-05-12 SVD filter analytic-gradient execution

User request:
- continue with the second subplan under the nonlinear filtering master
  program;
- read and tighten the plan if needed;
- audit as another developer;
- execute phase by phase with tests, audit, tidy, reset memo update, and commit;
- stay in the BayesFilter V1 lane and avoid other agents' structural,
  monograph, MacroFinance, and DSGE lanes.

Plan under execution:

```text
docs/plans/bayesfilter-v1-svd-filter-analytic-gradient-audit-implementation-plan-2026-05-12.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-svd-filter-analytic-gradient-audit-result-2026-05-12.md
```

Initial lane status:
- branch `main` was ahead of `origin/main` by 9 commits;
- out-of-lane dirty/untracked files existed before this pass and were left
  untouched;
- protected structural plans, Chapter 18b, shared monograph reset memo,
  MacroFinance, and DSGE files were not edited by this lane.

### Phase G0: lane recovery and baseline

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Result:
- `20 passed, 2 warnings`;
- warnings were TensorFlow Probability `distutils` deprecation warnings.

Interpretation:
- existing nonlinear/SVD value filters and the old branch-gated SVD-CUT
  derivative tests were clean before implementation.

Next phase justified?
- Yes.  Proceed to G1 derivation-to-code audit.

### Phase G1: derivation-to-code audit and plan tightening

Audit result:
- MathDevMCP label lookup confirmed the Chapter 18 score and reconstruction
  labels in `docs/chapters/ch18_svd_sigma_point.tex`;
- the existing `TFStructuralStateSpace` callback contract did not contain the
  state, innovation, observation, or parameter derivatives needed for a true
  analytic score;
- the plan was tightened to forbid hidden production `GradientTape` and to
  require an explicit first-order structural derivative provider.

Interpretation:
- G1 passed after tightening;
- G3 could proceed only by adding a derivative-provider contract, not by
  reusing raw autodiff.

Next phase justified?
- Yes.  Proceed to G2 oracle migration.

### Phase G2: raw tape moved to testing oracle

Implementation:
- added `bayesfilter/testing/tf_svd_cut_autodiff_oracle.py`;
- updated tests and branch diagnostics to import the raw SVD-CUT
  score/Hessian from the testing module;
- removed `tf_svd_cut4_score_hessian` from top-level and
  `bayesfilter.nonlinear` public exports.

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `10 passed, 2 skipped, 2 warnings`.

Interpretation:
- raw autodiff remains available as a testing oracle;
- production public exports no longer advertise it as an analytic SVD-CUT
  derivative backend.

Next phase justified?
- Yes.  Proceed to G3-G5 with an explicit first-order derivative contract.

### Phase G3-G5: analytic score core, integration, and tests

Implementation:
- added `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`;
- introduced `TFStructuralFirstDerivatives`;
- added shared analytic smooth-branch score machinery for:
  - `tf_svd_cubature_score`;
  - `tf_svd_ukf_score`;
  - `tf_svd_cut4_score`;
  - `tf_svd_sigma_point_score_with_rule`;
- added `tests/test_nonlinear_sigma_point_scores_tf.py`.

Score contract:
- production path is TF-only;
- score is the derivative of the implemented sigma-point likelihood;
- SVD/eigen factors reconstruct the implemented covariance branch;
- active hard floors and weak spectral gaps are blockers;
- Hessian is explicitly `None` and marked deferred.

Focused validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  -p no:cacheprovider
```

Result:
- `6 passed, 2 warnings`.

Checks:
- finite-difference score parity for SVD cubature and SVD-UKF;
- finite-difference and testing-oracle score parity for SVD-CUT4;
- branch blockers for active floors and weak spectral gaps;
- eager/graph parity for the cubature score.

Interpretation:
- the shared analytic score core is functional on the smooth affine fixture;
- Models B-C still need explicit analytic derivative providers before nonlinear
  score claims can be extended beyond the affine smooth fixture.

Next phase justified?
- Yes for G6 Hessian gate and provenance.

### Phase G6: Hessian gate

Decision:
- Hessian remains deferred.

Reason:
- HMC needs a score, not a Hessian;
- the current production score requires only first derivatives;
- a Hessian implementation would require second derivatives of structural
  maps, moment/covariance tensors, and the eigensystem branch, plus memory
  tests and a named downstream consumer.

Next phase justified?
- Yes.  Proceed to provenance, broader tests, and commit.

### Phase G7: provenance, tidy, and final validation

Provenance:
- added `docs/plans/bayesfilter-v1-svd-filter-analytic-gradient-audit-result-2026-05-12.md`;
- registered the result in `docs/source_map.yml`;
- updated `docs/plans/bayesfilter-v1-api-freeze-criteria-2026-05-10.md` so the
  stable nonlinear derivative candidates are score-only functions with explicit
  structural derivative providers, while the raw SVD-CUT tape path is
  testing-only.

Post-audit tidy:
- retained `bayesfilter/nonlinear/svd_cut_derivatives_tf.py` only as a
  migration guard that raises a clear error;
- the raw SVD-CUT `GradientTape` implementation itself is now only in
  `bayesfilter.testing.tf_svd_cut_autodiff_oracle`.

Final validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `43 passed, 2 skipped, 2 warnings`.

Additional checks:
- `python -m py_compile` passed for touched Python modules/tests;
- `git diff --check` passed;
- `docs/source_map.yml` parsed with `yaml.safe_load`;
- production NumPy scan in `bayesfilter/nonlinear` found no NumPy imports.

Interpretation:
- second subplan is complete for the affine smooth-branch score rung;
- raw SVD-CUT autodiff is testing-only;
- analytic first-order smooth-branch scores exist for SVD cubature, SVD-UKF,
  and SVD-CUT4;
- Hessian remains explicitly deferred.

Remaining hypotheses to test next:
- H-NL1: nonlinear Models B-C can be given explicit analytic first-derivative
  providers without changing the production structural callback contract.
- H-NL2: SVD cubature, SVD-UKF, and SVD-CUT4 analytic scores on Models B-C match
  finite differences of their implemented likelihoods under the same branch
  gates.
- H-NL3: branch-frequency diagnostics over the default nonlinear parameter
  boxes identify a usable smooth-score region for HMC readiness.
- H-NL4: optional GPU/XLA score benchmarks should be attempted only after the
  CPU score branch and nonlinear finite-difference tests are stable.

Next phase justified?
- Yes.  The next phase should add explicit first-derivative providers for
  nonlinear Models B-C and extend the score test ladder beyond the affine
  fixture.

## 2026-05-12 execution: nonlinear filtering remaining master program

User request:
- finish the third nonlinear filtering subplan;
- read and tighten the plan if needed;
- audit it as another developer;
- execute phases with `plan -> execute -> test -> audit -> tidy -> update reset memo`;
- stay in the BayesFilter V1 lane and avoid files owned by other agents.

Plan under execution:

```text
docs/plans/bayesfilter-v1-nonlinear-filtering-remaining-master-program-plan-2026-05-12.md
```

Initial lane status:
- branch `main` was ahead of `origin/main` by 10 commits;
- out-of-lane dirty/untracked files existed before this pass and are not owned
  by this lane;
- protected structural plans, Chapter 18b, shared monograph reset memo,
  MacroFinance, and DSGE files are not part of this pass.

### Phase R0: baseline and lane audit

Baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Result:
- `16 passed, 2 skipped, 2 warnings`;
- warnings were TensorFlow Probability `distutils` deprecation warnings.

Interpretation:
- pre-existing value and branch baseline was clean;
- continuation to R1 was justified.

### Phase R1-R2: value consolidation and branch diagnostics

Plan tightening and audit:
- added an execution addendum to the remaining master-program plan;
- explicitly blocked Models B-C score/HMC claims until first-derivative
  providers exist;
- kept GPU/XLA as an optional escalated gate and HMC as a readiness gate, not
  an execution claim;
- required benchmark artifacts to distinguish exact references from dense
  one-step Gaussian projection references.

Implementation:
- added `bayesfilter/testing/nonlinear_diagnostics_tf.py`;
- exported testing-only nonlinear diagnostic helpers from `bayesfilter.testing`;
- added `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`.

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `11 passed, 2 warnings`.

Interpretation:
- SVD cubature, SVD-UKF, and SVD-CUT4 now share a testing diagnostic
  vocabulary for value results;
- value branch summaries cover Models A-C across all three backends;
- score branch summaries are intentionally affine-only until nonlinear Models
  B-C derivative providers exist;
- continuation to R3 is justified.

### Phase R3: CPU approximation benchmark

Implementation:
- added `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`;
- generated:

```text
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.json
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.md
```

Benchmark command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py --repeats 1
```

Result:
- Model A matches exact Kalman to roundoff for SVD cubature, SVD-UKF, and
  SVD-CUT4;
- Model B dense one-step projection log-likelihood errors were about
  `1.68e-2` for cubature/UKF and `3.64e-3` for CUT4;
- Model C dense one-step projection log-likelihood errors were about
  `4.93e-2` for cubature, `1.49e-1` for UKF, and `3.39e-2` for CUT4;
- all tiny value branch parameter boxes were `3/3` finite;
- the JSON artifact is strict JSON and converts infinite scalar diagnostics to
  null.

Interpretation:
- CUT4 improves the one-step projection diagnostics in these two nonlinear
  examples, but at larger point count;
- Models B-C still do not have exact full nonlinear likelihood references in
  this pass;
- continuation to R4-R7 is justified.

### Phase R4-R7: CI tiers, deferrals, documentation, and provenance

R4 result:
- `pytest.ini` already has `extended`, `hmc`, `external`, and `gpu` markers;
- new branch diagnostic tests are fast CPU tests;
- benchmark scripts remain explicit artifacts outside default pytest.

R5 result:
- GPU/XLA gate deferred because this pass did not require escalated CUDA
  execution and because nonlinear Models B-C score providers are still absent.

R6 result:
- nonlinear HMC readiness remains blocked for Models B-C pending explicit
  derivative providers and score branch diagnostics.

R7 implementation:
- updated `docs/chapters/ch28_nonlinear_ssm_validation.tex` with current V1
  evidence and limitations;
- added `docs/plans/bayesfilter-v1-nonlinear-filtering-remaining-master-program-result-2026-05-12.md`;
- registered the result and benchmark artifacts in `docs/source_map.yml`.

Next phase justified?
- Yes, but the next phase should be derivative-provider work for Models B-C,
  not GPU/XLA or HMC.

Final validation for this pass:
- focused nonlinear suite:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `54 passed, 2 skipped, 2 warnings`.

Additional checks:
- `python -m py_compile` passed for touched Python modules/tests/scripts;
- `git diff --check` passed;
- `docs/source_map.yml` parsed with `yaml.safe_load`;
- nonlinear benchmark JSON parsed with `python -m json.tool`;
- NumPy scan in `bayesfilter/nonlinear` and the new diagnostic helper found no
  matches;
- `latexmk -cd -pdf -interaction=nonstopmode -halt-on-error docs/main.tex`
  completed successfully, with pre-existing undefined citation/reference
  warnings elsewhere in the monograph.

Completion interpretation:
- third subplan is complete for CPU value diagnostics, branch summaries,
  benchmark artifacts, and provenance;
- GPU/XLA and nonlinear HMC are intentionally deferred;
- next justified phase is derivative-provider work for Models B-C.

## 2026-05-12 audit: nonlinear implementation correctness

User request:
- audit the whole implementation to ensure correctness;
- commit the resulting scoped changes.

Lane boundary:
- stayed inside the BayesFilter V1 nonlinear-filtering lane;
- left MacroFinance, DSGE, Chapter 18b, structural SVD/SGU plans, and the
  shared monograph reset memo untouched;
- ignored unrelated dirty/untracked files from other lanes.

Audit artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-implementation-audit-result-2026-05-12.md
```

Result:
- no major implementation correctness issue was found in the audited nonlinear
  filtering lane;
- one diagnostic exposure gap was fixed: placement/innovation floor counts and
  PSD-projection residuals are now exposed through `diagnostics.extra` for SVD
  cubature, SVD-UKF, and SVD-CUT4 value wrappers;
- focused tests now assert that the shared nonlinear diagnostic snapshot sees
  those fields.

Verification:
- full default CPU suite: `185 passed, 5 skipped, 2 warnings`;
- extended SVD-CUT CPU branch diagnostics: `2 passed, 2 warnings`;
- focused nonlinear regression after the fix: `24 passed, 2 warnings`;
- `py_compile`, `git diff --check`, `docs/source_map.yml` YAML parsing, and
  nonlinear benchmark JSON parsing passed.

Interpretation:
- current V1 nonlinear claims remain valid: exact Kalman parity for Model A,
  value-filter and dense one-step projection diagnostics for Models B-C, and
  smooth affine score certification only;
- derivative providers for Models B-C, nonlinear Hessian certification,
  GPU/XLA scaling, and nonlinear HMC remain gated follow-up work.

## 2026-05-13 execution start: nonlinear derivative-provider gap closure

User request:
- create a plan to close the remaining nonlinear gaps and test the hypotheses;
- tighten and audit the plan;
- execute phase-by-phase with plan, execute, test, audit, tidy, reset-memo
  updates;
- continue automatically only when primary criteria and veto diagnostics pass;
- commit scoped V1-lane files at completion.

Plan under execution:

```text
docs/plans/bayesfilter-v1-nonlinear-derivative-provider-gap-closure-plan-2026-05-13.md
```

Audit artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-derivative-provider-gap-closure-plan-audit-2026-05-13.md
```

Lane boundary:
- stay inside the BayesFilter V1 nonlinear-filtering lane;
- do not edit or stage MacroFinance, DSGE, Chapter 18b, structural SVD/SGU
  plans, or the shared monograph reset memo;
- existing unrelated dirty/untracked files remain out of scope.

Tightened execution direction:
- close derivative-provider and score-diagnostic gaps for Models B-C where the
  current smooth SVD score branch supports the law;
- keep Hessian, GPU/XLA, and nonlinear HMC gated until score parity and branch
  diagnostics pass;
- treat the default Model C zero phase variance as a branch-diagnostic issue,
  not as something to hide with a silent numerical floor.

### Phase D0: baseline and drift check

Baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Result:
- `22 passed, 2 warnings`.

Additional status:
- branch `main` remains ahead of `origin/main` by 12 commits before this pass;
- the shared monograph reset memo and unrelated untracked files are dirty but
  out of lane and unstaged;
- scan for NumPy imports in production nonlinear code and the nonlinear testing
  helpers under this pass found no matches.

Interpretation:
- baseline is clean;
- continuation to Model B derivative-provider implementation is justified.

### Phase D1-D2: Model B and Model C derivative providers

Implementation:
- added `make_nonlinear_accumulation_first_derivatives_tf`;
- added `make_univariate_nonlinear_growth_first_derivatives_tf`;
- changed the testing fixtures so Model B and Model C parameters can be tensors
  as well as Python scalars;
- added optional `initial_phase_variance` to the Model C testing fixture while
  preserving the default zero-variance phase benchmark;
- exported the derivative providers from `bayesfilter.testing`.

Formula audit:
- Model B derivatives match
  \(m_t=\rho m_{t-1}+\sigma\varepsilon_t\),
  \(k_t=\alpha k_{t-1}+\beta\tanh(m_t)\), and \(y_t=m_t+k_t+u_t\);
- Model C derivatives match the phase-state nonlinear growth law, quadratic
  observation, observation variance derivative \(2\sigma_y\), and initial
  covariance derivative with respect to \(P_{0,x}\);
- all derivative providers obey the analytic score contract: map Jacobians and
  parameter partials are evaluated with sigma-point inputs held fixed.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  -p no:cacheprovider
```

Result:
- `13 passed, 2 warnings`.

Interpretation:
- H-D1 is supported: Model B has an explicit derivative provider without
  changing the production structural callback contract;
- H-D2 is supported: Model B analytic scores match centered finite differences
  for SVD cubature, SVD-UKF, and SVD-CUT4 on the selected smooth branch;
- H-D3 is partially supported: Model C analytic scores match centered finite
  differences for all three backends on a nondegenerate phase-state testing
  variant;
- H-D4 is supported: the default zero-phase-variance Model C fixture blocks the
  current smooth SVD score branch by active floor, so it is not promoted to a
  default score-ready law.

### Phase D3: branch summary extension

Implementation:
- extended score branch summaries from affine-only to affine, Model B, and
  smooth-phase Model C;
- added a default Model C branch-summary blocker test that counts the zero
  phase variance as an active-floor blocker.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `12 passed, 2 warnings`.

Interpretation:
- branch diagnostics now separate score-ready nonlinear fixtures from the
  default deterministic-degenerate Model C fixture;
- continuation to documentation and final validation is justified.

### Phase D4: documentation, hygiene, and final validation

Documentation/result updates:
- updated `docs/chapters/ch28_nonlinear_ssm_validation.tex` with Model B and
  Model C derivative-provider equations and current score-claim boundaries;
- added result artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-derivative-provider-gap-closure-result-2026-05-13.md
```

- registered plan, audit, result, code, test, and chapter artifacts in
  `docs/source_map.yml`.

Focused nonlinear/V1 validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `62 passed, 2 skipped, 2 warnings`.

Full default CPU validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:
- `193 passed, 5 skipped, 2 warnings`.

Hygiene:
- `py_compile` passed for touched Python modules/tests;
- `git diff --check` passed;
- `docs/source_map.yml` parsed with `yaml.safe_load`;
- scan for NumPy imports in production nonlinear code and the touched nonlinear
  testing helpers found no matches.

Completion interpretation:
- derivative-provider gap is closed for Model B on the selected score branch;
- Model C derivative-provider correctness is established on a nondegenerate
  phase-state testing variant;
- default Model C remains correctly blocked by the smooth SVD active-floor
  gate;
- Hessian, GPU/XLA, nonlinear HMC, and default-Model-C structural sigma-point
  score work remain future phases.

## 2026-05-13 master consolidation

User request:
- create one single coherent V1 master program because the direction had become
  dispersed and could cause drift.

Created:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

Role:
- this is now the controlling roadmap for the BayesFilter V1 lane;
- earlier V1 plans remain supporting evidence, not competing control documents;
- future subplans should cite the master and name the gap they close.

Master-program ordering:
1. Chapter 18b structural sigma-point score derivation and implementation for
   deterministic-completion models, with default Model C as the first target;
2. wider nonlinear score branch diagnostics;
3. nonlinear benchmark refresh with score metadata;
4. Model B nonlinear HMC readiness plan and tiny CPU smoke;
5. optional nonlinear GPU/XLA scaling diagnostics;
6. nonlinear Hessian need assessment;
7. optional exact-reference strengthening for Models B-C;
8. future MacroFinance/DSGE integration plan.

Anti-drift rules recorded:
- do not start GPU/XLA, HMC, Hessian, or external-client work before its
  predecessor gate passes;
- do not promote diagnostic artifacts to certified claims;
- do not edit MacroFinance, DSGE, Chapter 18b, structural SVD/SGU plans, or the
  shared monograph reset memo from this lane;
- do not add production NumPy dependencies;
- every future V1 subplan must name the master-program gap it closes.

## 2026-05-13 structural sigma-point score pivot

User correction:
- the next phase should not invent a separate fixed-null SVD derivative theory;
- Chapter 18b already gives the correct structural sigma-point filtering
  derivation and should govern the solution.

Master update:
- replaced the old fixed-null phrasing in
  `docs/plans/bayesfilter-v1-master-program-2026-05-13.md`;
- R1 is now Chapter 18b structural sigma-point score implementation for
  deterministic-completion models, with default Model C as the first target.

New subplan:

```text
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md
```

Subplan phases:
1. derive the needed structural score mathematics from Chapter 18b with
   MathDevMCP support and write the concrete document block;
2. update the master/subplan from the derivation result;
3. implement the derivative according to that derivation;
4. audit code-document consistency with MathDevMCP/code comparison tools;
5. test against default Model C;
6. finish with result artifacts, source-map updates, tests, and scoped commit.

Interpretation:
- default Model C should be treated as a structural deterministic-completion
  score problem, not as a collapsed full-state SVD null-space problem;
- HMC, GPU/XLA, and nonlinear Hessian remain blocked until this structural
  score phase resolves.

## 2026-05-13 structural sigma-point score execution: plan tightening and audit

Phase:
- pre-S0 plan tightening and independent audit.

Files updated:

```text
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-audit-2026-05-13.md
```

Result:
- tightened the plan to preserve the backend's declared sigma-rule dimension
  while representing structurally null phase directions by zero factor columns
  and zero factor derivatives;
- recorded an independent audit finding that resizing the CUT4 rule to the
  active rank would drift from the current V1 backend comparison;
- confirmed that HMC, GPU/XLA, Hessian, MacroFinance, DSGE, and SGU work stay
  out of this phase.

Interpretation:
- continuation to S0 is justified because the plan now has a concrete
  non-drift path: derive the structural pre-transition score, then implement a
  structural fixed-support placement check without adding any fake phase
  nugget.

## 2026-05-13 structural sigma-point score execution: S0 derivation

Phase:
- S0, derivation from Chapter 18b with MathDevMCP support.

Files updated:

```text
docs/chapters/ch18_svd_sigma_point.tex
```

MathDevMCP evidence:
- looked up `prop:bf-structural-ukf-pushforward`;
- looked up `eq:bf-structural-ukf-moment-objects`;
- looked up `eq:bf-structural-ukf-loglik-object`;
- checked the Gaussian innovation score algebra against the Chapter 18b
  likelihood sign convention.

Derivation result:
- added `sec:bf-svd-sp-structural-fixed-support-score` to Chapter 18;
- stated the structural score variable as
  \(A_t=(x_{t-1},\varepsilon_t)\), with state completion
  \(X_t^{(r)}=F_\theta(A_t^{(r)})\) and observation points
  \(Z_t^{(r)}=h_\theta(X_t^{(r)})\);
- derived the fixed-support spectral branch for structural zero directions:
  active columns use the usual simple-spectrum eigenderivative formula, fixed
  null columns are zero and have zero derivative, and the null derivative block
  must be checked rather than floored;
- derived the structural point, moment, innovation, and likelihood score
  equations.

Interpretation:
- S0 gives a concrete implementation contract, so S1/S2 remain justified;
- the default Model C path should be a structural fixed-support score branch,
  not a positive phase nugget and not a collapsed full-state regularization
  branch;
- HMC, GPU/XLA, and nonlinear Hessian remain blocked.

## 2026-05-13 structural sigma-point score execution: S1 plan update

Phase:
- S1, update the controlling V1 plan from the S0 derivation.

Files updated:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md
```

Result:
- the master now points R1 to the Chapter 18 structural fixed-support score
  subsection;
- the subplan now records the concrete S2 contract: preserve declared
  sigma-rule dimension, keep structural null factor columns at zero, and block
  moving-null, floor-regularized, or weak-gap branches.

Interpretation:
- S2 implementation is justified because the derivation has produced an
  auditable code contract and a falsifiable default Model C test.

## 2026-05-13 structural sigma-point score execution: S2 implementation

Phase:
- S2, implement the structural fixed-support score path.

Files updated:

```text
bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py
bayesfilter/testing/nonlinear_diagnostics_tf.py
tests/test_nonlinear_sigma_point_scores_tf.py
tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py
```

Implementation:
- added an opt-in `allow_fixed_null_support` branch to the SVD sigma-point
  score path;
- preserved the declared sigma-rule dimension for cubature, UKF, and CUT4;
- represented structural zero directions by zero factor columns and zero
  factor derivatives;
- separated `structural_null_count` from true placement floor activation;
- added diagnostics for `sigma_point_variable`,
  `fixed_null_derivative_residual`, and the structural fixed-support branch;
- left the old smooth no-active-floor branch unchanged unless the new option is
  explicitly requested.

Validation:

```bash
python -m py_compile \
  bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py \
  bayesfilter/testing/nonlinear_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `py_compile` passed;
- focused pytest result: `35 passed, 2 warnings`.

Interpretation:
- S2 supports H-S1 on the focused test scope: default Model C now has an
  analytic structural fixed-support score path for SVD cubature, SVD-UKF, and
  SVD-CUT4;
- the old smooth branch still blocks default Model C, which is the desired
  guard against accidental promotion without the structural option;
- continuation to S3 is justified.

## 2026-05-13 structural sigma-point score execution: S3 audit

Phase:
- S3, code-document consistency audit.

Tool status:
- MathDevMCP `compare_label_code`, `compare_doc_code`, and one
  `extract_latex_context` call failed internally on the new local label, so
  those calls are not counted as certifying evidence;
- MathDevMCP `search_code_docs` did find the implementation terms
  `pre_transition_structural`, `structural_fixed_support_no_active_floor`, and
  `fixed_null_derivative_residual` in the code;
- manual audit checked the code against
  `sec:bf-svd-sp-structural-fixed-support-score`.

Manual audit result:
- code places points on the augmented pre-transition variable
  `[state, innovation]`;
- code completes state points with `model.transition` and observation points
  with `model.observe`;
- code reports `sigma_point_variable = pre_transition_structural`;
- code keeps structural null factor columns at zero in the opt-in branch;
- code blocks positive placement floors, positive covariance in a structural
  null direction, moving null derivative blocks, and weak active spectral gaps;
- code reports `structural_null_count`,
  `structural_null_covariance_residual`,
  `fixed_null_derivative_residual`,
  `factor_derivative_reconstruction_residual`, and
  `deterministic_residual`.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py::test_model_c_default_structural_fixed_support_score_matches_finite_difference \
  tests/test_nonlinear_sigma_point_scores_tf.py::test_model_c_structural_fixed_support_blocks_positive_placement_floor \
  tests/test_nonlinear_sigma_point_scores_tf.py::test_model_c_structural_fixed_support_blocks_moving_null_covariance \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py::test_default_model_c_structural_fixed_support_score_branch_summary_passes \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py::test_default_model_c_structural_fixed_support_diagnostics \
  -p no:cacheprovider
```

Result:
- `11 passed, 2 warnings`;
- `git diff --check` passed.

Interpretation:
- code-document consistency is supported by manual audit and targeted tests,
  with transparent tool limitation noted;
- continuation to S4 is justified.

## 2026-05-13 structural sigma-point score execution: S4 default Model C validation

Phase:
- S4, test against default Model C and focused V1 regressions.

Documentation update:
- updated `docs/chapters/ch28_nonlinear_ssm_validation.tex` so default Model C
  is no longer described as waiting for a future fixed-null branch;
- Chapter 28 now states that default Model C is score-ready under the
  structural fixed-support branch, while the old smooth branch still blocks it.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `73 passed, 2 skipped, 2 warnings`.

Full default CPU validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:
- `204 passed, 5 skipped, 2 warnings`.

Interpretation:
- default Model C structural fixed-support score is validated against finite
  differences for SVD cubature, SVD-UKF, and SVD-CUT4;
- R1 is closed for the local V1 testing target;
- continuation to S5 finishing and commit is justified.

## 2026-05-13 structural sigma-point score execution: S5 completion

Phase:
- S5, finish artifacts and update provenance.

Artifacts:

```text
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-result-2026-05-13.md
docs/source_map.yml
```

Completion interpretation:
- R1 is closed on the tested V1 scope;
- wider branch-grid diagnostics are still justified as the next phase;
- nonlinear HMC, GPU/XLA scaling, nonlinear Hessians, and external switch-over
  remain blocked or deferred until their master-program gates are reached.

## 2026-05-13 structural sigma-point score execution: final pre-commit check

Phase:
- final lane audit before commit.

Lane status:
- staged scope should include only V1 structural sigma-point score files,
  tests, Chapter 18/28 updates, V1 plan/result/audit artifacts, the V1 reset
  memo, and `docs/source_map.yml`;
- the shared monograph reset memo and unrelated structural/DSGE/MacroFinance
  untracked files remain out of lane and must not be staged.

Validation rerun:

```bash
python -m py_compile \
  bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py \
  bayesfilter/testing/nonlinear_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py

git diff --check

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider

python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"
```

Result:
- `py_compile` passed;
- `git diff --check` passed;
- focused score/branch tests: `36 passed, 2 warnings`;
- full default CPU suite: `204 passed, 5 skipped, 2 warnings`;
- `source_map ok`.

Completion interpretation:
- the structural fixed-support score branch is ready to commit for the V1
  lane;
- the next justified phase remains wider nonlinear branch-grid diagnostics for
  Model B and default Model C before nonlinear HMC, GPU/XLA, or Hessian work.
  This next-phase statement was superseded by the 2026-05-14 master
  reconciliation below, which inserts P1 derivative-validation matrix before
  branch-grid diagnostics.

## 2026-05-14 V1 master derivative-validation reconciliation

User request:
- rewrite the master program and affected subplans after identifying that the
  master did not explicitly include a derivative-validation matrix for the
  nonlinear model suite.

Files updated:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p2-branch-diagnostics-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p3-benchmark-refresh-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p8-external-integration-plan-2026-05-14.md
docs/plans/bayesfilter-v1-master-and-subplans-audit-2026-05-14.md
docs/source_map.yml
```

Result:
- the master now makes P1 a nonlinear derivative-validation matrix over
  Models A-C and SVD cubature, SVD-UKF, and SVD-CUT4;
- the matrix must record value status, score status, branch label, derivative
  provider, reference target, compiled/eager parity where available, and
  Hessian status for every model/backend cell;
- wider branch diagnostics moved to P2;
- benchmark refresh moved to P3;
- nonlinear HMC, Hessian consumer assessment, GPU/XLA, exact references, and
  external integration shifted to P4-P8 with explicit gates;
- fresh master-aligned subplans now exist for all remaining phases.
- post-audit tightening updated the master phase table and P1/P2 action text
  so the master points to the existing subplans instead of implying they still
  need to be created.

Audit:
- `docs/plans/bayesfilter-v1-master-and-subplans-audit-2026-05-14.md`
  confirms that the rewrite fixes the derivative-validation planning gap,
  preserves Hessian deferral unless a consumer is named, keeps default Model C
  tied to the structural fixed-support branch, and stays inside the V1 lane.

Interpretation:
- P1, not branch-grid diagnostics, is now the next safe execution phase;
- P1 is documentation/test-evidence consolidation first and should implement
  new tests only for explicit missing matrix cells;
- nonlinear Hessians remain status-recorded but not implemented unless P5 names
  a concrete consumer.

## 2026-05-14 V1 P1 derivative-validation matrix execution

Phase:
- P1 / R2 nonlinear derivative-validation matrix.

Plan:

```text
docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-result-2026-05-14.md
```

Execution result:
- built the consolidated matrix for Models A-C and SVD cubature, SVD-UKF, and
  SVD-CUT4;
- recorded value status, score status, score branch, derivative provider,
  reference target, branch/null diagnostics, compiled/eager status, Hessian
  status, and public-claim scope for every row;
- clarified that Model A score evidence is from the parameterized smooth
  affine structural score fixture, while Model A value evidence is from the
  fixed affine Gaussian structural oracle;
- kept production nonlinear Hessians deferred and kept the SVD-CUT4 autodiff
  score/Hessian oracle as testing-only evidence;
- confirmed default Model C score certification only under the Chapter 18
  structural fixed-support branch with `allow_fixed_null_support=True`.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `41 passed, 2 warnings`.

Interpretation:
- P1 passes: no matrix cell has unknown value or score status, and Hessian
  status is explicit without implying production Hessian support;
- P2 is justified next: run wider nonlinear score branch diagnostics for
  Model B and default Model C, with default Model C using
  `allow_fixed_null_support=True`.

## 2026-05-14 V1 P2 branch diagnostics execution

Phase:
- P2 / R3 wider nonlinear score branch diagnostics.

Plan:

```text
docs/plans/bayesfilter-v1-p2-branch-diagnostics-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p2-branch-diagnostics-result-2026-05-14.md
```

Implementation:
- extended the V1 testing diagnostic summary with structural-null residual
  maxima and bounded representative failure labels;
- tightened branch diagnostic tests so active-floor, weak-gap, and structural
  fixed-support diagnostics remain visible.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `18 passed, 2 warnings`.

Branch grid result:
- Model B passed `5/5` score rows for SVD cubature, SVD-UKF, and SVD-CUT4;
- smooth-phase Model C passed `5/5` score rows for all three backends;
- default Model C passed `5/5` score rows for all three backends only under
  `allow_fixed_null_support=True`;
- no active floors, weak spectral gaps, nonfinite scores, or hidden failure
  labels appeared on the tested boxes;
- default Model C structural-null residuals remained at numerical zero scale.

Interpretation:
- P2 passes;
- P3 is justified next: refresh nonlinear benchmark artifacts with score
  branch metadata and keep exactness claims tied to the actual reference type;
- HMC, Hessians, GPU/XLA, and external integration remain gated.

## 2026-05-14 V1 P3 benchmark refresh execution

Phase:
- P3 / R4 nonlinear benchmark refresh with score metadata.

Plan:

```text
docs/plans/bayesfilter-v1-p3-benchmark-refresh-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p3-benchmark-refresh-result-2026-05-14.md
```

Implementation:
- refreshed `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`
  so benchmark rows include score status, score branch labels, finite-score
  status, failure labels, structural-null diagnostics, and
  `score_allow_fixed_null_support`;
- corrected the benchmark's parameterized affine score branch grid to avoid a
  repeated-spectrum weak-gap artifact;
- removed stale benchmark blocked claim that Models B-C analytic scores are
  blocked; retained blocked claims for exact full nonlinear likelihood,
  nonlinear HMC readiness, GPU/XLA speedup, and nonlinear Hessian readiness.

Generated:

```text
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json
docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.md
```

Validation:

```bash
python -m py_compile docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Result:
- syntax check passed;
- `23 passed, 2 warnings`.

Benchmark command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter \
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py \
  --repeats 1 \
  --output docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.md
```

Result:
- benchmark completed with CPU-only logical devices in the JSON artifact;
- all Model A, Model B, and default Model C rows have value branch `3/3` and
  score branch `3/3` for SVD cubature, SVD-UKF, and SVD-CUT4.

Interpretation:
- P3 passes;
- P4 is justified next with Model B as the first nonlinear HMC target
  candidate;
- HMC convergence, nonlinear Hessians, GPU/XLA speedups, and external
  integration remain gated.

## 2026-05-14 V1 P4 nonlinear HMC target execution

Phase:
- P4 / R5 nonlinear HMC target selection and tiny CPU smoke.

Plan:

```text
docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-result-2026-05-14.md
```

Implementation:
- selected Model B nonlinear accumulation with SVD-CUT4 analytic score as the
  first nonlinear HMC target;
- added opt-in testing helper `ModelBNonlinearSVDTarget` and
  `run_model_b_nonlinear_svd_cut4_hmc_smoke`;
- added opt-in tests in
  `tests/test_hmc_nonlinear_model_b_readiness_tf.py`;
- wrote tiny-smoke diagnostics to
  `docs/benchmarks/bayesfilter-v1-model-b-nonlinear-hmc-smoke-2026-05-14.json`.

Validation:

```bash
python -m py_compile \
  bayesfilter/testing/tf_hmc_readiness.py \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py \
  -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py \
  -p no:cacheprovider
```

Results:
- syntax check passed;
- default HMC test behavior: `3 skipped, 2 warnings`;
- public API guard: `2 passed, 2 warnings`;
- opt-in HMC readiness: `3 passed, 2 warnings`.

Tiny-smoke diagnostics:
- finite samples: `8`;
- nonfinite samples: `0`;
- acceptance rate: `1.0`;
- branch diagnostics: `5/5`, no active floors, no weak spectral gaps, no
  nonfinite branch rows;
- convergence was not claimed.

Interpretation:
- P4 passes at tiny CPU smoke scope;
- nonlinear Hessians are not needed for this ordinary score-only HMC target;
- P5 is justified as a decision-only Hessian consumer assessment, and should
  keep Hessian work deferred unless a concrete consumer is named.

## 2026-05-14 V1 P5 Hessian consumer assessment

Phase:
- P5 / R6 nonlinear Hessian consumer assessment.

Plan:

```text
docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-result-2026-05-14.md
```

Decision:
- no concrete V1 nonlinear Hessian consumer is currently named;
- P4's Model B SVD-CUT4 HMC smoke used ordinary score-only HMC dynamics and
  therefore did not create a Hessian requirement;
- production nonlinear SVD sigma-point results keep `hessian=None` and
  Hessian-deferred status;
- the SVD-CUT4 autodiff Hessian-like path remains testing-only in
  `bayesfilter.testing.tf_svd_cut_autodiff_oracle`;
- no nonlinear Hessian implementation subplan was opened.

Validation:

```bash
git diff --check
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"
```

Result:
- whitespace check passed;
- source-map YAML parsed successfully.

Interpretation:
- P5 passes by the explicit-deferral branch;
- P6 remains optional and diagnostic only;
- any P6 GPU/CUDA/XLA command must use escalated sandbox permissions and any
  speedup claim must be limited to the tested shapes.

## 2026-05-14 V1 P6 GPU/XLA scaling diagnostic

Phase:
- P6 / R7 optional GPU/XLA scaling diagnostics.

Plan:

```text
docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-result-2026-05-14.md
```

Implementation:
- added benchmark-only diagnostic harness
  `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py`;
- generated CPU-hidden control artifacts:
  `docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.json`
  and `.md`;
- generated GPU-visible diagnostic artifacts:
  `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json`
  and `.md`;
- no production filter behavior or public API was changed.

Validation:

```bash
python -m py_compile docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py
nvidia-smi  # escalated
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py \
  --device-scope cpu --repeats 2 --warmup-calls 1 --timesteps 24 \
  --backends tf_svd_cut4 --modes eager,graph,xla --devices cpu \
  --output docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.md
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py \
  --device-scope visible --repeats 2 --warmup-calls 1 --timesteps 24 \
  --backends tf_svd_cut4 --modes eager,graph,xla --devices cpu,gpu \
  --output docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.md
```

Result:
- escalated `nvidia-smi` saw the NVIDIA GPU;
- GPU-visible TensorFlow saw `/device:GPU:0`;
- all Model B SVD-CUT4 rows had branch `3/3` with no active floors, weak
  spectral gaps, or nonfinite rows;
- CPU-hidden control steady seconds: eager `0.088768`, graph `0.004820`,
  XLA `0.000391`;
- GPU-visible steady seconds: CPU eager `0.086636`, CPU graph `0.005458`,
  CPU XLA `0.022035`, GPU eager `0.305209`, GPU graph `0.061104`, GPU XLA
  `0.022038`.

Interpretation:
- P6 passes as a diagnostic artifact;
- GPU/XLA is operational for this tested Model B SVD-CUT4 shape;
- this tiny shape does not support a broad GPU speedup claim;
- future GPU work should test larger horizons, larger dimensions, batched
  parameter points, or batched independent filters before making performance
  claims.

## 2026-05-14 V1 P7 exact-reference strengthening assessment

Phase:
- P7 / R8 optional exact nonlinear reference strengthening.

Plan:

```text
docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-result-2026-05-14.md
```

Plan tightening:
- P7 now explicitly allows a deferral branch when no current V1 claim requires
  a stronger nonlinear reference than exact Model A and dense one-step
  projection diagnostics for Models B-C.

Decision:
- no dense multi-step quadrature, high-particle SMC, or production reference
  dependency was added;
- Model A keeps exact Kalman reference status;
- Models B-C remain dense one-step Gaussian projection diagnostics only;
- exact full nonlinear likelihood for Models B-C remains blocked/deferred.

Validation:

```bash
git diff --check
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"
```

Result:
- whitespace check passed;
- source-map YAML parsed successfully after replacing a colon-bearing
  provenance sentence with YAML-safe wording.

Interpretation:
- P7 passes by explicit deferral;
- P8 is justified as planning-only external integration because local V1
  evidence is current and optional claims are scoped.

## 2026-05-14 V1 P8 external integration planning

Phase:
- P8 / R9 external client integration plan.

Plan:

```text
docs/plans/bayesfilter-v1-p8-external-integration-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-p8-external-integration-result-2026-05-14.md
```

Decision:
- MacroFinance linear QR compatibility remains the first external client
  target;
- the first bridge should be test-only or optional-live, not a production
  switch-over;
- DSGE remains future/test-only, with Rotemberg preferred over SGU if the DSGE
  lane supplies a causal local filtering law and residual contract;
- SGU remains blocked as a V1 production filtering target;
- no MacroFinance, DSGE, shared monograph, structural-lane, or Chapter 18b file
  was edited.

Integration gates:
- BayesFilter default CPU and focused V1 suites must pass on a release
  candidate;
- optional live MacroFinance compatibility should run on a clean recorded
  external checkout before any client switch-over branch;
- client adapters must pin parameter order, dtype, Hessian sign convention,
  deterministic-completion diagnostics, and regularization metadata;
- optional external checks remain outside default CI.

Interpretation:
- P8 passes as a reviewable integration plan;
- final validation should now run the master-program focused/default gates
  before final summary and push.

## 2026-05-14 V1 master execution closeout

Final summary artifact:

```text
docs/plans/bayesfilter-v1-master-program-execution-summary-2026-05-14.md
```

Final validation:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Result:
- `2 passed, 2 warnings`.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `73 passed, 2 skipped, 2 warnings`.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:
- `204 passed, 8 skipped, 2 warnings`.

Interpretation:
- the BayesFilter V1 master program P1-P8 execution is complete at the scoped
  evidence level;
- nonlinear value/score evidence is current;
- tiny nonlinear HMC, GPU/XLA, Hessian, exact-reference, and external-client
  claims are all labeled at their true scope;
- remaining work should proceed through opt-in MacroFinance compatibility,
  longer nonlinear HMC diagnostics, larger/batched GPU ladders, and
  consumer-gated Hessian/reference phases.

## 2026-05-15 V1 Model B/C BC0 baseline reconciliation

Phase:
- BC0 / baseline reconciliation for the Model B/C thorough testing master
  program.

Plan:

```text
docs/plans/bayesfilter-v1-model-bc-bc0-baseline-reconciliation-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-model-bc-bc0-baseline-matrix-result-2026-05-15.md
```

Execution:
- read the current nonlinear benchmark, HMC smoke, GPU/XLA diagnostic, and
  P1-P7 result artifacts;
- populated the six Model B/C x SVD-filter baseline rows;
- did not run new numerical experiments.

Gate result:
- BC0 passed;
- no Model B/C/filter cell has unknown status;
- default Model C rows explicitly use the structural fixed-support branch with
  `allow_fixed_null_support=True`;
- dense one-step projection remains diagnostic only;
- tiny HMC smoke and one-shape GPU/XLA rows remain diagnostic only;
- nonlinear Hessians remain deferred because no consumer is named.

Continuation:
- BC1 is justified for wider branch-box testing.

## 2026-05-15 V1 Model B/C BC1 branch boxes

Phase:
- BC1 / wider value-score branch boxes.

Plan:

```text
docs/plans/bayesfilter-v1-model-bc-bc1-wider-branch-boxes-plan-2026-05-14.md
```

Result artifact:

```text
docs/plans/bayesfilter-v1-model-bc-bc1-branch-boxes-result-2026-05-15.md
```

Benchmark artifacts:

```text
docs/benchmarks/bayesfilter-v1-model-bc-branch-boxes-2026-05-15.json
docs/benchmarks/bayesfilter-v1-model-bc-branch-boxes-2026-05-15.md
docs/benchmarks/benchmark_bayesfilter_v1_model_bc_testing.py
```

Execution:
- added a BayesFilter-local Model B/C testing harness for row-level BC1-BC3
  artifacts;
- ran the predeclared deterministic and seeded BC1 branch boxes on CPU with
  `CUDA_VISIBLE_DEVICES=-1`;
- used seed `20260515` for seeded random rows.

Gate result:
- BC1 passed;
- all 60 predeclared rows were stable;
- Model B stayed on the smooth simple-spectrum score branch;
- Model C score rows used structural fixed support with
  `allow_fixed_null_support=True`;
- no row reported active score floors, weak gaps, nonfinite values/scores, or
  unlabeled failures.

Continuation:
- BC2 is justified for all six Model B/C x filter cells.
