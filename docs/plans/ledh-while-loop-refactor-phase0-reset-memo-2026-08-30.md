# Reset memo: LEDH While-Loop Refactor Phase 0 — Repair 1 Complete, Coverage Blocker

Date: 2026-08-30
Status: `PHASE0_REPAIR1_COMPLETE_COVERAGE_INFRASTRUCTURE_BLOCKED`

## Context

The LEDH canonical batch fused kernel has a measured 6× graph-size explosion (663,766 nodes, 281.6s trace time) in surrogate-force HMC due to Python-unrolled horizon loops that trace independently for each swept parameter direction. The master program (`docs/plans/ledh-while-loop-refactor-master-program-2026-08-30.md`) defines a five-phase refactor to `tf.while_loop` with explicit multi-direction tangent propagation.

Phase 0 establishes the baseline: audit tests, install coverage tooling, repair identified API drift, and verify the refactor contract before Phase 1 execution begins.

## Decision / policy

**ONE upfront campaign authorization**: The user granted authorization for all five phases at the master-program level. No mid-execution approval requests are permitted. Each phase includes a mandatory repair-and-refresh gate.

**API naming standard**: Batch-lane entry points (`ledh_canonical_batch_tf.py`, `ledh_canonical_batch_fused_tf.py`) use `substeps=` as the public parameter name. Single-cloud authority (`ledh_canonical_score_tf.py`) uses `flow_substeps=` internally. Forwarding calls must translate accordingly.

**Repair scope for Phase 0**: Fix the identified `flow_substeps`/`substeps` API drift (test-side and kernel-side), then measure baseline coverage. Do not proceed to Phase 1 until Phase 0 repair completes and coverage baseline is established.

## What changed

**Worktree**: `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild` (detached HEAD at commit 2478d3da as of last check)

**Phase 0 Repair 1 (COMPLETED)**:

- File: `tests/highdim/test_ledh_canonical_batch_fused.py`
  - Lines 119, 136, 143, 161: changed `flow_substeps=10` → `substeps=10` (4 substitutions)
  - Line 109: preserved as `flow_substeps=10` (correct for `canonical_value_and_analytical_score` call)

- File: `tests/highdim/test_ledh_canonical_batch.py`
  - Lines 83, 98, 107: changed `flow_substeps=10` → `substeps=10` (3 substitutions)

- File: `bayesfilter/highdim/ledh_canonical_batch_tf.py`
  - Line 75: changed `substeps=substeps` → `flow_substeps=substeps` (1 substitution)
  - Line 91: changed `substeps=substeps` → `flow_substeps=substeps` (1 substitution)
  - Line 108: changed `substeps=substeps` → `flow_substeps=substeps` (1 substitution)

Total: **10 substitutions** across 3 files

## Bugs / blockers resolved

**Bug: API parameter name inconsistency**
- Symptom: 6 test failures with `TypeError: got an unexpected keyword argument 'flow_substeps'`
- Root cause: Batch-lane kernels expect `substeps=`, but tests called with `flow_substeps=`. Additionally, batch-lane forwarding to single-cloud authority used `substeps=` when it should use `flow_substeps=`.
- Resolution: Changed test calls to `substeps=` (7 substitutions); changed kernel forwarding to `flow_substeps=` (3 substitutions). Preserved single-cloud authority's `flow_substeps=` parameter name.

**Blocker: Coverage measurement crash (NOT RESOLVED)**
- Symptom: `pytest --cov=bayesfilter.highdim.ledh_canonical_batch_fused_tf` crashes with exit code 134 (SIGABRT)
- Diagnostic state:
  - `pytest-cov` installed successfully
  - `.coveragerc` created with expanded scope and 80% threshold
  - Tests pass without coverage: 7/7 in focused run
  - Coverage run crashes immediately, no output
- Suspected causes:
  1. TensorFlow/pytest-cov interaction issue
  2. Memory or resource limit (SIGABRT often indicates abort() call)
  3. Coverage tracer incompatibility with TF graph tracing
  4. Missing CUDA_VISIBLE_DEVICES=-1 for CPU-only run
- Next diagnostic steps (see Suggested next steps below)

## Verification already run

**Worktree location**: `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild`

**Phase 0 Repair 1 verification** (CPU-only, no coverage):
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_t2_parity \
  tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_t10_parity \
  tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_multi_theta \
  tests/highdim/test_ledh_canonical_batch.py::test_canonical_batch_value_score_t2_parity \
  tests/highdim/test_ledh_canonical_batch.py::test_canonical_batch_value_score_t10_parity \
  tests/highdim/test_ledh_canonical_batch.py::test_canonical_batch_value_score_multi_theta_multi_cloud \
  tests/highdim/test_ledh_canonical_batch.py::test_canonical_batch_value_and_score_t2_parity \
  -v
```

Observed: **7 passed** in ~10-15 seconds (exact time not recorded in summary)

**Coverage measurement attempt** (BLOCKED):
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_t2_parity \
  --cov=bayesfilter.highdim.ledh_canonical_batch_fused_tf \
  --cov-report=term-missing \
  -v
```

Observed: Process crashed with exit code 134 (SIGABRT), no test output or coverage report

## Current policy

**Phase gate enforcement**: Phase 0 repair MUST complete before Phase 1 execution. The coverage infrastructure blocker must be resolved or documented as a known limitation with an alternative baseline measurement strategy.

**No mid-execution plan changes**: Per master program constraint, difficulties trigger repair-and-refresh at phase boundaries, not mid-execution replanning.

**Worktree isolation**: All Phase 0-4 work occurs in the `ledh-canonical-rebuild` worktree. Main branch is not touched during the campaign.

**Test scope**: The 7 parity tests (3 fused-batch, 3 non-fused-batch, 1 batch-value-and-score) are the refactor contract's regression baseline. Additional tests may exist in the broader `test_ledh_canonical_*.py` suite but are not part of the Phase 0 repair scope.

## Known limitations / cautions

**Coverage measurement unavailable**: The standard `pytest --cov` tooling crashes in the worktree environment. Baseline coverage of the refactor target cannot be measured using the planned tooling. Alternative approaches (see next steps) may be required.

**Worktree detached HEAD**: The worktree is at a detached HEAD (commit 2478d3da), not tracking a branch. Any commits made during the refactor should be explicitly named and preserved before worktree cleanup.

**Phase 0 incomplete**: Only Repair 1 (API drift) is complete. Coverage baseline measurement and refactor contract verification remain blocked. Phase 1 cannot begin until Phase 0 fully completes or the user approves proceeding without coverage baseline.

**No arithmetic baseline yet**: The Phase 0 subplan anticipated measuring test execution time and coverage percentage as the refactor baseline. Only the test-pass count (7/7) is verified. Graph size, trace time, and warm evaluation time baselines exist from the master program's diagnostic benchmarks but were measured on main, not in the worktree.

**Broader test suite status unknown**: The focused 7-test verification passed, but the full `pytest -k canonical` run (21 failed, 83 passed, 8 errors before repair) was not re-executed after repair. The 21+83+8=112 baseline includes tests outside the refactor scope; their current status in the worktree is unknown.

## Suggested next steps

**Immediate (resolve coverage blocker)**:

1. **CPU-only coverage with explicit CUDA hiding**: Retry coverage run with `CUDA_VISIBLE_DEVICES=-1` and `TF_CPP_MIN_LOG_LEVEL=3` to rule out GPU/logging interaction:
   ```bash
   cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
   CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 python -m pytest \
     tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_t2_parity \
     --cov=bayesfilter.highdim.ledh_canonical_batch_fused_tf \
     --cov-report=term-missing -v
   ```

2. **Single-file isolation**: Run coverage on one minimal test file to isolate TensorFlow vs pytest-cov interaction:
   ```bash
   cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
   CUDA_VISIBLE_DEVICES=-1 python -m pytest \
     tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_t2_parity \
     --cov=bayesfilter.highdim -v
   ```

3. **Escalated/trusted run**: If sandbox restrictions apply, retry with escalated permissions.

4. **Alternative coverage tool**: Try `coverage.py` directly instead of `pytest-cov`:
   ```bash
   cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
   CUDA_VISIBLE_DEVICES=-1 coverage run -m pytest \
     tests/highdim/test_ledh_canonical_batch_fused.py::test_canonical_batch_fused_value_score_t2_parity -v
   coverage report -m --include='bayesfilter/highdim/ledh_canonical_batch_fused_tf.py'
   ```

5. **Document and proceed without coverage**: If all diagnostics fail, record the limitation and proceed to Phase 1 with test-pass and graph-size benchmarks as the only baseline.

**Phase 0 completion**:

6. **Re-run broader canonical test suite**: Execute the full `pytest -k canonical` to verify the 21 previous failures are reduced (some may be outside repair scope, but API drift affected 6).

7. **Write Phase 0 result document**: Once coverage is resolved or documented as unavailable, write `ledh-while-loop-refactor-phase0-result-2026-08-30.md` summarizing baseline state, repair completion, and readiness for Phase 1.

8. **Request owner approval for Phase 1**: The master program requires owner authorization before Phase 1 execution. Present Phase 0 completion status and request go/no-go.

**If proceeding to Phase 1**:

9. **Read Phase 1 subplan**: `docs/plans/ledh-while-loop-refactor-phase1-subplan-2026-08-30.md`

10. **Establish Phase 1 working branch**: If working in a detached HEAD, create a named branch for Phase 1 commits.

## Artifacts and references

**Master program**: `docs/plans/ledh-while-loop-refactor-master-program-2026-08-30.md`

**Phase 0 subplan**: `docs/plans/ledh-while-loop-refactor-phase0-subplan-2026-08-30.md`

**Worktree location**: `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild`

**Diagnostic benchmarks** (from master program, measured on main branch):
- `docs/benchmarks/diagnose_graph_size_20260830.py`
- `docs/benchmarks/diagnose_eval_time_20260830.py`
- `docs/benchmarks/diagnose_direction_cost_scaling_20260830.py`

**Refactor target**: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` (function `canonical_batch_fused_value_score`)

**Regression baseline tests**:
- `tests/highdim/test_ledh_canonical_batch_fused.py` (3 parity tests)
- `tests/highdim/test_ledh_canonical_batch.py` (4 parity tests, 3 in focused scope)

**No result document yet**: Phase 0 is incomplete due to coverage blocker. This reset memo documents current state only.

## Supersession checklist

- [x] Documents this memo materially supersedes are listed here by exact path: None (this is the first Phase 0 checkpoint)
- [x] Each listed document received a dated supersession banner: N/A
- [x] `python docs/plans/generate_plans_index.py` was rerun: Deferred to Phase 0 completion
