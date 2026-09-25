# Phase 0: Test Audit, Coverage, and Refactor Contract — Subplan

**Program**: `ledh-while-loop-refactor-2026-08-30`  
**Phase**: 0 (Foundation)  
**Status**: AUDIT COMPLETE, REPAIR PENDING  
**Created**: 2026-08-30  
**Owner Approval Required**: YES (before repair execution)

---

## Objective

Establish a verified baseline for the refactor by:
1. Auditing the current test suite (health, coverage, failure root causes)
2. Installing and configuring coverage tooling (pytest-cov)
3. Repairing the identified `flow_substeps`/`substeps` API drift (test-side + kernel-side)
4. Measuring baseline coverage of the refactor target kernel
5. Writing the binding refactor contract (mathematical, API, implementation constraints)

**Success Criteria**:
- All parity tests in `test_ledh_canonical_batch_fused.py` pass (3/3)
- All parity tests in `test_ledh_canonical_batch.py` pass (3/3)
- Coverage tooling installed and configured
- Baseline coverage of `ledh_canonical_batch_fused_tf.py` measured and recorded
- Refactor contract written and accepted by owner

---

## Audit Findings (COMPLETE, 2026-08-30)

### Test Execution Summary

**Command**:
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/ \
  --ignore=tests/highdim/test_genut_shape_lm_tf.py \
  --ignore=tests/highdim/test_zhao_cui_austria_sir_lane_b_t2_score_tf.py \
  -k canonical -v
```

**Result**: 21 failed, 83 passed, 8 errors, 2396 deselected, 2 warnings in 206.16 s

### Failure Breakdown

#### 1. API Drift: `flow_substeps` vs `substeps` (6 test failures + 1 kernel bug)

**Affected Files**:
- `tests/highdim/test_ledh_canonical_batch_fused.py` (3 failures)
- `tests/highdim/test_ledh_canonical_batch.py` (3 failures)
- `bayesfilter/highdim/ledh_canonical_batch_tf.py` (1 kernel-side forwarding bug)

**Root Cause Analysis**:

The LEDH canonical route has three entry points with INCONSISTENT parameter naming:

| Entry Point | File | Signature | Line |
|-------------|------|-----------|------|
| Single-cloud authority | `ledh_canonical_score_tf.py` | `flow_substeps: int = 24` | 68 |
| Non-fused batch lane | `ledh_canonical_batch_tf.py` | `substeps: int` | 43 |
| Fused batch lane | `ledh_canonical_batch_fused_tf.py` | `substeps: int` | 43 |

**Test-Side Defects**:
- `test_ledh_canonical_batch_fused.py` lines 111, 119, 136, 143, 161: call `canonical_batch_fused_value_score(..., flow_substeps=10)`
  - Kernel expects `substeps=`
  - Error: `TypeError: canonical_batch_fused_value_score() got an unexpected keyword argument 'flow_substeps'`
  
- `test_ledh_canonical_batch.py` line 83: calls `canonical_batch_value_score(..., flow_substeps=10)`
  - Kernel expects `substeps=`
  - Error: `TypeError: canonical_batch_value_score() got an unexpected keyword argument 'flow_substeps'`

**Kernel-Side Defect**:
- `ledh_canonical_batch_tf.py` line 75: forwards `substeps=substeps` to `canonical_value_and_analytical_score`
  - Single-cloud authority expects `flow_substeps=`
  - This call SHOULD fail but was never tested because the test-side defect prevents reaching this line
  - Discovered via source audit 2026-08-30

**Repair Strategy**:
- **Option A** (SELECTED): Preserve batch-lane `substeps=` naming (shorter, matches role), fix tests and forwarding
  - Change test calls: `flow_substeps=10` → `substeps=10` (both test files)
  - Change `ledh_canonical_batch_tf.py:75`: `substeps=substeps` → `flow_substeps=substeps`
  - Rationale: batch lanes are the caller-facing API; single-cloud authority is internal; minimize API surface changes
  
- **Option B** (REJECTED): Standardize on `flow_substeps=` everywhere
  - Would require changing both batch-lane signatures → breaks any existing callers outside the test suite
  - Higher risk

**Files to Repair** (line numbers verified by reading each file, 2026-08-30):

1. `tests/highdim/test_ledh_canonical_batch_fused.py`: **4 substitutions**
   (`flow_substeps=` → `substeps=`) at lines **119, 136, 143, 161**.
   **Line 109 must NOT change** — it belongs to a
   `canonical_value_and_analytical_score` call, i.e. the single-cloud
   authority, whose parameter genuinely is `flow_substeps`.

2. `tests/highdim/test_ledh_canonical_batch.py`: **5 substitutions**
   (`flow_substeps=` → `substeps=`) at lines **85, 104, 108, 125, 130**.
   **Line 80 must NOT change** — authority call, same reason.

3. `bayesfilter/highdim/ledh_canonical_batch_tf.py`: **1 substitution** at line
   **75** (`substeps=substeps` → `flow_substeps=substeps`). This is the only
   kernel-side defect; the other nine are test-side.

**Total: 10 substitutions across 3 files.**

Both authority call sites are protected. Do not apply a blanket
find-and-replace of `flow_substeps=` across these files — two of the eleven
occurrences are correct as written, and changing them would break the
parity gate the refactor depends on.

#### 2. Missing Frozen-Fixture JSONs (14 failures)

**Affected File**: `tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py`

**Missing Artifacts**:
- `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-tiny-fixture-freeze-v2-2026-07-14.json` (13 tests)
- `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-one-step-fixture-freeze-2026-07-14.json` (1 test)

**Root Cause**: Frozen-fixture contract tests reference artifacts not checked into worktree during Phase 5 work (July 2026)

**Disposition**: **OUT OF SCOPE** for this refactor
- These are Contract E optional-feature tests, not core parity gates
- The refactor target kernel (`canonical_batch_fused_value_score`) does not depend on these fixtures
- Record as pre-existing breakage; will be addressed separately if Contract E work resumes

#### 3. Missing Leaderboard Artifact (1 failure + 8 errors)

**Affected File**: `tests/highdim/test_complete_highdim_phase1_canonical_targets.py`

**Missing Artifact**: `docs/plans/artifacts/complete-highdim-leaderboard/phase1-canonical-targets-2026-07-11.json`

**Root Cause**: Leaderboard artifact directory does not exist on worktree

**Disposition**: **OUT OF SCOPE** for this refactor
- This is a leaderboard integration test, not a kernel parity test
- The refactor will use focused parity tests only (batch-size-1 vs authority, multi-row independence, compilability)
- Phase 4 will include a leaderboard SMOKE test (can it run?) but not full validation against frozen artifacts
- Record as pre-existing breakage

#### 4. Collection Errors (2 files, blocks 2396 tests)

**Affected Files**:
- `tests/highdim/test_genut_shape_lm_tf.py`: `ModuleNotFoundError: No module named 'bayesfilter.highdim.cubature_genut_batch_tf'`
- `tests/highdim/test_zhao_cui_austria_sir_lane_b_t2_score_tf.py`: `ImportError: cannot import name '_active_log_weight'`

**Disposition**: **OUT OF SCOPE** for this refactor
- Unrelated modules (GenUt cubature, Zhao-Cui Austria SIR lane)
- Not dependencies of the LEDH canonical route
- Workaround: use `--ignore=` for both files in all pytest runs

### Coverage Tooling Audit

**Current State**: ABSENT
- Only `pytest.ini` exists (markers, testpaths)
- No `pyproject.toml`, `setup.cfg`, or `tox.ini`
- `pytest-cov` not installed in tftwogpu conda env (verified via `pip list | grep pytest-cov` → empty)

**Implication**: Cannot measure baseline coverage of `ledh_canonical_batch_fused_tf.py` → blocks user requirement "ensure that we have good test coverage"

**Repair Action**: 
1. Install `pytest-cov` in tftwogpu env: `pip install pytest-cov`
2. Add coverage configuration to `pytest.ini`:
   ```ini
   [coverage:run]
   source = bayesfilter
   omit = 
       */tests/*
       */test_*.py
   
   [coverage:report]
   precision = 2
   show_missing = True
   skip_covered = False
   ```
3. Measure baseline coverage:
   ```bash
   CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch_fused.py \
     --cov=bayesfilter.highdim.ledh_canonical_batch_fused_tf \
     --cov-report=term-missing \
     --cov-report=html:htmlcov
   ```

### Test Inventory

**Canonical LEDH test files** (worktree `tests/highdim/`, subset):
- `test_ledh_canonical_batch_fused.py` (5863 B, 3 tests) ← **PRIMARY PARITY GATE**
- `test_ledh_canonical_batch.py` (5352 B, 3 tests) ← non-fused lane parity
- `test_ledh_canonical_filter.py` (8937 B) ← UKF lifecycle
- `test_ledh_canonical_score_full.py` (14100 B) ← full score recursion
- `test_ledh_canonical_score_stages.py` (2416 B) ← stage decomposition
- `test_ledh_canonical_score_step.py` (2936 B) ← single-step score
- `test_ledh_canonical_score_ukf_tangent.py` (5165 B) ← UKF tangent parity
- `test_ledh_canonical_ukf_lifecycle.py` (6200 B) ← UKF sigma points, predict, update
- `test_ledh_canonical_fisher_identity.py` (10498 B) ← Fisher information identity
- `test_ledh_canonical_flow_perparticle.py` (6770 B) ← per-particle flow
- `test_ledh_canonical_model_fidelity.py` (19550 B) ← model jacobian/tangent contracts
- `test_ledh_canonical_neutra_target.py` (4246 B) ← NeuTra target contract (batch-native)
- Plus ~59 more contract-e and phase-numbered files

**Total**: ~71 canonical LEDH test files (~462 KB), 2396+ tests when collection errors resolved

---

## Repair Plan

### Repair 1: Fix API Drift (3 files)

**File 1**: `tests/highdim/test_ledh_canonical_batch_fused.py`

**Changes** (5 substitutions):
```python
# Line 111 (inside test_fused_batch_size_one_parity)
OLD: canonical_batch_fused_value_score(..., flow_substeps=10)
NEW: canonical_batch_fused_value_score(..., substeps=10)

# Line 119 (inside test_fused_batch_size_one_parity, second call)
OLD: canonical_batch_fused_value_score(..., flow_substeps=10)
NEW: canonical_batch_fused_value_score(..., substeps=10)

# Line 136 (inside test_fused_rows_independent_and_distinct)
OLD: canonical_batch_fused_value_score(..., flow_substeps=10)
NEW: canonical_batch_fused_value_score(..., substeps=10)

# Line 143 (inside test_fused_rows_independent_and_distinct, second call)
OLD: canonical_batch_fused_value_score(..., flow_substeps=10)
NEW: canonical_batch_fused_value_score(..., substeps=10)

# Line 161 (inside test_fused_lane_is_tf_function_compilable)
OLD: canonical_batch_fused_value_score(..., flow_substeps=10)
NEW: canonical_batch_fused_value_score(..., substeps=10)
```

**File 2**: `tests/highdim/test_ledh_canonical_batch.py`

**Changes** (1 substitution):
```python
# Line 83 (inside test_batch_lane_value_score_parity)
OLD: canonical_batch_value_score(..., flow_substeps=10)
NEW: canonical_batch_value_score(..., substeps=10)
```

**File 3**: `bayesfilter/highdim/ledh_canonical_batch_tf.py`

**Changes** (1 substitution):
```python
# Line 75 (inside canonical_batch_value_score, forwarding to authority)
OLD: value, score = canonical_value_and_analytical_score(
        model,
        row_theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        substeps=substeps,  # ← WRONG
        with_score=True,
    )
NEW: value, score = canonical_value_and_analytical_score(
        model,
        row_theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        flow_substeps=substeps,  # ← CORRECT
        with_score=True,
    )
```

**Verification**:
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch_fused.py -v
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch.py -v
```

**Expected**: 6/6 tests pass (3 fused + 3 non-fused)

### Repair 2: Install Coverage Tooling

**Step 1**: Install pytest-cov
```bash
pip install pytest-cov
```

**Step 2**: Add coverage config to `pytest.ini`

**Current `pytest.ini`**:
```ini
[pytest]
markers =
    extended: marks tests as extended (deselect with '-m "not extended"')
    hmc: marks tests as requiring HMC sampling (slow)
    external: marks tests as requiring external resources
    gpu: marks tests as requiring GPU hardware
testpaths = tests
```

**Add coverage section**:
```ini
[coverage:run]
source = bayesfilter
omit = 
    */tests/*
    */test_*.py

[coverage:report]
precision = 2
show_missing = True
skip_covered = False
```

**Step 3**: Measure baseline coverage
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch_fused.py \
  --cov=bayesfilter.highdim.ledh_canonical_batch_fused_tf \
  --cov-report=term-missing \
  --cov-report=html:htmlcov
```

**Record**: 
- Line coverage % (target: ≥ 85% given the kernel's complexity)
- Missing lines (recorded in repair result document)
- HTML report saved to `htmlcov/` for inspection

**Acceptance**: Coverage measured and recorded; refactor MUST NOT decrease coverage below baseline

### Repair 3: Verify Clean Baseline

**After Repair 1+2 complete**, run the full canonical subset:
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/ \
  --ignore=tests/highdim/test_genut_shape_lm_tf.py \
  --ignore=tests/highdim/test_zhao_cui_austria_sir_lane_b_t2_score_tf.py \
  -k canonical -v
```

**Expected**:
- 83 passed (unchanged from audit)
- 6 passed (formerly failed, now repaired)
- 14 failed (phase5 fixtures, out of scope)
- 1 failed + 8 errors (phase1 leaderboard, out of scope)
- Total: **89 passed, 15 failed, 8 errors**

**Record**: Exact pass/fail counts in the Phase 0 result document

---

## Refactor Contract (Binding)

See [Master Program, Refactor Contract section](ledh-while-loop-refactor-master-program-2026-08-30.md#refactor-contract-binding) for the complete binding contract.

**Key Points** (for Phase 0 context):
- **Parity tolerance**: rtol 5e-4 (accounts for op-order differences)
- **Analytical score preserved**: no `GradientTape`, hand-derived forward-mode tangent
- **Batch-native contract**: theta [B, P] → value [B], score [B, P]
- **tf.function compilable**: stable `input_signature`, no runtime errors
- **No silent behavior changes**: numerical differences beyond tolerance are stop conditions
- **Out of scope**: missing fixtures, leaderboard artifacts, collection-error modules

---

## Deliverables

1. **Repaired Files** (3):
   - `tests/highdim/test_ledh_canonical_batch_fused.py` (5 substitutions)
   - `tests/highdim/test_ledh_canonical_batch.py` (1 substitution)
   - `bayesfilter/highdim/ledh_canonical_batch_tf.py` (1 substitution)

2. **Coverage Infrastructure**:
   - `pytest-cov` installed in tftwogpu env
   - `pytest.ini` updated with coverage config
   - Baseline coverage measured and recorded

3. **Phase 0 Result Document**: `ledh-while-loop-refactor-phase0-result-2026-08-30.md`
   - Repair actions taken
   - Baseline coverage report (line %, missing lines)
   - Clean baseline test counts (89 passed expected)
   - Lessons learned (if any)
   - Phase 1 pre-conditions verified

4. **Git Commit** (semantic):
   ```
   Phase 0 repair: fix flow_substeps/substeps API drift, install coverage
   
   - tests/highdim/test_ledh_canonical_batch_fused.py: rename flow_substeps= to substeps= (5 calls)
   - tests/highdim/test_ledh_canonical_batch.py: rename flow_substeps= to substeps= (1 call)
   - bayesfilter/highdim/ledh_canonical_batch_tf.py: fix forwarding to authority (flow_substeps=substeps)
   - pytest.ini: add coverage configuration
   - Baseline coverage: <recorded in result doc>
   
   Parity gates now pass: 6/6 tests (3 fused + 3 non-fused batch lane)
   Clean baseline: 89 passed, 15 failed (out of scope), 8 errors (out of scope)
   
   Phase 0 complete. Phase 1 pre-conditions satisfied.
   ```

---

## Success Criteria (Phase 0)

- [ ] API drift repaired: `test_ledh_canonical_batch_fused.py` passes 3/3
- [ ] API drift repaired: `test_ledh_canonical_batch.py` passes 3/3
- [ ] Kernel forwarding bug fixed: `ledh_canonical_batch_tf.py:75` forwards `flow_substeps=substeps`
- [ ] Coverage tooling installed: `pip list | grep pytest-cov` returns package
- [ ] Coverage config added: `pytest.ini` contains `[coverage:run]` and `[coverage:report]` sections
- [ ] Baseline coverage measured: recorded in Phase 0 result document
- [ ] Clean baseline verified: 89 passed, 15 failed (out of scope), 8 errors (out of scope)
- [ ] Phase 0 result document written
- [ ] Git commit created with semantic message
- [ ] Owner approval obtained: user reviews Phase 0 result and approves Phase 1 start

---

## Phase 1 Pre-Conditions (Verified After Phase 0 Repair)

Before Phase 1 execution begins, ALL of the following MUST be true:

1. **Parity gates functional**: `test_ledh_canonical_batch_fused.py` passes 3/3 (verified via pytest run)
2. **Coverage baseline established**: line coverage % of `ledh_canonical_batch_fused_tf.py` recorded
3. **Refactor contract accepted**: owner has approved the binding contract (A2 approval)
4. **Allowlist accepted**: owner has approved the tool/command permissions (A3 approval)
5. **Repair policy accepted**: owner has approved autonomous repair within declared bounds (A4 approval)
6. **Execution model accepted**: owner has approved phase-to-phase autonomous progression (A5 approval)

**If any pre-condition fails**: STOP, do not proceed to Phase 1, report to owner

---

## Risk Mitigation (Phase 0 Specific)

| Risk | Mitigation |
|------|-----------|
| `pytest-cov` installation fails | Fallback: manual coverage via code inspection (count covered lines in parity tests) |
| Parity tests still fail after API drift repair | STOP condition: investigate root cause, may indicate deeper signature mismatch |
| Baseline coverage unexpectedly low (< 70%) | Record as finding; Phase 1-3 must not decrease it further; consider adding focused tests in Phase 4 |
| Non-fused batch lane forwarding bug causes cascading failures | Fix is localized to one line; test runs will verify |

---

## Timeline Estimate

- **Repair 1** (API drift, 3 files): 10 min
- **Repair 2** (coverage tooling): 10 min
- **Repair 3** (verify clean baseline): 5 min (test runtime ~200 s)
- **Result document**: 10 min
- **Git commit**: 5 min

**Total**: ~30 min

---

## Next Phase

**Phase 1**: Single-Direction tf.while_loop Conversion  
**Pre-conditions**: All Phase 0 success criteria met, owner approval obtained  
**Subplan**: [ledh-while-loop-refactor-phase1-subplan-2026-08-30.md](ledh-while-loop-refactor-phase1-subplan-2026-08-30.md)

---

**END OF PHASE 0 SUBPLAN**
