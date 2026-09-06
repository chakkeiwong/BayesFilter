# Phase 1 Execution Handoff

**Date:** 2026-09-07  
**Authority:** ledh-surrogate-hmc-executable-master-program-2026-09-07.md  
**Estimated Time:** 0.5 day  
**GPU Required:** Yes (escalated, memory-growth check)

---

## Goal

Re-run existing diagnostics to verify current implementation state before building new tests.

---

## Prerequisites

Phase 0 must be complete (tolerance, seed policy, coverage implemented).

---

## Tasks

### Task 1.1: JVP Parity Tests (0.2 day)

**Goal:** Verify Contract E streaming JVP implementation matches reference.

**Files to check (DO NOT READ FULL FILES):**
- `bayesfilter/highdim/ledh_contract_e_streaming_tf.py` (lines 1188-1300 only — the two JVP functions)
- Existing test file location: `grep -r "contract_e.*jvp" tests/`

**Commands:**
```bash
# Find existing JVP tests
grep -r "test.*contract_e.*jvp\|test.*ledh.*jvp" tests/ -l

# Run tests (CPU-only, no GPU needed for parity check)
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  conda run -n tftwogpu python -m pytest \
  [test_file_path] -v --tb=short \
  > results/phase1-jvp-parity.txt 2>&1
```

**Success Criterion:**
- All JVP parity tests pass
- Write summary: `results/phase1-jvp-summary.json`

```json
{
  "task": "jvp_parity",
  "status": "PASS" | "FAIL",
  "tests_run": 5,
  "tests_passed": 5,
  "output_file": "results/phase1-jvp-parity.txt"
}
```

**Bounded I/O:**
- Use `grep` to find test file, don't search manually
- Read only test output summary (last 50 lines), not full output
- If tests fail, read only the failure traceback (< 100 lines)

---

### Task 1.2: Sinkhorn Convergence Check (0.1 day)

**Goal:** Verify Sinkhorn solver converges within iteration limits.

**Files to check:**
- Find Sinkhorn convergence test: `grep -r "sinkhorn.*converge\|sinkhorn.*iteration" tests/ -l`

**Commands:**
```bash
# Run Sinkhorn convergence tests
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  conda run -n tftwogpu python -m pytest \
  [test_file_path] -k "sinkhorn" -v --tb=short \
  > results/phase1-sinkhorn-convergence.txt 2>&1
```

**Success Criterion:**
- Sinkhorn converges within max iterations on test fixtures
- Write summary: `results/phase1-sinkhorn-summary.json`

**On Failure:**
- Sinkhorn non-convergence is a **veto** for Contract E
- Stop and report to user
- Do not proceed to Phase 2

---

### Task 1.3: GPU Memory Growth Verification (0.2 day)

**Goal:** Verify TensorFlow GPU memory growth is enabled (CLAUDE.md requirement).

**What to check:**
- Find initialization code: `grep -r "set_memory_growth\|memory_growth" bayesfilter/ -B 3 -A 3`
- Check if it's in a guaranteed-early location (before any GPU op)

**Test script (create):**
```python
# tests/infrastructure/test_gpu_memory_growth.py
import tensorflow as tf

def test_memory_growth_enabled():
    """Verify memory growth is enabled on all visible GPUs."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        # CPU-only environment, skip test
        return
    
    for gpu in gpus:
        config = tf.config.experimental.get_memory_growth(gpu)
        assert config, f"Memory growth not enabled on {gpu.name}"
        
def test_memory_growth_before_init():
    """Verify memory growth is set before any TF GPU operation."""
    # This test should be run in isolation (fresh Python process)
    # If it fails, it means some module initialized GPU before setting growth
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        return
        
    # Try to set memory growth — should succeed if not initialized yet
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            if "cannot be modified" in str(e):
                raise AssertionError(
                    f"GPU {gpu.name} already initialized before memory growth was set. "
                    "This violates CLAUDE.md TensorFlow GPU Memory Rule."
                )
```

**Commands:**
```bash
# Run with GPU access (escalated)
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 \
  conda run -n tftwogpu python -m pytest \
  tests/infrastructure/test_gpu_memory_growth.py -v \
  > results/phase1-memory-growth.txt 2>&1
  
# Also verify nvidia-smi works
nvidia-smi > results/phase1-nvidia-smi.txt 2>&1
```

**Success Criterion:**
- Memory growth enabled on GPU(s)
- `nvidia-smi` output confirms GPU visible
- Write summary: `results/phase1-memory-growth-summary.json`

**On Failure:**
- If memory growth not enabled → **blocking defect**
- Must fix before any GPU run
- See CLAUDE.md: "serious GPU runs must fail closed if this cannot be done"

---

## Success Criteria (Phase 1 Complete)

1. ✅ JVP parity tests pass
2. ✅ Sinkhorn convergence tests pass
3. ✅ GPU memory growth verified enabled
4. ✅ `nvidia-smi` confirms GPU access works

**All four must pass. Any failure blocks Phase 2.**

---

## Result Summary File

**Write:** `results/phase1-summary.json`

```json
{
  "phase": "1",
  "status": "PASS" | "FAIL",
  "date": "2026-09-07",
  "tasks_completed": [
    "jvp_parity",
    "sinkhorn_convergence",
    "gpu_memory_growth"
  ],
  "tests_passed": ["list of test names"],
  "blocking_failures": [],
  "artifacts": [
    "results/phase1-jvp-parity.txt",
    "results/phase1-sinkhorn-convergence.txt",
    "results/phase1-memory-growth.txt",
    "results/phase1-nvidia-smi.txt"
  ],
  "next_phase": "2"
}
```

---

## On Failure

**If JVP parity fails:**
- Blocking defect in Contract E implementation
- Write diagnosis: `results/phase1-jvp-failure-diagnosis.md`
- Stop and report to user

**If Sinkhorn convergence fails:**
- Blocking defect in OT solver
- Write diagnosis: `results/phase1-sinkhorn-failure-diagnosis.md`
- Stop and report to user

**If memory growth not enabled:**
- Blocking defect per CLAUDE.md policy
- Find where GPU is initialized: `grep -r "list_physical_devices\|list_logical_devices" bayesfilter/`
- Write diagnosis: `results/phase1-memory-growth-failure-diagnosis.md`
- Stop and report to user

**Do not proceed to Phase 2 if any task fails.**

---

## Bounded I/O Instructions

**DO:**
- Use `grep` to find test files and initialization code
- Read only targeted line ranges (±10 lines around grep matches)
- Write test outputs to files, read only summaries
- Run `nvidia-smi` to confirm GPU access

**DO NOT:**
- Read full test files "to understand what they do"
- Read full implementation files
- Print full pytest output to terminal
- Re-read files already grep'd

**If you need to verify a function exists:**
```bash
# Find function definition
grep -n "^def function_name" file.py

# Read just that function (if line 150)
# Use Read tool with offset=150, limit=30
```

---

## Completion Note Template

After Phase 1 completes, write `docs/plans/phase1-complete.md`:

```markdown
# Phase 1 Complete — [DATE]

**Status:** PASS / FAIL

**Diagnostics run:**
- JVP parity: PASS / FAIL
- Sinkhorn convergence: PASS / FAIL
- GPU memory growth: PASS / FAIL

**Tests passed:** [count]

**Blocking failures:** [list if any]

**Artifacts:**
- results/phase1-summary.json
- results/phase1-*.txt (test outputs)

**Next:** Phase 2 (toy potential test)

**Issues encountered:** [if any]

**GPU status:**
- Device: [from nvidia-smi]
- Memory growth: enabled / not enabled
```

---

**Execute Phase 1 with these constraints. Write the completion note when done.**
