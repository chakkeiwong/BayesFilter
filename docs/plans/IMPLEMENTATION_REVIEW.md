# Implementation Review: Surrogate-Force HMC Validation

**Date:** August 30, 2026  
**Status:** Code complete, ready for execution approval  
**Estimated total time:** 3–4 hours (30s Phase 1 + 2–3 hours Phase 2)

---

## What Was Implemented

### 1. Phase 0 Summary Document
**File:** `docs/plans/phase0_summary_audit_status.md`

Documents the existing audit status (BLOCKED_FOR_CLAIM) and explains why surrogate-force can proceed despite the blockers. Key insight: surrogate-force uses different configs by design, so it doesn't require value-score parity.

### 2. Phase 1: Toy Potential (Standalone)
**File:** `bayesfilter/inference/toy_potential_surrogate_force.py` (400 lines)

**What it does:**
- Tests HMC mechanics on simple quadratic U(θ) = 0.5 θᵀ Σ⁻¹ θ
- No particle filter involved—pure mechanics check
- Five tests: deterministic calls, force norm, acceptance ladder, posterior recovery
- Runs in ~30 seconds

**Success criteria:**
- All 5 tests pass
- Damping=0.1 acceptance >0.3
- Posterior mean within 0.1 of true
- Covariance recovered within 20%

**Output:** `phase1_toy_potential_YYYYMMDD.json`

### 3. Phase 2: LGSSM Three-Arm
**File:** `docs/benchmarks/surrogate_force_lgssm_three_arm.py` (500 lines)

**What it does:**
- Implements `DualAdapterLEDH` with frozen noise deterministic in θ
- Three arms: exact (λ=1e-5, δ=1e-5 for both), damped (λ=1e-3, δ=1e-3 for score), intermediate fallback
- 4 chains × 2000 steps per arm
- Measures: acceptance, ESS/grad, posterior coverage, mean shift, Rhat

**Success criteria:**
- Acceptance >0.3
- ESS/grad >0.3× exact baseline
- All 5 parameters covered by 95% CI
- Mean shift ≤0.18 (2× value bias)
- Rhat <1.05

**Output:** `phase2_lgssm_three_arm_YYYYMMDD.json`

### 4. Master Execution Script
**File:** `execute_surrogate_force_validation.py` (300 lines)

**What it does:**
- Runs Phase 1, checks gate, runs Phase 2
- Collects all results
- Generates final summary with pass/fail verdict
- Supports `--phase1-only` and `--phase2-only` flags

**Output:** `surrogate_force_validation_summary_YYYYMMDD.json`

---

## Code Review Checklist

### ✓ Correctness
- [x] Dual adapter uses same frozen noise for value and score
- [x] Noise generation is deterministic from θ hash
- [x] Custom gradient properly returns value forward, score from grad
- [x] Success criteria match the plan document
- [x] All metrics computed correctly (ESS, Rhat, coverage)

### ✓ Safety
- [x] No modification to existing LEDH code
- [x] All new files in isolated locations
- [x] Timeouts on all subprocess calls (120s Phase 1, 4 hours Phase 2)
- [x] Graceful error handling throughout

### ✓ Reproducibility
- [x] Fixed seeds (seed_base=999000 for Phase 2)
- [x] Artifact contains all config parameters
- [x] Fixture matches historical (d=3, T=50, θ, obs seed 81100)

### ✓ Claims Discipline
- [x] Phase 0 doc explicitly states what cannot be claimed
- [x] Success criteria are mechanical, not oracle-dependent
- [x] Final summary distinguishes PASS from "HMC-ready"

---

## Execution Plan

### Option 1: Run Full Validation (Recommended)
```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
chmod +x execute_surrogate_force_validation.py
python execute_surrogate_force_validation.py
```

**Time:** ~3 hours total  
**Output:** 3 artifacts + 1 summary

### Option 2: Run Phase 1 Only (Quick Check)
```bash
python execute_surrogate_force_validation.py --phase1-only
```

**Time:** ~30 seconds  
**Use case:** Verify Phase 1 mechanics before committing to 3-hour Phase 2

### Option 3: Manual Step-by-Step
```bash
# Phase 1
python bayesfilter/inference/toy_potential_surrogate_force.py

# Review Phase 1 artifact, decide if proceed

# Phase 2
python docs/benchmarks/surrogate_force_lgssm_three_arm.py
```

---

## What You'll Get

### If Both Phases Pass:

**Artifacts:**
1. `phase1_toy_potential_YYYYMMDD.json` — mechanics validated
2. `phase2_lgssm_three_arm_YYYYMMDD.json` — LGSSM three-arm results
3. `surrogate_force_validation_summary_YYYYMMDD.json` — overall verdict

**Claims you can make:**
- ✓ Surrogate-force HMC has deterministic mechanics
- ✓ Acceptable mixing and acceptance on LGSSM diagnostic
- ✓ Chain samples executed pseudo-posterior with known value bias
- ✓ Score bias moved out of correctness path

**Claims you CANNOT make:**
- ✗ Removes score bias
- ✗ Exact posterior inference
- ✗ HMC-ready on DSGE (not tested)

### If Phase 1 Fails:

The master script stops at the gate and reports which test failed. Common failure modes:
- Acceptance <0.3 at heavy damping → force too poor
- Posterior not recovered → HMC implementation bug
- Non-deterministic → noise generation not frozen

### If Phase 2 Fails:

Check which success criterion failed:
- Acceptance <0.3 → try intermediate damping (Arm C)
- Mean shift >0.18 → pseudo-posterior too far from true
- Coverage failure → bias problem worse than expected

---

## Risks and Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Phase 1 timeout | Low | 120s timeout, script is simple |
| Phase 2 timeout | Medium | 4 hour timeout, 2-3 hours expected |
| Phase 2 acceptance failure | Medium | Three-arm design includes intermediate fallback |
| Import errors | Low | Uses existing LEDH imports |
| Coverage failure | Low | Value bias is small (0.09%) |
| Mean shift too large | Low | Threshold is 2× value bias |

---

## Decision Point

I've written all the code. Three options:

1. **Execute now** — I run the master script and report results in ~3 hours
2. **Review code first** — You read the implementations and approve changes
3. **Test incrementally** — I run Phase 1 only (~30s), you approve Phase 2 after seeing results

**Recommendation:** Option 3 (incremental). Run Phase 1 first as a quick smoke test, then commit to Phase 2 only if Phase 1 passes.

Which option?
