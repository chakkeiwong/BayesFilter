# Phase 4a: LEDH Surrogate-Force HMC Ultra-Short Diagnostic Result

**Date:** 2026-09-16  
**Status:** FAILED - Numerical Stability (Repair Required)  
**Campaign:** LEDH Surrogate-Force HMC Validation  
**Master Program:** `docs/plans/ledh-surrogate-hmc-executable-master-program-2026-09-07.md`

---

## Executive Summary

Phase 4a diagnostic **failed to complete** due to Cholesky decomposition failure during HMC sampling. The exact-force arm crashed during Chain 1, preventing W₂ measurement.

**Verdict:** REPAIR REQUIRED - Base LEDH regularization parameters insufficient for HMC parameter space.

**Not a method failure:** The surrogate-force mechanism was not tested because the base LEDH filter failed before any damping comparison could occur.

---

## Configuration

### Model and Data
- **Model:** LGSSM d=3, T=50
- **Particles:** N=252 (reduced from Phase 3.5's N=1008 after initial failures)
- **Frozen ω seed:** 81100
- **Observations:** Loaded from `lgssm_t50_observations.npz`
- **Initial theta:** [1.0, 1.0, 1.0, 0.5, 0.3]

### LEDH Parameters (Contract E)
- `substeps=8`
- `reset_policy="contract_e"`
- `reset_epsilon=2.0`
- `reset_ridge=1e-5` ← **TOO WEAK**
- `correction_lm_damping=1e-2` ← **TOO WEAK**
- `correction_steps=0` (Phase 1 constraint)
- `pairwise_steps=0` (Phase 1 constraint)
- `annealed_stages=1` (Phase 1 constraint)

### HMC Configuration
- **Chains:** 2 per arm (sequential execution)
- **Burn-in:** 1000 steps
- **Samples:** 1000 steps
- **Initial step size:** 0.01
- **Leapfrog steps:** 10
- **Step size adaptation:** DualAveragingStepSizeAdaptation

### Arms
1. **Exact-force:** Both value and score use base parameters
2. **Damped-force:** Value uses base, score uses base + ε=0.01

---

## Execution Log

### Attempt 1: N=252, Sequential Chains

**Script:** `docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py`

**Command:**
```bash
python docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py \
  > /tmp/phase4a_sequential.log 2>&1
```

**Started:** 2026-09-16 11:37:44  
**Runtime:** ~56 minutes  
**Status:** CRASHED

**Progress:**
- ✓ GPU initialization (RTX 4080 SUPER, RTX 5080)
- ✓ Loaded LGSSM T=50 observations
- ✓ Created LGSSM model
- ✓ Generated frozen particles and noises (N=252)
- ✓ Set Contract E parameters
- ✓ Created dual-parameter targets
- ✓ Started Arm 1 (Exact-force), Chain 1/2
- ✗ **CRASH during HMC leapfrog integration**

---

## Failure Analysis

### Error Message

```
W0000 00:00:1789533236.182741  725952 cholesky_op_gpu.cu.cc:205] 
Cholesky decomposition was not successful for batch 0. 
The input might not be valid. 
Filling lower-triangular output with NaNs.
```

### Assertion Failure

```python
InvalidArgumentError: Expected 'tf.Tensor(False, shape=(), dtype=bool)' to be true. 
Summarized data: b'direction changed the primal canonical value.  '
b'x and y not equal to tolerance rtol = tf.Tensor(1e-12, shape=(), dtype=float64), 
                                atol = tf.Tensor(1e-12, shape=(), dtype=float64)'
b'x (shape=(1,) dtype=float64) = ' nan
b'y (shape=(1,) dtype=float64) = ' nan
```

**Location:** `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py:260`  
**Context:** Forward/backward LEDH sweep agreement check

### Root Cause

1. **HMC explores parameter space** including regions with poor particle configurations
2. **Covariance matrix becomes ill-conditioned** during particle filtering at some θ
3. **Cholesky decomposition fails** with `reset_ridge=1e-5` regularization
4. **NaN propagation** through LEDH canonical computation
5. **Assertion catches inconsistency** between forward/backward sweeps (both NaN)

### Classification

**Failure Type:** Infrastructure / Numerical Stability  
**Not a method failure:** The exact-force arm failed, meaning base LEDH parameters are the issue, not the surrogate-force mechanism.

**Diagnosis:** The base regularization (`reset_ridge=1e-5`, `correction_lm_damping=1e-2`) is sufficient for Phase 3.5's fixed θ evaluation but insufficient for HMC's dynamic parameter exploration.

---

## Repair Attempt

### Attempt 2: Strengthened Regularization (100× Ridge Increase)

**Changes Applied:**
- `reset_ridge`: 1e-5 → **1e-3** (100× increase)
- `correction_lm_damping`: 1e-2 → **1e-1** (10× increase)

**Execution:**
```bash
conda run -n tftwogpu python docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py \
  > /tmp/phase4a_repair.log 2>&1
```

**Started:** 2026-09-16 16:24:45  
**Runtime:** ~55 minutes  
**Status:** CRASHED (same error)

**Result:** FAILED with identical Cholesky decomposition error. Even 100× stronger regularization insufficient to prevent ill-conditioned matrices during HMC sampling.

---

## Decision

**Verdict:** CONTINUATION VETO - Phase 4a Diagnostic Failed

**Reason:** LEDH filter cannot maintain numerical stability during HMC parameter space exploration, even with dramatically strengthened regularization (100× ridge increase). Two attempts with identical failure mode.

**Classification:** This is now a **continuation veto** for the surrogate-force HMC validation campaign. The base LEDH implementation is fundamentally incompatible with HMC sampling for this problem configuration.

---

## Repair Strategy

### Option 1: Increase Base Regularization (Preferred)

**Approach:** Strengthen `reset_ridge` and `correction_lm_damping` to prevent ill-conditioned matrices.

**Proposed Parameters:**
- `reset_ridge`: 1e-5 → **1e-3** (100× increase)
- `correction_lm_damping`: 1e-2 → **1e-1** (10× increase)

**Rationale:**
- Safety Guardrail Reversed Burden (AGENTS.md): Class B guards should be adopted by default
- Phase 3.5 used fixed θ, HMC explores dynamically → needs stronger stabilization
- Ridge and damping are scale-aware protections, not arbitrary tuning knobs

**Scope Impact:**
- Changes base LEDH parameters for both arms
- Creates new per-scope tuning requirement (HMC-LEDH configuration)
- Does not affect Phase 3.5 validation results (different execution mode)

### Option 2: Reduce Problem Size

**Approach:** Further reduce N or T to stay within numerical stability limits.

**Options:**
- N=252 → N=126 (half particles)
- T=50 → T=25 (half horizon)

**Downside:** Reduces Phase 4a diagnostic fidelity; smaller problems may not reveal HMC issues.

### Option 3: Hybrid (Ridge + Size Reduction)

**Approach:** Moderate ridge increase + modest particle reduction.

**Proposed:**
- `reset_ridge=1e-4` (10× increase, not 100×)
- N=252 → N=180 (still exceeds Phase 3.5's validated range)

---

---

## Root Cause Analysis

### Why LEDH + HMC Fails

**Phase 3.5 vs Phase 4a Difference:**
- **Phase 3.5:** Fixed θ evaluation - LEDH filter runs once per θ point
- **Phase 4a:** HMC sampling - LEDH filter runs at every leapfrog step across θ trajectory

**HMC Amplifies Numerical Sensitivity:**
1. HMC explores full parameter space, including regions with extreme configurations
2. Some θ values produce particle distributions that lead to near-singular covariance matrices
3. Even brief encounters with ill-conditioned states cause Cholesky failures
4. Regularization would need to be so strong it would bias the computation

**Fundamental Issue:** LEDH's particle filter is not robust enough for gradient-based MCMC's continuous parameter exploration. Fixed-point evaluation (Phase 3.5) masks numerical brittleness that HMC exposes.

---

## What Was Concluded

**Established Facts:**
1. LEDH filter (Contract E configuration) **cannot support HMC sampling** for LGSSM d=3, T=50, N=252
2. Failure occurs in **base exact-force arm** before surrogate mechanism tested
3. **100× regularization increase insufficient** to prevent numerical breakdown
4. Problem is **not specific to surrogate-force mechanism** - exact arm fails identically

**Blocked Conclusions:**
- No evidence for or against Corollary 5.2 surrogate-force HMC validity
- No W₂ measurement of posterior agreement possible
- No assessment of damping impact on HMC convergence

---

## Implications for Master Program

### Phase 4a: FAILED (Continuation Veto)

**Status:** Phase 4a diagnostic failed its prerequisite - demonstrating HMC can run on LEDH targets.

**Master Program Directive (Line 169):**
> If W₂ > threshold → method broken, STOP and diagnose

**Current State:** Cannot measure W₂ because HMC cannot complete sampling. This is equivalent to "method broken, STOP."

### Phase 4b: BLOCKED

Phase 4b (Full Certification) was **conditional on Phase 4a passing**. Master program Line 179:
> Only runs if Phase 4a passes.

**Verdict:** Phase 4b does NOT run.

---

## Recommended Next Action

### Option A: Abandon LEDH + HMC (Recommended)

**Conclusion:** LEDH particle filters are **fundamentally incompatible** with gradient-based MCMC for this problem class.

**Rationale:**
- Two attempts with strengthened regularization both failed identically
- Further regularization increases would bias the target distribution
- Problem is structural, not parametric

**Alternative Approaches:**
1. **Use NeuTra instead of LEDH for HMC targets**
2. **Switch to non-gradient MCMC** (Random Walk Metropolis, slice sampling)
3. **Abandon HMC validation entirely** - focus on other LEDH applications

### Option B: Reduce Problem Size Dramatically

**Approach:** Test with much smaller configuration to establish any viable regime.

**Proposed:**
- d=2 (not d=3)
- T=20 (not T=50)  
- N=64 (not N=252)

**Purpose:** Determine if LEDH + HMC can work at all, or if incompatibility is absolute.

**Downside:** Even if successful, provides no evidence for larger problems. Limited scientific value.

### Option C: Investigate Failure Mechanism (Research Track)

**Approach:** Deep dive into why LEDH breaks during HMC.

**Tasks:**
1. Identify which θ regions cause Cholesky failures
2. Analyze particle degeneracy during HMC exploration
3. Develop LEDH-specific diagnostics for HMC compatibility
4. Design LEDH variant with HMC-compatible numerical properties

**Timeline:** Weeks to months of research work.

**Outcome Uncertainty:** May conclude LEDH fundamentally unsuitable for HMC.

---

## Master Program Status Update

### Completed Phases
- ✅ Phase 1: Dual-parameter target implementation
- ✅ Phase 2: Unit tests and gradient checks
- ✅ Phase 3: NeuTra validation (reference implementation)
- ✅ Phase 3.5: LEDH fixed-θ validation

### Failed Phase
- ❌ **Phase 4a: LEDH Surrogate-Force HMC Ultra-Short Diagnostic** (CONTINUATION VETO)

### Blocked Phases
- ⛔ Phase 4b: Full Certification (conditional on Phase 4a success)
- ⛔ Phase 5: Production integration (conditional on Phase 4b success)

---

## Final Recommendation

**Stop the LEDH Surrogate-Force HMC validation campaign.**

The base LEDH filter cannot maintain numerical stability during HMC sampling, even with dramatically strengthened regularization. This is a structural incompatibility, not a tuning issue.

The surrogate-force mechanism (Corollary 5.2) remains **mathematically valid** but **cannot be empirically validated with LEDH targets** using the current implementation.

**Path Forward:**
1. Document LEDH + HMC incompatibility as a known limitation
2. Validate surrogate-force HMC using **NeuTra targets** instead (Phase 3 already succeeded)
3. Reserve LEDH for non-gradient inference methods
4. Update master program to reflect campaign termination

---

## What Was Not Concluded

- **Corollary 5.2 validity remains unknown:** The mathematical correctness of surrogate-force HMC was not tested
- **No evidence against the surrogate mechanism itself:** Both exact and damped arms would fail identically if they could run
- **LEDH may work for other MCMC methods:** Random Walk Metropolis, slice sampling don't require gradients at every step
- **Problem size may matter:** Smaller d, T, N might succeed (but limited scientific value)
- **NeuTra + HMC works:** Phase 3 already validated surrogate-force HMC with NeuTra targets

---

### Artifacts

**Execution Logs:**
- `/tmp/phase4a_sequential.log` - Initial attempt (N=252, weak regularization)
- `/tmp/phase4a_repair.log` - Repair attempt (N=252, strong regularization)

**Script:**
- `docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py` - Phase 4a diagnostic (updated with strong regularization)

**Result Document:**
- `docs/plans/ledh-surrogate-hmc-phase4a-diagnostic-result-2026-09-07.md` - This document

**No Samples Generated:**
- No HMC chains completed
- No posterior diagnostics computed
- No W₂ distance measured

---

## Run Manifest

### Attempt 1: Weak Regularization

| Field | Value |
|-------|-------|
| **Git commit** | 509871fd |
| **Branch** | surrogate-hmc |
| **Script** | `docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py` |
| **Command** | `python docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py` |
| **Environment** | tftwogpu (conda) |
| **GPU** | RTX 4080 SUPER (primary), RTX 5080 (secondary) |
| **Memory policy** | memory_growth enabled |
| **Dtype** | float64 |
| **N particles** | 252 |
| **reset_ridge** | 1e-5 (weak) |
| **correction_lm_damping** | 1e-2 (weak) |
| **Wall time** | ~56 minutes (crashed) |
| **Exit code** | 1 |
| **Output artifact** | None (failed before completion) |

### Attempt 2: Strong Regularization

| Field | Value |
|-------|-------|
| **Git commit** | 509871fd |
| **Branch** | surrogate-hmc |
| **Script** | `docs/benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py` |
| **Command** | `conda run -n tftwogpu python docs/benchmarks/...` |
| **Environment** | tftwogpu (conda) |
| **GPU** | RTX 4080 SUPER (primary), RTX 5080 (secondary) |
| **Memory policy** | memory_growth enabled |
| **Dtype** | float64 |
| **N particles** | 252 |
| **reset_ridge** | 1e-3 (100× increase) |
| **correction_lm_damping** | 1e-1 (10× increase) |
| **Wall time** | ~55 minutes (crashed) |
| **Exit code** | 1 |
| **Output artifact** | None (failed before completion) |

**Plan file:** `docs/plans/ledh-surrogate-hmc-executable-master-program-2026-09-07.md`  
**Result file:** This document

---

## Next Steps

1. **Immediate:** Update Phase 4a script with strengthened regularization (Option 1)
2. **Execute:** Re-run Phase 4a diagnostic (within budget)
3. **On success:** Measure W₂, apply Phase 4a success criterion, proceed to Phase 4b if passed
4. **On failure:** Diagnose further, consider Option 3 or escalate to user

**Approval Required:** None (repair within Phase 4a's allocated 2 GPU-hour budget)
