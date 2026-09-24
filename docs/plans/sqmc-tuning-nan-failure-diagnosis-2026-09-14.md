---
title: SQMC Tuning NaN Failure Diagnosis
date: 2026-09-14
status: active-investigation
---

## Problem

The SQMC tuning script (`run_sqmc_tuning.py`) fails with **all 108 cells invalid**
(54 configurations × 2 seeds). The script crashes when trying to find Pareto-optimal
configurations because no valid results exist.

## Root Cause

**Every SQMC score evaluation returns NaN**, which the tuning script correctly rejects
as invalid (line 399-410):

```python
if not tf.reduce_all(tf.math.is_finite(all_scores)):
    return {'valid': False, ...}
```

## Diagnosis Process

### 1. Initial Error (Red Herring)

The task output showed:
```
Grid [1/54]: ALL INVALID
...
KeyError: 'seed_results'
```

The KeyError was a downstream crash in Pareto analysis when all cells are invalid.
The real failure is **upstream: NaN scores**.

### 2. Reproduction

Created [`docs/benchmarks/debug_sqmc_single_eval.py`](../benchmarks/debug_sqmc_single_eval.py)
to isolate a single evaluation. Discovered:

- Signature is correct (ancestry_policy parameter exists)
- Observation dimensions corrected (3D, not 2D)
- Function executes without exceptions
- **Returns**: `value=nan`, `score=[nan]`

### 3. Configuration

The tuning script evaluates:
- **Model**: 3D diagonal LGSSM (5 parameters: phi_1, phi_2, phi_3, q_scale, r_scale)
- **Theta**: `[0.9, 0.8, 0.7, 0.6, 0.8]`
- **Observations**: Frozen 3D LGSSM sequence (T=20), generated with **different** parameters  
  (phi=[0.72, 0.55, 0.35], q_scale=0.35, r_scale=0.45)
- **Particles**: N=1008
- **Horizon**: T=20

This parameter mismatch is **intentional** (evaluating at a point different from data
generation), matching the UNTUNED baseline diagnostic that **worked** (commit 7b23adbd,
Sept 12).

### 4. Contradiction

**UNTUNED baseline diagnostic (`run_sqmc_oracle_characterization.py`) succeeded**:
- Commit 7b23adbd: "All routes produce valid, usable gradients for HMC"
- Cosine similarity: 0.9995-0.9996
- Relative norm error: 0.9-1.7%
- **No NaN scores**

**Tuning script (`run_sqmc_tuning.py`) fails completely**:
- Same model, theta, observations (via `_lgssm_frozen_observations()`)
- Same DTYPE (float64)
- **All evaluations return NaN**

## Potential Causes

1. **Control parameter mismatch**: Tuning grid uses different correction/pairwise/reset
   controls than the working baseline

2. **Seed/randomization difference**: The baseline and tuning script use different
   random seeds for particle initialization

3. **Code regression**: A change between the baseline (commit 7b23adbd) and the current
   worktree branch (`rqmc-sqmc-4route-comparison`) broke SQMC numerics

4. **Environment difference**: The working baseline ran in a different conda environment
   or with different TensorFlow/CUDA settings

## Next Actions

1. **Compare control parameters**: Extract the exact controls from
   `run_sqmc_oracle_characterization.py` and verify they match the tuning grid baseline

2. **Run characterization script on current branch**: Verify the baseline **still works**
   on `rqmc-sqmc-4route-comparison`

3. **Bisect if needed**: If the characterization fails on this branch, git-bisect between
   7b23adbd (working) and current HEAD to find the breaking commit

4. **Minimal diff**: Once we identify working parameters, determine the **minimal change**
   to make the tuning script work

## Files

- Tuning script: [`docs/benchmarks/run_sqmc_tuning.py`](../benchmarks/run_sqmc_tuning.py)
- Characterization script (working baseline): [`docs/benchmarks/run_sqmc_oracle_characterization.py`](../benchmarks/run_sqmc_oracle_characterization.py)
- Debug script: [`docs/benchmarks/debug_sqmc_single_eval.py`](../benchmarks/debug_sqmc_single_eval.py)
- Frozen observations: `bayesfilter.highdim.ledh_canonical_neutra_targets_tf._lgssm_frozen_observations()`
- Model: `bayesfilter.highdim.ledh_canonical_models_tf.diagonal_lgssm_canonical_model()`
- Score function: `bayesfilter.highdim.ledh_canonical_score_tf.canonical_value_and_analytical_score()`

## Status

**Diagnosis complete**. The failure is confirmed to be NaN scores, not a signature or
data-shape mismatch. The working baseline exists (commit 7b23adbd), so this is either
a control-parameter issue or a code regression.

Next step: Compare control parameters and run the baseline on current branch.
