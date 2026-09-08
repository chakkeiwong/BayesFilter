# Phase 2B Step 1: Characterisation Tests

**Date:** 2026-09-09  
**Branch:** `surrogate-hmc`  
**Parent Program:** [ledh-surrogate-hmc-unified-program-2026-09-06.md](ledh-surrogate-hmc-unified-program-2026-09-06.md)  
**Status:** IN_PROGRESS

---

## Goal

Create golden-master fixtures and characterisation tests to lock down current `canonical_value_and_analytical_score` behavior before refactoring in Phase 2B Step 3.

---

## Deliverables

### 1. Golden-Master Fixtures

**Test matrix:**
- T ∈ {1, 2, 3, 10, 50}
- N ∈ {6, 64, 1008}
- d ∈ {2, 3}
- K (directions) ∈ {1, 2, 5}
- dtype ∈ {float64, float32}
- `annealed_stages` ∈ {1, 8}

**Total configurations:** 5 × 3 × 2 × 3 × 2 × 2 = 360 fixtures

**Tuned controls (from 2026-09-03 LGSSM tuning):**
```python
reset_policy="contract_e"
reset_epsilon=2.0
reset_sinkhorn_steps=8
reset_balance_steps=8
reset_ridge=1e-05
correction_steps=4
correction_strength=0.2
correction_lm_damping=0.001
correction_lm_scale_floor=1e-06
correction_trust_radius=0.1
pairwise_steps=4
pairwise_strength=0.02
pairwise_rms_cap=2.0
coordinate_cap=0.98
coordinate_cap_power=8
flow_substeps=24
```

**Storage:**
- Directory: `tests/highdim/fixtures/ledh_golden_master_20260909/`
- Format: JSON, one file per configuration
- Naming: `golden_T{T}_N{N}_d{d}_K{K}_{dtype}_anneal{annealed_stages}.json`

**Content per fixture:**
```json
{
  "schema": "ledh_golden_master_v1",
  "config": { ... },
  "inputs": {
    "theta": [...],
    "initial_states": [...],
    "initial_covariances": [...],
    "noises": [...],
    "observations": [...]
  },
  "outputs": {
    "value": ...,
    "score": [...],
    "diagnostics": { ... }
  },
  "generation_metadata": {
    "git_commit": "...",
    "timestamp": "...",
    "python_version": "...",
    "tensorflow_version": "...",
    "seed": ...
  }
}
```

### 2. Row-Independence Tests

Test at B ∈ {1, 2, 4} with distinct θ rows:
- **Identical rows** → identical outputs (bitwise)
- **Distinct rows** → distinct outputs
- **Row isolation** → Row `i` output unchanged by row `j` perturbation

This catches the principal design risk: naive flattening that mixes clouds across θ rows.

### 3. Capability-Matrix Tests

One test per union capability:
- Contract-E reset
- Dual-cap correction
- Trust-region correction
- Pairwise correction
- Annealed stages
- Multi-direction tangent (K > 1)

### 4. Tolerance Policy

- Golden-master: rtol=1e-12 (float64), rtol=1e-6 (float32)
- Post-refactor: same tolerance (no regression allowed)
- Cross-lane parity (different lanes): rtol=5e-4 (not applicable here)

---

## Implementation

### Step 1.1: Fixture Generator Script

**File:** `tests/highdim/generate_golden_master_fixtures.py`

Creates all 360 fixtures by:
1. Loading tuned controls
2. Creating LGSSM model (d=2 or d=3)
3. Generating deterministic inputs from fixed seeds
4. Running `canonical_value_and_analytical_score`
5. Serializing outputs to JSON

**Estimated time:** 2-3 hours (generation + validation)

### Step 1.2: Characterisation Test Suite

**File:** `tests/highdim/test_ledh_golden_master.py`

Parametrized pytest that:
1. Discovers all fixture files
2. Loads each fixture
3. Recreates inputs
4. Runs `canonical_value_and_analytical_score`
5. Compares outputs to fixture (rtol=1e-12 for float64)

**Estimated time:** 1-2 hours

### Step 1.3: Row-Independence Tests

**File:** `tests/highdim/test_ledh_row_independence.py`

Three test classes:
- `test_identical_rows_produce_identical_outputs`
- `test_distinct_rows_produce_distinct_outputs`
- `test_row_isolation` (row i unchanged by row j perturbation)

**Estimated time:** 2-3 hours

### Step 1.4: Capability-Matrix Tests

**File:** `tests/highdim/test_ledh_capability_matrix.py`

Each capability gets one focused test exercising that feature.

**Estimated time:** 1-2 hours

---

## Exit Criterion

All tests pass against unmodified `canonical_value_and_analytical_score`:
- ✅ 360 golden-master fixtures generated
- ✅ All golden-master tests pass (rtol=1e-12 for float64)
- ✅ All row-independence tests pass
- ✅ All capability-matrix tests pass

---

## Timeline

- **Day 1 (2026-09-09):** 
  - Morning: Fixture generator script + generate fixtures
  - Afternoon: Golden-master test suite
- **Day 2 (2026-09-10):**
  - Morning: Row-independence tests
  - Afternoon: Capability-matrix tests + documentation

**Total:** 1-2 days

---

## Notes

- Keep fixture files small by using short horizons (T=1,2,3) for most cases
- Use deterministic seeds for reproducibility
- Document any numerical issues (e.g., Cholesky failures at extreme settings)
- Golden-master tolerance is strict (1e-12) — refactor must match exactly

---

## Status Tracking

- [ ] Step 1.1: Fixture generator script
- [ ] Step 1.2: Characterisation test suite
- [ ] Step 1.3: Row-independence tests
- [ ] Step 1.4: Capability-matrix tests
- [ ] Documentation complete
- [ ] All tests pass
