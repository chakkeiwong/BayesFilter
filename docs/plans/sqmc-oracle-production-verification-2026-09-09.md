# SQMC Oracle Comparison: Production Code and Tuning Verification

**Date:** 2026-09-09  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Purpose:** Verify we use correct production code and understand tuning requirements

---

## Production Code Path: ✓ VERIFIED

### Austria SIR Runner Uses Production Code

The completed Austria SIR SQMC comparison (`run_sqmc_rerun_corrected_filter_20260906.py`) calls:

```python
finite_value_standard_score_initial_rqmc(
    model.callbacks,
    theta,
    observations,
    ...
    reset_policy="contract_e",                    # Contract-E reset
    dual_cap_enabled=True,                        # Dual-cap enabled
    dual_cap_diagonal_steps=...,                  # Per-route
    dual_cap_pairwise_steps=...,                  # Per-route
    dual_cap_pairwise_particle_rms_cap=...,      # Per-route (radial_cap)
    dual_cap_coordinate_cap=...,                  # Per-route
    trust_region_enabled=(reset == "trust_region"),  # Trust region
    trust_region_lm_damping=1.0e-2,              # Fixed across routes
    trust_region_lm_scale_floor=1.0e-4,          # Fixed across routes
    trust_region_radius=0.5,                     # Fixed across routes
    epsilon=8.0,                                  # Fixed across routes
    sinkhorn_steps=8,                            # Fixed across routes
    balance_steps=8,                             # Fixed across routes
    ridge=1.0e-5,                                # Fixed across routes
    transport_plan_mode="streaming",             # Streaming/dense choice
    transport_row_chunk_size=...,                # From chunk policy
    transport_col_chunk_size=...,                # From chunk policy
)
```

**Source file:** `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`

**Verification:**
- ✓ This is production LEDH PFPF-OT code (not experimental)
- ✓ Uses Contract-E reset with GenUT
- ✓ Uses dual-cap trust region
- ✓ Uses streaming transport (complies with K≤3000 chunk rule per CLAUDE.md)
- ✓ TF32 enabled (`tf.config.experimental.enable_tensor_float_32_execution(True)`)
- ✓ GPU execution (`/GPU:0`)

**This matches CLAUDE.md Default Execution Target:**
> "The default production algorithm target is the GPU-oriented LEDH-PFPF-OT TF32 route"

---

## Route Definitions

The 4 SQMC routes tested are **transport ancestry policies**, not different OT algorithms:

| Route | Ancestry Policy | Description |
|---|---|---|
| `iid_dual_cap` | `existing_one_to_one` | No reordering, identity ancestors |
| `previous_inverse_cdf` | `hilbert_inverse_cdf` | Hilbert curve + inverse CDF sampling |
| `repaired_fixed_previous_controls` | `hilbert_permutation_one_to_one` | Hilbert curve + fixed permutation |
| `repaired_permutation` | `hilbert_permutation_one_to_one` | Hilbert curve + permutation (different controls) |

**Shared across all routes:**
- Reset: Contract-E with GenUT
- Transport: Sinkhorn (ε=8.0, 8 steps, balance 8 steps)
- Trust region: Enabled (damping 1e-2, floor 1e-4, radius 0.5)
- Ridge: 1e-5

**Different per route:**
- Dual-cap parameters (diagonal/pairwise steps/strength, radial_cap, coordinate_cap)
- State map multiplier (2.0 vs 3.0)
- Hilbert bits (always 12)

---

## Tuning Status

### CLAUDE.md Requirement

> "Every claim-bearing LEDH model run requires an offline tuning artifact for the exact model/target, route/reset family, horizon/prepared-data regime, particle count, dimensions, dtype/backend, chunk policy, and route-specific control family used by that run."

**Implication:** Each (model, horizon, N, route) needs its own tuning artifact.

### Austria SIR Tuning Status

The Austria SIR runner **does NOT reference tuning artifacts**. Instead, it uses:

1. **Hard-coded route controls** in `ROUTE_CONTROLS` dict
2. **Hard-coded MAP location/scale** specific to Austria SIR T=20

**These are warm-start/inherited settings**, not per-model tuned values.

### LGSSM/KSC-SV Tuning Status

**LGSSM:**
- Trust region tuning exists for T=50: `docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/`
- But this tunes trust-region controls (damping, floor, radius), not route-specific dual-cap controls
- No SQMC route comparison tuning found for LGSSM

**KSC-SV:**
- No SQMC route tuning artifacts found

---

## Tuning Scope for Full Compliance

To comply with CLAUDE.md's tuning rule for the full 896-cell campaign:

**Models:** LGSSM (3 horizons), KSC-SV (4 horizons) = 7 model configs
**Particle counts:** N=1008, N=2016 = 2
**Routes:** 4

**Total tuning artifacts needed:** 7 × 2 × 4 = **56 tuning artifacts**

**Each tuning artifact requires:**
- Tuning campaign (grid search or optimization)
- Validation runs
- Selection of best controls
- Documentation

**Estimated time:** Weeks to months of work

---

## Practical Options

### Option A: Use Warm-Start Settings (Austria SIR Approach)

**What it means:**
- Use the same route controls from Austria SIR runner for LGSSM/KSC-SV
- Adjust only model-specific items (MAP location/scale or disable for adaptive)
- Trust region controls: use published values (damping 1e-2, floor 1e-4, radius 0.5)

**Justification:**
- Austria SIR comparison found all 4 routes **statistically indistinguishable** with warm-start settings
- If routes are equivalent with sub-optimal tuning, they'll likely be equivalent with optimal tuning
- Oracle comparison measures **accuracy**, not optimal performance
- Route comparison is relative, not absolute

**Compliance:**
- ✗ Not per-model tuned (violates CLAUDE.md strict interpretation)
- ✓ Uses production code path
- ✓ Uses reviewed default controls

**Timeline:** Can start immediately

---

### Option B: Use Available LGSSM Tuning + Warm-Start for KSC-SV

**What it means:**
- LGSSM T=50: Use trust-region tuning from existing artifact
- LGSSM T=20/T=360: Use warm-start settings
- KSC-SV: Use warm-start settings

**Justification:**
- Partial compliance with tuning rule
- Best available without full tuning campaign

**Compliance:**
- ~ Mixed: LGSSM T=50 compliant, rest warm-start

**Timeline:** Can start immediately with LGSSM T=50, document warm-start for others

---

### Option C: Adaptive State Map (No MAP Tuning Required)

**What it means:**
- Use `state_map_policy="adaptive_empirical"` instead of fixed MAP
- Dual-cap and trust-region controls: warm-start
- Let the algorithm adapt to each model

**Justification:**
- Austria SIR runner uses fixed MAP only for SQMC routes (not iid_dual_cap)
- Adaptive is more general and doesn't require per-model MAP tuning
- Transport and reset controls may be more transferable than MAP

**Code from Austria SIR runner:**
```python
if route == "iid_dual_cap":
    location = tf.zeros_like(location)  # Adaptive
    scale = tf.ones_like(scale)         # Adaptive
else:
    # Use fixed MAP for other routes
```

**Compliance:**
- ~ Reduces tuning surface (MAP not needed)
- Still needs route control tuning per strict rules

**Timeline:** Can start immediately

---

### Option D: Full Tuning Campaign (Strict Compliance)

**What it means:**
- Run 56 separate tuning campaigns
- One per (model, horizon, N, route) combination
- Document each with tuning artifact

**Timeline:** Weeks to months before oracle comparison can begin

**Risk:** Route comparison may be equivalent anyway (like Austria SIR), making tuning effort unnecessary

---

## Recommended Approach

**Option A + C: Warm-Start with Adaptive State Map**

**Rationale:**

1. **Austria SIR precedent:** Used warm-start settings, found all routes equivalent
2. **Oracle comparison goal:** Measure accuracy vs ground truth, not find optimal performance
3. **Relative comparison:** Comparing routes to each other, not to an absolute standard
4. **Adaptive state map:** Generalizes better across models than fixed MAP
5. **Production code:** Still uses correct LEDH PFPF-OT production path
6. **Falsifiable:** If oracle reveals route differences that Austria SIR missed, we can tune then

**Configuration:**
- Use Austria SIR route controls as warm-start
- Set `state_map_policy="adaptive_empirical"` (no MAP tuning needed)
- Use published trust-region controls (damping 1e-2, floor 1e-4, radius 0.5)
- Document as "warm-start configuration, not per-model tuned"

**Documentation requirement:**
In results, state:
- "Configuration uses warm-start controls from Austria SIR, not per-model tuned"
- "Comparison measures relative route performance, not absolute optimality"
- "If route differences emerge, follow-up tuning campaign recommended"

---

## User Decision Required

The user stated: "Ensure that it is tuned correctly for each model."

**Question:** Which option do you prefer?

**A. Warm-start (fast, precedented, Austria SIR approach)**
- Can start oracle comparison immediately
- Document as non-tuned
- If routes differ, tune later

**B. Partial tuning (LGSSM T=50 only)**
- Use available tuning where it exists
- Warm-start elsewhere
- Mixed compliance

**C. Full tuning (strict compliance)**
- 56 tuning artifacts
- Weeks/months before oracle comparison
- May be unnecessary if routes are equivalent

**D. Adaptive + warm-start (recommended)**
- Generalizes better than fixed MAP
- Can start immediately
- Document limitations

---

## Next Steps

**After user decision:**

1. **If warm-start approved:** Adapt Austria SIR runner for LGSSM/KSC-SV with adaptive state map
2. **If tuning required:** Create tuning campaign master program (separate from oracle comparison)
3. **If adaptive approved:** Modify runner to use `adaptive_empirical` state map policy

**Files to create:**
- LGSSM model construction adapter
- KSC-SV model construction adapter
- Oracle integration wrapper
- Minimal characterization runner (8 cells)

**Timeline estimate (after decision):**
- Warm-start: 2-3 hours to working characterization
- Partial tuning: 1 day (run LGSSM T=50 tuning check)
- Full tuning: Weeks (new master program required)

