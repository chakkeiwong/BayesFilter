# SQMC Parameters: Mathematical Roles and Tuning Criticality

**Date:** 2026-09-09  
**Purpose:** Identify which parameters are critical to tune vs which have minor effects

---

## Parameter Categories

### Category 1: Transport Parameters (Sinkhorn/OT)

**Purpose:** Control the optimal transport that resets weighted particles to equal weights while preserving moments.

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `epsilon` | 8.0 | Entropic regularization strength in Sinkhorn algorithm | **MODERATE** |
| `sinkhorn_steps` | 8 | Number of Sinkhorn iterations | **LOW** |
| `balance_steps` | 8 | Number of balancing iterations after Sinkhorn | **LOW** |
| `ridge` | 1e-5 | Numerical regularization for covariance gap | **LOW** |

**Mathematical context:**
- Sinkhorn solves entropic-regularized optimal transport with cost ε
- Larger ε = more regularization = smoother transport = faster convergence
- Smaller ε = closer to true OT = sharper transport = slower convergence
- `sinkhorn_steps` and `balance_steps` control convergence; more steps = tighter marginal matching

**Evidence from docs:**
- LGSSM/KSC-SV/Austria SIR used ε=2.0, 8 steps
- Austria current used ε=8.0, 8 steps (more regularized)
- **Tuning impact:** Affects transport residual and convergence, but usually not accuracy once converged

**Recommendation:** **Use warm-start values**. These are numerical convergence parameters with well-understood behavior. If transport residuals pass validity gates, values are adequate.

---

### Category 2: Dual-Cap Parameters (Covariance Stabilization)

**Purpose:** Prevent covariance explosion in GenUT higher-moment corrections.

#### 2A: Diagonal Correction

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `diagonal_steps` | 4 | Number of diagonal third/fourth-moment correction steps | **LOW** |
| `diagonal_strength` | 0.2 | Step size for diagonal correction | **MODERATE** |

#### 2B: Pairwise Correction

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `pairwise_steps` | 4 | Number of pairwise co-skewness/co-kurtosis correction steps | **LOW** |
| `pairwise_strength` | 0.02 | Step size for pairwise correction | **MODERATE** |

#### 2C: Radial Cap

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `radial_cap` | 2.0 | Cap on rowwise RMS of pairwise correction direction: $s_n = (1 + r_n^2/c_r^2)^{-1/2}$ | **HIGH** |

**Mathematical role:** Limits particle displacement from pairwise corrections to prevent divergence.

#### 2D: Coordinate Cap

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `coordinate_cap` | 0.98 | Smooth cap on standardized coordinates: $f_b(z) = z/\{1+(z/b)^p\}^{1/p}$ | **HIGH** |
| `coordinate_cap_power` | 8 | Smoothness of coordinate cap | **LOW** |

**Mathematical role:** Prevents individual standardized coordinates from exceeding bounds, controls tail behavior.

**Evidence from docs:**
> "High cap activity indicates frequent covariance corrections"  
> Austria SIR trust-region reduced cap activity from 85.7% to 66.9%

**Mathematical insight:** 
- Caps control **covariance explosion**, the main failure mode
- `radial_cap` and `coordinate_cap` directly limit particle displacement
- Strength parameters (`diagonal_strength`, `pairwise_strength`) control correction aggressiveness
- Step counts control number of iterations (usually 4 is sufficient)

**Recommendation:** 
- **Caps (radial_cap, coordinate_cap): Use warm-start values**. These have been validated across models (LGSSM, KSC-SV, Austria SIR, Predator-Prey). From docs: selected defaults are radial_cap=2.0, coordinate_cap=0.98, power=8.
- **Strengths: Use model-inherited warm-start**. These varied by model in evidence (0.02 for LGSSM/KSC-SV/Austria, 0.05 for Predator-Prey), but differences were small.
- **Steps: Use defaults** (4 for both diagonal and pairwise).

---

### Category 3: Trust-Region Parameters (Levenberg-Marquardt Damping)

**Purpose:** Regularize the Contract-E linear system solve that computes the covariance-restoring affine map.

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `lm_damping` | 1e-2 | LM regularization added to normal equations: $(J^TJ + \lambda I)\delta = -J^T r$ | **MODERATE** |
| `lm_scale_floor` | 1e-4 | Minimum scale for LM damping relative to problem scale | **LOW** |
| `radius` | 0.5 | Trust-region radius limiting affine displacement | **MODERATE** |

**Mathematical context:**
- Contract-E solves for affine map $X^+ = \mu + L\hat{Z}$ that restores mean and covariance
- Trust region adds LM damping to stabilize ill-conditioned systems
- Larger damping = more conservative = more stability = less correction
- Radius limits the norm of the correction

**Evidence from docs:**
> Austria SIR T20 trust-region tuning selected: damping=0.001, floor=1e-06, radius=0.1  
> Result: 26% better objective than dual-cap alone, 66.9% cap activity vs 85.7%

**Tuning criticality:**
- **Model-dependent**: Austria SIR selected damping=0.001, but Austria SIR current uses damping=0.01 (10× larger)
- Affects numerical stability of covariance restoration
- **But**: Austria SIR comparison used damping=0.01 for all routes and found them equivalent

**Recommendation:** **Use warm-start damping=0.01, floor=1e-4, radius=0.5** (Austria SIR current values). These are more conservative than the tuned values but have been validated to work. Fine-tuning can improve objectives by ~26% but doesn't change route rankings.

---

### Category 4: State Map Parameters (Ancestry Policy)

| Parameter | Default | Mathematical Role | Tuning Criticality |
|---|---|---|---|
| `hilbert_bits` | 12 | Precision of Hilbert curve for particle ordering | **LOW** |
| `map_multiplier` | 2.0-3.0 | Scale multiplier for state map (fixed MAP only) | **HIGH if fixed MAP** |
| `map_location` | model-specific | Center of state map (fixed MAP only) | **HIGH if fixed MAP** |
| `state_map_policy` | adaptive/fixed | Adaptive (empirical) vs fixed (pre-tuned) MAP | **HIGH** |

**Mathematical context:**
- Hilbert curve orders particles for ancestry selection
- State map transforms particles to [0,1]^d for Hilbert ordering
- Fixed MAP: pre-tuned location/scale specific to model
- Adaptive: uses empirical mean/std from particles

**Evidence:**
Austria SIR runner uses:
- `iid_dual_cap`: adaptive (location=0, scale=1) 
- Other routes: fixed MAP with model-specific location/scale

**Recommendation:** **Use adaptive state map policy** for LGSSM/KSC-SV. This eliminates the need for per-model MAP tuning and is more general. From code:
```python
if route == "iid_dual_cap":
    location = tf.zeros_like(location)  # Adaptive
    scale = tf.ones_like(scale)
```

---

## Criticality Summary

### Must Tune (if not using adaptive):
1. ❌ **MAP location/scale** — highly model-specific (but **avoid by using adaptive**)

### Should Tune (for optimal performance):
2. 🟡 **Trust-region damping/radius** — 26% objective improvement possible
3. 🟡 **Pairwise/diagonal strength** — model-specific aggressiveness

### Can Use Warm-Start (minor effect):
4. ✅ **Radial cap** (2.0) — validated across 4 models
5. ✅ **Coordinate cap** (0.98, power 8) — validated across 4 models  
6. ✅ **Sinkhorn epsilon** (8.0) — affects convergence, not accuracy
7. ✅ **Sinkhorn/balance steps** (8, 8) — numerical convergence
8. ✅ **Ridge** (1e-5) — numerical regularization
9. ✅ **Diagonal/pairwise steps** (4, 4) — convergence parameters
10. ✅ **Hilbert bits** (12) — sufficient precision
11. ✅ **LM scale floor** (1e-4) — numerical floor
12. ✅ **Coordinate cap power** (8) — smoothness parameter

---

## Recommended Configuration for LGSSM/KSC-SV Oracle Comparison

**Strategy:** Use production algorithm with warm-start controls + adaptive state map

```python
# Transport (warm-start from Austria SIR)
epsilon = 8.0
sinkhorn_steps = 8
balance_steps = 8
ridge = 1e-5

# Dual-cap (validated defaults from four-model evidence)
diagonal_steps = 4
diagonal_strength = 0.2  # Austria SIR value
pairwise_steps = 4
pairwise_strength = 0.02  # LGSSM/KSC-SV value from evidence
radial_cap = 2.0  # Universal default
coordinate_cap = 0.98  # Universal default
coordinate_cap_power = 8

# Trust-region (conservative Austria SIR current values)
lm_damping = 1e-2  # More conservative than tuned 1e-3
lm_scale_floor = 1e-4
trust_radius = 0.5  # More conservative than tuned 0.1

# State map (CRITICAL: use adaptive to avoid MAP tuning)
state_map_policy = "adaptive_empirical"  # No MAP tuning needed
hilbert_bits = 12

# Ancestry policies (these are the 4 routes being compared)
# - iid_dual_cap: existing_one_to_one
# - previous_inverse_cdf: hilbert_inverse_cdf  
# - repaired_fixed_previous_controls: hilbert_permutation_one_to_one
# - repaired_permutation: hilbert_permutation_one_to_one
```

**Justification:**
1. **No MAP tuning** — use adaptive instead of fixed
2. **Conservative trust-region** — damping=0.01 more stable than tuned 0.001
3. **Validated dual-cap** — caps and strengths from four-model evidence
4. **Standard transport** — convergence parameters with known behavior

**What this avoids:**
- ❌ No per-model MAP tuning (56 artifacts)
- ❌ No per-model trust-region tuning grid search
- ❌ No per-model strength tuning

**What this preserves:**
- ✅ Production LEDH PFPF-OT code path
- ✅ Validated dual-cap covariance stabilization
- ✅ Conservative trust-region for numerical stability
- ✅ Standard transport with validated convergence

**Risk assessment:**
- Trust-region tuning could improve objectives by ~26% (Austria SIR evidence)
- But Austria SIR found all routes **equivalent** even with warm-start values
- Oracle comparison measures **relative route performance**, not absolute optimality
- If routes differ with warm-start settings, targeted tuning can follow

---

## Documentation Requirement

Results must state:
> "Configuration uses warm-start controls from Austria SIR and four-model dual-cap evidence, with adaptive state map (no per-model MAP tuning). Trust-region uses conservative values (damping=0.01, radius=0.5) rather than model-specific tuned values. Comparison measures relative route accuracy against oracle, not absolute optimally-tuned performance."

---

## References

1. `docs/genut-dual-cap-default-algorithm-integration-note-2026-08-07.md` — Four-model dual-cap evidence, selected defaults
2. `docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md` — Austria SIR trust-region tuning (damping=0.001 selected, but 0.01 also valid)
3. Austria SIR runner `run_sqmc_rerun_corrected_filter_20260906.py` — Current production configuration
4. CLAUDE.md — Production algorithm definition and per-scope tuning rule

