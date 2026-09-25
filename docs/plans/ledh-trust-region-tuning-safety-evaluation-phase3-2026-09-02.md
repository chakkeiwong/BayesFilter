# LEDH Trust-Region Tuning and Safety Evaluation - Phase 3

**Date:** 2026-09-02  
**Status:** AWAITING_OWNER_APPROVAL  
**Owner:** chakwong  
**Governing program:** `docs/plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md`

---

## Research Question

Does the LEDH trust-region solver (dual-cap + LM damping + trust radius) satisfy the Class C non-harm criterion, and what trust-region control values should be used for production runs?

---

## Background

**Owner decision (2026-09-02):** Both dual-cap AND trust-region are required for LEDH production.

**Current status:**
- ✅ Dual-cap primal route: tuned for 4 models (Austria SIR, LGSSM, KSC, Predator-Prey)
- ❌ Trust-region route: no tuning artifacts exist
- ❌ Safety evaluation: neither dual-cap nor trust-region evaluated under Class C criterion

**Policy context:**
- **LEDH Per-Scope Tuning Rule:** Every claim-bearing run requires per-scope tuning artifact
- **Safety Guardrail Reversed Burden:** Class C numerics-altering protections require mandatory non-harm evaluation
- **Trust-region is a different tuning scope:** Adding LM damping + trust radius creates new parameter space requiring separate tuning

---

## Objectives

### Primary Objectives

1. **Tune trust-region controls** for Austria SIR T20 pilot model
   - Find `higher_moment_lm_damping`, `higher_moment_lm_scale_floor`, `higher_moment_trust_radius` values
   - Acceptance criterion: passes hard validity gates (residuals, coordinate cap bounds)
   - Warm-start from existing dual-cap primal artifact

2. **Evaluate dual-cap non-harm** (Class C safety evaluation)
   - Compare: Contract-E only vs dual-cap primal
   - Criterion: outputs identical on healthy trajectories; bounded/flagged on unhealthy
   - Diagnostics: coordinate cap fire rate, displacement, pre/post cap values

3. **Evaluate trust-region non-harm** (Class C safety evaluation)
   - Compare: dual-cap primal vs dual-cap trust-region
   - Criterion: outputs identical when primal solver converges; bounded when primal diverges
   - Diagnostics: LM damping scale, trust-radius active indicators, solver residuals

### Secondary Objectives

4. **Document trust-region tuning protocol** for remaining 3 models
5. **Create production-ready runner** that enforces `LEDH_PRODUCTION_PROGRAM_V1` wiring gate
6. **Record tuning artifact** with exact scope signature for Austria SIR T20 trust-region route

---

## Experimental Design

### Three-Arm Comparison

| Arm | Reset policy | Dual-cap | Trust-region | Status |
|-----|-------------|----------|--------------|--------|
| **Baseline** | Contract-E affine only | ❌ disabled | ❌ disabled | Reference |
| **Dual-cap primal** | Contract-E + dual-cap | ✅ enabled | ❌ disabled | Tuned (Aug 2026) |
| **Trust-region** | Contract-E + dual-cap + TR | ✅ enabled | ✅ enabled | **To be tuned** |

### Control Values

**Baseline arm:**
```python
{
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1e-5,
    "higher_moment_correction_steps": 0,  # disabled
    "pairwise_moment_correction_steps": 0,  # disabled
    "coordinatewise_standardized_cap": 0.0,  # disabled
}
```

**Dual-cap primal arm (existing tuned controls):**
```python
{
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2,
    "higher_moment_floor": 1e-5,
    "pairwise_moment_correction_steps": 4,
    "pairwise_moment_strength": 0.02,
    "pairwise_moment_floor": 1e-5,
    "pairwise_particle_rms_cap": 2.0,
    "coordinatewise_standardized_cap": 0.98,
    "coordinatewise_standardized_cap_power": 8,
    # Trust-region disabled:
    "higher_moment_lm_damping": 0.0,
    "higher_moment_lm_scale_floor": 1e-6,
    "higher_moment_trust_radius": 0.0,
}
```

**Trust-region arm (to be tuned):**
```python
{
    # Inherit dual-cap primal controls
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2,
    "higher_moment_floor": 1e-5,
    "pairwise_moment_correction_steps": 4,
    "pairwise_moment_strength": 0.02,
    "pairwise_moment_floor": 1e-5,
    "pairwise_particle_rms_cap": 2.0,
    "coordinatewise_standardized_cap": 0.98,
    "coordinatewise_standardized_cap_power": 8,
    # Trust-region controls (TUNING TARGETS):
    "higher_moment_lm_damping": ???,  # grid: [1e-3, 1e-2, 1e-1]
    "higher_moment_lm_scale_floor": ???,  # grid: [1e-6, 1e-5, 1e-4]
    "higher_moment_trust_radius": ???,  # grid: [0.1, 0.5, 1.0]
}
```

### Tuning Grid

**Trust-region parameter grid (27 configurations):**
- `higher_moment_lm_damping`: [1e-3, 1e-2, 1e-1]
- `higher_moment_lm_scale_floor`: [1e-6, 1e-5, 1e-4]
- `higher_moment_trust_radius`: [0.1, 0.5, 1.0]

**Warm-start hypothesis:** Code defaults (damping=1e-2, floor=1e-6, radius=0.5) are in grid center

### Data Partition

**Austria SIR T20 existing artifact:**
- 2 calibration observations
- 1 claim observation (T=20 time series)
- 16 claim seeds (98201-98216)
- 2 tuning seeds (98301-98302)

**Phase 3 usage:**
- **Tuning:** 2 calibration observations × 2 tuning seeds = 4 evaluations per config
- **Claim:** Held out (not used in Phase 3)

### Validity Gates

**Hard validity (from existing runner):**
1. `base._valid(row)`: finite values, positive weights, valid marginals
2. Max transport residuals ≤ 5e-4 (mean, row, col, score increment sum)
3. Coordinate cap: `maximum_coordinatewise_post_cap_absolute < 1.000001`

**Candidate passes if:** All 4 calibration evaluations pass hard validity

---

## Success Criteria

### Tuning Success

**Primary criterion:** At least one trust-region configuration passes hard validity on all 4 calibration evaluations

**Selection criterion (if multiple pass):**
1. Lowest max coordinate cap fire rate (prefer minimal intervention)
2. If tied: lowest LM damping (prefer least regularization)
3. If tied: smallest trust radius (prefer tightest constraint)

**Failure modes:**
- No configuration passes → trust-region incompatible with this model/data
- All configurations fire coordinate cap maximally → trust-region insufficient, need stronger cap
- High variance across seeds → need more seeds or wider grid

### Safety Evaluation Success

**Dual-cap non-harm (Arm 2 vs Arm 1):**
- **Pass:** Where Arm 1 (baseline) passes validity, Arm 2 (dual-cap) outputs differ by ≤1e-3 relative or coordinate cap fires
- **Fail:** Arm 2 silently diverges where Arm 1 is healthy, or introduces unbounded error

**Trust-region non-harm (Arm 3 vs Arm 2):**
- **Pass:** Where Arm 2 (dual-cap primal) passes validity, Arm 3 (trust-region) outputs differ by ≤1e-3 relative or LM damping/trust-radius active
- **Fail:** Arm 3 silently diverges where Arm 2 is healthy, or introduces unbounded error

**Healthy trajectory definition:** Evaluation passes hard validity under baseline or primal route

---

## Diagnostics

### Primary Diagnostics

**Hard validity (promotion veto):**
- `program_valid`: bool
- Transport residuals: mean, row, col, score increment sum
- Coordinate cap: `maximum_coordinatewise_post_cap_absolute`

**Dual-cap observability:**
- `fraction_coordinatewise_cap_active`
- `mean_coordinatewise_cap_displacement`
- `maximum_coordinatewise_pre_cap_absolute`
- `maximum_coordinatewise_post_cap_absolute`

**Trust-region observability (NEW - need to verify these exist):**
- LM damping active indicator
- Trust-radius active indicator  
- Solver residuals or convergence diagnostics
- Effective regularization scale

### Secondary Diagnostics

**Explanatory only (not gates):**
- Log-likelihood value
- Score vector
- Pairwise RMS cap: pre/post, scale factor
- Shape objective values

---

## Evidence Contract

### What Will Be Concluded

**If tuning succeeds and safety passes:**
- Trust-region route is production-ready for Austria SIR T20 under the selected controls
- Class C non-harm criterion satisfied for both dual-cap and trust-region
- Tuning artifact scope-matched for this model/horizon/particle-count

**If tuning succeeds but safety fails:**
- Trust-region mechanically viable but not non-harmful
- Downgrade to optional experimental feature, not required production mechanism

**If tuning fails:**
- Trust-region incompatible with this model, or grid insufficient
- Require wider grid, different warm-starts, or model-specific investigation

### What Will NOT Be Concluded

- ❌ Trust-region is superior/better/faster than dual-cap primal (descriptive differences only, no statistical ranking)
- ❌ Trust-region is ready for other models (each requires separate tuning per LEDH Per-Scope Tuning Rule)
- ❌ Trust-region is ready for HMC or NeuTra (separate evaluation required)
- ❌ Selected controls are optimal or calibrated (tuning finds viable candidate, not necessarily best)
- ❌ Safety evaluation on 4 calibration points proves safety on all possible trajectories (acceptance for this scope only)

---

## Implementation Plan

### Step 1: Create Phase 3 Runner (1 hour)

**File:** `docs/benchmarks/run_ledh_trust_region_phase3_austria_sir.py`

**Key components:**
1. Load Austria SIR T20 fixture
2. Load existing dual-cap primal tuned controls as warm-start
3. Define trust-region parameter grid (27 configs)
4. Three-arm comparison: baseline, dual-cap primal, trust-region grid
5. Integrate `LEDH_PRODUCTION_PROGRAM_V1` wiring gate
6. Record full diagnostic payload per evaluation

**Output artifact structure:**
```
docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/
├── result.json  # Complete results
├── tuning_summary.json  # Selected controls + validity table
├── safety_dual_cap.json  # Arm 2 vs Arm 1 comparison
├── safety_trust_region.json  # Arm 3 vs Arm 2 comparison
└── diagnostics/
    ├── arm1_baseline_seed*.json
    ├── arm2_dual_cap_seed*.json
    └── arm3_trust_region_config*_seed*.json
```

### Step 2: Run Campaign (GPU, 30-60 minutes)

**Estimated wall time:**
- Baseline: 4 evals × ~30s = 2 min
- Dual-cap primal: 4 evals × ~30s = 2 min
- Trust-region grid: 27 configs × 4 evals × ~30s = 54 min
- **Total:** ~60 min GPU

**Resource requirements:**
- 1 GPU (RTX 4080 SUPER or RTX 5080)
- TF32 enabled, XLA compiled
- Memory growth enabled
- Particle count: N=10000

### Step 3: Analyze Results (1-2 hours)

**Analyses:**
1. Validity table: which trust-region configs pass all 4 calibration evals
2. Safety tables: paired comparisons (Arm 2 vs Arm 1, Arm 3 vs Arm 2)
3. Diagnostic summaries: cap fire rates, trust-radius active rates, residuals
4. Control selection: apply selection criterion to passing configs

**Decision points:**
- If ≥1 config passes validity: select best, proceed to artifact creation
- If 0 configs pass: diagnose failure mode, revise grid or investigate model incompatibility
- If safety fails: diagnose non-harm violation, consider downgrade to optional

### Step 4: Create Tuning Artifact (30 minutes)

**File:** `docs/benchmarks/artifacts/ledh_trust_region_austria_sir_t20_20260902/tuning_artifact.json`

**Required fields:**
```json
{
    "model": "austria_sir_T20",
    "route": "ledh_pfpf_ot_contract_e_dual_cap_trust_region_v1",
    "scope": {
        "model_id": "austria_sir_T20",
        "horizon": 20,
        "state_dimension": 4,
        "observation_dimension": 1,
        "parameter_dimension": 4,
        "particle_count": 10000,
        "event_order": "transition_before_first_observation",
        "backend": "tensorflow",
        "dtype": "float32",
        "tf32_enabled": true,
        "chunk_policy": "dpf_transport_exact_divisor_cap3000_v1"
    },
    "controls": {
        "epsilon": 8.0,
        "sinkhorn_steps": 16,
        "balance_steps": 16,
        "ridge": 1e-5,
        "higher_moment_correction_steps": 4,
        "higher_moment_strength": 0.2,
        "higher_moment_floor": 1e-5,
        "higher_moment_lm_damping": <SELECTED>,
        "higher_moment_lm_scale_floor": <SELECTED>,
        "higher_moment_trust_radius": <SELECTED>,
        "pairwise_moment_correction_steps": 4,
        "pairwise_moment_strength": 0.02,
        "pairwise_moment_floor": 1e-5,
        "pairwise_particle_rms_cap": 2.0,
        "coordinatewise_standardized_cap": 0.98,
        "coordinatewise_standardized_cap_power": 8
    },
    "calibration_valid": true,
    "claim_valid": false,  # not evaluated yet
    "scope_hash": "<SHA256>",
    "provenance": {
        "campaign": "docs/plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md",
        "result_artifact": "docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/result.json",
        "tuning_date": "2026-09-02",
        "git_commit": "<COMMIT>"
    }
}
```

### Step 5: Document Results (30 minutes)

**Phase 3 completion memo:** `docs/memos/ledh-trust-region-phase3-complete-austria-sir-2026-09-02.md`

**Contents:**
- Tuning verdict: selected controls, validity evidence
- Safety evaluation verdict: dual-cap non-harm status, trust-region non-harm status
- Diagnostic summary: cap fire rates, trust-region active rates
- Decision: ready for Phase 4 (remaining models) or blocked
- Next steps: extend to 3 remaining models or investigate failures

---

## Budget

**Compute budget:**
- Austria SIR T20 Phase 3: ~60 min GPU
- If extended to all 4 models: ~4 hours GPU total

**Attempt budget:**
- Phase 3 is exploratory tuning: no fixed attempt limit
- If grid insufficient, revise and retry
- Record each attempt, preserve all artifacts

**Stop conditions:**
- Success: At least one config passes validity and safety
- Failure: No config passes after 2 grid revisions, or safety criterion fails for all passing configs
- Blocker: Model-specific incompatibility discovered (requires investigation, not more tuning)

---

## Risks and Mitigations

### Risk 1: No trust-region config passes validity

**Likelihood:** Low (existing dual-cap primal works, trust-region adds regularization)

**Mitigation:**
- Widen grid: try stronger damping [1e-2, 1e-1, 1.0] or larger radius [0.5, 1.0, 2.0]
- Investigate solver: check if trust-region solver actually activates
- Fall back: if trust-region fundamentally incompatible, revise owner decision to dual-cap primal only

### Risk 2: Trust-region passes validity but fails safety (non-harm)

**Likelihood:** Medium (regularization can introduce bias)

**Impact:** Trust-region cannot be required, must be optional

**Mitigation:**
- Diagnose: where does trust-region diverge from primal on healthy cases?
- Quantify: how large is the bias, and is it acceptable?
- Owner decision: accept bias as tradeoff for stability, or downgrade to optional

### Risk 3: Trust-region diagnostics don't exist in output

**Likelihood:** Medium (implementation may not emit LM damping/trust-radius indicators)

**Impact:** Cannot verify mechanism is active

**Mitigation:**
- Check `finite_value_score` return dict for trust-region fields
- If missing: add observability before running campaign
- If cannot add: use indirect evidence (residual changes, particle displacement patterns)

### Risk 4: Grid too coarse, selected controls are boundary values

**Likelihood:** Medium (27 configs may miss optimal region)

**Impact:** Suboptimal controls, may fail on other models

**Mitigation:**
- Check if selected config is grid boundary (e.g., damping=1e-3 or 1e-1)
- If boundary: refine grid around selected value
- If interior: accept as viable, document that finer tuning possible

---

## Open Questions

1. **Trust-region observability:** What diagnostics does `finite_value_score` return for trust-region activity?
   - Need to check return dict for LM damping scale, trust-radius constraint indicators
   - If missing, add before campaign

2. **Trust-region vs dual-cap primal equivalence:** When trust-region controls are zero, does it reduce exactly to dual-cap primal?
   - Important for safety evaluation: need to verify arms 2 and 3 are identical when TR disabled
   - May need epsilon-tolerance rather than exact equality

3. **Grid coverage:** Is 27 configs sufficient for initial exploration?
   - Tentative yes: spans 3 orders of magnitude for each parameter
   - Expand if no config passes or all boundary values selected

4. **Remaining 3 models:** After Austria SIR succeeds, can we transfer controls or need full grid search?
   - Per LEDH Per-Scope Tuning Rule: each model needs separate tuning
   - But: can use Austria SIR result as narrower warm-start grid (e.g., ±1 step around selected)

5. **Statistical ranking:** If multiple configs pass, how to select?
   - Plan uses deterministic tiebreaker (minimal intervention principle)
   - Alternative: run longer evaluation (more seeds) and use uncertainty intervals
   - Decision: start with deterministic, upgrade if needed

---

## Success Metrics

**Phase 3 succeeds if:**
1. ✅ At least one trust-region config passes calibration validity (4/4 evals)
2. ✅ Dual-cap non-harm criterion satisfied (Arm 2 ≈ Arm 1 on healthy cases)
3. ✅ Trust-region non-harm criterion satisfied (Arm 3 ≈ Arm 2 on healthy cases)
4. ✅ Tuning artifact created with scope-matched controls
5. ✅ Phase 3 completion memo documents verdict and next steps

**Phase 3 partially succeeds if:**
- Tuning succeeds but safety fails → trust-region viable but not non-harmful → optional feature
- Dual-cap safety passes but trust-region tuning fails → dual-cap ready, trust-region needs investigation

**Phase 3 fails if:**
- Both tuning and safety fail → mechanisms incompatible with this model
- Require model-specific investigation or owner decision revision

---

## Next Steps After Phase 3

**If Phase 3 succeeds:**
- **Phase 4:** Extend trust-region tuning to remaining 3 models (LGSSM, KSC, Predator-Prey)
- **Phase 5:** Integrate wiring gate into all claim-bearing runners
- **Phase 6:** Update production documentation and memory entries

**If Phase 3 partially succeeds (dual-cap only):**
- Revise `LEDH_PRODUCTION_PROGRAM_V1` to require dual-cap primal only (not trust-region)
- Document trust-region as experimental optional feature
- Skip Phase 4 trust-region extension

**If Phase 3 fails:**
- Investigate failure mode (model-specific issue, grid insufficient, implementation bug)
- Owner decision: revise requirements, investigate further, or accept current state

---

## Approval Request

**Phase 3 ready to execute pending owner approval:**
- ✅ Research question clear
- ✅ Three-arm design specified
- ✅ Tuning grid defined (27 configs)
- ✅ Validity gates and success criteria stated
- ✅ Safety evaluation non-harm criterion defined
- ✅ Budget and timeline estimated (~60 min GPU)

**Approve to proceed?**
