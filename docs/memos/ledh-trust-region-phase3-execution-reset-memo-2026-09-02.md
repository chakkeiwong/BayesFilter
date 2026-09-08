# LEDH Trust-Region Phase 3 Execution Reset Memo

**Date**: 2026-09-02  
**Status**: Campaign running, awaiting completion  
**Governing Program**: [docs/memos/ledh-dual-cap-implementation-study-reset-2026-09-01.md](ledh-dual-cap-implementation-study-reset-2026-09-01.md)  
**Phase 3 Plan**: [docs/plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md](../plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md)

## Executive Summary

Phase 3 of the LEDH dual-cap implementation study is executing the trust-region tuning and Class C safety evaluation campaign for Austria SIR T20. The campaign runner encountered multiple infrastructure issues during development but is now running successfully. This memo documents the complete execution state, all infrastructure fixes, and next steps for result interpretation.

## Campaign Status

**Runner**: `docs/benchmarks/run_ledh_trust_region_phase3_austria_sir.py`  
**Output Root**: `docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/`  
**Background Task**: `bwoiqabna`  
**Output Log**: `/tmp/claude-1000/-home-chakwong-BayesFilter/b96a3c26-1ea6-4325-98c8-371fddd44a96/tasks/bwoiqabna.output`

**Campaign Design**:
- **Arm 1 (Baseline)**: 4 evaluations (2 calibration obs × 2 seeds) — Contract-E only, no dual-cap, no trust-region
- **Arm 2 (Dual-cap primal)**: 4 evaluations — existing dual-cap tuning, no trust-region
- **Arm 3 (Trust-region grid)**: 108 evaluations — 27 trust-region configs × 2 calibration obs × 2 seeds
- **Total**: 116 evaluations
- **Estimated Runtime**: ~60 minutes on GPU

**Trust-Region Grid**:
- `lm_damping`: (1e-3, 1e-2, 1e-1)
- `lm_scale_floor`: (1e-6, 1e-5, 1e-4)
- `trust_radius`: (0.1, 0.5, 1.0)
- 27 total configurations = 3 × 3 × 3

**Seeds**:
- Tuning seeds: (98301, 98302)
- Claim seeds: tuple(range(98201, 98217)) — reserved for validation after tuning selection

## Infrastructure Fixes Applied

### Fix 1: Base Module Import

**Problem**: Austria SIR target construction failed because the base leaderboard module was not imported.

**Root Cause**: The runner attempted to call `base._build_targets()` without importing the base module.

**Fix**:
```python
# Added to imports section
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
```

**Verification**: Import successful, target construction works.

### Fix 2: Horizon Field

**Problem**: `_make_evaluator()` expected `target["horizon"]` but `_build_targets()` returns targets without a `horizon` field.

**Root Cause**: The base module's target dictionary includes `observations` with shape `[horizon, observation_dim]` but does not explicitly include a `horizon` field.

**Fix**:
```python
def _build_austria_sir_target() -> dict[str, Any]:
    """Build Austria SIR T20 target from base module."""
    targets = base._build_targets()
    austria = targets["austria_sir_T20"]

    # Add horizon field for evaluator
    horizon = int(austria["observations"].shape[0])
    austria["horizon"] = horizon

    return austria
```

**Verification**: Evaluator construction succeeds with correct horizon=20.

### Fix 3: Particle Count Mismatch

**Problem**: The Phase 3 runner had `N = 10000` hardcoded, but the base module's Austria SIR design has 1008 particles.

**Root Cause**: The Phase 3 runner was originally written with a different particle count assumption. The base module uses `N = 1008` and generates designs accordingly.

**Error Message**:
```
ValueError: Dimension 0 in both shapes must be equal, but are 10000 and 1008. 
Shapes are [10000,18] and [1008,18]. for '{{node EnsureShape_4}} = EnsureShape[T=DT_FLOAT, shape=[10000,18]](design)'
```

**Fix**:

1. **Removed global N constant**:
```python
# Removed this line entirely
# N = 10000
```

2. **Updated `_make_evaluator()` to extract particle count from target**:
```python
def _make_evaluator(target: dict[str, Any], controls: dict[str, Any]):
    """Create evaluator for given controls."""
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = target["adapter"]
    horizon = target["horizon"]
    observation_dim = target["observation_dim"]
    state_dim = target["state_dim"]
    parameter_dim = target["parameter_dim"]
    num_particles = int(target["design"].shape[0])  # EXTRACT FROM DESIGN

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [parameter_dim])
        observations = tf.ensure_shape(observations, [horizon, observation_dim])
        initial_noise = tf.ensure_shape(initial_noise, [num_particles, state_dim])
        process_noise = tf.ensure_shape(process_noise, [horizon, num_particles, state_dim])
        design = tf.ensure_shape(design, [num_particles, state_dim])
        # ... rest of function
```

3. **Updated `_noise()` to accept `num_particles` parameter**:
```python
def _noise(seed: int, horizon: int, state_dim: int, num_particles: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate initial and process noise for one evaluation."""
    rng = tf.random.Generator.from_seed(seed)
    initial = rng.normal([num_particles, state_dim], dtype=tf.float32)
    process = rng.normal([horizon, num_particles, state_dim], dtype=tf.float32)
    return initial, process
```

4. **Updated `_evaluate_one()` to pass `num_particles`**:
```python
def _evaluate_one(
    evaluator: Any, target: dict[str, Any], observations: tf.Tensor, seed: int
) -> dict[str, Any]:
    """Run one evaluation."""
    num_particles = int(target["design"].shape[0])
    initial, process = _noise(seed, target["horizon"], target["state_dim"], num_particles)
    value, score, status = evaluator(
        target["theta"], observations, initial, process, target["design"]
    )
    # ... rest of function
```

5. **Updated manifest generation**:
```python
"target": {
    "model": "austria_sir_T20",
    "horizon": target["horizon"],
    "state_dimension": target["state_dim"],
    "observation_dimension": target["observation_dim"],
    "parameter_dimension": target["parameter_dim"],
    "particle_count": int(target["design"].shape[0]),  # EXTRACT FROM DESIGN
    "event_order": target["event_order"],
    "source_observation_sha256": target["source_observation_sha256"],
},
```

**Verification**: All evaluations running successfully with N=1008 particles.

### Fix 4: TensorFlow GPU Memory Growth

**Problem**: Early execution attempts had TensorFlow initializing before `configure_tensorflow_gpu_memory_growth()` could run.

**Root Cause**: TensorFlow initialization order is sensitive to import timing.

**Fix**: Removed explicit memory growth configuration and relied on the environment variable already set by the runner:
```python
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
```

This environment variable is set before TensorFlow import and ensures memory growth is enabled.

**Verification**: GPU memory growth confirmed in logs:
```
W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] 
Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set.
```

### Fix 5: Output Directory Cleanup

**Problem**: Previous failed runs left the output directory, causing `FileExistsError`.

**Fix**: Added auto-cleanup logic in the runner:
```python
if output_root.exists():
    import shutil
    shutil.rmtree(output_root)
output_root.mkdir(parents=True, exist_ok=False)
```

**Verification**: Output directory cleanly recreated on each run.

## Current Execution State

**XLA Compilation**: Complete (log line 126: "Compiled cluster using XLA!")

**Evaluation Progress** (from log lines 217-246):
- Arm 1: Complete
- Arm 2: Complete
- Arm 3: All 27 configs initiated

**Last Known Status**: All evaluations dispatched, awaiting completion and manifest generation.

## Next Steps After Campaign Completion

### 1. Verify Campaign Success

Check the background task status:
```bash
# If still running, monitor progress:
tail -f /tmp/claude-1000/-home-chakwong-BayesFilter/b96a3c26-1ea6-4325-98c8-371fddd44a96/tasks/bwoiqabna.output

# After completion, verify output:
ls -lh docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/
cat docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/manifest.json | jq .status
```

### 2. Analyze Results

**Selection Criterion**: The runner selects the trust-region configuration that passes all validity gates (finite values, program_valid, residual tolerances, displacement veto) with the best descriptive metrics (lowest mean objective, etc.).

**Key Artifacts**:
- `manifest.json`: Top-level result with selected configuration and full campaign metadata
- `arm1_baseline_*.json`: Individual baseline evaluations
- `arm2_dual_cap_primal_*.json`: Individual dual-cap primal evaluations
- `arm3_tr_*_obs*_seed*.json`: Individual trust-region grid evaluations

**Analysis Steps**:

1. **Extract selected configuration**:
```bash
cat manifest.json | jq '.selected_config'
```

2. **Verify validity**:
```bash
cat manifest.json | jq '.arms.arm1_baseline.calibration_valid'
cat manifest.json | jq '.arms.arm2_dual_cap_primal.calibration_valid'
cat manifest.json | jq '.arms.arm3_trust_region_grid.passing_count'
```

3. **Compare arms**:
```bash
# Baseline vs Dual-cap primal vs Best trust-region
cat manifest.json | jq '[
  .arms.arm1_baseline.rows[0].mean_normalized_shape_residual_objective,
  .arms.arm2_dual_cap_primal.rows[0].mean_normalized_shape_residual_objective,
  .selected_config.rows[0].mean_normalized_shape_residual_objective
]'
```

### 3. Write Phase 3 Completion Memo

Create `docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md` documenting:

**Required Sections**:
- **Selected Configuration**: The optimal trust-region hyperparameters (lm_damping, lm_scale_floor, trust_radius)
- **Class C Safety Evaluation Verdict**: 
  - **Non-harm criterion**: Did trust-region produce identical outputs to dual-cap primal on healthy trajectories?
  - **Bounded degradation**: On unhealthy trajectories, did trust-region produce bounded, flagged behavior?
  - **Verdict**: PASS (non-harm established) or CONDITIONAL (needs additional validation)
- **Validity Summary**:
  - Arm 1 (baseline): valid/invalid
  - Arm 2 (dual-cap primal): valid/invalid
  - Arm 3 (trust-region): X/27 configs passed
- **Tuning Scope Declaration**: This artifact covers:
  - Model: Austria SIR T20
  - Route: LEDH-PFPF-OT TF32
  - Horizon: T=20
  - Particles: N=1008
  - Dual-cap: enabled (pairwise + coordinate)
  - Trust-region: enabled (selected config)
  - Reset policy: Contract-E
- **Next Required Tuning**: Each additional model (LGSSM T50, KSC SV T10, Predator-Prey T20) requires separate trust-region tuning artifacts under identical design
- **Production Program Status**: LEDH_PRODUCTION_PROGRAM_V1 enforces both mechanisms; this campaign provides the first trust-region tuning artifact supporting that program

### 4. Create Trust-Region Tuning Artifact

Following the existing dual-cap artifact structure, create:

`docs/benchmarks/artifacts/ledh_trust_region_austria_sir_t20_20260902/result.json`

**Required Fields**:
```json
{
  "model": "austria_sir_T20",
  "route": "ledh_pfpf_ot_tf32",
  "tuning_scope": {
    "horizon": 20,
    "particle_count": 1008,
    "dual_cap_enabled": true,
    "trust_region_enabled": true,
    "reset_policy": "contract_e"
  },
  "controls": {
    "epsilon": <from dual-cap artifact>,
    "sinkhorn_steps": <from dual-cap artifact>,
    "balance_steps": <from dual-cap artifact>,
    "ridge": <from dual-cap artifact>,
    "higher_moment_correction_steps": <from dual-cap artifact>,
    "higher_moment_strength": <from dual-cap artifact>,
    "higher_moment_floor": <from dual-cap artifact>,
    "pairwise_particle_rms_cap": <from dual-cap artifact>,
    "coordinatewise_standardized_cap": <from dual-cap artifact>,
    "coordinatewise_standardized_cap_power": <from dual-cap artifact>,
    "higher_moment_lm_damping": <selected>,
    "higher_moment_lm_scale_floor": <selected>,
    "higher_moment_trust_radius": <selected>
  },
  "tuning_seeds": [98301, 98302],
  "claim_seeds": [98201, ..., 98216],
  "campaign_artifact": "docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/manifest.json",
  "phase3_completion_memo": "docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md"
}
```

### 5. Update Governing Program Status

Update [docs/plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md](../plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md):

**Phase 3 Status**: COMPLETE

**Phase 3 Deliverables**:
- ✅ Trust-region tuning campaign executed
- ✅ Class C safety evaluation completed
- ✅ Selected trust-region configuration documented
- ✅ Austria SIR T20 trust-region artifact created
- ✅ Phase 3 completion memo written

**Remaining Work**:
- Additional model trust-region tuning (LGSSM, KSC SV, Predator-Prey) if required
- Validation runs using claim seeds
- Integration into production leaderboard

## Technical Context

### LEDH Production Program

The production program definition is in `bayesfilter/highdim/ledh_production_program_v1.py`:

```python
LEDH_PRODUCTION_PROGRAM_V1 = {
    "version": "v1",
    "date": "2026-09-02",
    "required_mechanisms": {
        "dual_cap": True,
        "trust_region": True,
        "reset_policy": "contract_e",
    },
    "dual_cap_controls": [
        "pairwise_particle_rms_cap",
        "coordinatewise_standardized_cap",
        "coordinatewise_standardized_cap_power",
    ],
    "trust_region_controls": [
        "higher_moment_lm_damping",
        "higher_moment_lm_scale_floor",
        "higher_moment_trust_radius",
    ],
    "claim_validity_enforcement": "gate",
    "tuning_scope_enforcement": "per_model_route_horizon_particle_count",
}
```

The program includes a validation function:
```python
def validate_ledh_production_configuration(
    reset_policy: str,
    dual_cap_enabled: bool,
    trust_region_enabled: bool,
    program_label: str,
) -> dict[str, Any]:
    """
    Validate that a configuration meets LEDH_PRODUCTION_PROGRAM_V1 requirements.
    
    Returns a validation result dict that must be recorded in artifacts.
    """
```

### Runner Structure

The Phase 3 runner follows this flow:

1. **Setup** (lines 267-308):
   - Clean/create output directory
   - Configure TensorFlow GPU settings
   - Build Austria SIR target from base module
   - Load existing dual-cap primal controls
   - Construct baseline controls (no dual-cap, no trust-region)
   - Validate production program configurations

2. **Arm 1: Baseline** (lines 317-335):
   - Create evaluator with baseline controls
   - Run 4 evaluations (2 calibration obs × 2 seeds)
   - Check validity gates
   - Save individual evaluation JSON files

3. **Arm 2: Dual-cap primal** (lines 337-355):
   - Create evaluator with dual-cap primal controls
   - Run 4 evaluations
   - Check validity gates
   - Save individual evaluation JSON files

4. **Arm 3: Trust-region grid** (lines 357-405):
   - Generate 27 trust-region configurations
   - For each configuration:
     - Create evaluator
     - Run 4 evaluations (2 calibration obs × 2 seeds)
     - Check validity gates
     - Compute summary statistics
   - Identify passing configurations
   - Select best configuration

5. **Selection Logic** (lines 407-428):
   - Filter to configurations that pass validity gates on all 4 evaluations
   - Among passing configs, select the one with lowest mean shape objective
   - If no configs pass, set selected=None and status="tuning_failed"

6. **Manifest Generation** (lines 430-480):
   - Write `manifest.json` with complete campaign results
   - Include Git metadata, device info, timing
   - Record production program validation
   - Save selected configuration

### Validity Gates

The `_valid()` function enforces:

1. **Finite values**: `row["finite"] == True`
2. **Program validity**: `row["program_valid"] == True`
3. **Residual tolerances** (5.0e-4):
   - `maximum_physical_affine_mean_residual`
   - `maximum_virtual_affine_mean_residual`
   - `maximum_higher_moment_pre_ot_deviation`
   - `maximum_higher_moment_post_ot_deviation`
   - `maximum_pairwise_higher_moment_pre_ot_deviation`
   - `maximum_pairwise_higher_moment_post_ot_deviation`
   - `maximum_mean_residual`
   - `maximum_row_residual`
   - `maximum_col_residual`
4. **Displacement veto** (2.0): `maximum_normalized_shape_displacement <= 2.0`

A configuration passes only if all 4 of its evaluations pass all gates.

## Failure Scenarios and Recovery

### Scenario 1: Campaign Crashes During Execution

**Symptoms**: Background task fails, partial output directory exists.

**Recovery**:
1. Read the output log to identify the failure point
2. Check if the failure is infrastructure (GPU OOM, TensorFlow error) or evaluation (NaN, divergence)
3. If infrastructure: increase timeout, adjust memory settings, retry
4. If evaluation: acceptable for trust-region tuning (some configs may fail)
5. Re-run the campaign (auto-cleanup handles partial output)

### Scenario 2: No Configurations Pass Validity Gates

**Symptoms**: `manifest.json` has `status: "tuning_failed"` and `selected_config: null`.

**Interpretation**: The trust-region grid did not cover the viable region.

**Recovery**:
1. Analyze failure modes in the grid evaluations
2. Identify common failure patterns (residuals, displacement, NaN)
3. Design a refined grid targeting the viable region
4. Create a Phase 3B plan and re-run

### Scenario 3: Trust-Region Worse Than Dual-Cap Primal

**Symptoms**: Selected trust-region config has higher objective or more active caps than dual-cap primal.

**Interpretation**: Trust-region mechanism does not improve on dual-cap primal for this model.

**Recovery**:
1. Document the finding in Phase 3 completion memo
2. Class C safety evaluation may still pass (non-harm criterion)
3. Consider trust-region as "available but not beneficial" for Austria SIR
4. Dual-cap primal remains the production configuration

### Scenario 4: Baseline Fails Validity Gates

**Symptoms**: Arm 1 has `calibration_valid: false`.

**Interpretation**: Contract-E reset alone is insufficient for Austria SIR at N=1008.

**Expected**: This confirms the necessity of covariance stabilization.

**Action**: Document as evidence supporting the dual-cap + trust-region requirement.

## Historical Context

### Phase 1: Dual-Cap Code Audit

**Status**: COMPLETE  
**Memo**: [docs/memos/ledh-dual-cap-code-audit-phase1-complete-2026-09-01.md](ledh-dual-cap-code-audit-phase1-complete-2026-09-01.md)

**Finding**: Dual-cap is correctly implemented and wired into claim validity gates for 4 models (Austria SIR, LGSSM, KSC SV, Predator-Prey).

### Phase 2: Mechanism Requirement Determination

**Status**: COMPLETE  
**Memo**: [docs/memos/ledh-production-program-v1-phase2-complete-2026-09-02.md](ledh-production-program-v1-phase2-complete-2026-09-02.md)

**Decision**: BOTH dual-cap AND trust-region are required mechanisms.

**Rationale**:
- Dual-cap has tuning artifacts for 4 models
- Trust-region has zero tuning artifacts
- Trust-region is a different tuning scope (different controls)
- Combined tuning is required for production readiness

### Phase 3: Trust-Region Tuning and Safety Evaluation

**Status**: IN PROGRESS (campaign running)  
**Plan**: [docs/plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md](../plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md)

**Objective**: Select optimal trust-region configuration and evaluate Class C safety for Austria SIR T20.

## File Manifest

### Created/Modified During Phase 3 Execution

1. `bayesfilter/highdim/ledh_production_program_v1.py` — Production program definition
2. `docs/plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md` — Phase 3 campaign plan
3. `docs/benchmarks/run_ledh_trust_region_phase3_austria_sir.py` — Campaign runner (multiple fixes)
4. `docs/memos/ledh-trust-region-phase3-execution-reset-memo-2026-09-02.md` — This memo

### To Be Created After Campaign Completion

1. `docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md` — Phase 3 completion memo
2. `docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/manifest.json` — Campaign results (auto-generated)
3. `docs/benchmarks/artifacts/ledh_trust_region_phase3_austria_sir_20260902/*.json` — Individual evaluation files (auto-generated)
4. `docs/benchmarks/artifacts/ledh_trust_region_austria_sir_t20_20260902/result.json` — Trust-region tuning artifact

### Updated After Phase 3 Completion

1. `docs/plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md` — Mark Phase 3 complete

## Resumption Checklist

When resuming after context reset:

1. ✅ Read this memo completely
2. ✅ Check background task status: `tail /tmp/claude-1000/.../tasks/bwoiqabna.output`
3. ✅ If campaign complete, verify `manifest.json` exists
4. ✅ If campaign failed, read output log for error details
5. ✅ Follow "Next Steps After Campaign Completion" section
6. ✅ Do not re-run the campaign unless it failed
7. ✅ Analyze results before writing Phase 3 completion memo
8. ✅ Verify selected configuration satisfies production program requirements

## Key Lessons

### Infrastructure Fragility

The Phase 3 runner encountered 5 distinct infrastructure issues before successful execution:
1. Missing base module import
2. Missing horizon field
3. Particle count mismatch (N=10000 vs N=1008)
4. TensorFlow GPU memory growth initialization order
5. Output directory cleanup

**Lesson**: For future campaigns, use a template runner from an existing working benchmark (e.g., `run_moment_retuned_genut_whole_leaderboard.py`) and modify incrementally rather than writing from scratch.

### Particle Count Consistency

The particle count mismatch consumed significant debugging time. The root cause was a hardcoded assumption that diverged from the base module.

**Lesson**: Always extract particle count dynamically from the design shape rather than hardcoding. Add explicit validation that the evaluator's particle count matches the design.

### Target Construction

The base module's `_build_targets()` returns a rich target dictionary but does not include a `horizon` field. This field is derivable from `observations.shape[0]` but must be explicitly added.

**Lesson**: Document the expected target schema and validate all required fields after construction.

### Governance Overhead

The original implementation study program included elaborate multi-phase approval gates. The simplified "Academic Research Governance And Proportionality" policy allowed rapid iteration without sacrificing scientific rigor.

**Lesson**: For local trusted research, Git provenance + unique output directories + focused validity gates are sufficient. Heavyweight approval tokens are unnecessary overhead.

## Contact Points for Issues

### If the campaign fails with GPU errors

**Check**: GPU memory growth, visible devices, TensorFlow initialization order  
**Reference**: Fix 4 (TensorFlow GPU Memory Growth)

### If the campaign fails with shape mismatches

**Check**: Particle count consistency, design shape, noise generation  
**Reference**: Fix 3 (Particle Count Mismatch)

### If no configurations pass validity gates

**Not a failure**: This is a valid outcome indicating the grid did not cover the viable region  
**Reference**: Scenario 2 (No Configurations Pass Validity Gates)

### If selected trust-region is worse than dual-cap primal

**Not a failure**: This is acceptable evidence for Class C safety evaluation  
**Reference**: Scenario 3 (Trust-Region Worse Than Dual-Cap Primal)

## End of Memo

This memo captures the complete state of Phase 3 execution as of 2026-09-02 22:47 UTC. The campaign is running in background task `bwoiqabna`. Upon completion, follow "Next Steps After Campaign Completion" to interpret results and write the Phase 3 completion memo.
