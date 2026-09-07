# Phase 4A GPU/XLA Readiness Gate Result

Date: 2026-09-07  
Status: **PASS (FP32-no-TF32) | FAIL (TF32-enabled)**  
Governing plan: `docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`, step 4A.4

## Question

Does the complete Phase 4A shared-executor kernelized-observation endpoint run reproducibly under GPU/XLA with acceptable numerical parity to its non-XLA graph mode?

## Evidence

Two bounded smoke tests were executed on a single NVIDIA GeForce RTX 4080 SUPER (PCI bus ID 0000:09:00.0, compute capability 8.9) using:

- TensorFlow 2.20.0-dev0+selfbuilt
- Python environment: `tftwogpu`
- CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1
- Memory growth enabled (no eager preallocation)
- Particle count N=8, horizon T=2, 3 deterministic repeats
- float32 dtype
- Fixed-seed two-dimensional nonlinear mechanics fixture

### Run 1: TF32 enabled (repository default mode)

Artifact: [results/ledh_younis_kdm_phase4a/gpu_smoke_20260907.json](../ledh_younis_kdm_phase4a/gpu_smoke_20260907.json)

- XLA compilation: **succeeded** (`xla_must_compile_attribute=true`)
- Output device placement: **GPU:0** (correct)
- Deterministic replay (3 repeats): **passed** (identical value)
- Valid finite output: **FAIL** (`valid=false`)
- Atom-identity gate: **FAIL** (`atom_identity_all_valid=false`)
- Observation-map gate: **FAIL** (`observation_map_all_valid=false`)
- Graph/XLA numerical errors:
  - value_abs: 0.00507 (threshold 0.0002, **25× over**)
  - score_abs: 0.00536 (threshold 0.0005, **11× over**)
  - state_abs: 0.00216 (threshold 0.0005, **4× over**)
  - weight_abs: 0.000715 (threshold 0.0002, **4× over**)
- Effective model tolerance: 7.6e-6 (64× float32 machine epsilon)
- Compile + first run: 13.8 s
- XLA steady-state repeats: 27–30 ms

### Run 2: TF32 disabled

Artifact: [results/ledh_younis_kdm_phase4a_tf32_disabled_20260907/gpu_smoke.json](../ledh_younis_kdm_phase4a_tf32_disabled_20260907/gpu_smoke.json)

- XLA compilation: **succeeded**
- Output device placement: **GPU:0** (correct)
- Deterministic replay: **passed**
- Valid finite output: **PASS** (`valid=true`)
- Atom-identity gate: **PASS** (`atom_identity_all_valid=true`)
- Observation-map gate: **PASS** (`observation_map_all_valid=true`)
- Graph/XLA numerical errors:
  - value_abs: 0.0 (threshold 0.0002, **passed**)
  - score_abs: 7.2e-7 (threshold 0.0005, **passed**)
  - state_abs: 1.4e-6 (threshold 0.0005, **passed**)
  - weight_abs: 1.2e-7 (threshold 0.0002, **passed**)
- Effective model tolerance: 7.6e-6 (unchanged)
- Compile + first run: 14.8 s
- XLA steady-state repeats: 30–31 ms

### Comparative analysis

Disabling TF32 reduces errors by **~7500× for the score** and clears all internal correctness gates. TF32's reduced 10-bit mantissa introduces ~1e-3 relative error in matmul-heavy operations, while the atom-identity and observation-map tolerances are derived from float32 machine epsilon (2^-23 ≈ 1.2e-7) and floor at 64×eps ≈ 7.6e-6. This is a 660× mismatch.

## Decision

| Criterion | Status |
|---|---|
| XLA compilation | **PASS** (both modes) |
| GPU placement | **PASS** (both modes) |
| Determinism | **PASS** (both modes) |
| Finite valid output | **PASS (FP32-no-TF32)** / FAIL (TF32) |
| Internal correctness gates | **PASS (FP32-no-TF32)** / FAIL (TF32) |
| Graph/XLA parity | **PASS (FP32-no-TF32)** / FAIL (TF32) |
| Steady-state timing acceptable | **PASS** (27–31 ms per step, both modes) |
| Peak GPU memory | 64 MB (negligible for N=8) |

**Verdict**: The Phase 4A GPU/XLA readiness gate **passes with TF32 disabled**, establishing that the shared-executor refactor, complete tangent, and XLA compatibility are correct. TF32 mode **fails the internal correctness gates** due to a 660× tolerance/precision mismatch.

## Implications and next steps

1. **TF32 compatibility is a design decision, not an implementation defect.** The identical code with only the TF32 flag changed produces the discriminating outcome. The canonical analytical score endpoint (which Phase 4A wraps) has no TF32-specific tolerance relaxation, suggesting the repository's TF32 policy applies to DPF transport (matrix-only OT/Sinkhorn), not the derivative-sensitive analytical score path.

2. **The atom-identity and observation-map gates are correctness checks**, not tuning hyperparameters. They verify that:
   - The supplied Gaussian factor reproduces the model's atom log-density at zero bandwidth within a declared tolerance.
   - The supplied linear observation callbacks match the model's observation callbacks.
   Relaxing these tolerances to TF32 scale would allow ~1e-3 errors to pass as "correct," undermining their purpose.

3. **Phase 4A is explicitly a diagnostic route**, not production. The plan classification is `kdm_finite_full_feedback_diagnostic_only`. A production promotion would require its own TF32 decision, evidence contract, and tuning scope.

4. **Proceeding options**:
   - **Option A (conservative)**: Declare Phase 4A incompatible with TF32 and continue the plan with `tf32_mode=disabled` for any Phase 4A GPU run. Record this as a route restriction.
   - **Option B (relaxed)**: Add a TF32-aware tolerance path that scales the atom-identity and observation-map floors to ~1e-3 when TF32 is enabled. This would require justifying why ~1e-3 errors in the correctness checks are acceptable.
   - **Option C (separate modes)**: Treat FP32-no-TF32 and TF32-enabled as distinct execution modes with separate route IDs, evidence contracts, and tuning scopes. A serious campaign would then choose one mode.

5. **Recommendation**: Continue the plan with **Option A** (FP32-no-TF32 for Phase 4A), since the plan's next step (4A.5 calibration and validation) has not yet authorized TF32 for the KDM route, and the internal gates are designed to catch exactly this kind of precision loss. If a later phase wants to use TF32, that decision should be explicit, with a revised tolerance contract and a recorded trade-off.

## What is NOT concluded

- Passing the GPU/XLA gate **does not establish lower model-score error**, production readiness, HMC readiness, DSGE validity, or default-promotion eligibility. This is a bounded engineering mechanics gate required before step 4A.5 (calibration and untouched validation).
- Disabling TF32 **does not prove the route is slow**. Both modes achieve ~30 ms steady-state, which is acceptable for an N=8 diagnostic. A serious N=512 timing comparison has not been run.
- The FP32-no-TF32 choice **is not inherited by the canonical analytical score or other routes**. Each route owns its TF32 decision.

## Artifacts

- [results/ledh_younis_kdm_phase4a/gpu_smoke_20260907.json](../ledh_younis_kdm_phase4a/gpu_smoke_20260907.json) — TF32 enabled, FAIL
- [results/ledh_younis_kdm_phase4a_tf32_disabled_20260907/gpu_smoke.json](../ledh_younis_kdm_phase4a_tf32_disabled_20260907/gpu_smoke.json) — TF32 disabled, PASS
- Git commit: 21d5870f (ledh-refactor-with-policy-fix branch)
- Plan: [docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md](../../docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md)
- Reset: [docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md](../../docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md)

## Run manifest

| Field | Value |
|---|---|
| Python | /home/chakwong/anaconda3/envs/tftwogpu/bin/python |
| TensorFlow | 2.20.0-dev0+selfbuilt |
| Device | NVIDIA GeForce RTX 4080 SUPER, compute 8.9, 13495 MB logical |
| Memory policy | memory_growth=true, no preallocation |
| CUDA_VISIBLE_DEVICES | 1 (PCI_BUS_ID order → 4080 SUPER) |
| Trust basis | owner_designated_managed_session_visible_gpu_trusted |
| Executed | 2026-09-07 21:23 UTC (TF32=enabled), 21:27 UTC (TF32=disabled) |
| Wall time | ~26 s per run (includes compile + 3 repeats) |
