# Phase 4A Execution Summary — 2026-09-08

## Current status

This file contains historical pre-repair timing and campaign planning
material. The authoritative Phase 4A result is the all-bandwidth Attempt 05
under docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt05-n32t5-all-rho-f32-current/.
It selected rho=0.8, found 4.84% higher validation MSE than ATOM-FINITE,
and found no positive point-MSE gain for any prespecified positive bandwidth.
The broad Phase 4A ladder is paused. Phase 4B is no longer a mathematical
blocker: its separately labelled `RESKDM-IWSG-FINITE` reference is implemented
and has passed bounded CPU correctness checks. A source and author-code audit
found that the earlier GPU-smoked implementation computed the distinct
`RESKDM-SN-FINITE` derivative by prematurely normalizing IWSG ratios; those
smokes must be rerun for the repaired route. Its score quality remains
unevaluated.

**Governing plan:** [docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md](../bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md)  
**Reset:** [docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md](../bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md)

## Correction: current evidence supersedes the historical timing block

The timing and campaign paragraphs below were written before the Contract-E
covariance-carry repair and are retained as historical provenance only.  They
must not be used as the current budget or readiness statement.  The repaired
`N=128,T=20`, float64/XLA/no-TF32 probe required `296.1017 s` for compile plus
first execution, `0.2318 s` per warm execution, and `31,457,280` peak GPU
bytes. The authoritative Attempt 05 `N=32,T=5`, float32/no-TF32
all-bandwidth result completed with `NO_PROMOTION_EVIDENCE`: KDM-FINITE at the
selected bandwidth was descriptively `4.84%` worse than ATOM-FINITE, and its
paired bootstrap interval crossed zero. See
[`bayesfilter-ledh-younis-kdm-phase4a-campaign-result-20260908.md`](bayesfilter-ledh-younis-kdm-phase4a-campaign-result-20260908.md)
and the complete artifact directory
`docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt05-n32t5-all-rho-f32-current/`.

The broad ladder is paused.  The TF32 arm also remains vetoed by the repaired
identity gate; only float64 or float32 without TF32 is eligible for further
diagnostic work.  No canonical-default, HMC, DSGE, or Phase 4B claim follows
from the pilot.

## Phase 4A: Integrated Kernelized Observation Weighting

Phase 4A implements a fully differentiated, full-feedback diagnostic route that replaces the point observation factor with the exact linear-Gaussian convolution `N(y; C x, R + C B C^T)`, then sends the resulting weights and their total tangent through the shared canonical Contract-E/GenUT/dual-cap executor. This is a **KDM-FINITE** route — it changes the finite target and is not the canonical ATOM-FINITE score.

## Execution Status

### ✅ Phase 4A.1: Shared-engine refactor (Complete)

- Extracted one internal analytical executor (`_value_and_analytical_score_impl`)
- Canonical endpoint calls it with ordinary observation factor
- Phase 4A wrapper calls it with convolved observation factor
- No copy of flow, UKF, PF-PF, Contract-E, GenUT, or dual-cap recurrence
- All entry points resolve, existing tests pass

### ✅ Phase 4A.2: Complete Gaussian factor and total tangent (Complete)

- Implemented `linear_gaussian_kdm_observation_factors` with complete `dx`, `dC`, `dR`, `dB` tangents
- Zero-bandwidth branch explicit (no Cholesky/inverse of zero covariance)
- Observation-map and atom-identity checks verify supplied Gaussian representation matches model callbacks
- Rank-deficient PSD bandwidth accepted (only effective observation covariance is factored)

### ✅ Phase 4A.3: Full-feedback gates (Complete)

**CPU reference tests:** [tests/highdim/test_ledh_younis_kdm_integrated_tf.py](../tests/highdim/test_ledh_younis_kdm_integrated_tf.py)  
**Result:** 9/9 tests passed (CUDA_VISIBLE_DEVICES=-1, CPU-only)

- Zero-bandwidth value, score, weights, reset states, and trace parity with canonical
- Positive-bandwidth value shift reported and nonzero on discriminating fixture
- Analytical score matches central finite difference of complete finite program
- Individual `dx`, `dC`, `dR`, `dB` component tests pass
- Singular PSD bandwidth accepted, indefinite bandwidth rejected
- Two-step test proves changed KDM weights feed through Contract-E and dual caps into later states

### ✅ Phase 4A.4: Static graph/GPU readiness (Complete)

**GPU/XLA gate artifact:** [results/ledh_younis_kdm_phase4a_gate_20260907.md](ledh_younis_kdm_phase4a_gate_20260907.md)

**Verdict:** **PASS with FP32-no-TF32** | FAIL with TF32-enabled

- XLA compilation: ✅ succeeded (`xla_must_compile_attribute=true`)
- GPU placement: ✅ GPU:0 correct
- Deterministic replay: ✅ passed (3 repeats, identical value)
- TF32 disabled: ✅ all gates pass, errors ~1e-6
- TF32 enabled: ❌ atom-identity and observation-map gates fail, errors ~5e-3

**Root cause:** TF32's reduced 10-bit mantissa (~1e-3 relative precision) is incompatible with the float32-eps-derived tolerance floor (7.6e-6). Disabling TF32 reduces errors by ~7500× and clears all internal correctness gates.

**Decision:** Phase 4A continues with **FP32-no-TF32** (the conservative choice that passes all gates). The canonical analytical score has no TF32-specific tolerance relaxation, so this is consistent with repository policy.

**Timing pilot artifact:** [results/ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json](ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json)

Representative ladder (6 rows):

```text
N=32  T=5  rho=0.0 →  61 ms/row  (peak 0.6 MB)
N=32  T=5  rho=0.2 →  66 ms/row  (peak 0.6 MB)
N=128 T=20 rho=0.0 → 348 ms/row  (peak 16 MB)
N=128 T=20 rho=0.2 → 169 ms/row  (peak 16 MB)
N=512 T=50 rho=0.0 → 473 ms/row  (peak 550 MB)
N=512 T=50 rho=0.2 → 340 ms/row  (peak 550 MB)
```

All configurations valid. Peak GPU memory 550 MB (4% of available 13.5 GB). Total steady-state time for 6 rows: 1.46 seconds.

### ⬜ Phase 4A.5: Calibration and untouched validation (Completed; no promotion)

**Campaign amendment:** [docs/plans/bayesfilter-ledh-younis-kdm-phase4a-campaign-amendment-20260908.md](../docs/plans/bayesfilter-ledh-younis-kdm-phase4a-campaign-amendment-20260908.md)

**Status:** Small-cell all-bandwidth campaign completed; no promotion evidence

**Completed implementation:**
- ✅ Kalman oracle baseline (3/3 tests passed)
- ✅ Calibration/validation runner with disjoint path/stream discipline (5/5 tests passed)
- ✅ Paired score MSE computation verified

**Proposed scope:**
- Calibration: 81 configurations (3 N × 3 T × 9 rho), 10 replications = 810 rows
- Validation: 9 configurations (3 N × 3 T, calibration-selected rho), 10 replications = 90 rows
- Total: 900 rows, 0.10 GPU-hours estimate, 0.15 GPU-hour cap

## Current Checkpoint

Phase 4A engineering (steps 4A.1–4A.4) and the bounded small-cell campaign
are complete. The integrated endpoint:
- ✅ Shares the canonical executor (no copy of LEDH recurrence)
- ✅ Differentiates the complete Gaussian factor with `dx`, `dC`, `dR`, `dB`
- ✅ Passes zero-bandwidth atom-identity gates
- ✅ Feeds changed weights through Contract-E and dual caps (full feedback)
- ✅ Compiles under GPU/XLA with FP32-no-TF32
- ✅ Runs deterministically with recorded steady-state timing

The route is not promoted. The consumed Attempt 05 holdout gives no positive
bandwidth a point-MSE advantage and does not support a new default.

## What Phase 4A Does NOT Establish

- Does not show lower model-score error (that's the 4A.5 question)
- Does not establish HMC readiness, production readiness, or default promotion
- Does not implement the complete Younis mixture-density particle filter (Phase 4B is a separate route)
- Does not prove DSGE validity or degenerate-support correctness
- The KDM-FINITE target is a new scalar; zero-bandwidth parity does not make positive bandwidth equal to ATOM-FINITE

## Artifacts

- CPU tests: [tests/highdim/test_ledh_younis_kdm_integrated_tf.py](../tests/highdim/test_ledh_younis_kdm_integrated_tf.py)
- GPU/XLA gate: [results/ledh_younis_kdm_phase4a_gate_20260907.md](ledh_younis_kdm_phase4a_gate_20260907.md)
- Timing pilot: [results/ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json](ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json)
- Implementation: [bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py](../bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py)
- Git branch: `ledh-refactor-with-policy-fix`
- Current source checkpoint: 76f09a6d plus the subsequent Phase 4B audit changes
