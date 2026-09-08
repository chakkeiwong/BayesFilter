# Phase 4A Campaign Ready — Approval Requested

**Date:** 2026-09-08  
**Governing plan:** [bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md](bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md)  
**Campaign amendment:** [bayesfilter-ledh-younis-kdm-phase4a-campaign-amendment-20260908.md](bayesfilter-ledh-younis-kdm-phase4a-campaign-amendment-20260908.md)

## Status

**All implementation gates passed.** The Phase 4A campaign is ready for approval and launch.

## Implementation Complete (17 tests passed)

| Component | Tests | Status |
|---|---|---|
| Phase 4A integrated endpoint | 9/9 | ✅ CPU tests passed |
| GPU/XLA readiness | smoke | ✅ FP32-no-TF32 passes all gates |
| Timing pilot | 6 configs | ✅ 0.10 GPU-hour estimate established |
| Kalman oracle baseline | 3/3 | ✅ Analytical score verified |
| Campaign runner | 5/5 | ✅ Disjoint seeds, paired MSE verified |

**Total:** 17/17 tests passed

## Campaign Specification

**Research question:**  
Does Phase 4A kernelized observation weighting reduce score error relative to the Kalman oracle?

**Scope:**
- **Calibration:** 810 rows (3 N × 3 T × 9 rho, 10 reps each)
- **Validation:** 90 rows (3 N × 3 T, calibration-selected rho, 10 reps each)
- **Total:** 900 rows

**Budget:**
- **Estimate:** 0.10 GPU-hours (~6 GPU-minutes)
- **Cap:** 0.15 GPU-hours, 1000 rows maximum

**Primary criterion:**  
Paired score MSE on validation data (held-out paths/streams) for calibration-selected rho per (N, T) cell.

**Execution mode:**  
FP32-no-TF32, GPU/XLA, single 4080 SUPER

**Disjoint discipline:**
- Calibration: path seeds 100000+, noise seeds 200000+
- Validation: path seeds 300000+, noise seeds 400000+

## What This Campaign Establishes

✅ **Establishes:**
- Whether Phase 4A reduces score error vs oracle on LGSSM
- Best rho per (N, T) cell on calibration data
- Validation score MSE with confidence intervals

❌ **Does NOT establish:**
- HMC readiness or production readiness
- Default promotion for canonical score
- Comparison with ATOM-FINITE baseline
- DSGE validity or mixture-model readiness
- Phase 4B feasibility

## Approval Request

**This campaign requires approval because:**
1. It's a serious research campaign (not a smoke test)
2. It will produce statistical evidence for a research decision
3. It consumes GPU budget and creates durable artifacts

**The implementation is complete.** All gates passed. The campaign can launch immediately upon approval.

**Estimated wall time:** 15 minutes (serial execution with XLA compile overhead)

## Command to Launch

```bash
source ~/anaconda3/bin/activate tftwogpu && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 && \
python docs/benchmarks/run_ledh_younis_kdm_phase4a_campaign.py \
  --output results/ledh_younis_kdm_phase4a_campaign_20260908/result.json \
  --calibration-reps 10 \
  --validation-reps 10 \
  --dtype float32
```

## Artifacts

All implementation artifacts are committed and documented:
- [results/phase4a_implementation_complete_20260908.md](../results/phase4a_implementation_complete_20260908.md)
- [results/phase4a_execution_summary_20260908.md](results/phase4a_execution_summary_20260908.md)
- [results/ledh_younis_kdm_phase4a_gate_20260907.md](../results/ledh_younis_kdm_phase4a_gate_20260907.md)
- [results/ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json](../results/ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json)

**Ready for approval.**
