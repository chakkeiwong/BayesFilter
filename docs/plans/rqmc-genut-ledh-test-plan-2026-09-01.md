# RQMC GenUT Initialization Test Plan

**Date:** 2026-09-01  
**Author:** Claude Code (Opus 5)  
**Status:** Draft for review

## Research Question

Can randomized quasi-Monte Carlo (RQMC) methods improve LEDH-PFPF-OT particle initialization over standard Monte Carlo (MC) for the dual-cap trust-region algorithm?

## Background

The GenUT guided initialization work (rescued from `experiment/genut-guided-initialization`) tested four RQMC initialization methods against equal-weight MC initialization on three benchmark models (Austria SV, KSC, J0-common-target). The current LEDH production route (`LEDH_PRODUCTION_PROGRAM_V1`) uses the dual-cap trust-region mechanism enabled by default, which was not available during the original GenUT investigation.

The four RQMC methods under test:
1. **Sobol-Matousek scrambling** (Matousek 1998 nested uniform scrambling)
2. **Sobol-Owen scrambling** (Owen 1995 random digital shift + scramble)
3. **Halton-Owen scrambling** (Owen 2017 randomized Halton with drop-and-permute)
4. **GenUT guided proposal** (Ebeigbe et al. 2021 generalized unscented transformation)

## Mechanism Under Test

RQMC initialization of the N=1008 particle cloud at t=0, before the first LEDH prediction-update step. The initialization methods differ only in how they sample from the prior; all subsequent transport, resampling, and covariance updates use the identical LEDH-PFPF-OT dual-cap route.

## Test Models

Three benchmark models from the rescued GenUT artifacts:
1. **Austria SV (N=1008, T=100, D=2):** stochastic volatility with Student-t observations
2. **KSC (N=1008, T=945, D=2):** Nile river flow with seasonal dynamics
3. **J0 common-target validation (N=1008, T=50, D=5):** synthetic 5D linear-Gaussian target

## Arms

1. **Equal-weight MC baseline:** standard `tf.random.normal` initialization
2. **Sobol-Matousek:** Sobol sequence with Matousek scrambling
3. **Sobol-Owen:** Sobol sequence with Owen scrambling  
4. **Halton-Owen:** Halton sequence with Owen randomized permutation
5. **GenUT guided:** generalized unscented transformation with prior covariance

## Production Program Configuration

All arms run under `LEDH_PRODUCTION_PROGRAM_V1`:
- Route: `ledh_pfpf_alg1_ukf_tf` with per-particle UKF predict/update
- Covariance: triple {x,P,w} resampling with OT/Sinkhorn carry
- Trust region: **dual-cap enabled** (coordinate and pairwise caps)
- Chunk policy: `dpf_transport_exact_divisor_cap3000_v1` (K=N for N≤3000)
- Backend: TensorFlow/TFP with TF32 enabled on GPU
- Dtype: `float32`

## Tuning Status

Per-scope tuning is **required** before claim-bearing runs. The original GenUT tuning artifacts used N=1008 but predate the dual-cap mechanism. Fresh tuning scopes:
- Model: Austria SV / KSC / J0
- Route: LEDH-PFPF-OT dual-cap
- Particle count: N=1008
- Horizon: T=100 / T=945 / T=50
- Dtype/backend: float32 / TF32-GPU
- Controls: Sinkhorn epsilon, balance iterations, dual-cap radii

**Budget:** 5 tuning runs per model (warm-start from rescued artifacts where applicable), 3 claim-bearing seeds per arm per model.

## Primary Criterion

**Log-likelihood estimate at final time T.** Higher is better. The LEDH score lane computes the incremental log-weight sum; we compare the terminal value across arms.

## Promotion Criterion

An RQMC arm is **viable for promotion** if:
1. All N=1008 × T time steps complete without divergence, NaN, or dual-cap saturation
2. Terminal log-likelihood mean across 3 seeds is within 2 standard errors of the MC baseline or higher
3. No continuation veto fires (see below)

An RQMC arm **replaces MC as the default** only if terminal log-likelihood shows statistical superiority (bootstrap 95% CI excludes zero difference) across all three models.

## Promotion Veto

Hard veto (immediate disqualification):
- Divergence, NaN, or non-finite score at any time step
- Dual-cap saturation >10% of particles at any time step
- Missing required per-scope tuning artifact

## Continuation Veto

Stop the entire campaign if:
- Tuning fails to converge for >1 model after 5 attempts
- All RQMC arms show terminal log-likelihood statistically inferior to MC baseline (bootstrap 95% CI excludes positive difference) on Austria SV
- Infrastructure failure (GPU OOM, serialization corruption, harness crash) persists after 2 repair attempts

## Explanatory Diagnostics

Recorded per arm per model per seed, not used for promotion:
- Per-time-step log-likelihood increment (trace plot)
- Effective sample size (ESS) at T
- Sinkhorn iteration count (mean, max)
- Dual-cap activation rate (pairwise, coordinate)
- Runtime (wall seconds, GPU utilization)

## What Will Not Be Concluded

- **Not claiming correctness:** RQMC initialization does not alter the LEDH target distribution; only sampling efficiency and Monte Carlo variance are under test.
- **Not claiming horizon-independence:** results are specific to T=100 / T=945 / T=50; longer horizons require fresh evidence.
- **Not claiming model-independence:** results are specific to Austria SV, KSC, J0; other models require their own validation.
- **Not claiming tuning-free:** each RQMC arm requires its own per-scope tuning; transferred settings are warm-start only.

## Evidence Artifacts

Each run produces:
- **Manifest:** git commit, command, conda env, GPU status, tuning artifact path, seeds
- **Score trace:** per-seed terminal log-likelihood and per-step increment
- **Diagnostic bundle:** ESS, Sinkhorn counts, dual-cap rates, runtime
- **Decision table:** promotion status, veto status, statistical ranking

Output root: `docs/benchmarks/artifacts/rqmc_genut_ledh_dual_cap_20260901/`

## Commands

Tuning phase (one per model):
```bash
conda run -n tftwogpu python docs/benchmarks/tune_ledh_dual_cap_austria_sv_n1008.py \
  --output docs/benchmarks/artifacts/rqmc_genut_ledh_dual_cap_20260901/tuning/austria_sv_n1008/
```

Claim-bearing runs (3 seeds × 5 arms × 3 models = 45 runs):
```bash
for model in austria_sv ksc j0_common_target; do
  for arm in mc sobol_matousek sobol_owen halton_owen genut_guided; do
    for seed in 1001 1002 1003; do
      conda run -n tftwogpu python docs/benchmarks/run_ledh_rqmc_initialization.py \
        --model $model \
        --arm $arm \
        --seed $seed \
        --tuning docs/benchmarks/artifacts/rqmc_genut_ledh_dual_cap_20260901/tuning/${model}_n1008/ \
        --output docs/benchmarks/artifacts/rqmc_genut_ledh_dual_cap_20260901/runs/${model}_${arm}_seed${seed}/
    done
  done
done
```

Result assembly:
```bash
conda run -n tftwogpu python docs/benchmarks/assemble_rqmc_results.py \
  --input docs/benchmarks/artifacts/rqmc_genut_ledh_dual_cap_20260901/runs/ \
  --output docs/benchmarks/artifacts/rqmc_genut_ledh_dual_cap_20260901/results.json
```

## Next Steps After Completion

1. If all RQMC arms fail promotion veto on Austria SV: stop, write result note, archive.
2. If ≥1 RQMC arm passes promotion criterion on all 3 models: write promotion proposal, request owner approval for default change.
3. If RQMC arms are viable but not statistically superior: write result note, keep as optional feature, do not change default.

## Pre-Execution Checklist

- [ ] Tuning runner exists for each model
- [ ] Claim-bearing runner exists and accepts --model, --arm, --seed, --tuning, --output
- [ ] Result assembler exists and computes bootstrap CIs
- [ ] GPU memory growth enabled and verified in runner
- [ ] Configuration-status-first reporting enforced in result assembler
- [ ] Continuation veto logic implemented in runner or supervisor script
- [ ] Fresh output directory created with unique timestamp

## Review Notes

(To be filled by reviewer)

---

**Plan approved by:** (awaiting review)  
**Execution authorized:** (awaiting approval)
