# RQMC LEDH Campaign: Process-Noise Confound Analysis

**Date:** 2026-09-06  
**Status:** Campaign invalidated — process-noise confound violates master program  
**Related:** docs/memos/rqmc-ledh-correction-and-phase2b-completion-2026-09-05.md  

## Summary

The RQMC LEDH initialization campaign (45 runs, 3 models, 5 arms) is **scientifically invalid** due to a process-noise confound. The `mc` arm and all RQMC arms receive **different process-noise trajectories** for the same seed, violating the master program's requirement that "all subsequent LEDH transport, resampling, and covariance updates are identical across arms."

The measured penalties (KSC: −0.050, LGSSM: −0.099, Predator-Prey: −1.026) conflate initialization quality with uncontrolled trajectory differences. The confound accounts for **62% of the KSC penalty** and **137% of the LGSSM penalty** (i.e. the sign reverses after correction).

## Root Cause

The runner (`run_rqmc_ledh_initialization.py`) draws initial noise and process noise from **one shared `tf.random.Generator`**:

```python
def _run_evaluation(...):
    rng = replication_generator(seed)
    initial_noise = _generate_initial_noise(arm, state_dim, seed, rng)
    process_noise = _generate_process_noise(horizon, state_dim, seed, rng)
```

- **MC arm**: `_generate_initial_noise` calls `rng.normal([1008, state_dim])` → advances the stream by `1008 × state_dim` elements → process noise starts at offset `1008 × state_dim`
- **RQMC arms** (sobol, halton, genut): initial noise comes from scipy/tfp or a fixed design → `rng` is **not advanced** → process noise starts at offset `0`

For the same seed, arms get different random trajectories *by construction*.

## Measured Confound

Using [diagnose_rqmc_process_noise_confound.py](diagnose_rqmc_process_noise_confound.py), I held the initial cloud fixed and varied only the process-noise stream offset:

| Model            | As-run penalty | Offset-matched | Confound contribution |
|------------------|---------------:|---------------:|----------------------:|
| KSC SV T10       |         −0.050 |         −0.019 |   −0.031 (62% of penalty) |
| LGSSM T50        |         −0.099 |         +0.037 |   −0.136 (137%: sign flips!) |
| Predator-Prey T20|         −1.026 |         −0.939 |   −0.088 (9% of penalty) |

**KSC SV T10** (state_dim=1, horizon=10):
- Process-noise-only effect: max |Δ| = 0.05 (held initialization fixed)
- Confound accounts for 62% of the measured −0.050 penalty
- True initialization effect: −0.019 (not statistically distinguishable at n=3)

**LGSSM T50** (state_dim=3, horizon=50):
- Process-noise-only effect: max |Δ| = 0.44
- Confound accounts for −0.136, **larger than** the measured −0.099 penalty
- True initialization effect: **+0.037** (sobol superior after correction, but n=3 too small)

**Predator-Prey T20** (state_dim=2, horizon=20):
- Process-noise-only effect: max |Δ| = 1.30
- Confound accounts for −0.088 (9% of penalty)
- True initialization effect: −0.939 (still a large penalty, now attributable to initialization)

## Secondary Defect: GenUT Cloud Degeneracy

The `genut_guided` arm uses `gaussian_genut_design(dim=d)` → `replicate_positive_genut(num_particles=1008)`. This produces **3-6 distinct particles out of 1008**:

- d=1: 3 distinct rows (0.3%)
- d=2: 5 distinct rows (0.5%)
- d=3: 6 distinct rows (0.6%)

For Predator-Prey T20, that extreme degeneracy (5 distinct particles) explains the massive −8.18 penalty. This is a **separate defect** from the confound, but it further disqualifies the campaign.

## Why This Invalidates All 45 Runs

The master program states:

> **Mechanism tested**: Particle cloud initialization at t=0 only. All subsequent LEDH transport, resampling, and covariance updates are identical across arms.

The confound violates that contract. Every comparison in the campaign conflates:
1. Initialization quality (the intended mechanism)
2. Process-noise trajectory differences (uncontrolled confound)

There is **no valid way to separate these** post hoc. The measured penalties are scientifically meaningless.

## What Would Be Required for a Valid Campaign

1. **Fix the runner**: seed initial and process noise **independently**
   ```python
   initial_rng = replication_generator(seed)
   process_rng = replication_generator(seed + OFFSET)  # or SeedSequence
   initial_noise = _generate_initial_noise(arm, state_dim, seed, initial_rng)
   process_noise = _generate_process_noise(horizon, state_dim, seed, process_rng)
   ```

2. **Fix GenUT degeneracy**: either use the full symmetric design (not just positive quadrant replication), or replace `genut_guided` with a non-degenerate RQMC sequence

3. **Increase replication**: n=3 seeds is too small to establish statistical significance even without confounds

4. **Re-run the entire campaign**: all 45 cells must be regenerated with the fixed runner

## Verdict

The campaign is **scientifically invalid** and its artifacts **cannot be used for any claim**. The verdicts in [rqmc-ledh-correction-and-phase2b-completion-2026-09-05.md](rqmc-ledh-correction-and-phase2b-completion-2026-09-05.md) are correct *relative to the confounded data*, but that data does not answer the research question.

**Do not repair, re-analyze, or build on this campaign.** Start fresh with a corrected runner.

---

## Artifacts

- Confound diagnostic script: [diagnose_rqmc_process_noise_confound.py](../benchmarks/diagnose_rqmc_process_noise_confound.py)
- Confound measurements:
  - [process_noise_confound_ksc_sv_T10.json](../benchmarks/artifacts/rqmc_ledh_init_v1_20260904/process_noise_confound_ksc_sv_T10.json)
  - [process_noise_confound_lgssm_T50.json](../benchmarks/artifacts/rqmc_ledh_init_v1_20260904/process_noise_confound_lgssm_T50.json)
  - [process_noise_confound_predator_prey_T20.json](../benchmarks/artifacts/rqmc_ledh_init_v1_20260904/process_noise_confound_predator_prey_T20.json)
- Invalidated campaign artifacts: [docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/](../benchmarks/artifacts/rqmc_ledh_init_v1_20260904/)
