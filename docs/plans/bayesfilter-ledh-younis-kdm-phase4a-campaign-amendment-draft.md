# Phase 4A.5 Campaign Amendment

**Status:** READY for approval  
**Parent plan:** [bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md](bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md)

## Timing pilot outcome

Timing pilot: [results/ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json](../../results/ledh_younis_kdm_phase4a_timing_pilot_20260907/timing.json)

Representative ladder (6 rows spanning N=32 to N=512, T=5 to T=50):

```text
N=32  T=5  rho=0.0 →  61 ms/row  (peak 0.6 MB)
N=32  T=5  rho=0.2 →  66 ms/row  (peak 0.6 MB)
N=128 T=20 rho=0.0 → 348 ms/row  (peak 16 MB)
N=128 T=20 rho=0.2 → 169 ms/row  (peak 16 MB)
N=512 T=50 rho=0.0 → 473 ms/row  (peak 550 MB)
N=512 T=50 rho=0.2 → 340 ms/row  (peak 550 MB)
```

**Peak GPU memory:** 550 MB (negligible — 4% of 13.5 GB available on 4080 SUPER)  
**Total 6-row ladder steady-state time:** 1.46 seconds  
**Compile overhead per configuration:** ~30-40 seconds (one-time per distinct N×T×rho)

All 6 configurations passed validity checks. The atom branch (rho=0) is slower than positive bandwidth for larger configurations due to additional boundary checks.

## Proposed calibration scope

Once the timing pilot completes, the calibration scope will be:

```text
horizon:       5, 20, 50
particle N:    32, 128, 512
rho grid:      0, .025, .05, .10, .20, .40, .80, 1.20, 1.60  (9 values)
replications:  TBD from pilot variance or power calculation
total rows:    3 × 3 × 9 × replications = 81 × replications
```

**Selection criterion:** Paired score MSE against Kalman oracle on disjoint calibration paths and streams.

**What is being selected:** The single best rho value per (N, T) cell, yielding 9 output pairs (one per cell).

**What is NOT being concluded:**
- Calibration does not establish HMC readiness, production readiness, or default promotion.
- The selected rho is specific to Phase 4A (kernelized observation, full feedback, diagnostic route only).
- Oracle comparison measures score accuracy, not posterior quality or computational cost relative to the canonical baseline.

## Proposed validation scope

After calibration selects one rho per (N, T) cell, run untouched validation on those 9 configurations:

```text
configurations: 9 (one per N×T cell, using calibration-selected rho)
replications:   TBD from pilot variance or paired-interval width target
total rows:     9 × replications
```

**Validation criterion:** Paired score MSE confidence/credible interval plus MCSE-adjusted comparison.

**What validation establishes:** Statistical support (or lack thereof) for ranking the Phase 4A route against the Kalman oracle on held-out paths.

**What validation does NOT establish:**
- Does not compare with the canonical analytical baseline or other LEDH routes.
- Does not establish DSGE validity, mixture-model readiness, or Phase 4B feasibility.
- Does not tune or re-select rho — holdout data never tune hyperparameters.

## Resource budget

Based on timing pilot extrapolation across the full intended ladder:

```text
Calibration configurations:  81 (3 N × 3 T × 9 rho)
Validation configurations:   9 (3 N × 3 T, using calibration-selected rho)
Proposed replications:       10 per configuration

Calibration rows:   81 × 10 = 810
Validation rows:    9 × 10 = 90
Total rows:         900

Estimated GPU-hours per calibration replication:  0.0092
Estimated GPU-hours per validation replication:   0.0010
Estimated total GPU-hours (10 replications):      0.10 hours (6 minutes)
Estimated peak GPU memory:                        550 MB (4% of available)
Estimated wall time (serial, with XLA compile):   ~15 minutes
```

**Budget cap:** 0.15 GPU-hours (9 GPU-minutes), 1000 total rows maximum. The campaign will NOT exceed this without renewed approval.

**Replication justification:** 10 replications per configuration is conservative for a bounded diagnostic campaign where the primary goal is to verify that the Phase 4A endpoint runs without divergence and produces finite oracle-comparison scores. Statistical ranking between rho values is secondary to establishing basic viability. A serious production campaign would require variance-driven power calculations.

## Hard gates before campaign launch

1. ✅ CPU reference analytical score passes (9/9 tests)
2. ✅ GPU/XLA readiness gate passes (FP32-no-TF32)
3. ✅ Timing pilot completes with realistic budget estimates (0.10 GPU-hours for 10 replications)
4. ⬜ Calibration and validation runner implements disjoint path/stream discipline
5. ⬜ Oracle Kalman baseline implemented and tested
6. ⬜ Paired score MSE computation verified
7. ⬜ This amendment approved

## Repair and retry policy

Within this campaign:
- Infrastructure, harness, or resource failures may be repaired and retried without renewed approval if the scientific contract (target, method, promotion criteria, vetoes, budgets) remains unchanged.
- Each attempt consumes campaign budget and uses a fresh versioned output directory.
- A scientific veto (divergence, invalid artifact, correctness-gate failure) stops the campaign for new direction.

## Next actions

The timing pilot is complete and budgets are established. The remaining implementation tasks before campaign launch are:

1. **Implement Kalman oracle baseline** — For a linear-Gaussian state-space model, compute the exact analytical log-likelihood and its score using the Kalman filter and smoother. This is the primary oracle for Phase 4A score comparisons.

2. **Implement calibration/validation runner** — A runner that:
   - Generates disjoint observation paths and random streams for calibration vs validation
   - Runs Phase 4A integrated endpoint and Kalman oracle on paired paths
   - Computes paired score MSE for each (N, T, rho) configuration
   - Selects best rho per (N, T) cell on calibration data
   - Reports paired confidence intervals on validation data
   - Preserves path-by-stream score vectors (not only averages) for recomputation

3. **Verify paired score MSE computation** — Unit test the paired-difference and MSE logic before campaign launch.

4. **Request approval** — Present this amendment with concrete budgets (0.15 GPU-hour cap, 1000 row cap, 10 replications) for user authorization.

After approval, the campaign launches and runs to completion under the declared budget. Results will inform whether Phase 4A kernelized observation weighting reduces score error relative to the Kalman oracle, which is a necessary (but not sufficient) condition for considering it further.
