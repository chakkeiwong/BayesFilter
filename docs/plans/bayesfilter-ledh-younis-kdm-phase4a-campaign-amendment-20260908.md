# Phase 4A.5 Campaign Amendment

**Status:** REVISED_AFTER_REPAIRED_TIMING; SMALL_CELL_PILOT_COMPLETED; NO_PROMOTION_EVIDENCE; BROAD_LADDER_PAUSED  
**Parent plan:** [bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md](bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md)

## Skeptical audit and supersession

The original timing/budget paragraph below was written against an older,
pre-repair artifact and is retained only as history.  It is not an execution
budget.  The current repaired probe at `N=128,T=20`, float64, XLA, TF32
disabled took `296.1017 s` for compile plus first execution, `0.2318 s` per
warm execution, and `31,457,280` peak GPU bytes.  The old six-row estimate of
30--40 second compiles and 0.10 GPU-hours is therefore stale relative to the
current call chain.  The campaign must begin with a small-cell compile/variance
pilot and must not infer N=512/T=50 feasibility from the stale estimate.

The runner, independent Kalman reference, disjoint split logic, and paired
bootstrap calculation are implemented and pass focused CPU tests.  The first
runner attempt was deliberately withheld until the synthetic observations were
corrected to draw the declared stationary initial state; no research artifact
was created from the rejected generator.

The authorized small-cell pilot is now complete.  The powered `N=32,T=5`,
float32/no-TF32 run used 20 calibration and 100 validation paths, selected
`rho=0.8`, and produced `NO_PROMOTION_EVIDENCE`: KDM-FINITE had 4.84% higher
validation score MSE than ATOM-FINITE, with a paired bootstrap interval that
crossed zero.  The full result is recorded in
`docs/plans/results/bayesfilter-ledh-younis-kdm-phase4a-campaign-result-20260908.md`.
The broad ladder is paused; the historical budget below is not an execution
authorization.

## Historical pre-repair timing note

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

## Historical proposed calibration scope (superseded)

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

## Historical proposed validation scope (superseded)

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

## Historical resource budget (superseded)

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

## Pilot gate record

1. ✅ CPU reference analytical score and covariance-carry tests pass.
2. ✅ Complete GPU/XLA endpoint passes float64 and float32 without TF32.
3. ✅ The repaired timing probe is recorded; the older six-row estimate is
   superseded by the 296.10-second `N=128,T=20` compile probe.
4. ✅ Independent scalar Kalman reference and runner pass focused tests.
5. ✅ Calibration and validation use disjoint path/stream discipline.
6. ✅ Paired score MSE and bootstrap calculation are preserved in row-level
   artifacts.
7. ✅ Small-cell pilot completed; the promotion criterion failed and the broad
   ladder is paused.

## Repair and retry policy

Within this campaign:
- Infrastructure, harness, or resource failures may be repaired and retried without renewed approval if the scientific contract (target, method, promotion criteria, vetoes, budgets) remains unchanged.
- Each attempt consumes campaign budget and uses a fresh versioned output directory.
- A scientific veto (divergence, invalid artifact, correctness-gate failure) stops the campaign for new direction.

## Next actions

1. Run a focused same-stream diagnostic over the complete rho grid at the
   already compiled small cell, comparing every candidate to both ATOM-FINITE
   and the Kalman oracle.  Keep this diagnostic separate from tuning and do not
   spend the superseded broad-ladder budget.
2. Use that table to decide whether the observed loss is finite-particle atom
   error, positive-bandwidth bias, or a route-level implementation issue.  A
   larger `N,T` cell requires its own compile probe and a new evidence contract.
3. Keep Phase 4B blocked until the complete Younis mixture proposal law,
   support measure, numerator, ancestry rule, and anchored derivative are
   written and independently reviewed.
4. Repair or quarantine the registered `batch_fused`/NeuTra lane separately;
   its current reduced recurrence bypasses Contract-E, GenUT, and the dual
   caps and cannot support a canonical full-algorithm claim.
