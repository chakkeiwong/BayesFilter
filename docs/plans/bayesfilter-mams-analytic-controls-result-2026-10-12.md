# D4R3 analytic-control result

Date: 12 October 2026. The [reviewed plan](bayesfilter-mams-analytic-controls-2026-10-12.md)
was executed. The model-map defects are repaired, the analytic consumer now
monitors declared model/tail quantities, and the oscillation and failure
controls pass. The repeated reference experiment also exposed limited MCSE
availability under strong antithetic dependence. It does **not** establish
general calibration, MAMS robustness or posterior accuracy.

## Changes and independent checks

The partial-funnel model map now implements its declared density:
`v=3*z0`, `x=exp(v/2+delta)*zchild`, `delta=.5*tanh(z0/2)`. Its identity is
versioned. The exact-map control represents the same scale-three funnel.
Independent centered-density and finite-difference Jacobian checks pass.
Both fixtures now start from the same model points by applying the inverse
partial chart; identical latent numbers would give different model starts.

The new diagnostic quantity module supplies model neck/flare probabilities,
conditional child-tail events, Student-t tails and known moments. Model child
moments with highly unstable short-run uncertainty are retained descriptively.
Binary events use the existing explicitly declared probability diagnostics.
Missing/nonpositive MCSE, constant tails and missing state-space references
remain unavailable. Neither reference failure nor any posterior diagnostic
changes tuning membership.

Forty regressions passed from the immutable `analytic-source-eca11e7f0`
archive in 35.81 seconds (38.04 including startup). They cover the new
quantities, actual XLA randomized-HMC periodic returns, signed antithetic
correlations with slow squares, existing pilot recovery/streams/work and
validation lifecycle. The earlier focused run passed 24 tests in 8.75 seconds
including startup. Two additional review regressions passed in 4.86 seconds:
matched model starts, and a bounded finite bank with **exact** mean zero and
second moment `5/3` but no Student-t tails. That last control isolates the new
tail check: means/squares agree while the absent-tail uncertainty is correctly
inconclusive. These are 42 distinct engineering regressions, not 66 independent
scientific replications.

## Calibration and provenance

The exact-law calibration ran from commit
`eca11e7f0c0a6a32bc9b94769775c73ad8677169`, tree
`788d28d521bccb9a473972277fa1d671bd721416`, in TensorFlow 2.20.0 / TFP
0.25.0. It used CPU/XLA with GPUs deliberately hidden and two TensorFlow
threads. Exact synthetic reference generation used diagnostic NumPy/SciPy;
the quantities and lugsail calculations used TensorFlow. No MAMS chain was
generated in this calibration; the separate kernel regressions and prior
public-pipeline runs must not be conflated with it.

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/calibrate_mams_analytic_diagnostics.py \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/analytic-controls-01 \
  --seed 5700 --replications 32 --draws 2048 --max-wall-seconds 1200
```

The [terminal audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/analytic-controls-01/terminal-audit.json)
verified all 144 source hashes, the twelve cells, 384 distinct positive
replicate seeds, 192 paired negative controls, and all quantity-family
denominators. The sibling manifest and result preserve command, source,
environment, quantities, errors, MCSEs, intervals, seeds and wall time. The
driver completed in 23.603492 seconds, within its existing local allowance.
The later start-bank correction does not rewrite this archived experiment;
that experiment draws exact reference states and does not use fixture starts.

Each entry below is `discrepancy / inconclusive / no discrepancy`, out of
32 independent replicate chains, each with four chains and 2048 draws.
Rho is the stationary latent Gaussian AR(1) parameter; Student-t margins
are transformed by the exact inverse CDF.

| Reference target | rho=0 | rho=0.9 | rho=-0.9 |
| --- | --- | --- | --- |
| Gaussian | 1 / 0 / 31 | 1 / 0 / 31 | 2 / 15 / 15 |
| Student-t(5) | 2 / 0 / 30 | 4 / 1 / 27 | 3 / 17 / 12 |
| Exactly whitened funnel | 2 / 2 / 28 | 4 / 0 / 28 | 3 / 16 / 13 |
| Partially whitened funnel | 2 / 1 / 29 | 3 / 0 / 29 | 1 / 9 / 22 |

All 192 deliberate clipped-tail or wrong-map cases raised a discrepancy;
none falsely passed. The additional moment-matched test above verifies that
this is not solely sensitivity to changed first or second moments.

The positive-control alarm frequencies range from 1/32 to 4/32. Their
pointwise 95% exact binomial intervals range from approximately
`[.00079,.16217]` to `[.03513,.28995]`. None excludes the working .05 rate,
but those intervals are too wide to establish nominal coverage. They are
pointwise calibration intervals, not a simultaneous guarantee across twelve
conditions. Non-rejection is not validation of the proposed nominal rate.

For rho=-.9, unavailable coordinate-mean MCSE accounts for most inconclusive
reports: at least one per-chain lugsail long-run variance was nonpositive.
Squared quantities still expose persistence in the oscillation regression.
Do not clip the variance, replace missing uncertainty by zero, relax the
threshold after seeing this result, or use the missing MCSE to reject a tuning
candidate. The estimator needs separate stationary and stopped-output
calibration before any broad posterior-policy claim.

## Decision and continuation

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept fixture/quantity mechanics; keep statistical calibration open | Map/Jacobian identities, complete quantities and false-reassurance controls pass | No shared numerical or source defect in these checks; reference mutation is detected | Strong antithetic MCSE availability and broad false-alarm intervals | Finish the running full-model price ladder; exercise the changed quantities in fresh public posterior runs; calibrate MCSE/stopping before promotion | Robust tuning, posterior correctness, superiority or default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Wrong map and missing-tail evidence cannot produce a successful reference screen; no change to tuning admission. |
| Statistically supported ranking | None; no methods were compared. |
| Descriptive-only differences | Counts and MCSE availability across targets/rho. Per-cell binomial intervals quantify their sampling uncertainty but do not validate nominal coverage. |
| Default-readiness | Unchanged; both method and calibration remain experimental. |
| Next evidence needed | Fresh public-pipeline quantities, calibrated uncertainty under antithetic/transient/stopped chains, then adequately searched classical comparisons and paired state-space confirmation. |

Known local charges for this work are 75.253492 seconds, within the reserved
1800-second suballocation of D4R4b's 3600-second local allowance. Together
with its previously recorded 624.042147 seconds, known local usage is
699.295639 seconds. Keep the full local reserve until phase settlement; do
not charge this sum twice. The GPU first rung and conditional second rung
retain their existing allocations.

The strongest alternative explanation is that exact stationary reference
chains are substantially easier than adaptively initialized or stopped MCMC.
The weakest evidence is 32 replications per condition and the high antithetic
unavailability. A fresh stopped-chain study with adequate uncertainty could
overturn any optimistic interpretation. These limitations concern diagnostic
evidence; they do not invalidate the corrected model maps or abandon the
planned MAMS tuning repair.
