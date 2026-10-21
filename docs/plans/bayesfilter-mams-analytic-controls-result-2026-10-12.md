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

## Fresh public-pipeline diagnostic

The continuation specified in the plan also finished. From immutable commit
`031f0dfef315825697678fa658b40a1ffff68000`, the actual public tuner,
export/reload and posterior controller ran on Student-t(5), the exact funnel
and the partial funnel with root seed 5800. This explicitly CPU-only graph
reference/debug run used the unchanged short feasibility allocations and
completed in 93.538730 supervisor seconds, including its 92.322754-second
worker. The [terminal audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/analytic-pipeline-01/terminal-audit.json)
checks 174 source hashes, all three targets, all six created/verified members,
export/reload inventories and declared binary quantity identities.

| Target | Verified and retained in candidate set | Short posterior diagnostics passed | Approximate reference screen without alarm |
| --- | ---: | ---: | ---: |
| Student-t(5) | 2 / 2 | 2 / 2 | 1 / 2 |
| Exactly whitened funnel | 2 / 2 | 2 / 2 | 2 / 2 |
| Partially whitened funnel | 2 / 2 | 2 / 2 | 2 / 2 |

The first Student-t member estimated `E[q0^2]=1.334773` against `5/3`.
Its lugsail MCSE was `.104065`; the approximate quantity-family-adjusted
interval was `[1.060222,1.609324]`, which excludes the exact expectation.
The member remains verified for tuning, while
`posterior_reference_screen_passed=false`. This demonstrates why a local
posterior diagnostic pass must remain distinct from reference agreement.
A single alarm after sequential stopping does not establish a biased sampler
or an implementation error. Heavy-tail finite-sample variability, incomplete
exploration and imperfect MCSE/stopping calibration remain alternatives.
Do not retune against that failed heldout reference or delete its outcome.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept new public-pipeline wiring | Six of six verified policies exported/reloaded and assessed with the declared quantities | One Student-t reference alarm vetoes that member's reference-agreement claim; it is not a tuning or research-continuation veto | Short, one-seed CPU diagnostics and uncalibrated stopped intervals | Retain all outcomes; finish GPU pricing and the predeclared MCSE/stopping and full-model confirmation work | GPU qualification, sampler bias, broad posterior validity or ranking |

Inference status remains: no statistically supported ranking; the six rows
are descriptive feasibility evidence, and MAMS stays experimental. Default
readiness is unchanged. The next statistical evidence must address stopped
outputs and antithetic/heavy-tail quantities without selecting only successful
members. The strongest alternative explanation for the alarm is an ordinary
finite-sample or stopping-related coverage failure; fresh predeclared
replications can distinguish persistent error from that explanation.

Known local-control usage is now 168.792222 seconds, and known total D4R4b
local usage is 792.834369 seconds inside the unchanged 3600-second reserve.
This supersedes the earlier cost snapshot above. The changed quantity wiring
now has an actual public-pipeline reference/debug execution; GPU/XLA
confirmation and statistical calibration remain open.

## Exact antithetic batch-means diagnostic

The unavailable antithetic MCSE has a concrete mathematical warning behind it:
lugsail is not always conservative. For a stationary variance-one AR(1)
process with autocovariance `gamma(k)=rho^|k|`, write

```
V_b = b Var(batch_mean)
    = 1 + 2 sum_(k=1)^(b-1) (1-k/b) rho^k
    = (1+rho)/(1-rho) - 2 rho (1-rho^b)/(b (1-rho)^2).
```

For `a=floor(n/b)` complete batches, centering by their common sample mean
gives `E[BM_b] = a/(a-1) (V_b - V_(a*b)/a)`: expand the sum of squared
batch means about that mean, take expectations, and use
`Var(mean of a batches)=V_(a*b)/(a*b)`. This retains inter-batch dependence
and the actual discarded remainder; independent batches are not assumed.
Apply the formula separately at the two batch lengths. At `n=2048`, batch
lengths 45 and 15, and lugsail coefficient one half:

| rho | Exact long-run variance | Expected BM45 | Expected 2 BM45 minus BM15 |
| --- | ---: | ---: | ---: |
| 0 | 1 | 1 | 1 |
| .9 | 19 | 14.9468164 | 20.4928731 |
| -.9 | .05263158 | .06405705 | .03510238 |

The [calculation](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/analytic-controls-01/antithetic-lugsail-expectation.json)
was independently checked against finite autocovariance sums. The batch sizes
are those used in the recorded calibration, not new defaults. For negative
rho the lugsail combination can reverse an upward bias into a downward bias;
finite-sample nonpositivity is a separate possible outcome. This exact
stationary result does not diagnose the particular Student-t alarm or measure
the false-alarm probability after posterior stopping.

Keep MCSE policy calibration open. A proposed repair must compare batch-size
sensitivity and a nonnegative spectral/batch alternative against exact AR(1)
variance, then test transformed heavy tails and the actual stopping rule with
fresh replications. A variance floor or choosing whichever estimator allows
earliest stopping cannot establish coverage. No precision-policy default or
tuning membership changes on the basis of this calculation.
