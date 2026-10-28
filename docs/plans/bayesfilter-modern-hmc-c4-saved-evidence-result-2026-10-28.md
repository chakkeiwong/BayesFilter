# C4 saved-evidence diagnosis: arithmetic passes; precision limits remain

28 October 2026. This completes the saved-scalar audit in the
[C4 plan](bayesfilter-modern-hmc-c4-saved-precision-audit-2026-10-28.md), not
the precision-reliability repair or the modern-HMC master. No new trajectories
were generated. Compact results, commands, source identities and measured costs
are in [the evidence directory](../validation/modern-hmc-20261028-c4/README.md).

## Findings

The precision audit checked all 1,152 record identities and independently
recomputed all 288 generated validation endpoints. Stored means and MCSE agree
with standard-library arithmetic and the original frozen TensorFlow/XLA
reduction. All 96 refused validation slots remain in the denominator. The first
reader failed because a warm-up refusal correctly has no precision report; the
repaired reader preserves 396 such assessments and rejects other unexplained
missing reports. The failed attempt remains preserved.

The two antithetic failures are correct refusals under the original precision
criterion. At replications 1 and 19, squared-coordinate MCSE is respectively
0.0311469491 and 0.0321495992 against tolerance 0.0282842712. Estimated variances
are 3.1209 and 3.3251 times the fixed stationary reference variance. The largest
chain contributes about 60.8% and 62.2% of the variance numerator. These are
explanations of small-chain variance uncertainty; the stationary reference is
not the variance conditional on the actual warm-up and delivery decisions and
cannot override them. Strong positive dependence also remains unresolved:
at 160,000 draws the four-chain calibration has 23 precision and four warm-up
failures out of 64 replications, and all 32 validation plans are refused.

The tail audit checked 1,792 records and 5,376 profiles across seven cases,
three profiles and all eight original interval arms. Means, widths, variance
ratios and coverage classifications agree with preserved outcomes. All 240
original screens are unchanged. For the lognormal squared quantity, replacing
estimated variance with the stored exact stationary variance while keeping the
same Student multiplier gives the following descriptive counterfactual:

| Profile | Original misses | Exact-variance misses | Original misses rescued | New misses introduced |
|---|---:|---:|---:|---:|
| 4 chains, 10,000 draws | 27 | 3 | 27 | 3 |
| 4 chains, 160,000 draws | 16 | 3 | 16 | 3 |
| 64 chains, 10,000 draws | 16 | 8 | 15 | 7 |

The short four-chain intervals missing below the truth have median
estimated/exact variance ratio 0.04727. Underestimated variance and inadequate
representation of the upper tail therefore warrant investigation. The table
does not validate a variance multiplier, establish a Student law for nonlinear
means, or rank estimators. Exact variance is unavailable to an ordinary consumer.
All 1,792 original raw-record paths are now absent; this audit checks surviving
scalar evidence and cannot replay raw trajectories or invent per-chain records.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close arithmetic diagnosis | All generated endpoints agree | No surviving source/arithmetic mismatch | Raw-generation and conditional-law questions remain | Preserve original refusals; proceed to mechanism-specific work | Precision reliability |
| Keep antithetic refusals | Two estimated-MCSE thresholds fail | Original precision veto retained | Variance from four chain means is uncertain | Qualify any future planning repair on fresh evidence | Permission to pass failed candidates |
| Keep nonlinear-tail question open | Scalar interval audit passes | Original coverage failures retained | Unseen-tail risk and nonnormal finite-sample means | Specify a tail-aware qualification contract before new sampling | A calibrated new estimator |
| Continue C2 independently | Local compatibility already passes | Official source binding unresolved | Consumer provenance and clean-root deployment | Implement scoped binding with negative tests | Public admission or posterior readiness |

| Inference class | Status |
|---|---|
| Hard veto screen | Original precision, warm-up and coverage failures preserved; failed reader repaired |
| Statistically supported ranking | None; no new ranking analysis was specified |
| Descriptive-only differences | Variance ratios, chain contributions and counterfactual miss counts |
| Default-readiness | Not established; numerical policies unchanged |
| Next evidence needed | Source-bound consumer integration; separately specified fresh tail/precision validation |

## Verification, cost and red-team review

Twenty focused tests pass: independent covariance sums, variance arithmetic,
translation, failed precision preservation, warm-up refusals, record corruption
and unchanged-multiplier counterfactuals. The original reduction is identified
by SHA-256 26167994b0aa06a26c277a5a914d39bb8b14c26b2e33deb5ef8fc7b786d1abeb.
The audit used deliberately CPU-only reference execution, not GPU qualification.

The failed reader, passing precision audit and tail audit cost 2.372230816,
2.790352042 and 0.858639302 inclusive seconds. Prior focused-test tool durations
add 0.940977597 seconds. A final 20-test verification adds 0.422075025 seconds. These measured charges total 7.384274782 seconds, leaving
at most 235743.412140 seconds before unmetered preparation, coding and packaging.
This is an upper bound, not an exact fully settled balance. SIR's separate
7363.064379416 seconds are unchanged. No numerical worker remains active.

The strongest alternative explanation is selection or raw-generation behavior
that scalar reductions cannot reveal. An intact raw replay disagreeing with the
saved statistics would overturn the arithmetic conclusion for that trajectory.
The weakest evidence is finite-replication, outcome-conditioned tail explanation.
Neither the candidate refusals nor these limits reject the research direction.
The next concrete engineering step is C2 source/provenance binding; C3 posterior
qualification, C4 estimator reliability, C5 independent-fit SIR and C6/N8 release
reconciliation remain open. Do not rerun completed count ladders or choose a
variance multiplier from this confirmation data.
