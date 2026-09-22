# Independent d40 confirmation: both accuracy screens pass

2026-09-20. The frozen optional R iAPF completed all 64 planned repetitions.
Both datasets pass the predeclared accuracy, fit, tail and conditional
heuristic screens. The earlier upward discrepancy on the selected d40
dataset did not recur with fresh method randomness. This completes the
[confirmation plan](../../iapf-r-d40-confirmation-2026-09-20.md).

The target is the marginal likelihood of the paper's linear-Gaussian model,
with dimension 40 and 100 observations. The returned quantity is the fresh
final APF likelihood estimate, divided by the exact Kalman likelihood on the
same observations. The candidate uses the optional log-quadratic guide fit
and positive-floor power 8. That fit differs from the paper's equation 15.

## Accuracy and numerical validity

Intervals are conditional on the recorded observations: 2,000 percentile
bootstrap resamples of each batch's 32 likelihood ratios, with seed
`data_seed + 900`. The criterion is containment in [0.9, 1.1], evaluated
separately. The preceding batch is shown for context and was not pooled with
either confirmation batch.

| Dataset and method randomness | Mean likelihood ratio | Bootstrap 95% interval | Declared screen |
| --- | ---: | --- | --- |
| Previous selected d40 batch, seed 80000040, repetitions 701:732 | 1.041181 | [1.005136, 1.078378] | Passed; motivated confirmation |
| Same observations, fresh repetitions 801:832 | 0.967538 | [0.932404, 1.002977] | Pass |
| Fresh observations, seed 82000040, repetitions 901:932 | 1.022861 | [0.982292, 1.061802] | Pass |

Both new intervals include one. That is consistent with finite-sample
fluctuation as an explanation of the earlier discrepancy; it neither proves
unbiasedness nor bounds unseen rare tails. No persistent upward discrepancy
was established. The two new datasets were evaluated separately throughout.

| Numerical diagnostic | Same observations | Fresh observations | Role and finding |
| --- | ---: | ---: | --- |
| Completed iAPF repetitions | 32/32 | 32/32 | Completeness check passes |
| Valid QR fits | 19,300/19,300 | 19,300/19,300 | Fit validity veto absent |
| Maximum design condition | 34.7923 | 35.1697 | Numerical explanation |
| Maximum normal-equation gradient | 7.39e-16 | 6.25e-16 | Fit check passes |
| Gaussian-limit second-moment checks | 3,200/3,200 | 3,200/3,200 | Tail screen passes |
| Minimum relative tail margin | 0.320317 | 0.316396 | Positive; no tail veto |
| Final particle counts | 1,000 or 2,000 | 1,000 or 2,000 | Controller explanation |
| Final controller iterations | 7 or 8 | 7 or 8 | No controller-cap failure |
| Maximum recorded floor mixture probability | 7.90e-22 | 1.11e-20 | No observed prior-component takeover |
| Minimum final-pass ESS | 417.299 | 371.096 | Explanatory only |

Large relative fit residuals remain possible: maxima were 0.9136 and 0.9904.
Passing a QR numerical check does not mean the diagonal guide equals the
optimal guide. The exact importance correction, rather than an exact guide
fit, is what preserves the likelihood target in ideal arithmetic.

## Conditional heuristic checks

Entries are observed mean squared errors of cumulative log-likelihoods
against Kalman, separated using the preregistered Kalman innovation threshold.
The cheap alternatives are bootstrap filtering with 10,000 particles,
fully-adapted filtering with 5,000, and prior-proposal SIS with 10,000.

| Dataset | Observation situation | iAPF | Bootstrap PF | Fully-adapted PF | SIS |
| --- | --- | ---: | ---: | ---: | ---: |
| Same observations | Ordinary | 0.025280 | 85,310.27 | 0.181291 | 6,373,831.86 |
| Same observations | Large innovation | 0.021652 | 121,718.62 | 0.224414 | 9,231,704.87 |
| Fresh observations | Ordinary | 0.031346 | 87,216.60 | 0.140584 | 6,449,568.20 |
| Fresh observations | Large innovation | 0.018760 | 83,501.86 | 0.144301 | 6,114,318.28 |

No observed heuristic veto fired. These differences are descriptive, with
unequal particle counts and costs; no statistical or efficiency ranking is
claimed. SIS likelihood ratios underflow to zero, but its finite log ratios
and all repetitions are preserved and enter this log-scale diagnostic.

## Mathematical and implementation audit

The [derivation and source trace](likelihood-audit.md) checks the proposal,
positive-floor mixture, importance weights, both adaptive-resampling branches
and the controller's fresh final evaluation. For frozen guides and particle
count, cancellation of the twist and transition normalizers gives the
original likelihood. Both resampling branches preserve the same conditional
expectation of the unnormalized particle estimate.

If H denotes the learned guides, chosen particle count and stopping history,
the independent final evaluation therefore satisfies
`E[Z_hat_final | H] = Z` for the idealized algorithm. Reusing the estimate that
triggered stopping would not have this justification. The strengthened test
checks the value returned after stopping, as well as the extra filter call;
it rejects a mutation that draws a fresh value but returns the stopping value.
All 11 focused pytest cases pass. The numerical R implementation was unchanged.

This derivation is about likelihoods, not unbiased log likelihoods. It does
not establish exact unbiasedness of finite-precision R or certify arbitrary
models. The original-paper replication gap also remains: equation 15, the
author's optimizer and constraints, floor definition, early doubling
convention, and original code/data provenance require reconciliation. The
pinned public R comparator uses a different scaled objective and has not been
established as author code. Two attempted external provenance lookups failed
with HTTP 502; no stronger provenance conclusion is supported.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this confirmation stage; retain the optional R reference for further validation | Both conditional intervals wholly inside [0.9, 1.1] | No fit, tail, completeness, source/data or conditional heuristic veto | Two datasets and 32 draws per dataset do not resolve rare tails or general validity | Reconcile the paper's fitting procedure before a paper-replication label or larger study | Author-paper replication, default promotion, or TensorFlow/LEDH/KDM correctness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No hard veto in either complete confirmation batch |
| Statistically supported ranking | None; the intervals support only the declared conditional accuracy screen |
| Descriptive-only differences | Heuristic errors, runtimes, ESS, fit residuals and empirical tails |
| Default-readiness | Not established; default objective and floor remain unchanged |
| Next evidence needed | Source-grounded fitting/solver reconciliation, then a separately planned replication ladder; eventual executable TensorFlow parity |

Post-run skeptical review: PASS for closing this bounded confirmation. The
strongest alternative explanation is that 32 repetitions miss consequential
rare weights, particularly with a very small positive floor. The positive
Gaussian-limit margins are useful diagnostics, not a guarantee of bootstrap
coverage. Broader fresh-data evidence or a material failure of the checked
likelihood identity would overturn the current interpretation. The weakest
evidence is generalization beyond this linear-Gaussian setting. No threshold,
fit setting, seed, or repetition was selected after observing these results.

## Reproducibility and compute

Branch `surrogate-hmc`, commit
`6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`; captured source closures preserve
the relevant uncommitted files. R 4.1.2 ran with GPU deliberately hidden,
`CUDA_VISIBLE_DEVICES=-1`, and one BLAS/OpenMP thread. This is an explicit
independent CPU reference, not the repository's GPU/XLA implementation.

The two filtering attempts used 501.244841 and 496.449970 worker seconds.
Tail diagnostics and reporting used 22.571156 seconds. Total:
**1,020.265967 / 1,500 summed worker seconds**, leaving **479.734033** unused.
Both planned launches succeeded; no retry or additional scientific launch was
needed. Focused test logs report 3.45 and 4.40 seconds inside the separate
120-second mechanics allowance. The stage is closed; unused time is not an
authorization to add a different scientific experiment.

Exact commands, environment, seeds, data/source hashes and wall times are in
[attempt 1](attempt01-same-data/manifest.json),
[attempt 2](attempt02-fresh-data/manifest.json), their diagnostic manifests,
and the [report manifest](report-manifest.json). Complete CSV/RDS results and
captured source trees remain in each attempt directory. The
[machine-readable summary](summary.json) records both decisions separately.
The [audit manifest](audit-manifest.json) preserves the strengthened tests
added after the filter source snapshots, with their logs and source hashes.

The next master-program stage is a bounded source and fitting reconciliation:
inspect official supplementary/author materials for the actual objective,
solver and constraints; distinguish explicit paper instructions from local
reconstruction; then specify the smallest discriminating fitting check. If
the missing choices cannot be recovered, report them as unknown and retain
the successful log-quadratic method under its own identity. Scaling that
method alone cannot establish equation-15 conformance.
