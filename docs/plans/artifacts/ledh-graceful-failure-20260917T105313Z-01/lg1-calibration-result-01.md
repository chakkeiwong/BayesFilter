> Stage record preserved. Current result: all three LG1 attempts failed
> calibration; exact-block transport was implemented and checked before the
> N4096 continuation. See the [terminal result](../../ledh-graceful-failure-execution-result-2026-09-18.md)
> and `decision.json` for final scope decisions.

# LG1 first calibration: approximation screens failed

The full shared finite program has a checked analytical derivative, but the
tested N=256 and N=1024 settings fail the proposed Kalman approximation budgets.
This rejects these calibration candidates. It does not reject LEDH or establish
a posterior error, and it does not authorize loosening held-out thresholds.

The target is stationary LG1, T=10, phi uniform on (-0.95,0.95), Q=1, R=0.5.
The independent comparator is the exact scalar Kalman recursion at the same
represented observations, parameters and stationary initialization. Only
calibration replicates 0 and 2 were inspected in the control curves. All four
calibration replicates were inspected in the preceding diagnostic. Validation
and claim observations remain unread.

Every row below is descriptive: worst error across the frozen three-parameter
design and two datasets, not uncertainty-supported rankings. All candidates
retained the canonical reset, GenUT correction, pairwise controls and dual caps.

| Candidate | N | Worst whitened mean | Worst whitened covariance | Worst centered log-ratio |
|---|---:|---:|---:|---:|
| warm | 256 | 0.604707 | 0.578749 | 1.674500 |
| flow-20 | 256 | 0.629212 | 0.638979 | 2.098161 |
| flow-40 | 256 | 0.632664 | 0.668536 | 2.349292 |
| flow-80 | 256 | 0.631733 | 0.682829 | 2.484776 |
| ot-32 | 256 | 0.604706 | 0.578749 | 1.674504 |
| ot-128 | 256 | 0.604706 | 0.578749 | 1.674504 |
| epsilon-0.5 | 256 | 0.626701 | 0.578749 | 1.743047 |
| epsilon-1.0 | 256 | 0.617864 | 0.578749 | 1.708226 |
| epsilon-4.0 | 256 | 0.610432 | 0.578749 | 1.659192 |
| ridge-1e-06 | 256 | 0.604705 | 0.578749 | 1.674500 |
| ridge-0.0001 | 256 | 0.604708 | 0.578749 | 1.674489 |
| correction-4 | 256 | 0.608400 | 0.578749 | 1.680142 |
| lm-0.01 | 256 | 0.604497 | 0.578749 | 1.674237 |
| trust-0.5 | 256 | 0.604861 | 0.578749 | 1.674302 |
| N1024-warm | 1024 | 0.621522 | 0.506228 | 1.356063 |
| refined | 256 | 0.638642 | 0.682829 | 2.487004 |
| N1024-refined | 1024 | 0.663920 | 0.635533 | 2.036719 |

Budgets are 0.05 for each moment error and 0.02 for centered log-ratio.
All value and force masks in this sweep were valid. Finite values are therefore
insufficient evidence of approximation quality. Increasing flow steps did not
resolve the discrepancy; this observation motivates localization, not a claim
that one setting is statistically better.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Reject tested calibration candidates | Kalman screens failed | Approximation promotion veto; no numerical continuation veto | Finite particle resolution versus implementation or diagnostic mismatch | Complete exact-block N=4096 capability, derivative check and planned calibration | LEDH research direction rejected |

| Inference status | Finding |
|---|---|
| Hard veto screen | No nonfinite candidate in this sweep; all fail approximation screens |
| Statistically supported ranking | None |
| Descriptive-only differences | Every continuous comparison in the table |
| Default-readiness | Not established |
| Next evidence needed | Larger-particle localization, independent moment checks, disjoint validation, issued tuning identity, posterior reference and actual HMC qualification |

The strongest alternative explanation is an implementation or diagnostic error
that survives self-consistent differentiation. A correct finite-program derivative
does not refute that explanation. Per-step ESS and before/after-reset moments
are now preserved to distinguish it from sampling error. The weakest evidence
is tail approximation on only two calibration datasets in the control sweep.

Evidence: `lg1-calibration-diagnostic-01/diagnostic.json`,
`lg1-control-curves-01/curves.json`, their saved target manifests, exact commands
and process logs. Full resource charges, including failures, are in `campaign.json`.

Additional capability findings: the old shared reset had no policy-compliant
multi-block computation. The repair retains the same Sinkhorn and analytical
quotient equations, with policy-selected blocks; the full transport matrices
remain materialized for the existing diagnostic return, so total memory is
still quadratic. It must not be described as a fully streaming memory route.
The TensorFlow public HMC tuner currently explicitly limits its configuration
to diagnostic evidence and rejects XLA. The legacy admitting tuner has runtime
NumPy use. Neither can yet issue an eligible production-path artifact for this
campaign without a bounded, tested migration. Bare transition tests cannot fill
that gap.

## N4096 continuation of attempt 1

All runs used the canonical features and exact policy K=2048. All value and force statuses were valid. Calibration replicates 0 and 2 only.

| Schedule | Replicate | Worst mean error | Worst covariance error | Centered log-ratio error | Screen |
|---|---:|---:|---:|---:|---|
| N4096-warm | 0 | 0.125327 | 0.241315 | 0.165512 | fail |
| N4096-warm | 2 | 0.465984 | 0.229133 | 1.06846 | fail |
| N4096-flow-3 | 0 | 0.0881405 | 0.083524 | 0.0707318 | fail |
| N4096-flow-3 | 2 | 0.449867 | 0.182502 | 0.614379 | fail |
| N4096-flow-1 | 0 | 0.0998322 | 0.113432 | 0.0961892 | fail |
| N4096-flow-1 | 2 | 0.697581 | 0.375296 | 0.506961 | fail |
| N4096-flow-40 | 0 | 0.172475 | 0.340983 | 0.265823 | fail |
| N4096-flow-40 | 2 | 0.519873 | 0.34961 | 1.43436 | fail |

All four schedules fail the frozen 0.05 moment or 0.02 log-ratio budgets. These are descriptive calibration differences, with no statistically supported ranking. Reset means and covariances match weighted pre-reset moments within rounding; this alone does not establish the correctness of higher moments or the next-step proposal. Diagnose the finite flow/weight program independently before attributing the discrepancy to sampling. No posterior or default promotion follows.
