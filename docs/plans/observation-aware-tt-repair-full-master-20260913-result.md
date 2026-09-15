> Superseded interpretation (2026-09-14): runs 01–20 do not validate the requested repair or exact filtering corrections. Review found a wrong initial prior, bounded proposal support, correlated deterministic innovations, and absent SGQF pullback/retained TT lifecycle. The numerical records below are preserved as historical mechanics only. Current program and audit: [complete program](observation-aware-tt-repair-complete-program-20260913.md).

# Observation-aware TT repair master: run result (2026-09-13)

The bounded CPU diagnostic completed four recursive steps on the scalar
stochastic-volatility fixture.  The final artifact is
`docs/benchmarks/artifacts/observation_aware_tt_repair_full_master_20260913/run-20/`.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep the lifecycle as a diagnostic candidate | PASS: four TT fits, finite exact corrections, monotone normalized CDFs, ESS above 4, fit residual below 1 | No veto fired | One scalar seed; bounded grid and defensive mixture | Add an independent reference and multi-seed horizon before any promotion | No source-faithful paper-scale KR, posterior correctness, convergence, HMC, or production claim |

The SGQF guide changed its mean by `1.6802460932` between the low and tail
observation checks.  After extending the regular TT design over the complete
`[-8,8]^2` basis domain and adding SGQF likelihood-weighted rows for the
retained ancestors, TT fit residuals were `0.1347`, `0.1321`, `0.1326`, and
`0.1316`; deterministic midpoint holdout residuals were `0.1335`, `0.1284`,
`0.1282`, and `0.1268`.  TT proposal ESS values were `25.44`, `25.48`,
`24.23`, and `22.77` out of 32.  The quadratic CDF inverse reproduced every
requested probability within the recorded tolerance.  The exact
transition/observation correction remained finite at every step, although its
most negative observed log value was `-712.0`, which is retained as a tail
diagnostic rather than hidden.  The prior-proposal SIR comparator had
pre-resampling ESS values `20.54`, `16.62`,
`13.21`, and `18.41`; deterministic resampling fired after step two.  These
differences are descriptive and are not a statistically supported ranking.

The previous un-repaired attempt is preserved as `run-06`; it passed finite
checks but produced boundary-collapsed TT conditionals (ESS approximately one
and implausible means).  That failure triggered the regular fixed-design fit,
the SGQF defensive mixture, and the ESS veto in the current program.

The final holdout check first caught a float32/float64 mismatch in `run-15`;
the failed attempt is preserved in its `failure.txt`.  The subsequent
float64 repair passed in `run-16`; `run-17` repeats the repaired program after
adding explicit guide, fit, and manifest veto checks; `run-18` adds SGQF rows
to the TT regression; `run-20` is the terminal execution and adds the
quadratic CDF-inversion consistency guard.

The strongest alternative explanation is that the defensive SGQF component,
rather than the TT fit, supplies most of the usable proposal.  An independent
dense reference, a grid-resolution calibration, and multiple seeds are needed
to separate that explanation from genuine TT benefit.  The current result
therefore validates the executable call chain and its failure guards only.

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for run-20 |
| Statistically supported ranking | None; one seed |
| Descriptive differences | TT ESS and posterior means are recorded in `result.json` |
| Default readiness | Not assessed |
| Next evidence needed | Dense conditional reference, independent seeds, longer horizon, and source-route call-chain review |
