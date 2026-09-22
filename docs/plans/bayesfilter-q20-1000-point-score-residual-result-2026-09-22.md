# q20 1,000-point latent Gaussian score-residual result

The diagnostic evaluated 1,000 fresh iid `N(0,I4)` latent points against the
frozen `direct-w16-lr0.0005-r0-beta1-u512` map and the exact beta=1 q20/T30
transformed target.  All 1,000 values, scores, status fields, and residuals are
preserved in the worker [points artifact](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00006-gaussian-score-residual-1000/worker/points.json).

## Numerical result

For `g(z) = grad_z log pi_z(z) + z`, the valid-row residual norm had:

| quantity | value |
|---|---:|
| rows | 1,000 / 1,000 |
| minimum | 0.4188 |
| 1% quantile | 1.4462 |
| 5% quantile | 2.7149 |
| 25% quantile | 5.7075 |
| median | 7.6008 |
| mean | 9.6903 |
| RMS | 11.7209 |
| 75% quantile | 11.7753 |
| 95% quantile | 23.1927 |
| 99% quantile | 33.7290 |
| maximum | 46.7441 |

The residual exceeded 0.5 for 99.9% of rows, 1 for 99.8%, 2 for 97.4%, 5
for 82.2%, and 10 for 32.1%.  The residual exceeded the corresponding
Gaussian score norm `||z||` for 98.3% of rows.  The latent norm itself ranged
from 0.1513 to 3.9768, with median 1.8638, so this result is not caused only by
an extremely remote tail sample.

The companion `r(z) = log pi_z(z) + ||z||^2/2` ranged from -82.9191 to
-35.5108, a 47.4083 log-unit span.  Up to the unknown normalizing constant,
that is a target-to-standard-normal density-ratio span of about
`3.88e20`.  The pointwise ratio is not a global divergence estimate, but the
variation is far beyond a small whitening residual.

Every row had finite value and score and passed target-status validation.  The
saved four-point canary values and scores were reproduced exactly (maximum
absolute error 0), which checks the diagnostic's map/target wiring.  The
compiled value/score function ran with one XLA trace on GPU in 50 fixed
20-row batches; the target evaluation took 110.78 seconds and the worker took
132.33 seconds (134.04 seconds including supervision). TensorFlow memory growth was enabled and verified before
initialization.

## Interpretation and decision

The current map is materially inconsistent with the exact standard-normal
score assumption in the tested Gaussian-sized region.  This rejects the
Gaussian-derived step-size approximation for this map and scope.  It does not
show that NeuTra is invalid, that the transformed target is mathematically
wrong, or that no smaller HMC step can work.

| decision | primary criterion | veto status | uncertainty | next justified action | not concluded |
|---|---|---|---|---|---|
| Gaussian score approximation for the frozen map | Strongly failed descriptively: residual median 7.60 and 98.3% larger than `||z||` | No finite/status veto; all rows valid | Points follow the Gaussian proposal law, not posterior draws; score path lacks an independent derivative oracle here | Measure geometry on independent posterior/reference points and repair or retune the map under a new scope | Global posterior non-Gaussianity, score correctness, or NeuTra failure |
| 1,000-point diagnostic harness | All values/scores/statuses finite; saved canary reproduced | Passed engineering screen | Reproduction is not independent score validation | Preserve artifact and keep engineering, sampler, and scientific ledgers separate | Posterior correctness or HMC readiness |

| inference status | finding |
|---|---|
| hard veto screen | No target-health veto; the Gaussian hypothesis is rejected by its declared explanatory residual evidence |
| statistically supported ranking | None; this was not a method comparison or ranking run |
| descriptive-only differences | Residual means, quantiles, exceedance fractions, and density-ratio span |
| default-readiness | No new step size or map default is authorized |
| next evidence needed | Independent posterior/reference points, score parity, local curvature/scale checks, then fresh target-specific HMC tuning |

The failed first attempt (6.01 seconds) stopped before numerical evaluation
because its initial batching assertion incorrectly required 1,000 to be divisible by 32.
The repair changed only the diagnostic harness to fixed batches of 20, preserved
the failed script and receipt, and reran under the same 900-second cumulative
stage allocation.  The successful retry consumed 134.04 seconds, for a total
diagnostic charge of 140.04 seconds.

## Post-run red-team

The strongest alternative explanation is a target-score or transport-pullback
implementation error rather than a poorly learned map.  The saved canary
reproduction verifies that this run uses the same value/score path as the prior
diagnostic, but it is not an independent derivative check.  A score-parity test
against an eligible independent derivative/reference path would address that
alternative.  The other limitation is support: iid Gaussian points need not be
posterior draws, so a future posterior/reference-point audit could reduce the
scope of the mismatch.  A result showing small residuals at independently
validated posterior points, or an independent score check that changes the
residuals, would overturn the current map-specific conclusion.

The run therefore validated the harness and artifact path, weakened the
Gaussian-whitening hypothesis for this frozen export, and left the NeuTra
research direction open for map repair and target-specific tuning.

## Provenance and budget

- Plan: `docs/plans/bayesfilter-q20-1000-point-score-residual-plan-2026-09-22.md`
- Successful [worker manifest](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00006-gaussian-score-residual-1000/worker/manifest.json)
- Successful [summary](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00006-gaussian-score-residual-1000/worker/result.json)
- Successful [budget receipt](artifacts/q20-1000-point-score-residual-2026-09-22/budget-receipt.json)
- Preserved [failed setup attempt](artifacts/q20-1000-point-score-residual-2026-09-22/setup-failure-01/budget-receipt.json)

After the run, the campaign has 155,794.7589 seconds (43.2763 hours) of
campaign allowance and 1,004.9098 seconds (16.7485 minutes) of diagnostic
allowance remaining.  The master remains `ESTIMATION_BUDGET_PAUSED`; this
diagnostic did not change its production or estimation state.
