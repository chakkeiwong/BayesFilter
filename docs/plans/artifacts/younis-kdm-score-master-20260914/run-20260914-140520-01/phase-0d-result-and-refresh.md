# Phase 0D execution and repair

The actual integrated KDM and resampling IWSG consumers now call the shared
analytical LEDH executor with working trace, observation-factor and post-reset
callbacks. All six model parameter directions include the initial law and
parameter-dependent bandwidth. Resampling finite differences hold the sampled
proposal bank fixed, matching the implemented derivative target.

| Evidence | Result |
|---|---|
| Full-direction consumer tests | 2 passed, CPU/XLA, 25.84 s |
| Conditional identities and trace checks | 12 passed, 3 unrelated subspace tests deselected, 30.14 s |
| Known-center/control/blend tests | 4 passed, 4.95 s; includes validation-mutation independence check |
| Actual calibration/validation CLI | 48 rows complete, 31.589 s, combinations-cpu-01 |
| GPU consumers | 3 rows complete, 39.515 s, kdm-consumers-gpu-01; RTX 4080 SUPER, FP32/TF32/XLA, verified memory growth |

The new control is a fixed mixture's translation/log-scale density score,
whose mean is zero. Its mixture pilot is independent; component uniforms are
independent and Gaussian innovations are shared with LEDH initialization.
Calibration fits the coefficient using oracle errors; validation consumes it
unchanged. This preserves the expectation of the baseline, including its bias.
The separate blend minimizes calibration oracle MSE and has no such identity.
The manuscript now derives this construction and the zero-covariance failure
of an independently sampled conditional-zero control.

The CPU fixture uses two calibration and two validation datasets, four
particle replicates each. The fitted control design has rank 3. Relative to
LEDH, validation squared-error differences (candidate minus LEDH) are
integrated KDM -0.1304 (dataset MCSE 0.1087), exact-center CV +0.2512 (0.5599),
and calibrated blend -0.5341 (0.4799). The blend coefficient is 5.2689;
it is an unconstrained quadratic optimum, not a convex weight. These are
descriptive mechanics results. In particular, the CV's higher observed error
must not be hidden, and this fixture does not support a ranking.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue implementation | Consumer/identity/partition checks pass | No nonfinite/missing row in final runs | Small calibration; coupling may be weak | Test covariance providers and corrected proposals | Unbiased model score or superiority |
| Retain CV as unpromoted hypothesis | Exact centering proven | Observed validation worsening prevents promotion | Only two validation datasets | Larger independent calibration and coupling study later | CV improves this filter |

| Inference status | Finding |
|---|---|
| Hard veto screen | Final recorded runs pass numerical validity and coverage |
| Statistically supported ranking | None |
| Descriptive-only differences | All reported error differences and MCSEs |
| Default-readiness | Not established |
| Next evidence needed | Scope-specific tuning, more independent datasets, conditional heuristic comparisons and full cost accounting |

Repairs: the shared executor previously rejected required hooks. Restoring
them exposed two tests relying on Python branching with a Tensor time index
and one bitwise identity assertion differing by one FP64 ULP after tracing.
Graph-native branching and a scale-aware 32-epsilon identity tolerance repair
those tests. Failed logs remain preserved. No density, bias or scientific
acceptance criterion was relaxed.

Post-run red team: fitting a coefficient to eight rows can overfit, especially
when the two biased scores are nearly collinear. Independent validation exposes
this risk but cannot resolve it with two datasets. The strongest alternative
explanation for the apparent blend advantage is calibration noise. More
independent validation can overturn it. The weakest evidence is comparative
quality; algebra and executable call-chain checks are stronger.

Next implementation slice: SGQF's existing standalone score uses a scalar
host loop. Build a shared batched quadrature-moment provider using the actual
repository sparse-grid rule, verify parity against the standalone reference,
and connect prediction, observation conditioning, covariance reset and total
tangents to the same LEDH executor. Signed weights remain integration weights.
Invalid covariance fails closed; no silent ridge or clipping is introduced.
UKF remains the canonical comparator; alternative providers are explicit
research candidates. Then implement corrected lookahead and stochastic FD.

Budget: four GPU framework launches used in the campaign; two remain in the
current three-launch integration slice. The 48+3 new rows plus the earlier
three-row consumer fixture leave at least 66 rows of the 120-row slice. No
large scientific matrix or final claim run has been executed. Detailed commands,
source hashes, streams and device metadata are in each run's state and results.
