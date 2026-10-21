# Precision of the bounded surrogate-HMC inference schedule

The thirteen-function, four-comparison schedule is poorly sized for establishing
median-CDF equivalence at 2,000 retained draws per chain. Even under ideal IID
sampling and zero reference error, its nominal interval radius is .02146,
larger than the .02 equivalence margin. This exposes a design limitation of
the bounded pilots; it is not evidence against surrogate HMC.

The [prospective calculation](surrogate-hmc-inference-precision-audit-2026-09-22.md)
uses the actual Student-t critical value 3.83913389080146 with 36 degrees of
freedom and the frozen multiplicity allocation. Reconstructing the critical
value from an already closed M10 interval agrees within 1e-12. No new posterior
look was computed. The [report and exact script](artifacts/surrogate-hmc-all-models-20260922-precision-audit-01/report.json)
preserve the calculation; it cost 4.422476 CPU seconds and no GPU time.

| Draws per chain, four chains | Ideal median-CDF radius, 13 functions/four families |
|---:|---:|
| 1,000 | .03035 |
| 2,000 | .02146 |
| 4,000 | .01518 |
| 8,000 | .01073 |

At least 2,303 IID draws per chain bring this nominal radius just below .02;
9,212 bring it below .01. Neither count is a power guarantee. Actual estimates
are displaced from the reference, reference allowances are positive, and
dependence changes variance. Estimated batch variances also fluctuate, so the
calculation does not prove that a 2,000-draw result can never pass. The older
six-function/one-family design has radius .01711 at 2,000 and must be assessed
separately. A small scalar result does not justify transferring its schedule
to a larger family of functions and comparisons.

The active SV20 run retains its two declared looks. LG2 endpoint qualification
may still measure complete transition and diagnostic costs, but its 2,000-draw
forecast is the cost of another bounded pilot, not evidence of adequate power.
Before another retained inference scope is funded, justify its uncertainty
policy and precision prospectively. The earlier binary-function calibration
remains limited to its tested Gaussian regimes; this calculation does not
promote that diagnostic to a runtime default.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Treat current schedules as bounded pilots | Ideal IID radius exceeds the median margin for 13/four | No new veto or changed active look | Joint power, dependence, reference allowance | Require prospective precision justification and complete cost for new inference | No sampler failure or general sample-size recommendation |

| Inference status | Finding |
|---|---|
| Hard veto screen | No additional candidate veto from this calculation |
| Statistically supported ranking | None; no stochastic methods compared |
| Descriptive-only differences | None; table is a deterministic idealized calculation |
| Default-readiness | No diagnostic or runtime default changed |
| Next evidence needed | Scope-specific diagnostic calibration and a funded precision design |

Post-run red team: antithetic sampling can yield greater precision than this
IID calculation, while positive dependence and reference error reduce it.
The weakest part is the lack of a joint power calculation. That prevents
declaring a replacement sample count adequate, but does not justify treating
the inherited 2,000-draw schedule as adequately powered.
