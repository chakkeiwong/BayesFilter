# Range–bearing observation likelihood needs a wrapped-density reference

The existing range–bearing density callback is wrong relative to its
simulator's wrapped-angle law. It evaluates only the nearest Gaussian alias.
At bearing standard deviation pi, its integral over the circle is
0.68268949334, while an independently summed wrapped density integrates to
1.00000000000. The missing term is not a parameter-independent constant that
can be dropped from noise-scale inference.

The [pre-run audit](surrogate-hmc-range-bearing-scope-audit-2026-09-22.md)
derives the distinction from the wrapping map and cites the simulator and
density callback. The [executed diagnostic](../../tests/test_surrogate_hmc_range_bearing_scope.py)
calls that callback and integrates its angular density after dividing out the
same independent radial Gaussian factor. With4096 and8192 midpoint angles,
the callback masses are0.68268949694 and0.68268949334, agreeing with the
analytic value erf(1/sqrt(2))=0.68268949214. Increasing the alias count from
4 to8 on each side changes the wrapped density by at most2.8e-17.

At the original fixture's standard deviation.04, both integrals equal one
to floating-point precision and their largest density difference is5.4e-15.
Thus the audit does not show a material error in that fixed narrow-noise
fixture's posterior. It shows that this callback cannot be assumed to be the
normalized model likelihood across the proposed noise-scale parameter space.
No prior-tail probability, posterior effect or performance ranking was tested.

Two tests passed under CPU/XLA with GPU devices intentionally hidden, costing
6.674001 CPU-worker seconds and5.915082 wall seconds. No GPU time was used.
The [manifest](artifacts/surrogate-hmc-all-models-20260922-range-bearing-scope-audit-01/run-manifest.json),
source snapshot, [complete log](artifacts/surrogate-hmc-all-models-20260922-range-bearing-scope-audit-01/pytest.log)
and accounting preserve the command and numerical results.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| M13 reference remains unqualified | Callback fails unit mass for the simulator's wrapped law | Both numerical refinements pass | Posterior relevance of broad-noise regimes is unmeasured | Qualify a normalized wrapped likelihood over the declared parameter scope before reference inference | No rejection of surrogate HMC or fixed narrow-noise results |

| Inference status | Finding |
|---|---|
| Hard veto screen | Cannot treat this single-alias callback as the exact wrapped model likelihood |
| Statistically supported ranking | None; deterministic law check |
| Descriptive-only differences | Density differences and execution time are diagnostic |
| Default-readiness | No range–bearing posterior admission |
| Next evidence needed | Same-law normalized endpoint/reference, full DCTR consumer, scoped tuning and posterior comparisons |

The proposal field may remain imperfect and need not differentiate the
acceptance endpoint. The finding concerns which observation density is called
the model likelihood. If a single-alias finite target is retained as an
explicit approximation, it needs its own D reference and a separate D/R
comparison; it cannot silently replace the simulator's law.

Red team: broad angular noise may have negligible posterior probability for a
particular dataset. A qualified tail bound could establish a restricted
approximation error for that dataset; this diagnostic supplies no such bound.
Renormalizing the nearest term alone defines a different truncated angular
law and does not establish equality with the wrapped Gaussian law.
