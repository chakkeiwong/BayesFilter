# M13 surrogate-HMC: corrected cumulative posterior result

The 9,000 fixed-kernel draws formerly labeled verification
are posterior samples. Prefix status: M13_POSTERIOR_SAMPLING_COMPLETE.
There is no separate validation sample requirement. Warmup is excluded.

The old master generated another 3,750 draws per
chain. All 12,750 saved post-warmup draws are now included.
Complete-history status: M13_POSTERIOR_MIXING_INCONCLUSIVE.
An earlier passing prefix cannot override adverse evidence from later samples.

Parameter order below is log process scale / log observation scale; four chains.

| Samples | Draws/chain | R-hat | Bulk ESS | Tail ESS | Posterior mean | Mean MCSE |
|---|---:|---|---|---|---|---|
| Earlier stopping point | 9,000 | 1.0074 / 1.0055 | 512 / 1076 | 324 / 1956 | 0.0765 / -0.1410 | 0.0112 / 0.0052 |
| Complete saved history | 12,750 | 1.0542 / 1.0088 | 77 / 321 | 30 / 483 | 0.0303 / -0.1299 | 0.0418 / 0.0097 |

Acceptance by chain: earlier 0.8058 / 0.8104 / 0.8174 / 0.7838; complete
0.8099 / 0.6812 / 0.7784 / 0.8057. Acceptance is descriptive.

All archived checksums, fixed-step and binding identities, seed histories,
exact saved state handoffs and health checks passed. No force fallback or
divergence was found. CPU reconstruction allows rounding only in the affine
map; archived handoffs remain exact. The earlier R-hat was reproduced.
No new HMC transitions were generated.

| Decision | Primary criterion | Veto checks | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Earlier prefix: M13_POSTERIOR_SAMPLING_COMPLETE | R-hat 1.0074 / 1.0055 | Health passed | Later history may reveal slow exploration | Preserve the earlier stopping point | Permanent convergence |
| Complete archive: M13_POSTERIOR_MIXING_INCONCLUSIVE | R-hat 1.0542 / 1.0088 | Health passed | Slow exploration and MCSE | Preserve every sample; follow the refreshed plan | Failure of surrogate HMC as a method |

| Inference status | Interpretation |
|---|---|
| Hard veto screen | No numerical, target-status, seed or state-continuity veto found |
| Statistically supported ranking | None; this reclassification compares no methods |
| Descriptive-only differences | ESS, MCSE, means and acceptance; low ESS is a precision warning |
| Default-readiness | Not evaluated; the all-model campaign remains incomplete |
| Next evidence needed | Stable cumulative mixing; reference agreement is a separate scientific assessment, not an independent validation chain |

The target remains the frozen importance-likelihood posterior for
M13/data1704/T20. The samples use that endpoint with a frozen analytical
proposal field. This accounting repair does not establish equality with the
original exact model likelihood. The old equivalence calibration cannot be
applied to arbitrary pooled sample counts and has not been relabeled passed.

Post-run review: a favorable early diagnostic can miss slow exploration. Later
deterioration is counterevidence to durable convergence. A longer healthy
cumulative history could resolve this; it must not discard the adverse suffix.

The old deadline has expired. The total cap is 20,000
per chain; 12,750 have been used. The runtime repair and archive
reclassification are complete. Any later continuation belongs to this same
master and starts from the saved chains; it requires a new time allocation.
