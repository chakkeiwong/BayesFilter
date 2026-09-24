# Merged-source pilots and fresh fixed-count estimator confirmation

Update: the subsequent 272-fit public confirmation is complete and audited in
[the M25 result](bayesfilter-hmc-gap-closure-result-2026-09-22.md). The decisions
below record this earlier pilot/estimator stage. They are not outstanding launch
instructions. General posterior calibration remains open.

The Gaussian and beta-binomial pilots completed on committed source
`f9c86f41ac44efc5e6a5a5220633073fce803431`, package identity
`13917d3625fa5a09dfd3bf1ab9d2894618ed7bc4259ca7603f16c766f6213d66`.
All 23 Gaussian and 17 beta-binomial verified candidates remain retained.
The predeclared first L=3 member in each fit supplied posterior draws and an
independent fixed-count comparator. Both complete public-path inventories
passed; a single fit is execution/cost evidence only. Worker times were
188.58306595892645 and 227.3817994639976 seconds, within the per-fit
confirmation reservations. Pilot observations are excluded from confirmation.

The fresh estimator study completed all 800 predeclared fixed-count arms,
400 per exact AR(1) regime. It used four chains, 10000 discarded and 10000
retained transitions per chain, root seed 2026092244, and the same array for
the three predeclared estimators in each replication. Exact Gaussian oracle
coverage passed both Bonferroni screens. Every result, array, hash, command,
source, environment and seed is preserved under
`artifacts/hmc-repair-master-2026-09-16/m21-r2/estimator-confirmation-cpu-r1`.
This is CPU reference work with GPUs intentionally hidden, not GPU evidence.

| Regime | Estimator | Available / 400 | Covered / 400 | Pointwise 95% interval | Declared screen |
| --- | --- | ---: | ---: | --- | --- |
| Stationary rho 0.98 | Lugsail b=100 | 400 | 367 | 0.8861--0.9425 | Not passed |
| Stationary rho 0.98 | Lugsail b=500 | 396 | 371 | 0.8975--0.9509 | Not passed |
| Stationary rho 0.98 | TFP autocorrelation | 400 | 374 | 0.9062--0.9571 | Passed |
| Dispersed rho 0.995 | Lugsail b=100 | 400 | 296 | 0.6941--0.7823 | Not passed |
| Dispersed rho 0.995 | Lugsail b=500 | 400 | 367 | 0.8861--0.9425 | Not passed |
| Dispersed rho 0.995 | TFP autocorrelation | 400 | 373 | 0.9033--0.9550 | Passed |

The screen requires all estimates available and a pointwise coverage lower
bound >=0.90. It is not a test proving exact 95% coverage. Unavailable
estimates remain in the denominator. The larger lugsail batch reduces bias
in the persistent case but introduces unavailable estimates in four of the
rho 0.98 replications; this is the expected variance/bias tradeoff, not evidence
that nonpositive estimates should be clipped to a passing value.

The existing autocorrelation option passes the fixed-count eligibility screen
in these two regimes. No method ranking was predeclared or established. Its
random-stop behavior, heavy-tailed cases, unseen modes and general HMC
performance remain unproved. The lugsail formula remains algebraically checked,
while its square-root bandwidth is inadequate in the strongly persistent
case. The result calls for target-specific bandwidth/estimator checks and
separate stopping validation, not an automatic new default.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute merged public confirmation | Complete pilot inventories and cost checks pass | No invalid pilot artifact | One fit per control | Run the fixed 272-fit inventory with independent fixed arms | Whole-pipeline coverage |
| Preserve optional autocorrelation as eligible here | Fixed-count screen passes both exact regimes | Both exact oracles pass | Only two laws and deterministic counts | Plan fresh random-stop validation before any default change | Anytime coverage or superiority |
| Keep lugsail bandwidth gap open | Neither tested bandwidth passes both regimes | Four unavailable b=500 estimates remain failures of precision availability | Bias/variance tradeoff and target dependence | Diagnose bandwidth with separate evidence; retain original accuracy targets | Universal lugsail adequacy |
| Continue null-size study | M22 pilot valid and committed source available | No shared invalidity | Rejection-rate uncertainty | Execute 512 experiments per arm | Subtle complete-fit power |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No broken exact oracle or corrupted array; unavailable precision is explicit |
| Statistically supported ranking | None |
| Descriptive-only differences | Estimated/exact SE ratios, timings and between-estimator differences |
| Default readiness | No default changed; fixed-count eligibility is narrower |
| Next evidence needed | Public confirmation, random-stop validation of nominated estimator, GPU replication and target-specific cases |

The estimator worker consumed 207.83763966301922 seconds. The two public pilot
workers consumed 415.96486542292405 seconds. `m21-r2/reconcile.py` combines
completed worker receipts without double-counting coordinators; live queue
reservations are reported separately. The strongest alternative explanation
for an apparently favorable estimator is favorable finite-sample behavior in
these two Gaussian regimes. A fresh random-stop or non-Gaussian failure would
overturn a broader adequacy claim; no such claim is made here.
