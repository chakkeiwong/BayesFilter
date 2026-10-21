# Frozen randomized-HMC terminal result — 16 October 2026

The full NL-PP-SGQF randomized-HMC cohort completed at 17:13 China time.
All eight declared roots qualified, were exported, and received posterior
assessment. Five passed both posterior and reference screens. No repair
descendants were created and no verified members remain unassessed. Every
qualified member remains in the tuning set, including the three posterior
delivery failures. There were no numerical-health hard vetoes.

The immutable source is `full-model-source-e68cdf1c4`, commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`. The frozen design is
`designs-02/randomized_hmc.json`, seed 71410, four chains, full SGQF data and
geometry, the original lugsail precision policy, and 18 reference quantities.
Acceptance/health determined tuning qualification; R-hat, ESS, MCSE and
reference agreement were used only for posterior assessment. The estimand is
the declared finite SGQF posterior, not the exact nonlinear state-space
posterior. The executed command and GPU/XLA/TF32/memory-growth provenance are
preserved in the run manifest and launch receipt.

Let `C` denote
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01`
relative to this directory, and `P=C/randomized-feasibility-01`.

| Member | Maximum steps | Epsilon | Warm-up per chain | Retained per chain | Posterior checks | Reference screen |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1 | 1 | 2000 | 10000 | Precision failed | Failed |
| 2 | 3 | 0.5 | 2000 | 10000 | Precision failed | Passed |
| 3 | 3 | 1 | 2000 | 5000 | Passed | Passed |
| 4 | 7 | 0.5 | 2000 | 3000 | Passed | Passed |
| 5 | 7 | 1 | 2000 | 5000 | Passed | Passed |
| 6 | 15 | 0.5 | 2000 | 3000 | Passed | Passed |
| 7 | 15 | 1 | 2000 | 7000 | Passed | Passed |
| 8 | 31 | 0.5 | 2000 | 3000 | Passed | Failed |

Members 1 and 2 exhausted the retained-count allowance with insufficient
precision. Member 1 also missed the normalized reference screen for `second_4`
(0.1036477867 against 0.1). Member 8 passed posterior checks but missed the
reference screen for `second_2` (0.1107636266 against 0.1); it is neither
precision-limited nor count-censored. These screen values are descriptive.
The policy does not permit extra draws or selective reruns after inspecting
these outcomes. Their failure rejects these deliveries, not randomized HMC as
a research direction. No reference tolerance, precision estimator, seed or
count limit changes are warranted by this result alone.

The terminal read-only audit passed: 197 source/input hashes, 896 qualification
chunks and 1,488 saved posterior tensor files were checked. All three services
exited successfully, with no restart. The authoritative settlement is
`P/terminal-audit-01/settlement.json`; its observer record is adjacent. This
note does not create a second settlement or replay any sampler.

The numerical supervisor consumed 68082.38066303777 seconds (18.9118 hours),
charged once without adding its overlapping worker. It released an old-pool
balance of 69278.81444643158 seconds. The already-held 3600-second local
reserve and historical 1579.5929542018566-second balance remain separate.
The observer's 40.50036860210821 active seconds are charged once to the new
grants in `C/additional-24h-20261016-01/reconciliation-02.json`; its sleeping
time is excluded. That reconciliation also carries forward the previously
charged 60.07246739137918 seconds of preparation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this feasibility attempt | All roots and verified members accounted for | Evidence audit passed | One target and seed | Preserve all eight outcomes | Broad validity or release |
| Accept five deliveries within this design | Posterior and reference screens passed | No numerical-health veto | Stopped precision/reference calibration remains limited | Retain for matched descriptive comparison | Exact posterior correctness or superiority |
| Reject three deliveries | Declared precision/reference screens failed | Delivery veto only; tuning unchanged | Dependence, reference error and Monte Carlo variation | Preserve failures and continue planned comparators | Rejection of the method or research direction |

| Inference status | Interpretation |
| --- | --- |
| Hard veto screen | No numerical-health veto; two precision failures and two reference failures across three members |
| Statistically supported ranking | None; this is one seed and target |
| Descriptive-only differences | Candidate delivery counts, retained counts, screen margins and occupied time |
| Default-readiness | Not established; N5/N8 remain open |
| Next evidence needed | Frozen SNAPER comparator, independent/model replication and scoped consumer/release checks |

Post-run skeptical review: the result is a complete engineering lifecycle
observation, not a calibrated method comparison. Passing diagnostics could
still miss bias or underestimate uncertainty under sequential stopping; two
reference misses show why a small R-hat alone is insufficient. The weakest
evidence is single-seed posterior reliability and the existing precision
calibration. Independent replications or an audited reference discrepancy
could overturn a favorable interpretation. The next comparator uses its
already-frozen proposals and fresh verification, without selecting proposals
using these posterior outcomes.
