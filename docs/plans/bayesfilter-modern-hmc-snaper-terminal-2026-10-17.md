# Frozen SNAPER-proposal terminal result — 17 October 2026

The full NL-PP-SGQF frozen SNAPER-proposal cohort completed at 10:21 China
time; the terminal audit closed at 10:22. All six declared roots qualified,
were retained and received posterior assessment. Five passed both posterior
and reference screens. The first reached the retained-count cap with
insufficient lugsail precision. No repair descendants, unassessed members or
numerical-health vetoes remain. All six retain their tuning membership.

The immutable numerical source is `full-model-source-e68cdf1c4`, commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`. The unchanged design is
`priced-comparison-01/designs-02/snaper.json`, seed 71410, four chains, full
SGQF data and frozen geometry. These are previously frozen, target-specific
discarded-pilot proposals; this phase does not replicate fresh adaptation.
The comparator ladder retains fixed HMC, randomized HMC and fully refreshed
MAMS. NUTS is excluded. Acceptance/health determined tuning membership;
R-hat, ESS, lugsail MCSE and reference agreement assessed posterior delivery
only. The reference concerns the declared finite-filter posterior, not the
exact nonlinear state-space posterior.

Let `P` denote
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/snaper-feasibility-01`
relative to this directory. The exact command, environment, data/geometry
identities, seeds, TensorFlow memory-growth/XLA settings and device provenance
are in `P/run/manifest.json`, `P/run/supervisor.json` and the launch records.

| Member | Maximum steps | Epsilon | Warm-up per chain | Retained per chain | Posterior checks | Reference screen |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2 | 0.6269862024585631 | 2000 | 10000 | Precision failed | Passed |
| 2 | 2 | 0.7837327530732038 | 2000 | 7000 | Passed | Passed |
| 3 | 4 | 0.6269862024585631 | 2000 | 6000 | Passed | Passed |
| 4 | 4 | 0.7837327530732038 | 2000 | 8000 | Passed | Passed |
| 5 | 7 | 0.7837327530732038 | 2000 | 4000 | Passed | Passed |
| 6 | 8 | 0.6269862024585631 | 2000 | 3000 | Passed | Passed |

Member 1 passed R-hat, bulk/tail ESS and all 18 reference quantities. Its
declared lugsail precision report gives MCSE/SD
`0.020537019105533617` for `second_4`, exceeding `0.02`. The diagnostic's
separate default mean-MCSE summary is not the lugsail policy criterion and
must not replace it when interpreting this decision. This is a posterior
delivery failure under the frozen count/precision policy. It rejects neither
the tuning candidate nor the SNAPER research direction. No extra draws,
relaxed threshold or selective posterior retry is authorized by this outcome.

The read-only terminal audit passed: 197 source/input hashes, 480
qualification numerical chunks and 1,200 posterior tensor files were checked.
`P/run/result.json["posterior"]["rows"]` and
`P/run/posterior/result.json` hold the final posterior dispositions. The
top-level tuning rows retain earlier `posterior_not_started` values; those
are not the final posterior result. The authoritative settlement is
`P/terminal-audit-01/settlement.json`; this note does not settle the run again.

The inclusive supervisor charge was `42670.32444858877` seconds. Closeout
active inspection cost `19.84129042737186` seconds and monitoring cost
`15.325063032563776` seconds are charged separately, excluding sleep. From
the existing 180000-second reservation, the net release is
`137294.5091979513` seconds. `P/released-reservation-01.json` reconciles this
release once. Its scope excludes the terminal settlement's stale global
balance, concurrent SIR charges, the held 3600-second reserve and historical
1579.5929542018566-second balance. The release remains separate from running
SIR caps; it is not a new grant.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this feasibility attempt | All six roots and members have terminal dispositions | Integrity audit passed | One target and seed | Preserve every outcome and continue the SIR phase | Master completion or release |
| Retain five deliveries | Posterior precision/convergence and reference screens passed | No numerical-health veto | Repeated-look calibration and reference approximation | Use as descriptive feasibility evidence | Exact posterior correctness or superiority |
| Reject member 1's delivery | Declared lugsail precision failed at the retained cap | Posterior-only veto | Dependence and MCSE uncertainty | Preserve failure; continue planned calibration | Tuning failure or method rejection |

| Inference status | Interpretation |
| --- | --- |
| Hard veto screen | No numerical-health veto; one declared posterior precision failure |
| Statistically supported ranking | None; single target/seed and frozen proposals |
| Descriptive-only differences | Delivery and retained counts, precision margins and occupied time |
| Default-readiness | Not established; N5 calibration/confirmation and N8 consumer/release work remain open |
| Next evidence needed | SIR all-method feasibility, broader model/seed evidence and stopped-precision calibration |

Post-run skeptical review: finite-filter agreement and favorable R-hat can
coexist with underestimated uncertainty or bias. The lugsail screen is an
operational repeated-look rule, not an anytime coverage theorem. Frozen
proposals also cannot establish adaptation reliability. Independent
replications, stopping calibration or an audited reference discrepancy could
overturn a favorable interpretation. Delivery fractions and runtimes do not
support ranking these methods under the current evidence.
