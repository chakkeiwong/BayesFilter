# M13 surrogate-HMC terminal review — 2026-09-29

The repaired master executed, but M13 and the all-model campaign are incomplete.
Recovery03 ended at 23:11 HKT on 2026-09-28 with
`M13_PUBLIC_TUNING_BUDGET_EXHAUSTED`. The supervisor is no longer running.
The public tuning worker used 45,013.60 seconds of elapsed GPU allocation
(about 12.5 hours) and 46,884.55 CPU-worker seconds. The complete guard receipt
and final reports were inspected. This was an orderly budget stop.

The question remains testing **surrogate HMC**: canonical analytical DCTR forces
propose moves, and the separately frozen trajectory-importance endpoint plus
the prior defines the Metropolis correction. The endpoint is a finite
approximation; these results do not establish exact-likelihood inference.
Scope remains M13/data1704/T20, four state variables, two observation variables,
and two parameters. The force uses FP32/TF32/GPU/XLA; the endpoint uses FP64.

## Checked result

Four chains completed 6,250 fresh warm-up transitions each. Readiness uses the
maximum of rank-normalized split and folded R-hat over the latest 1,000 draws
per chain, requiring both parameters to be at most 1.05.

| Warm-up draws per chain | Parameter 1 R-hat | Parameter 2 R-hat |
| --- | --- | --- |
| 2,000 | 1.2890 | 1.6291 |
| 3,000 | 2.2701 | 1.4222 |
| 4,000 | 2.3790 | 2.1246 |
| 5,000 | 2.0519 | 1.3115 |
| 6,000 | 1.7966 | 1.1741 |

Every scheduled readiness check failed. The final 250-draw chunk had a health
check; its next readiness check was not yet due. All 25 fresh warm-up chunks
passed the recorded finite-value, endpoint, energy, divergence, movement,
state-continuity, and no-force-fallback checks. Earlier adaptation recorded one
force fallback; nomination and fresh warm-up recorded none. Do not describe the
entire run as fallback-free.

The frozen kernel used step size 0.06976348499025516 and one leapfrog step.
Poor exploration with this kernel is a hypothesis for the failed readiness
checks, not a diagnosed cause. No fresh verification or retained posterior
draws were produced. The final master report records unchanged execution sources.

## Budget and decision

The additional allocation was 60,000 CPU-worker and 60,000 GPU seconds, with
12,000 of each reserved for retained sampling and one fresh scientific attempt.
Across development, infrastructure retries, and recovery03, measured charges
are 47,662.586245 CPU-worker and 45,384.544549 GPU seconds. Remaining balances
are 12,337.413755 and 14,615.455451 seconds respectively. The tuning allowance,
not the entire allocation, stopped this attempt; most remaining compute is
reserved for a retained phase that cannot start without readiness and
verification. Earlier interrupted CPU usage remains unknown and separate.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| No M13 posterior promotion | Warm-up readiness failed; verification and retention not reached | Fresh numerical health passed; tuning resource cap stopped continuation | Whether kernel tuning, initialization, or a longer run explains slow mixing | Existing master phase review_candidate: inspect nomination and trajectory-length selection, then specify a prospective repair and budget | Rejection of surrogate HMC, exact posterior correctness, or all-model completion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No recorded fresh numerical-health veto; resource cap fired |
| Statistically supported ranking | None |
| Descriptive-only differences | Readiness-window R-hat history and runtime; no monotone improvement established |
| Default readiness | Not established by this run |
| Next evidence needed | Qualified tuning followed by cumulative retained ESS and the existing MCSE-aware reference comparison |

The next review is part of the existing master. A second scientific attempt or
transfer of the retained reserve is not automatic under this plan. No mid-phase
resume has been implemented. Repeating the same long run without diagnosing its
tuning would leave the cause unresolved.

Post-run skeptical review: the archives support failure to meet the specified
readiness criterion, not a mathematical failure of surrogate HMC. A better
tuned kernel passing readiness and retained reference checks would overturn any
broader negative inference. The weakest evidence is the absence of verification
and retained draws; finite health cannot substitute for them.

Evidence root: `artifacts/surrogate-hmc-m13-sequential-recovery-03/`.
Inspected `report.json`, `accounting.json`, `run-manifest.json`,
`m13-public-tuning/report.json`, `m13-public-tuning/guard-receipt.json`, and
`m13-public-tuning/tuning/fresh-progress.json`.
Plan: `surrogate-hmc-sequential-recovery-master.md`.
Execution commit: `8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`, with dirty execution
sources preserved in the run snapshot. No new experiment was run for this review.
