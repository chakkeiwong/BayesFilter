# Phase 41 Report Repair and Refresh Note

Date: 2026-08-26  
Continuation version: `v2.3-independent-audit-bank`

## Attempt ledger

| Attempt | Failure class | Evidence | Repair | Status |
|---|---|---|---|---|
| CPU-hidden fresh pilot | none | passing C0/M0 theta-measure receipt; exact target and arm-specific protocol hashes; distinct tensor hashes | continue | pass |
| GPU frozen-training attempts 1 and 2 | harness defects | arm-specific C0 protocol mismatch was caught before training; then static validation shape relaxation was caught after GPU/XLA initialization | corrected arm hash checks and shape-specific validation compilation; focused regression passed | repaired |
| GPU frozen-training attempt 3 | none | all four arms pass v2.3 boundary; fresh rows marked unused for training and selection | continue to read-only report | pass |
| CPU report attempt 1 | reporter dtype defect | JSON-decoded affine covariance was float32 and `_max_abs` requested float64 without an explicit cast; no result was emitted | cast diagnostic tensors to float64 before reduction; preserve failed report root | repaired |

The report failure is an engineering/reporting issue only. It did not alter
the source pilots or the successful GPU audit and did not produce a scientific
comparison result. The next report uses a new output root
`phase41-independent-audit-bank/report-attempt2/`.

## Refresh boundary

The report remains read-only and bound to the same target signature, M0/C0
protocol hashes, Phase 40 measure report, and Phase 41 terminal audit receipt.
No support summary, moment, loss, or branch label is used for optimizer
selection. A valid report may trigger the next repair, but it cannot promote
IID whitening, posterior correctness, HMC, canonical LEDH, or a statistical
ranking from one fresh bank and one training seed.
