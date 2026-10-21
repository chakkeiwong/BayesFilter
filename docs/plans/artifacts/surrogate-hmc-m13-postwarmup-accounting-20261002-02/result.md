# Corrected M13 posterior sample accounting

The owner's objection is correct. The 9,000 draws called verification are eligible posterior samples. They followed warmup, used a frozen kernel with adaptation disabled, and the next 3,750 draws continued the same four chain states. Excluding the first 9,000 from posterior accounting was wrong. The prior suffix-only interpretation is superseded.

No new HMC transitions were generated. This CPU-only TensorFlow diagnostic checked tensor/manifest checksums, counts, target/kernel binding, unique seeds, numerical health, exact saved state handoffs, and the affine transform against archived raw samples. It reproduced the saved prefix R-hat values exactly.

Parameters below are log multipliers of process and observation standard deviations, in that order: Q=Q0 exp(2 lambda_Q), R=R0 exp(2 lambda_R).

| Draws per chain | R-hat, Q / R | Bulk ESS, Q / R | Tail ESS, Q / R | Mean MCSE / posterior SD, Q / R |
|---|---|---|---|---|
| First 9,000 post-warmup | 1.00742 / 1.00547 | 512 / 1,076 | 324 / 1,956 | 4.27% / 3.05% |
| All 12,750 post-warmup | 1.05418 / 1.00884 | 77 / 321 | 30 / 483 | 13.44% / 5.64% |

At 9,000, the owner's R-hat stopping rule was satisfied. There was no need to restart posterior counting. Prefix means were 0.07647 and -0.14097 with estimated MCSE 0.01123 and 0.00517. The old ESS>=400 screen flags Q's tail ESS, but this does not make the draws ineligible for inference. Precision depends on the quantity estimated.

The later draws now exist and reveal slower mixing than the prefix suggested. Cumulative means are 0.03027 and -0.12992 with estimated MCSE 0.04178 and 0.00968. We must report this evidence rather than select only the favorable prefix. It does not retroactively justify the unnecessary second sampling phase.

The combined sample archive is all-postwarmup-samples.tfbin, shape [12750,4,2], with its checksum in report.json. These draws target the unchanged frozen importance-likelihood posterior. This is not an exact-likelihood, cross-model, comparative-efficiency, or default-readiness result. The old six-look equivalence calibration is not reused at unsupported counts.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Count all eligible post-warmup draws | Frozen kernel and exact continuity pass | Archive/health checks pass | Finite-chain precision | Use 12,750 cumulatively | No need for a separate sample set |
| Report current mixing issue | Q R-hat 1.05418 exceeds 1.01 | No numerical veto; deadline expired | Slow exploration | Repair cumulative consumer before any budgeted continuation | Not rejection of surrogate HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical health passed; expired deadline stops more execution |
| Statistically supported ranking | No ranking or method comparison performed |
| Descriptive-only differences | Prefix/cumulative R-hat, ESS and acceptance; these nested samples are not independent replications |
| Default-readiness | Not assessed |
| Next evidence needed | Any authorized continuation must assess all post-warmup draws cumulatively and report MCSE |

Post-run review: a late visit to a slowly explored region is a plausible explanation for the change. Target/kernel continuity is checked; localization of that region is not. Further healthy cumulative mixing could overturn the current R-hat failure. MCSE estimates are less reliable with poor mixing and are estimates, not guarantees.

Attempt 01 stopped on an unnecessarily exact CPU/GPU reconstruction of the original initial coordinates. The measured difference is 2.22e-16. Attempt 02 allows only FP64 rounding there while retaining exact checks on every saved state handoff. It passed in 3.23 seconds wall time / 6.96 CPU seconds. Logs and scripts preserve both attempts.

Runtime consumer repair remains pending. This artifact corrects the stopped run's sample accounting and interpretation; it does not claim the old launcher is repaired. No further sampling budget is inferred.
