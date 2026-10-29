# Independent R campaign phase result

Phase: phase01-controller-pending. Fixed-count CPU reference analysis.

| Data | d | Method | Complete | Mean ratio (95% interval) | SD (95% interval) | Mean N | Practical | Heuristic | Literal pattern |
|---|---:|---|---:|---|---|---:|---|---|---|
| 89500080 | 80 | qr | 8/8 | 0.98075 [0.8862, 1.096] | 0.16135 [0.06085, 0.214] | 2000 | False | passed observed screen | False |
| 89500080 | 80 | short_qr | 8/8 | 1.0202 [0.88, 1.179] | 0.2366 [0.0915, 0.3136] | 1000 | True | passed observed screen | False |
| 89500080 | 80 | bpf | 8/8 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89500080 | 80 | fully_adapted | 8/8 | 0.12887 [0.07681, 0.2031] | 0.10015 [0.02531, 0.1416] | 5000 | N/A | comparator | N/A |
| 89500080 | 80 | sis | 8/8 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |

A heuristic veto means an observed conditional loss to a simple comparator; it does not
establish population inferiority when the paired interval includes zero. Intervals are
pointwise, conditional on each fixed dataset, and are not simultaneous or cross-dataset inference.
Finite log ratios with exponentiation underflow are counted in summary.json. A displayed zero
sample SD for an underflowed baseline is not zero theoretical variance.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| 89500080 qr: not promoted | False | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89500080 short_qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Exact record identity, source hashes, seeds, fit/tail and missing-pair status are preserved per cell. |
| Statistically supported ranking | No global ranking. Paired pointwise intervals in summary.json support only their specified contrast and fixed data. |
| Descriptive-only differences | Raw means, extreme ratios, resampling counts and unmatched-cost timing do not establish superiority. |
| Default-readiness | No algorithm or production default changes. QR differs from paper Equation15. |
| Next evidence needed | Complete the remaining frozen campaign; exact original settings/data and multivariate full-filter R/TF parity remain open. |

Post-run skeptical review: the strongest alternative explanation is data-specific Monte Carlo
variation or unobserved rare large weights. Larger independent fixed-count runs could overturn
a small-sample screen. The weakest evidence is literal matching to unpublished original data
and implementation choices. These results do not prove correctness or failure of LEDH, KDM or HMC.
