# Independent R campaign phase result

Phase: phase03-first-study-1000. Fixed-count CPU reference analysis.

| Data | d | Method | Complete | Mean ratio (95% interval) | SD (95% interval) | Mean N | Practical | Heuristic | Literal pattern |
|---|---:|---|---:|---|---|---:|---|---|---|
| 89800005 | 5 | qr | 1000/1000 | 0.99883 [0.9964, 1.001] | 0.040856 [0.03914, 0.04246] | 1000 | True | passed observed screen | False |
| 89800005 | 5 | short_qr | 1000/1000 | 0.99883 [0.9964, 1.001] | 0.040856 [0.03914, 0.04246] | 1000 | True | passed observed screen | False |
| 89800005 | 5 | bpf | 1000/1000 | 1.0055 [0.9636, 1.048] | 0.68127 [0.6065, 0.7563] | 10000 | N/A | comparator | N/A |
| 89800005 | 5 | fully_adapted | 1000/1000 | 0.99939 [0.9933, 1.005] | 0.096616 [0.09238, 0.1005] | 5000 | N/A | comparator | N/A |
| 89800005 | 5 | sis | 1000/1000 | 2.6339e-100 [1.039e-107, 7.902e-100] | 8.3286e-99 [3.284e-106, 1.441e-98] | 10000 | N/A | comparator | N/A |
| 89800010 | 10 | qr | 1000/1000 | 1.0046 [1, 1.009] | 0.069193 [0.0662, 0.07212] | 1000 | True | passed observed screen | False |
| 89800010 | 10 | short_qr | 1000/1000 | 1.0046 [1, 1.009] | 0.069193 [0.0662, 0.07212] | 1000 | True | passed observed screen | False |
| 89800010 | 10 | bpf | 1000/1000 | 1.1575 [0.8856, 1.46] | 4.6259 [3.149, 5.942] | 10000 | N/A | comparator | N/A |
| 89800010 | 10 | fully_adapted | 1000/1000 | 1.0006 [0.9921, 1.009] | 0.13852 [0.132, 0.145] | 5000 | N/A | comparator | N/A |
| 89800010 | 10 | sis | 1000/1000 | 1.7489e-270 [3.866e-283, 5.247e-270] | 5.5305e-269 [0, 0] | 10000 | N/A | comparator | N/A |
| 89800020 | 20 | qr | 1000/1000 | 0.99931 [0.9929, 1.006] | 0.10338 [0.0988, 0.1082] | 1000 | True | passed observed screen | False |
| 89800020 | 20 | short_qr | 1000/1000 | 0.99931 [0.9929, 1.006] | 0.10338 [0.0988, 0.1082] | 1000 | True | passed observed screen | False |
| 89800020 | 20 | bpf | 1000/1000 | 2.4073e-14 [1.36e-16, 7.125e-14] | 7.3114e-13 [3.936e-15, 1.265e-12] | 10000 | N/A | comparator | N/A |
| 89800020 | 20 | fully_adapted | 1000/1000 | 0.98874 [0.9741, 1.003] | 0.23122 [0.2188, 0.2431] | 5000 | N/A | comparator | N/A |
| 89800020 | 20 | sis | 1000/1000 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89800040 | 40 | qr | 1000/1000 | 0.9988 [0.9893, 1.008] | 0.15437 [0.1471, 0.1616] | 1092 | True | passed observed screen | False |
| 89800040 | 40 | short_qr | 1000/1000 | 0.99818 [0.9884, 1.008] | 0.15843 [0.1511, 0.1657] | 1001 | True | passed observed screen | False |
| 89800040 | 40 | bpf | 1000/1000 | 2.7795e-196 [1.452e-201, 8.338e-196] | 8.7882e-195 [0, 0] | 10000 | N/A | comparator | N/A |
| 89800040 | 40 | fully_adapted | 1000/1000 | 1.0219 [0.9756, 1.07] | 0.75189 [0.6619, 0.8396] | 5000 | N/A | comparator | N/A |
| 89800040 | 40 | sis | 1000/1000 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89800080 | 80 | qr | 1000/1000 | 0.99517 [0.9842, 1.006] | 0.17948 [0.1712, 0.188] | 1971 | False | veto | False |
| 89800080 | 80 | short_qr | 1000/1000 | 0.99071 [0.9757, 1.007] | 0.24963 [0.2375, 0.262] | 1068 | True | veto | True |
| 89800080 | 80 | bpf | 1000/1000 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89800080 | 80 | fully_adapted | 1000/1000 | 0.83927 [0.5587, 1.292] | 6.6238 [1.366, 11.23] | 5000 | N/A | comparator | N/A |
| 89800080 | 80 | sis | 1000/1000 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |

A heuristic veto means an observed conditional loss to a simple comparator; it does not
establish population inferiority when the paired interval includes zero. Intervals are
pointwise, conditional on each fixed dataset, and are not simultaneous or cross-dataset inference.
Finite log ratios with exponentiation underflow are counted in summary.json. A displayed zero
sample SD for an underflowed baseline is not zero theoretical variance.

| Data | d | Method | Observed / paper SD | Observed / paper mean N | Observed / paper resampling | Whole-run seconds |
|---|---:|---|---|---|---|---:|
| 89800005 | 5 | qr | 0.040856 / 0.09 | 1000 / 1000 | 1.016 / 6.93 | 1.9649 |
| 89800005 | 5 | short_qr | 0.040856 / 0.09 | 1000 / 1000 | 1.016 / 6.93 | 1.7612 |
| 89800005 | 5 | bpf | 0.68127 / 0.51 | 10000 / 10000 | 99 / 99 | 0.31435 |
| 89800005 | 5 | fully_adapted | 0.096616 / 0.1 | 5000 / 5000 | 27.037 / 26.04 | 0.16339 |
| 89800005 | 5 | sis | 8.3286e-99 / unreported | 10000 / unreported | 0 / unreported | 0.23017 |
| 89800010 | 10 | qr | 0.069193 / 0.14 | 1000 / 1000 | 4.002 / 15.11 | 2.6833 |
| 89800010 | 10 | short_qr | 0.069193 / 0.14 | 1000 / 1000 | 4.002 / 15.11 | 2.4812 |
| 89800010 | 10 | bpf | 4.6259 / 6.4 | 10000 / 10000 | 99 / 99 | 0.55813 |
| 89800010 | 10 | fully_adapted | 0.13852 / 0.17 | 5000 / 5000 | 48.857 / 52.71 | 0.25498 |
| 89800010 | 10 | sis | 5.5305e-269 / unreported | 10000 / unreported | 0 / unreported | 0.49805 |
| 89800020 | 20 | qr | 0.10338 / 0.19 | 1000 / 1000 | 9.297 / 27.61 | 4.3909 |
| 89800020 | 20 | short_qr | 0.10338 / 0.19 | 1000 / 1000 | 9.297 / 27.61 | 4.0393 |
| 89800020 | 20 | bpf | 7.3114e-13 / unreported | 10000 / unreported | 99 / unreported | 1.0005 |
| 89800020 | 20 | fully_adapted | 0.23122 / 0.53 | 5000 / 5000 | 84.533 / 84.98 | 0.49333 |
| 89800020 | 20 | sis | 0 / unreported | 10000 / unreported | 0 / unreported | 0.94825 |
| 89800040 | 40 | qr | 0.15437 / 0.23 | 1092 / 1033 | 19.563 / 42.41 | 8.1129 |
| 89800040 | 40 | short_qr | 0.15843 / 0.23 | 1001 / 1033 | 19.569 / 42.41 | 7.5351 |
| 89800040 | 40 | bpf | 8.7882e-195 / unreported | 10000 / unreported | 99 / unreported | 2.1998 |
| 89800040 | 40 | fully_adapted | 0.75189 / unreported | 5000 / unreported | 99 / unreported | 1.1554 |
| 89800040 | 40 | sis | 0 / unreported | 10000 / unreported | 0 / unreported | 2.1811 |
| 89800080 | 80 | qr | 0.17948 / 0.35 | 1971 / 1142 | 37.19 / 71.88 | 20.859 |
| 89800080 | 80 | short_qr | 0.24963 / 0.35 | 1068 / 1142 | 37.348 / 71.88 | 16.076 |
| 89800080 | 80 | bpf | 0 / unreported | 10000 / unreported | 99 / unreported | 4.8525 |
| 89800080 | 80 | fully_adapted | 6.6238 / unreported | 5000 / unreported | 99 / unreported | 2.7533 |
| 89800080 | 80 | sis | 0 / unreported | 10000 / unreported | 0 / unreported | 5.3542 |

Paper values come from the preserved GJL first-study tables. Different datasets and
QR's different fitting objective prevent literal equivalence claims. Whole-run timing
includes learning and final evaluation and is not matched-cost or cross-machine evidence.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| 89800005 qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800005 short_qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800010 qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800010 short_qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800020 qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800020 short_qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800040 qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800040 short_qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800080 qr: not promoted | False | heuristic=veto; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89800080 short_qr: not promoted | True | heuristic=veto; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |

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
