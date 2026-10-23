# Independently calibrated SIR guidance

OT remains (1,5). These are conditional comparisons of the optional three-density consumer, not canonical LEDH or original neural KDM claims.

## Data 26100831

| Arm | Valid / failed | Beta | Mean score | Score RMSE |
|---|---:|---|---|---:|
| plain | 8 / 0 | [0.1, 1.9834122663984284e-26, 0.9] | -205.2177, 121.8254, 63.0800 | 94.6240 |
| csmc_quadratic | 8 / 0 | [0.9360533833503722, 5.551115123125783e-17, 0.06394661664962772] | -205.4216, 127.4172, 64.7511 | 96.3592 |
| kernel | 8 / 0 | [0.850125241279602, 1.0534040281555085e-13, 0.14987475872029268] | -193.6360, 122.4223, 63.9493 | 84.5113 |
| plain_quadratic_beta | 8 / 0 | [0.9360533833503722, 5.551115123125783e-17, 0.06394661664962772] | -190.8559, 115.4455, 61.8914 | 82.2319 |
| plain_kernel_beta | 8 / 0 | [0.850125241279602, 1.0534040281555085e-13, 0.14987475872029268] | -193.5394, 117.3565, 62.3434 | 83.8621 |
| prior | 8 / 0 | [1.0, 0.0, 0.0] | -190.0426, 114.0994, 61.8929 | 82.1886 |
| local | 8 / 0 | [0.1, 0.0, 0.9] | -205.2177, 121.8254, 63.0800 | 94.6240 |

| Contrast | Pairs | Difference in squared score error | 99.375% interval |
|---|---:|---:|---|
| quadratic_matched | 8 | 2523.0197713378884 | [-5902.929147319651, 15949.534744032822] |
| kernel_matched | 8 | 109.30256786146076 | [-4087.144042433851, 4433.123943723924] |
| quadratic_calibrated | 8 | 331.39908197936836 | [-10795.918617603882, 15524.884309350455] |
| kernel_calibrated | 8 | -1811.5508129425598 | [-6298.607895741311, 3270.145472827745] |

Component-wise heuristic losses: [{'candidate': 'csmc_quadratic', 'comparator': 'plain', 'coordinate': 1}, {'candidate': 'csmc_quadratic', 'comparator': 'plain', 'coordinate': 2}, {'candidate': 'csmc_quadratic', 'comparator': 'prior', 'coordinate': 0}, {'candidate': 'csmc_quadratic', 'comparator': 'prior', 'coordinate': 1}, {'candidate': 'csmc_quadratic', 'comparator': 'prior', 'coordinate': 2}, {'candidate': 'csmc_quadratic', 'comparator': 'local', 'coordinate': 1}, {'candidate': 'csmc_quadratic', 'comparator': 'local', 'coordinate': 2}, {'candidate': 'kernel', 'comparator': 'plain', 'coordinate': 1}, {'candidate': 'kernel', 'comparator': 'prior', 'coordinate': 0}, {'candidate': 'kernel', 'comparator': 'prior', 'coordinate': 1}, {'candidate': 'kernel', 'comparator': 'prior', 'coordinate': 2}, {'candidate': 'kernel', 'comparator': 'local', 'coordinate': 1}]

## Data 26100832

| Arm | Valid / failed | Beta | Mean score | Score RMSE |
|---|---:|---|---|---:|
| plain | 8 / 0 | [0.1, 7.035073878712627e-28, 0.9] | -179.0975, 66.6768, 10.0754 | 77.6984 |
| csmc_quadratic | 8 / 0 | [0.8078540563583374, 5.551115123125783e-17, 0.1921459436416626] | -169.4087, 64.9340, 9.7124 | 70.8741 |
| kernel | 8 / 0 | [0.5347591717154986, 2.2057542903297644e-08, 0.46524080622695857] | -164.2021, 62.5397, 10.1978 | 65.4509 |
| plain_quadratic_beta | 8 / 0 | [0.8078540563583374, 5.551115123125783e-17, 0.1921459436416626] | -180.6295, 67.4894, 9.9719 | 82.4412 |
| plain_kernel_beta | 8 / 0 | [0.5347591717154986, 2.2057542903297644e-08, 0.46524080622695857] | -179.9901, 67.6657, 10.0839 | 83.6040 |
| prior | 8 / 0 | [1.0, 0.0, 0.0] | -180.4842, 68.0019, 9.9259 | 83.8851 |
| local | 8 / 0 | [0.1, 0.0, 0.9] | -179.0975, 66.6768, 10.0754 | 77.6984 |

| Contrast | Pairs | Difference in squared score error | 99.375% interval |
|---|---:|---:|---|
| quadratic_matched | 8 | -1773.4244523312116 | [-3845.2583675601027, -400.5493106271887] |
| kernel_matched | 8 | -2705.806546929686 | [-7192.819453617393, -102.71974362445172] |
| quadratic_calibrated | 8 | -1013.9009494281183 | [-3192.336294954463, 647.1506162070337] |
| kernel_calibrated | 8 | -1753.2101576943803 | [-5215.549294908207, 820.0837682242618] |

Component-wise heuristic losses: [{'candidate': 'kernel', 'comparator': 'plain', 'coordinate': 2}, {'candidate': 'kernel', 'comparator': 'prior', 'coordinate': 2}, {'candidate': 'kernel', 'comparator': 'local', 'coordinate': 2}]

## Decision and uncertainty

| Field | Finding |
|---|---|
| Decision | No canonical/default promotion; inspect conditional contrasts |
| Primary criterion | Joint conditional gains: {'csmc_quadratic': False, 'kernel': False} |
| Veto status | 0 invalid rows; heuristic verdict {'csmc_quadratic': 'fail', 'kernel': 'fail'} |
| Main uncertainty | Eight seeds, one guide fit per data set, approximate reference not propagated |
| Next justified action | Resolve remaining message/consumer mismatch before claiming improved LEDH |
| Not concluded | Canonical LEDH correctness, original KDM performance, equal-compute or fresh-data superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | 0 failures; full valid design=True |
| Statistically supported ranking | Only contrasts with intervals excluding zero, conditional on the frozen fits and references |
| Descriptive-only differences | Scores, RMSE, timings and heuristic component comparisons |
| Default-readiness | Not established |
| Next evidence needed | Independent guide fits, reference uncertainty, fresh data and canonical conformance |

Lower pilot J is not a score-accuracy result. Matched-beta comparisons isolate guidance from mixture selection. A remaining loss could reflect the positive-information projection or limited kernel activity, rather than failure of controlled SMC/KDM as research directions. The weakest evidence remains the single fitted guide and approximate reference; fresh fits or a canonical consumer can overturn these conditional findings.
