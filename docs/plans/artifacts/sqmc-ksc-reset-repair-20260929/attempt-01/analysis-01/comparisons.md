# KSC reset repair: untouched comparisons

Full seven-mixture reference. Means average fixed-design replications; both analytical score coordinates are included.
FP64 GPU/XLA diagnostic variants, N=1008. No canonical/default admission.

| Method | T | Data | Selected arm | Mean absolute value error: old → new | Mean score L2 error: old → new | Paired L2 difference 95% interval |
|---|---:|---:|---|---:|---:|---|
| IID | 10 | 243001 | steps8 | 0.164099 → 0.0754135 | 0.271655 → 0.126182 | [-0.202014, -0.0862003] |
| IID | 10 | 243002 | steps8 | 0.121287 → 0.0565945 | 0.213359 → 0.0623996 | [-0.184526, -0.122814] |
| IID | 120 | 243001 | epsilon25 | 0.613882 → 0.344707 | 0.345598 → 0.224582 | [-0.265375, 0.0442521] |
| IID | 120 | 243002 | epsilon25 | 0.305787 → 0.477468 | 0.651389 → 0.244133 | [-0.589761, -0.206242] |
| Inverse CDF | 10 | 243001 | epsilon25 | 0.0962199 → 0.00863129 | 0.314437 → 0.0591738 | [-0.280041, -0.232049] |
| Inverse CDF | 10 | 243002 | epsilon25 | 0.09536 → 0.0182682 | 0.16842 → 0.0301148 | [-0.151708, -0.12451] |
| Inverse CDF | 120 | 243001 | steps8 | 0.536856 → 0.163187 | 0.329639 → 0.144456 | [-0.291356, -0.0845927] |
| Inverse CDF | 120 | 243002 | steps8 | 0.125318 → 0.213643 | 0.681811 → 0.146093 | [-0.654304, -0.399304] |
| Permutation | 10 | 243001 | epsilon25 | 0.0962675 → 0.00951738 | 0.315293 → 0.0577065 | [-0.281743, -0.234343] |
| Permutation | 10 | 243002 | epsilon25 | 0.0957508 → 0.0198976 | 0.168134 → 0.0307832 | [-0.150573, -0.123897] |
| Permutation | 120 | 243001 | steps8 | 0.535648 → 0.169851 | 0.329005 → 0.146068 | [-0.290023, -0.0807225] |
| Permutation | 120 | 243002 | steps8 | 0.125434 → 0.218893 | 0.672389 → 0.141498 | [-0.641491, -0.396889] |
| Permutation ablation | 10 | 243001 | epsilon25 | 0.096687 → 0.00951738 | 0.316445 → 0.0577065 | [-0.283861, -0.234865] |
| Permutation ablation | 10 | 243002 | epsilon25 | 0.0964303 → 0.0198976 | 0.169011 → 0.0307832 | [-0.151596, -0.124324] |
| Permutation ablation | 120 | 243001 | steps8 | 0.536366 → 0.169851 | 0.333713 → 0.146068 | [-0.303814, -0.0821004] |
| Permutation ablation | 120 | 243002 | steps8 | 0.124767 → 0.218893 | 0.692296 → 0.141498 | [-0.669999, -0.411548] |

## Actual values and scores: data 243001, T=10

| Method/arm | Mean log likelihood | Mean gamma score | Mean log-beta score | Mean-error SE: gamma, log-beta |
|---|---:|---:|---:|---|
| Full seven-mixture reference | -23.997617283 | 1.775756807 | -2.155586535 | quadrature checked |
| Gaussian Kalman heuristic | -24.277318301 | 1.397330823 | -1.876718752 | deterministic approximation |
| IID / baseline | -23.873544056 | 1.605961995 | -1.967129611 | 0.018954, 0.044008 |
| IID / candidate | -23.999192898 | 1.690016318 | -2.101782239 | 0.021877, 0.029619 |
| Inverse CDF / baseline | -23.901397376 | 1.534274759 | -1.955026825 | 0.01473, 0.0098586 |
| Inverse CDF / candidate | -23.994790511 | 1.725547845 | -2.133386402 | 0.0066477, 0.0093586 |
| Permutation / baseline | -23.901349786 | 1.534895006 | -1.953254156 | 0.014089, 0.010439 |
| Permutation / candidate | -23.995385248 | 1.727266797 | -2.133958555 | 0.0056986, 0.0098224 |
| Permutation ablation / baseline | -23.900930284 | 1.533061892 | -1.953773718 | 0.014909, 0.010006 |
| Permutation ablation / candidate | -23.995385248 | 1.727266797 | -2.133958555 | 0.0056986, 0.0098224 |

## Actual values and scores: data 243001, T=120

| Method/arm | Mean log likelihood | Mean gamma score | Mean log-beta score | Mean-error SE: gamma, log-beta |
|---|---:|---:|---:|---|
| Full seven-mixture reference | -270.702896202 | -0.508659556 | -1.291005193 | quadrature checked |
| Gaussian Kalman heuristic | -275.542306325 | -0.813750759 | -0.892859070 | deterministic approximation |
| IID / baseline | -271.316778239 | -0.432267129 | -1.085171066 | 0.10131, 0.055871 |
| IID / candidate | -270.879543409 | -0.633906248 | -1.217855750 | 0.099404, 0.026526 |
| Inverse CDF / baseline | -271.239752570 | -0.375158888 | -1.089143021 | 0.097776, 0.018019 |
| Inverse CDF / candidate | -270.821584886 | -0.580225487 | -1.269873809 | 0.053617, 0.012488 |
| Permutation / baseline | -271.238543975 | -0.371483139 | -1.087564459 | 0.097647, 0.01813 |
| Permutation / candidate | -270.823316362 | -0.577390069 | -1.269882173 | 0.054245, 0.013293 |
| Permutation ablation / baseline | -271.239262624 | -0.368794285 | -1.087245064 | 0.10179, 0.017 |
| Permutation ablation / candidate | -270.823316362 | -0.577390069 | -1.269882173 | 0.054245, 0.013293 |

## Actual values and scores: data 243002, T=10

| Method/arm | Mean log likelihood | Mean gamma score | Mean log-beta score | Mean-error SE: gamma, log-beta |
|---|---:|---:|---:|---|
| Full seven-mixture reference | -23.033735926 | -0.404750834 | -1.640443124 | quadrature checked |
| Gaussian Kalman heuristic | -22.796587340 | -0.373607301 | -1.218765426 | deterministic approximation |
| IID / baseline | -22.912448476 | -0.204826464 | -1.698341712 | 0.020362, 0.019468 |
| IID / candidate | -22.987823955 | -0.358005361 | -1.646108808 | 0.012178, 0.014354 |
| Inverse CDF / baseline | -22.938375969 | -0.242594702 | -1.684594126 | 0.011378, 0.0068138 |
| Inverse CDF / candidate | -23.023042755 | -0.384340817 | -1.649200319 | 0.011136, 0.0071708 |
| Permutation / baseline | -22.937985131 | -0.243081499 | -1.684825108 | 0.011426, 0.0071832 |
| Permutation / candidate | -23.021882052 | -0.384751260 | -1.648241534 | 0.011167, 0.0076686 |
| Permutation ablation / baseline | -22.937305646 | -0.242113153 | -1.684603395 | 0.011461, 0.0070468 |
| Permutation ablation / candidate | -23.021882052 | -0.384751260 | -1.648241534 | 0.011167, 0.0076686 |

## Actual values and scores: data 243002, T=120

| Method/arm | Mean log likelihood | Mean gamma score | Mean log-beta score | Mean-error SE: gamma, log-beta |
|---|---:|---:|---:|---|
| Full seven-mixture reference | -284.105359942 | 2.358935308 | 0.299061235 | quadrature checked |
| Gaussian Kalman heuristic | -289.119307428 | 2.448686517 | 0.655680321 | deterministic approximation |
| IID / baseline | -284.231222413 | 2.856512226 | 0.299130287 | 0.17132, 0.055216 |
| IID / candidate | -284.459443680 | 2.338869846 | 0.347294048 | 0.085933, 0.056453 |
| Inverse CDF / baseline | -284.160036239 | 3.035272629 | 0.331255873 | 0.062907, 0.028701 |
| Inverse CDF / candidate | -284.225769251 | 2.368748070 | 0.350155526 | 0.05522, 0.02619 |
| Permutation / baseline | -284.159942453 | 3.026332424 | 0.330404077 | 0.059098, 0.027898 |
| Permutation / candidate | -284.227196844 | 2.366669388 | 0.350044821 | 0.05302, 0.02552 |
| Permutation ablation / baseline | -284.158477299 | 3.046262376 | 0.333190588 | 0.062821, 0.027793 |
| Permutation ablation / candidate | -284.227196844 | 2.366669388 | 0.350044821 | 0.05302, 0.02552 |

Intervals are conditional on each fixed synthetic dataset and on four/eight design replications. They are exploratory, unadjusted for multiple cells, and do not establish a population-wide method ranking.
Component SE is replicate error SD divided by sqrt(n). The JSON also contains the full covariance of the mean error vector, its RMS vector SE (square root of covariance trace), and bootstrap SD/interval for the norm of the mean error vector. These differ from the SD of replicate error norms.
