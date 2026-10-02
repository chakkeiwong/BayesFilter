# KSC oracle comparison: numerical results

This is a numerical reading of the saved September 29 residual-design and
coordinate-cap comparison, not a new filtering run. The October safety guard
has not yet been evaluated against these oracles. Every number below comes
from the saved source summary or actual-value table.

The scope is scalar KSC with the full seven-component observation mixture,
N=1008, gamma=1.5, log_beta=0, FP64 GPU/XLA with TF32 off, and horizons 10 and
120. Dataset 243001 is A and 243002 is B. Each untouched cell has eight paired
particle designs. The full-mixture references were checked by quadrature
refinement and an independent density-grid recurrence to 1e-7.

Old denotes the original reset and cap. New denotes the selected richer-design,
identity-core-cap repair with eight moment-correction steps. The terminal OT
regularization is 25.6 for the epsilon25 arm and 102.4 for the steps8 arm; thus
this is a comparison of the selected combined configurations, not an isolated
causal estimate of the residual-design change.

Log-likelihood error is the mean of absolute per-design log-likelihood errors.
Score error is the mean Euclidean norm of the per-design error in the two score
coordinates (gamma, log_beta). These are not the absolute error or score-error
norm of the averaged estimates. A negative score change means lower error.
The saved paired 95% intervals describe particle-design variation conditional
on a fixed dataset; they are exploratory and unadjusted for multiple comparisons.
No likelihood-error interval was supplied by this summary, so those mean changes
are descriptive. No population-wide method ranking is inferred.

## Untouched numerical comparisons

| Method | T | Dataset | Selected arm | Mean absolute log-likelihood error: old → new | Mean score L2 error: old → new | Paired score-error change, 95% interval |
|---|---:|---:|---|---:|---:|---:|

| IID | 10 | 243001 | steps8 | 0.164099 → 0.075413 | 0.271655 → 0.126182 | -0.145472 [-0.202014, -0.086200] |
| Inverse CDF | 10 | 243001 | epsilon25 | 0.096220 → 0.008631 | 0.314437 → 0.059174 | -0.255263 [-0.280041, -0.232049] |
| Permutation | 10 | 243001 | epsilon25 | 0.096267 → 0.009517 | 0.315293 → 0.057707 | -0.257587 [-0.281743, -0.234343] |
| Permutation ablation | 10 | 243001 | epsilon25 | 0.096687 → 0.009517 | 0.316445 → 0.057707 | -0.258739 [-0.283861, -0.234865] |
| IID | 10 | 243002 | steps8 | 0.121287 → 0.056595 | 0.213359 → 0.062400 | -0.150959 [-0.184526, -0.122814] |
| Inverse CDF | 10 | 243002 | epsilon25 | 0.095360 → 0.018268 | 0.168420 → 0.030115 | -0.138305 [-0.151708, -0.124510] |
| Permutation | 10 | 243002 | epsilon25 | 0.095751 → 0.019898 | 0.168134 → 0.030783 | -0.137350 [-0.150573, -0.123897] |
| Permutation ablation | 10 | 243002 | epsilon25 | 0.096430 → 0.019898 | 0.169011 → 0.030783 | -0.138228 [-0.151596, -0.124324] |
| IID | 120 | 243001 | epsilon25 | 0.613882 → 0.344707 | 0.345598 → 0.224582 | -0.121016 [-0.265375, 0.044252] |
| Inverse CDF | 120 | 243001 | steps8 | 0.536856 → 0.163187 | 0.329639 → 0.144456 | -0.185184 [-0.291356, -0.084593] |
| Permutation | 120 | 243001 | steps8 | 0.535648 → 0.169851 | 0.329005 → 0.146068 | -0.182937 [-0.290023, -0.080722] |
| Permutation ablation | 120 | 243001 | steps8 | 0.536366 → 0.169851 | 0.333713 → 0.146068 | -0.187645 [-0.303814, -0.082100] |
| IID | 120 | 243002 | epsilon25 | 0.305787 → 0.477468 | 0.651389 → 0.244133 | -0.407256 [-0.589761, -0.206242] |
| Inverse CDF | 120 | 243002 | steps8 | 0.125318 → 0.213643 | 0.681811 → 0.146093 | -0.535718 [-0.654304, -0.399304] |
| Permutation | 120 | 243002 | steps8 | 0.125434 → 0.218893 | 0.672389 → 0.141498 | -0.530891 [-0.641491, -0.396889] |
| Permutation ablation | 120 | 243002 | steps8 | 0.124767 → 0.218893 | 0.692296 → 0.141498 | -0.550798 [-0.669999, -0.411548] |

## Validation dataset, reported numerically

These frozen-configuration comparisons used two designs per cell on a separate
validation dataset. Their small replication count limits uncertainty assessment.
They show whether the untouched results recur on the earlier validation data.

| Method | T | Mean absolute log-likelihood error: old → new | Mean score L2 error: old → new |
|---|---:|---:|---:|
| IID | 10 | 0.078530 → 0.141842 | 0.082175 → 0.093201 |
| IID | 120 | 0.596813 → 0.433676 | 0.381482 → 0.431605 |
| Inverse CDF | 10 | 0.011069 → 0.012307 | 0.066595 → 0.010528 |
| Inverse CDF | 120 | 0.548057 → 0.084772 | 0.073887 → 0.121654 |
| Permutation | 10 | 0.010833 → 0.015942 | 0.066328 → 0.009831 |
| Permutation | 120 | 0.552193 → 0.083290 | 0.066158 → 0.123429 |
| Permutation ablation | 10 | 0.010872 → 0.015942 | 0.066767 → 0.009831 |
| Permutation ablation | 120 | 0.554177 → 0.083290 | 0.073987 → 0.123429 |

The inverse-CDF T=120 validation comparison has score error 0.073887 → 0.121654
and log-likelihood error 0.548057 → 0.084772. Thus the score/likelihood trade-off
changes across datasets; the favorable untouched score changes alone do not
establish uniform improvement.

## Fourth-moment mechanism at the saved first reset

The incoming weighted cloud has standardized fourth moment 2.916221. This is a
particle-cloud target, not an oracle posterior fourth moment.

| Quantity | Original reset/cap | Richer design and identity-core cap |
|---|---:|---:|
| Raw reset fourth moment | 1.000473 | 2.972274 |
| After moment correction | 1.577655 | 2.966277 |
| Final fourth moment | 1.328293 | 2.966277 |
| Absolute discrepancy from incoming-cloud target | 1.587928 | 0.050056 |

This local discrepancy is about 96.85% smaller. It explains the mechanism without
establishing the true filtering fourth moment or long-horizon accuracy.


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


## Provenance and interpretation

The figure and CSV display all sixteen untouched cells, without binary verdicts.
The original predeclared criteria and decisions remain preserved in the source
report; this readout changes presentation, not the experimental criteria.
The source paths and checksums are recorded in manifest.json. Only deterministic
post-run arithmetic and plotting were executed; no new likelihood/score evidence
was generated. CPU-only reporting; GPUs were intentionally hidden.

