# SIR optional mixed-precision fresh-scope result, 2026-10-12

The selected candidate passes the agreed1% numerical comparison on both fresh
validation datasets, at both parameter rows, with all50 observations completed.
FP32 remains the default. This candidate uses FP64 recursive/model/moment
arithmetic and centered FP32 pairwise Gaussian/transport kernels. N4032/K2016,
TF32off, GPU/XLA and memory growth are unchanged.

The comparator is the identical finite filter in uniform FP64. It is an
arithmetic reference, not an oracle for the SIR model score. Calibration uses
seeds26101231/32; validation uses26101233/34. Selection is descriptive nomination
using importance-weight variance times measured cost, not an HMC-efficiency
ranking. Candidate2 is selected. Both candidates0 and2 remain numerically viable.

| Candidate | Flow/reset/epsilon | Precision viable | Maximum scaled score gap, two datasets | Warm B2 seconds |
|---|---|---|---|---|
| 0 | 16/40/0.5 | True | 0.0735611% / 0.175862% | 61.96 |
| 1 | 32/80/0.5 | False | 183.628% / 1078.61% | 83.57 |
| 2 | 16/40/0.25 | True | 0.218942% / 0.320293% | 57.93 |

The scale is max(1,absolute FP64 component); near-zero components therefore use
an absolute tolerance of0.01. The doubled-step candidate1 fails materially,
despite tiny likelihood changes and healthy states. More iterations do not
guarantee numerical score robustness. The selected candidate's largest gap is
0.3203%; candidate0's is0.1759%. Neither comparison establishes exact model-score
accuracy or posterior convergence.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue with candidate2 for exploratory mechanics | Same-algorithm value/score discrepancy below1% on both datasets | No invalid rows/nonfinite outputs; candidate1 rejected | Only two validation datasets, local parameter rows; no untouched downstream run | Freeze selection; finish range precision and diagnose shared cost | Canonical LEDH admission, exact posterior, default readiness |
| Continue cost repair | Warm B2 score about58s | No speed promotion test | Possible competing GPU load and parameter dependence | Measure shared primitives against fused step | Statistically supported speed ranking |

| Inference status | Finding |
|---|---|
| Hard veto screen | Candidate1 fails score precision; candidates0/2 pass the stated local numerical screen |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, scaled precision errors and variance-times-cost nomination |
| Default-readiness | FP32 unchanged; mixed arithmetic optional, not promoted |
| Next evidence needed | Frozen-selection HMC mechanics, new seeds26101235/36 for generalization, downstream cumulative convergence/reference checks |

Pre-mortem/alternative: success could be local to these parameter rows and
datasets; propagation can amplify pairwise rounding elsewhere. Any fresh
frozen-selection precision failure overturns robustness beyond this screen.
The weakest evidence is posterior-region coverage. The saved campaign data
has already been inspected and may support exploratory mechanics only.

Command/environment/source/device/memory manifest: ../surrogate-hmc-mixed-scope-sir-20261012-01/sir_d18/result.json.
Selection and complete per-component values/scores: ../surrogate-hmc-mixed-scope-sir-20261012-01/sir_d18/filter-selection.json.
Wrapper time3736.7306s, charged to the ten-hour ledger. After this result,
25250.4133s used,10749.5867s remaining. Range mixed scope is now running with
a2500-second cap in ../surrogate-hmc-mixed-scope-range-20261012-01.
