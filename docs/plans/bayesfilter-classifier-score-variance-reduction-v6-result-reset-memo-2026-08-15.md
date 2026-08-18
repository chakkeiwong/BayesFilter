# V6 Classifier-Score Variance-Reduction Result And Reset Memo

Date: 2026-08-15  
Status: `COMPLETED`

## Direct Answer

Increasing the training bank from 2,048 to 8,192 paths per class and delta
reduced classifier-score training-bundle variance reliably. Common random
numbers also helped strongly in the exact Gaussian problem and modestly on the
SIR 128-path audit bank, but CRN was not reliable on the one fixed SIR path.

The combined `crn_n8192` arm passed the predeclared Gaussian variance and exact
MSE guards. In SIR it was descriptively favorable but did not pass the primary
two-part criterion because the fixed-path variance-ratio interval crossed 1.
There is no exact SIR score claim.

## Primary Results

Ratios below 1 favor the numerator treatment. Intervals are paired 95%
bootstrap intervals over matched bundle/path clusters.

| Model | Comparison | 128-path audit variance ratio | Fixed-path variance ratio | Verdict |
|---|---|---:|---:|---|
| Gaussian | CRN at `n=2048` | 0.406 [0.365, 0.452] | 0.299 [0.118, 0.832] | supported |
| Gaussian | CRN at `n=8192` | 0.493 [0.450, 0.541] | 0.319 [0.192, 0.497] | supported |
| Gaussian | 8192 vs 2048, independent | 0.334 [0.301, 0.373] | 0.305 [0.150, 0.684] | supported |
| Gaussian | 8192 vs 2048, CRN | 0.405 [0.377, 0.435] | 0.326 [0.219, 0.459] | supported |
| Gaussian | combined vs baseline | 0.165 [0.146, 0.185] | 0.097 [0.040, 0.269] | primary success |
| SIR | CRN at `n=2048` | 0.890 [0.787, 1.003] | 0.626 [0.245, 1.416] | not supported |
| SIR | CRN at `n=8192` | 0.901 [0.811, 0.997] | 1.488 [0.480, 3.377] | audit only; fixed inconclusive |
| SIR | 8192 vs 2048, independent | 0.483 [0.441, 0.531] | 0.380 [0.193, 0.886] | supported |
| SIR | 8192 vs 2048, CRN | 0.489 [0.432, 0.559] | 0.904 [0.269, 2.325] | audit only; fixed inconclusive |
| SIR | combined vs baseline | 0.435 [0.387, 0.489] | 0.566 [0.236, 1.593] | descriptively favorable; primary failure |

The Gaussian exact-score guard also improved rather than merely concentrating
around a worse answer:

| Gaussian comparison | 128-path exact MSE ratio | Fixed-path exact MSE ratio |
|---|---:|---:|
| combined vs baseline | 0.139 [0.124, 0.154] | 0.077 [0.035, 0.162] |

All Gaussian and SIR bundle cells were finite, had positive calibration
temperatures, and satisfied the frozen optimizer-completion rule. Pairing,
CRN identity, independent-noise nonidentity, and exact prefix nesting passed.

## SIR Fixed-Path Score Diagnostics

These are descriptive means with sample standard deviations across ten trained
bundles. They are not exact SIR scores.

| Cell | independent 2048 | CRN 2048 | independent 8192 | CRN 8192 |
|---|---:|---:|---:|---:|
| `T20_j0` | 50.978 (24.149) | 45.851 (30.044) | 48.152 (21.171) | 47.636 (25.749) |
| `T20_j1` | -107.498 (11.225) | -98.602 (34.831) | -112.191 (14.419) | -115.566 (5.423) |
| `T20_j2` | -2.132 (4.345) | -2.038 (3.407) | -1.478 (2.115) | -1.337 (1.456) |
| `T40_j0` | 10.234 (48.253) | 17.482 (33.699) | 47.548 (24.362) | 32.418 (35.909) |
| `T40_j1` | -98.962 (22.117) | -98.929 (13.462) | -111.451 (9.092) | -109.788 (10.409) |
| `T40_j2` | -7.768 (7.167) | -5.821 (3.755) | -6.586 (5.461) | -6.079 (2.392) |
| `T50_j0` | 48.161 (50.502) | 44.709 (31.827) | 63.561 (25.612) | 41.569 (45.456) |
| `T50_j1` | -100.207 (9.815) | -94.665 (20.518) | -108.516 (9.063) | -109.861 (8.414) |
| `T50_j2` | -19.095 (9.005) | -19.131 (5.835) | -17.821 (8.481) | -16.717 (4.064) |

The fixed-path CRN effect is heterogeneous: it reduces variance sharply in
some cells and increases it in others. The joint fixed statistic therefore has
high uncertainty. The 128-path audit bank is the more stable evidence that CRN
has a small SIR effect at `n=8192`; that evidence must not be upgraded into a
fixed-path or score-correctness claim.

## Decision Tables

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Use more training paths in the next classifier-score experiment | passed in Gaussian and SIR | no hard veto | cost versus diminishing returns beyond 8192 | test a path-count ladder with paired bundles | no exact SIR score or default readiness |
| Treat CRN as optional, not a universal repair | Gaussian passed; SIR fixed failed | no implementation veto | strong fixed-path heterogeneity | replicate more fixed SIR paths or use a predeclared multi-path primary target | no claim CRN always lowers variance |
| Do not promote `crn_n8192` as a general SIR solution | SIR two-part primary criterion failed | fixed interval crosses 1 | ten bundles give a wide single-path variance interval | increase bundles only if the fixed-path estimand remains scientifically important | no method ranking or filter validation |

| Inference status | Result |
|---|---|
| Hard veto screen | passed for all 20 bundles and 720 fitted cells |
| Statistically supported ranking | more paths lowers variance; Gaussian CRN lowers variance; SIR CRN is supported only on the `n=8192` audit effect |
| Descriptive-only differences | SIR fixed-path CRN and combined-arm reductions |
| Default readiness | not established |
| Next evidence needed | additional paired bundles for a single fixed-path claim, or a predeclared multi-path estimand if path-averaged stability is the real target |

## Negative-Result Interpretation And Red Team

- Implementation failure: no. Equality tests showed the cached training banks
  exactly match the original per-arm generator; all hard execution gates pass.
- Tuning failure: not established. Frozen V5 controls were intentionally held
  fixed for the causal ablation, so `n=8192` was not retuned and is not claimed
  optimal.
- Diagnostic failure: no. The fixed-path interval is wide because it estimates
  variance from ten bundle replicates at one path; that is uncertainty in the
  stated estimand, not a broken test.
- Evidence against the idea: the evidence weakens the claim that CRN is a
  universal SIR variance repair. It supports more paths much more clearly.
- Strongest alternative explanation: CRN benefits depend on observation path,
  horizon, coordinate, and nonlinear feature fit; the joint audit average can
  improve while one fixed path remains unstable.
- Result that would overturn the conclusion: a larger predeclared paired-bundle
  replication showing the SIR fixed CRN or combined upper interval below 1.
- Weakest evidence: the single fixed-path variance ratio, because ten bundle
  observations yield a broad interval and strong cell heterogeneity.

## Execution Record

| Field | Gaussian | SIR |
|---|---|---|
| Bundles | 10 | 10 |
| Cells per bundle | 36 | 36 |
| Environment | `tftwogpu` | `tftwogpu` |
| GPU route | TensorFlow/XLA, RTX 5080, FP32, TF32 off | TensorFlow/XLA, RTX 5080, FP32, TF32 off |
| Memory policy | verified growth | verified growth |
| Maximum allocator peak | 2,279,525,888 bytes | 2,400,928,256 bytes |
| Sum of bundle wall times | 761.06 seconds | 4,576.47 seconds |
| Campaign wall interval | 03:27:22-03:40:42 HKT | 03:43:35-05:00:47 HKT |
| Aggregation | CPU-only, 5,000 paired bootstrap draws | CPU-only, 5,000 paired bootstrap draws |
| Git commit | `18cfe60984252a9656d1d818c29a2fa86dbc8118` | same |

The capacity and exact-timing diagnostics repaired a measured harness issue:
shared stateless simulation banks were being regenerated for every arm. The
final coordinate-scoped generator reuses identical shared splits and exact
large-bank prefixes while releasing tensors after each coordinate. This
changed runtime, not the scientific design.

## Canonical Artifacts

- Gaussian aggregate:
  `docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/gaussian_full/aggregate_result_cpu_v4.json`
  (`sha256 2bde7f2a6448ac98372bc29355e42a362357d839c205d1c3e9eb8e7c1621e31a`)
- SIR aggregate:
  `docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/sir_full/aggregate_result_cpu_v4.json`
  (`sha256 0fb1d3731abe1023347a4f51fbc852723baf3ab7cf1492cdd5a6563a72bc7aab`)
- Plan:
  `docs/plans/bayesfilter-classifier-score-variance-reduction-v6-plan-2026-08-15.md`
- Plan review:
  `docs/plans/bayesfilter-classifier-score-variance-reduction-v6-plan-review-2026-08-15.md`

Focused final verification: `16 passed` for the variance-reduction and anchored
estimator tests. Aggregation was deliberately CPU-only with
`CUDA_VISIBLE_DEVICES=-1`; TensorFlow emitted a CUDA initialization message
during import, but the aggregate artifact records the CPU-only policy and no
GPU computation was used.

## Restart State

The V6 question is closed. On restart, do not rerun the 20 completed bundles.
Use the two `aggregate_result_cpu_v4.json` files as the canonical summaries.
The next smallest discriminating experiment is a predeclared path-count ladder
(`2048`, `8192`, and one larger count) using paired bundles, with CRN retained
as a factorial arm rather than assumed as the repair. Decide in advance whether
the target is one fixed observation path or path-averaged estimator stability;
do not mix those estimands after results are visible.

