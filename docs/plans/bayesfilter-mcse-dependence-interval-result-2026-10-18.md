# MCSE dependence and interval diagnostic: terminal result

18 October 2026. **Completed and independently audited; no policy promoted.**
The current precision policy does not support a broad 95% interval-calibration
claim on this stationary stress suite. Twenty-two of 108 fixed-endpoint method
screens and three of 108 stopped-endpoint method screens flagged undercoverage.
These findings concern posterior precision assessment, not acceptance/health
tuning membership. The SIR campaign and all numerical defaults remain unchanged.

## Design, provenance and validity

The controlling plan is [the interval-study plan](bayesfilter-mcse-dependence-interval-study-2026-10-18.md).
Let `A=artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/mcse-dependence-interval-20261018-01`.
The source-bound run manifest is `A/attempt-01/manifest.json`, terminal output
`A/attempt-01/result.json`, independent audit `A/audit-01.json`, and descriptive
reader `A/summary-01.json`. Exact commands, source hashes and environment are
preserved in the manifest, `A/launch-submission-01.json` and `A/frozen-sources-01.json`.
The manifest's initial `running` field is historical; terminal status comes
from the result, audit and supervisor, which all completed successfully.

Checkout commit: `139cd11fe7fea37e971f2197a544cb26ccd39c44`. The immutable numerical
authority is the `e68cdf1c4` source with the reviewed precision overlay under
`N5/generic-adaptation/priced-comparison-01/precision-repair-01/source`.
Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; TensorFlow 2.20.0,
TFP 0.25.0, NumPy 2.1.3 and SciPy
1.17.1. NumPy/SciPy are independent-reference/diagnostic
tools only. GPUs were intentionally hidden before import, with two TF/OMP
threads and XLA on. No HMC transitions or transport training were performed.
Synthetic data and seed 75117 are fully specified by the manifest.

All six conditions completed 128/128 replications: Gaussian and `exp(Gaussian)`
laws at latent AR(1) correlations 0, +.99 and -.99. Each record has four
stationary chains and ten looks, 1000 through 10000 draws per chain.
Gaussian/lognormal laws share latent paths: 768 condition records correspond
to **384 independent latent four-chain units**, not 768 independent units.
Quantities are the coordinate, its square, and `abs(coordinate)>2`.
The six arms, common thresholds and two longer-batch hypotheses were frozen
before launch. No seed, threshold, batch size or arm was selected on this result.

The independent audit passed all 768 records, 84 numerical source dependencies,
24 covariance-oracle checks and 231 screen definitions. It regenerated streams,
means, batch components, controls and first-pass histories. It **did not replay
the complete TensorFlow ESS/R-hat implementation independently**. The frozen
suite passed 89 tests and the supervisor passed five fault tests before launch.
The terminal reader initially rejected bitwise differences between independent
bisection and SciPy beta intervals (maximum observed `4.490852134608758e-13`).
A reader-only repair used the already audited absolute tolerance `1e-12` for
these intervals; every other screen field remains exact. No scientific output,
membership, stopping decision or estimator changed.

## Coverage screens and denominators

The prespecified family uses exact-binomial miss-probability intervals with
Bonferroni alpha `.05/231`. A lower endpoint above `.05` flags undercoverage.
The family contains 108 fixed method screens, 108 conditional stopped method
screens and 15 fixed control screens. Dependence between arms/laws does not
require independence for the Bonferroni bound. Nonflags do not establish
calibration, and these are not paired superiority tests or anytime-valid intervals.

| Endpoint/type | Screens | Flagged | Interpretation |
| --- | ---: | ---: | --- |
| Fixed method, 10000 draws/chain | 108 | 22 | Evidence against broad fixed-endpoint calibration |
| Stopped method | 108 | 3 | Conditional first-passing-look failures; nonstoppers retained separately |
| Fixed controls | 15 | 0 | No flagged control; calibration and interval-shape adequacy remain unproved |

Every flagged fixed screen below has 128 valid intervals and no missing record.

| Law | Latent rho | Arm | Quantity | Misses/valid | Family-adjusted miss-probability interval |
| --- | ---: | --- | --- | ---: | --- |
| gaussian | +0.99 | `lugsail` | `coordinate` | 18/128 | [0.051592, 0.283049] |
| gaussian | +0.99 | `lugsail_max_batch_means` | `coordinate` | 18/128 | [0.051592, 0.283049] |
| gaussian | +0.99 | `batch_means` | `coordinate` | 32/128 | [0.126609, 0.410345] |
| lognormal | +0.99 | `lugsail` | `square` | 28/128 | [0.103719, 0.375398] |
| lognormal | +0.99 | `lugsail` | `abs_gt_2` | 18/128 | [0.051592, 0.283049] |
| lognormal | +0.99 | `lugsail_max_batch_means` | `square` | 28/128 | [0.103719, 0.375398] |
| lognormal | +0.99 | `lugsail_max_batch_means` | `abs_gt_2` | 18/128 | [0.051592, 0.283049] |
| lognormal | +0.99 | `batch_means` | `coordinate` | 30/128 | [0.115040, 0.392992] |
| lognormal | +0.99 | `batch_means` | `square` | 30/128 | [0.115040, 0.392992] |
| lognormal | +0.99 | `batch_means` | `abs_gt_2` | 30/128 | [0.115040, 0.392992] |
| lognormal | +0.99 | `autocorrelation` | `square` | 24/128 | [0.081891, 0.339411] |
| lognormal | +0.99 | `lugsail_twice_sqrt_cap20` | `square` | 24/128 | [0.081891, 0.339411] |
| lognormal | +0.99 | `lugsail_max_batch_means_twice_sqrt_cap20` | `square` | 24/128 | [0.081891, 0.339411] |
| gaussian | -0.99 | `batch_means` | `square` | 21/128 | [0.066331, 0.311629] |
| gaussian | -0.99 | `batch_means` | `abs_gt_2` | 20/128 | [0.061320, 0.302198] |
| lognormal | -0.99 | `lugsail` | `square` | 24/128 | [0.081891, 0.339411] |
| lognormal | -0.99 | `lugsail_max_batch_means` | `square` | 24/128 | [0.081891, 0.339411] |
| lognormal | -0.99 | `batch_means` | `coordinate` | 22/128 | [0.071433, 0.320973] |
| lognormal | -0.99 | `batch_means` | `square` | 29/128 | [0.109348, 0.384226] |
| lognormal | -0.99 | `autocorrelation` | `square` | 24/128 | [0.081891, 0.339411] |
| lognormal | -0.99 | `lugsail_twice_sqrt_cap20` | `square` | 23/128 | [0.076621, 0.330232] |
| lognormal | -0.99 | `lugsail_max_batch_means_twice_sqrt_cap20` | `square` | 23/128 | [0.076621, 0.330232] |

The three stopped-screen flags all concern the IID lognormal square:

| Law | Latent rho | Arm | Quantity | Misses/valid | Family-adjusted miss-probability interval |
| --- | ---: | --- | --- | ---: | --- |
| lognormal | +0 | `lugsail` | `square` | 19/128 | [0.056405, 0.292673] |
| lognormal | +0 | `lugsail_max_batch_means` | `square` | 19/128 | [0.056405, 0.292673] |
| lognormal | +0 | `batch_means` | `square` | 20/128 | [0.061320, 0.302198] |

At rho=0 every arm stopped in all 128 replications of each law, mostly at
1000 draws, with some first passes at 2000–4000. At rho=+.99 neither law
stopped in any arm (0/128 for each cell). At rho=-.99 Gaussian also had 0/128
stops in every arm. The lognormal negative-correlation batch-means arm stopped
in 12/128 records: two at 8000, three at 9000 and seven at 10000; all other
arms had 0/128 stops. The 12 stopped records missed none of the three truths,
but the pointwise 95% upper miss-probability bound is `.2646484694`.
They do not validate stopped coverage. These zero-stop cells are absent
evidence about stopped intervals, not successful screens.

Unavailable fixed intervals are distinct from nonstopping. All other fixed
method/quantity cells had 128/128 available intervals:

| Law | Latent rho | Arm | Quantity | Unavailable/planned |
| --- | ---: | --- | --- | ---: |
| gaussian | +0 | `lugsail_twice_sqrt_cap20` | `coordinate` | 1/128 |
| lognormal | +0 | `lugsail_twice_sqrt_cap20` | `coordinate` | 1/128 |
| gaussian | -0.99 | `lugsail` | `coordinate` | 128/128 |
| gaussian | -0.99 | `autocorrelation` | `coordinate` | 72/128 |
| gaussian | -0.99 | `lugsail_twice_sqrt_cap20` | `coordinate` | 8/128 |

## Variance scale and interval-shape controls

Estimated variance divided by the exact finite-length variance of the pooled
mean isolates scale error at the fixed endpoint. The following summaries are
descriptive; there is no supported ranking or tail comparison. Full distributions
remain in the result, and `summary-01.json` records quantile conventions.

| Law | Latent rho | Quantity | Arm | Available | Median ratio | Mean ratio |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| gaussian | +0.99 | `coordinate` | `lugsail` | 128 | 0.576336 | 0.581984 |
| gaussian | +0.99 | `coordinate` | `lugsail_max_batch_means` | 128 | 0.576336 | 0.581984 |
| gaussian | +0.99 | `coordinate` | `batch_means` | 128 | 0.361875 | 0.364386 |
| gaussian | +0.99 | `coordinate` | `autocorrelation` | 128 | 1.012411 | 1.045736 |
| gaussian | +0.99 | `coordinate` | `lugsail_twice_sqrt_cap20` | 128 | 0.847738 | 0.856738 |
| gaussian | +0.99 | `coordinate` | `lugsail_max_batch_means_twice_sqrt_cap20` | 128 | 0.847738 | 0.856738 |
| lognormal | +0.99 | `square` | `lugsail` | 128 | 0.272304 | 0.509472 |
| lognormal | +0.99 | `square` | `lugsail_max_batch_means` | 128 | 0.272304 | 0.509472 |
| lognormal | +0.99 | `square` | `batch_means` | 128 | 0.189707 | 0.348124 |
| lognormal | +0.99 | `square` | `autocorrelation` | 128 | 0.354539 | 0.592204 |
| lognormal | +0.99 | `square` | `lugsail_twice_sqrt_cap20` | 128 | 0.342161 | 0.575247 |
| lognormal | +0.99 | `square` | `lugsail_max_batch_means_twice_sqrt_cap20` | 128 | 0.342161 | 0.575789 |
| lognormal | -0.99 | `square` | `lugsail` | 128 | 0.367329 | 0.692389 |
| lognormal | -0.99 | `square` | `lugsail_max_batch_means` | 128 | 0.367329 | 0.695084 |
| lognormal | -0.99 | `square` | `batch_means` | 128 | 0.265540 | 0.484139 |
| lognormal | -0.99 | `square` | `autocorrelation` | 128 | 0.400751 | 0.731044 |
| lognormal | -0.99 | `square` | `lugsail_twice_sqrt_cap20` | 128 | 0.406291 | 0.781120 |
| lognormal | -0.99 | `square` | `lugsail_max_batch_means_twice_sqrt_cap20` | 128 | 0.414720 | 0.785956 |

Substantial variance underestimation is visible in several strongly dependent
cells. Longer batches do not remove all undercoverage or unavailable estimates;
the maximum safeguard does not remove the lognormal-square flags. These
observations support a finite-sample precision-assessment problem, rather than
a claim that a successful acceptance tuner guarantees adequate estimation.

Exact-variance normal controls cover coordinate and square; the four-chain
Student control covers only Gaussian coordinates, where its normal independent
chain-mean premise holds. Neither is a stopping input or a replacement policy.
Their fixed-endpoint miss counts are:

| Law | Latent rho | Control | Quantity | Misses/valid |
| --- | ---: | --- | --- | ---: |
| gaussian | +0 | `exact_variance_normal` | `coordinate` | 10/128 |
| gaussian | +0 | `exact_variance_normal` | `square` | 9/128 |
| gaussian | +0 | `gaussian_chain_mean_student` | `coordinate` | 7/128 |
| lognormal | +0 | `exact_variance_normal` | `coordinate` | 15/128 |
| lognormal | +0 | `exact_variance_normal` | `square` | 8/128 |
| gaussian | +0.99 | `exact_variance_normal` | `coordinate` | 9/128 |
| gaussian | +0.99 | `exact_variance_normal` | `square` | 3/128 |
| gaussian | +0.99 | `gaussian_chain_mean_student` | `coordinate` | 3/128 |
| lognormal | +0.99 | `exact_variance_normal` | `coordinate` | 8/128 |
| lognormal | +0.99 | `exact_variance_normal` | `square` | 4/128 |
| gaussian | -0.99 | `exact_variance_normal` | `coordinate` | 10/128 |
| gaussian | -0.99 | `exact_variance_normal` | `square` | 5/128 |
| gaussian | -0.99 | `gaussian_chain_mean_student` | `coordinate` | 9/128 |
| lognormal | -0.99 | `exact_variance_normal` | `coordinate` | 6/128 |
| lognormal | -0.99 | `exact_variance_normal` | `square` | 1/128 |

None of these controls flagged after family adjustment. With 128 replications
and wide adjusted intervals this does not establish normal interval-shape
adequacy. In particular, there is no matched exact-variance control at random
stopping times, no exact control for the event quantity, and no justification
for extrapolating the Gaussian Student result to nonlinear transformed means.

## Decisions and remaining uncertainty

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this diagnostic | Complete planned records and terminal audit passed | No source/artifact veto | Full ESS/R-hat not independently replayed | Preserve result and settle once | Sampler or posterior release validity |
| Reject broad precision calibration claim | 25 prespecified undercoverage flags | Coverage limitation affects precision claims | Tail shape, estimator variability and stopping contribute differently | Design a source-grounded precision-method comparison | Universal failure of lugsail or HMC |
| Leave estimator/default unchanged | No selection/promotion criterion existed | Several longer-batch arms still fail coverage/availability | No validated replacement | Keep experimental alternatives diagnostic until separately validated | A longer-batch or maximum arm is superior |
| Continue SIR | Separate frozen scientific contract remains intact | This diagnostic is no continuation veto | Model-specific precision/delivery remains unsettled | Let the existing controller finish and audit each method | Changing tuning membership or retrying only failed members |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | Scientific attempt and integrity audit passed; declared coverage limitations remain |
| Statistically supported ranking | None; only prespecified one-arm undercoverage tests were performed |
| Descriptive-only differences | Variance-ratio summaries, stop counts and observed arm differences |
| Default-readiness | Not established for an alternative; broad current-policy calibration is unsupported |
| Next evidence needed | Source-grounded finite-sample/long-run-variance diagnosis, independently audited ESS/R-hat, then fresh fixed/stopped coverage and nonlinear confirmation |

The next precision phase must distinguish variance-scale error from interval-
shape error and stopping selection. It should first use cheap derivations and
saved paths, justify any automatic batch choice or confidence correction, and
declare a fresh-data confirmation criterion before changing runtime defaults.
Increasing the sample cap alone is not a validated repair. Broader matched
model/seed comparisons, an actual external consumer and scoped release audit
remain separate master-program obligations.

Post-run red-team: the strongest alternative explanation is a finite-sample
interaction among heavy transformed tails, strong dependence and normal
intervals, rather than a coding defect or universal estimator failure.
Exact-variance controls weaken a pure variance-free normal-shape explanation
for these fixed endpoints but do not rule it out or isolate stopping effects.
A fresh, source-grounded policy passing predeclared fixed and actual-stopping
calibration would overturn the current lack of a validated replacement.
The weakest evidence concerns stopped dependent cells, where there were few
or no stops. Stationary initialization supplies no evidence about burn-in.

## Cost and settlement

The worker plus independent audit consumed `3462.895169735886` inclusive
supervisor seconds; preflight consumed `53.66638870770112`, giving the original
settlement charge `3516.5615584435873` out of 7200 seconds. Submission overhead
`.041790587361902` is separate and charged once. The failed/successful terminal
summary commands consumed `.039637597` and `2.787860932` observed command
seconds, respectively; the successful summary's internal elapsed time is
nested and not added again. This closeout's measured reporting cost is recorded
in `A/reconciled-settlement-01.json`.

The original `final-settlement-01.json` remains unchanged: its historical
balance predates the concurrent saved-nonlinear replay. The reconciled receipt
subtracts that replay's `131.3668131614104` charge and terminal reporting once,
with no new grant or SIR-cap change. The separate 3600-second hold and historical
1579.592954-second balance are excluded. The reconciled receipt is authoritative
for the released pool after both diagnostics; it is not a second release.
