# Covariance-guided mixture master program: completed campaign

All 25 independent-validation jobs completed successfully: 192 filter rows, 48 bootstrap reference runs, eight author-code fits and eight quadratic-score calculations. There were no invalid filter rows, failed jobs or missing reference results. The campaign used 8.29 wall hours and 16.56 aggregate worker-hours, within the 12/24-hour limits.

The large SIR discrepancy remains repaired relative to the old filter on both new datasets. KSC comparisons are favorable on both datasets. LGSSM likelihood errors are descriptively larger, and predator–prey scores are mixed. This is an optional candidate with useful evidence, not a uniformly better filter or a default promotion.

## Actual comparison

All rows below use T=50, N=1008, eight paired particle designs per new dataset, FP64 GPU/XLA and TF32 disabled. Entries are means of individual-run errors. LGSSM uses exact Kalman values/scores, KSC a refined grid, and the nonlinear rows use the pooled rank-20 Zhao–Cui path/quadratic reference. The full report also compares both nonlinear models with the independent bootstrap ladder.

| Model | Data seed | Old mean absolute log-likelihood error | New | Old mean score L2 error | New |
|---|---:|---:|---:|---:|---:|
| lgssm | 26100831 | 0.027987 | 0.047157 | 0.104756 | 0.096283 |
| lgssm | 26100832 | 0.024913 | 0.046382 | 0.086007 | 0.093887 |
| ksc | 26100831 | 1.653558 | 0.150613 | 1.369230 | 0.117371 |
| ksc | 26100832 | 1.106189 | 0.185479 | 1.098802 | 0.162100 |
| predator_prey | 26100831 | 0.073403 | 0.061615 | 0.643835 | 0.827193 |
| predator_prey | 26100832 | 0.134680 | 0.123375 | 0.816316 | 0.650316 |
| sir_d18 | 26100831 | 235.401120 | 1.076124 | 16212.282857 | 44.203409 |
| sir_d18 | 26100832 | 148.268102 | 0.531596 | 14189.008006 | 40.736320 |

The conditional paired 95% intervals exclude zero in the favorable direction for both KSC metrics on both datasets and for SIR likelihood on both datasets. The SIR score interval excludes zero only on dataset 26100832. LGSSM and predator–prey intervals contain zero. These intervals condition on a fixed reference and fixed observations; only two new observation datasets were generated, so no broad ranking is claimed. Larger observed errors against simple comparators veto universal non-deterioration, without establishing statistically significant deterioration in LGSSM or predator–prey.

## Algorithm and implementation

The proposal mixes the transition, a whole-cloud UKF-guided affine map, and conditional-covariance LEDH maps. Importance weights use the actual proposal mixture over all branches and ancestors. Analytical recursive scores differentiate the same finite likelihood with beta frozen after independent pilot calibration. The equal-weight reset restores weighted mean and covariance. Optional bounded orthogonal corrections fit marginal and pairwise third/fourth moments while preserving the first two moments. The principal matched and independent runs use zero optional correction steps; their accuracy does not certify fourth-moment repair.

Shared numerical authorities are `bayesfilter/highdim/covariance_proposal_tf.py`, `covariance_proposal_beta_tf.py` and `covariance_proposal_moments_tf.py`. The implemented instance has common additive Gaussian process noise and dense reset, d < N <= 3000. Canonical admission, large-N streaming, calibrated FP32/TF32 operation and HMC readiness remain separate work. The old/new comparison changes complete algorithm variants and their numerical controls; it does not isolate a single code-line change.

## Programs and evidence

| Stage | Program under docs/benchmarks | Evidence |
|---|---|---|
| Implementation, calibration and initial validation | `run_ledh_covariance_proposal.py` | [Initial results](ledh-covariance-proposal-results-20261008.md), [LaTeX/code audit](ledh-covariance-proposal-audit-20261008.md) |
| Original comparison conditions | `run_ledh_matched_comparison.py` | [Matched results](ledh-matched-comparison-results-20261008.md): one original dataset, four paired designs |
| Frozen independent validation | `run_ledh_independent_validation.py` | [Final results](ledh-independent-validation-results-20261008.md): two new datasets, eight paired designs each |
| Reporting only | `summarize_ledh_independent_validation.py` | Raw likelihood/score CSV, paired errors, reference stability and conditional heuristic comparisons |

The independent supervisor calls the existing filter and author-reference implementations. Frozen tuning and observation hashes were checked at launch and at completion. Every method triplet has identical initial/process tensor hashes. All eight filter workers used the RTX 4080 SUPER, FP64/XLA, TF32 disabled and verified GPU memory growth. Author-code reference jobs deliberately hid GPUs; bootstrap references used the recorded GPU route.

## Reference assessment

The Zhao–Cui computation uses the pinned author path-importance route with current-model callbacks and a local quadratic score fit. It is a numerical reference, not an exact oracle or a fresh replication of the published experiment. The supporting paper/source anchors are Eq.26/Algorithm 4 and `models/full_sol.m:139–206`, commit `80034dccb99eb1d86284a1839b4a12067d13b9da`. Raw importance weights are averaged before taking logs. The author mean-log-weight diagnostic is preserved as a different quantity.

Each nonlinear dataset has two independent rank-20 fits with 100000 paths each. Both regression radii and each fit remain in the final report. SIR target-density parity errors are below 4.6e-12. At the smaller regression radius, maximum heldout log-likelihood residuals are below 8.8e-6; SIR path ESS ranges from 11146 to 18900. These explain numerical stability but cannot bound shared rank/support bias or establish score correctness.

For SIR, Zhao minus bootstrap likelihood differences are -0.143 and +0.128. Their first score coordinates differ by -76.8 and -17.6; the corresponding bootstrap MCSEs are 36.6 and 23.6. Thus the remaining score discrepancy must not be judged against a supposedly exact bootstrap oracle. Between-fit uncertainty with only two fits is weak and can be smaller than conditional path uncertainty. Both uncertainties are displayed separately, and neither includes approximation bias.

## Verification and limitations

The implementation audit maps Algorithms 0–3 to runtime consumers and executable call-chain tests. Its final suite passed 41 tests. The independent-data loader and proposal/reference harness passed 36 focused CPU tests. The final reporting suite passed four tests covering cancellation, likelihood-weighted score pooling, paired seeds and incomplete references. These suites overlap; their counts are not a count of distinct scientific validations. MathDevMCP left formal obligations unverified, as recorded in the audit. Finite differences are derivative diagnostics, not the claim-bearing score path.

The merged 746-page monograph and 25-page standalone paper compile with no unresolved references or citations. The monograph reports 217 overfull-box warnings; the standalone has none. The result table and surrounding prose were checked in the rendered PDFs. The earlier development-only build was 650 pages with 215 warnings; remote main added three chapters. Both sets of build and inspection receipts are preserved under `docs/plans/artifacts/ledh-independent-validation-docs-20261009-final/` and `docs/plans/artifacts/ledh-main-integration-20261009-01/`.

The merged repository passed all 40 focused integration checks on CPU with GPUs intentionally hidden. Two tests initially lacked the local Zhao–Cui author-source cache; both passed after copying the existing campaign cache, without code changes. The bibliography merge preserves all 220 unique citation keys and the publisher-verified corrections to two shared entries. These checks validate integration and document consistency; they do not add independent scientific replications. Final branch synchronization is recorded separately in `/tmp/bayesfilter-ledh-final-sync-20261009.json`.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the candidate optional; complete documentation and Git integration | All planned likelihood/score comparisons available | No implementation/collection failure; conditional comparator losses veto universal promotion | Two datasets; nonlinear reference approximation bias | Examine model-specific proposal geometry and reference convergence before new tuning or broad claims | Universal improvement, canonical/default admission, HMC validity |

| Inference item | Status |
|---|---|
| Hard veto screen | 192/192 valid rows; 25/25 completed jobs; no missing evidence |
| Statistically supported ranking | Conditional paired findings above; no broad model/regime ranking |
| Descriptive-only differences | Raw means, LGSSM losses, mixed predator–prey scores, two-dataset trends and nonlinear reference differences |
| Default readiness | Not evaluated or promoted |
| Next evidence needed | More independent data and nonlinear reference rank/path convergence before a broad ranking |

The strongest alternative explanation is data-specific proposal geometry combined with numerical reference bias. A more accurate reference could alter the residual nonlinear errors, and further independent datasets could reverse favorable comparisons. The weakest element is the small number of observation datasets. Eight particle designs per dataset do not make sixteen independent datasets. Candidate limitations do not reject the covariance-guided research direction.

Plan: `docs/plans/ledh-independent-validation-20261008.md`. Final structured comparison: `docs/plans/artifacts/ledh-independent-validation-20261008-01/report-20261009-011148/comparison.json`. Raw author checkpoints remain local; compact evidence and file hashes are committed.
