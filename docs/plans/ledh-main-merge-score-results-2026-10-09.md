# Main integration and saved LEDH score results, 2026-10-09

The exact requested `ledh-matched-comparison-20261008-01/comparison.json` is already committed to **local main**, at `285b43e111124b0f3337f81032905591b9a54324`. Its Git blob is `2e0ce61c6cff22b82f07c85ab1494f41615f4d0b`, identical to the file in `/home/chakwong/BayesFilter-SQMC`. It was introduced by `a43d0e3c3ff7c2dcab56bab7ca7298a8e8c607e1`. After a fresh fetch, `origin/main` remains at `b1ccb8678c1ca68811eaa301bd4e1fd368e4bc23` and does not contain it. No push was performed.

## Integration and verification

Local main was merged into `ledh-graceful-failure-execution-20260917` in `7ed1d1ecd339a2a9a6314a430f11d436a6e4aad1`. The merge preserves the branch's shared batch executor, structural-transition support, failure diagnostics and precision protections while integrating main's shared Gaussian mixture density and optional covariance-guided proposal implementation.

The first focused CPU-only attempt had 38 passes and nine failures. All failures arose during graph construction: the imported proposal expected a scalar validity flag, whereas this branch's shared quadrature routine returns validity per cloud. Commit `10884528ca8b1a7463895256d30b9655d69c0955` applies `tf.reduce_all` to that flag at the proposal consumer. It changes neither means, covariances nor their tangents. All 20 affected proposal tests then passed. With the 27 unaffected checks, **47 distinct focused checks pass**, covering analytical-score finite differences, mixture weights, consumer wiring, precision diagnostics, proposal validity and reporting. Both commit hooks also passed all three oracle-contract checks.

Tests exercised the merged working tree, including preserved uncommitted campaign edits. CUDA was intentionally hidden for the focused tests; this is engineering verification, not a new GPU or scientific comparison. Commands, source hashes, environments, logs and timings are in the [run manifest](artifacts/ledh-main-merge-20261009T053052Z/manifest.json). The original five-file staged patch is byte-for-byte preserved; nine originally unstaged files remain unstaged after conflict resolution; no previous untracked path is missing. Recovery stash `1f5dc7092b208bc1418ab4257903d9911e239ddc` and both patch layers are retained. No conflict remains.

## Saved scientific evidence

The requested October 8 comparison used one fixed dataset per model and four paired designs. All four paired score-difference intervals include zero. Its lower observed score errors alone do not establish a ranking.

The later study also merged from main used two fresh datasets per model, eight paired designs per dataset, T=50, N=1008, FP64 GPU/XLA, and TF32 disabled. All 25 jobs completed with 192 valid filter rows and no missing references. Several controls differ between the complete filter variants, so this does not isolate one code change.

| Model | Mean score L2 error, old → new, dataset 1 | Dataset 2 | Interpretation against the saved reference |
|---|---:|---:|---|
| LGSSM | 0.104756 → 0.096283 | 0.086007 → 0.093887 | Mixed; both paired intervals include zero |
| KSC SV | 1.369230 → 0.117371 | 1.098802 → 0.162100 | Both conditional intervals favor the candidate; likelihood-error intervals also favor it |
| Predator–prey | 0.643835 → 0.827193 | 0.816316 → 0.650316 | Mixed; both paired intervals include zero |
| SIR, state dimension 18 | 16212.282857 → 44.203409 | 14189.008006 → 40.736320 | Large descriptive reductions; only dataset 2's score interval excludes zero; likelihood-error intervals favor the candidate on both datasets |

LGSSM uses exact Kalman values/scores, KSC refined integration, and the other models a pooled rank-20 Zhao–Cui path/quadratic reference, with separate bootstrap comparisons also recorded. Conditional intervals treat numerical references as fixed. Nonlinear reference bias and the small number of datasets limit broader inference. LGSSM mean likelihood errors increase by about 0.02 with intervals including zero; material deterioration is not established.

The saved heuristic verdict is `descriptive_losses_present_promotion_veto`. It blocks uniform promotion under the study's contract; it does not establish statistically significant inferiority or reject further research. The new filter remains optional. These comparisons contain no range-bearing/M13 or posterior-sampling result and do not establish FP32/TF32 default readiness.

The original objective remains **testing surrogate HMC**. This likelihood/score work addresses a prerequisite. The existing `marginal_weight_campaign.py` forwards an importance-weight policy but does not call the covariance-proposal filter. Merging the optional implementation therefore does not automatically change the existing HMC target or complete posterior evaluation.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete bounded merge and evidence review | Main integrated; preservation and focused tests pass | Initial shape defect repaired | GPU behavior of merged working tree not rerun | Use integrated branch for a specifically scoped continuation of the existing surrogate-HMC master program | Posterior correctness or HMC convergence |
| Retain optional candidate | Conditional KSC and specified SIR findings favorable | Comparator losses block uniform promotion | Two datasets and nonlinear reference bias | Resolve target scope, tuning and actual consumer wiring before another campaign | Universal improvement or canonical/default admission |

| Inference item | Status |
|---|---|
| Hard veto screen | Saved study has no invalid rows or missing evidence; integration defect repaired |
| Statistically supported ranking | Conditional KSC and specified SIR findings against fixed references only |
| Descriptive-only differences | Raw means, mixed LGSSM/predator–prey scores, two-dataset trends and nonlinear reference differences |
| Default readiness | Not evaluated or promoted |
| Next evidence needed | More datasets and reference convergence for broader rankings; eligible tuned consumer and posterior sampling for the HMC question |

The strongest alternative explanation is dataset-specific proposal geometry combined with reference approximation bias. Better references or additional datasets could alter the ranking. The weakest scientific evidence is the number of independent observation datasets; engineering tests do not resolve it. No new scientific experiment or long sampling run was launched.

Sources: [plan](ledh-main-merge-score-check-2026-10-09.md), [requested comparison](artifacts/ledh-matched-comparison-20261008-01/comparison.json), [independent comparison](artifacts/ledh-independent-validation-20261008-01/report-20261009-011148/comparison.json), [upstream result note](../benchmarks/ledh-covariance-proposal-master-summary-20261009.md).
