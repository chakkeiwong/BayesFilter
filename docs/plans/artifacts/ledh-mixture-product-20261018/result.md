# Gaussian-product moment repair — 18 October 2026

The repaired global consumer still fails the conditional heuristic screen and
is not promoted. Four of the 24 predeclared simultaneous contrasts support
lower score MSE than the matching old fitted-kernel consumer; none supports
improvement over unguided global proposals. Both repaired arms remain
numerically viable. This completes the bounded repair comparison, not the
larger LEDH score-improvement program.

The [pre-run plan](../../ledh-mixture-product-moment-repair-20261018.md) remains
unchanged so its checksum still identifies the executed contract. The
[manifest](attempt-01/manifest.json), [run result](attempt-01/result.json),
[rows](attempt-01/rows.jsonl) and [independent analysis](attempt-01/analysis.json)
preserve the computation. Work remains in checkout `ledh-iapf-execution-20260922`,
branch `ledh-refactor-with-policy-fix`, base
`ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus the recorded source closure.
Unrelated working-tree changes are preserved.

## Target and implemented change

The scientific target is the seven-mixture KSC model's likelihood score at
three fixed two-observation datasets. The computed score is the total
analytical derivative of the finite particle likelihood program with fitted
future messages held fixed. These are different quantities: derivative
checks establish the latter, and the independent model reference measures its
error relative to the former. Equality to the model score is not established.

The optional `mixture_product_moments` global consumer computes the exact mean
and covariance of the product of the current-observation Gaussian and the
frozen Gaussian-mixture future message. It retains component covariance and
between-component uncertainty, including derivatives of normalized component
responsibilities. The shared TensorFlow routine supports multiple dimensions
and batched inputs. Local maps, proposal-density correction and reset retain
their existing definitions. `positive_information` remains the default.

This is exact conditioning of an approximate Gaussian, not an exact nonlinear
filtering or smoothing update. Both product mean and covariance change, so
score differences cannot be attributed solely to between-component variance.
The optional mixture route is not canonical UKF/dual-cap/Contract-E admission,
original KDM training, TF32 readiness or HMC evidence.

## Engineering and numerical evidence

Twenty focused CPU tests pass in 29.06 seconds. They cover independent
multidimensional moment/tangent references, invalid covariance rejection,
inactive and single-component parity, unchanged identity/local maps and an
executable call-chain check from `make_filter` to the shared routine with a
finite-program directional derivative check. An extended multi-component
global-map moment assertion subsequently passes with the two selected map
tests in 5.43 seconds. Python compilation passes. Logs:
[focused suite](focused-tests-01.log), [extended map checks](map-reference-tests-01.log).

The GPU/XLA attempt completes all 24 fits and 2,304 rows. The independent
terminal audit passes 13,824 trace checks, all 74 source hashes, expected
row/fit/cloud pairing and 24 two-step finite-difference checks. The largest
absolute derivative discrepancy is 7.5112631e-7 and passes the predeclared
mixed tolerance. No numerical continuation veto fires. Logs:
[attempt](attempt-01.log), [terminal audit](analysis-01.log).

The manuscript proposition derives product moments, total derivatives and
positive definiteness. MathDev verifies the scoped scalar covariance and mean
derivative identities. Its full implementation audit is **not a certificate**:
all 19 proof obligations remain inconclusive with `parser_limit`. The corrected
operation audit finds the required Cholesky, covariance and quadratic-form
operations; shapes are supported and code contracts consistent. Its aggregate
`mismatch` comes from lexical matching of 80 LaTeX/prose tokens against Python
identifiers, including `let`, `every`, `align` and `qquad`, rather than a
refuted identity. The first request also treated inline code as a pathname;
the second used unsupported operation category names. All responses remain in
[initial audit](mathdev-audit.json), [scoped audit](mathdev-scoped-audit.json)
and [corrected audit](mathdev-corrected-audit.json). Manual formula review and
independent executable checks support the implementation on the stated
fixtures; automated certification of the general matrix proof remains open.

## Conditional score evidence

Each condition has eight independent fitting clusters and eight paired clouds
per fit. Fitting controls remain 256 components, 16 transition replicates and
bandwidth factor 0.5. Counts, beta arms, data and numerical controls match.
Fresh seeds are disjoint from the previous campaign. The three datasets were
already inspected; they are not new data holdouts. OT epsilon 1 / five
iterations is the owner's untuned comparator.

The following MSEs are descriptive. Scores use coordinates Phi^-1(gamma)
and log(beta). All twelve arms and six heuristic comparisons are preserved
conditionally in the machine-readable analysis.

| Fixed condition | Score coordinate | Old fitted/global | Product fitted/global | Unguided/global |
| --- | --- | ---: | ---: | ---: |
| Ordinary | Phi^-1(gamma) | 0.00315993 | 0.00027810 | 0.00033313 |
| Ordinary | log(beta) | 0.16439711 | 0.03589405 | 0.02872097 |
| Negative future tail | Phi^-1(gamma) | 0.00259606 | 0.00047951 | 0.00051886 |
| Negative future tail | log(beta) | 0.24072748 | 0.05401497 | 0.06357852 |
| Positive future tail | Phi^-1(gamma) | 0.03683554 | 0.05598996 | 0.04748061 |
| Positive future tail | log(beta) | 1.54878813 | 0.79766265 | 0.55865533 |

Paired squared-error differences are averaged over clouds within each fit,
then assessed with Student-t intervals with seven degrees of freedom and
Bonferroni coverage over all 24 contrasts. The critical value is 4.7500743386.
Four intervals exclude zero, all for product/global minus the matching old
fitted/global consumer:

| Fixed condition | Coordinate | MSE difference | Simultaneous 95% interval |
| --- | --- | ---: | --- |
| Ordinary | Phi^-1(gamma) | -0.00288184 | [-0.00467766, -0.00108601] |
| Ordinary | log(beta) | -0.12850306 | [-0.24958800, -0.00741812] |
| Negative future tail | Phi^-1(gamma) | -0.00211655 | [-0.00416811, -0.00006499] |
| Negative future tail | log(beta) | -0.18671251 | [-0.33268640, -0.04073861] |

The other 20 intervals include zero. None supports improvement over unguided
global proposals; none supports harm under this interval criterion. With
only eight fitting clusters, approximate normality materially limits coverage.
There is no statistically supported overall ordering of the viable methods.

Product/global has seven descriptive losses to the six constructed heuristics
across condition/coordinate pairs: ordinary log(beta) against unguided and
Gaussian-future global; both negative-tail coordinates against Gaussian-future
global; both positive-tail coordinates against unguided global; and
positive-tail log(beta) against Gaussian-future global. Product/equal has 21
losses. These predeclared promotion vetoes are not statistically supported
harm claims or numerical continuation vetoes.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain optional global repair; no promotion | 4/12 global contrasts improve against old fitted consumer; required all-comparator improvement fails | Seven heuristic losses; numerical checks pass | Eight clusters, known data; mean and covariance both change | Compare exact and fitted messages under the same product consumer | Benefit over unguided global or default readiness |
| Equal-mixture arm remains numerically viable only | 0/12 contrasts support improvement | 21 heuristic losses; numerical checks pass | Local-map/reset effects remain entangled | Keep as a controlled comparison if needed | Ranking from descriptive MSEs |
| Continue research direction | Bounded conditional evidence against old consumer | No continuation veto | Fit error versus Gaussian/reset approximation | Plan exact-message/product-consumer control first | That more fitting capacity is appropriate |

| Inference category | Status |
| --- | --- |
| Hard veto screen | No numerical invalidity; descriptive heuristic promotion veto for both repaired arms |
| Statistically supported ranking | Four conditional improvements against matching old global consumer under the declared approximate intervals; none against unguided global; no overall ranking |
| Descriptive-only differences | Remaining MSE, bias, variance, heuristic losses and runtime comparisons |
| Default-readiness | Not established; runtime default and canonical admission unchanged |
| Next evidence needed | Exact and fitted messages under the same repaired consumer with fresh paired clouds; later fresh data and scope-specific tuning for broader claims |

## Execution and budget

The run uses `tftwogpu`, TensorFlow 2.20.0-dev0+selfbuilt, FP64, XLA on, TF32
off, trusted GPU access and GPU:0 (RTX 4080 SUPER). Memory growth is verified
before initialization on both visible GPUs. This is an explicit reference
exception to the production TF32 direction. Commands from the LEDH worktree:

```sh
env TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=. /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_mixture_product_diagnostic.py --output docs/plans/artifacts/ledh-mixture-product-20261018/attempt-01 --budget-seconds 600
env CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/analyze_ledh_mixture_product_diagnostic.py --run docs/plans/artifacts/ledh-mixture-product-20261018/attempt-01
```

The stage closes after one of two permitted launches, using
407.033164738/900 GPU-process wall seconds. CPU reference time is 0.618702775
seconds (also included in process wall time); analysis takes 0.943810340
seconds, for 1.562513115/180 CPU diagnostic seconds. Focused tests use
34.49/300 seconds. TensorFlow allocator current/peak bytes are 45,824/517,888;
these are allocator measurements, not total process GPU memory. No process
is running and unused budget does not transfer to another campaign.

The [manuscript build and rendered-page review](document-build/review.md)
passes in 3.36/300 build seconds. The refreshed PDF has 75 pages. Its new
proof, result table, next comparison and audit limitations were visually
inspected; the final build has no warnings or unresolved references.

## Red-team and next discriminating question

The strongest alternative explanation is that the product consumer fixes one
approximation while fitted future density, current-observation Gaussian and
reset still dominate score error. Existing exact-future controls use the old
positive-information consumer, so this experiment cannot separate those
possibilities. Mean changes also prevent covariance-only causal attribution.
The weakest statistical evidence is the eight-cluster normal approximation
on three already-known datasets.

Next, prepare a bounded comparison of exact seven-component future messages
and frozen fitted messages **under the same product-moment consumer**, retaining
unguided global and old-consumer controls. Use fresh paired clouds and
distinguish saved-fit mechanism diagnostics from confirmation requiring fresh
independent fits. Exact-message/product scores that still lose to unguided
global would overturn the explanation that residual fitting error alone
causes the remaining failure. The next study needs its own evidence contract
and budget; increasing kernel capacity is not justified yet.
