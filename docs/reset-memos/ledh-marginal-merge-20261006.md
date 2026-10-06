# Active LEDH/Zhao–Cui checkpoint

Question: marginal versus ancestor LEDH likelihoods/scores for PP and SIR d18,
T10/20/40/50, with original author TT and bootstrap references. Branch
sqmc-development; shared correction committed b3f4ca646. User requested commits,
scoped permissions, matched comparisons and remaining reference-harness repairs.
Plan: docs/plans/ledh-zhao-horizon-comparison-20261006.md.
Root: docs/plans/artifacts/ledh-zhao-horizons-20261006-01/.

All64 main GPU FP64/XLA LEDH,32 covariance-only and bootstrap N1008/32768/
131072 x4 checks complete. Three PP rank20 fits and all PP scores complete.
SIR rank20 fit17 has all four likelihoods and T10/20 quadratic scores. T40/50
score parents timed out after finishing the large radius. Repair watcher
session4604 now retries only the missing fixed radius, separate ledger
score-repair-attempts.json. Do not relaunch it or write its ledger. Primary
session15473 runs rank40 fit2 (T10 completed); cap115200sec. Its four score jobs
follow the fit. Original supervisor53905 finished. Do not create another
attempts.json writer. Rank20/40 remain separate; recovered radii are one fit.

Budget172800 aggregate job-seconds, including failed attempts and derivative
checks. Last08:37UTC snapshot used46910sec including active jobs; reserved146933
including full active caps, plus9600sec future rank40 score slots held by repair
watcher. Plan allows the same missing-radius repair for later rank40 timeouts
only within this cap. Fresh parent launches use shared budget accounting.

45 focused tests pass, including separate repair-ledger budget accounting. GPU derivative check002 passes both SIR
policies, max normalized errors7.44e-6 ancestor/6.04e-5 mixture over all steps;
trace value/coordinate0 score exactly equal. No stable analytical defect found.
Mixture moment-safety rejection changes under a small perturbation: local
agreement does not establish global smoothness or statistical score accuracy.

Report007: all completed values/scores, rank-separated author references,
conditional versus between-fit uncertainties, raw author lml, derivative
checks, reference errors and full budget. SIR error remains about255 ancestor/
233 mixture afterT20; even bootstrap N1008 is much closer. PP differences small.
No statistically supported ranking, oracle/default/HMC claim. Rank40 and long
scores remain open. Monograph built614pages, refs/cites settled, pages402–404
visually inspected; documentation-01 preserves baseline/build evidence.

Next: finish source/harness audit, commit tested campaign code/docs and compact
evidence, complete references within budget, write final continuous comparison.
Preserve prior uncommitted replication work and huge local MAT/source trees.
No main merge/push this stage.
