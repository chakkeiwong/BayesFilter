# Adaptive R replication stage1000

30000 records; 8 failed/capped learners. All scheduled labels retained.

Pointwise95% empirical bootstrap intervals use2000 resamples and describe Monte Carlo uncertainty conditional on one data set per dimension.
They do not identify author code or establish full-paper replication, population ranking, simultaneous inference or a new default.

| d | Method | Complete | Ratio mean | Ratio SD [95% interval] | Relative RMSE [95% interval] | Mean final N |
|---|---|---:|---:|---|---|---:|
|5|score_after_k|1000/1000|0.99887|0.067374 [0.064107, 0.070483]|0.067349 [0.0641, 0.070501]|1000|
|5|qr_after_k|1000/1000|0.99988|0.041041 [0.039247, 0.042814]|0.04102 [0.039243, 0.042841]|1000|
|10|score_after_k|1000/1000|0.99678|0.093694 [0.089599, 0.097605]|0.093703 [0.089673, 0.097615]|1000|
|10|qr_after_k|1000/1000|0.99729|0.067746 [0.064599, 0.070687]|0.067767 [0.064686, 0.070713]|1000|
|20|score_after_k|1000/1000|0.98976|0.14425 [0.13815, 0.15072]|0.14454 [0.13861, 0.15081]|1000|
|20|qr_after_k|1000/1000|0.99587|0.10372 [0.099126, 0.10814]|0.10375 [0.099188, 0.10821]|1001|
|40|score_after_k|1000/1000|0.98933|0.22674 [0.21273, 0.24161]|0.22687 [0.21325, 0.24161]|1001|
|40|qr_after_k|1000/1000|1.0003|0.158 [0.15035, 0.16539]|0.15792 [0.15037, 0.16539]|1106|
|80|score_after_k|992/1000|NA|NA [NA, NA]|NA [NA, NA]|1073|
|80|qr_after_k|1000/1000|1.0004|0.1795 [0.17064, 0.1883]|0.17941 [0.17066, 0.18827]|1960|

Observed heuristic promotion vetoes: d80 score_after_k vs bootstrap; d80 score_after_k vs fully_adapted; d80 score_after_k vs current_observation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
|Stage recorded|Complete labels and exact oracle checked|3 observed heuristic comparisons veto promotion|Conditional Monte Carlo/tail uncertainty|Inspect terminal results|Paper replication/default status|

| Inference item | Status |
|---|---|
|Hard veto screen|Source, finite-state, oracle, wiring and label controls passed; candidate failures preserved|
|Statistically supported ranking|No automatic ranking claim; intervals are pointwise only|
|Descriptive-only differences|Observed moments, tails, counts and CPU times; bootstrap intervals conditional on the fixed data|
|Default-readiness|Not evaluated; independent R extensions|
|Next evidence needed|Review tail concentration and intervals; original implementation/data identity remains missing|

Red-team: rare importance-weight events may be missing even when bootstrap intervals are narrow. Check maximum_loss_share, ratio tails and underflow counts.
Particle counts differ between fitters; R times include explanatory score refits. An Eq15/source-faithfulness conclusion does not follow.
