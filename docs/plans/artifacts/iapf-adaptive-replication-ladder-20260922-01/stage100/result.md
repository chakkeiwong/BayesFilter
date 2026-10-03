# Adaptive R replication stage100

3000 records; 0 failed/capped learners. All scheduled labels retained.

Pointwise95% empirical bootstrap intervals use2000 resamples and describe Monte Carlo uncertainty conditional on one data set per dimension.
They do not identify author code or establish full-paper replication, population ranking, simultaneous inference or a new default.

| d | Method | Complete | Ratio mean | Ratio SD [95% interval] | Relative RMSE [95% interval] | Mean final N |
|---|---|---:|---:|---|---|---:|
|5|score_after_k|100/100|1.0005|0.063566 [0.053845, 0.071463]|0.06325 [0.053871, 0.071607]|1000|
|5|qr_after_k|100/100|0.9981|0.042577 [0.03667, 0.047964]|0.042406 [0.036712, 0.048019]|1000|
|10|score_after_k|100/100|0.98867|0.086066 [0.073991, 0.097001]|0.086381 [0.075148, 0.097812]|1000|
|10|qr_after_k|100/100|1.0089|0.073551 [0.062678, 0.083632]|0.073716 [0.063064, 0.084565]|1000|
|20|score_after_k|100/100|0.99151|0.15382 [0.13421, 0.17061]|0.15328 [0.13408, 0.17108]|1000|
|20|qr_after_k|100/100|1.0082|0.10213 [0.088498, 0.11386]|0.10195 [0.089033, 0.11421]|1000|
|40|score_after_k|100/100|1.0062|0.25503 [0.21599, 0.29307]|0.25383 [0.21673, 0.29489]|1000|
|40|qr_after_k|100/100|1.0356|0.14717 [0.12352, 0.17127]|0.1507 [0.12631, 0.17828]|1140|
|80|score_after_k|100/100|0.98107|0.36345 [0.28637, 0.43399]|0.36212 [0.29344, 0.4335]|1060|
|80|qr_after_k|100/100|0.99641|0.16588 [0.1398, 0.18936]|0.16509 [0.14138, 0.18917]|1940|

Observed heuristic promotion vetoes: none in this sample.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
|Stage recorded|Complete labels and exact oracle checked|0 observed heuristic comparisons veto promotion|Conditional Monte Carlo/tail uncertainty|Continue to next frozen stage|Paper replication/default status|

| Inference item | Status |
|---|---|
|Hard veto screen|Source, finite-state, oracle, wiring and label controls passed; candidate failures preserved|
|Statistically supported ranking|No automatic ranking claim; intervals are pointwise only|
|Descriptive-only differences|Observed moments, tails, counts and CPU times; bootstrap intervals conditional on the fixed data|
|Default-readiness|Not evaluated; independent R extensions|
|Next evidence needed|Review tail concentration and intervals; original implementation/data identity remains missing|

Red-team: rare importance-weight events may be missing even when bootstrap intervals are narrow. Check maximum_loss_share, ratio tails and underflow counts.
Particle counts differ between fitters; R times include explanatory score refits. An Eq15/source-faithfulness conclusion does not follow.
