# Adaptive R replication stage300

9000 records; 2 failed/capped learners. All scheduled labels retained.

Pointwise95% empirical bootstrap intervals use2000 resamples and describe Monte Carlo uncertainty conditional on one data set per dimension.
They do not identify author code or establish full-paper replication, population ranking, simultaneous inference or a new default.

| d | Method | Complete | Ratio mean | Ratio SD [95% interval] | Relative RMSE [95% interval] | Mean final N |
|---|---|---:|---:|---|---|---:|
|5|score_after_k|300/300|0.99362|0.063211 [0.057298, 0.069113]|0.063427 [0.058057, 0.06921]|1000|
|5|qr_after_k|300/300|0.99906|0.041162 [0.038244, 0.043871]|0.041104 [0.038264, 0.04387]|1000|
|10|score_after_k|300/300|0.99207|0.09257 [0.084925, 0.099441]|0.092755 [0.085299, 0.099715]|1000|
|10|qr_after_k|300/300|0.99753|0.070238 [0.064266, 0.07547]|0.070165 [0.06427, 0.075553]|1000|
|20|score_after_k|300/300|0.98419|0.14683 [0.1342, 0.15915]|0.14744 [0.13504, 0.15953]|1000|
|20|qr_after_k|300/300|1.0044|0.10231 [0.093718, 0.1106]|0.10224 [0.093832, 0.11072]|1000|
|40|score_after_k|300/300|1.0121|0.24043 [0.21918, 0.26071]|0.24033 [0.21911, 0.26173]|1000|
|40|qr_after_k|300/300|1.0189|0.16088 [0.14739, 0.17551]|0.16172 [0.14781, 0.17674]|1106.7|
|80|score_after_k|298/300|NA|NA [NA, NA]|NA [NA, NA]|1080|
|80|qr_after_k|300/300|0.99422|0.16444 [0.15001, 0.17759]|0.16427 [0.15028, 0.17748]|1960|

Observed heuristic promotion vetoes: d80 score_after_k vs bootstrap; d80 score_after_k vs fully_adapted; d80 score_after_k vs current_observation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
|Stage recorded|Complete labels and exact oracle checked|3 observed heuristic comparisons veto promotion|Conditional Monte Carlo/tail uncertainty|Continue to next frozen stage|Paper replication/default status|

| Inference item | Status |
|---|---|
|Hard veto screen|Source, finite-state, oracle, wiring and label controls passed; candidate failures preserved|
|Statistically supported ranking|No automatic ranking claim; intervals are pointwise only|
|Descriptive-only differences|Observed moments, tails, counts and CPU times; bootstrap intervals conditional on the fixed data|
|Default-readiness|Not evaluated; independent R extensions|
|Next evidence needed|Review tail concentration and intervals; original implementation/data identity remains missing|

Red-team: rare importance-weight events may be missing even when bootstrap intervals are narrow. Check maximum_loss_share, ratio tails and underflow counts.
Particle counts differ between fitters; R times include explanatory score refits. An Eq15/source-faithfulness conclusion does not follow.
