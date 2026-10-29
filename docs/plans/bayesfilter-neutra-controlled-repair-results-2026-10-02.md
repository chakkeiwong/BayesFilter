# Controlled NeuTra repair: recorded results

Execution status: **terminal_review_complete**. Next recorded action: `terminal_artifacts_audited_and_monograph_rebuilt`.

This is a bounded FP64 diagnostic campaign on the canonical IAF. Checkpoint loss and geometry cannot establish posterior correctness. No ranking or FP32/TF32 readiness is claimed.

**8/8 cases settled; 0 passed fresh posterior confirmation.** An unqualified case means the bounded procedure did not deliver its declared result; it does not by itself reject the research direction.

Charged GPU process time: 8.2593 h of 12 h. Charged CPU core time: 8.5576 h of 24 h. Active workers are additional, still reserved work. Failed attempts are included.

## Controls

| Target | Numerical exact transport | Pooled teacher | iid joint passes | Exact Gaussian HMC |
|---|---|---|---|---|
| mixture | True | True | 8/8 | True |
| warped_mixture | True | True | 8/8 | True |

The iid results use eight independent streams paired across targets; they measure finite-sample feasibility, not nominal coverage. The Gaussian controls include exact physical decoding and separate reference streams. The numerical transport check evaluates target-plus-log-Jacobian cancellation separately. The practical teacher screen covers event probabilities and marginal moments with between-replication uncertainty; it does not establish equality of the entire conditional density within every mode.

## Training and posterior decisions

| Target / teacher / seed | Completed fit arms | Posterior confirmation |
|---|---|---|
| mixture / oracle / 11 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| mixture / oracle / 37 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| mixture / estimated / 11 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| mixture / estimated / 37 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| warped_mixture / oracle / 11 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| warped_mixture / oracle / 37 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| warped_mixture / estimated / 11 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |
| warped_mixture / estimated / 37 | parent, continue, forward, reverse, joint, joint-continuation | not reached: no map qualified in bounded checks |

| Completed fit | Final Adam step | Lifetime map updates | Stop classification | Heldout FKL cross entropy | Estimated RKL | Gaussian residual median | Directed residual max |
|---|---:|---:|---|---:|---:|---:|---:|
| parent-mixture-oracle-s11 | 32768 | 32768 | budget_limited | 3.5303 | 0.156656 | 0.311841 | 432.56 |
| continue-mixture-oracle-s11 | 65536 | 65536 | budget_limited | 3.5124 | 0.103172 | 0.253681 | 493.199 |
| forward-mixture-oracle-s11 | 32768 | 65536 | budget_limited | 3.51852 | 0.120892 | 0.259855 | 499.622 |
| reverse-mixture-oracle-s11 | 32768 | 65536 | budget_limited | 3.78498 | 0.0576407 | 0.260773 | 1056.63 |
| joint-mixture-oracle-s11 | 32768 | 65536 | budget_limited | 3.5118 | 0.0565832 | 0.249321 | 661.953 |
| joint-continuation-mixture-oracle-s11 | 98304 | 131072 | budget_limited | 3.50029 | 0.0299385 | 0.167722 | 1181.46 |
| parent-mixture-oracle-s37 | 32768 | 32768 | budget_limited | 3.49773 | 0.0420978 | 0.239347 | 1326.62 |
| continue-mixture-oracle-s37 | 65536 | 65536 | budget_limited | 3.4966 | 0.0479171 | 0.278709 | 2078.31 |
| forward-mixture-oracle-s37 | 32768 | 65536 | budget_limited | 3.49263 | 0.0297487 | 0.233406 | 2077.42 |
| reverse-mixture-oracle-s37 | 32768 | 65536 | budget_limited | 3.49001 | 0.00967217 | 0.120318 | 3457.77 |
| joint-mixture-oracle-s37 | 32768 | 65536 | budget_limited | 3.48665 | 0.00814138 | 0.129009 | 3567.49 |
| joint-continuation-mixture-oracle-s37 | 40960 | 73728 | budget_limited | 3.49658 | 0.0254977 | 0.15625 | 3701.96 |
| parent-mixture-estimated-s11 | 32768 | 32768 | budget_limited | 3.49385 | 0.0229748 | 0.154473 | 1851.09 |
| continue-mixture-estimated-s11 | 65536 | 65536 | budget_limited | 3.48656 | 0.0126233 | 0.151765 | 3007.89 |
| forward-mixture-estimated-s11 | 32768 | 65536 | budget_limited | 3.49214 | 0.0136962 | 0.136399 | 2998.27 |
| reverse-mixture-estimated-s11 | 32768 | 65536 | budget_limited | 3.49488 | 0.0164461 | 0.171398 | 4972.41 |
| joint-mixture-estimated-s11 | 32768 | 65536 | budget_limited | 3.48352 | 0.00880881 | 0.122953 | 4593.44 |
| joint-continuation-mixture-estimated-s11 | 40960 | 73728 | budget_limited | 3.48415 | 0.0204999 | 0.135156 | 4873.3 |
| parent-mixture-estimated-s37 | 32768 | 32768 | budget_limited | 3.48984 | 0.0321457 | 0.127977 | 1253.54 |
| continue-mixture-estimated-s37 | 65536 | 65536 | budget_limited | 3.49205 | 0.03862 | 0.189626 | 1653.03 |
| forward-mixture-estimated-s37 | 32768 | 65536 | budget_limited | 3.49137 | 0.0410085 | 0.159256 | 1671.07 |
| reverse-mixture-estimated-s37 | 32768 | 65536 | budget_limited | 3.4916 | 0.014173 | 0.163797 | 4052.72 |
| joint-mixture-estimated-s37 | 32768 | 65536 | budget_limited | 3.49046 | 0.0229172 | 0.198793 | 3261.55 |
| joint-continuation-mixture-estimated-s37 | 40960 | 73728 | budget_limited | 3.4803 | 0.00840964 | 0.102619 | 3286.01 |
| parent-warped_mixture-oracle-s11 | 8192 | 8192 | budget_limited | 3.54692 | 0.208097 | 0.289685 | 401.049 |
| continue-warped_mixture-oracle-s11 | 16384 | 16384 | budget_limited | 3.53028 | 0.148549 | 0.255246 | 595.597 |
| forward-warped_mixture-oracle-s11 | 8192 | 16384 | budget_limited | 3.52281 | 0.150769 | 0.240204 | 598.004 |
| reverse-warped_mixture-oracle-s11 | 32768 | 40960 | budget_limited | 3.50493 | 0.0280242 | 0.150397 | 2577.28 |
| joint-warped_mixture-oracle-s11 | 8192 | 16384 | budget_limited | 3.49903 | 0.0405365 | 0.225586 | 989.122 |
| joint-continuation-warped_mixture-oracle-s11 | 16384 | 24576 | budget_limited | 3.50461 | 0.0253255 | 0.221864 | 1563.33 |
| parent-warped_mixture-oracle-s37 | 8192 | 8192 | budget_limited | 3.53137 | 0.125563 | 0.295535 | 382.103 |
| continue-warped_mixture-oracle-s37 | 40960 | 40960 | budget_limited | 3.49958 | 0.0293775 | 0.274324 | 1118.48 |
| forward-warped_mixture-oracle-s37 | 32768 | 40960 | budget_limited | 3.49089 | 0.0220826 | 0.188047 | 1126.63 |
| reverse-warped_mixture-oracle-s37 | 32768 | 40960 | budget_limited | 3.50345 | 0.0195009 | 0.228385 | 2689.7 |
| joint-warped_mixture-oracle-s37 | 32768 | 40960 | budget_limited | 3.49238 | 0.0199665 | 0.226259 | 2292.35 |
| joint-continuation-warped_mixture-oracle-s37 | 98304 | 106496 | budget_limited | 3.48277 | 0.0101372 | 0.116889 | 3635.48 |
| parent-warped_mixture-estimated-s11 | 32768 | 32768 | budget_limited | 3.49147 | 0.0592382 | 0.136924 | 1042.39 |
| continue-warped_mixture-estimated-s11 | 65536 | 65536 | budget_limited | 3.48301 | 0.0271511 | 0.119837 | 1903.88 |
| forward-warped_mixture-estimated-s11 | 32768 | 65536 | budget_limited | 3.48763 | 0.0343236 | 0.156621 | 2056.82 |
| reverse-warped_mixture-estimated-s11 | 32768 | 65536 | budget_limited | 3.48996 | 0.0253459 | 0.182613 | 4140.4 |
| joint-warped_mixture-estimated-s11 | 32768 | 65536 | budget_limited | 3.4853 | 0.0211848 | 0.15695 | 3126.58 |
| joint-continuation-warped_mixture-estimated-s11 | 98304 | 131072 | budget_limited | 3.48304 | 0.0209626 | 0.0911837 | 4123.85 |
| parent-warped_mixture-estimated-s37 | 32768 | 32768 | budget_limited | 3.51455 | 0.115484 | 0.21613 | 403.4 |
| continue-warped_mixture-estimated-s37 | 65536 | 65536 | budget_limited | 3.50645 | 0.0798227 | 0.217255 | 690.018 |
| forward-warped_mixture-estimated-s37 | 32768 | 65536 | budget_limited | 3.50575 | 0.0954013 | 0.175211 | 734.39 |
| reverse-warped_mixture-estimated-s37 | 32768 | 65536 | budget_limited | 3.58677 | 0.0372024 | 0.216388 | 1278.82 |
| joint-warped_mixture-estimated-s37 | 32768 | 65536 | budget_limited | 3.49826 | 0.0430846 | 0.152448 | 1019.72 |
| joint-continuation-warped_mixture-estimated-s37 | 98304 | 131072 | budget_limited | 3.49348 | 0.0260629 | 0.149814 | 1672.9 |

These diagnostics are descriptive. Optimizer step counts include restored Adam iterations for preserved-state continuation and restart at zero for reset arms. Lifetime map updates follow the selected map’s ancestry; discarded pilots also consume compute, but do not belong to that map’s ancestry. Use each history’s additional updates and measured cost for fair work accounting. Resource-limited fits are not converged fits. Teacher arms share a search protocol but may nominate different widths or learning rates, so their differences do not isolate teacher quality alone. Objective branches within a case share their initial map. The two development seeds and paired streams across targets/teachers do not make eight statistically independent method replications.

## Standard 1,000-point probes of final joint maps

| Target / teacher / seed | Median | Mean | q95 | q99 | Maximum | Fraction > 1 | Log-density offset range |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixture-oracle-s11 | 0.167722 | 6.38692 | 3.23774 | 102.011 | 1028.84 | 0.092 | 7.64366 |
| mixture-oracle-s37 | 0.15625 | 7.73422 | 2.43637 | 12.8449 | 3535.16 | 0.091 | 10.4991 |
| mixture-estimated-s11 | 0.135156 | 8.75537 | 1.93741 | 18.8186 | 4124.65 | 0.078 | 5.60036 |
| mixture-estimated-s37 | 0.102619 | 4.72086 | 2.22081 | 15.6402 | 1415.16 | 0.083 | 3.33576 |
| warped_mixture-oracle-s11 | 0.221864 | 4.38851 | 2.64997 | 42.5361 | 1191.65 | 0.084 | 8.19154 |
| warped_mixture-oracle-s37 | 0.116889 | 11.8497 | 3.17884 | 77.6743 | 2891.09 | 0.096 | 6.85494 |
| warped_mixture-estimated-s11 | 0.0911837 | 4.17799 | 1.89492 | 20.4844 | 1411.29 | 0.08 | 8.6362 |
| warped_mixture-estimated-s37 | 0.149814 | 9.85543 | 5.46742 | 106.616 | 1777.47 | 0.156 | 12.0436 |

These are the prescribed Gaussian-base probes of the final trained joint checkpoint, including its one continuation where completed. They do not select or qualify a map. Mean and tail summaries can expose errors hidden by the median; finite probes cannot establish uniform geometry. Repeated use of the same probes and two fitted seeds does not support a statistical ranking.

## Continuation and plateau evidence

| Case | Chosen width | LR | Additional continuation checkpoints | Last paired objective gain | SE | Stop |
|---|---:|---:|---:|---:|---:|---|
| mixture-oracle-s11 | 16 | 0.001 | 3 | 0.00476104 | 0.00745882 | budget_limited |
| mixture-oracle-s37 | 32 | 0.001 | 1 | not available | not available | budget_limited |
| mixture-estimated-s11 | 32 | 0.001 | 1 | not available | not available | budget_limited |
| mixture-estimated-s37 | 32 | 0.001 | 1 | not available | not available | budget_limited |
| warped_mixture-oracle-s11 | 32 | 0.001 | 1 | not available | not available | budget_limited |
| warped_mixture-oracle-s37 | 16 | 0.001 | 3 | 0.00653488 | 0.00346027 | budget_limited |
| warped_mixture-estimated-s11 | 32 | 0.001 | 3 | -0.00208124 | 0.00548157 | budget_limited |
| warped_mixture-estimated-s37 | 32 | 0.001 | 3 | 0.0146184 | 0.00776954 | budget_limited |

The operational plateau requires two successive distinct-map comparisons with `abs(gain) + 3*SE < 0.001` nats. A single continuation checkpoint has no within-job paired change. A noisy gain near zero is inconclusive; it is not a plateau. This repeated-look rule is an engineering resolution test, not a global-optimization or formal sequential-coverage guarantee.

## Frozen-map qualification outcomes

| Completed qualification job | Checkpoint outcomes | Retained posterior outcomes |
|---|---|---|
| joint-mixture-oracle-s11 | qualification_budget_exhausted: 3 | warmup count cap: 2; resource-limited/incomplete: 3 |
| continue-mixture-oracle-s11 | qualification_budget_exhausted: 2; no_verified_member_passed_posterior_screen: 1 | resource-limited/incomplete: 3; warmup count cap: 2 |
| forward-mixture-oracle-s11 | qualification_budget_exhausted: 2; no verified pair within bounded search; map/kernel repair remains possible: 1 | warmup count cap: 2; resource-limited/incomplete: 2 |
| reverse-mixture-oracle-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3 |
| parent-mixture-oracle-s11 | no verified pair within bounded search; map/kernel repair remains possible: 2; qualification_budget_exhausted: 1 | resource-limited/incomplete: 1 |
| repair-mixture-oracle-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 1 |
| joint-continuation-mixture-oracle-s11 | qualification_budget_exhausted: 3 | none completed |
| terminal-allocation-mixture-oracle-s11 | qualification_budget_exhausted: 1 | retained cap: broad-event precision, event coverage per chain: 1; resource-limited/incomplete: 1 |
| joint-mixture-oracle-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2 |
| continue-mixture-oracle-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2 |
| forward-mixture-oracle-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2 |
| reverse-mixture-oracle-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 1 |
| parent-mixture-oracle-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 1 |
| repair-mixture-oracle-s37 | qualification_budget_exhausted: 3 | none completed |
| joint-continuation-mixture-oracle-s37 | qualification_budget_exhausted: 1 | resource-limited/incomplete: 1 |
| terminal-allocation-mixture-oracle-s37 | qualification_budget_exhausted: 1 | retained cap: broad-event precision, event coverage per chain: 1; resource-limited/incomplete: 1 |
| resource-completion-mixture-oracle-s11 | no_verified_member_passed_posterior_screen: 1 | retained cap: broad-event precision, event coverage per chain: 3 |
| resource-completion-mixture-oracle-s37 | no_verified_member_passed_posterior_screen: 1 | retained cap: broad-event precision, event coverage per chain: 2; warmup count cap: 1 |
| joint-mixture-estimated-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2; retained cap: broad-event precision, event coverage per chain: 1 |
| continue-mixture-estimated-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3 |
| forward-mixture-estimated-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2 |
| reverse-mixture-estimated-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2 |
| parent-mixture-estimated-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; qualification_budget_exhausted: 2 | none completed |
| repair-mixture-estimated-s11 | qualification_budget_exhausted: 3 | none completed |
| joint-continuation-mixture-estimated-s11 | no_verified_member_passed_posterior_screen: 1 | warmup count cap: 2; resource-limited/incomplete: 1 |
| terminal-allocation-mixture-estimated-s11 | no_verified_member_passed_posterior_screen: 1 | warmup count cap: 3 |
| joint-mixture-estimated-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3 |
| continue-mixture-estimated-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3 |
| forward-mixture-estimated-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 2 |
| reverse-mixture-estimated-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3; warmup count cap: 1 |
| parent-mixture-estimated-s37 | qualification_budget_exhausted: 3 | none completed |
| repair-mixture-estimated-s37 | qualification_budget_exhausted: 3 | none completed |
| joint-continuation-mixture-estimated-s37 | qualification_budget_exhausted: 1 | resource-limited/incomplete: 1 |
| terminal-allocation-mixture-estimated-s37 | qualification_budget_exhausted: 1 | resource-limited/incomplete: 1 |
| resource-completion-mixture-estimated-s37 | no_verified_member_passed_posterior_screen: 1 | warmup count cap: 2; retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 1 |
| joint-warped_mixture-oracle-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 1 | resource-limited/incomplete: 1; warmup count cap: 2 |
| continue-warped_mixture-oracle-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 1 | warmup count cap: 2; resource-limited/incomplete: 1 |
| forward-warped_mixture-oracle-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 1 | warmup count cap: 2; resource-limited/incomplete: 1 |
| reverse-warped_mixture-oracle-s11 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3 |
| parent-warped_mixture-oracle-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 1 | warmup count cap: 2; resource-limited/incomplete: 1 |
| repair-warped_mixture-oracle-s11 | qualification_budget_exhausted: 2 | resource-limited/incomplete: 2 |
| joint-continuation-warped_mixture-oracle-s11 | qualification_budget_exhausted: 1 | resource-limited/incomplete: 1; warmup count cap: 1 |
| terminal-allocation-warped_mixture-oracle-s11 | qualification_budget_exhausted: 1 | resource-limited/incomplete: 1; warmup count cap: 1 |
| resource-completion-warped_mixture-oracle-s11 | no_verified_member_passed_posterior_screen: 1 | warmup count cap: 3 |
| joint-warped_mixture-oracle-s37 | qualification_budget_exhausted: 3 | resource-limited/incomplete: 3 |
| continue-warped_mixture-oracle-s37 | no verified pair within bounded search; map/kernel repair remains possible: 1; qualification_budget_exhausted: 2 | warmup count cap: 2; resource-limited/incomplete: 1 |
| forward-warped_mixture-oracle-s37 | no verified pair within bounded search; map/kernel repair remains possible: 1; qualification_budget_exhausted: 2 | warmup count cap: 2; resource-limited/incomplete: 2 |
| reverse-warped_mixture-oracle-s37 | no_verified_member_passed_posterior_screen: 1; qualification_budget_exhausted: 2 | resource-limited/incomplete: 3 |
| parent-warped_mixture-oracle-s37 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 1 | warmup count cap: 3 |
| repair-warped_mixture-oracle-s37 | qualification_budget_exhausted: 1; no_verified_member_passed_posterior_screen: 2 | resource-limited/incomplete: 2; warmup count cap: 4; numerical/health veto: warmup_chunk_health_failed: 1; retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 1 |
| joint-continuation-warped_mixture-oracle-s37 | no_verified_member_passed_posterior_screen: 3 | warmup count cap: 7; numerical/health veto: warmup_chunk_health_failed: 2 |
| terminal-allocation-warped_mixture-oracle-s37 | no_verified_member_passed_posterior_screen: 1 | retained cap: broad-event precision, event coverage per chain: 1; numerical/health veto: warmup_chunk_health_failed: 1; warmup count cap: 1 |
| joint-warped_mixture-estimated-s11 | no_verified_member_passed_posterior_screen: 3 | warmup count cap: 6; numerical/health veto: warmup_chunk_health_failed: 1; retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 1 |
| continue-warped_mixture-estimated-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 2 | warmup count cap: 5 |
| forward-warped_mixture-estimated-s11 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 2 | warmup count cap: 5; retained cap: broad-event precision, event coverage per chain: 1 |
| reverse-warped_mixture-estimated-s11 | no verified pair within bounded search; map/kernel repair remains possible: 2; no_verified_member_passed_posterior_screen: 1 | retained cap: broad-event precision, event coverage per chain: 2; warmup count cap: 1 |
| parent-warped_mixture-estimated-s11 | no verified pair within bounded search; map/kernel repair remains possible: 2; no_verified_member_passed_posterior_screen: 1 | warmup count cap: 2 |
| repair-warped_mixture-estimated-s11 | no_verified_member_passed_posterior_screen: 3 | warmup count cap: 7; resource-limited/incomplete: 2 |
| joint-continuation-warped_mixture-estimated-s11 | no_verified_member_passed_posterior_screen: 2; qualification_budget_exhausted: 1 | retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 1; warmup count cap: 5; numerical/health veto: warmup_chunk_health_failed: 2 |
| terminal-allocation-warped_mixture-estimated-s11 | no_verified_member_passed_posterior_screen: 1 | warmup count cap: 1; numerical/health veto: warmup_chunk_health_failed: 1; retained cap: broad-event precision, event coverage per chain: 1 |
| joint-warped_mixture-estimated-s37 | no_verified_member_passed_posterior_screen: 3 | warmup count cap: 4; retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 2; retained cap: broad-event precision, event coverage per chain: 2 |
| continue-warped_mixture-estimated-s37 | no_verified_member_passed_posterior_screen: 3 | retained cap: broad-event precision, event coverage per chain: 1; warmup count cap: 7; retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 1 |
| forward-warped_mixture-estimated-s37 | no_verified_member_passed_posterior_screen: 3 | warmup count cap: 6; retained cap: broad-event precision, event coverage per chain: 1 |
| reverse-warped_mixture-estimated-s37 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 2 | retained cap: broad-event precision, event coverage per chain: 3; retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 2; retained cap: precision, broad-event precision, event coverage per chain: 1 |
| parent-warped_mixture-estimated-s37 | no verified pair within bounded search; map/kernel repair remains possible: 1; no_verified_member_passed_posterior_screen: 2 | warmup count cap: 6 |
| repair-warped_mixture-estimated-s37 | qualification_budget_exhausted: 3 | retained cap: rhat, bulk_ess, tail_ess, binary_outcomes_observed, precision, broad-event precision, event coverage per chain: 2; resource-limited/incomplete: 2; retained cap: broad-event precision, event coverage per chain: 1; warmup count cap: 1 |
| joint-continuation-warped_mixture-estimated-s37 | qualification_budget_exhausted: 3 | retained cap: broad-event precision, event coverage per chain: 1; resource-limited/incomplete: 3; warmup count cap: 2 |
| terminal-allocation-warped_mixture-estimated-s37 | no_verified_member_passed_posterior_screen: 1 | warmup count cap: 1; retained cap: broad-event precision, event coverage per chain: 2 |

Recorded member outcomes across the completed jobs: numerical/health veto: 8; resource-limited/incomplete: 76; retained cap: 36; warmup count cap: 106. These include repeated checkpoints and kernels; they are not independent replications.

Failed chunk-health fields: all_states_moved: 2; log_accept_ratio_all_finite: 6. Counts refer to failed checks, not necessarily distinct maps or methods.

A resource stop is incomplete evidence. It does not establish a numerical defect, poor stationary sampling, or rejection of NeuTra. Count caps are failures to deliver the declared result within the fixed procedure. Kernel observations and repeated checkpoints are not independent method replications.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Analytic controls | Read control table above | Failed search/settings preserved | Finite-run reliability | Assess learned maps only after controls pass | Learned-map success |
| Learned transports | Fresh posterior confirmation required | See per-checkpoint/member assessments | Optimization, residual geometry, retained event information | Follow recorded repair/remaining budget | Convergence from loss/probes |
| Optional geometry extension | Requires adequate stationary density fit and failed geometry | Trigger must be demonstrated | No calibrated objective coefficient yet | Price/test derivatives only if trigger is supported | Implemented or validated extension |
 
| Inference status | Conclusion |
|---|---|
| Hard veto screen | Numerical/health failures reject the affected evidence; a precision cap rejects that candidate’s delivery. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Loss, score norms, tails, timings, and individual fit trajectories. |
| Default readiness | Not established; FP64 references and two small targets cannot promote TF32/q20. |
| Next evidence needed | Completed confirmations, target-specific optimization evidence, and independent replication appropriate to the claimed reliability. |

## Engineering repairs and audit

The Gaussian control’s scope mismatch was fixed before numerical tuning. Initial step-size/trajectory choices hit analytic resonances; the wider search and remaining-member phase preserved rejected settings and used unchanged posterior criteria. A duplicate export after successful confirmation was recovered from consistent saved evidence without rerunning simulation. Diagnostic graphs are reused across current map variables; focused tests checked that reuse does not freeze stale parameters. Earlier checkpoints are eligible for downstream screening, and the selected filename is passed to fresh mode-spanning confirmation.

Focused checks passed for the combined gradient, exact next-update restoration, invalid-update rollback, numerical controls, diagnostic graph reuse, checkpoint selection and controller adoption/resource allocation. The 22 shared rare-region/assessment regressions also passed. Fourteen pre-existing precision-suite failures conflict with the dirty working-tree configuration API; these have not been repaired or hidden. This campaign exercises the declared FP64 route and does not establish FP32/TF32 readiness.

The strongest alternative explanation for poor learned-map sampling is unfinished or inadequate optimization, rather than a failure of the IAF research direction. A frozen map passing independent posterior confirmation would overturn rejection of that candidate. Rare-event precision and the small number of fitted seeds limit stronger conclusions.

Plan: `docs/plans/bayesfilter-neutra-controlled-repair-master-2026-10-02.md`. Every attempt has a command/environment/source/device/seed/time manifest. The campaign state preserves attempt order, failures, repairs, and costs. Original results and source snapshots have not been overwritten. The reproducible provenance audit is `scripts/audit_neutra_controlled_repair.py`; the terminal JSON audit is stored beside the campaign state.
