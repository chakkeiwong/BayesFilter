# Independent T=50 validation of the frozen covariance-guided mixture

The settings from the matched comparison were frozen before these new observations and particle designs. Each dataset has eight paired designs at N=1008, T=50, FP64 GPU/XLA. The raw likelihood and every score coordinate are in the linked CSV; reference fits and uncertainty are preserved in the JSON.

Campaign complete: **False**. Default promotion: **not evaluated**.
Heuristic comparison: `descriptive_losses_present_promotion_veto`.

| Model | Data seed | Reference | Method | Mean log likelihood | Mean absolute likelihood error | Mean score L2 error | Error of mean score |
|---|---:|---|---|---:|---:|---:|---:|
| lgssm | 26100831 | model_reference | old | -138.361379 | 0.027987 | 0.104756 | 0.007578 |
| lgssm | 26100831 | model_reference | old_covariance_only | -138.361242 | 0.027962 | 0.104865 | 0.009078 |
| lgssm | 26100831 | model_reference | new | -138.332676 | 0.047157 | 0.096283 | 0.029790 |
| lgssm | 26100832 | model_reference | old | -120.111796 | 0.024913 | 0.086007 | 0.013569 |
| lgssm | 26100832 | model_reference | old_covariance_only | -120.111736 | 0.024908 | 0.085986 | 0.014116 |
| lgssm | 26100832 | model_reference | new | -120.085168 | 0.046382 | 0.093887 | 0.034927 |
| ksc | 26100831 | model_reference | old | -113.300322 | 1.653558 | 1.369230 | 1.367188 |
| ksc | 26100831 | model_reference | old_covariance_only | -113.293815 | 1.648086 | 1.358657 | 1.355194 |
| ksc | 26100831 | model_reference | new | -112.080705 | 0.150613 | 0.117371 | 0.092221 |
| ksc | 26100832 | model_reference | old | -118.207316 | 1.106189 | 1.098802 | 1.095662 |
| ksc | 26100832 | model_reference | old_covariance_only | -118.162383 | 1.081316 | 1.111285 | 1.109031 |
| ksc | 26100832 | model_reference | new | -117.205552 | 0.185479 | 0.162100 | 0.106687 |
| predator_prey | 26100831 | bootstrap_N131072 | old | -267.364613 | 0.075275 | 0.564018 | 0.286775 |
| predator_prey | 26100831 | bootstrap_N131072 | old_covariance_only | -267.364670 | 0.071228 | 0.572526 | 0.385356 |
| predator_prey | 26100831 | bootstrap_N131072 | new | -267.353725 | 0.062841 | 0.830933 | 0.144444 |
| predator_prey | 26100831 | zhao_rank20 | old | -267.364613 | 0.073403 | 0.643835 | 0.405118 |
| predator_prey | 26100831 | zhao_rank20 | old_covariance_only | -267.364670 | 0.069356 | 0.688778 | 0.545801 |
| predator_prey | 26100831 | zhao_rank20 | new | -267.353725 | 0.061615 | 0.827193 | 0.142772 |
| predator_prey | 26100832 | zhao_rank20 | old | -248.219561 | 0.134680 | 0.816316 | 0.416385 |
| predator_prey | 26100832 | zhao_rank20 | old_covariance_only | -248.215557 | 0.130430 | 0.826660 | 0.502039 |
| predator_prey | 26100832 | zhao_rank20 | new | -248.120605 | 0.123375 | 0.650316 | 0.487377 |

Negative paired differences favor the new method. These intervals vary particle designs conditional on one dataset and a fixed numerical reference. They exclude reference bias.

| Model | Data seed | Reference | Error | Paired new minus old | Conditional 95% interval |
|---|---:|---|---|---:|---|
| lgssm | 26100831 | model_reference | absolute_value_error | 0.019170 | [-0.010356, 0.048696] |
| lgssm | 26100831 | model_reference | score_l2_error | -0.008472 | [-0.044405, 0.027460] |
| lgssm | 26100832 | model_reference | absolute_value_error | 0.021469 | [-0.025917, 0.068855] |
| lgssm | 26100832 | model_reference | score_l2_error | 0.007880 | [-0.022631, 0.038391] |
| ksc | 26100831 | model_reference | absolute_value_error | -1.502945 | [-2.350914, -0.654975] |
| ksc | 26100831 | model_reference | score_l2_error | -1.251860 | [-1.794456, -0.709263] |
| ksc | 26100832 | model_reference | absolute_value_error | -0.920710 | [-1.449450, -0.391970] |
| ksc | 26100832 | model_reference | score_l2_error | -0.936702 | [-1.234986, -0.638419] |
| predator_prey | 26100831 | bootstrap_N131072 | absolute_value_error | -0.012434 | [-0.070641, 0.045773] |
| predator_prey | 26100831 | bootstrap_N131072 | score_l2_error | 0.266915 | [-0.092578, 0.626408] |
| predator_prey | 26100831 | zhao_rank20 | absolute_value_error | -0.011788 | [-0.069395, 0.045819] |
| predator_prey | 26100831 | zhao_rank20 | score_l2_error | 0.183357 | [-0.255942, 0.622657] |
| predator_prey | 26100832 | zhao_rank20 | absolute_value_error | -0.011305 | [-0.102890, 0.080280] |
| predator_prey | 26100832 | zhao_rank20 | score_l2_error | -0.166001 | [-0.815344, 0.483343] |

## Decision and uncertainty

| Item | Finding |
|---|---|
| Completion | False |
| Primary criterion | Continuous likelihood and score errors reported per model and dataset; descriptive_losses_present_promotion_veto |
| Veto diagnostics | 0 invalid rows; 3 failed/running attempts; 9 missing-evidence findings |
| Main uncertainty | Two datasets; finite particle, TT rank/support, and regression-radius bias |
| Next justified action | Inspect conditional failures and reference stability; preserve frozen settings |
| Not concluded | Universal improvement, exact nonlinear oracle, default readiness or HMC validity |

| Inference item | Status |
|---|---|
| Hard veto screen | See explicit invalid/failed/missing records in comparison.json |
| Statistically supported ranking | none claimed across data-generating regimes; reference bias not bounded for nonlinear models |
| Descriptive-only differences | raw differences, per-design means, two-dataset trends, nonlinear reference differences |
| Default readiness | not evaluated or promoted |
| Next evidence needed | more independent datasets and nonlinear reference rank/path convergence before a broad ranking |

The strongest alternative explanation is favorable or unfavorable data-specific geometry combined with shared reference approximation error. Agreement of two fits alone cannot exclude shared rank or support bias. Independent-data reversals or a stable, more accurate reference would overturn a favorable interpretation. The weakest inferential element is the small number of datasets.

Artifacts: `docs/plans/artifacts/ledh-independent-validation-20261008-01/report-20261008-161240/comparison.json` and `values-and-scores.csv`. 
Aggregate worker time: 1.242 hours. 
Plan: `docs/plans/ledh-independent-validation-20261008.md`.
