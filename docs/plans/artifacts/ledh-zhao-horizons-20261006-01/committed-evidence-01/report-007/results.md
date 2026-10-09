# Matched likelihoods and scores

Values are means ± one Monte Carlo standard error where available. Each model uses one frozen dataset and exact time prefixes. Four paired LEDH designs quantify seed variation. References are approximate; their SE excludes bias.

| Model | T | Ancestor LEDH | Mixture LEDH | Original-author TT | Bootstrap N=131072 |
|---|---:|---:|---:|---:|---:|
| predator_prey | 10 | -47.370765 ± 0.0366 | -47.37532 ± 0.0343 | -47.404361 ± 2.47e-06 | -47.401235 ± 0.00699 |
| predator_prey | 20 | -93.905989 ± 0.0397 | -93.896008 ± 0.0418 | -93.905788 ± 2.67e-06 | -93.905077 ± 0.00839 |
| predator_prey | 40 | -193.31695 ± 0.0543 | -193.31562 ± 0.0492 | -193.30227 ± 3.02e-06 | -193.30022 ± 0.0277 |
| predator_prey | 50 | -240.27721 ± 0.0858 | -240.27585 ± 0.0707 | -240.23645 ± 7.64e-07 | -240.24014 ± 0.035 |
| sir_d18 | 10 | -600.4985 ± 37.1 | -579.65913 ± 25.9 | -345.65464 ± 0.00364 | -345.69116 ± 0.0623 |
| sir_d18 | 20 | -942.71305 ± 36.7 | -921.05419 ± 25.9 | pending | -687.67693 ± 0.0535 |
| sir_d18 | 40 | -1602.3714 ± 36.6 | -1580.725 ± 25.9 | pending | -1347.2656 ± 0.0501 |
| sir_d18 | 50 | -1924.1996 ± 36.6 | -1902.5538 ± 25.9 | pending | -1669.1106 ± 0.0511 |

Scores are derivatives in the displayed physical parameter coordinates for predator–prey and log-scale coordinates for SIR.

| Model | T | Parameter | Ancestor LEDH | Mixture LEDH | Original-author quadratic | Bootstrap N=131072 |
|---|---:|---|---:|---:|---:|---:|
| predator_prey | 10 | r | 11.515533 ± 0.0829 | 11.441445 ± 0.0275 | 11.574997 ± 0.04 | 11.937205 ± 0.172 |
| predator_prey | 10 | K | 1.1435818 ± 0.00406 | 1.1434418 ± 0.00359 | 1.149469 ± 0.000453 | 1.1562879 ± 0.00518 |
| predator_prey | 10 | a | 0.025251262 ± 0.000982 | 0.025223896 ± 0.000832 | 0.025564789 ± 1.9e-05 | 0.025770807 ± 5.81e-05 |
| predator_prey | 10 | s | -2.1625702 ± 0.0229 | -2.1380304 ± 0.0234 | -2.2007438 ± 0.00318 | -2.2428437 ± 0.0183 |
| predator_prey | 10 | u | -5.7107424 ± 0.249 | -5.733408 ± 0.209 | -5.7854773 ± 0.00165 | -5.8024062 ± 0.00763 |
| predator_prey | 10 | v | 6.8712828 ± 0.308 | 6.8953526 ± 0.258 | 6.9589584 ± 0.00216 | 6.9795019 ± 0.0084 |
| predator_prey | 20 | r | 11.725961 ± 0.0916 | 11.645634 ± 0.0338 | 11.793711 ± 0.0379 | 12.05359 ± 0.0713 |
| predator_prey | 20 | K | 1.3957971 ± 0.00617 | 1.3943994 ± 0.00577 | 1.397244 ± 0.00265 | 1.4123931 ± 0.00586 |
| predator_prey | 20 | a | 0.03771476 ± 0.000968 | 0.037370637 ± 0.000936 | 0.037657708 ± 1.57e-05 | 0.037894267 ± 9.06e-05 |
| predator_prey | 20 | s | -2.0591507 ± 0.0243 | -1.9897683 ± 0.0295 | -2.0059481 ± 0.00492 | -2.0361299 ± 0.0156 |
| predator_prey | 20 | u | -9.2554276 ± 0.259 | -9.2161729 ± 0.238 | -9.2793258 ± 0.00181 | -9.3048443 ± 0.017 |
| predator_prey | 20 | v | 11.18937 ± 0.319 | 11.138148 ± 0.293 | 11.215729 ± 0.00243 | 11.250297 ± 0.0202 |
| predator_prey | 40 | r | 13.91303 ± 0.0865 | 13.812974 ± 0.0127 | 13.951608 ± 0.0354 | 14.354466 ± 0.157 |
| predator_prey | 40 | K | 4.0687282 ± 0.0112 | 4.0708182 ± 0.00902 | 4.0795801 ± 0.00383 | 4.0902289 ± 0.0109 |
| predator_prey | 40 | a | 0.065132269 ± 0.00133 | 0.064603995 ± 0.00128 | 0.065050721 ± 4.85e-05 | 0.065945279 ± 6.76e-05 |
| predator_prey | 40 | s | -11.589692 ± 0.114 | -11.485066 ± 0.135 | -11.526532 ± 0.00751 | -11.521995 ± 0.0479 |
| predator_prey | 40 | u | -11.055599 ± 0.32 | -10.985607 ± 0.288 | -11.075883 ± 0.00795 | -11.279223 ± 0.0295 |
| predator_prey | 40 | v | 13.400582 ± 0.394 | 13.312013 ± 0.354 | 13.423028 ± 0.0102 | 13.681099 ± 0.0358 |
| predator_prey | 50 | r | 13.932362 ± 0.0743 | 13.830014 ± 0.0112 | 13.922329 ± 0.014 | 14.500892 ± 0.269 |
| predator_prey | 50 | K | 3.4708848 ± 0.011 | 3.4726058 ± 0.00904 | 3.481381 ± 0.00535 | 3.4808036 ± 0.0165 |
| predator_prey | 50 | a | 0.069236042 ± 0.00174 | 0.068795967 ± 0.00159 | 0.068990377 ± 5.59e-05 | 0.069805584 ± 4.72e-05 |
| predator_prey | 50 | s | -10.310515 ± 0.114 | -10.211398 ± 0.146 | -10.271913 ± 0.00909 | -10.277717 ± 0.028 |
| predator_prey | 50 | u | -12.910456 ± 0.428 | -12.860692 ± 0.364 | -12.874511 ± 0.0151 | -13.037979 ± 0.0102 |
| predator_prey | 50 | v | 15.675575 ± 0.526 | 15.61188 ± 0.447 | 15.628927 ± 0.0191 | 15.840188 ± 0.011 |
| sir_d18 | 10 | log_kappa_scale | -1683.9464 ± 1.55e+03 | 1198.8577 ± 1.76e+03 | pending | 100.70251 ± 10.5 |
| sir_d18 | 10 | log_nu_scale | 809.86873 ± 668 | -321.52377 ± 833 | pending | -34.975705 ± 1.31 |
| sir_d18 | 10 | log_observation_noise_scale | 641.8544 ± 230 | 313.4928 ± 587 | pending | 9.2621382 ± 0.193 |
| sir_d18 | 20 | log_kappa_scale | -1621.8655 ± 1.51e+03 | 1046.5636 ± 1.78e+03 | pending | 90.377245 ± 10.3 |
| sir_d18 | 20 | log_nu_scale | 746.65782 ± 642 | -278.39007 ± 847 | pending | -41.024817 ± 2.52 |
| sir_d18 | 20 | log_observation_noise_scale | 658.4524 ± 227 | 337.30572 ± 580 | pending | 21.741712 ± 0.255 |
| sir_d18 | 40 | log_kappa_scale | -1624.897 ± 1.51e+03 | 998.21414 ± 1.79e+03 | pending | 87.951368 ± 12.5 |
| sir_d18 | 40 | log_nu_scale | 746.03023 ± 642 | -264.65355 ± 852 | pending | -38.966773 ± 2.33 |
| sir_d18 | 40 | log_observation_noise_scale | 634.31836 ± 227 | 314.26465 ± 578 | pending | -2.5966581 ± 0.208 |
| sir_d18 | 50 | log_kappa_scale | -1624.9051 ± 1.51e+03 | 999.18552 ± 1.79e+03 | pending | 87.551025 ± 13.7 |
| sir_d18 | 50 | log_nu_scale | 745.92735 ± 642 | -265.06359 ± 851 | pending | -37.113419 ± 3.06 |
| sir_d18 | 50 | log_observation_noise_scale | 606.63908 ± 227 | 286.51134 ± 578 | pending | -30.267602 ± 0.205 |

The separate SIR rank-20 calculation is a rank diagnostic. Its uncertainty is conditional on one fitted proposal. It is not pooled with rank40.

| T | Quantity | Original-author rank20 | Original-author rank40 |
|---:|---|---:|---:|
| 10 | log_likelihood | -345.66223 ± 0.00391 | -345.65464 ± 0.00364 |
| 10 | log_kappa_scale | 108.15945 ± 3.11 | pending |
| 10 | log_nu_scale | -39.349353 ± 1.23 | pending |
| 10 | log_observation_noise_scale | 9.0423837 ± 0.0349 | pending |
| 20 | log_likelihood | -687.65346 ± 0.00433 | pending |
| 20 | log_kappa_scale | 108.12348 ± 3.88 | pending |
| 20 | log_nu_scale | -47.475585 ± 2.07 | pending |
| 20 | log_observation_noise_scale | 21.580917 ± 0.0319 | pending |
| 40 | log_likelihood | -1347.2406 ± 0.0061 | pending |
| 40 | log_kappa_scale | pending | pending |
| 40 | log_nu_scale | pending | pending |
| 40 | log_observation_noise_scale | pending | pending |
| 50 | log_likelihood | -1669.0939 ± 0.00494 | pending |
| 50 | log_kappa_scale | pending | pending |
| 50 | log_nu_scale | pending | pending |
| 50 | log_observation_noise_scale | pending | pending |

Finite-program derivative diagnostics (implementation checks, not statistical score accuracy):

| Check | Model | Policy | Largest normalized error over recorded steps | Trace parity | Screen |
|---|---|---|---:|---|---|
| score-check-001 | predator_prey | ancestor | 2.1477e-09 | True | True |
| score-check-001 | predator_prey | marginal_mixture | 2.1692e-09 | True | True |
| score-check-001 | sir_d18 | ancestor | 0.04675 | True | False |
| score-check-001 | sir_d18 | marginal_mixture | 1.8763e-05 | False | False |
| score-check-002 | sir_d18 | ancestor | 7.4353e-06 | True | True |
| score-check-002 | sir_d18 | marginal_mixture | 6.0374e-05 | True | True |

The author TT column uses logmeanexp of the original raw importance weights. The raw author lml diagnostic (mean log weight), conditional path SE, ESS and fit seeds are in original-author-values.json. TT score uncertainties use between-fit SE when independent fits exist, otherwise conditional path jackknife SE from one fit; the CSV labels this distinction. Rank20 SIR checks are retained separately from rank40 and never pooled. TT scores use the smallest predeclared regression radius; all radii and heldout checks remain in their run directories.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Hold promotion | Numerical comparisons reported continuously | See invalid and unfinished attempts in decision.json | Reference bias, one dataset, limited independent fits | Complete pending runs; assess particle/rank/radius stability | Superiority, oracle certification, HMC/default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | 0 invalid returned LEDH evaluations; incomplete jobs are separately listed |
| Statistically supported ranking | None established |
| Descriptive differences | Per-component errors and paired changes retained |
| Default readiness | Not assessed; controls are untuned in these scopes |
| Next evidence | Stable independent references and fresh scope-specific tuning/holdout data |

Post-run alternative explanation: reference bias and untuned moment/reset controls can dominate a weighting difference. Agreement of two approximations alone is insufficient; the weakest evidence is long-horizon score accuracy with limited independent fits.
