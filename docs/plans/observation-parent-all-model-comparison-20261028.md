# Original, uniform-parent and weighted-parent comparison

All five models are included. Uniform means observation-conditioned child proposals with equal parent-selection probabilities; weighted uses observation-dependent parent probabilities. Original is the earlier three-branch proposal. Defensive branches remain present in the conditional-mixture arms.

These are saved FP64 GPU/XLA diagnostic runs, with N=1008 and eight paired seeds per model. No new numerical experiment, tuning, default change or algorithm edit was performed for this comparison. The main SIR row is the nominal T=50 case, chosen to align horizons; the CSV contains all 20 usable SIR parameter/horizon cells, including nominal T=10 and T=20. Comparisons are within each model and parameter coordinate; error magnitudes across models or differently scaled parameters are not pooled.

Skeptical reporting audit: preserve each saved comparator and target, use the completed nonlinear extension for KSC/range–bearing, distinguish mean error from RMSE, retain both SIR reference fits, and do not substitute ESS for accuracy. Source hashes and per-seed arithmetic are verified by the accompanying script. Existing run plans remain the governing evidence contracts.

## Log-likelihood estimates

Each method cell is mean estimate (RMSE against the displayed reference). Lower RMSE is favorable descriptively. These are log likelihoods, not raw likelihoods.

| Model | T | Reference | Original | Uniform parents | Weighted parents |
| --- | --- | --- | --- | --- | --- |
| LGSSM | 50 | -138.358 | -138.377 (0.0513889) | -138.377 (0.0523155) | -138.377 (0.0525897) |
| KSC SV | 50 | -111.975 | -112.039 (0.206717) | -112.025 (0.135892) | -112.077 (0.115135) |
| Predator–prey | 50 | -267.317 | -267.328 (0.145348) | -267.317 (0.148873) | -267.335 (0.17887) |
| Range–bearing | 20 | 39.8555 | 39.6365 (0.331358) | 39.6231 (0.349486) | 39.7131 (0.231912) |
| SIR nominal | 50 | -1669.09 | -1669.27 (0.854424) | -1669.16 (0.750419) | -1669.19 (0.654556) |

## Score estimates

Each method cell is mean score (component RMSE against the displayed reference). The score is the parameter gradient of log likelihood. Means can appear close while individual runs have large errors; RMSE includes both mean error and seed dispersion.

| Model | Parameter | Reference | Original | Uniform parents | Weighted parents |
| --- | --- | --- | --- | --- | --- |
| LGSSM | transition_parameter | 5.86731 | 5.87984 (0.0980073) | 5.87915 (0.0993004) | 5.86609 (0.0950208) |
| LGSSM | log_process_variance | 7.32385 | 7.32594 (0.0144067) | 7.32557 (0.0149315) | 7.32618 (0.0142106) |
| LGSSM | log_observation_variance | 4.13012 | 4.13771 (0.0411005) | 4.1381 (0.0414811) | 4.14015 (0.0424457) |
| LGSSM | initial_mean_scale | 0.099617 | 0.100207 (0.00193968) | 0.100258 (0.00201672) | 0.100278 (0.00195216) |
| KSC SV | persistence_parameter | -2.0026 | -2.02203 (0.0744471) | -1.39563 (0.609442) | -1.59661 (0.407061) |
| KSC SV | log_observation_scale | -0.824124 | -0.698378 (0.142015) | -0.436223 (0.389578) | -0.733437 (0.0944128) |
| Predator–prey | r | -33.9911 | -34.1748 (0.302258) | -34.2382 (0.353482) | -34.3073 (0.397328) |
| Predator–prey | carrying_capacity | -0.769203 | -0.782646 (0.0228249) | -0.783775 (0.0282161) | -0.834925 (0.0689603) |
| Predator–prey | half_saturation | -0.00369203 | -0.00375312 (0.00139447) | -0.00405471 (0.00150476) | -0.0051175 (0.00174575) |
| Predator–prey | s | 4.04338 | 3.94216 (0.240421) | 3.98077 (0.168025) | 4.47244 (0.459779) |
| Predator–prey | u | -4.47828 | -4.4094 (0.383359) | -4.35234 (0.423594) | -4.22303 (0.424911) |
| Predator–prey | v | 4.87586 | 4.79117 (0.468782) | 4.72069 (0.51788) | 4.60354 (0.495951) |
| Range–bearing | log_process_scale | -0.334787 | -0.304327 (0.634585) | -0.130327 (0.426624) | -0.252753 (0.502737) |
| Range–bearing | log_observation_scale | -5.79515 | -5.23519 (0.758656) | -5.45566 (0.715763) | -5.394 (0.817229) |
| SIR nominal | log infection multiplier | 102.855 | 6338.21 (18648.6) | 110.512 (32.7596) | 131.782 (44.4186) |
| SIR nominal | log recovery multiplier | -48.6029 | -2996.01 (7905.32) | -53.6253 (7.23488) | -57.2394 (13.0812) |
| SIR nominal | log observation-noise multiplier | -30.3076 | -5999.53 (9593.82) | -27.9792 (3.85256) | -25.5908 (4.94545) |

## Particle ESS

These are averages over time and design seeds. Higher ESS alone does not establish a more accurate likelihood or score.

| Model | Original | Uniform parents | Weighted parents |
| --- | --- | --- | --- |
| LGSSM | 957.19 | 957.143 | 957.185 |
| KSC SV | 883.396 | 941.015 | 996.65 |
| Predator–prey | 972.369 | 936.209 | 972.323 |
| Range–bearing | 727.786 | 718.443 | 733.762 |
| SIR nominal | 796.728 | 816.117 | 911.872 |

## Interpretation and reference limitations

Uniform parents have lower observed score RMSE than weighted parents in both range–bearing components, five of six predator–prey components and all three nominal SIR T=50 components. Weighted parents have lower observed score RMSE in both KSC components and three of four LGSSM components. Weighted log-likelihood RMSE is lower for KSC, range–bearing and nominal SIR T=50; uniform is lower for LGSSM and predator–prey. These mixed results do not support a universally more accurate parent rule.

The KSC persistence component is less accurate under either conditional scheme than under the original three-branch proposal. Predator–prey weighted scores have higher RMSE than the original in all six components. These adverse comparisons remain important despite the ESS increase.

LGSSM uses exact Kalman likelihood and score. KSC uses a refined grid for the seven-mixture observation target, not the unapproximated original stochastic-volatility likelihood. Predator–prey uses the mean of two adapted Zhao–Cui local-score fits; fit/support and radius errors are unresolved. Range–bearing uses 16 particle/Fisher reference replicates at 65,536 particles, with score MCSE [0.06502,0.02268] and possible finite-particle bias. SIR uses the saved rank40/fit2 adapted Zhao–Cui reference for the displayed values and retains rank20/fit17 as a sensitivity check. The nominal T=50 second-fit score is [108.57912,-50.42205,-30.38459], compared with [102.85487,-48.60288,-30.30761] in the displayed fit; uniform has lower score RMSE in all three components against either fit.

Across the 20 usable SIR cells, the following counts require the same direction against both reference fits. They describe observed RMSE, not a statistical ranking:

| Quantity | Uniform lower | Weighted lower | Reference dependent |
| --- | --- | --- | --- |
| log_likelihood | 5 | 15 | 0 |
| infection | 13 | 3 | 4 |
| recovery | 15 | 2 | 3 |
| noise | 13 | 7 | 0 |

The implemented analytical score is the derivative of the finite calculation with categorical selections locally fixed. Local finite-difference agreement checks that quantity; it does not prove equality to the model score or to a derivative after averaging over parameter-dependent selections.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete consolidated descriptive comparison | All requested score and likelihood estimates and RMSE shown | Saved validity checks pass; inadequate SIR references excluded | Eight seeds, fixed data, numerical-reference error and untuned conditional controls | Diagnose score propagation and categorical selection effects before accuracy promotion | Universal superiority or scientific default readiness |

| Inference item | Status |
| --- | --- |
| Hard veto screen | No numerical execution failure in the reported arms; 20 SIR reference cells meet saved screens |
| Statistically supported ranking | None newly established; this aggregation performs no new ranking test. The SIR source report retains two exploratory, uncorrected componentwise findings |
| Descriptive differences | All displayed means, RMSE comparisons and counts; conditional methods remain numerically viable |
| Default readiness | Not established; heuristic losses and unresolved score errors remain promotion vetoes under the source plans |
| Next evidence needed | Mechanism checks, scope-specific calibration on disjoint data, stronger references where needed, and predeclared paired uncertainty on independent datasets |

Strongest alternative explanations are finite-reference error, controls calibrated for the original proposal, and the distinction between branchwise finite-program derivatives and model scores. More accurate references and independently calibrated repeated datasets could alter the descriptive ordering. Failure of a candidate on a score component is not a rejection of the research direction.

## Evidence and reproducibility

- Four-model report: observation-parent-four-model-results-20261028.md.
- Nonlinear extension: observation-parent-nonlinear-results-20261028.md.
- SIR campaign: observation-parent-sir-accuracy-result-20261028.md.
- This report's script, CSV, JSON and source-hash manifest: artifacts/observation-parent-all-model-comparison-20261028/.

Status: COMPLETE. This is CPU standard-library post-run reporting only, with no scientific framework import or GPU use. No experiments or background processes remain for this reporting task.
