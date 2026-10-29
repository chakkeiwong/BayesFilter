# Epsilon 1, five Sinkhorn iterations: paired materiality results

Baseline: current corrected-source selected controls. Candidate: epsilon=1, iterations=5; all other controls and mixture weights held fixed.
N4032/K2016, GPU/XLA, TF32 off; FP32 except SIR FP64 recursion/FP32 pairs.
Saved plus two fresh datasets, six local parameter positions per model. These are filter log likelihoods and physical-parameter scores, not log priors or posterior samples.

Changes below use 100*abs(candidate-baseline)/max(1,abs(baseline)). The 1% screen is componentwise; max score change is the largest component over all inspected inputs.

| Model | Status | Max log-likelihood change % | Max score change % | Max absolute log-likelihood change | Max centered change | Baseline / candidate seconds per B2 call |
|---|---|---:|---:|---:|---:|---:|
| ksc | MATERIAL_DIFFERENCE | 0.0952942 | 7.29252 | 0.113701 | 0.00953674 | 8.020 / 1.838 |
| lgssm | WITHIN_ONE_PERCENT_ON_INSPECTED_INPUTS | 0.000128416 | 0.026363 | 0.000167847 | 0.000106812 | 21.850 / 8.213 |
| predator_prey | MATERIAL_DIFFERENCE | 0.0184672 | 5.57833 | 0.0601501 | 0.0733032 | 14.104 / 7.894 |
| range_bearing | MATERIAL_DIFFERENCE | 0.0433676 | 18.6176 | 0.0180244 | 0.0177078 | 5.155 / 2.631 |
| sir_d18 | MATERIAL_DIFFERENCE | 0.0790664 | 1482.48 | 1.41367 | 1.40324 | 21.462 / 14.849 |

Centered change subtracts each arm's nominal-parameter log likelihood within the same dataset. It is explanatory, not an additional predeclared veto.

## Nominal parameter, saved dataset

| Model | Baseline log likelihood | Candidate log likelihood | Baseline score | Candidate score |
|---|---:|---:|---|---|
| ksc | -112.03359985351562 | -112.05841064453125 | -2.0497148, -0.77476215 | -2.0793977, -0.70350921 |
| lgssm | -138.3419189453125 | -138.3418731689453 | 5.8540277, 7.3201647, 4.1385508, 0.10007131 | 5.8544421, 7.3201647, 4.1385088, 0.10007129 |
| predator_prey | -267.2937316894531 | -267.2898254394531 | -34.027348, -0.76881051, -0.0025934263, 4.0212712, -4.7631063, 5.2267089 | -34.000763, -0.76525712, -0.0016144174, 3.9706991, -5.0015507, 5.5182719 |
| range_bearing | 40.2177848815918 | 40.21420669555664 | -0.70928067, -6.738297 | -0.76136434, -6.6700211 |
| sir_d18 | -1716.7618403491638 | -1716.7514085490743 | -175.02541, 98.351819, 56.140837 | -170.1669, 90.86408, 59.453225 |

## Transport and reset diagnostics

| Model | Arm | Max Sinkhorn row residual | Max column residual | Max reset covariance relative error |
|---|---|---:|---:|---:|
| ksc | baseline | 0.000583891 | 1.40667e-05 | 4.8611e-07 |
| ksc | candidate | 0.00866052 | 3.45707e-06 | 5.16607e-07 |
| lgssm | baseline | 8.73115e-10 | 1.78814e-06 | 3.74337e-06 |
| lgssm | candidate | 8.67294e-09 | 1.3113e-06 | 3.21294e-06 |
| predator_prey | baseline | 1.22845e-05 | 9.29832e-06 | 3.89422e-06 |
| predator_prey | candidate | 0.000955602 | 6.3777e-06 | 3.53782e-06 |
| range_bearing | baseline | 2.6077e-08 | 1.3113e-06 | 3.03869e-05 |
| range_bearing | candidate | 2.6077e-08 | 1.43051e-06 | 2.76384e-05 |
| sir_d18 | baseline | 0.142929 | 7.14064e-05 | 4.82821e-15 |
| sir_d18 | candidate | 0.168516 | 5.47171e-05 | 6.23405e-15 |

## Decision and inference status

| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain existing selections while reporting this ablation | Per-model 1% comparison above | See invalid-arm/source status | Limited data/parameter coverage; both epsilon and iterations change | Assess model-specific materiality before any new tuning | No default or HMC promotion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Kernel validity, finite output, full horizon and source identity retained; failures shown explicitly |
| Statistically supported ranking | None; no runtime or method-quality superiority claim |
| Descriptive-only differences | Exact differences on these frozen inputs and three warmed timing repeats |
| Default-readiness | No changes to selected controls or default precision |
| Next evidence needed | Broader target-specific assessment for any proposed default change; separate HMC tuning and posterior checks |

Post-run red team: the selected baseline is not a model oracle. Small relative changes can coexist with an important absolute or centered log-likelihood change. Frozen beta isolates the requested intervention but may not be optimal for it. The weakest evidence is coverage outside six local positions and three datasets. A different result on a new dataset or posterior region would overturn a broad similarity claim. This experiment does not estimate likelihood MCSE, because its fresh replications change datasets as well as particle designs.

Charged worker-wall GPU time: 0.4137 h. All per-position values, scores and health traces are retained beside this report.

## Interpretation of this comparison

The tested pair means epsilon=1 and five alternating row/column Sinkhorn updates.
The baseline epsilon/iteration pairs were LGSSM (0.5,80), KSC (0.25,40),
predator-prey (1,40), SIR (0.5,40), and range bearing (4,40). Hence only
predator-prey isolates iteration count. This experiment cannot establish that
40 or 80 updates are necessary for the other models.

All five arms pass the existing numerical validity checks. LGSSM is numerically
similar under the declared 1% screen on these inputs. The other four have
material score differences. These are differences between finite likelihood
programs; the selected baseline is not an oracle and the comparison does not
establish which score is closer to the true model score.

The largest SIR component percentage is magnified by a baseline component of
-1.32578 changing to -20.98031. The same row changes the full score vector by
113.774% of its baseline norm, including a sign change in another component;
the effect is not confined to a near-zero denominator. Its centered likelihood
change also reaches 1.40324 log units despite the small percentage of the total
log likelihood. The other models have much smaller centered differences.

The five-iteration candidate has measured B2 evaluation-time ratios of 0.376,
0.229, 0.560, 0.692 and 0.510 versus each baseline, respectively. Three warmed
repeats are descriptive runtime evidence, not a statistically supported ranking.
A separate same-epsilon iteration ablation would distinguish the mechanisms;
it was not executed or substituted for this requested comparison. It would
refine the existing master computational-cost work, not change its target.

Terminal verification: 90 paired rows, 45 health files, finite outputs, all
existing validity traces, complete filtering horizons, fixed non-OT controls,
and snapshot/current-source hashes passed. The parent and all workers exited.
No default or tuning authority was changed. Full raw nominal vectors and every
local-position pair are retained in the model result JSON files.
