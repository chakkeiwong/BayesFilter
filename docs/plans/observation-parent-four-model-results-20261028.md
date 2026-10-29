# Four-model likelihood and score check

All 96 evaluations completed the full horizon with finite values, finite analytical scores, and all recorded validity checks passing. Observation-weighted parents increased mean particle ESS against uniform parents in all eight paired seeds for both supported models. Accuracy did not improve consistently: the predator–prey weighted arm had larger log-likelihood RMSE and larger RMSE in all six score components than the original three-branch arm. These are descriptive comparisons, not statistically established rankings.

## Scope and reproducibility

This is one fixed parameter configuration and one saved dataset per model, with N=1008 and eight paired design seeds, 281301–281308. LGSSM, KSC SV and predator–prey use T=50 and dataset 26100831; wrapped range–bearing uses T=20 and dataset 1704. Candidate evaluation used TensorFlow/TFP on GPU, FP64, XLA enabled, TF32 disabled, with verified memory growth. FP64 is an explicit diagnostic exception to the production precision policy. The CPU reference calculations are independent checks.

The claim being checked is agreement with the model log likelihood and its parameter gradient. The filter returns the finite-particle log-likelihood estimate and its implemented analytical recursive score. The references below approximate or compute the model quantities independently. The measurements do not establish equality of the finite filter score with the exact model score or prove the implementation of every derivative term.

The conditional route currently requires affine Gaussian observations. KSC SV uses a seven-mixture observation law and range–bearing uses a nonlinear wrapped observation law, so their conditional routes are unsupported. Their runs explicitly select the original proposal; they do not validate observation-weighted parents in those models. The transition-only control uses branch weights [1,0,0] with the same recurrence and OT reset. It is not an independent standard bootstrap particle filter.

Original-proposal controls and branch weights were frozen from saved settings. They were not retuned for the conditional arms, and the saved datasets were reused for this diagnostic. These are not untouched promotion runs. Each manifest records the actual parameters, data, source hashes, command, environment, device, precision, settings and runtime.

The four successful launches consumed 364.186 seconds (6.07 minutes) of the 3,600-second execution budget; no experimental launch required a retry. The initial environment probe exposed a custom-op ABI mismatch in tf-gpu; evaluation used tftwogpu, as specified in the pre-run plan.

Execution began at commit 14e6fbd59 on branch ledh-refactor-with-policy-fix. The worktree also contained pre-existing modified numerical dependencies. The source-and-input snapshot preserves every loaded local source and reference input by its original manifest hash, including the earlier LGSSM runner version. The later runner adds predator–prey identity assertions and stronger range–bearing reference validation without changing the numerical algorithm. Unrelated live source edits are not included in the new source-code commit.

Artifacts: [summary](artifacts/observation-parent-four-model-check-20261028/summary.json), [snapshot index](artifacts/observation-parent-four-model-check-20261028/snapshot-index.json), and [pre-run plan](observation-parent-four-model-check-20261028.md). Per-model folders preserve every seed, every score component, ESS and validity traces, manifests, logs and smoke checks.

## Accuracy by model

Error is estimate minus reference. RMSE is sqrt(mean(error squared)) across the eight seeds. Score vectors are reported in the parameter order shown for each model; their components have different units, so they are not collapsed into a cross-parameter norm. All likelihood columns refer to log likelihood.

### LGSSM

Target: sqmc_lgssm_p44_d3_predict_first_v2; T=50; theta=[0.25, -1.7148, -2.12026, 0.04].

Score order: transition_parameter, log_process_variance, log_observation_variance, initial_mean_scale.

Reference (exact_Kalman): log likelihood -138.358; score [5.86731, 7.32385, 4.13012, 0.099617].

| Arm | Log-likelihood bias | Log-likelihood RMSE | Score RMSE by component | Mean ESS |
|---|---:|---:|---|---:|
| Original three-branch | -0.018837 | 0.0513889 | [0.0980073, 0.0144067, 0.0411005, 0.00193968] | 957.19 |
| Transition-only control | -1.00936 | 1.60017 | [0.835043, 0.976886, 3.16487, 0.0135532] | 198.819 |
| Conditional, uniform parents | -0.0190183 | 0.0523155 | [0.0993004, 0.0149315, 0.0414811, 0.00201672] | 957.143 |
| Conditional, weighted parents | -0.0188208 | 0.0525897 | [0.0950208, 0.0142106, 0.0424457, 0.00195216] | 957.185 |

| Arm | Mean score by component | Score bias by component |
|---|---|---|
| Original three-branch | [5.87984, 7.32594, 4.13771, 0.100207] | [0.0125212, 0.00208246, 0.00759629, 0.000590481] |
| Transition-only control | [5.68165, 7.97789, 6.78622, 0.103857] | [-0.185669, 0.654036, 2.6561, 0.00423985] |
| Conditional, uniform parents | [5.87915, 7.32557, 4.1381, 0.100258] | [0.0118378, 0.00172147, 0.00798157, 0.000640936] |
| Conditional, weighted parents | [5.86609, 7.32618, 4.14015, 0.100278] | [-0.00122568, 0.00232254, 0.010035, 0.000661012] |

Weighted versus conditional, uniform parents: ESS higher in 8/8 paired seeds; mean ESS difference 0.0417608.

Weighted versus original three-branch: ESS higher in 4/8 paired seeds; mean ESS difference -0.00474563.

The exact Kalman solution gives an unusually strong reference here. The three non-transition arms have similar observed errors. Weighted parents produce slightly lower RMSE in two score components and slightly higher RMSE in two relative to the original proposal; likelihood RMSE is also slightly higher. The ESS increase over uniform parents is only about 0.042 out of 1008 particles.

### KSC SV

Target: sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1; T=50; theta=[1.5, 0].

Score order: persistence_parameter, log_observation_scale.

Reference (independently_refined_grid): log likelihood -111.975; score [-2.0026, -0.824124].

| Arm | Log-likelihood bias | Log-likelihood RMSE | Score RMSE by component | Mean ESS |
|---|---:|---:|---|---:|
| Original three-branch | -0.0639836 | 0.206717 | [0.0744471, 0.142015] | 883.396 |
| Transition-only control | -0.135443 | 0.229338 | [0.0929878, 0.168793] | 754.146 |

| Arm | Mean score by component | Score bias by component |
|---|---|---|
| Original three-branch | [-2.02203, -0.698378] | [-0.0194338, 0.125745] |
| Transition-only control | [-2.03344, -0.696446] | [-0.030841, 0.127677] |

The independently refined grid uses 401, 801 and 1201 nodes with bound 40, plus 1201 nodes with bound 48. Maximum reported refinement differences are 5.68e-14 in log likelihood and 1.31e-14 in score. This checks the implemented seven-mixture target, rather than the original non-mixture stochastic-volatility likelihood. The observation-scale score has a visible positive mean error despite well-resolved grid refinement. The original proposal has lower observed RMSE than the transition-only control in all reported components.

### Predator–prey

Target: canonical_predator_prey_x0_then_transition_observe_v1; T=50; theta=[0.6, 114, 25, 0.3, 0.5, 0.5].

Score order: r, carrying_capacity, half_saturation, s, u, v.

Reference (adapted_Zhao_Cui_not_certified): log likelihood -267.317; score [-33.9911, -0.769203, -0.00369203, 4.04338, -4.47828, 4.87586].

| Arm | Log-likelihood bias | Log-likelihood RMSE | Score RMSE by component | Mean ESS |
|---|---:|---:|---|---:|
| Original three-branch | -0.0102833 | 0.145348 | [0.302258, 0.0228249, 0.00139447, 0.240421, 0.383359, 0.468782] | 972.369 |
| Transition-only control | -0.0863074 | 0.679934 | [1.51881, 0.156057, 0.0115473, 0.639898, 3.02541, 3.72243] | 357.297 |
| Conditional, uniform parents | 0.000662696 | 0.148873 | [0.353482, 0.0282161, 0.00150476, 0.168025, 0.423594, 0.51788] | 936.209 |
| Conditional, weighted parents | -0.0176713 | 0.17887 | [0.397328, 0.0689603, 0.00174575, 0.459779, 0.424911, 0.495951] | 972.323 |

| Arm | Mean score by component | Score bias by component |
|---|---|---|
| Original three-branch | [-34.1748, -0.782646, -0.00375312, 3.94216, -4.4094, 4.79117] | [-0.183703, -0.0134437, -6.10869e-05, -0.101219, 0.0688864, -0.0846969] |
| Transition-only control | [-33.956, -0.715923, 6.81629e-05, 3.95507, -5.44732, 6.06663] | [0.0351827, 0.0532793, 0.00376019, -0.0883134, -0.969031, 1.19076] |
| Conditional, uniform parents | [-34.2382, -0.783775, -0.00405471, 3.98077, -4.35234, 4.72069] | [-0.247013, -0.0145722, -0.00036268, -0.0626148, 0.125946, -0.155171] |
| Conditional, weighted parents | [-34.3073, -0.834925, -0.0051175, 4.47244, -4.22303, 4.60354] | [-0.316123, -0.0657228, -0.00142547, 0.42906, 0.255259, -0.272325] |

Weighted versus conditional, uniform parents: ESS higher in 8/8 paired seeds; mean ESS difference 36.1141.

Weighted versus original three-branch: ESS higher in 4/8 paired seeds; mean ESS difference -0.045654.

The central reference is the predeclared mean of two existing adapted Zhao–Cui fits, evaluated at the smallest local score radius, 0.0003125. It is not a certified oracle. The score was estimated by local quadratic fits in the physical parameter coordinates. The saved reference used 128 training and 64 holdout points, 100,000 paths and 20 jackknife groups. Its conditional standard errors exclude fit/support error and radius bias.

| Reference fit seed | Log likelihood | Score | Conditional score SE |
|---|---:|---|---|
| 41 | -267.3174480123 | [-33.9296, -0.766754, -0.00365762, 4.02452, -4.46908, 4.86609] | [0.0560195, 0.00591084, 0.000116203, 0.0254511, 0.0255629, 0.030904] |
| 53 | -267.3174382302 | [-34.0527, -0.771651, -0.00372644, 4.06225, -4.48749, 4.88563] | [0.052294, 0.00414343, 9.06649e-05, 0.0211551, 0.0207588, 0.0255363] |

All individual-fit and radius results are retained. Recomputing the arm errors against each fit separately is included in summary.json; the larger weighted likelihood RMSE and all six larger weighted score RMSEs versus original occur against each individual fit as well. Averaging the fits does not establish that their systematic errors cancel. Relative to uniform parents, weighting raises ESS by about 36 particles, yet log-likelihood RMSE and five of six score RMSEs increase. Relative to the original proposal, log-likelihood RMSE and all six score RMSEs increase. The selected conditional settings or score propagation may warrant investigation; this check does not identify the cause or reject the broader research direction.

### Range–bearing

Target: M13_wrapped_range_bearing; T=20; theta=[0, 0].

Score order: log_process_scale, log_observation_scale.

Reference (finite_particle_Fisher_reference_not_exact): log likelihood 39.8555; score [-0.334787, -5.79515].

| Arm | Log-likelihood bias | Log-likelihood RMSE | Score RMSE by component | Mean ESS |
|---|---:|---:|---|---:|
| Original three-branch | -0.218936 | 0.331358 | [0.634585, 0.758656] | 727.786 |
| Transition-only control | -0.0787775 | 0.284445 | [0.551729, 0.812272] | 510.588 |

| Arm | Mean score by component | Score bias by component |
|---|---|---|
| Original three-branch | [-0.304327, -5.23519] | [0.0304608, 0.559957] |
| Transition-only control | [-0.338022, -5.33842] | [-0.003235, 0.456725] |

The reference averages 16 particle-filter/Fisher replicates at 65,536 particles and retains the 16,384-particle comparison. MCSE for [log likelihood, score components] is [0.00708354, 0.0650232, 0.022681]. Increasing the reference particle count changes log likelihood by -0.00525051 and score by [0.135807, -0.0622632]. Monte Carlo standard errors do not bound finite-particle bias. The original arm has lower observation-scale score RMSE than the transition-only control, but higher likelihood and process-scale score RMSE. This mixed comparison prevents a dominance claim.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete the requested diagnostic | Every likelihood and score component compared | All 96 runs finite, valid, full horizon | One configuration/data set per model; numerical references for three models | Inspect the predator–prey component errors before a separate tuned validation | General accuracy or a supported ranking |
| Preserve the existing engineering default decision | This run has no promotion threshold | No numerical execution veto; incomplete heuristic comparison | Untuned conditional settings; no KSC/RB conditional implementation | Treat weighted parents as the selected engineering direction, with unresolved accuracy questions | Cross-model scientific default readiness |

| Inference item | Status |
|---|---|
| Hard veto screen | PASS: every saved validity entry, finite value/score and full horizon checked; target and reference assertions passed |
| Statistically supported ranking | None; eight paired seeds were predeclared for descriptive comparison, without a ranking test |
| Descriptive-only differences | Component RMSE, bias, seed dispersion, ESS and runtime; the mean ESS rises against uniform in all 16 supported-model pairs |
| Default readiness | Not established; KSC/RB conditional support absent, conditional settings not retuned, FP64 diagnostic rather than production-precision admission |
| Next evidence needed | Scope-specific calibration on disjoint data, untouched multi-dataset replication with declared paired uncertainty, score-term/parity checks, and stronger numerical references where required |

The heuristic-dominance check is incomplete. These runs construct a transition-only control and retain original/uniform comparisons where available, but do not supply the full set of independent heuristic filters needed for promotion. Range–bearing is already mixed against the transition-only control. No candidate is rejected by a numerical validity veto; numerical viability is distinct from an accuracy ranking.

## Post-run skeptical review

The strongest alternative explanation for the predator–prey result is that controls calibrated for the original proposal fit the changed conditional proposal poorly. Reference approximation and finite-filter derivative behavior are additional explanations. Repeating the comparison with independently calibrated conditional settings and a strengthened score reference could overturn the observed pattern. The present result shows that higher ESS alone cannot certify lower score error.

The engineering evidence consists of completed runs and checked data/source identity. Numerical evidence consists of finite full-horizon outputs and the saved validity traces. Scientific evidence consists of the descriptive reference errors with the reference limitations above. Passing the first two does not establish the third as a superiority claim. No new mathematical method, Zhao–Cui source-faithfulness claim, canonical LEDH admission, HMC claim or broad default-policy promotion is made.
