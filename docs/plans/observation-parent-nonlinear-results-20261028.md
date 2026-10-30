# KSC SV and range-bearing observation-weighted score results

Both requested models now run the conditional observation-weighted proposal through the existing analytical recurrence. All 64 full-horizon runs pass the saved numerical validity checks. The observed score results are mixed: weighting has lower error in one parameter and higher error in the other, compared with the original three-branch method, in each model. Higher ESS does not establish more accurate scores.

## Design and implementation

One saved configuration and dataset per model; N=1008; eight paired seeds 281301–281308. KSC uses T=50 and theta=[1.5,0]; range-bearing uses T=20 and theta=[0,0]. All four arms use the saved controls and branch weights. These controls were not tuned for the new proposal. FP64 GPU/XLA with TF32 disabled is an explicit diagnostic precision exception. TensorFlow verified memory growth before initialization. The actual recorded device is RTX 5080; CUDA device numbering did not match the nvidia-smi index assumed at launch. The runs used at most 131 MiB of TensorFlow allocator memory and did not change device settings.

KSC conditions exactly on each of the seven mixture components, selecting a parent/component pair using its observation likelihood. The marginal conditional density is evaluated using the exact mixture Bayes identity. The uniform-parent arm retains conditional component probabilities within each parent. Range-bearing instead uses a local linear Gaussian approximation to the polar observation, including wrapped residuals, and evaluates its weighted Gaussian-mixture proposal density explicitly. Its target remains the original wrapped observation law. This is an approximate conditional proposal, not an exact range-bearing conditional distribution.

The optional component-weight tangent was added to the existing nonshared Gaussian-mixture density kernel. The shared analytical score recurrence calls that kernel; no separate model-specific score recurrence or autodiff score was introduced. The existing proposal-policy setting was not changed in this task. The formerly rejected KSC and range-bearing laws now have implementations behind it, but this diagnostic does not establish scientific default readiness.

## Scores and likelihoods

Scores below are means over eight seeds. RMSE is computed across the individual seed errors, not from the mean score alone. Every difference in these tables is descriptive.

### KSC SV

Parameter order: persistence_parameter, log_observation_scale. Branch weights: [0.1, 0.9, 0].

Reference score [-2.0026, -0.824124]; log likelihood -111.9746052563.

| Method | Mean score | Score RMSE | Log-likelihood bias | Log-likelihood RMSE | Mean ESS |
|---|---|---|---:|---:|---:|
| Original three-branch | [-2.02203, -0.698378] | [0.0744471, 0.142015] | -0.0639836 | 0.206717 | 883.396 |
| Transition only | [-2.03344, -0.696446] | [0.0929878, 0.168793] | -0.135443 | 0.229338 | 754.146 |
| Conditional, uniform parents | [-1.39563, -0.436223] | [0.609442, 0.389578] | -0.0505524 | 0.135892 | 941.015 |
| Conditional, weighted parents | [-1.59661, -0.733437] | [0.407061, 0.0944128] | -0.102447 | 0.115135 | 996.650 |

| Method | Score standard deviation across seeds |
|---|---|
| Original three-branch | [0.0768278, 0.0705588] |
| Transition only | [0.0937813, 0.118029] |
| Conditional, uniform parents | [0.0585894, 0.0386133] |
| Conditional, weighted parents | [0.0315575, 0.0280742] |

Weighted parents have higher mean ESS in 8/8 paired seeds versus original three-branch; average increase 113.254 particles.
Weighted parents have higher mean ESS in 8/8 paired seeds versus conditional, uniform parents; average increase 55.636 particles.

The refined grid evaluates the same seven-mixture target, not the original unapproximated stochastic-volatility likelihood. Its 401/801/1201-node and domain-bound checks differ by at most 5.69e-14 in log likelihood and 1.32e-14 in score. Relative to the original proposal, weighted conditioning has much larger persistence-score error (RMSE 0.07445 to 0.40706), lower observation-scale error (0.14201 to 0.09441), and lower likelihood RMSE (0.20672 to 0.11513). Relative to uniform conditional parents, both score RMSEs and likelihood RMSE are lower. The persistence result prevents a general score-accuracy claim.

### Range–bearing

Parameter order: log_process_scale, log_observation_scale. Branch weights: [0.1, 0.025177, 0.874823].

Reference score [-0.334787, -5.79515]; log likelihood 39.8554771041.

| Method | Mean score | Score RMSE | Log-likelihood bias | Log-likelihood RMSE | Mean ESS |
|---|---|---|---:|---:|---:|
| Original three-branch | [-0.304327, -5.23519] | [0.634585, 0.758656] | -0.218936 | 0.331358 | 727.786 |
| Transition only | [-0.338022, -5.33842] | [0.551729, 0.812272] | -0.0787775 | 0.284445 | 510.588 |
| Conditional, uniform parents | [-0.130327, -5.45566] | [0.426624, 0.715763] | -0.232366 | 0.349486 | 718.443 |
| Conditional, weighted parents | [-0.252753, -5.394] | [0.502737, 0.817229] | -0.142341 | 0.231912 | 733.762 |

| Method | Score standard deviation across seeds |
|---|---|
| Original three-branch | [0.677618, 0.547208] |
| Transition only | [0.589813, 0.718083] |
| Conditional, uniform parents | [0.400291, 0.673637] |
| Conditional, weighted parents | [0.530245, 0.761161] |

Weighted parents have higher mean ESS in 8/8 paired seeds versus original three-branch; average increase 5.977 particles.
Weighted parents have higher mean ESS in 8/8 paired seeds versus conditional, uniform parents; average increase 15.320 particles.

The reference averages 16 Fisher/particle-filter replicates at 65,536 particles; it is not an exact oracle. Its MCSE for [log likelihood, process-scale score, observation-scale score] is [0.00708,0.06502,0.02268]. Moving from 16,384 to 65,536 particles changes the scores by [0.13581,-0.06226]; MCSE does not bound finite-particle bias. Relative to the original proposal, the weighted arm has lower process-scale score RMSE (0.63459 to 0.50274), higher observation-scale score RMSE (0.75866 to 0.81723), and lower likelihood RMSE (0.33136 to 0.23191). Relative to uniform conditional parents, both score RMSEs are higher despite higher ESS. Its observation-scale RMSE also slightly exceeds the transition-only control.

## What the score check establishes

The target quantity for accuracy is the derivative of the model marginal log likelihood, in the recorded parameter coordinates. The implemented quantity is the analytical derivative of the finite particle/transport value program, with categorical selections and angle winding numbers locally fixed. Density/tangent checks and full-program finite differences pass for both nonlinear observation laws and both conditional policies. That verifies this local derivative calculation on the tested cases; it does not establish equality to the model score. In particular, finite differences away from categorical thresholds do not test the contribution from changing selections to an expectation over random designs. This is a concrete possible explanation for persistent score bias, alongside transport/reset approximation and inherited controls; the present run does not distinguish them.

## Decision and inference status

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete the requested comparison | All score components and likelihood errors recorded for both models | All 64 runs finite, valid, full horizon; all validity trace entries pass | Eight seeds and one dataset per model; approximate references | Preserve component results and examine KSC persistence bias | General score improvement |
| Retain as a numerically viable diagnostic implementation | Density and derivative checks pass; accuracy is mixed | No implementation/harness validity veto; uniform/transition comparisons prevent dominance | Selection-boundary effects, reset approximation, untuned allocation | Small controlled derivative-of-expectation versus local-derivative diagnostic before another accuracy campaign | Model-score correctness, HMC readiness, canonical LEDH admission, scientific default readiness |

| Inference item | Status |
|---|---|
| Hard veto screen | PASS for the stated numerical, density, derivative and data checks; this does not certify score accuracy |
| Viable candidates | All four arms remain numerically viable for each model |
| Statistically supported ranking | None; no predeclared paired ranking test or interval was part of this diagnostic |
| Descriptive differences | All reported means, RMSEs, dispersions, ESS counts and runtime differences |
| Default readiness | Not established; one component is worse in each model versus original, controls are inherited, and heuristic coverage is incomplete |
| Next evidence needed | Discriminate selection-boundary and reset effects, then scope-specific tuning on separate data and untouched multi-dataset paired inference, including reference uncertainty |

## Reproducibility and attempts

Plan: observation-parent-nonlinear-extension-20261028.md. Checkout: .localresources/worktrees/ledh-iapf-execution-20260922; branch ledh-refactor-with-policy-fix; base commit 127309b45285f61d77b2c470250e857806ab39ee. Unrelated working-tree edits were preserved. Each run saves imported source snapshots and SHA-256 hashes, exact inputs and reference paths/hashes, settings, environment, command, memory policy, seeds, timing, per-seed output and all validity traces. All saved source snapshots and input hashes were rechecked after execution.

CPU test command: CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_observation_parent_nonlinear.py tests/highdim/test_observation_parent_default.py tests/highdim/test_covariance_proposal_shared_gaussian.py. Result: 40 passed in 61.32 seconds; GPU intentionally hidden; graph/non-XLA independent checks.

GPU command prefix: CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 timeout 900 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_observation_parent_nonlinear_extension.py. Arguments are --model ksc --output docs/plans/artifacts/observation-parent-nonlinear-extension-20261028/ksc-02 and --model range_bearing --output docs/plans/artifacts/observation-parent-nonlinear-extension-20261028/range-bearing-01. Both executed with trusted/elevated GPU permissions.

Attempt 1 (ksc-01) failed at Python parse time due to an extra closing parenthesis in the runner; no scientific computation or output directory was created. Its complete log is preserved. The fix passed py_compile. KSC attempt 2 completed in 91.11 seconds; range-bearing attempt 1 completed in 136.74 seconds. One CPU test attempt and three GPU-launch attempts were consumed. The CPU test plus successful GPU runs consumed 289.17 seconds; conservatively charging 60 seconds for the failed launch and initialization leaves at least 3250 seconds of the 3600-second budget unused. Five GPU launches and three CPU test attempts remain; no additional campaign was launched.

Artifacts: artifacts/observation-parent-nonlinear-extension-20261028/{ksc-02,range-bearing-01}/{manifest,result,rows,status,smoke}.json; source_snapshot/; complete logs; summary.json; this post-run summarizer.

## Post-run skeptical review

The strongest alternative explanation is that better weight concentration reduces likelihood variability while the local derivative through categorical selection remains a poor approximation to the model score. KSC likelihood bias is more negative under weighting even though its RMSE falls. Range-bearing reference finite-particle error and the small conditional-branch allocation weaken causal claims there. More particles, independent datasets, explicit selection-boundary diagnostics and an independently constructed score would overturn or sharpen this interpretation. Nothing in these runs invalidates the data, harness, density formula or tested local analytical derivative. The result is mixed candidate accuracy, not rejection of conditional proposals as a research direction. There is no planned downstream repair phase in this bounded diagnostic; further tuning or broad validation requires its own evidence contract.
