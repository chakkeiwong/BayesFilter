# Paired N32/N64 stress diagnostic, 18 October 2026

## Question and checked result

Does increasing the particle count reduce the positive-tail score error when
the saved messages and product consumer remain fixed? This second launch of
the [message/consumer campaign](../../ledh-message-consumer-continuation-20261018.md)
follows the [predeclared diagnostic](../../ledh-particle-count-stress-repair-20261018.md).
The checked artifacts are in `attempt-01/`: `manifest.json`, `result.json`,
`rows.jsonl` and `analysis.json`; full logs are adjacent to that directory.

Eight saved fits, six fresh nested clouds per fit, two particle counts and
eight arms produce 768 valid rows. The terminal audit passes 4,608 trace
checks, 73 source hashes, 15 input hashes and 16 analytical-score versus
finite-difference checks; maximum derivative error is 1.21954459864e-8.
Neither numerical validity nor source/coverage checks fire a continuation veto.
All 14 predeclared simultaneous paired contrasts remain unresolved. No ranking
is statistically supported.

## Conditional score error

This table is squared error of score coordinate 1, averaged over the same
positive-tail data and fixed fits; it is descriptive, not a ranking.

| Arm | N32 MSE | N64 MSE |
| --- | ---: | ---: |
| Unguided identity | 13.485263 | 7.329337 |
| Unguided global | 1.460713 | 0.208329 |
| Unguided local | 3.503439 | 0.600185 |
| Unguided equal mixture | 3.324714 | 0.467409 |
| Gaussian-future global | 0.736282 | 0.162246 |
| Gaussian-future equal mixture | 2.088151 | 0.321812 |
| Fitted product/global | 0.866750 | 0.205119 |
| Exact product/global | 1.010306 | 0.204109 |

Both product arms lose descriptively to Gaussian-future global in coordinate 1
at N64. This is a heuristic promotion veto, not statistical evidence of harm.
The N64-minus-N32 differences are -0.661631 for the fitted product and -0.806197
for the exact product. Their simultaneous 95% intervals are respectively
[-1.565620, 0.242358] and [-1.842354, 0.229960]. These intervals do not establish
a particle-count effect. At N64, fitted-minus-unguided-global MSE is -0.003210
[-0.175248, 0.168827]; exact-minus-unguided-global is -0.004220
[-0.202635, 0.194195]. The saved-fit conditional analysis does not estimate
generalization to new fits or datasets.

## Decision and inference status

| Item | Status |
| --- | --- |
| Decision | Reject promotion of both tested product candidates; continue the bounded mechanism diagnosis. |
| Primary criterion | All 14 simultaneous contrasts include zero; no established count effect or gain. |
| Veto diagnostics | Numerical/reference/derivative/coverage audits pass; descriptive heuristic losses veto promotion only. |
| Main uncertainty | Eight saved-fit groups, six clouds per group, and small-cluster t calibration. |
| Next justified action | Fresh nested N64/N128 clouds with twelve replicates under the final permitted launch. |
| Not concluded | No asymptotic bias, fitting-capacity, canonical LEDH, scope-tuning, default or HMC conclusion. |

| Inference class | Status |
| --- | --- |
| Hard veto screen | No continuation veto; both candidates remain numerically viable. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Observed MSE reductions with count and Gaussian-future losses. |
| Default-readiness | Not established; this is an FP64 GPU/XLA diagnostic exception. |
| Next evidence needed | Larger fresh paired clouds for count sensitivity, then a consumer/reset attribution study if needed. |

The current candidates fail promotion, not the backward-guidance research
direction. Exact-message residual error makes fitting error alone an inadequate
explanation, but does not isolate reset, observation approximation, flow or
finite-particle effects. The strongest alternative explanation is sampling
variation in a few conditional groups. Replicated contrasts with intervals
excluding zero would change the present unresolved verdict; a new-fit study
would still be needed for generalization.

## Reproducibility and automatic continuation

The manifest records commit `ea1018de1da55ed02a29a8d60f08a6827fce8a42`, dirty
source hashes, `tftwogpu`, GPU device, verified memory growth, FP64/XLA with
TF32 disabled, data, seeds and exact command. The GPU command is
`TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_particle_count_stress_diagnostic.py --budget-seconds 600 --output docs/plans/artifacts/ledh-particle-count-stress-20261018/attempt-01`.
The CPU terminal analyzer deliberately hides GPUs with `CUDA_VISIBLE_DEVICES=-1`.
GPU-process wall time is 160.733202406 seconds; CPU reference plus analysis is
1.705762587 seconds. Cumulative campaign use is 646.162868663/1,200 GPU-process
seconds, two of three launches, and 3.464404211/180 CPU reference/analysis seconds.

The next phase has already been audited and launched under
`../../ledh-particle-count-followup-20261018.md`, capped at 500 seconds within
the remaining 553.837131337. This is the automatic repair/refresh mechanism:
audit the evidence, classify the failure, update the contract and budget, then
execute within existing authorization. Phase completion itself creates no
permission gate. The final launch limit remains binding. Original runner and
analyzer snapshots are preserved before the follow-up driver refresh; the
completed evidence is not overwritten.
