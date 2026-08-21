# GenUT Four-Model Leaderboard Rerun Result

Date: 2026-08-16  
Plan: `docs/plans/bayesfilter-genut-four-model-leaderboard-rerun-plan-2026-08-16.md`  
Terminal artifact: `docs/benchmarks/artifacts/genut_four_model_leaderboard_rerun_20260816/attempt04/`  
Status: complete comparison artifact; no default or HMC promotion

## Question And Scope

The question was whether the established `b=0.98` coordinate cap plus radial
RMS cap GenUT ladder behaves consistently across LGSSM, KSC-SV, predator-prey,
and Austria-SIR. The tested arms were diagonal, pairwise, coordinate cap, and
dual cap. The run used `N=1008`, FP32, XLA, deterministic stateless seeds,
and the current `tftwogpu` TensorFlow `2.20.0-dev0+selfbuilt` environment.

This is the legacy dual-cap leaderboard route. It does not test the newer
`genut_column_scaled_lm_smooth_rms_trust_v1` route because the harness does not
set its LM damping, scale-floor, or trust-radius controls.

## Engineering Repair

The first launch failed before evaluation because the checked-in custom op was
linked against the old `tf-gpu` TensorFlow/Abseil ABI. Rebuilding against
`tftwogpu` initially omitted CUDA registration because the environment bundles
CUDA under `site-packages/nvidia/cuda_runtime`. `CMakeLists.txt` now searches
that environment-local include and library path. The old binary is preserved
as `bayesfilter/ops/_symmetric_sylvester_ops.so.pre_tftwogpu_20260816`.

The rebuilt op was checked by eager and XLA GPU Sylvester/principal-square-root
probes. Both visible GPUs were detected in the probe, with memory growth
verified before logical-device initialization. The leaderboard harness also
constructs setup-only fixtures on CPU after memory-policy verification; claim
evaluators remain pinned to GPU:0.

Some separate retries terminated during TF32-enabled or cap-arm XLA execution
without a Python traceback. These are preserved as failed attempts (`attempt01`
through `attempt03`, plus the partial TF32-off `attempt05`) and are not treated
as numerical evidence. The terminal `attempt04` serialized all 16 cells.

## Hard Validity

All 16 model/arm cells in `attempt04` passed the declared hard gates:

| Model | Horizon | Diagonal | Pairwise | Coordinate cap | Dual cap |
|---|---:|---:|---:|---:|---:|
| LGSSM | 50 | pass | pass | pass | pass |
| KSC-SV | 10 | pass | pass | pass | pass |
| Predator-prey | 20 | pass | pass | pass | pass |
| Austria-SIR | 20 | pass | pass | pass | pass |

Pass means finite value and score, valid program status, and maximum declared
mean/row/column/score-increment residual below `5e-4` for both calibration and
claim seeds. The largest observed residual was below the gate in every cell.

The internal same-program finite-difference diagnostic passed for all KSC-SV
arms. It failed its predeclared tolerance for LGSSM, predator-prey, and
Austria-SIR. This diagnostic is explanatory, not an external score oracle, and
does not overturn the hard finite/residual result.

## Representative Values And Score Variability

The terminal artifact contains full per-seed rows. Representative 16-seed
means (sample SD in parentheses) are:

| Model | Arm | Value mean (SD) | Score mean vector |
|---|---|---:|---|
| LGSSM | diagonal | -136.3355 (0.4680) | [5.7922, -4.0508, 0.2395, -1.9765, 5.5621] |
| LGSSM | pairwise | -136.3344 (0.4707) | [5.7812, -4.0328, 0.2203, -2.0496, 5.5516] |
| LGSSM | coordinate cap | -136.3350 (0.4698) | [5.6895, -3.9893, 0.1983, -2.1028, 5.5629] |
| LGSSM | dual cap | -136.3319 (0.4677) | [5.7071, -3.9915, 0.2029, -2.0920, 5.5298] |
| KSC-SV | diagonal/pairwise | -19.95395 (0.04760) | [-0.6944, 0.6077] |
| KSC-SV | coordinate/dual | -19.95785 (0.04894) | [-0.7068, 0.5755] |
| Predator-prey | diagonal | -102.7391 (0.2923) | [-27.7734, 0.0778, -0.0875, 1.0416, 18.3634, -23.6463] |
| Predator-prey | pairwise | -102.7442 (0.3071) | [-27.8043, 0.0724, -0.0877, 1.0409, 18.4310, -23.7305] |
| Predator-prey | coordinate cap | -102.7264 (0.3064) | [-27.7679, 0.0759, -0.0871, 1.0164, 18.2564, -23.5180] |
| Predator-prey | dual cap | -102.7272 (0.3054) | [-27.7464, 0.0755, -0.0870, 1.0146, 18.2419, -23.4998] |
| Austria-SIR | diagonal | -682.7208 (1.2381) | [495.4189, -353.7767, -90.2724] |
| Austria-SIR | pairwise | -682.0848 (1.5107) | [91.3748, -148.1293, -27.9601] |
| Austria-SIR | coordinate cap | -681.6991 (0.4016) | [31.4426, -100.9350, 13.0493] |
| Austria-SIR | dual cap | -681.6369 (0.7310) | [8.5927, -104.9195, -2.5832] |

The Austria diagonal score remains much more variable than its capped arms.
The caps materially change the score means in that model, so lower spread is a
descriptive stability observation only, not evidence of unbiasedness.

## Scope And Reproducibility

The unchanged predator-prey and Austria observation hashes match the prior
2026-08-07 artifact. The current checkout regenerated LGSSM and KSC fixtures:

| Model | Prior source hash | Current source hash | Interpretation |
|---|---|---|---|
| LGSSM | `9d16f6ad...` | `8aa2e810...` | new scope; old controls warm start only |
| KSC-SV | `65553817...` | `b223a996...` | new scope; old controls warm start only |
| Predator-prey | `fea0681d...` | same | reproducibility scope |
| Austria-SIR | `cd794ad6...` | same | reproducibility scope |

The harness admits only the two explicitly measured regenerated hashes and
rejects any other mismatch. Calibration rows passed for every terminal cell,
but no fresh grid search was performed for regenerated LGSSM/KSC; inherited
controls therefore remain warm starts, not newly promoted defaults.

## Decision And Inference Status

| Decision | Status | Evidence | Limitation |
|---|---|---|---|
| Four-model finite/residual viability | pass | 16/16 hard-valid cells | no exact nonlinear oracle |
| Similar behavior across models | descriptive pass | all models finite; capped Austria spread lower | raw scales differ |
| Statistically supported arm ranking | not established | paired 16-seed summaries | no predeclared superiority test |
| Current LM/trust-region route | not tested | legacy controls only | separate harness required |
| Default/HMC/NeuTra readiness | not established | no downstream posterior run | FD failures and Austria variability remain |

Hard vetoes: none in terminal `attempt04`. Failed intermediate launches are
preserved infrastructure/route evidence, not numerical evidence against GenUT.

## Nonclaims

This result does not prove exact nonlinear likelihood or score correctness,
absence of systematic bias, statistical superiority of dual cap, posterior
correctness, NeuTra training readiness, HMC readiness, production default
readiness, or source-faithful Zhao-Cui equivalence.
