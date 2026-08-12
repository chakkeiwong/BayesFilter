# Zhao-Cui/GenUT Dual-Cap Cross-Model Result

Date: 2026-08-07

Status: `FD_VETO`

Primary artifacts:
- `docs/benchmarks/artifacts/zhao_cui_genut_dual_cap_cross_model_20260807/attempt03/result.json`
- `docs/benchmarks/artifacts/zhao_cui_genut_dual_cap_cross_model_20260807/attempt03/result.md`
- `docs/benchmarks/artifacts/zhao_cui_genut_dual_cap_cross_model_20260807/attempt04_fd_attribution/result.json`

## Verdict

The dual-cap cross-model test does not support a universal successful algorithm. All selected rows were finite and transport-valid, but only KSC SV passed the predeclared same-scalar finite-difference gate. LGSSM and predator-prey failed that gate. The separate FD-step attribution shows that their diagonal baselines also fail at `h=1e-3`, so those two failures are inherited FP32/TF32 derivative diagnostics rather than cap-only failures. The capped map is therefore a viable numerical experiment for KSC, but not an admitted score route for the other two models under this contract.

## Scope

| Model | Horizon | N | Selected arm | Claim rows | FD status | Reference role |
|---|---:|---:|---|---:|---|---|
| lgssm_T50 | 50 | 1008 | pairwise_coordinate_cap | 6 | FAIL | Kalman exact value/score |
| ksc_sv_T10 | 10 | 1008 | dual_cap | 6 | PASS | dense transformed-mixture diagnostic |
| predator_prey_T20 | 20 | 1008 | pairwise_coordinate_cap | 6 | FAIL | none |

## Detailed Results

### lgssm_T50

Selected controls: `{'balance_steps': 8, 'coordinatewise_standardized_cap': 0.98, 'coordinatewise_standardized_cap_power': 8, 'epsilon': 2.0, 'higher_moment_correction_steps': 4, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.2, 'pairwise_moment_correction_steps': 4, 'pairwise_moment_floor': 1e-05, 'pairwise_moment_strength': 0.02, 'pairwise_particle_rms_cap': 0.0, 'ridge': 1e-05, 'sinkhorn_steps': 8}`

| Metric | Baseline | Selected |
|---|---:|---:|
| value mean | -136.524089 | -136.559807 |
| value sample SD | 0.311065389 | 0.341392348 |
| score_0 mean | 6.30811906 | 6.06628537 |
| score_0 sample SD | 2.15294621 | 2.13572439 |
| score_1 mean | -4.10177286 | -4.03423889 |
| score_1 sample SD | 0.657831968 | 0.653117656 |
| score_2 mean | 0.148978581 | 0.138097291 |
| score_2 sample SD | 0.372321218 | 0.350196549 |
| score_3 mean | -1.036433 | -1.10465936 |
| score_3 sample SD | 2.90607405 | 2.78885669 |
| score_4 mean | 7.03632053 | 7.34644206 |
| score_4 sample SD | 2.72584394 | 2.76260398 |

Cap mean displacement: `0.239739`; maximum active fraction: `0.683201`; minimum cap derivative: `2.46463e-06`.

FD rows:

| Parameter | Absolute residual | Normalized residual | Gate |
|---:|---:|---:|---|
| 0 | 0.333217621 | 0.0581816025 | FAIL |
| 1 | 0.246581554 | 0.0510325581 | FAIL |
| 2 | 0.0235163867 | 0.0235163867 | PASS |
| 3 | 0.0523911715 | 0.0511955768 | FAIL |
| 4 | 0.932461739 | 0.161286354 | FAIL |

Reference: `{'kind': 'exact_affine_kalman_analytical_score', 'score': [5.65544620078834, -3.835056867532365, 0.3023618942730066, -1.9171764199511037, 4.35427564127945], 'value': -136.07597463460453}`

Paired candidate-minus-baseline shifts are descriptive only:

- `score_0`: mean `-0.24183369`, SD `0.26410264`, signs `-/5 +/1`.
- `score_1`: mean `0.06753397`, SD `0.12654566`, signs `-/2 +/4`.
- `score_2`: mean `-0.01088129`, SD `0.084235453`, signs `-/4 +/2`.
- `score_3`: mean `-0.068226357`, SD `0.20067378`, signs `-/2 +/4`.
- `score_4`: mean `0.31012154`, SD `0.48509293`, signs `-/1 +/5`.
- `value`: mean `-0.035718282`, SD `0.059601567`, signs `-/5 +/1`.
### ksc_sv_T10

Selected controls: `{'balance_steps': 8, 'coordinatewise_standardized_cap': 0.98, 'coordinatewise_standardized_cap_power': 8, 'epsilon': 2.0, 'higher_moment_correction_steps': 4, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.2, 'pairwise_moment_correction_steps': 4, 'pairwise_moment_floor': 1e-05, 'pairwise_moment_strength': 0.02, 'pairwise_particle_rms_cap': 2.0, 'ridge': 1e-05, 'sinkhorn_steps': 8}`

| Metric | Baseline | Selected |
|---|---:|---:|
| value mean | -19.9847047 | -19.9884758 |
| value sample SD | 0.0428153539 | 0.0431492377 |
| score_0 mean | -0.701667647 | -0.71548596 |
| score_0 sample SD | 0.016273556 | 0.0182400495 |
| score_1 mean | 0.625490641 | 0.591312319 |
| score_1 sample SD | 0.0969988645 | 0.090558383 |

Cap mean displacement: `0.228122`; maximum active fraction: `0.749008`; minimum cap derivative: `1.33535e-05`.

FD rows:

| Parameter | Absolute residual | Normalized residual | Gate |
|---:|---:|---:|---|
| 0 | 8.9943409e-05 | 8.9943409e-05 | PASS |
| 1 | 0.000510811806 | 0.000510811806 | PASS |

Reference: `{'fd_order_gap': 0.0, 'fd_step_gap': 2.3684743055696345e-10, 'fd_steps': [1e-05, 3e-05], 'kind': 'sequential_dense_transformed_mixture_value_converged_fd_score_diagnostic', 'orders': [401, 601], 'radius': 8.0, 'score': [-0.7056728072403947, 0.6354925923564755], 'score_provenance': 'diagnostic centered finite difference of converged dense value', 'value': -19.956279204514765, 'value_order_gap': 1.0302869668521453e-13}`

Paired candidate-minus-baseline shifts are descriptive only:

- `score_0`: mean `-0.013818314`, SD `0.006660597`, signs `-/6 +/0`.
- `score_1`: mean `-0.034178322`, SD `0.0084456593`, signs `-/6 +/0`.
- `value`: mean `-0.0037711461`, SD `0.0031666605`, signs `-/5 +/1`.
### predator_prey_T20

Selected controls: `{'balance_steps': 8, 'coordinatewise_standardized_cap': 0.98, 'coordinatewise_standardized_cap_power': 8, 'epsilon': 2.0, 'higher_moment_correction_steps': 4, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.2, 'pairwise_moment_correction_steps': 4, 'pairwise_moment_floor': 1e-05, 'pairwise_moment_strength': 0.05, 'pairwise_particle_rms_cap': 0.0, 'ridge': 1e-05, 'sinkhorn_steps': 8}`

| Metric | Baseline | Selected |
|---|---:|---:|
| value mean | -102.7609 | -102.742517 |
| value sample SD | 0.314150828 | 0.292251285 |
| score_0 mean | -27.2595002 | -27.1445548 |
| score_0 sample SD | 0.642530602 | 0.615160739 |
| score_1 mean | 0.0953194847 | 0.100458997 |
| score_1 sample SD | 0.0747877053 | 0.0772287129 |
| score_2 mean | -0.0876026886 | -0.0870213285 |
| score_2 sample SD | 0.00401202991 | 0.00411160935 |
| score_3 mean | 0.725335876 | 0.6117585 |
| score_3 sample SD | 0.328605422 | 0.390282141 |
| score_4 mean | 18.6205311 | 18.5379903 |
| score_4 sample SD | 1.08236747 | 0.910856701 |
| score_5 mean | -23.957613 | -23.8548641 |
| score_5 sample SD | 1.30870839 | 1.11726873 |

Cap mean displacement: `0.229211`; maximum active fraction: `0.684028`; minimum cap derivative: `3.17636e-07`.

FD rows:

| Parameter | Absolute residual | Normalized residual | Gate |
|---:|---:|---:|---|
| 0 | 0.0919475555 | 0.00337673747 | FAIL |
| 1 | 0.0953889638 | 0.0953889638 | FAIL |
| 2 | 0.00255079567 | 0.00255079567 | PASS |
| 3 | 0.271680832 | 0.254554927 | FAIL |
| 4 | 0.4562397 | 0.0242922604 | FAIL |
| 5 | 0.436693192 | 0.0179951396 | FAIL |
Paired candidate-minus-baseline shifts are descriptive only:

- `score_0`: mean `0.11494541`, SD `0.141266`, signs `-/1 +/5`.
- `score_1`: mean `0.0051395123`, SD `0.01325249`, signs `-/1 +/5`.
- `score_2`: mean `0.00058136011`, SD `0.0030588283`, signs `-/3 +/3`.
- `score_3`: mean `-0.11357738`, SD `0.11330994`, signs `-/5 +/1`.
- `score_4`: mean `-0.08254083`, SD `0.76964951`, signs `-/3 +/3`.
- `score_5`: mean `0.10274887`, SD `0.94632868`, signs `-/3 +/3`.
- `value`: mean `0.018383026`, SD `0.035335626`, signs `-/3 +/3`.

## Attribution Diagnostic

The FD-step ladder in `attempt04_fd_attribution/result.json` evaluates baseline and selected arms at `h` in `{3e-4,1e-3,3e-3,1e-2}`. LGSSM and predator-prey baseline and selected arms both improve substantially at larger steps, while neither meets the frozen `h=1e-3` gate. This identifies a precision/finite-difference sensitivity of the inherited route; it does not establish exact score accuracy.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain cap implementation | Primitive disabled-parity, affine-restoration, and JVP tests pass | none | generic cap changes moments | Keep opt-in only | no model-wide success |
| Admit KSC capped arm | finite rows and FD gate pass | reference comparison is descriptive regression | dense reference is not runtime target | investigate cap removal/less-active chart before promotion | no score accuracy/posterior claim |
| Admit LGSSM capped arm | finite rows pass, FD fails; baseline FD also fails | hard derivative veto | FP32/TF32 FD cancellation and cap activity | repair baseline derivative authority or use higher-precision reference lane | no cap score claim |
| Admit predator-prey capped arm | finite rows pass, FD fails; no exact authority | hard derivative veto | no nonlinear score oracle and cap changes map | keep diagnostic-only; do not promote | no accuracy/superiority claim |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | KSC passes; LGSSM and predator-prey fail the declared FD screen; all finite/program/transport gates pass |
| Statistically supported ranking | None; six seeds and descriptive shifts do not support ranking |
| Descriptive differences | KSC cap changes value/score slightly and is descriptively worse against dense reference; other shifts are descriptive only |
| Default readiness | No; cap remains opt-in extension/invention outside Austria |
| Next evidence needed | A scope-specific higher-precision score authority and a less-active chart/cap calibration; no changed defaults under this artifact |

## Run Manifest

- Plan: `docs/plans/bayesfilter-zhao-cui-genut-dual-cap-cross-model-plan-2026-08-07.md`
- Git commit: `6a11b689295bfb0e58de6e6d2f84918671b5a685`
- Environment: `/home/chakwong/anaconda3/envs/tf-gpu`
- TensorFlow: `2.19.1`
- GPU: `['/device:GPU:0', '/device:GPU:1']`
- Dtype/TF32/XLA: `float32` / `True` / `True`
- Wall time: `210.028s`

## Nonclaims

- coordinatewise standardized cap is extension_or_invention outside Austria bounded-teacher chart
- no exact nonlinear score claim for predator-prey
- KSC dense reference is diagnostic only
- six seeds give descriptive uncertainty, not statistically supported ranking
- no default, HMC, NeuTra, posterior, or broad superiority claim
