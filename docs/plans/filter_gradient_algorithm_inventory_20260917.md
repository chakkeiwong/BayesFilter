# Filtering and gradient algorithm inventory — 2026-09-17

Baseline: `3582b4ac`. This inventory distinguishes mathematical algorithms,
implementation variants, score definitions, and reference code. A name such as
`canonical`, `analytical`, `batched`, or `xla` does not establish policy compliance
or scientific admission. Findings and measured memory are in the
[audit result](filter_gradient_policy_memory_result_20260917.md).

This is the frozen pre-campaign inventory. Its execution dispositions describe
the baseline, not current compliance. Follow the
[active repair master](filter_gradient_repair_master_20260917.md) and
[completion ledger](filter_gradient_repair_ledger_20260917.json) for subsequent
repairs, qualification and still-open consumer coverage.

The discovery pass covers all **2,695 tracked Python files**, including package
code, executable benchmark scripts under `docs/benchmarks`, experiments, tests,
archived harnesses and vendored sources. The compressed
[function-level inventory](artifacts/filter-gradient-policy-memory-20260917/static-complete.json.gz)
records every parsed function, defaults, decorators, Python loop site, NumPy
import/call, TensorFlow boundary, source hash and resolvable local call edge.
The [module index](artifacts/filter-gradient-policy-memory-20260917/static-complete.md)
lists all 1,448 owned source/harness modules. It is broader than the algorithms
below because it also includes orchestration, validation and reporting.

## Filtering families and implementation variants

Paths below are relative to the repository root. Closely related scalar,
masked, batch and history-output entry points are grouped rather than counted
as different mathematical algorithms.

| Family | Implementations and variants | Gradient calculation / execution disposition |
|---|---|---|
| Covariance Kalman | `linear/kalman_tf.py`, `experimental_batched_kalman_tf.py`, `kalman_covariance_derivatives_tf.py`: dense, missing-observation/masked, checked, batch and history variants | Explicit first-order mean/covariance sensitivity recursion; shared second-order recursion where exposed. Repaired TensorFlow recurrences; XLA defaults. |
| Square-root QR Kalman | `linear/kalman_qr_tf.py`, `kalman_qr_derivatives_tf.py`: compact, factorized, masked, static batch and history variants | Analytical score and Hessian through factor/solve recursions. Repaired numerical loops. |
| SVD Kalman | `linear/kalman_svd_tf.py`, `kalman_svd_derivatives_tf.py`, `batched_kalman_svd_derivatives_tf.py` | First-order score and second-order/Hessian interfaces with spectral validity/status checks. Repaired recurrences; distinguish SVD-valued and nonspectral derivative constructions. |
| Correlated-noise linear Gaussian | `linear/correlated_kalman_tf.py` | Conditional-Gaussian Kalman handling of cross-covariance; not a separate generic EKF. |
| Stationary linear-Gaussian initialization | `linear/stationary_lgssm_derivatives_tf.py` | Stationary mean/covariance derivative construction feeding Kalman targets. This is initialization, not an additional filter. |
| Direct factor SRUKF | `nonlinear/factor_srukf_tf.py`, `srukf_factor_tf.py`, `factor_srukf_compat.py` | Augmented-noise sigma points, QR prediction, block conditional QR, analytical first derivatives. `tf_default_srukf_value_and_score` selects this family. XLA defaults; fresh GPU memory/parity check completed. |
| Rectangular SRUKF | `nonlinear/rectangular_srukf_tf.py`, `linear/rectangular_factor_tf.py` | Singular-support value evaluation, direct SVD support discovery, frozen-rank/chart QR analytical score. Fresh memory fixture tests the full-rank fixed-chart branch only. |
| Spectral/principal-square-root UKF and cubature | `nonlinear/sigma_points_tf.py`, `svd_sigma_point_derivatives_tf.py`, `experimental_batched_svd_sigma_point_tf.py`, `batched_svd_sigma_point_tf.py` | UKF and spherical-radial cubature, scalar and batch, explicit sensitivity and output-cotangent/custom-gradient interfaces. Historical factor choices remain distinct from the direct SRUKF default. |
| Conjugate unscented transform CUT4 | `nonlinear/cut_tf.py`, `svd_cut_tf.py`, `svd_cut_derivatives_tf.py`; experimental `tf_tfp/cubature/cut4_tf.py` | CUT4 Gaussian integration and analytical sigma-point score; experimental one-dimensional CUT4 reference filters also exist. |
| Fixed sparse-grid quadrature filter (SGQF) | `nonlinear/fixed_sgqf_tf.py`, `fixed_sgqf_compiled_tf.py`, `fixed_sgqf_derivatives_tf.py` | Fixed sparse Gauss-Hermite/Smolyak branches, signed weights, explicit recursive score. Numerical recurrence is compiled by default; Python quadrature construction and an autodiff structural adapter remain separate findings. |
| SSL-LSTM nonlinear targets | `nonlinear/ssl_lstm_posterior_tf.py`, `ssl_lstm_complexity_target_tf.py`, `ssl_lstm_complexity_batched_target_tf.py`, `ssl_lstm_sgqf_ukf_adapters.py`, `ssl_lstm_zhaocui_fixed_adapter.py` | Model adapters over UKF, SGQF and fixed TT; parameter-mask, prior and coordinate-chain scores. They are consumers, not new filtering algorithms. Debug artifact builders have explicit non-JIT defaults. |
| Stochastic-volatility filter specializations | `highdim/actual_sv_srukf_tf.py`, `generalized_sv_sgqf_tf.py`, `sv_mixture_cut4.py` | Actual-SV augmented-noise SRUKF; raw-y generalized-SV SGQF manual score; transformed-SV independent-panel Gaussian-mixture Kalman/UKF/CUT4/SGQF/TT comparators, including exact-transformed-observation variants. NumPy reference/preparation boundaries need isolation. |
| Bootstrap particle filter | `experiments/dpf_implementation/tf_tfp/filters/bootstrap_pf_tf.py`; NumPy `filters/particles.py` | Importance-weight recursion with multinomial/systematic resampling. TensorFlow experimental runner has Python date iteration and host ESS branching; NumPy implementation is explicitly a reference. |
| Relaxed OT differentiable particle filter | `experiments/dpf_implementation/tf_tfp/filters/dpf_ot_tf.py` | Bootstrap propagation with finite Sinkhorn/other relaxed resampling. Host filtering loop; differentiating a relaxed finite program is distinct from a statistical observed-data score. |
| Single-cloud LEDH/PF-PF | `tf_tfp/filters/ledh_pfpf_ot_tf.py`, `tf_tfp/flows/ledh_tf.py`, `tf_tfp/structural/structural_filter_tf.py` under `experiments/dpf_implementation` | Local affine flow, proposal-density correction, resampling; structural stochastic/completion-state variant. Python date/particle/pseudo-time loops and host diagnostics prevent a fully compiled enclosing filter. Historical implementations do not obtain current canonical status from their names. |
| Li-Coates Algorithm 1 with per-particle UKF covariance | `experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_alg1_ukf_tf.py` | Particle UKF predict/update, flow and covariance lifecycle; eager general route, row-mapped variant, specialized scalar-SV graph/value kernels, classical and OT resampling state. Graph scalar kernels default to XLA; general loops remain. |
| Dense batched LEDH-PFPF-OT | `experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py` | Batch-native flow and transport; Python time unrolling; score wrapper uses `GradientTape`. The file's default-production prose is not current analytical/canonical admission evidence. |
| Streaming batched LEDH-PFPF-OT | `experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_streaming_tf.py` | TensorFlow particle/time loops with optional histories and streaming transport; enclosing score wrapper still uses `GradientTape`. |
| Contract E–Cholesky candidates | `highdim/ledh_contract_e_canonical_lgssm_tf.py`, `ledh_contract_e_latent_sir_tf.py`, `ledh_contract_e_reset_tf.py`, `ledh_contract_e_streaming_tf.py` | Total moment/weight/transport derivatives and reset primitives; compiled factories, LGSSM and latent-SIR variants, including two-node variant. Parameter loops remain in LGSSM derivative helpers. Canonical reset eligibility does not establish the full post-reset LEDH algorithm's admission. |
| Cubature/GenUT and higher-moment particle candidates | `highdim/cubature_genut_filter.py`, `cubature_genut_batch_tf.py`, `higher_moment_contract_e.py`, `genut_shape_lm_tf.py` | Scalar and batch finite-program value/JVP variants, moment restoration, marginal/pairwise/projected corrections and LM/trust controls. Main batch score currently runs ForwardAccumulator per parameter; manual diagnostic alternatives must not silently become claim-bearing replacements. |
| Guided and initialization-design particle variants | `highdim/genut_guided_proposal_tf.py`, `ledh_pfpf_genut_initialization_tf.py`, `ledh_pfpf_genut_initial_rqmc_tf.py` | Defensive/guided proposals; IID or randomized-QMC initialization; one-to-one, Hilbert-ordered and inverse-CDF ancestry. Standard backward-filtering score is explicitly not the derivative of the finite likelihood. Some date loops remain. |
| SQMC design and ancestry primitives | `highdim/sqmc_tf.py` | Randomized Halton uniforms/Gaussians, calibrated logistic map, Hilbert ordering and inverse-CDF ancestors. Used by particle variants; Hilbert bit/axis loops are Python. This is not a standalone newly admitted canonical LEDH filter. |
| TT-assisted Contract E moment-teacher filters | `highdim/zhao_cui_moment_teacher_lgssm_tf.py`, `zhao_cui_moment_teacher_nonlinear_tf.py`, `zhao_cui_moment_teacher_xla.py`, `zhao_cui_moment_teacher.py`, `zhao_cui_moment_teacher_als.py` | Independent squared-TT moments and fixed-ALS value/JVP replay; LGSSM, predator-prey, Austria-SIR and actual-SV particle variants. Packed compiled primitives coexist with Python setup, parameter and historical reference loops. |
| Experimental Contract E–TP | `highdim/ledh_contract_e_tp_tf.py`, `ledh_contract_e_tp_lgssm_tf.py`, `ledh_contract_e_tp_scalar_sv_tf.py`, `ledh_contract_e_tp_structural_tf.py` | Dense/streaming teacher constraints, diagonal/dense KKT, square/overcomplete controls; value, manual JVP/VJP and autodiff variants. Recursive unrolled and TensorFlow-loop implementations coexist. Not a substitute for the Cholesky reset policy. |
| Gaussian/dense/retained-grid filtering | `highdim/filtering.py` | Gaussian retained filter, scalar dense quadrature filter, scalar/multistate TT retained-grid value/score paths. Generic multistate retained-grid Zhao-Cui routes are policy-restricted to historical/diagnostic use. |
| Fixed adjacent-state squared TT | `highdim/zhao_cui_fixed_adjacent_tt_tf.py`, `zhao_cui_actual_sv_batched_tt_tf.py` | Scalar adjacent-state fixed fitting and batch-native actual-SV finite likelihood; analytical directional replay and separate reverse-autodiff score. Python time, parameter and ALS-sweep loops remain. |
| Generic squared-TT filters | `highdim/squared_tt_engine_v0_tf.py`, `squared_tt_engine_adapted_tf.py`, `squared_tt_engine_gaussian_tf.py` and corresponding `*_xla_tf.py` files | Branch-axis retention, adapted/truncation-corrected coordinates and Gaussian/Hermite reference. The XLA variants compile steps but retain host date loops and unrolled ALS fitting. |
| Squared-TT manual adjoint filter | `highdim/squared_tt_adjoint_engine_tf.py`, `squared_tt_adjoint_tf.py` | Forward checkpoints, reverse date/ALS sweeps, QR/Cholesky/design/normalizer/marginal pullbacks. Analytical adjoint, but Python numerical loops and host materialization remain. |
| Frozen-TT proposal APF and fixed-variant predator-prey evaluator | `highdim/zhao_cui_frozen_proposal_apf_tf.py`, `zhao_cui_predator_prey_fixed_variant_tf.py` | Frozen SIRT samples/proposals, importance/auxiliary corrections and explicit score marks. Both provide XLA-default factories; APF still unrolls time, predator-prey uses a TensorFlow time loop. Preparation is a separate numerical boundary. No source-faithfulness claim is made here. |
| Austria-SIR Lane-B TT variants | `highdim/zhao_cui_austria_sir_packed_xla_tf.py`, `zhao_cui_austria_sir_lane_b_tf.py`, `zhao_cui_austria_sir_lane_b_t1_score_tf.py`, `zhao_cui_austria_sir_lane_b_t2_score_tf.py`, `zhao_cui_austria_sir_*training_jvp_tf.py` | T1/T2 learned parameter-density/marginal TT, packed replay, origin/Fisher/manual carried scores and training-program tangents. These are specialized learned/fixed-program score estimators, not generic exact analytical filtering scores. |
| Source-route sequential TT/SIRT operations | `highdim/source_route.py`, `tt.py`, `fitting.py`, `transport.py`, `retained_quadratic_form_tf.py`, `retained_moments_tf.py` | Frozen fitting, TT contraction/marginalization, coordinate frames/recentering and sequential reapproximation; direct analytical LS/TT derivatives in `derivatives.py`. Many Python axis/sweep/preparation loops; no blanket compiled-default certificate. |
| Hard-bound models and reference filtering | `hardbound/censored_scalar_tf.py`, `truncated_gaussian_tf.py`, `dns_curve_tf.py`, `joint_target_tf.py`; `gate_grid_reference.py` | Closed-form censoring probabilities/densities and truncated-Gaussian moments; DNS observation map; latent joint targets (not marginal filters); independent gate grid/Kalman reference. Joint target state recurrences use Python; DNS preparation uses NumPy. |
| Independent simulation-score authorities | `highdim/simulation_score_tf.py`, `sir_online_score_teacher_tf.py` | Self-normalized Fisher identity over simulated paths and an online O(N²) SIR teacher with analytical local-density scores. The teacher has an XLA-default factory; Fisher reduction is a plain TensorFlow helper. |
| Learned observation-only score estimators | `independent_score/classifier_ratio_score_tf.py`, `joint_k_classifier_ratio_score_tf.py`, `anchored_orthogonal_ratio_score_tf.py` | Calibrated classifier log ratios / central score, joint-K and anchored orthogonal variants, epsilon-squared extrapolation. Batched optimizer steps default to XLA; Python epoch orchestration is distinct from filter recurrence. These are learned estimators, not exact recursive analytical scores. |
| Independent/reference/quarantined filters | `filters/{kalman,sigma_points,particles}.py`, `linear/kalman_derivatives_numpy.py`, `testing/`, `experiments/dpf_implementation/{filters,references,resampling}`, `experiments/controlled_dpf_baseline`, `experiments/student_dpf_baselines` | NumPy covariance/solve/SVD Kalman and score/Hessian, cubature/UKF, bootstrap PF, finite Sinkhorn and flow-assisted reference baselines. Vendored FilterFlow and student EDH/PFPF/kernel-PFF/UKF/Kalman code is separately indexed. These are not eligible substitutes for owned TF/XLA runtime. |

The `linear/`, `nonlinear/`, `highdim/`, `filters/`, `hardbound/`, and
`independent_score/` paths in the table are under `bayesfilter/`. No separate
owned general extended-Kalman or ensemble-Kalman implementation was identified
in the package/TF-DPF discovery. This observation does not classify every
external vendor's API as a BayesFilter algorithm.

## Gradient mechanisms that must not be conflated

| Mechanism | Where it occurs | Meaning |
|---|---|---|
| Analytical forward filtering sensitivities | Kalman covariance/QR/SVD; SRUKF; SGQF | Derivatives propagated with mean, covariance/factor and likelihood recurrences. |
| Analytical Hessian/second-order recursion | `linear/kalman_second_order_tf.py`, QR/SVD Hessian front ends | Second derivatives of the selected fixed numerical branch, with supplied structural derivatives. |
| Matrix differential primitives | `linear/qr_factor_tf.py`, `stack_qr_tf.py`, `block_qr_conditional_tf.py`, `lower_rank_downdate_tf.py`, `rectangular_factor_tf.py`, `svd_factor_tf.py`, `ops/` | QR/Cholesky first/second derivatives, rank-update and conditional-factor derivatives, fixed-support solves, principal-square-root Fréchet/Sylvester calculations. |
| Analytical finite transport JVP/VJP | `ledh_contract_e_reset_tf.py`, `ledh_contract_e_streaming_tf.py`, experimental `resampling/annealed_transport_tf.py` | Derivatives of finite Sinkhorn iterations, row normalization, transported cloud and direct source moments/weights. Total and stopped-scale/keys variants are different derivatives. |
| Transport approximations | Experimental `resampling/{sinkhorn,nystrom_transport,positive_feature_transport,low_rank_coupling_transport,low_rank_coupling_solver}_tf.py`, warm-start student helpers | Dense, annealed, streaming, Nyström, positive-feature and low-rank transport calculations. A custom gradient may still call autodiff internally. Tensor-returning primitives and host-reporting wrappers must be audited separately. |
| Analytical TT/least-squares JVP and adjoint | `highdim/derivatives.py`, TT teacher and adjoint modules | Frozen-branch fitting, basis/core contraction, squared-density normalization and retained-marginal derivatives. |
| Analytical statistical score recursion | Online SIR teacher, `ledh_pfpf_genut_initialization_tf.py`, guided proposals and fixed-variant importance evaluators | Expected complete-data score / backward score marks. It need not equal a derivative of a fixed Monte Carlo likelihood program. |
| TensorFlow reverse autodiff | `GradientTape`, `score_api.py`, actual-SV batch TT AD endpoint, dense/streaming LEDH wrappers, model/transport/training helpers | AD-generated derivative. Not an analytical LEDH recursive score; model-local AD also needs disclosure. |
| TensorFlow forward autodiff | `ForwardAccumulator` in GenUT and diagnostic Contract E–TP variants | AD/JVP replay. Per-parameter Python replay remains a policy violation for admitted numerical runtime. |
| Custom-gradient bridges | `nonlinear/batched_svd_sigma_point_tf.py`, `inference/batched_value_score.py`, NeuTra target bridges | Supply a precomputed explicit score to consumers; decorator alone does not establish whether the underlying score is analytical, batched or XLA. |
| Learned/neural scores and forces | `independent_score/`, inference `neural_force_*`, learned NeuTra transport | Optimization gradients, ratio-score estimators and learned proposal forces. Not interchangeable with exact target score. |
| Prior, Jacobian and coordinate chain rules | `adapters/bgs.py`, stationary/model adapters, posterior targets, `inference/batched_value_score.py` | Analytical prior derivatives and affine/nonlinear coordinate pullbacks; must be included in an enclosing target audit. |
| Diagnostic derivative authorities | Testing autodiff oracles, finite-difference/complex-step checks where present, `score_diagnostics_tf.py`, benchmark comparison code | Parity checks only; cannot silently replace an analytical/default computation. |

HMC, NeuTra, MAP/curvature estimation, whitening and CPU value/score pools are
gradient **consumers**. Their execution/admission boundaries are included in
the static inventory and relevant findings, but this audit does not claim to
requalify every sampler or optimizer. HMC public authority remains governed by
`docs/reference/hmc-tuning-interface.md` and the capability registry.
