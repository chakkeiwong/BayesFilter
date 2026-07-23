# Higher-Moment Contract E Candidate Result

Date: 2026-07-23

## Outcome

The finite higher-moment Contract E candidate is engineering-valid and has no statistically supported regression against the prior same-target finite route on the oracle-backed LGSSM and genuine transformed-SV scopes. It does not solve the score-bias problem and is not promoted to canonical, default, HMC, or leaderboard status.

## Algorithm Audited

The documented and implemented route is:

`particle transition -> likelihood/value increment -> normalized weights -> finite entropic OT/Sinkhorn -> row-quotient barycenter -> Contract E residual injection -> realized-covariance Cholesky restoration -> finite projected third/fourth-moment correction -> exact mean/covariance rewhitening -> next transition`.

The candidate is not a classical unscented transform because it does not evaluate the nonlinear map at sigma points. GenUT motivates selected diagonal skewness/kurtosis targets; the actual candidate keeps all particle model evaluations and applies a finite post-reset shape map. The correction is a damped local Gauss-Newton step with a coupled 2x2 moment Jacobian per coordinate. It is not an exact nonconvex moment projection.

## Mathematical Audit

- LaTeX chapter: `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`.
- MathDevMCP artifact: `docs/plans/bayesfilter-higher-moment-contract-e-genut-mathdevmcp-audit-2026-07-23.md`.
- Propositions cover the finite OT row quotient, Contract E ridged covariance identity, current-increment value boundary, projected moment Jacobian, exact rewhitening mean/covariance, and total derivative of the executed finite program.
- MathDevMCP parsed the chapter successfully. Its proposition-label audit selected zero display-equation targets; focused derivation-label calls were `inconclusive` with no semantic counterexample because the scalar backend cannot encode the matrix/finite-program obligations. This is diagnostic audit evidence, not a formal proof certificate.
- Independent primitive/JVP tests passed, including a TensorFlow forward-mode comparison.

## Verification

`CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_higher_moment_contract_e.py tests/highdim/test_cubature_genut_candidate.py tests/highdim/test_cubature_genut_filter.py tests/highdim/test_cubature_genut_adapters.py`

Result: `26 passed`.

The full monograph build reached and compiled the new chapter, then stopped on a pre-existing missing SSL-LSTM figure later in the repository:

`plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation/ssl-lstm-launch-traces-z.pdf`.

## Campaign

Artifact: `docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/attempt02/result.json`.

Comparison: `docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/comparison_attempt01/comparison.md`.

Execution used FP32, TF32, XLA, TensorFlow memory growth, RTX 4080 SUPER, and `N>1000`. Scope-specific tuning used separate calibration and selection/validation observation partitions, then 16 untouched particle seeds per oracle-backed claim scope. Runtime score was recursive forward sensitivity; no runtime autodiff or finite differences were used.

### Tuning Audit

The controls were tuned correctly for the declared engineering objective, but
not for oracle accuracy. For each model/horizon scope, the runner evaluated a
16-point grid using two calibration observations and two selection/validation
observations, with two particle seeds per observation. It selected the
minimum validation variance proxy, breaking ties with calibration variance and
then correction size. The proxy was the maximum, over value and score
coordinates, of the sample variance divided by fixed coordinate scales.

This establishes:

- scope-specific controls;
- no claim observations or claim particle seeds read during selection;
- frozen controls in the repository-issued route identity; and
- an untouched 16-seed claim run for the oracle-backed scopes.

It does **not** establish optimal value/score accuracy. The tuning objective
contains no exact-oracle error and no direct moment-residual term. It can select
a low-variance but biased estimator. The validation partitions are selection
data, not an untouched final validation set, despite the historical label
`validation`; the untouched evidence is the claim seed set (and, for the
oracle-backed rows, the fixed claim observation). The grid was also small and
the selection used only two particle seeds per partition, so the objective is
noisy. For example, selected controls were on different grid boundaries across
the LGSSM horizons, and the selected LGSSM `T=2` controls were not the
calibration-variance minimizer. These are acceptable for feasibility tuning but
not evidence of global or statistically optimal tuning.

Accordingly, the correct conclusion is: **the candidate was tuned according to
its predeclared variance-stability proxy, but it was not tuned in a way that
could guarantee or demonstrate oracle-score improvement.** A stronger tuning
study would require more independent calibration/selection replicates, a
predeclared distributional/moment objective, and an untouched validation
observation set; oracle error may be used only for diagnostic evaluation on
oracle-bearing models, not as a general runtime tuning requirement.

The first launch attempt was a harness initialization-order failure and did not start a scientific run. It is preserved at `docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/attempt01/failure.json`. The repaired second attempt completed with `hard_valid=true` in 627.25 seconds. TensorFlow allocator peak in the closed process was 134,416,128 bytes; this is allocator telemetry, not a hard memory cap.

## Results

The table below reports candidate mean and Student-t 95% CI over 16 seeds, dense-oracle error mean, and the paired 95% CI for candidate-minus-prior absolute oracle error. An interval crossing zero supports neither improvement nor regression.

| Scope | Quantity | Candidate mean [95% CI] | Prior mean [95% CI] | Candidate oracle error mean | Paired abs-error delta 95% CI |
|---|---|---:|---:|---:|---:|
| LGSSM T=2 | value | -8.82024 [-8.91467, -8.72580] | -8.82053 [-8.91498, -8.72609] | 0.04191 | [-0.000431, 0.000421] |
| LGSSM T=2 | phi1 | 3.95554 [3.67523, 4.23584] | 3.95294 [3.67170, 4.23418] | 0.12752 | [-0.00421, 0.00298] |
| LGSSM T=2 | phi2 | -0.37402 [-0.43169, -0.31635] | -0.37491 [-0.43356, -0.31626] | 0.01016 | [-0.00282, 0.00118] |
| LGSSM T=2 | phi3 | -0.04905 [-0.09802, -0.00009] | -0.04935 [-0.09821, -0.00049] | 0.03404 | [-0.00038, 0.00161] |
| LGSSM T=2 | q_scale | 4.65774 [4.27721, 5.03827] | 4.65674 [4.27552, 5.03796] | 0.24053 | [-0.00560, 0.00260] |
| LGSSM T=2 | r_scale | 11.06065 [10.43459, 11.68670] | 11.06234 [10.43636, 11.68811] | -0.07749 | [-0.00364, 0.00508] |
| LGSSM T=10 | value | -32.14010 [-32.24105, -32.03914] | -32.13954 [-32.24070, -32.03807] | -0.08748 | [-0.00166, 0.00055] |
| LGSSM T=10 | phi1 | 11.15960 [10.76518, 11.55403] | 11.15960 [10.76460, 11.55460] | -0.12039 | [-0.00943, 0.00413] |
| LGSSM T=10 | phi2 | -0.28891 [-0.47669, -0.10113] | -0.28757 [-0.47602, -0.09912] | 0.01513 | [-0.00333, 0.00030] |
| LGSSM T=10 | phi3 | -1.33979 [-1.45392, -1.22567] | -1.34084 [-1.45489, -1.22678] | -0.03476 | [-0.00123, 0.00181] |
| LGSSM T=10 | q_scale | 9.68801 [9.04147, 10.33454] | 9.67916 [9.03397, 10.32442] | 0.19935 | [-0.00699, 0.01408] |
| LGSSM T=10 | r_scale | 14.60188 [13.87969, 15.32407] | 14.60026 [13.87582, 15.32470] | 0.53354 | [-0.00815, 0.00307] |
| LGSSM T=50 | value | -136.06404 [-136.34559, -135.78250] | -136.06503 [-136.34604, -135.78402] | 0.01193 | [-0.000014, 0.00198] |
| LGSSM T=50 | phi1 | 5.71923 [5.33231, 6.10614] | 5.70907 [5.32096, 6.09718] | 0.06378 | [-0.00857, 0.00398] |
| LGSSM T=50 | phi2 | -4.02410 [-4.31015, -3.73805] | -4.02513 [-4.31005, -3.74022] | -0.18904 | [-0.00026, 0.00317] |
| LGSSM T=50 | phi3 | 0.22075 [-0.01368, 0.45518] | 0.22049 [-0.01376, 0.45473] | -0.08161 | [-0.00089, 0.00229] |
| LGSSM T=50 | q_scale | -2.21452 [-3.27136, -1.15769] | -2.23526 [-3.29294, -1.17758] | -0.29735 | [-0.01397, 0.00761] |
| LGSSM T=50 | r_scale | 4.40299 [2.70200, 6.10398] | 4.38064 [2.68266, 6.07862] | 0.04871 | [-0.00892, 0.02051] |
| Fresh SV T=50 | value | -116.80421 [-116.85621, -116.75221] | -116.80256 [-116.85464, -116.75047] | -0.00492 | [-0.000864, 0.00107] |
| Fresh SV T=50 | theta_gamma | -0.81244 [-0.87238, -0.75250] | -0.80873 [-0.86869, -0.74877] | 0.03959 | [-0.00230, 0.00206] |
| Fresh SV T=50 | theta_log_beta | -2.28351 [-2.39521, -2.17181] | -2.28550 [-2.39735, -2.17365] | -0.05194 | [-0.00195, 0.00112] |

The dense fresh-SV reference is value `-116.799295`, score `(-0.852029, -2.231567)`. The candidate value is close; score differences remain descriptive and not statistically supported as an improvement.

Predator-prey T=20 has no exact score oracle. Candidate means are value `-103.16426`, `(r,K,a,s,u,v)=(-22.11031, 1.19975, -0.001464, -3.16670, -0.64445, 0.15989)`. These are descriptive comparisons only. Austria SIR canonical fixed source-order SGQF value is `-691.369206826` for both prior and candidate, difference `0`; score is not applicable because there is no free parameter.

Maximum candidate moment residuals (skewness, kurtosis) were: LGSSM T=2 `(0.8186, 1.0613)`, T=10 `(0.6107, 0.9003)`, T=50 `(0.8550, 2.3369)`, fresh SV `(0.5819, 0.8160)`, predator-prey `(0.4262, 0.7934)`. These reject any claim of exact third/fourth-moment matching.

## Decision

| Decision | Status | Reason |
|---|---|---|
| Engineering validity | `PASS` | finite GPU/XLA rows, route identity, covariance/score-sum checks |
| No-regression screen | `PASS` | no paired oracle-error CI entirely above zero |
| Candidate improvement | `UNSUPPORTED` | no paired oracle-error CI entirely below zero |
| Score-bias resolution | `FAIL` | LGSSM T=50 score remains descriptively biased; residuals are large |
| Canonical/default/leaderboard promotion | `NOT READY` | exactness and broad nonlinear evidence remain absent |

The result is evidence against promoting this particular finite correction as the solution, not evidence against the broader research direction of improving the carried distribution. The next discriminating work should target a correction with a stronger distributional contract and an independently audited score/value comparison.
