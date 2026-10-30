# Covariance-guided proposal: implementation and results, 8 October 2026

The master and shared TensorFlow kernels implement the documented finite algorithm for four additive-Gaussian-transition adapters. The implementation checks pass, but this candidate is not ready to replace the existing filter: FP32/TF32 failed a SIR covariance invariant, a global-heavy SIR proposal collapsed, and an equal-mixture SIR likelihood has extreme local sensitivity. The fitted mixture avoids those particular failures in the two held-out seeds; this is not a statistically supported ranking or a proof that SIR is repaired.

Code: `docs/benchmarks/run_ledh_covariance_proposal.py`; plan: `docs/plans/ledh-covariance-proposal-implementation-20261008.md`. Full per-run values, all scores and reference MCSEs: [numerical tables](../plans/artifacts/ledh-covariance-implementation-20261008-01/numerical-tables.md). All structured records and failed attempts remain in that directory.

## What changed and what was checked

- One shared candidate implements conditional-Q local affine flows, a UKF global map, categorical mixture sampling, all-ancestor densities, the finite Sinkhorn/Cholesky reset and analytical total derivatives. Invalid recursion stops and records its completed steps; partial values are not full-horizon likelihoods.
- Independent-pilot beta fitting uses a defensive simplex, backtracking projected gradients, a convex gap certificate and separate calibration/validation/final data. Only beta was fitted: floor 0.1, Euler16, Sinkhorn40 and epsilon1 remain hypotheses. The artifact is a diagnostic candidate scope, not canonical admission.
- Optional correction includes all marginal third/fourth moments and pairwise powers (2,1), (1,2), (3,1), (1,3), (2,2), four fixed cosine-pair skew directions, a smooth Cayley map and analytical optimizer derivatives. Both standardized displacement caps are 0.25. No general moment-attainability claim is made.
- KSC gained a dtype argument while retaining its prior FP64 default; its exact mixture observation law remains in the weights. No existing filtering default was replaced.
- 41 CPU-hidden tests passed, including every parameter in all four models with and without correction, independent mixture-density evaluation, convex derivatives, reset invariants, caps, Cayley solve residuals/mixed derivatives, singular rejection and executable shared-call-chain checks. Existing marginal-weight, KSC and LGSSM regressions passed.
- The 34-child GPU/XLA matrix completed. Its SIR global-heavy arm returned invalid; the other matrix arms completed. Seven focused follow-ups and two finer finite-difference runs followed four initial attempts: 48 model/calibration attempts in total, including a final identical-input check of added observability and fail-closed guards. All used verified memory growth. These are bounded diagnostics, not a full tuning campaign.

## Held-out T=50 values and every score coordinate

N=128, FP64 GPU/XLA, final data seed 26100830, candidate seeds 261008100/101. Calibration/validation used 26100820/21. Parameter vectors are saved beside every result. PP order is (r, carrying_capacity, half_saturation, s, u, v); SIR order is (log_kappa_scale, log_nu_scale, log_observation_noise_scale). KSC uses its transformed persistence and log-scale coordinates; LGSSM uses the five frozen_3d adapter coordinates. Do not compare coordinates across models.

| Model / estimator | Beta (transition, global, local) | Log likelihood | Score vector |
|---|---|---:|---|
| lgssm, seed 261008100 | [0.100000, 0.900000, 0.000000] | -235.488023 | [-21.097526, -4.226782, -5.026579, 3.047367, 13.271472] |
| lgssm, seed 261008101 | [0.100000, 0.900000, 0.000000] | -233.523209 | [-11.380205, -6.900280, -2.337906, 0.794430, 16.130195] |
| lgssm, Kalman exact | — | -234.064961 | [-14.318744, -6.232475, -3.523503, 3.580154, 15.048428] |
| ksc, seed 261008100 | [0.100000, 0.900000, 0.000000] | -122.720613 | [-1.673710, 0.732997] |
| ksc, seed 261008101 | [0.100000, 0.900000, 0.000000] | -122.231158 | [-2.129109, 0.453984] |
| ksc, independently refined KSC grid | — | -122.397506 | [-1.623850, 0.677745] |
| predator_prey, seed 261008100 | [0.100000, 0.900000, 0.000000] | -241.611502 | [-5.008690, 3.233081, 0.021280, -4.018390, -3.360355, 4.258475] |
| predator_prey, seed 261008101 | [0.100000, 0.900000, 0.000000] | -241.233471 | [-4.808240, 3.199668, 0.021830, -5.300951, -2.760591, 3.519942] |
| predator_prey, bootstrap Fisher diagnostic, not exact oracle | — | -241.336261 | [-1.122989, 3.360679, 0.029525, -5.355769, -4.718079, 5.914300] |
| predator_prey, reference MCSE | — | 0.051457 | [1.049110, 0.072710, 0.000791, 0.354580, 0.218772, 0.283764] |
| sir_d18, seed 261008100 | [0.100000, 0.000000, 0.900000] | -1679.557415 | [7.650856, 135.306409, -11.799591] |
| sir_d18, seed 261008101 | [0.100000, 0.000000, 0.900000] | -1678.657591 | [-123.129080, 170.696043, -20.842317] |
| sir_d18, bootstrap Fisher diagnostic, not exact oracle | — | -1677.300976 | [-53.035709, 120.162474, -19.999914] |
| sir_d18, reference MCSE | — | 0.360010 | [72.891029, 4.997353, 1.774160] |

The bootstrap references use N=8192 and four independent replications. Their MCSEs do not measure finite-particle bias. SIR retains as few as 22 initial ancestors; its first-score MCSE is 72.89, so that coordinate is a weak reference. No eligible Zhao–Cui run on these identical fresh data was available; historical numbers were not substituted.

## Simple competitors and horizon checks

Each T50 arm below uses the identical held-out dataset, N and seeds. Global/local arms retain 0.1 transition mass. These actual values are descriptive; the two replications do not support a ranking. The transition-only proposal remains an essential comparator. A large equal-mixture SIR score error relative to both it and the bootstrap reference is a promotion veto.

| Model / arm | Seed | Log likelihood | Score vector | Status |
|---|---:|---:|---|---|
| lgssm/equal | 261008100 | -235.886672 | [-16.758486, -4.703665, -4.988218, 6.703190, 16.089190] | complete |
| lgssm/equal | 261008101 | -233.273091 | [-14.575247, -4.826123, -1.864329, -0.498829, 15.638959] | complete |
| lgssm/identity | 261008100 | -235.804910 | [-21.885464, -3.488341, -1.139142, -0.957670, 18.458597] | complete |
| lgssm/identity | 261008101 | -232.623980 | [-14.540005, -2.256590, -2.895466, 4.927614, 16.016959] | complete |
| lgssm/global | 261008100 | -235.488023 | [-21.097526, -4.226782, -5.026579, 3.047367, 13.271472] | complete |
| lgssm/global | 261008101 | -233.523209 | [-11.380205, -6.900280, -2.337906, 0.794430, 16.130195] | complete |
| lgssm/local | 261008100 | -235.167544 | [-19.370242, -3.975224, -5.015662, 3.404240, 13.576393] | complete |
| lgssm/local | 261008101 | -233.755850 | [-12.818196, -6.396233, -3.186063, 2.488250, 16.368689] | complete |
| ksc/equal | 261008100 | -122.503154 | [-1.655213, 0.550829] | complete |
| ksc/equal | 261008101 | -121.951203 | [-2.330234, 0.196066] | complete |
| ksc/identity | 261008100 | -123.158587 | [-1.339908, 0.462746] | complete |
| ksc/identity | 261008101 | -121.512107 | [-1.872275, 0.556556] | complete |
| ksc/global | 261008100 | -122.720613 | [-1.673710, 0.732997] | complete |
| ksc/global | 261008101 | -122.231158 | [-2.129109, 0.453984] | complete |
| ksc/local | 261008100 | -122.892909 | [-1.377162, 0.755443] | complete |
| ksc/local | 261008101 | -121.715382 | [-2.006296, 0.326001] | complete |
| predator_prey/equal | 261008100 | -241.884715 | [-5.491504, 3.196681, 0.022114, -4.112090, -3.575309, 4.516324] | complete |
| predator_prey/equal | 261008101 | -241.292760 | [-5.367501, 3.225167, 0.022383, -5.576469, -2.905795, 3.654583] | complete |
| predator_prey/identity | 261008100 | -243.584428 | [-8.492361, 3.187291, 0.020292, -5.107675, -2.737072, 3.453132] | complete |
| predator_prey/identity | 261008101 | -241.306551 | [-6.483805, 3.010932, -0.013136, -6.008252, 6.651338, -8.151152] | complete |
| predator_prey/global | 261008100 | -241.611502 | [-5.008690, 3.233081, 0.021280, -4.018390, -3.360355, 4.258475] | complete |
| predator_prey/global | 261008101 | -241.233471 | [-4.808240, 3.199668, 0.021830, -5.300951, -2.760591, 3.519942] | complete |
| predator_prey/local | 261008100 | -241.728471 | [-4.957819, 3.218167, 0.025727, -4.436056, -4.377434, 5.489725] | complete |
| predator_prey/local | 261008101 | -241.169815 | [-5.143804, 3.191304, 0.021301, -5.208087, -2.711936, 3.452662] | complete |
| sir_d18/equal | 261008100 | -1679.018236 | [-119.157128, 572.500428, -2193.071144] | complete |
| sir_d18/equal | 261008101 | -1678.181273 | [-225.059899, 192.258584, -14.080445] | complete |
| sir_d18/identity | 261008100 | -1679.765503 | [-35.205677, 152.517726, -6.906400] | complete |
| sir_d18/identity | 261008101 | -1677.923028 | [-37.657265, 152.506025, -20.021134] | complete |
| sir_d18/global | 261008100 | -141.211610 | [-134.363312, 62.744137, 36.104219] | invalid after 4/50; partial value |
| sir_d18/global | 261008101 | -102.348611 | [20.784940, -8.951159, -8.744272] | invalid after 3/50; partial value |
| sir_d18/local | 261008100 | -1679.557415 | [7.650856, 135.306409, -11.799591] | complete |
| sir_d18/local | 261008101 | -1678.657591 | [-123.129080, 170.696043, -20.842317] | complete |

PP and SIR T10/T20/T40 comparisons, with all reference scores and MCSEs, appear in the linked full numerical tables. They use data seed 26100820 and equal mixture probabilities, explicitly without tuned claims. Thus their data must not be confused with the T50 held-out seed.

## Derivative and precision diagnosis

At T3, the maximum scaled error |FD-score|/max(1,|score|) at step 5e-5 was 1.58e-8 (LGSSM), 9.35e-10 (KSC), 9.41e-8 (PP) and 1.23e-4 (SIR). GPU correction-enabled checks also completed in all four models; their maxima were 1.17e-8, 3.88e-10, 9.88e-8 and 9.59e-5 on their separately recorded data.

For SIR T50 equal mixture, the coarse steps 1e-4 and 5e-5 failed to resolve the derivative. The subsequent common-noise ladder converged toward the analytical result. The third score -2193.071144 had an absolute discrepancy of 3431.54 at 5e-5, 0.7403 at 1e-7, 0.1847 at 5e-8 and 0.000797 at 1e-8. The largest scaled discrepancy across all coordinates/seeds at 1e-8 is about 2.54e-5. This supports the finite-program derivative while revealing extreme curvature; it does not validate that score as an approximation to the model score. The tuned T50 mixture already agrees closely at the coarser ladder. Full ladders are retained, including the failures.

On identical saved SIR T3 inputs, FP32/TF32 failed covariance preservation (roughly 3–4e-4 relative error), whereas FP32 with TF32 disabled passed. Separate FP64 simulated data differ because stateless random draws depend on dtype; those runs are not a controlled dtype comparison. No tolerance was relaxed and the FP32/TF32 default was not promoted.

The global-heavy SIR arm stopped after 4 and 3 steps. Its final ESS was 1.82 and 7.24; reset covariance margins were 1.16e-16 and 5.30e-12. The second seed also failed the final covariance residual check (2.91e-11 versus its FP64 dimension-scaled limit). The equal mixture completed but had maximum Sinkhorn row-mass residuals 0.116/0.075 and reset displacements 37.30/47.58. These are explanatory diagnostics: the optional Cayley cap bounds only its additional rotation, not these preceding reset moves.

## Optional skewness/kurtosis correction

SIR T50 with two correction steps and fixed beta=(0.1,0,0.9) completed both seeds. Values were -1679.557370 and -1678.657590; scores were [7.652537,135.306893,-11.800022] and [-123.127171,170.694964,-20.842685]. Maximum standardized displacement was 1.82e-4/8.74e-5 against cap0.25; final covariance relative error stayed below 3.58e-15. The combined selected-moment objective never increased in the 100 observed reset steps. The changes were very small. This verifies mechanics and observed boundedness, not a meaningful fourth-moment accuracy gain or non-harm in every regime. The low-rank basis, feature scales, steps and caps still need dedicated calibration.

## Decisions and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain optional implementation | Shared finite algorithm and derivative/invariant tests pass in checked scopes | FP32/TF32 and global-heavy SIR remain invalid | Broader dimensions, tails, reset conditioning | Calibrate precision and finite reset sensitivity before larger claims | Production/canonical admission |
| Reject equal-mixture SIR as a usable score setting | Extreme held-out score despite finite-program parity | Accuracy and sensitivity veto | Longer particle/seed ladder | Diagnose transport scale/iteration and cloud sensitivity on fresh calibration data | Rejection of the whole proposal family |
| Retain fitted mixture as a diagnostic candidate | Valid runs and pilot gap certificate | No statistical superiority established | Small N, two seeds, bootstrap bias/MCSE | Larger scope-specific calibration and independent reference convergence | SIR fixed or HMC ready |
| Retain optional moment repair mechanics | Caps, first/second moments and derivative checks pass | No observed numerical veto in checked cases | Almost negligible movement and restricted basis | Dedicated correction non-harm/calibration study | Accurate fourth moments |

| Inference status | Finding |
|---|---|
| Hard veto screen | FP32/TF32 SIR covariance failure; global-heavy SIR numerical failure; extreme equal-mixture score |
| Statistically supported ranking | None: two candidate seeds and only four reference replications |
| Descriptive-only differences | Every displayed candidate comparison, beta validation loss, runtime and ESS |
| Default readiness | Not established; existing default unchanged |
| Next evidence needed | Precision-safe kernels, calibrated reset controls, particle/reference refinement and multi-seed uncertainty on fresh holdouts |

## Terminal review

The strongest alternative explanation for apparent gains is Monte Carlo variation plus the unusually weak first-score bootstrap reference. The sharp equal-mixture finite-difference convergence makes an omitted derivative less plausible for that checked trajectory, but does not rule out untested defects. A well-conditioned particle/refinement study with converged references could overturn the current accuracy diagnosis. The weakest evidence is statistical accuracy and generality, not the checked algebraic mechanics. Failed settings reject those settings; no continuation veto rejected the research direction. The bounded stage ends with implementation, diagnostics and documentation complete, without scientific promotion.

[Detailed equation/function audit](ledh-covariance-proposal-audit-20261008.md). Both shared-source documents compile: monograph 648 pages and standalone 24 pages, with no unresolved references. MathDevMCP returned unverified obligations; no formal verification is claimed. Initial latexmk absence and a BibTeX absolute-path restriction were repaired with installed pdflatex/bibtex and relative paths. Earlier failed numerical/test attempts remain preserved.

Run provenance records the documentation base commit 03b50cd6f and working-source hashes. The final implementation commit contains the shared numerical filter used by the completed matrix. Later edits added diagnostic step controls, reports, explicit column/coordinate diagnostics and fail-closed finite/cap guards. The final identical-input SIR rerun checks that these last observability/guard changes leave healthy values and scores unchanged to 1e-7; the complete final CPU regression suite is recorded separately. No earlier artifact source hash is relabeled as the final source hash.
