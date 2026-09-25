# Frozen first-transition TT diagnosis

Question: why does the d4 amplitude fit jump from 3% error at t0 to 47% at t1, and does the implementation fail to model the adjacent states jointly?

This is a diagnostic within the user's authorized investigation. Freeze campaign-01 d4 t0/t1 coefficients, SGQF charts, physical parameters, observations and scaling. Do not train or select a runtime candidate on these reported holdout observations. Classification: independent diagnostic of the local extension, not source-faithful algorithm development.

## Evidence contract and skeptical audit

- Target A: exactly the joint target used at t1, using the saved normalized t0 retained TT mixture. Target B: replace only the incoming density by normalized p0(x0)g(y0|x0), independently integrated at t0. Both use the same transition, likelihood, coordinates and amplitude scaling.
- Comparator: the saved t1 amplitude; unrestricted degree-3 tensor-product projection; optimal rank-r separation of its current/past coefficient matrix; algebraic TT-SVD with grouped and paired axis order. These are diagnostic approximations only, not filtering implementations or candidates.
- Primary question: decompose measured squared amplitude error into polynomial truncation, necessary rank-3 separation error, and remaining approximation/optimization/regularization error. Compare Target A with B to measure inherited error separately.
- Check the actual call chain, and compare the reduced target formula with public physical-density evaluations. Check Gaussian quadrature normalization, basis Gram identity, exact fitted polynomial mass, projection error decompositions, and singular-value tails. A mismatch or nonfinite result invalidates diagnosis.
- Quadrature orders 5,7,9; optional 11 only if the 7-to-9 changes in mass or error exceed 0.001 and budget permits. Record differences; no continuous-space certification from unconverged quadrature. Tensor-product integration here is an independent diagnostic, never a revived retained-grid production route.
- Rank tails are exact for the specified finite weighted matrix. TT-SVD errors are descriptive algebraic approximations; an attained improvement on this target is not a general method ranking. A failed current rank is a repair trigger, not a continuation veto for TT research.
- No conclusion of optimal runtime settings, general superiority, full filtering repair, HMC readiness, or high-dimensional performance.

Defaults: use the executed degree 3, rank 3 and saved scaling, not freshly chosen training controls. The polynomial projection supplies an independent best linear-space diagnostic. Paired ordering is motivated by the nearly diagonal transition and must not be interpreted as a drop-in change to the existing marginal/KR consumers. Product Gaussian quadrature may underresolve tails; order comparisons expose this. Only one first-transition target is localized; fresh calibration and replication are required later.

Pre-mortem: separately whitened coordinates are not independence assumptions. Their transforms preserve the joint transition density, and separate invertible transformations cannot eliminate intrinsic block separation rank. A coupled triangular transform may reduce a rank burden while invalidating one current consumer. Verify both facts analytically before suggesting a map repair. Hold the incoming law fixed for fitting comparisons; distinguish inherited-law error from fitter error.

Audit outcome: plan answers the active question with frozen artifacts and explicit diagnostic exceptions. It does not rerun a particle comparison or tune on the holdout. Source anchors inspected: Zhao–Cui Sec.3.1/equation13 and Algorithms2–3; author `@TTSIRT/eval_cirt_reference.m` suffix conditioning and `marginalise.m` suffix Gram propagation. Local behavior is an extension.

## Execution

Program: `docs/benchmarks/diagnose_observation_tt_first_transition.py`. TensorFlow float64 on the RTX5080 with escalated access and verified memory growth. Stable-signature target/projection kernels default to XLA; diagnostic SVD, small eigensystems and artifact setup are explicit non-XLA reference exceptions. No NumPy or pfor. Verify matrix computations and source hashes in a manifest.

One diagnostic launch, at most 600 seconds (within the previously unspent 2300.426 numerical seconds); at most one localized infrastructure repair/retry within that same total. New output root: `docs/benchmarks/artifacts/observation_tt_first_transition_20260914/run-01/`. Preserve each attempted order before moving on. Stop at budget, failed numerical invariant, or nonfinite target; preserve traceback and partial outputs.

Result note: `docs/plans/observation-tt-first-transition-root-cause-20260914-result.md`. Write decision and inference status, remaining uncertainty, actual command/hardware/wall time, and the next justified repair.

Localized diagnostic repair: run-01 completed orders 5,7,9 in 13.104 seconds. Its optional-order check monitored only saved-fit error and mass; the declared projection/ordering/inherited-error diagnostics still moved by more than 0.001. Extend that check to every declared quantity and use the one permitted fresh-directory retry, run-02, under the same 600-second total budget. Target and numerical computations are unchanged. Run-01 remains valid at its reported orders; it does not establish convergence of omitted checks.
