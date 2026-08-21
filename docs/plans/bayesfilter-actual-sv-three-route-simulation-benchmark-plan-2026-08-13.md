# Experiment plan: actual-SV three-route simulation benchmark

## Question
On common simulated datasets from the exact actual-SV DGP, how do the three Zhao-Cui-style approximation families compare against their own dense same-target references and against the dense Gaussian-mixture / Kalman approximation after exact transformation correction?

## Mechanism being tested
We simulate paths from the exact actual-SV model and evaluate, on the same datasets:
1. the fixed-variant actual-SV batch TT route,
2. the exact-transformed Zhao-Cui route,
3. the KSC-surrogate Zhao-Cui route,
4. the dense Gaussian-mixture / Kalman approximation.

Each route is first judged against its own dense same-target benchmark. Only then are cross-route gaps interpreted, and only after exact transformation/Jacobian correction puts all reported likelihoods into the same raw-`y` representation.

## Scope
- DGP: exact actual-SV model.
- Data: simulated paths, not the two-row deterministic fixture.
- Dimensions: 1 / 2 / 3.
- Horizon: moderate (e.g. 20) so the benchmark is still cheap but statistically meaningful.
- Runtime target: CPU-only.

## Success criteria
- The benchmark emits one consolidated artifact with separate sections for:
  - fixed-variant actual-SV batch TT,
  - exact-transformed Zhao-Cui,
  - KSC-surrogate Zhao-Cui,
  - dense Gaussian-mixture / Kalman approximation.
- Each route has a same-target dense reference comparison.
- Cross-route comparisons are reported only after exact transformation correction and never treated as same-target proofs.
- The benchmark identifies whether any route is internally inconsistent or whether disagreements are only cross-family approximation differences.

## Diagnostics
Primary:
- fixed-variant actual-SV batch TT vs its own dense same-target gap,
- exact-transformed Zhao-Cui vs its own dense same-target gap,
- KSC-surrogate Zhao-Cui vs its own dense KSC reference gap,
- dense Kalman 7 / 14 / 28 component refinement ladder,
- transformed-back raw-`y` cross-route likelihood gaps.

Secondary:
- route-level score vs finite-difference error,
- route-level status / validity flags,
- whether dense Kalman refinement changes are below 1%.

## Interpretation rule
- If a route fails against its own dense same-target reference, do not trust it in cross-route interpretation.
- If all routes pass same-target checks but differ across families, interpret the gaps as approximation-family differences rather than bugs.
- If dense Kalman stabilizes under refinement, treat it as converged at that budget on the tested path.

## Skeptical audit
- Do not use the tiny deterministic fixture as evidence; it is diagnostic-only.
- Do not compare transformed-space and raw-`y` likelihoods without the exact Jacobian correction.
- Do not conflate the KSC surrogate target with the exact actual-SV transformed target.
- Keep route families separate in both JSON and markdown output.

## Files to touch
- `docs/benchmarks/benchmark_actual_sv_two_lane_comparison.py` or a new three-route benchmark script
- `tests/test_actual_sv_two_lane_benchmark_script.py` or a new benchmark-schema test
- `docs/plans/bayesfilter-actual-sv-three-route-simulation-benchmark-plan-2026-08-13.md`
- result artifacts to be emitted under `docs/benchmarks/`

## Skeptical pre-execution audit (2026-08-14, recovery session)

Audit outcome: plan is sound but three scope revisions are required before the
commands would answer the stated question.

1. Dense Kalman refinement cost. The existing
   `independent_panel_sv_mixture_kalman_filter` enumerates `K**dim` component
   tuples per step in a Python loop; at `K=28, dim=3, horizon=20` this is
   ~4.4e5 sequential Kalman updates and would not finish in a benchmark
   session. Because the panel target is an independent product across
   coordinates (diagonal transition, diagonal process noise, per-coordinate
   observation), the mixture-Kalman log-likelihood factorizes exactly as the
   sum of per-coordinate scalar mixture-Kalman log-likelihoods. The benchmark
   computes the refinement ladder coordinate-wise and includes an internal
   factorization check against the joint enumeration at `dim=2, K=7`
   (tolerance 1e-8) so the shortcut is verified, not assumed.
2. Fixed-variant actual-SV batch TT scope. The batch TT route
   (`bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf`) is a scalar
   fixed-`sigma=1` route with UKF-frozen cores. It is reported for `dim=1`
   only, with cores rebuilt (center-frozen at truth) on the simulated dataset,
   because the shipped adapter's frozen T10 seed-81101 dataset is not the
   simulated fixture this plan requires. Its same-target dense reference is
   the exact-transformed dense reference at the same physical parameters.
3. Fitted 7/14/28-component mixtures. No fitted-mixture builder exists in the
   repository; the 2026-08-12 refinement plan calls for fitted Gaussian
   mixtures to the exact `log(chi^2_1)` density. The benchmark implements a
   deterministic quadrature-weighted EM fit in TensorFlow float64
   (KSC-7-initialized, deterministic component splitting for 14/28), records
   fit quality (weighted L1 density error) per K, and pins mixtures in the
   artifact manifest. The KSC 1998 pinned 7-component mixture is also reported
   separately as the historical baseline.

Score checks (secondary diagnostics) run at `dim=1` only to bound runtime;
value comparisons run at all requested dims. Raw-`y` cross-route correction
uses `exact_transformed_sv_jacobian_log_abs_det` for offset-0 exact routes and
the generalized `sum log((y^2+c)/|y|)` correction for offset-`c` surrogate
routes; the two agree at `c=0` and the script asserts this.

Evidence contract restated: same-target gaps are promotion-relevant veto
diagnostics for route internal consistency; cross-family raw-`y` gaps are
descriptive/explanatory only; the 1% refinement rule is an empirical
stabilization screen, not a convergence proof; single-path simulated data at
horizon 20 supports no statistical ranking of families.

## Execution record (2026-08-14)

Executed after the audit above, in the recovery session that continued the
stalled "load delegated dancing plan" session. New files:
`docs/benchmarks/benchmark_actual_sv_three_route_simulation.py`,
`tests/test_actual_sv_three_route_benchmark_script.py` (passed), artifacts
under `docs/benchmarks/artifacts/actual_sv_three_route_simulation_20260814/attempt01/`.
The two-lane harness's in-progress simulation migration was repaired
(per-coordinate independent simulated paths) and its schema test passes.
Result note with decision and inference-status tables:
`docs/plans/bayesfilter-actual-sv-three-route-simulation-benchmark-result-2026-08-14.md`.
Headline: all routes pass their own same-target references; dense
Gaussian-mixture route stabilizes under the fitted 7/14/28 ladder; the
remaining cross-family raw-`y` difference is dominated by KSC-7 mixture bias
(shrinks ~10x under fitted-28), all descriptive only.
