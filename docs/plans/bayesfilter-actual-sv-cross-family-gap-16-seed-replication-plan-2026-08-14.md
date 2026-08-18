# Experiment plan: actual-SV cross-family gap 16-seed replication

## Question
Is the dim-2 cross-family raw-`y` gap observed in
`actual_sv_three_route_simulation_20260814/attempt01`
(fitted-28 Kalman minus exact dense = -0.0144, larger than dims 1/3 and
sign-flipped) a property of that one simulated path (seed scatter) or a
systematic dim-2 / parameter-set effect (or a bug)?

## Mechanism tested
Replicate the dense-route comparison over 16 independent simulated actual-SV
paths per dimension and measure the seed distribution of paired per-path
differences. TT routes are excluded: they already passed same-target checks
to <=6e-5 and are not the quantity in question.

## Scope
- 16 seed bases: 83120 + 20000*k, k = 0..15 (k=0 reproduces attempt01 paths).
- Dims 1/2/3, horizon 20, CPU-only float64, deterministic.
- Quantities per (seed, dim), raw-`y` after exact Jacobian correction:
  - exact-transformed dense reference (order 401, radius 8),
  - dense KSC-7 same-target reference (coordinate-wise),
  - mixture-Kalman with fitted 7/14/28-component mixtures (fitted once,
    identical across seeds).
- Differences: d_K = Kalman(K) - exact dense for K in {7f,14f,28f};
  dKSC = KSC-7 dense - exact dense.

## Evidence contract
- Primary diagnostic: seed distribution (mean, sd, 95% t-interval, min/max)
  of d_28 per dim, and where the k=0 (original) value falls in it.
- Veto: any non-finite value or factorization-check failure.
- Explanatory only: dKSC, d_7f, d_14f distributions; refinement stability
  across seeds.
- Interpretation rule:
  - if the original dim-2 value lies inside the seed scatter and the dim-2
    mean is comparable to dims 1/3, classify the -0.0144 as path-level
    Monte Carlo variation (nothing incorrect);
  - if dim-2's mean is shifted relative to dims 1/3 beyond its t-interval,
    classify as parameter-set-dependent approximation bias and open a
    follow-up (not a bug claim yet);
  - if any seed shows erratic/non-monotone refinement, suspect a bug.
- Not concluded regardless of outcome: no family ranking, no HMC/posterior
  claims, no default changes. 16 seeds supports interval statements about
  these fixture distributions only.

## Commands
`python docs/benchmarks/run_actual_sv_cross_family_gap_seed_sweep.py
  --output docs/benchmarks/artifacts/actual_sv_cross_family_gap_seed_sweep_20260814/attempt01/result.json`

## Artifacts
- JSON + markdown under
  `docs/benchmarks/artifacts/actual_sv_cross_family_gap_seed_sweep_20260814/`
- result note `bayesfilter-actual-sv-cross-family-gap-16-seed-result-2026-08-14.md`
