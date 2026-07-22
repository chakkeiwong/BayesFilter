# Matched Three-Horizon Recursive-Score Comparison Result

Date: 2026-07-21

> Post-review clarification, 2026-07-21: raw physical-score error intervals in
> the JSON are the primary score-agreement object. The percentage tables below
> are coordinate-wise relative errors and are descriptive near zero Kalman
> coordinates. Although the implementation formed them after multiplying both
> scores by positive diagonal HMC chain factors, those factors cancel from each
> coordinate-wise ratio; the numerical percentages are unchanged. The report
> generator now leads with raw physical-score errors and emits an explicit
> `coordinate_relative_error` field.

## Outcome

The matched `T=2,10,50` comparison completed with `hard_valid=true`. Original
Contract E with a centered Gaussian residual design and Cubature/Gaussian GenUT
used the same `N=1008`, 16 seeds, target, finite Sinkhorn controls, Cholesky
restoration, float32/TF32 GPU execution, and recursive no-autodiff score.

Gaussian GenUT is algebraically and bitwise identical to Cubature for the
Gaussian moments `s=0`, `k=3`; it is therefore an alias in this result, not an
independent stochastic arm.

No paired simultaneous interval comparing absolute Kalman error excludes zero.
The two reset designs are statistically indistinguishable under this 16-seed
comparison. Some descriptive mean differences are visible, but none supports a
method-wide or coordinate-specific ranking.

## Artifacts

- Plan: `docs/plans/bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-plan-2026-07-21.md`
- JSON: `docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/attempt02/result.json`
- Generated Markdown: `docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/attempt02/result.md`
- Run manifest: `docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/attempt02/run_manifest.json`
- Command: `python docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py --output-root docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/attempt02`

The first launch (`attempt01`) failed before GPU work because the direct script
did not add the repository root to `sys.path`. No artifact was created. The
localized bootstrap repair was verified and `attempt02` completed in `209.3 s`.

## Likelihood Value

Intervals are the same 16-seed simultaneous 95% convention used in the prior
LGSSM comparisons. Relative error is `(particle - Kalman)/abs(Kalman)`.

| T | Kalman value | Method | Mean particle value | Mean relative error [simultaneous interval] |
|---:|---:|---|---:|---:|
| 2 | `-8.862151` | Contract E Gaussian | `-8.855339` | `+0.077% [-1.461%, +1.615%]` |
| 2 | `-8.862151` | Cubature = GenUT | `-8.857111` | `+0.057% [-1.506%, +1.620%]` |
| 10 | `-32.052616` | Contract E Gaussian | `-32.138622` | `-0.268% [-0.603%, +0.067%]` |
| 10 | `-32.052616` | Cubature = GenUT | `-32.176457` | `-0.386% [-0.750%, -0.022%]` |
| 50 | `-136.075975` | Contract E Gaussian | `-135.918271` | `+0.116% [-0.135%, +0.367%]` |
| 50 | `-136.075975` | Cubature = GenUT | `-136.084744` | `-0.006% [-0.240%, +0.227%]` |

At `T=10`, the Cubature value interval excludes zero, so that arm has evidence
of a small negative value bias relative to Kalman under this screen. This does
not establish that Contract E Gaussian is better: the paired difference in
absolute value error still includes zero.

## Physical Score

### Raw Means

Coordinates are `(phi1, phi2, phi3, q_scale, r_scale)`.

| T | Arm | Physical score vector |
|---:|---|---|
| 2 | Kalman | `(3.828014, -0.384181, -0.083087, 4.417213, 11.138136)` |
| 2 | Contract E Gaussian mean | `(3.906515, -0.372409, -0.070174, 4.453207, 11.316507)` |
| 2 | Cubature = GenUT mean | `(3.899511, -0.388288, -0.069649, 4.449878, 11.280953)` |
| 10 | Kalman | `(11.279994, -0.304041, -1.305039, 9.488660, 14.068336)` |
| 10 | Contract E Gaussian mean | `(11.763834, -0.338939, -1.447541, 9.652405, 14.540095)` |
| 10 | Cubature = GenUT mean | `(11.879055, -0.367203, -1.401704, 10.108611, 14.714762)` |
| 50 | Kalman | `(5.655446, -3.835057, 0.302362, -1.917176, 4.354276)` |
| 50 | Contract E Gaussian mean | `(5.068498, -3.902814, 0.409842, -2.464738, 4.192434)` |
| 50 | Cubature = GenUT mean | `(5.518827, -4.119177, 0.313136, -1.845925, 5.351534)` |

### Kalman-Relative Errors

| T | Method | phi1 | phi2 | phi3 | q_scale | r_scale |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Contract E Gaussian | `+2.051% [-9.483%, +13.585%]` | `+3.064% [-24.601%, +30.729%]` | `+15.542% [-70.121%, +101.206%]` | `+0.815% [-12.204%, +13.834%]` | `+1.601% [-6.867%, +10.070%]` |
| 2 | Cubature = GenUT | `+1.868% [-9.778%, +13.514%]` | `-1.069% [-31.973%, +29.835%]` | `+16.173% [-65.698%, +98.045%]` | `+0.740% [-13.124%, +14.603%]` | `+1.282% [-7.252%, +9.816%]` |
| 10 | Contract E Gaussian | `+4.289% [-0.807%, +9.385%]` | `-11.478% [-86.026%, +63.071%]` | `-10.919% [-27.813%, +5.974%]` | `+1.726% [-7.269%, +10.721%]` | `+3.353% [-0.863%, +7.570%]` |
| 10 | Cubature = GenUT | `+5.311% [-0.447%, +11.069%]` | `-20.774% [-82.970%, +41.422%]` | `-7.407% [-24.469%, +9.655%]` | `+6.534% [-1.631%, +14.699%]` | `+4.595% [-1.412%, +10.602%]` |
| 50 | Contract E Gaussian | `-10.378% [-22.236%, +1.479%]` | `-1.767% [-16.399%, +12.866%]` | `+35.547% [-25.025%, +96.119%]` | `-28.561% [-124.966%, +67.845%]` | `-3.717% [-57.624%, +50.190%]` |
| 50 | Cubature = GenUT | `-2.416% [-16.666%, +11.834%]` | `-7.408% [-15.626%, +0.809%]` | `+3.563% [-95.636%, +102.763%]` | `+3.717% [-94.669%, +102.102%]` | `+22.903% [-29.265%, +75.071%]` |

Every score interval includes zero. The largest uncertainties occur where the
Kalman coordinate is small or Monte Carlo variability is large, especially
`phi3`, `q_scale`, and `r_scale` at `T=50`. Raw physical-score errors in the
JSON should be used alongside these relative errors near zero denominators.

## Paired Method Comparison

Each entry is the paired difference in absolute Kalman-relative error,
`Cubature - Contract E Gaussian`, in percentage points. Negative favors
Cubature. Every interval includes zero.

| T | Value | phi1 | phi2 | phi3 | q_scale | r_scale |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `+0.021 [-0.130, +0.172]` | `+0.266 [-1.913, +2.445]` | `+3.883 [-3.041, +10.808]` | `-6.167 [-24.509, +12.175]` | `+1.781 [-0.351, +3.913]` | `+0.074 [-0.432, +0.579]` |
| 10 | `+0.088 [-0.193, +0.370]` | `+0.801 [-3.700, +5.303]` | `-6.791 [-52.395, +38.813]` | `-1.156 [-11.655, +9.343]` | `-0.508 [-7.638, +6.622]` | `+2.294 [-1.523, +6.110]` |
| 50 | `-0.030 [-0.227, +0.166]` | `+1.212 [-6.823, +9.247]` | `-3.217 [-12.032, +5.599]` | `+31.891 [-18.381, +82.164]` | `-4.933 [-88.087, +78.222]` | `+3.881 [-38.035, +45.798]` |

## Validity And Runtime

- All 96 executed particle rows were finite and bitwise replayable.
- Every row used
  `compact_forward_sensitivity_no_autodiff_cubature_genut_v1`.
- Candidate FD runtime score was disabled in every row.
- The Kalman reference used `analytic_recursive_kalman_score`.
- Reset mean/covariance and Sinkhorn marginal gates passed.
- GPU allocator peak was `474,037,248` bytes, about `452 MiB`.
- Total wall time was `209.3 s` on the RTX 4080 SUPER.
- TensorFlow emitted retracing warnings because deterministic seed-specific
  graphs are created separately. This affects timing interpretation, not the
  numerical comparison.

## Decision Table

| Decision | Status |
|---|---|
| Engineering correctness | passed hard-valid and replay gates |
| Value agreement with Kalman | generally centered around zero; small `T=10` Cubature negative bias detected |
| Score agreement with Kalman | all simultaneous intervals include zero, with substantial uncertainty in weak coordinates |
| Cubature/GenUT versus Contract E Gaussian | statistically indistinguishable under paired 16-seed evidence |
| Promotion/default decision | none |
| Next justified action | if narrower method comparison is needed, increase paired seeds or improve variance reduction before changing the algorithm |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | passed |
| Statistically supported ranking | none |
| Descriptive-only differences | all method mean differences and runtimes |
| Default readiness | not established |
| Next evidence needed | stronger paired precision and nonlinear-target validation |

## Post-Run Red Team

The strongest alternative explanation is that residual-design effects are
smaller than seed-level Monte Carlo variability at `N=1008`, so this campaign
is underpowered to rank them. The result does not show that the methods are
mathematically identical; it shows that no difference was resolved under the
declared paired uncertainty analysis. It also does not transfer automatically
to a non-Gaussian nonlinear model, where GenUT moments need not reduce to
Cubature and the residual design may matter more.
