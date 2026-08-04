# Austria Growth Versus Particle Count Result

Date: 2026-08-01
Status: `N_INCREASE_SCORE_DIAGNOSTIC_COMPLETED_DIAGONAL_VETO`

## Verdict

`N=4000` cannot be run on the current exact replicated cubature route because
Austria has `d=18` and the design requires `N % 36 == 0`. The nearest legal
count, `N=4032`, was run with the same Austria controls, seeds, TF32/XLA
settings, and eight fixed growth probes.

Increasing particle count did not make the full particle-filter tangent more
contractive after step 7. Relative to `N=1008`, diagonal growth was slightly
larger and pairwise growth was substantially larger in this diagnostic. This
does not support the hypothesis that the near-zero later growth in the earlier
plot is primarily a finite-`N` effect that disappears at larger `N`.

The physical deterministic RK4 transition is unchanged: it becomes
contracting after approximately step 6-7. The full filter remains mixed or
expansive at later steps, so filtering operations still contribute amplification
beyond the physical transition.

## Results

| Arm | N | Mean growth | Steps 8-20 mean | Steps 8-20 negative pooled steps | Total cumulative log growth |
|---|---:|---:|---:|---:|---:|
| Diagonal | 1008 | `+0.2724` | `+0.0834` | `3/13` | `+5.4474` |
| Diagonal | 4032 | `+0.3354` | `+0.0957` | `7/13` | `+6.7085` |
| Pairwise | 1008 | `+0.2394` | `+0.0776` | `5/13` | `+4.7877` |
| Pairwise | 4032 | `+0.3661` | `+0.2659` | `0/13` | `+7.3222` |

The matched physical transition cumulative log growth is `-1.0283` (factor
`0.358`). The full-filter factors are approximately `232x` and `120x` at
`N=1008`, versus `819x` and `1510x` at `N=4032` for diagonal and pairwise
arms respectively.

These are descriptive finite-horizon probe summaries, not estimates of an
asymptotic Lyapunov exponent or statistical rankings. The larger-N result is
also a different finite program, and controls were not retuned.

## Plot

[N=4032 physical/full-filter plot](../benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/derived_physical_vs_full_20260801_v2/austria_physical_vs_full_particle_growth.png)

[Per-step CSV](../benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/derived_physical_vs_full_20260801_v2/growth_by_step.csv)

[N=4032 raw result](../benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/attempt03/result.json)

## Score Diagnostic At N=4032

A follow-up score-only run used the same controls, observations, three seeds,
GPU/XLA, TF32, and verified memory growth. The artifact is
`docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/attempt06/`.

| Arm | Hard-valid | Score mean | Score SD across 3 seeds | SD/sqrt(3) descriptive seed-mean MCSE |
|---|---:|---|---|---|
| Diagonal | **No** | unavailable | unavailable | unavailable |
| Pairwise | Yes | `[-15.9340, -106.5231, 8.5470]` | `[34.2476, 20.8837, 5.4906]` | `[19.7738, 12.0579, 3.1709]` |

The diagonal arm is not a valid comparison arm at `N=4032`: seed `98202`
returned a non-finite value and score. Repeating the same score-only command
in attempt 05 produced the same failure for that seed, while the pairwise arm
was finite in both attempts. This is a hard numerical-validity veto, not an
estimate of diagonal score error.

For the finite pairwise arm, the `N=1008 -> N=4032` mean-score displacement is
`[-5.0776, +11.5519, -4.4866]`. Relative to the `N=1008` pairwise score SD,
the absolute displacement is `[0.14, 2.20, 1.23]` SDs by coordinate. The
pairwise coordinate SD ratios are `[0.965, 3.986, 1.505]`; aggregate score
variance (sum of coordinate variances) is `1.261x` the `N=1008` value and the
three-coordinate RMS SD rises from `20.81` to `23.38` (`+12.3%`). Thus this
run does not show a smaller pairwise score error at larger `N`; the second and
third coordinates are more dispersed.

The `SD/sqrt(3)` column is only the standard error of a three-seed mean under
an iid particle-seed interpretation. It is not an MCMC MCSE, and it does not
measure absolute bias. Austria has no exact nonlinear score oracle in this
campaign, so absolute score error remains unknown.

## Evidence And Limitations

- Attempt 03: `hard_valid=true`, TensorFlow 2.19.1, RTX 4080 SUPER, GPU/XLA,
  TF32, verified memory growth, `N=4032`, eight sequential probes, wall time
  `952.653 s`.
- Attempt 01 failed before the numerical route because the reused noise helper
  produced `N=1008` tensors. This was repaired by generating the same stateless
  noise salts `[101,102]` at the requested particle count.
- Attempt 02 reached the numerical route but failed with GPU
  `RESOURCE_EXHAUSTED` in dense Sinkhorn/JVP workspace. The sequential
  probe-batch repair preserved all eight probes and passed in attempt 03.
- `N=4000` was explicitly rejected by the design preflight; `N=4032` was
  accepted as `36 x 112` exact replication.
- Attempt 06 completed the score-only route for both arms, but the diagonal
  arm failed reproducibly at seed `98202`; only pairwise score dispersion is
  interpretable at `N=4032`.
- Three seeds and eight probes remain descriptive. No superiority, causal
  attribution, HMC, default, or scientific promotion claim is made.

## Decision Table

| Decision | Criterion | Status | Next action | Not concluded |
|---|---|---|---|---|
| Treat early physical expansion as real | Physical transition becomes negative after step 6-7 | Supported descriptively | Keep physical/full-filter curves separate | No global model instability theorem |
| Attribute later contraction to larger N | N=4032 post-step-7 growth lower than N=1008 | Rejected by this diagnostic | Test a legal N ladder with retuning only if needed | No universal N law from two counts |
| Retain pairwise as a stability repair | Full-filter growth non-positive after step 7 | Failed; N=4032 pairwise remains positive | Do not promote; investigate stage ablations/design route | No bias or HMC conclusion |
| Treat larger-N pairwise score as lower error | Pairwise score SD and N-displacement improve at N=4032 | Rejected by current diagnostic; aggregate SD is `1.261x` and diagonal is invalid | Keep pairwise as a candidate only; obtain an eligible score reference or larger replicated ladder | No absolute bias estimate or ranking |
