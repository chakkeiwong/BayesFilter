# Comparable Cubature/GenUT LGSSM Metric Result

Date: 2026-07-21

## Final Artifact

`docs/benchmarks/artifacts/lgssm_cubature_genut_comparable_metric_20260721_attempt5/`

Command:

```bash
python docs/benchmarks/run_lgssm_cubature_genut_fp32.py \
  --no-jit-compile \
  --output-root docs/benchmarks/artifacts/lgssm_cubature_genut_comparable_metric_20260721_attempt5
```

The run used the trusted NVIDIA GPU path, TensorFlow float32 candidate
arithmetic with TF32 enabled, and the explicit no-JIT exception already
required by the Cholesky-gradient compiler failure. It evaluated 16 particle
seeds (`82220..82235`) at `T=2,10,50` for both methods.

## Metric

This is the previous Contract E metric, not score L2. For each seed, the
artifact stores six relative errors in this order:

```text
value, phi1, phi2, phi3, q_scale, r_scale
```

The score is transformed from physical to HMC coordinates using
`(1-phi1^2, 1-phi2^2, 1-phi3^2, q_scale, r_scale)`. Each coordinate is
normalized by the corresponding exact Kalman value/score, then summarized over
16 seeds using sample SD, SE, and critical value `3.036283222821165`.

The observations are the prior dataset (`DATASET_SEED=81100`) and canonical
3-dimensional observation matrix. The candidate remains float32/TF32; the
Kalman normalization path matches the prior aggregation by casting those
float32 observations to float64 before calling `tf_kalman_log_likelihood`.

## Results

All values below are mean relative errors; intervals are the same simultaneous
95% intervals used by the previous Contract E aggregation. Cubature and
Gaussian GenUT are bitwise identical for all 96 paired seed-arm evaluations.

| T | Method | Value | phi1 | phi2 | phi3 | q_scale | r_scale |
|---:|---|---|---|---|---|---|---|
| 2 | Cubature = GenUT | `+0.000569 [-0.015064,+0.016201]` | `+0.018606 [-0.097853,+0.135065]` | `-0.011210 [-0.320330,+0.297910]` | `+0.161366 [-0.657504,+0.980237]` | `+0.007150 [-0.131536,+0.145835]` | `+0.012839 [-0.072497,+0.098175]` |
| 10 | Cubature = GenUT | `-0.003864 [-0.007504,-0.000223]` | `+0.053092 [-0.004484,+0.110668]` | `-0.207253 [-0.829292,+0.414785]` | `-0.074200 [-0.244829,+0.096429]` | `+0.065309 [-0.016350,+0.146967]` | `+0.045940 [-0.014134,+0.106014]` |
| 50 | Cubature = GenUT | `-0.000064 [-0.002397,+0.002268]` | `-0.024144 [-0.166643,+0.118356]` | `-0.073895 [-0.156099,+0.008308]` | `+0.038550 [-0.953546,+1.030646]` | `+0.039151 [-0.944743,+1.023045]` | `+0.228760 [-0.292948,+0.750467]` |

`hard_valid=true`; all values/scores are finite and replayable, reset
mean/covariance residuals are below `1.2e-7`, and Sinkhorn column residuals
are below `2.6e-5`. Peak TensorFlow allocator usage was `3,353,613,056`
bytes (about `3.12 GiB`).

## Interpretation

- The new results are directly comparable to the prior Contract E runs with
  respect to target, parameter order, HMC transformation, normalization, and
  six-coordinate interval metric.
- This does not make the numerical methods identical: this diagnostic uses
  `N=1008`, `epsilon=2`, `sinkhorn_steps=8`, and no JIT, while prior Contract E
  arms used their own tuned controls. Differences in the reported errors remain
  descriptive method/control evidence.
- `T=50` value relative error is centered near zero, but the score intervals,
  especially `phi3`, `q_scale`, and `r_scale`, are wide. The result is not an
  exact-filtering or superiority claim.
- Gaussian GenUT has `s=0`, `k=3`, hence zero central weight and the same six
  noncentral points as the cubature rule. A distinct GenUT result requires
  non-Gaussian moment inputs.

## Repair History

1. The first launch initialized TensorFlow while importing the dataset module,
   preventing memory-growth configuration. The dataset import is now deferred
   until after GPU configuration.
2. The first corrected oracle launch exposed a TensorFlow `tf.eye` positional
   argument typo, then a float32/float64 subtraction boundary. Both were fixed;
   the final run casts candidate outputs to float64 only for comparison with
   the prior float64 Kalman oracle.

The failed attempts wrote no result artifact and are not evidence. The final
artifact is `attempt5`.

## Evidence Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed |
| Same prior comparison metric | Passed |
| Statistically supported method ranking | None; methods are identical here |
| Descriptive differences | Present in the six-coordinate intervals above |
| Exact filtering validity | Not established |
| XLA/default readiness | Not established for this graph |
| Nonlinear-model or NAWM claim | Not made |
