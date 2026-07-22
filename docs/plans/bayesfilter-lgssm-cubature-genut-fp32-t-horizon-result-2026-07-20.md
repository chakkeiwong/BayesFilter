# Float32/TF32 Cubature and GenUT LGSSM Result

Date: 2026-07-20

## Final Artifact

Final successful run:

docs/benchmarks/artifacts/lgssm_cubature_genut_fp32_20260720_attempt3/

The artifact is GPU-backed TensorFlow float32 with TF32 enabled. XLA was
explicitly disabled for this run because the requested score graph hit a
reproducible XLA GPU layout failure in the Cholesky gradient fusion. This is an
execution exception recorded in the JSON configuration, not a change to the
method or dtype.

GPU: NVIDIA GeForce RTX 4080 SUPER.

The final artifact reports:

- hard_valid: true;
- finite values and scores for all six arms;
- bitwise replay for every arm;
- reset mean/covariance residual below 8e-8;
- Sinkhorn row residual below 1.4e-7;
- Sinkhorn column residual below 1.2e-5;
- peak TensorFlow allocator bytes: 3166347008.

## Configuration

- state dimension d=3;
- N=1008;
- N=6*168=7*144;
- horizons T=2,10,50;
- theta=(0.72,0.55,0.35,0.35,0.45);
- stationary Gaussian initial state;
- fixed observations generated from a fixed stateless seed;
- float32 tensors;
- TF32 enabled;
- positive Sinkhorn OT with eight iterations;
- ridge 1e-5;
- same finite-program directional finite-difference score check.

## Results

| T | Method | Particle value | Kalman value | Relative value error | Score L2 error | Same-scalar FD directional error |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Cubature | -1.4944053 | -1.4707679 | 1.607e-2 | 6.573e-2 | 9.19e-5 |
| 2 | GenUT | -1.4944053 | -1.4707679 | 1.607e-2 | 6.573e-2 | 9.19e-5 |
| 10 | Cubature | -9.3065357 | -9.2449560 | 6.661e-3 | 3.492e-1 | 1.29e-3 |
| 10 | GenUT | -9.3065357 | -9.2449560 | 6.661e-3 | 3.492e-1 | 1.29e-3 |
| 50 | Cubature | -56.1550293 | -56.0868721 | 1.215e-3 | 8.400e-1 | 3.68e-3 |
| 50 | GenUT | -56.1550293 | -56.0868721 | 1.215e-3 | 8.400e-1 | 3.68e-3 |

Cubature and GenUT are bitwise identical here because Gaussian GenUT has
s_a=0, k_a=3, giving u_a=v_a=sqrt(3), b_a=c_a=1/6, and w_0=0. The zero-mass
central point is omitted from the positive equal-weight realization.

## Review and Repairs

The implementation was reviewed in phases:

1. Import-time TensorFlow initialization prevented memory-growth configuration.
   Fixed by moving tensor creation behind GPU configuration.
2. Rank-1 matrix products in the particle and Sinkhorn paths failed.
   Replaced with tf.linalg.matvec.
3. XLA gradient compilation failed without explicit loop bounds.
   Added maximum_iterations.
4. XLA GPU layout/autotuning failed in the Cholesky gradient fusion. Retrying
   with XLA_FLAGS=--xla_gpu_autotune_level=0 did not repair it. The final run
   uses the explicit --no-jit-compile GPU diagnostic exception.
5. The first successful-looking run incorrectly carried posterior weights
   after an equal-weight reset. Fixed by setting weights to uniform after every
   reset. This materially reduced the long-horizon error.
6. The DGP initially used a zero initial state while the comparator used the
   stationary covariance. Fixed both to use the same stationary initial-state
   distribution.

The first two successful-looking artifacts before repairs are not evidence:

- lgssm_cubature_genut_fp32_20260720: historical failed/partial attempts;
- lgssm_cubature_genut_fp32_20260720_attempt2: invalid for scientific
  interpretation because the DGP initial-state mismatch was still present.

## Decision Table

| Decision | Primary criterion | Veto status | Interpretation | Next action |
|---|---|---|---|---|
| Program is executable on GPU float32/TF32 | finite, replayable, invariant-valid artifacts | passed | feasibility established | retain as diagnostic harness |
| Cubature reset | value/score rows available at all horizons | passed | viable staged diagnostic candidate | multi-seed run if comparison is needed |
| Gaussian GenUT reset | positive representation and finite rows | passed | identical to cubature for this Gaussian case | use non-Gaussian moments to test distinct GenUT behavior |
| Exact Kalman agreement | not a hard implementation invariant | not promoted | observed errors are descriptive only | do not claim exact filtering correctness |
| XLA default readiness | compiler failed on this score graph | failed for this graph | no-JIT is an explicit exception | repair graph/layout separately before XLA promotion |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for final attempt |
| Statistically supported ranking | none; one deterministic seed and methods are identical |
| Descriptive differences | value and score errors relative to Kalman above |
| Default readiness | not established |
| Next evidence needed | multi-seed replication, non-Gaussian GenUT moments, and a separate XLA-compatible gradient graph |

This run does not establish exact filtering likelihood, exact filtering score,
method superiority, nonlinear-model validity, or NAWM relevance.
