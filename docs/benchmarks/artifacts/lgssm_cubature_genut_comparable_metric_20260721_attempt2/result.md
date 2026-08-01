# Float32/TF32 LGSSM Cubature and GenUT Diagnostic

- hard_valid: True
- dtype: float32
- tf32_mode: enabled
- GPU allocator: {'current': 1279232, 'peak': 3361823232}

Comparison metric: previous Contract E six-coordinate HMC-relative-error metric.

| T | Method | Value mean rel. error | Value simultaneous 95% CI | q_scale mean rel. error | q_scale simultaneous 95% CI |
|---:|---|---:|---:|---:|---:|
| 2 | cubature | 5.687e-04 | [-1.506e-02, 1.620e-02] | 7.149e-03 | [-1.315e-01, 1.458e-01] |
| 2 | genut | 5.687e-04 | [-1.506e-02, 1.620e-02] | 7.149e-03 | [-1.315e-01, 1.458e-01] |
| 10 | cubature | -3.864e-03 | [-7.504e-03, -2.230e-04] | 6.531e-02 | [-1.635e-02, 1.470e-01] |
| 10 | genut | -3.864e-03 | [-7.504e-03, -2.230e-04] | 6.531e-02 | [-1.635e-02, 1.470e-01] |
| 50 | cubature | -6.453e-05 | [-2.397e-03, 2.268e-03] | 3.915e-02 | [-9.448e-01, 1.023e+00] |
| 50 | genut | -6.453e-05 | [-2.397e-03, 2.268e-03] | 3.915e-02 | [-9.448e-01, 1.023e+00] |

The JSON retains per-seed raw physical/HMC scores, raw score L2, and
finite-difference diagnostics as secondary metrics.

GenUT uses Gaussian moments s=0, k=3; its central weight is zero, so
the positive equal-weight realization omits the zero-mass center and
equals the six-point cubature design.

This is descriptive feasibility evidence, not a correctness or
superiority claim for exact filtering.
