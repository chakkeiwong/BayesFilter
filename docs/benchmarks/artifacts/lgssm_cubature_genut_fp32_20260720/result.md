# Float32/TF32 LGSSM Cubature and GenUT Diagnostic

- hard_valid: True
- dtype: float32
- tf32_mode: enabled
- GPU allocator: {'current': 1268992, 'peak': 3191230464}

| T | Method | Particle value | Kalman value | Value rel. error | Score L2 error | FD directional error |
|---:|---|---:|---:|---:|---:|---:|
| 2 | cubature | -1.456778 | -1.4241385 | 2.292e-02 | 9.550e-02 | 5.962e-05 |
| 2 | genut | -1.456778 | -1.4241385 | 2.292e-02 | 9.550e-02 | 5.962e-05 |
| 10 | cubature | -9.4477921 | -9.2026854 | 2.663e-02 | 3.675e+00 | 6.831e-03 |
| 10 | genut | -9.4477921 | -9.2026854 | 2.663e-02 | 3.675e+00 | 6.831e-03 |
| 50 | cubature | -72.500053 | -56.046276 | 2.936e-01 | 1.502e+02 | 5.690e-01 |
| 50 | genut | -72.500053 | -56.046276 | 2.936e-01 | 1.502e+02 | 5.690e-01 |

GenUT uses Gaussian moments s=0, k=3; its central weight is zero, so
the positive equal-weight realization omits the zero-mass center and
equals the six-point cubature design.

This is descriptive feasibility evidence, not a correctness or
superiority claim for exact filtering.
