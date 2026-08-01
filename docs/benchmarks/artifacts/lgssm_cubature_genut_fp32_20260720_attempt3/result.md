# Float32/TF32 LGSSM Cubature and GenUT Diagnostic

- hard_valid: True
- dtype: float32
- tf32_mode: enabled
- GPU allocator: {'current': 1264128, 'peak': 3166347008}

| T | Method | Particle value | Kalman value | Value rel. error | Score L2 error | FD directional error |
|---:|---|---:|---:|---:|---:|---:|
| 2 | cubature | -1.4944053 | -1.4707679 | 1.607e-02 | 6.573e-02 | 9.187e-05 |
| 2 | genut | -1.4944053 | -1.4707679 | 1.607e-02 | 6.573e-02 | 9.187e-05 |
| 10 | cubature | -9.3065357 | -9.244956 | 6.661e-03 | 3.492e-01 | 1.292e-03 |
| 10 | genut | -9.3065357 | -9.244956 | 6.661e-03 | 3.492e-01 | 1.292e-03 |
| 50 | cubature | -56.155029 | -56.086872 | 1.215e-03 | 8.400e-01 | 3.683e-03 |
| 50 | genut | -56.155029 | -56.086872 | 1.215e-03 | 8.400e-01 | 3.683e-03 |

GenUT uses Gaussian moments s=0, k=3; its central weight is zero, so
the positive equal-weight realization omits the zero-mass center and
equals the six-point cubature design.

This is descriptive feasibility evidence, not a correctness or
superiority claim for exact filtering.
