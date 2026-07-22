# Float32/TF32 LGSSM Cubature and GenUT Diagnostic

- hard_valid: True
- dtype: float32
- tf32_mode: enabled
- GPU allocator: {'current': 1264128, 'peak': 3174900992}

| T | Method | Particle value | Kalman value | Value rel. error | Score L2 error | FD directional error |
|---:|---|---:|---:|---:|---:|---:|
| 2 | cubature | -1.4437966 | -1.4241385 | 1.380e-02 | 6.076e-02 | 9.312e-05 |
| 2 | genut | -1.4437966 | -1.4241385 | 1.380e-02 | 6.076e-02 | 9.312e-05 |
| 10 | cubature | -9.2676773 | -9.2026854 | 7.062e-03 | 3.546e-01 | 3.747e-03 |
| 10 | genut | -9.2676773 | -9.2026854 | 7.062e-03 | 3.546e-01 | 3.747e-03 |
| 50 | cubature | -56.109772 | -56.046276 | 1.133e-03 | 8.336e-01 | 8.340e-03 |
| 50 | genut | -56.109772 | -56.046276 | 1.133e-03 | 8.336e-01 | 8.340e-03 |

GenUT uses Gaussian moments s=0, k=3; its central weight is zero, so
the positive equal-weight realization omits the zero-mass center and
equals the six-point cubature design.

This is descriptive feasibility evidence, not a correctness or
superiority claim for exact filtering.
