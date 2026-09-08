# Corrected Theta Measure Separation

Status: `PASS_THETA_MEASURE_SEPARATION_DIAGNOSTIC`

| Partition | N | ESS fraction | max normalized weight | negative-mode fraction | log-ratio min | log-ratio max | affine latent mean max | affine latent covariance offdiag max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 232 | 0.952075 | 0.007086 | 0.529969 | -36.825588 | -31.211405 | 0.000000 | 0.000000 |
| validation | 12 | 0.965518 | 0.102760 | 0.536211 | -37.120408 | -33.563111 | 1.320572 | 0.640088 |
| audit | 12 | 0.985267 | 0.095911 | 0.526569 | -35.410442 | -33.467838 | 0.215069 | 0.383038 |

Train affine oracle: mean max `5.204e-17`, covariance residual max `6.661e-16`.

This is an explanatory diagnostic. It does not establish IID Gaussian whitening, posterior correctness, or mode discovery.
