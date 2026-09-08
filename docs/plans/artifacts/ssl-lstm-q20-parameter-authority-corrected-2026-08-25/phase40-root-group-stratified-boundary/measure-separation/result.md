# v2.2 Root-Group-Stratified Theta Measure Separation

Status: `PASS_V2_2_THETA_MEASURE_SEPARATION_DIAGNOSTIC`

| Partition | rows | roots | ESS fraction | max weight | negative-mode fraction | theta mean[0] | affine latent mean max | affine covariance offdiag max | log-ratio min | log-ratio max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 232 | 108 | 0.950551 | 0.007095 | 0.533571 | 0.330821 | 0.000000 | 0.000000 | -37.120408 | -31.211405 |
| validation | 12 | 8 | 0.983826 | 0.101793 | 0.491953 | 1.181110 | 0.537741 | 0.733591 | -35.627232 | -33.467838 |
| audit | 12 | 6 | 0.996452 | 0.091052 | 0.491717 | -1.422781 | 0.399638 | 0.579953 | -34.682687 | -33.718001 |

Train affine oracle: mean max `4.510e-17`, covariance residual max `1.776e-15`; roots are disjoint across partitions.

This is an explanatory diagnostic. It does not establish IID Gaussian whitening, posterior correctness, or exhaustive mode discovery.
