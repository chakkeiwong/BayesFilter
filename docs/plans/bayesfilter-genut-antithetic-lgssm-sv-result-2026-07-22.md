# GenUT Antithetic LGSSM And SV Result

Date: 2026-07-22

Status: `ANTITHETIC_PARTIAL_COORDINATE_NOMINATION_FEASIBILITY_ONLY`

The primary comparator is an equal-cost average of two independent complete
GenUT runs. The single-cloud arm is descriptive only.

## LGSSM

Selected controls: `{'epsilon': 2.0, 'sinkhorn_steps': 4, 'ridge': 1e-06}`.

| Coordinate | Geometric variance ratio | Familywise 95% log-ratio CI | Datasets lower | Nominated |
|---|---:|---:|---:|---|
| value | 0.7298 | [-0.6909, 0.0609] | 7/8 | False |
| phi1 | 0.8034 | [-0.7946, 0.3569] | 4/8 | False |
| phi2 | 0.9237 | [-0.8104, 0.6518] | 4/8 | False |
| phi3 | 0.9345 | [-0.7344, 0.5989] | 5/8 | False |
| q_scale | 1.2851 | [-0.0614, 0.5631] | 1/8 | False |
| r_scale | 0.9641 | [-0.6655, 0.5925] | 4/8 | False |

No LGSSM coordinate passed the predeclared familywise variance screen. The
value ratio was descriptively below one on seven datasets, but its interval
crossed no change. The `q_scale` geometric variance ratio was `1.2851`, so
antithetic coupling was descriptively worse for the parameter that motivated
the original LGSSM investigation. No LGSSM ranking is statistically supported.

## SV

Selected controls: `{'epsilon': 4.0, 'sinkhorn_steps': 4, 'ridge': 1e-06}`.

| Coordinate | Geometric variance ratio | Familywise 95% log-ratio CI | Datasets lower | Nominated |
|---|---:|---:|---:|---|
| value | 0.4916 | [-1.1730, -0.2472] | 7/8 | True |
| theta_gamma | 0.6769 | [-0.8681, 0.0876] | 6/8 | False |
| theta_log_beta | 0.7091 | [-0.7321, 0.0444] | 7/8 | False |

Only the SV value coordinate passed the predeclared familywise t-interval
screen. Its geometric conditional-variance ratio was `0.4916`, and its
geometric MSE ratio was `0.4829` with familywise log-ratio interval
`[-1.2612,-0.1947]`. The exact sign test saw lower variance on seven of eight
datasets but had two-sided `p=0.0703`; thus the magnitude-based t screen passes
while distribution-free directional evidence remains weak at this dataset
count.

Neither SV score coordinate passed. `theta_log_beta` also had a nonzero signed
mean-error shift of `+0.02758`, pointwise 95% interval
`[+0.00128,+0.05388]`. Its MSE ratio remained descriptively below one, but the
familywise MSE interval crossed no change. This is a bias-warning diagnostic,
not evidence that antithetic score estimation is better.

## Decision

Antithetic averaging reduced conditional variance only for a subset of coordinates under the equal-cost screen. It remains an optional experimental coupling.

This feasibility campaign does not change the default. Dataset-level mean
error and MSE diagnostics are retained in `result.json`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| LGSSM antithetic GenUT | Reject nomination: 0/6 coordinates passed | Engineering, DGP, oracle, recursive-score, GPU/XLA, and residual vetoes passed | Eight outer datasets; value near the t boundary; `q_scale` descriptively worsened | Retain standard coupling for LGSSM | No proof antithetic can never help another LGSSM scope |
| SV value antithetic GenUT | Feasibility nomination: familywise t interval passed | All vetoes passed | Only eight datasets and exact sign `p=0.0703` | Replicate on at least 24 fresh DGP datasets before considering an optional policy | No default promotion or unbiasedness |
| SV score antithetic GenUT | Reject nomination: 0/2 score coordinates passed | All vetoes passed | Wide score intervals and a `theta_log_beta` signed-error shift | Keep standard score coupling; diagnose bias before another score claim | No score-variance or HMC improvement |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for attempt 2; attempt 1 was an FP32 finite-difference conditioning repair before claim execution |
| Statistically supported ranking | Limited to the predeclared magnitude-based SV value variance/MSE t screens; no LGSSM or SV-score ranking |
| Descriptive-only differences | All other variance ratios, signed error shifts except the reported pointwise beta shift, runtime, and allocator use |
| Default readiness | Failed; no default changed |
| Next evidence needed | At least 24 fresh SV DGP datasets, frozen controls, the same equal-cost comparator, exact sign evidence, and unchanged bias/MSE diagnostics |

## Engineering And Run Manifest

- FP32 tensors, TF32 enabled, TensorFlow GPU/XLA recursive-score path.
- `N=1008` for LGSSM and `N=1002` for SV; `T=50` for both.
- Eight claim DGP datasets and 16 particle-seed pairs per model.
- `128` auditable paired rows per model; each primary estimator uses two
  complete filters.
- Wall time: `144.74 s`; TensorFlow allocator peak: `101,545,216` bytes.
- Focused CPU-hidden verification before execution: `10 passed`.
- Artifact:
  `docs/benchmarks/artifacts/genut_antithetic_lgssm_sv_20260722/attempt02/`.

## Post-Run Red Team

The strongest alternative explanation for the lone SV-value pass is a small
number of outer datasets plus sensitivity of the t interval to magnitude. The
exact sign test does not independently clear 5%, and no score coordinate
passes. A 24-or-more-dataset replication that loses the SV-value interval or
shows increased absolute bias/MSE would overturn the nomination. The weakest
part of the evidence is outer-sample size, not engineering execution.
