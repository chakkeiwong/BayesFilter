# Float32/TF32 LGSSM Cubature and GenUT Experiment

Date: 2026-07-20

## Research Intent

Main question: can the staged Contract E reset use a Cubature or GenUT
residual design in a TensorFlow float32/TF32 LGSSM diagnostic while producing
finite value and score artifacts for \(T=2,10,50\)?

The LGSSM is a controlled harness for future high-dimensional nonlinear
filtering. It is not the scientific endpoint, an LGSSM-estimation target, or
NAWM implementation.

Candidate mechanism:

- \texttt{cubature}: replicated \(2d\) spherical-radial directions;
- \texttt{genut}: replicated \(2d+1\) GenUT points using the known Gaussian
  standardized moments \(s_a=0,\ k_a=3\) for this LGSSM diagnostic.

The route remains staged:

\[
\text{particle transition and current increment}
\to\text{positive Sinkhorn OT}
\to\text{barycentric cloud}
\to\text{residual injection}
\to\text{Cholesky restoration}.
\]

## Evidence Contract

| Item | Contract |
|---|---|
| Baseline | Exact Kalman value and score for the same observations and parameter |
| Candidate | Staged particle route with Cubature or GenUT residual |
| Primary outputs | finite value, finite score, value error, score error, reset mean/covariance residual |
| Hard vetoes | crash, non-finite tensors, invalid positive-definite factor, invalid Sinkhorn normalization, wrong shape, replay mismatch |
| Descriptive only | observed value/score error and runtime at one seed |
| No conclusion | no exact-filtering validity, no superiority, no nonlinear-model or NAWM claim |
| Artifact | \texttt{docs/benchmarks/artifacts/lgssm_cubature_genut_fp32_20260720/result.json} and \texttt{result.md} |

## Fixed Configuration

- State dimension: \(d=3\).
- Particle count: \(N=1008\).
- \(N=1008=6\cdot168=7\cdot144\), so both designs have exact equal-weight
  replication.
- Horizons: \(T=2,10,50\).
- Parameter vector:
  \[
  \theta=(\phi_1,\phi_2,\phi_3,q_{\rm scale},r_{\rm scale})
  =(0.72,0.55,0.35,0.35,0.45).
  \]
- Observation matrix: \(H=(1,1,1)/\sqrt3\).
- Fixed observations generated from a deterministic seed.
- Float32 tensors only.
- TensorFlow TF32 execution enabled.
- XLA JIT is enabled for the compiled route.
- Sinkhorn epsilon and iteration count are fixed before execution.

## Skeptical Audit

Potential misleading outcomes:

- The particle likelihood is not an unbiased log-likelihood estimator merely
  because the Kalman comparator is exact.
- The reset covariance identity is a finite ridged identity, not exact-filter
  covariance equality.
- The GenUT Gaussian moment choice tests the implementation and staged
  mechanics, not non-Gaussian GenUT advantage.
- A one-seed result is descriptive and cannot rank the methods statistically.
- TF32 affects matrix products; value/score errors include this execution mode.

Cheap diagnostics:

- exact row/column Sinkhorn marginal residuals;
- finite-value and finite-score checks;
- bitwise replay of the compiled call;
- reset mean and covariance residuals;
- finite-difference directional score check on the same finite scalar.

## Stop Conditions

Stop and classify the run as invalid if any hard veto fires. If a candidate
fails only the Kalman comparison while the harness invariants pass, classify
the candidate as a failed diagnostic arm, not as a failure of the staged
research direction.
