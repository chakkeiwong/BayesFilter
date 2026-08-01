# Matched Three-Horizon Recursive-Score Comparison Plan

Date: 2026-07-21

## Research Intent

Question: at `T=2,10,50`, how do the Cubature/Gaussian-GenUT reset and the
original Contract E Gaussian-residual reset compare on likelihood value and
recursive score, relative to the exact Kalman oracle?

This is an LGSSM diagnostic for reset and score behavior relevant to future
high-dimensional nonlinear filtering. It is not an LGSSM estimation objective
and is not a NAWM experiment.

## Matched Arms

All arms use `N=1008`, observations from dataset seed `81100`, particle seeds
`82220..82235`, float32/TF32 GPU arithmetic, no XLA, `epsilon=2`, eight finite
Sinkhorn steps, ridge `1e-5`, identical initial/process draws, and the compact
recursive no-autodiff score.

1. `contract_e_gaussian`: original Contract E-Chol reset with a fixed
   stateless Gaussian residual design, centered and scaled by
   `sqrt(N/(N-1))` independently at every time.
2. `cubature`: repeated `2d` spherical-radial design with exact zero mean and
   identity population covariance.
3. `genut`: Gaussian GenUT (`s=0`, `k=3`), algebraically and bitwise identical
   to Cubature in this scope. It is reported as a verified alias rather than
   redundantly executed.
4. `kalman`: exact float64 analytic recursive value and score oracle.

This comparison isolates the residual-design change within the current staged
Sinkhorn plus Contract E-Chol program. It does not claim equivalence to every
historical Contract E control or random-number convention.

## Evidence Contract

| Item | Contract |
|---|---|
| Primary outputs | value and five physical/HMC score coordinates versus Kalman at each horizon |
| Uncertainty | 16-seed mean, SD, SE, and simultaneous interval using critical value `3.036283222821165` |
| Paired comparison | per-seed difference in absolute Kalman-relative error: sigma-point minus Gaussian-residual Contract E |
| Score correctness | recursive score only; no candidate FD runtime score |
| Oracle | analytic recursive TensorFlow Kalman value/score, not autodiff or FD |
| Hard vetoes | crash, non-finite output, replay failure, wrong score route, FD in runtime, reset residual >= `5e-4`, or Sinkhorn marginal residual >= `5e-4` |
| Explanatory diagnostics | runtime, GPU allocator, raw score and raw error |
| Artifact | fresh versioned JSON/Markdown under `docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/` |

## Skeptical Audit

- Historical non-Cubature artifacts are not matched in particle count,
  precision, horizon coverage, controls, or score implementation; they cannot
  answer this question directly.
- The no-reset weighted filter changes reset cadence rather than only the
  residual design, so it is not the primary comparator.
- Gaussian GenUT and Cubature must not be counted as independent stochastic
  evidence because their designs are identical in this Gaussian scope.
- Raw relative score error is unstable near a zero Kalman coordinate. Preserve
  raw and relative results, but do not rank arms from a near-zero denominator
  without inspecting the absolute score error.
- A paired absolute-error difference can support a coordinate-specific ranking
  only if its simultaneous interval excludes zero and all hard vetoes pass.
- No-JIT remains an explicit diagnostic exception; this campaign cannot support
  XLA/default readiness.

The audit passes after replacing the unmatched historical comparator with the
matched Gaussian-residual arm and replacing the FD Kalman score with an
analytic recursive oracle.

## Budget And Stops

Budget: 16 seeds x 3 horizons x 2 executed particle arms, plus one algebraic
Cubature/GenUT equality check. Expected wall time is under 20 minutes on the
visible RTX 4080 SUPER. One localized harness repair/retry is allowed within the
same scope. Stop on a hard veto or missing analytic Kalman parity.

## Nonclaims

No exact nonlinear-filtering validity, method-wide superiority, `1/N` rate,
NAWM result, HMC readiness, XLA readiness, or default promotion follows from
this campaign.
