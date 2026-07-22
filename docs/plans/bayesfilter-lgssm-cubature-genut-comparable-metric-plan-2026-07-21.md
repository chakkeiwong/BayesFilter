# Comparable Cubature/GenUT LGSSM Metric Run

Date: 2026-07-21

## Research Intent

Question: can the float32/TF32 Cubature and Gaussian-GenUT staged Contract E
diagnostics be reported on the same metric as the previous 16-seed Contract E
LGSSM particle-bias runs?

Candidate: the existing staged route

\[
\text{transition} \to \text{positive Sinkhorn OT} \to
\text{barycentric cloud} \to \text{residual injection} \to
\text{Cholesky restoration}.
\]

This is an LGSSM feasibility diagnostic for future high-dimensional nonlinear
filtering. It is not an LGSSM estimation claim and is not a NAWM experiment.

## Evidence Contract

| Item | Contract |
|---|---|
| Target | Dataset seed `81100`, canonical 3-dimensional observation matrix, parameter vector `(0.72, 0.55, 0.35, 0.35, 0.45)` |
| Comparator | Exact Kalman value and five physical score coordinates for the same observations; score is transformed to HMC coordinates by `(1-phi1^2, 1-phi2^2, 1-phi3^2, q_scale, r_scale)` |
| Candidate | Cubature and Gaussian-GenUT residual designs, `N=1008`, 16 particle seeds `82220..82235`, horizons `T=2,10,50` |
| Primary metric | Six per-seed relative errors: value and five HMC-score coordinates; mean, sample SD, SE, and prior simultaneous critical value `3.036283222821165` |
| Hard vetoes | crash, non-finite tensors, non-replayable result, invalid reset moments, or Sinkhorn row/column residual at least `5e-4` |
| Secondary diagnostics | raw physical score, raw score L2, finite-difference directional score, wall time, allocator bytes |
| Nonclaims | no exact filtering validity, no superiority/ranking, no nonlinear-model or NAWM conclusion, no default-readiness claim |
| Artifact | fresh versioned directory under `docs/benchmarks/artifacts/` with `result.json` and `result.md` |

## Skeptical Audit Before Execution

- The old one-seed scalar-observation target would not be a fair comparison;
  this run uses the prior dataset and full 3-dimensional observation matrix.
- The six-coordinate metric is not raw score L2. HMC scaling and per-coordinate
  Kalman normalization are explicit in the result JSON.
- A 16-seed interval is descriptive evidence for this diagnostic. It cannot
  rank Cubature against GenUT, which are identical for Gaussian moments
  `s=0, k=3` in this implementation.
- The candidate remains float32/TF32 and the XLA escape hatch is explicit if
  required; no-JIT execution would not be XLA-readiness evidence.
- The Kalman target is computed in the same float32 wrapper for this run; the
  artifact records this precision. This is a comparability diagnostic, not a
  replacement for the prior float64 oracle artifact.

## Stop Conditions

Stop as invalid on any hard veto. If all invariants pass but Kalman relative
errors are large, classify that as candidate diagnostic failure, not as a
failure of the cubature research direction. Preserve all per-seed rows.

## Command

```bash
python docs/benchmarks/run_lgssm_cubature_genut_fp32.py \
  --no-jit-compile \
  --output-root docs/benchmarks/artifacts/lgssm_cubature_genut_comparable_metric_20260721_attempt1
```
