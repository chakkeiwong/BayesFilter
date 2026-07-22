# Recursive Score With Representative-Point FD Audit Plan

Date: 2026-07-21

## Research Intent

Question: can Cubature/GenUT compute its runtime score by explicit recursive
sensitivities without autodiff, while using finite differences only at a small
fixed representative set to tune and audit that score implementation?

This remains an LGSSM feasibility diagnostic for future high-dimensional
nonlinear filtering. It is not an LGSSM estimation study and is not a NAWM
experiment.

## Correction To The Previous Run

The previous runner used `tf.GradientTape` for the candidate score. Its
finite-difference calculation was only one directional diagnostic and tuning
called the evaluator with diagnostics disabled. Therefore that run did not
provide same-value-program finite-difference score evidence and its tuning
artifact must not be described as FD score-aware tuning.

The runtime score must be a compact forward sensitivity. It carries particle,
weight, and accumulated-likelihood tangents for all five parameters through the
time recursion. This includes transition and observation arithmetic, weight
normalization, every finite Sinkhorn iteration, barycentric transport, weighted
and uniform moments, residual injection, Cholesky factors, and the triangular
solve used for restoration. No `GradientTape` or `ForwardAccumulator` is
allowed. A central finite difference of the same value-only program is retained
only as a sparse audit oracle.

## Representative-Point Design

Do not evaluate finite differences over an unbounded parameter set. Generate a
fixed set of six valid points with TensorFlow stateless uniform draws from the
declared box:

```text
phi1, phi2, phi3 in [0.25, 0.85]
q_scale, r_scale in [0.25, 0.65]
```

The point-generation seed, count, domain, and exact point values are written to
the tuning artifact. All candidates and both calibration/validation partitions
reuse exactly these points. The claim partition never affects their selection.

The central difference step is parameter-scaled:

```text
h_j(theta) = max(1e-4, 2e-3 * abs(theta_j))
```

and uses the same fixed particle noise, process noise, observations, reset
controls, dtype, and backend for the plus and minus evaluations. The recursive
score and FD audit therefore target the same finite value program.

## Tuning Loss

The primary tuning validity condition is recursive-score agreement with
same-program central FD at every selected representative point. Among valid
candidates, the selection loss combines that parity residual with the value and
five HMC-score errors against the LGSSM Kalman reference. The artifact records
both the recursive score and FD audit, exact FD steps, and per-coordinate
residuals. The Kalman comparison is available only in this controlled LGSSM
diagnostic; FD parity is the transferable implementation audit.

The first bounded six-point smoke showed that an unregularized relative-error
loss can explode when one reference HMC-score component is near zero. The
selection loss therefore uses declared denominator floors `(1.0, 0.1, 0.1,
0.1, 0.1, 0.1)` for `(value, phi1, phi2, phi3, q_scale, r_scale)`; raw errors
and reference values remain in every row. This is a numerical conditioning
repair, not a claim that the small component is accurately estimated.

For a general nonlinear model without an analytical score, run the recursive
score at every model evaluation and use FD only on the fixed representative
audit set. Recursive-versus-FD agreement proves consistency with the finite
value program at those points, not equality of that finite program to the true
model likelihood.

## Skeptical Audit

- **Leakage:** fixed points and disjoint calibration/validation/claim seeds are
  required; claim points or claim seeds cannot select controls.
- **Wrong target:** FD is a derivative of the finite value program, not proof
  that the finite program equals the model likelihood. Results must retain this
  nonclaim.
- **Omitted derivative:** a transported-cloud-only derivative is wrong. The
  recursive score must include source-cloud moments, weights, transport, reset,
  and all parameter-dependent state propagated from earlier times.
- **Step noise:** float32 subtraction can amplify FD noise; record the steps and
  use combined absolute/relative parity tolerances.
- **Selection bias:** the finite grid remains only grid-optimal. The point set
  is a fixed tuning design, not a random after-the-fact choice.
- **Comparability:** prior artifacts used autodiff candidate scores. New
  recursive-score artifacts with sparse same-path FD audits are a new scope and
  must not silently overwrite or upgrade the prior claim.

## Verification And Budget

First run primitive and tiny-filter tests comparing recursive sensitivities to
same-value FD, plus a source audit excluding TensorFlow autodiff APIs. Then run
a bounded GPU representative-point parity smoke before any tuning grid. The
runtime claim evaluates only the recursive score; the expensive ten-value FD
audit is restricted to tuning/validation points.

Hard vetoes: non-finite value/score, invalid parameter points, replay mismatch,
reset/marginal residual failure, recursive/FD parity failure, or FD step
instability. A passing diagnostic does not establish exact likelihood, exact
score, nonlinear validity, or method superiority.

## Artifacts

Implementation: `docs/benchmarks/run_lgssm_cubature_genut_fp32.py` and
`docs/benchmarks/tune_lgssm_cubature_genut_fp32.py`.

The revised tuner must emit a new schema/versioned output root containing the
representative-point values, generation metadata, FD steps, per-point losses,
selected controls, source hashes, and the explicit score-reference/nonclaims.
