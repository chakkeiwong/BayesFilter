# Differentiable Singular and Ill-Conditioned SR-UKF Gap-Closure Plan

Date: 2026-08-18  
Scope: BayesFilter direct-factor SR-UKF; SSL-LSTM remains excluded.  
Canonical route: `direct_qr_block_conditional`.

## 1. Objective and release boundary

Close the remaining numerical and mathematical gaps between the current
full-rank direct block-QR SR-UKF and a differentiable filter for fixed-rank
singular covariance supports. The release target is deliberately local:

1. full-rank, badly conditioned factors have a scale-aware pivot admission
   policy and a temporal value/score test matrix;
2. fixed-rank singular state and innovation supports have a rectangular
   fixed-pivot QR value/score route;
3. unknown or changing rank remains a value-only rank-discovery route;
4. the `P_epsilon = G G' + epsilon I` limit is documented as a renormalized
   affine-support likelihood, not as a finite ambient Gaussian limit; and
5. every score-bearing claim is restricted to a fixed rank, pivot chart, sign
   convention, support, and observation branch.

This plan does not claim a differentiable score across rank changes, support
changes, repeated singular-vector gauges, angular cuts, or signed-weight PSD
boundaries. It does not silently add a nugget to the user's model.

## 2. Current gaps

| ID | Gap | Required closure |
|---|---|---|
| G1 | Default temporal QR passes `relative_pivot_tolerance=0.0` | Bind a positive, scale-aware threshold to the route and expose realized pivot diagnostics |
| G2 | Existing ill-conditioning evidence is mostly primitive and includes a uniform scale sweep | Add anisotropic temporal families, successful fixed-branch scores, and deterministic rejection tests |
| G3 | `TFFactorSRUKFModel` requires square factors | Add a rectangular fixed-rank model/derivative contract |
| G4 | Rectangular temporal route uses per-step direct-stack SVD and is value-only | Add a fixed-chart QR score route after rank/pivot preflight |
| G5 | Singular support likelihood has value but no derivative | Add QR support log likelihood and first derivative |
| G6 | Singular conditional update has value but no derivative | Add fixed-support gain, posterior rectangular-factor, mean, and derivative propagation |
| G7 | Sigma points are padded to ambient state dimension in singular route | Generate points in retained latent rank coordinates and preserve declared factor gauge |
| G8 | Rank/support/chart identity is not frozen across a claim run | Add a preflight object and fail closed when runtime telemetry differs |
| G9 | No temporal singular score authority | Add affine fixed-rank exact authorities and centered finite-difference tests |
| G10 | Canonical chapters do not describe the current block-QR and epsilon-limit contracts | Update Chapters 17, 18, 12, 14, and 23; keep historical spectral material explicitly historical |

## 3. Mathematics

### 3.1 Full-rank direct QR

For a direct residual/loading stack `A`, factor `A' = Q R` with positive
diagonal. The lower factor is `L=R'` and `L L'=A A'`. For the joint update,

```
 A = [Y; X],  A' = Q [[Ryy, Ryx], [0, Rxx]],
 Ly=Ryy', Lxy=Ryx', Lf=Rxx'.
```

Then `S=Ly Ly'`, `K=Lxy Ly^{-1}`, and `Pf=Lf Lf'`. The temporal runtime never
forms any of these covariance products.

### 3.2 Fixed-rank rectangular chart

Let `A` have rank `r <= n` and `B=A'`. A rank-revealing preflight chooses a
fixed permutation `Pi` with `B Pi=[B1 B2]`, `B1` full column rank. During the
score run compute only

```
 B1 = Q R11,       R12 = Q' B2,
 E2 = (I-Q Q') B2,
 G = Pi [R11 R12]' .
```

Require `||E2|| <= tau_chart` and all retained diagonal pivots above
`tau_pivot * max(1, ||B||_F)`. Then `G G'=A A'` on the fixed chart. The QR
derivative is obtained by differentiating the full-column-rank `B1`; `R12`
and `E2` use ordinary product rules. A chart, pivot, or rank change invalidates
the score and is reported as a branch event.

### 3.3 Singular epsilon limit

For `P_e=G G' + e I`, rank `r`, and innovation decomposition
`e_obs=e_parallel+e_perp`, the ambient log density is

```
ell_e = -1/2 [ n log(2 pi) + sum log(lambda_i+e)
               + (n-r) log e + sum e_i^2/(lambda_i+e)
               + ||e_perp||^2/e ].
```

Off support it tends to `-infinity`. On support it diverges by
`-(n-r)/2 log(2 pi e)`. The finite support value is

```
ell_support = lim_{e->0} [ell_e + (n-r)/2 log(2 pi e)]
             = -1/2 [ r log(2 pi) + log det^+(P)
                       + e_obs' P^+ e_obs ].
```

With a fixed thin QR `G=U R`, compute `z=R^{-1}U'e_obs` and

```
ell_support = -1/2 [ r log(2 pi) + 2 sum log diag(R) + z'z ].
```

Its first derivative is `-sum diag(Rdot)/diag(R) - z' zdot`, with `zdot`
from the differentiated triangular solve. This is the differentiable
fixed-support target; it is not the ambient-density derivative.

### 3.4 Singular conditional update

For observation stack `Y=U R V'` on a fixed support chart, let `e` be the
innovation and `z=R^{-1}U'e`. The observed state coordinates are `X V`, and
the residual state stack is `X(I-VV')`. The gain is `X V R^{-1} U'` and the
posterior factor is the fixed-chart rectangular QR factor of the residual
stack. Differentiate these solves and projections while `r`, `U/V` chart,
and support identity remain fixed.

## 4. Implementation phases

### Phase A: route identity and pivot policy

1. Add repository-owned constants for absolute/relative pivot and chart
   tolerances, with float64 defaults and explicit metadata.
2. Thread the relative pivot policy through prediction and block conditional
   QR. Do not set a nonzero threshold inside an XLA graph.
3. Add fail-closed diagnostics for active pivot threshold, chart residual,
   rank, support residual, and branch identity.

### Phase B: fixed-rank rectangular QR score primitives

1. Extend `rectangular_factor_tf.py` with fixed-chart QR preflight and a
   score-bearing `batched_fixed_support_qr_likelihood`.
2. Add derivative outputs for factor, support likelihood, support projection,
   gain, and conditional factor.
3. Keep direct-stack SVD only for rank discovery, diagnostics, and value-only
   fallback. Never use it as a hidden score fallback.

### Phase C: temporal rectangular SR-UKF

1. Add `TFRectangularSRUKFDerivatives` and a fixed-chart result contract.
2. Generate sigma points in retained latent dimensions.
3. Run prediction and conditional update through fixed rectangular QR.
4. Carry mean/factor derivatives through `tf.while_loop` with fixed static
   shapes and XLA support.
5. Return `score_valid`, `branch_status`, `rank`, `pivot`, and support telemetry.

### Phase D: documentation

Update the canonical `docs/main.tex` chapters:

- Chapter 17: current direct block-QR algorithm, conditional identity, pivot
  policy, signed-weight boundary, and full-rank score contract;
- Chapter 18: historical spectral route and its gauge/equal-eigenvalue risks;
- Chapter 12: rectangular QR factor derivatives and support likelihood score;
- Chapter 14: fixed-chart finite differences, branch-crossing vetoes, and
  epsilon-limit tests;
- Chapter 23: HMC boundary semantics and invalid-score handling.

Add a release evidence table referencing the versioned JSON artifacts and the
fact that singular score evidence is fixed-support only.

### Phase E: verification campaign

Unit tests:

- fixed-pivot rectangular QR reconstruction and derivative;
- support likelihood and derivative against the renormalized epsilon limit;
- conditional gain/factor derivative;
- repeated singular values with fixed QR chart;
- rank-zero/rank-one value-only branch;
- NaN/Inf, malformed chart, negative pivot, duplicate permutation.

Integration tests:

- affine full-rank temporal score against exact Kalman/SVD authority;
- ill-conditioned scales with condition numbers `1e4` through `1e28`;
- fixed-rank singular temporal score against support-coordinate reference;
- on/off support observations;
- deliberate rank, pivot, support, and chart crossings must invalidate score;
- batch permutation, eager/XLA parity, and GPU preference `3,2,1,0`.

Regression tests:

- current block QR equals archived QR/downdate values and scores;
- historical principal-root comparison remains diagnostic only;
- route guard forbids spectral decomposition in admitted temporal files;
- no accidental NumPy in runtime modules.

## 5. Review checklist

Before implementation is considered complete, review this plan for:

1. consistency of all dimensions and transpose orientations;
2. correct measure-theoretic treatment of the epsilon limit;
3. no hidden covariance construction or spectral fallback;
4. fixed-chart derivative validity and explicit branch failures;
5. signed-weight and negative-pivot boundaries;
6. XLA/static-shape compatibility;
7. independent exact authorities and finite-difference coverage; and
8. documentation matching code and artifact provenance.

## 6. Stop conditions and nonclaims

Stop score admission when rank, support, pivot pattern, chart residual, sign
convention, or angular branch changes. Do not silently add jitter or report a
value-only rank-discovery result as a differentiable score. This plan does not
establish exact nonlinear Bayesian inference, HMC readiness, GPU production
readiness, or universal model applicability.
