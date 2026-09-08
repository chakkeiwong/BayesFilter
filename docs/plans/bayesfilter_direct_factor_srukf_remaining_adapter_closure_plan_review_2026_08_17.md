# Review of the Direct-Factor SR-UKF Remaining-Adapter Closure Plan

Date: 2026-08-17  
Reviewed path: `docs/plans/bayesfilter_direct_factor_srukf_remaining_adapter_closure_plan_2026_08_17.md`  
Verdict: **PROCEED WITH THE INCORPORATED CONTROLS**

## Review questions and findings

### 1. Is the linear parity claim mathematically sound?

Yes, after fixing the exact parameter maps.  An affine transformation of DZ5
sigma points reproduces the first two moments, so the Common V2 and
LGSSM-EXACT likelihood recursions should match Kalman filtering apart from
floating-point/gauge effects.  The plan correctly requires an independent
linear-Gaussian authority and distinguishes likelihood from posterior for
LGSSM-EXACT.

Control incorporated: compare the 18-dimensional likelihood first, add the
same prior only afterward, and require full exact-authority score parity rather
than relying only on finite differences.

### 2. Is a wrapped final innovation sufficient for range/bearing?

No.  The initial idea of adding only an innovation callback would leave the
predicted bearing mean and observation covariance Euclidean and therefore
wrong when sigma points straddle the angular cut.  The plan now requires all
three manifold-aware operations: circular bearing mean, wrapped sigma-point
deviations, and wrapped observed innovation.

Control incorporated: add paired value/derivative callbacks, circular-
resultant validation, angular-cut telemetry, and fail-closed score behavior.

### 3. Is the range/bearing analytical derivative globally defined?

No.  `wrap` is discontinuous at odd multiples of `pi`, and the circular mean
is undefined when its resultant vanishes.  Away from those sets, the wrapped
residual derivative equals ordinary subtraction and the `atan2` derivative in
the plan is correct.

Control incorporated: the claim is explicitly fixed-branch only.  Tests must
force rejection near the cut and near a zero resultant; no smoothing is
silently introduced because that would change the statistical program.

### 4. Does the predator-prey adapter risk reusing the wrong fixture?

Yes.  The prior PP-UKF registry row uses a longer frozen dataset and probit
coordinates, while Common V2 uses `final_time=3`, seed `4404`, and physical
coordinate `r`.

Control incorporated: reconstruct the Common V2 source directly, assert the
three-observation horizon and source checksum, select only row zero of the
six-parameter RK4 Jacobian, and forbid PP-UKF artifacts as parity authority.

### 5. Does one-time Cholesky violate the direct-factor requirement?

No.  The fixed/parameter-dependent covariance contract is converted to a
factor at adapter construction.  The temporal recursion consumes only factors
and uses stack/block QR.  This is materially different from reconstructing and
decomposing a covariance at every time step.

Control incorporated: a source guard covers `_one_step` and the QR temporal
helpers.  Artifacts must name the one-time adapter boundary.

### 6. Are singular and ill-conditioned cases overclaimed?

The plan correctly limits these four adapters to positive-factor, fixed-rank
score branches.  It does not supersede the rectangular/SVD value-only route
for singular or rank-changing covariances.  A merely small but positive pivot
is recorded; an absent/non-positive pivot fails the row.

Control incorporated: no jitter repair or SVD fallback is allowed inside the
claimed direct-factor score route.  A failure remains evidence, not a reason
to relabel the model.

### 7. Is the testing scope adequate?

Yes after adding two details: finite-difference step halving and explicit
coverage of the four LGSSM-EXACT parameter families.  These detect a common
false positive where one step size agrees accidentally or entire derivative
tensor blocks are transposed/zero.

Execution audit addendum: the first closure harness summary did not
mechanically apply every threshold.  The final `r3` harness does.  For a
centered finite difference the gated estimate is the finer `h=5e-6` value
after the `h=1e-5` to `h=5e-6` comparison demonstrates convergence.  This
interpretation is mathematically preferable to rejecting a derivative because
the deliberately coarser truncation-error diagnostic is marginally above the
ceiling.  Both estimates remain immutable evidence.

## Execution decision

The plan is mathematically coherent, bounded, and preserves the correct
scientific boundaries.  Proceed in this order:

1. API callbacks and backward-compatibility tests;
2. generic/frozen adapters and unit tests;
3. per-model integration tests;
4. versioned campaign execution; and
5. inventory, result memo, and LaTeX updates.

Do not promote any row whose branch, pivot, exact-authority, finite-difference,
or XLA gate fails.
