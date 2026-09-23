# Proposed initializer clipping-count comparison

Status: proposal only; owner agreement pending. No runtime correction or
comparator has been installed. This proposal concerns a reporting count at the
zero coefficient boundary; it does not extend the ill-conditioned precision or
objective-resolution exceptions.

The complete one-residual-correction candidate matches all D1 original records,
events and calls on CPU02775 and GPU03252. D3 differs only in raw
`mu_clipped_count` fields and their copied events. All other numerical fields
pass the existing1e-10 absolute/relative comparison, all actual decisions and
target calls match, and all discrete fields other than those reporting counts
match. Independent100/70-digit references03248--03251 establish condition2.4994
and show that original NumPy frequently puts the essentially zero coefficient
on the wrong side of zero. See the [coefficient diagnosis](filter_gradient_initializer_coefficient_result_20260923.md).

Keep the raw clipping count and the actual clipped coefficients unchanged in
meaning. Add observational coefficient-resolution information, and allow a
before/after *reporting-count* difference only under all of these conditions:

1. Both recorded counts exactly equal the counts recomputed from their own raw
   coefficients and existing lower/upper clipping limits. No fabricated or
   inconsistent count is accepted.
2. Both least-squares designs retain full column rank and pass the existing
   design-resolution condition. Let m be training rows times dimension, p the
   number of coefficients, kappa the retained design condition, and c the vector
   containing raw_lambda0 and raw_mu. Define the reported roundoff scale
   `rho = eps * max(m,p) * kappa * ||c||_2`. This uses the existing
   design-roundoff indicator times coefficient magnitude. It is a diagnostic
   resolution scale, not a certified interval or a change to coefficient values.
3. Every coefficient whose lower-clipping side differs must satisfy
   `abs(raw_mu) <= rho` in **both** records. Upper-clipping decisions remain
   exact. Nonfinite values and rank/condition failures remain ineligible.
4. Every other field retains its existing comparison, including raw and clipped
   coefficients, precision/covariance, center selection, acceptance, refinement,
   statuses, event order and target rows/call order. The only new comparison
   exception is the diagnostic count explained by those checked lower-boundary
   sign changes. Preserve both counts and the resolution-scale observations in
   result artifacts; do not silently normalize saved evidence.

The standard-library diagnostic checker reviews all eight complete saved
D1/D3 scalar/batch CPU/GPU candidate records, events and calls under exactly this
proposal. It explains18 count occurrences, including copied reports, and rejects
five adverse changes: a resolved sign change whose coefficient difference is
still below the general1e-10 tolerance, rank loss, a forged count, upper clipping,
and a nonfinite coefficient. Receipt:
`artifacts/filter-gradient-repair-20260917/initializer-clip-reporting-proposal-review-03252.json`.
The exact checker hash is recorded there; its preserved source is
`artifacts/filter-gradient-repair-20260917/initializer-clip-reporting-proposal-checker-03252.py`.

If agreed, implement one residual correction with the same rank-selected QR/SVD
factors and add the observational fields. Renew complete geometry/initializer,
boundary, derivative, consumer and CPU/GPU tests at the existing tolerances,
then public integration and matched costs. A different healthy decision, target
order/count or unresolved numerical discrepancy still rejects the candidate.
No public initializer promotion follows merely from agreement with this plan.

Review: requiring a LAPACK rounding sign here would preserve an inaccurate
reporting accident. Silently dropping the count would hide the difference. The
proposed rule retains the raw evidence, checks each cause and preserves all
actual numerical/selection criteria. Its weak point is that rho is a diagnostic
scale, not a rigorous error certificate; it must never authorize accepting a
different numerical result or an uncertain scientific decision. This limited
change therefore requires explicit agreement under E2 before adoption.
