# Documentation-agent handoff: amend the squared-TT certificate request

- **Date:** 2026-08-08
- **Audience:** the documentation agent maintaining the standalone Fable rewrite
- **Supersedes only:** the scalar-only adequacy implication in
  `bayesfilter-fable-rewrite-squared-tt-handoff-request-2026-08-07.md`
- **Does not authorize:** unrelated chapter rewrites or global monograph changes

## Amendment required

The original handoff correctly requests a retained-first/right-contraction
certificate, but its required two-coordinate example is insufficient for the
vector-state claim. In the current Chapter 36b formula, the right recursion runs
through `j=2` and leaves only `H_1` explicit. That is correct for retained
dimension `m=1`. For a vector current state occupying coordinates `1:m`, it
would integrate out retained coordinates `2,...,m` and is therefore wrong
relative to the stated vector retained-block target.

Documentation must use the block formula

\[
  G_m(z_{1:m})=H_1(z_1)\cdots H_m(z_m),
  \qquad
  M_{>m}=\int H_{m+1}\cdots H_DH_D^\top\cdots H_{m+1}^\top
  \,\mathrm d\mu_{m+1:D},
\]

and retain

\[
  a_t(z_{1:m})=
  e^{-c_t}G_mM_{>m}G_m^\top
  +e^{-c_t}\tau_t\lambda_{t,\mathrm{ret}}(z_{1:m}).
\]

The right recursion must run only for `j=D,...,m+1`. The dotted formula must
differentiate both the retained prefix and the right mass contraction.

## Revised certificate requirement

The final documentation certificate must include both:

1. `m=1,D=2`, showing the scalar formula as a specialization; and
2. `m=2,D=4`, showing that the full retained prefix remains explicit.

Both cases must compare direct integration with right contraction and direct
differentiation with the dotted recursion. The branch must explicitly freeze or
differentiate the bases, mass matrices, coordinate map/domain, `c_t`, `tau_t`,
and defensive density. The current planned certificate freezes them and varies
only the TT cores.

## Chapter 37 amendment

For `m>1`, derive the saved quadratic evaluator from a retained-prefix
coefficient matrix `A_t`, so that the square contribution is
`Q_t^{sq}=A_tM_{>m}A_t^T`. Include the fixed defensive retained marginal in the
stored evaluator rather than silently dropping it. Preserve the existing
reference-coordinate query rule and next-step Jacobian ownership.

## Closure rule

Do not close the audit-ledger blocker from a scalar certificate alone. Close it
only after both exact certificate cases pass, every changed derivation has been
audited with MathDevMCP CLI with no reported mismatch, and the monograph builds.
MathDevMCP abstention remains diagnostic and is not a proof or a refutation.
