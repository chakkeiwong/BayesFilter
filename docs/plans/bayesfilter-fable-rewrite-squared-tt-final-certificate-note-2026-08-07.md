# Final squared-TT derivation certificate note

- **Date:** 2026-08-07
- **Scope:** retained-first squared-TT branch in the standalone rewritten monograph
- **Purpose:** document the final derivation check expected for the remaining blocker family, so the branch can be promoted only if the scalar/vector certificate is attached.
- **Resolution addendum, 2026-08-08:** this historical request is superseded by `bayesfilter-fable-rewrite-squared-tt-block-certificate-result-2026-08-08.md`. The completed work strengthened the requested scalar check with a genuine `m=2,D=4` vector-retained certificate.

## What the release branch now asserts

The retained-first branch is the concrete order

\[
  r_t=(x_t,x_{t-1})
\]

for the scalar case and the analogous concatenated order for vector states. The retained current state is the first block, not the last coordinate. The retained numerator and its derivative are therefore built by the right-side contractions already written in `ch36b`.

## What the remaining certificate must show

A final release must attach a scalar/vector identity check showing that, for at least one fixed numerical example:

1. the displayed right-side contractions produce the same retained numerator as direct integration of the concrete two-coordinate branch;
2. the same right-side recursion produces the same first derivative as direct differentiation of that concrete branch;
3. the defensive term remains on the same represented-density scale as the squared contraction;
4. the query-rule and derivative rule in `ch37` use the same retained/reference convention.

## What this note is not

This note is not yet the certificate itself. It states the exact form of the remaining check that must be attached before the squared-TT lane can be called fully release-closed.

## Status

At the time of this 2026-08-07 note, the remaining issue was the absence of a final explicit numerical/scalar-vector certificate. The 2026-08-08 resolution addendum above points to the completed, strengthened certificate.
