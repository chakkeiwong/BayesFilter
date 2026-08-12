# Squared-TT retained-prefix derivation certificate

- **Date:** 2026-08-09
- **Purpose:** explicit derivation certificate for the retained-first squared-TT branch in the standalone rewritten monograph.
- **Scope:** only the squared-TT retained-coordinate branch and its derivative certificate.

## Certificate statement

The retained-first convention is the intended branch convention for the active rewrite. The certificate therefore states the retained current block as the first block and uses right-side contractions that leave that block explicit.

For the scalar case, the certificate already states the special two-coordinate retained-prefix formula. For the genuine vector-retained case, the certificate now explicitly shows the retained prefix and suffix contraction pattern and its derivative recursion.

## Exact retained-prefix form

For retained dimension \(m>1\) and adjacent-state order \(D=2m\), the current block is

\[
  z_{\rm cur}=z_{1:m},
\]

and the trailing block is

\[
  z_{\rm prev}=z_{m+1:D}.
\]

The retained prefix is

\[
  G_m(z_{1:m}) = H_1(z_1)\cdots H_m(z_m),
\]

and the suffix contraction runs only over the previous-state block:

\[
  M_{>D}=1,
  \qquad
  M_{>j-1}[a,a']
  =
  \sum_{b,b',\ell,\ell'}
  C_j[a,\ell,b]
  C_j[a',\ell',b']
  B_j[\ell,\ell']
  M_{>j}[b,b'],
  \qquad j=D,
\ldots,m+1.
\]

The retained numerator is therefore

\[
  a_t(z_{1:m};\beta)
  =
  e^{-c_t}
  \{G_m(z_{1:m})M_{>m}G_m(z_{1:m})^\top
    +\tau_t\lambda_{t,\mathrm{ret}}(z_{1:m})\},
\]

with

\[
  \lambda_{t,\mathrm{ret}}(z_{1:m})
  =\int \lambda_t(z_{1:m},z_{m+1:D})\,dz_{m+1:D}.
\]

Its derivative is obtained by differentiating the retained prefix and suffix
recursion on the same frozen branch:

\[
  \dot G_m=\sum_{k=1}^{m}H_1\cdots H_{k-1}\dot H_kH_{k+1}\cdots H_m,
\]

with the dotted suffix recursion obtained by differentiating each product-rule
factor once.

## What the certificate proves

The certificate is valid only if it shows both:

1. the retained-first right-contraction formula equals direct integration of the
   same concrete branch; and
2. the dotted retained recursion equals direct differentiation of the same
   branch.

For the release note, this certificate must be attached as a scalar/vector
identity check, not merely described in prose.

## Relation to Chapter 37

Chapter 37 must remain consistent with this same retained/reference/Jacobian
ownership story:
- the stored evaluator consumes retained reference coordinates,
- physical fitting points are converted before query,
- the Jacobian is owned by the target-construction path,
- the route-audit note remains a convention boundary, not a theorem.

## Nonclaim

This note does not certify the whole monograph. It is only the final explicit
certificate statement needed for the squared-TT release blocker.
