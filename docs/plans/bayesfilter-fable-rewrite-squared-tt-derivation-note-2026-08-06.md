# Derivation note: retained-first squared-TT contract on the standalone rewrite branch

- **Date:** 2026-08-06
- **Location:** `docs/plans/`
- **Purpose:** document the retained-first squared-TT derivation convention used to repair the latest blocker in the standalone rewrite branch.
- **Status:** working derivation note for the rewrite branch; not a canonical-source publication artifact.

---

## 1. The convention being fixed

The repaired branch uses the concrete adjacent-state order

\[
  r_t=(x_t,x_{t-1})
\]

for the scalar case and the analogous concatenated order for vector states. In this branch, the **retained current state is the first block**, not the last coordinate.

That means the retained object must be built by **right-side contractions** that integrate out the trailing previous-state block.

---

## 2. Retained-object recursion

Let the retained block be denoted by \(z_{\rm cur}\) and the integrated block by \(z_{\rm prev}\). The relevant contraction recursion is:

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
  \qquad j=D,\ldots,2.
\]

The retained numerator is then the explicit first-block contraction:

\[
  a_t(z_{\rm cur};\beta)
  =
  e^{-c_t}
  \sum_{b,b',\ell,\ell'}
  M_{>1}[b,b']
  C_1[1,\ell,b]
  C_1[1,\ell',b']
  b_1(z_{\rm cur})[\ell]
  b_1(z_{\rm cur})[\ell']
  +
  e^{-c_t}\tau_t
  \int \lambda_t(z_{\rm cur},z_{\rm prev})\,dz_{\rm prev}.
\]

This form is consistent with the stated branch target

\[
  \overline q_t = e^{-c_t}(\phi_t^2 + \tau_t\lambda_t),
\]

so the defensive term carries the same represented-density scale as the squared contraction.

---

## 3. Derivative recursion on the same branch

The dotted recursion follows the same right-side contraction pattern:

\[
  \dot M_{>j-1}[a,a']
  =
  \sum_{b,b',\ell,\ell'}
  \dot C_j[a,\ell,b]
  C_j[a',\ell',b']
  B_j[\ell,\ell']
  M_{>j}[b,b']
  +
  \sum_{b,b',\ell,\ell'}
  C_j[a,\ell,b]
  \dot C_j[a',\ell',b']
  B_j[\ell,\ell']
  M_{>j}[b,b']
  +
  \sum_{b,b',\ell,\ell'}
  C_j[a,\ell,b]
  C_j[a',\ell',b']
  B_j[\ell,\ell']
  \dot M_{>j}[b,b'].
\]

Starting from \(\dot M_{>D}=0\), the retained numerator derivative is:

\[
  \dot a_t(z_{\rm cur};\beta)
  =
  e^{-c_t}
  \sum_{b,b',\ell,\ell'}
  \dot M_{>1}[b,b']
  C_1[1,\ell,b]
  C_1[1,\ell',b']
  b_1(z_{\rm cur})[\ell]
  b_1(z_{\rm cur})[\ell']
  +
  e^{-c_t}
  \sum_{b,b',\ell,\ell'}
  M_{>1}[b,b']
  \dot C_1[1,\ell,b]
  C_1[1,\ell',b']
  b_1(z_{\rm cur})[\ell]
  b_1(z_{\rm cur})[\ell']
  +
  e^{-c_t}
  \sum_{b,b',\ell,\ell'}
  M_{>1}[b,b']
  C_1[1,\ell,b]
  \dot C_1[1,\ell',b']
  b_1(z_{\rm cur})[\ell]
  b_1(z_{\rm cur})[\ell']
  +
  e^{-c_t}\tau_t
  \int \dot\lambda_t(z_{\rm cur},z_{\rm prev})\,dz_{\rm prev}.
\]

---

## 4. Why this fixes the earlier blocker

The earlier inconsistent draft mixed a retained-first narrative with a left-contraction algebra that actually left the last coordinate explicit. The corrected branch uses the same retained-first language and a right-contraction algebra that leaves the first block explicit.

That matters because the release note must not claim a retained-coordinate convention while implementing the opposite one.

---

## 5. Verification target

A future bounded check should certify:

1. the scalar two-coordinate case matches the displayed contraction exactly;
2. the vector-case contraction reduces to the same pattern by block concatenation;
3. the derivative recursion follows by a single product-rule application at each frozen core;
4. the stored evaluator and next-step query rule in Chapter 37 use the same coordinate convention.

---

## 6. Nonclaim

This note does **not** certify the whole monograph. It exists only to document the repaired squared-TT branch convention so the final rewrite can be audited consistently.
