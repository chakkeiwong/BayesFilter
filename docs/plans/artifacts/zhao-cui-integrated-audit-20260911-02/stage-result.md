# Zhao-Cui Integrated Audit: Stage 02 MathDev Verification

Date: 2026-09-11

Status: bounded proof checks complete; no runtime code changed.

## Scope

This stage checked the mathematical obligations identified in the source
inventory. MathDevMCP was used as a bounded symbolic/algebra service. Its
`doctor` report returned `ok=true`; SymPy 1.14.0 was available. Lean timed out
in the doctor probe and was not used. The high-level `derive_from` calls could
not encode prose-only claims, so they are not proof certificates.

## Machine-checked obligations

| Obligation | Encoded check | Result | Boundary |
|---|---|---|---|
| Positive scalar affine Jacobian normalization | `qu/l = qu*(1/l)`, assumption `l > 0` | `equivalent` by SymPy | Supports the scalar normalization in the physical-density change of variables. It does not prove the full multivariate conditional theorem. |
| Proposal-ratio cancellation | `q*(f*g/q) = f*g`, assumption `q != 0` | `equivalent` by SymPy | Supports the algebra used in Eq. (23) once the actual proposal density is positive. It does not establish proposal support. |
| Centered score has zero weighted mean | `w1*(h1-(w1*h1+(1-w1)*h2)) + (1-w1)*(h2-(w1*h1+(1-w1)*h2)) = 0` | `equivalent` by SymPy | Checks the two-particle centering algebra after weights sum to one. |
| Log-sum-exp weighted mean | `(e1*s1+e2*s2)/(e1+e2) = (e1/(e1+e2))*s1 + (e2/(e1+e2))*s2`, assumption `e1+e2 != 0` | `equivalent` by SymPy | Checks the scalar weighted-mean form used by the score recurrence. |

The exact service responses are compactly summarized here rather than copied
from the MCP envelope. The machine status for all four checks was
`status=equivalent`, with `backend_status=proved` or exact normalization.

## Manual/source-anchored obligations

1. **Squared-TT marginal:** the paper’s Proposition 2 and Algorithm 2(c), and
   the author `marginalise.m:25-51`, establish the source operation: propagate
   one-dimensional mass contractions through the squared TT and retain the
   resulting core factors. The local implementation’s Gram contraction at
   `zhao_cui_algorithm2_preparation_tf.py:124-131` has the corresponding
   structural shape. A general tensor proof of that implementation was not
   machine-encoded here; status `source-anchored, implementation not fully
   checked`.
2. **Upper conditional inverse:** paper Eq. (20)-(21), PDF p. 14, and
   `eval_cirt_reference.m:102-153` support the continuous inverse-CDF route.
   The local manuscript’s conditional inversion proposition is valid under
   positive continuous conditional mass, but the implementation’s bounded
   grid/bisection route needs a separately defined piecewise proposal law.
   Status `manual derivation; numerical-law distinction remains open`.
3. **Physical density:** for each fixed conditioning particle, the affine
   inverse and one current-state Jacobian give the density in the manuscript’s
   Eq. (algthree-physical-proposal). The scalar Jacobian normalization was
   machine-checked above. Status `manual multivariate derivation supported by
   scalar check`.
4. **Algorithm 3 weights:** paper Eqs. (22)-(23), Algorithm 3(c-d), and the
   author source support direct identity continuation and the model-to-proposal
   ratio, with no auxiliary ancestor probability. The cancellation algebra was
   machine-checked above. Status `source-anchored; support and finite empirical
   cloud assumptions remain required`.
5. **Frozen score:** the local evaluator fixes states and proposal log
   densities and differentiates only model terms at
   `zhao_cui_algorithm3_tf.py:148-161`. The centering and weighted-mean pieces
   were machine-checked. The complete horizon induction is a manual proof of
   the frozen finite scalar, not a proof of the adaptive TT total derivative.
   Status `manual induction supported by bounded algebra checks`.

## Negative and non-encodable checks

The first attempts used function-call notation such as `q(x)` and prose-only
targets. MathDevMCP returned `not_encodable` or `unverified`; these outcomes
are tool limitations and are not refutations. The prose-only conditional and
frozen-score `derive_from` calls explicitly requested typed `lhs`/`rhs` before
any certifying route can run. No unsupported tool response is promoted to a
mathematical verdict.

## Decision

The source-level mathematical structure is coherent under the stated fixed
program and positivity assumptions. The checks do not remove the active
implementation/evidence blockers:

- local rank initialization can collapse a configured rank to effective rank
  one;
- aggregate non-finiteness is not included in the evaluator validity mask;
- bounded numerical CDFs are not the same density as the smooth TT law unless
  the piecewise law is named and used consistently; and
- no non-test claim-bearing consumer call chain was found.

These are implementation and evidence defects, not failures of the basic
paper identities. Continue with a small deterministic CPU diagnostic stage to
reproduce each blocker and test only proposed fail-closed/labeling repairs.

