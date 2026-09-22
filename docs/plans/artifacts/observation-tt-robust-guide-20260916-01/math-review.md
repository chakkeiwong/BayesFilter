# A10 mathematical and manuscript review

The manuscript adds eight propositions and proofs in Section 17, equations
156--166, with the observed collapse, guard limits, convex SV mode, positive
change of measure, model covariance charts, physical mixture and smoothness.
All pre-existing manuscript text was retained. Two pdflatex passes succeeded;
rendered pages 55--61 were inspected for equation layout and narrative flow.
Human reader acceptance remains pending; no formal-proof certificate claimed.

## MathDevMCP audit and dispositions

`math-rigor-01` selected no targets because the tool indexes equation labels.
`math-rigor-02` inspected all eleven new equation labels and raised four issues.
The covariance dimension/invertibility exposition is strengthened immediately
before the SV potential. The suggested Neumann-series assumption concerns an
unrelated inverse of I-Omega: our inverse is S^-1 with S positive definite,
so that patch is mathematically inapplicable and was not inserted. The two
formalization requests concern signed-weight and chart definitions; their
stated properties are proved in the adjacent propositions, not by the tool.

`math-derivations-01` reached two of four requested targets, exhausted its
bounded attempt budget, and required Lean source/typed assumptions. It also
misclassified the continuous change-of-variables integral as needing finite
support; that is unnecessary under the stated integrability assumption.
Its mathematical-blocked label is a routing/assumption status, not a supplied
counterexample. The symbolic derivative attempt could not encode exp/diff
(`Symbol object is not callable`). These limitations remain recorded. No
MathDev approval or automated full proof is claimed. The tool reports include
legacy evidence-binding/publication quarantines; no publication is attempted.

## Executor proof audit

Verified directly: scale-invariant SPD counterexample; signed covariance
counterexample; Gaussian rounding bound under its explicit assumptions;
SV gradient/Hessian, coercivity and implicit-function condition; density ratio
and affine-spanning positive covariance proof; independent R recursion and
Jacobian; normalized physical mixture and exact importance identity; Gaussian
exponential-moment bound including zero observations; fixed-branch mixture
score. No proof guarantees realized ESS, quadrature accuracy, complete-filter
smoothness, or high-dimensional scalability. Numerical identity and endpoint
tests are required next. The review is an executor self-review, not an
independent reviewer verdict. Optional implementation may proceed under A10.

## Signed amplitude addendum

The ninth proposition derives the least-squares amplitude scale and proves
that its sign cancels in the normalized squared density. MathDevMCP's bounded
equation audit is preserved in mathdev-amplitude.json and mathdev-amplitude.md.
It selected the new equation but abstained on formalization/role, with partial
coverage. It supplied neither a concrete counterexample nor a proof certificate.
The final executor review makes the finite training set, finite row values,
nonnegative weights and finite positive continuous squared-amplitude integral
explicit; these strengthen the written hypotheses without changing the equation.
The integral is explicitly with respect to a fixed reference measure, Gaussian
for the implemented polynomial amplitude. A polynomial's Lebesgue squared
integral need not be finite; that would be the wrong reference measure for
the runtime TT. Sign and homogeneous-scale cancellation hold under either
measure when the stated finite-normalizer assumption holds.
The elementary quadratic derivative and normalizer cancellation are checked
directly. Tests-05 records 38 passing tests, including exact positive-branch
parity, sign equivalence and zero-scale rejection. Smoke-03 reproduces the
exposed negative scalar and completes the actual consumer. Neither those tests
nor the audit proves a complete analytical filter derivative.

## Final manuscript closeout

Section 17 now contains nine propositions/proofs and the fresh A10 filtering
results. LaTeX passes 6 and 7 succeed, producing 64 pages; the final pass has
no undefined references, LaTeX warnings or overfull boxes. Rendered pages
61--64 were inspected after the last changes, in addition to the earlier
proof pages. manuscript-preservation.json records one 610-line insertion and
no removed or replaced protected-baseline lines. This supersedes the earlier
intermediate insertion count. The final proof assumptions and tool limitations
above remain part of the result; the build does not upgrade partial MathDevMCP
coverage to formal verification. See result-review.md for terminal review.
