# A11 mathematical review

Scope: the additive Section 18, propositions 39–45, equations 168–182.
The complete pre-A11 manuscript and SHA are protected here. This review does
not reopen historical claims or certify the full document. The new section
contains all seven proofs next to their statements and explains the downstream
filtering question. Rendered pages 64–68 were inspected: equations and proof
endings fit the page, with no overflow or missing displays. A final build will
include the physical observation scale correction and numerical results.

## Proof review

| Proposition | Checked argument and limitations |
|---|---|
| Relative covariance floor | Whiten by R, diagonalize lambda I+(1-lambda)R^-1/2 P R^-1/2; eigenvalue, inverse, determinant and conditional upper bounds follow. P is PSD, R SPD, lambda positive. No claim that P<=R or every direction shrinks. |
| Design correction | r>=delta*phi; multiplying by the sampling density proves the fixed-h, fixed-scale identity. Explicitly excludes an unbiased training-error claim for data-dependent h and scale. |
| Capacity and initialization | Zero-padding gives nested function classes; orthogonal projection gives a Pythagorean error decomposition. Finite ALS, L1 core penalties and nonlinear-target error are expressly outside the monotonicity claim. |
| Conditional normalization | Finite Gaussian polynomial moments define a(v); tau>0 ensures both denominators and density are positive. The physical Jacobian cancels under integration. |
| Exact weights and ESS | q>=epsilon*f yields the weight and second-moment bounds. Integrability is verified for the actual Gaussian SV likelihood including beta. Strong laws concern conditionally iid draws at one fixed ancestor, not the complete weighted particle system. |
| Mixture-strength convexity | Differentiate the affine denominator twice. On an interior epsilon interval, bound f/q and qTT/q uniformly; the assumed integrable f*g^2 dominates both differentiated integrands. No monotone-ESS or convex-filter-MSE claim. |
| Local analytical derivatives | Differentiate C=LL' and solve the lower-triangular equation E+E'=B. Include both chart coordinates, coefficient dependence, tau and determinant derivatives. Mixture differentiation is exact for frozen epsilon. L1/guide branch switches and full-filter total derivatives remain unproved. |

Code-backed checks include the spectral floor at a guide covariance of 1e-28,
exact lambda=1 parity, finite-difference Cholesky derivative agreement, and a
real pair fit using the new charts followed by the physical-mixture consumer.
These supplement, rather than replace, the general proofs.

## MathDevMCP findings and dispositions

The first document derivation audit returned partial coverage: four of fourteen
requested equation labels were extracted; ten compound displays were
quarantined with unknown_row_shape. This is an extraction limitation, not a
counterexample. The full diagnostic report is math-derivation.json/.md.

| Target/finding | Disposition |
|---|---|
| Projection identity: missing formalized obligation | The adjacent proof states the Hilbert-space orthogonality argument. No Lean certificate is available. |
| Conditional density: conditional law/integrability | z is fixed; finite polynomial Gaussian moments define a(v). Normalization is proved by change of variables. |
| Conditional denominator nonzero | a(v)>=0 and tau>0 give a(v)+tau>0 explicitly in the proof. This is not an additional missing scientific assumption. |
| Determinant domain | L is the d-by-d Cholesky factor of an SPD covariance, with positive diagonal. Added an explicit common dimension declaration. |
| Cholesky derivative: formalized local obligation | The lower-triangular solution E=Phi(B) is derived and independently checked numerically. |
| TT score: denominator and differentiability | h^2+tau>0 and fixed-branch differentiability assumptions are explicit. The proposition includes all local dependencies and excludes a total-filter claim. |

The broad rigor tool failed twice at retrieve_label (KeyError). A third
bounded request against the exact new-section excerpt also failed. Error
artifacts are retained; these failures are not a successful full rigor audit.
The exact-label check (`math-floor-label.json`) found the covariance display,
but returned inconclusive: it could neither certify nor refute an obligation.
The scalar mixture-score identity was proved by the deterministic symbolic
backend (`math-mixture-score-algebra.json`). The second-moment derivative
request was routed to human review, and a bounded scalar retry failed encoding
the `diff` expression (`math-second-moment-scalar-retry.json`). These are tool
limitations, not evidence against the explicitly differentiated identity in
the manuscript. Their scope must not be enlarged into formal verification of
seven propositions or the complete manuscript.

Self-review also corrected the stochastic-volatility likelihood expression to
retain the fixture's positive observation scale beta. Its second-moment bound
now carries (2*pi*beta^2)^(-d), consistent with the implemented physical model.
No algorithm change followed from that documentation correction.

Verdict: the stated mathematical construction and proofs pass scoped manual
review and targeted numerical checks. MathDevMCP coverage remains partial;
no formal proof or human-reader acceptance certificate is claimed. Neither
limitation blocks the authorized bounded diagnostic experiment.
