# Terminal mathematical and document review — 2026-09-14

The complete program executed, and its density, correction and frozen-score claims have focused numerical support. A full MathDevMCP proof/audit pass is **not established**. Its successful checks below are deliberately reported at their actual scope.

## MathDevMCP results

| Check | Result | Meaning |
|---|---|---|
| Conditional Bayes ratio and affine Jacobian | `equivalent`, SymPy | Algebraic cancellation verified for positive denominators; integral identities and change-of-variables assumptions are supplied by the written proof. |
| Polynomial/Gaussian conditional mixture | `equivalent`, SymPy | Mixture expansion agrees with the reported density for positive polynomial mass and tau. The runtime fails closed for invalid/zero polynomial mass. |
| Centered covariance algebra | `equivalent`, SymPy | Expansion agrees with raw second moment minus the mean outer product after quadrature normalization; this does not prove quadrature exactness or covariance positivity. |
| Importance-density cancellation | `equivalent`, SymPy | Integrand cancellation verified for positive proposal density; integrability and conditioning remain explicit proof assumptions. |
| Actual conditional-density code | `scope_limited_match`, no missing terms | The log-density expression and genuine alias `logr → _log_standard_normal` match the actual sampler functions. The tool expressly does not verify function semantics. |
| Four proposition derivation audits | All `inconclusive` | SGQF statement yields no equation-like obligation; integral, matrix and gradient statements exceed the bounded backend or extraction capability. No proposition was certified or refuted. |
| Batch document rigor audit | Unavailable | Equation selection led to `retrieve_label` / `KeyError`, including a clean, byte-identical source-copy retry. No batch pass is claimed. |

The initial structural request included explanatory prose; the matcher treated English words as mathematical identifiers. Its `structural_mismatch` is preserved in `conditional-math-to-code.json` and is not mathematical evidence. The corrected request supplies only the actual log-density equation and records the resulting limited match in `conditional-density-structural.json`. The first proposition-only document-rigor request selected zero equations; this is zero coverage, not a pass. Lookup errors and all proposition/symbolic reports are preserved in this directory. No claim is based on `ok:true` alone.

The propositions' written proofs were checked directly: likelihood-weighted moment ratios under the predictive Gaussian; exact integration of a squared Hermite TT; conditional and physical-density normalization; conditional importance cancellation; and differentiation of the frozen finite log-normalizer recursion. The score derivation includes stationary-prior dependence on the transition matrix and noise scale. These are human-readable derivations plus Codex self-review, not independent formal certification.

## Program review and evidence limits

The implementation audit traces each consumer to SGQF, target preparation, L1 selection, retained marginalization, shared upper KR and weighting. Ten numerical tests and the CPU/GPU smoke supply executable wiring and density evidence. The complete d1/d4 T20 campaign checks downstream filtering and the three-control frozen score. All three numerical source hashes and the frozen plan match the launch manifest. No numerical code changed after the completed campaign.

The skeptical review checks initial Gaussian timing, previous-density dependence, both affine Jacobians, reference-row measure, actual mixture density, independent innovations, normalized weight carry, post-weight resampling and stationary-covariance differentiation. It also checks comparator fairness, validation/audit separation, declared controls and the distinction between promotion and continuation vetoes. The heuristic promotion veto is correctly retained. The d4 reference shares the particle-update kernel; independent random seeds do not make it a separate implementation. The scalar grid is separately implemented.

Remaining unproved or untested claims include uniform accuracy over states/parameters, finite importance-weight variance, low TT rank sufficiency, preservation of arbitrary higher moments, superiority, scaling above d4 and HMC readiness. Small fitting residuals, compiler success and a bounded symbolic identity cannot establish them. A Gaussian SGQF closure is useful here but is not the exact non-Gaussian filtering law.

## Rendered document and preservation

The protected baseline SHA-256 is `f62c14fd077b7d6d9b6220b3c1c0a4fba7ba48b40db3ba0b74a7db7272bdbbc6`. Its historical body between abstract and appendix is byte-identical in the revised note; no existing label or citation group was removed. The title/abstract now identify the executed construction, and section 14 adds its derivation, four propositions/proofs and actual results. This is a technical algorithm note retaining the failed approaches, not a shortened summary that drops their arguments.

The final 39-page PDF compiled with `pdflatex`; `latexmk` is not installed. There are no undefined-reference or overfull-box diagnostics. Existing underfull-box and PDF-bookmark warnings remain cosmetic. Pages 34–39 were rendered and inspected, including SGQF, regression, conditional/marginal equations, proofs, score formulas and the results table. A long result path overflow was corrected. The final source also explicitly states that the d4 reference shares the update kernel. Naturalness/readability remain provisional pending the user's reading; model review is not a human voice certificate.

Final LaTeX SHA-256: `5b48530cc210625dc2d974b63d382d31f124fdcc96bd8372e056ed13b81bc250`.

Final PDF SHA-256: `2ea15c3ba7d617744f5cde8deccc69be9fa486bd12078cb228f0d1a62caac3b7`.

Decision: deliver the completed executable, numerical evidence and updated PDF. Withhold comparative/default promotion and a full MathDevMCP proof-pass claim. The batch-audit tool failure is a recorded limitation, not permission to misreport either the mathematical result or the completed execution.
