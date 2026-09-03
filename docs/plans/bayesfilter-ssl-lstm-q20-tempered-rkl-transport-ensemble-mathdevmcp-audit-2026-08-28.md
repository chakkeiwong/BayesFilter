# MathDevMCP audit: tempered reverse-KL transport ensemble

Date: 2026-08-28  
Status: `PASS_WITH_EXPLICIT_TOOL_LIMITS_NO_FORMAL_CERTIFICATE`

Audited source:
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex`.

The isolated audit copy has SHA-256
`26dad3318f4558df148ca1d20138840c0c3c16e934f42b561d0d62e9e2e210d1`.
It is stored at
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-28/mathdevmcp/isolated-root/note.tex`.

## Verdict

No checked algebra contradicts the document, and manual review found no false
proposition under its stated assumptions. MathDevMCP did not certify the
measure-theoretic kernel-invariance, detailed-balance, or cold-marginal proofs.
Its label-scoped backend classified those obligations as inconclusive or not
encodable. The document therefore remains a proposition-and-proof research
note with bounded symbolic support, not a machine-verified proof.

The most important scientific repair did not come from a CAS identity. The
review recognized that fully optimizing every component at `beta=0` can erase
distributional diversity, so the proposal now requires an ablation between pure
continuation and predeclared fresh restarts or branching at positive beta. The
note also binds the current single-map tuner limitation and exact literature
sections.

## Tool environment

`mathdevmcp doctor` returned `ok: true` in the `tfgpu` environment with Python
3.13.13, SymPy 1.14.0, SageMath 10.7, LaTeXML 0.8.8, and Pandoc 3.10. Direct
Lean was unavailable because no default toolchain was configured; LeanDojo,
LeanExplore, and Pantograph were unavailable. No Lean certificate was attempted
or claimed.

## Audit ledger

| Check | Result | Interpretation |
|---|---|---|
| Document rigor plan on the isolated final source | 729 lines, 15 sections, 41 unique labels, 22 labeled equations selected, no duplicate labels, no missing references, four equation rows localized with parser uncertainty | Full labeled-equation inventory, not a proof. |
| Full `audit-math-document-rigor` with SymPy and Sage | Tool crashed in `document_exposition.py` with `KeyError: 'evidence_refs'` under both actionable and forensic attempts | Tool-side report failure; it supplies no pass or fail verdict and wrote no usable rigor report. |
| Deep applied-math audit | `completed_with_limits`, zero findings, 11 selected obligations marked `not_checkable`, no specialist execution | No contradiction was found, but zero findings is not certification. |
| `prop:separated-weights` label audit | `inconclusive`; one row not extracted and the finite-sum normalization not encodable | The Lagrange-multiplier proof required manual review. |
| `prop:chart-invariance` label audit | `inconclusive`; no equation-like obligation extracted | Measure pushforward/invariance proof required manual review. |
| `prop:replica-swap` label audit | `inconclusive`; swap density notation not encodable | Detailed balance required manual review plus a scalar reduction. |
| `thm:exact-cold-marginal` label audit | `inconclusive`; marginal statement not encodable | Common-invariant-kernel argument required manual review. |

The final deep-audit artifact is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-28/mathdevmcp/applied-deep-final/audit-7adce6cde7ed49c5bd4f5648af7e5706bdc4ef354385f9dd419dafb8966be254.json`
with SHA-256
`7adce6cde7ed49c5bd4f5648af7e5706bdc4ef354385f9dd419dafb8966be254`.

## Bounded symbolic checks

| Obligation | Backend result | Scope |
|---|---|---|
| `(b1-b2)(l2-l1) = b1*l2+b2*l1-b1*l1-b2*l2` | SymPy `equivalent` | Certifies the scalar pure-power swap log-ratio expansion only. |
| Difference of interpolated endpoint energies equals the interpolated differences | SymPy `equivalent` | Certifies the algebra in the barrier interpolation. |
| `c1*c2*c3 = exp(log(c1)+log(c2)+log(c3))` for positive factors | SymPy `equivalent` | Checks a finite product reduction; Tonelli and the general product theorem remain manual. |
| Common-normalizer cancellation in a two-component weight ratio | SymPy `equivalent` | Supports the normalized exponential weight form. |
| Two-component normalized weights sum to one after substituting their positive denominator | SymPy `equivalent` | Supports normalization only. |
| Logarithmic stationarity substitution for `alpha_i` | SymPy `unverified` because its bounded parser did not normalize nested `exp(-d)` and `log` terms | No result; the manual Lagrange-multiplier derivation is retained. |

## Manual proposition audit

| Result | Manual check | Disposition |
|---|---|---|
| Reverse-KL identity | Change variables under a diffeomorphism and substitute the unnormalized target. | Correct under integrability assumptions; source anchored to Hoffman et al., Section 2.2 equations (2)--(3) and Section 2.3. |
| Product importance mismatch | Factor the nonnegative integrand and use Tonelli; reciprocal bound follows from `c_j >= 1+delta`. | Correct; proves a failure example, not universal exponential collapse. |
| Missing sampled region | Independence gives `(1-m(A))^N`; an atomic estimator has no row in the region. | Correct; does not say formal support is absent. |
| Categorical transport mixture | Condition on the discrete component index. | Correct; the averaged-map counterexample is valid. |
| Mixture reverse KL | Split the mixture expectation and reparameterize each component by an IID Gaussian. | Correct if cross-component densities and differentiation envelopes exist. |
| Separated-region weights | Split disjoint supports; solve a strictly convex simplex problem. | Correct; `alpha_i` equals regional mass only when local component errors are equal to zero. |
| Conditional multi-start coverage | Independence and a union bound. | Correct but conditional on unknown, possibly zero basin probabilities. |
| Finite-query non-identification | Add a positive compactly supported smooth bump away from the finite query set. | Correct for the stated oracle model; it rules out generic finite certification, not structured-target guarantees. |
| Proper bridge and cold objective | Take logarithms of the geometric bridge and set `beta=1`. | Correct when every bridge normalizer is finite. An improper uniform endpoint is correctly excluded. |
| Chart-kernel invariance | Push the physical target through `T_i^{-1}`, apply invariant latent kernel, then push back. | Correct for frozen bijections and exact invariant component kernels. |
| Fixed chart-kernel mixture | Use linearity with fixed state-independent weights. | Correct; the two-state counterexample correctly rejects an uncorrected state-dependent selector. |
| Replica swap and cold marginal | Product-target Metropolis ratio cancels unaffected factors and normalizers; common-invariant compositions preserve the product law. | Correct; source anchored to Hukushima and Nemoto, Section II equations (2.1)--(2.7). It proves stationarity, not mixing. |
| Independence MH | Symmetric accepted probability flow. | Correct under proposal support; acceptance may still collapse in high dimension. |

## Claim boundary

The audit supports using the equations as an implementation specification. It
does not establish that a finite transport ensemble finds every mode, that the
maps are well trained, that replica exchange is irreducible or mixed, that q=20
predicts high-dimensional performance, or that the current code implements the
new proposal. Those are assigned to explicit implementation and experimental
phases in the companion plan.

