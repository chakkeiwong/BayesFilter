# MathDevMCP Audit And Fix Proposal

Question: Audit the active Zhao-Cui Algorithm 3 derivation, including recursive chart closure, pullback Jacobian, upper conditional inversion, physical proposal density, likelihood projection, and frozen score. Report algebraic counterexamples separately from formalization gaps; do not infer implementation correctness.
Status: no_proposal

## Certification Boundary

The audit-and-fix report is diagnostic guidance only; it does not apply edits, verify repaired text, or certify mathematical correctness.

## Audit Coverage

Mode: `explicit_labels`
Audited labels: 17 / 17
Skipped labels: 0
Complete for selected scope: True
Target file: `attempt05_n4_failure_analysis.tex`
Audited label list: `prop:algthree-chart-closure`, `eq:algthree-retained-coordinate-marginal`, `eq:algthree-retained-reprojection`, `prop:algthree-pullback`, `eq:algthree-block-determinant`, `eq:algthree-pullback-mass`, `prop:algthree-upper-kr`, `eq:algthree-upper-inverse`, `prop:algthree-physical-density`, `eq:algthree-physical-proposal`, `prop:algthree-importance`, `eq:algthree-importance-identity`, `prop:algthree-positive-projection`, `eq:algthree-likelihood-projection`, `prop:algthree-score`, `eq:algthree-score-recursion`, `eq:algthree-score-result`

## Tool Uses

| Tool | Purpose | Status | Output contract | Arguments |
| --- | --- | --- | --- | --- |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-chart-closure`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-chart-closure', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-retained-coordinate-marginal`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-retained-coordinate-marginal', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-retained-reprojection`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-retained-reprojection', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-pullback`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-pullback', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-block-determinant`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-block-determinant', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-pullback-mass`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-pullback-mass', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-upper-kr`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-upper-kr', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-upper-inverse`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-upper-inverse', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-physical-density`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-physical-density', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-physical-proposal`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-physical-proposal', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-importance`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-importance', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-importance-identity`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-importance-identity', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-positive-projection`. | inconclusive | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-positive-projection', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-likelihood-projection`. | inconclusive | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-likelihood-projection', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:algthree-score`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'prop:algthree-score', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-score-recursion`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-score-recursion', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:algthree-score-result`. | unverified | proof_audit_v2_result | `{'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'label': 'eq:algthree-score-result', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | diagnostic_only | high_level_workflow_result | `{'question': 'Audit the active Zhao-Cui Algorithm 3 derivation, including recursive chart closure, pullback Jacobian, upper conditional inversion, physical proposal density, likelihood projection, and frozen score. Report algebraic counterexamples separately from formalization gaps; do not infer implementation correctness.', 'evidence_count': 17, 'source': {'root': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05', 'labels': ['prop:algthree-chart-closure', 'eq:algthree-retained-coordinate-marginal', 'eq:algthree-retained-reprojection', 'prop:algthree-pullback', 'eq:algthree-block-determinant', 'eq:algthree-pullback-mass', 'prop:algthree-upper-kr', 'eq:algthree-upper-inverse', 'prop:algthree-physical-density', 'eq:algthree-physical-proposal', 'prop:algthree-importance', 'eq:algthree-importance-identity', 'prop:algthree-positive-projection', 'eq:algthree-likelihood-projection', 'prop:algthree-score', 'eq:algthree-score-recursion', 'eq:algthree-score-result'], 'file': 'attempt05_n4_failure_analysis.tex', 'source_digest': '8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69'}}` |
| `validate_proposed_fixes` | Attach deterministic backend-attempt accountability to concrete proposed fixes. | completed | proposal_fix_validation_summary | `{'policy': 'require_attempt_when_encodable', 'backend_order': ['lean', 'sympy'], 'detail_count': 12}` |

## Audited Evidence

- `prop:algthree-chart-closure`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-retained-coordinate-marginal`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-retained-reprojection`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:algthree-pullback`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-block-determinant`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-pullback-mass`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:algthree-upper-kr`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-upper-inverse`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:algthree-physical-density`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-physical-proposal`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:algthree-importance`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-importance-identity`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:algthree-positive-projection`: inconclusive - No proof-audit v2 obligation could be certified or refuted.
- `eq:algthree-likelihood-projection`: inconclusive - No proof-audit v2 obligation could be certified or refuted.
- `prop:algthree-score`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-score-recursion`: unverified - At least one obligation remains unverified or diagnostic-only.
- `eq:algthree-score-result`: unverified - At least one obligation remains unverified or diagnostic-only.

## Proposed Fix Validation

Enabled: True
Policy: `require_attempt_when_encodable`
Backend order: `['lean', 'sympy']`
Validated details: 12
Status counts: attempted_not_certified=3, not_encodable=9

## Proposed Changes

- No concrete proposed change could be derived safely.

## Evidence Gaps

1. `concretize_before_fix` for `obligation_1`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Observation-informed guide construction > line 3020`
   Problem: The localized source is prose or an incomplete fragment, so no concrete mathematical replacement is available.
   Why: Proof-audit v2 returned `inconclusive` with substatus `inconclusive:source_label_missing` on route `symbolic` and matrix-IR status `parsed`. The row could not be extracted as a safe proof obligation.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: \begin{proposition}[Validity of the positive likelihood-weighted projection]
   Evidence refs: proof_audit_v2:prop:algthree-positive-projection:obligation_1
2. `prove_reconstructed_obligation` for `invertibility_required`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > The upper conditional KR proposal > line 2832`
   Problem: The complete source-bound target is localized but remains uncertified.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:missing_assumption` on route `human_review` and matrix-IR status `parsed`. The obligation has missing shape, dimension, or regularity constraints.
   Proposed fix: Formalize the exact source-bound target, rerun a suitable deterministic backend, and retain the source digest and obligation digest for `prop:algthree-physical-density`.
   Proof target: `q_t^{U\text{-}TT}(x\mid c_i,y_t) = q_t^U\!\left((L_{t,i}^U)^{-1}(x-m_{t,i}^U)\mid v_i\right) \left\|\det L_{t,i}^U\right\|^{-1}`
   Derivation plan: Formalize this local proof obligation with explicit assumptions, then rerun proof-audit v2 on the same label.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: q_t^{U\text{-}TT}(x\mid c_i,y_t)
   Evidence refs: proof_audit_v2:prop:algthree-physical-density:obligation_1
3. `prove_reconstructed_obligation` for `determinant/logdet operand`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Reference coordinates and the squared-TT density > line 2708`
   Problem: The complete source-bound target is localized but remains uncertified.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `lean_candidate` and matrix-IR status `parsed_with_unresolved`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Formalize the exact source-bound target, rerun a suitable deterministic backend, and retain the source digest and obligation digest for `eq:algthree-retained-reprojection`.
   Proof target: `F_t^{\rm ret}(w,r) = \frac{\widehat\pi_t(\bar G_t(w;r,y_t),r) \|\det D_w\bar G_t(w;r,y_t)\|} {r_w(w)r_r(r)}`
   Derivation plan: Formalize this local proof obligation with explicit assumptions, then rerun proof-audit v2 on the same label.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: F_t^{\rm ret}(w,r)=
   Evidence refs: proof_audit_v2:eq:algthree-retained-reprojection:obligation_1
4. `concretize_before_fix` for `inverse/solve operand`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > The upper conditional KR proposal > line 2798`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `lean_candidate` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `U_k = \bigl(F_{t,k}^u\bigr)^{-1}`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: U_k=\bigl(F_{t,k}^u\bigr)^{-1}
   Evidence refs: proof_audit_v2:prop:algthree-upper-kr:obligation_1
5. `concretize_before_fix` for `derivative expression`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Analytical score of the same frozen finite program > line 3088`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `lean_candidate` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `s_0^{(i)} = \nabla_\theta\log p_\theta(x_0^{(i)})`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: s_0^{(i)}&=\nabla_\theta\log p_\theta(x_0^{(i)})
   Evidence refs: proof_audit_v2:prop:algthree-score:obligation_1
6. `prove_reconstructed_obligation` for `obligation_1`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Reference coordinates and the squared-TT density > line 2663`
   Problem: The complete source-bound target is localized but remains uncertified.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed_with_unresolved`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Formalize the exact source-bound target, rerun a suitable deterministic backend, and retain the source digest and obligation digest for `prop:algthree-chart-closure`.
   Proof target: `M_t(u,r) = \int F_t(u,r,e)r_e(e)\,de`
   Derivation plan: Formalize this local proof obligation with explicit assumptions, then rerun proof-audit v2 on the same label.
   Validation: `attempted_not_certified` - Configured backends were attempted, but none certified or refuted this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=unknown (diagnostic)
   Source: M_t(u,r)=\int F_t(u,r,e)r_e(e)\,de.
   Evidence refs: proof_audit_v2:prop:algthree-chart-closure:obligation_1
7. `concretize_before_fix` for `obligation_2`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > The upper conditional KR proposal > line 2800`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `\qquad k = d,d-1,\ldots,1.`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `attempted_not_certified` - Configured backends were attempted, but none certified or refuted this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=unknown (diagnostic)
   Source: \qquad k=d,d-1,\ldots,1.
   Evidence refs: proof_audit_v2:prop:algthree-upper-kr:obligation_2
8. `prove_reconstructed_obligation` for `obligation_1`
   Location: `attempt05_n4_failure_analysis.tex > line 3012`
   Problem: The complete source-bound target is localized but remains uncertified.
   Why: Proof-audit v2 returned `inconclusive` with substatus `unverified:parser_limit` on route `human_review` and matrix-IR status `parsed_with_unresolved`. Parser policy did not select a provenance-preserving backend for certification.
   Proposed fix: Formalize the exact source-bound target, rerun a suitable deterministic backend, and retain the source digest and obligation digest for `eq:algthree-likelihood-projection`.
   Proof target: `P^U(c,y_t) = \sum_j\alpha_j (\chi_j-m^U)(\chi_j-m^U)^\mathsf T`
   Derivation plan: Formalize this local proof obligation with explicit assumptions, then rerun proof-audit v2 on the same label.
   Validation: `attempted_not_certified` - Configured backends were attempted, but none certified or refuted this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=unknown (diagnostic)
   Source: P^U(c,y_t)&=\sum_j\alpha_j
   Evidence refs: proof_audit_v2:eq:algthree-likelihood-projection:obligation_1
9. `concretize_before_fix` for `obligation_3`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Analytical score of the same frozen finite program > line 3091`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed_with_unresolved`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `S_0 = \sum_iW_0^{(i)}H_0^{(i)},`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: &S_0&=\sum_iW_0^{(i)}H_0^{(i)},
   Evidence refs: proof_audit_v2:prop:algthree-score:obligation_3
10. `concretize_before_fix` for `obligation_4`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Analytical score of the same frozen finite program > line 3092`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `D_0^{(i)} = H_0^{(i)}-S_0,`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: &D_0^{(i)}&=H_0^{(i)}-S_0,\\
   Evidence refs: proof_audit_v2:prop:algthree-score:obligation_4
11. `concretize_before_fix` for `obligation_6`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Analytical score of the same frozen finite program > line 3095`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `H_t^{(i)} = D_{t-1}^{(i)}+s_t^{(i)},`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: H_t^{(i)}&=D_{t-1}^{(i)}+s_t^{(i)},
   Evidence refs: proof_audit_v2:prop:algthree-score:obligation_6
12. `concretize_before_fix` for `obligation_7`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Analytical score of the same frozen finite program > line 3096`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed_with_unresolved`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `S_t = \sum_iW_t^{(i)}H_t^{(i)},`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: &S_t&=\sum_iW_t^{(i)}H_t^{(i)},
   Evidence refs: proof_audit_v2:prop:algthree-score:obligation_7
13. `concretize_before_fix` for `obligation_8`
   Location: `attempt05_n4_failure_analysis.tex > Zhao--Cui Algorithm 3 with an Observation-Guided TT Proposal > Analytical score of the same frozen finite program > line 3097`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `D_t^{(i)} = H_t^{(i)}-S_t.`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Validation: `not_encodable` - No configured backend could encode this proposed fix target.
   Backend attempts: lean=not_encodable (diagnostic); sympy=not_encodable (diagnostic)
   Source: &D_t^{(i)}&=H_t^{(i)}-S_t.
   Evidence refs: proof_audit_v2:prop:algthree-score:obligation_8

## Next Actions

- `human_review`: Review proposed changes before applying edits.
- `rerun_relevant_audit`: After any manual repair, rerun the audit evidence that produced the proposal.

## Non-Claims

- `diagnostic_evidence_not_proof`: Diagnostic evidence is not a proof certificate.
- `general_theorem_proving_not_claimed`: This scoped workflow result does not claim general theorem-proving ability.
- `release_readiness_not_claimed`: This scoped workflow result does not claim release readiness.
- `fix_proposal_not_applied_or_verified`: Proposed fixes are diagnostic guidance only; they are not applied edits, proof certificates, or semantic implementation verification.
- `audit_fix_report_not_applied_or_certified`: The audit-and-fix report is diagnostic guidance only; it does not apply edits, verify repaired text, or certify mathematical correctness.
