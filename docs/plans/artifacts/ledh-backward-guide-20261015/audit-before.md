# MathDevMCP Audit And Fix Proposal

Question: Check the four proofs: Gaussian square completion and total tangents; frozen-mixture moving-anchor derivatives; exact Jacobian conditions for proposal correction; full derivative of the same finite OT/Contract-E recursion. Does the plan to add only a guided Gaussian pre-flow draw and its actual denominator preserve this mathematical specification? No score superiority, online refitting derivative, or unbiased full-reset likelihood is claimed.
Status: no_proposal

## Certification Boundary

The audit-and-fix report is diagnostic guidance only; it does not apply edits, verify repaired text, or certify mathematical correctness.

## Audit Coverage

Mode: `explicit_labels`
Audited labels: 4 / 4
Skipped labels: 0
Complete for selected scope: True
Target file: `ledh_backward_guidance.tex`
Audited label list: `prop:bg-gaussian`, `prop:bg-mixture`, `prop:bg-correction`, `prop:bg-total`

## Tool Uses

| Tool | Purpose | Status | Output contract | Arguments |
| --- | --- | --- | --- | --- |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:bg-gaussian`. | unverified | proof_audit_v2_result | `{'root': '/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score', 'label': 'prop:bg-gaussian', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'ledh_backward_guidance.tex', 'source_digest': None}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:bg-mixture`. | unverified | proof_audit_v2_result | `{'root': '/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score', 'label': 'prop:bg-mixture', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'ledh_backward_guidance.tex', 'source_digest': None}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:bg-correction`. | unverified | proof_audit_v2_result | `{'root': '/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score', 'label': 'prop:bg-correction', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'ledh_backward_guidance.tex', 'source_digest': None}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `prop:bg-total`. | unverified | proof_audit_v2_result | `{'root': '/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score', 'label': 'prop:bg-total', 'paragraph_context': True, 'before': 4, 'after': 1, 'summary_only': True, 'backend': 'sympy', 'task_context': 'general_math_audit', 'file': 'ledh_backward_guidance.tex', 'source_digest': None}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | diagnostic_only | high_level_workflow_result | `{'question': 'Check the four proofs: Gaussian square completion and total tangents; frozen-mixture moving-anchor derivatives; exact Jacobian conditions for proposal correction; full derivative of the same finite OT/Contract-E recursion. Does the plan to add only a guided Gaussian pre-flow draw and its actual denominator preserve this mathematical specification? No score superiority, online refitting derivative, or unbiased full-reset likelihood is claimed.', 'evidence_count': 4, 'source': {'root': '/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score', 'labels': ['prop:bg-gaussian', 'prop:bg-mixture', 'prop:bg-correction', 'prop:bg-total'], 'file': 'ledh_backward_guidance.tex'}}` |

## Audited Evidence

- `prop:bg-gaussian`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:bg-mixture`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:bg-correction`: unverified - At least one obligation remains unverified or diagnostic-only.
- `prop:bg-total`: unverified - At least one obligation remains unverified or diagnostic-only.

## Proposed Fix Validation

Enabled: False
Policy: `require_attempt_when_encodable`
Backend order: `['lean', 'sage', 'sympy']`
Validated details: 0

## Proposed Changes

- No concrete proposed change could be derived safely.

## Evidence Gaps

1. `concretize_before_fix` for `obligation_3`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 76`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `inconclusive` with substatus `inconclusive:source_label_missing` on route `symbolic` and matrix-IR status `parsed`. The row could not be extracted as a safe proof obligation.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `\dot P&=-P\dot QP,& \dot K&=\dot P+\dot H,&`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: \dot P&=-P\dot QP,& \dot K&=\dot P+\dot H,&
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_3
2. `concretize_before_fix` for `obligation_4`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > What a kernel mixture can contribute > line 123`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `inconclusive` with substatus `inconclusive:source_label_missing` on route `symbolic` and matrix-IR status `parsed_with_unresolved`. The row could not be extracted as a safe proof obligation.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `H&=\sum_k r_k\Lambda_k,& b&=\sum_k r_k\Lambda_kc_k.`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: H&=\sum_k r_k\Lambda_k,& b&=\sum_k r_k\Lambda_kc_k.
   Evidence refs: proof_audit_v2:prop:bg-mixture:obligation_4
3. `concretize_before_fix` for `obligation_2`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > The corrected finite likelihood and its derivative > line 202`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `inconclusive` with substatus `inconclusive:source_label_missing` on route `symbolic` and matrix-IR status `parsed_with_unresolved`. The row could not be extracted as a safe proof obligation.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `=\sum_{t=1}^T \log\sum_i\exp(\lambda_t^i),\qquad`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: =\sum_{t=1}^T \log\sum_i\exp(\lambda_t^i),\qquad
   Evidence refs: proof_audit_v2:prop:bg-total:obligation_2
4. `concretize_before_fix` for `invertibility_required`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 70`
   Problem: The audit reports a missing assumption or shape constraint, but the current tools have not derived the exact assumption statement to insert here.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:missing_assumption` on route `human_review` and matrix-IR status `parsed`. The obligation has missing shape, dimension, or regularity constraints.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `P&=Q^{-1},& K&=P+H,& V&=K^{-1},& m&=V(Pa+b).`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: P&=Q^{-1},& K&=P+H,& V&=K^{-1},& m&=V(Pa+b).
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_2
5. `concretize_before_fix` for `obligation_9`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 84`
   Problem: The audit reports a missing assumption or shape constraint, but the current tools have not derived the exact assumption statement to insert here.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:missing_assumption` on route `human_review` and matrix-IR status `parsed`. The obligation has missing shape, dimension, or regularity constraints.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `$\dot L = L\Phi(L^{-1}\dot V L^{-\top})$ and $\Phi$ retains the lower`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: $\dot L=L\Phi(L^{-1}\dot V L^{-\top})$ and $\Phi$ retains the lower
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_9
6. `concretize_before_fix` for `inverse/solve operand`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 70`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:missing_assumption` on route `human_review` and matrix-IR status `parsed`. The obligation has missing shape, dimension, or regularity constraints.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `P&=Q^{-1},& K&=P+H,& V&=K^{-1},& m&=V(Pa+b).`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: P&=Q^{-1},& K&=P+H,& V&=K^{-1},& m&=V(Pa+b).
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_2
7. `concretize_before_fix` for `determinant/logdet operand`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > The corrected finite likelihood and its derivative > line 171`
   Problem: The report entry is not concrete enough to appear as an applied-document repair.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `lean_candidate` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `bijection on the proposal support and $j = \log\|\det D_UF\|$ is its actual`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: bijection on the proposal support and $j=\log\|\det D_UF\|$ is its actual
   Evidence refs: proof_audit_v2:prop:bg-correction:obligation_1
8. `concretize_before_fix` for `obligation_1`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 68`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `$q^0 = \mathcal N(m,V)$, where`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: $q^0=\mathcal N(m,V)$, where
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_1
9. `concretize_before_fix` for `obligation_2`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 70`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:missing_assumption` on route `human_review` and matrix-IR status `parsed`. The obligation has missing shape, dimension, or regularity constraints.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `P&=Q^{-1},& K&=P+H,& V&=K^{-1},& m&=V(Pa+b).`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: P&=Q^{-1},& K&=P+H,& V&=K^{-1},& m&=V(Pa+b).
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_2
10. `concretize_before_fix` for `obligation_4`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 77`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `\dot V = -V\dot K V,\nonumber`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: \dot V&=-V\dot K V,\nonumber\\
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_4
11. `concretize_before_fix` for `obligation_5`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 78`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `\dot m = V(\dot P a+P\dot a+\dot b-\dot K m).`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: \dot m&=V(\dot P a+P\dot a+\dot b-\dot K m).
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_5
12. `concretize_before_fix` for `obligation_6`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 81`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `For fixed $\xi\sim\mathcal N(0,I)$, put $U = m+L\xi$ with`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: For fixed $\xi\sim\mathcal N(0,I)$, put $U=m+L\xi$ with
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_6
13. `concretize_before_fix` for `obligation_7`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 82`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `$L = \operatorname{chol}(V)$. Its tangent is`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: $L=\operatorname{chol}(V)$. Its tangent is
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_7
14. `concretize_before_fix` for `obligation_8`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 83`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `$\dot U = \dot m+\dot L\xi$, where`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: $\dot U=\dot m+\dot L\xi$, where
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_8
15. `concretize_before_fix` for `obligation_9`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > A precisely defined, frozen-guide proposal > line 84`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:missing_assumption` on route `human_review` and matrix-IR status `parsed`. The obligation has missing shape, dimension, or regularity constraints.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `$\dot L = L\Phi(L^{-1}\dot V L^{-\top})$ and $\Phi$ retains the lower`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: $\dot L=L\Phi(L^{-1}\dot V L^{-\top})$ and $\Phi$ retains the lower
   Evidence refs: proof_audit_v2:prop:bg-gaussian:obligation_9
16. `concretize_before_fix` for `obligation_3`
   Location: `ledh_backward_guidance.tex > Backward information as a proposal, not a replacement score > The corrected finite likelihood and its derivative > line 203`
   Problem: The audit reports that this claim needs formalization or human review; that is a certification gap, not a document edit by itself.
   Why: Proof-audit v2 returned `unverified` with substatus `unverified:manual_formalization_required` on route `human_review` and matrix-IR status `parsed_with_unresolved`. The obligation uses notation outside the bounded algebraic backend and needs formalization or human review.
   Proposed fix: Do not edit the document from this item alone. First produce a concrete assumption statement, replacement LaTeX, or proof obligation tied to the source line.
   Proof target: `w_t^i = \frac{\exp(\lambda_t^i)}{\sum_j\exp(\lambda_t^j)}.`
   Derivation plan: Use the referenced proof-audit obligation to derive a concrete local obligation; if the source is prose, refine the parser/provenance before proposing an edit.
   Source: w_t^i=\frac{\exp(\lambda_t^i)}{\sum_j\exp(\lambda_t^j)}.
   Evidence refs: proof_audit_v2:prop:bg-total:obligation_3

## Next Actions

- `human_review`: Review proposed changes before applying edits.
- `rerun_relevant_audit`: After any manual repair, rerun the audit evidence that produced the proposal.

## Non-Claims

- `diagnostic_evidence_not_proof`: Diagnostic evidence is not a proof certificate.
- `general_theorem_proving_not_claimed`: This scoped workflow result does not claim general theorem-proving ability.
- `release_readiness_not_claimed`: This scoped workflow result does not claim release readiness.
- `fix_proposal_not_applied_or_verified`: Proposed fixes are diagnostic guidance only; they are not applied edits, proof certificates, or semantic implementation verification.
- `audit_fix_report_not_applied_or_certified`: The audit-and-fix report is diagnostic guidance only; it does not apply edits, verify repaired text, or certify mathematical correctness.
