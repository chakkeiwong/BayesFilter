# Math Document Rigor Audit

Target: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`

## Executive Summary

- Coverage status: `partial_coverage`
- Selected targets: 25 / 122 labeled equation rows
- Gaps: 6
- Proposals: 3
- Concrete repairs: 3
- Diagnostic abstentions: 0
- This report is diagnostic and proposal-oriented; it is not a proof of the document.

## Backend Provenance

- Active Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`
- LeanDojo status: `available`
- LeanDojo environment scope: `backend_python`
- LeanDojo backend env: `mathdevmcp-backends`
- Certification boundary: LeanDojo availability is proof-search evidence only. A proof is certified only by direct Lean checking with no placeholders, or by another certifying backend under the scoped contract.

## Document Inventory

- Lines: 2665
- Sections: 59
- Equation rows: 174
- Labeled equation rows: 122
- Duplicate labels: 0
- Missing refs: 0

## Tool Uses

| Tool | Purpose | Status | Output contract | Arguments |
| --- | --- | --- | --- | --- |
| `locate_equations_in_file` | Localize display equations in the exact target file. | `completed` | `equation_rows` | `{"root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "tex_path": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex"}` |
| `summarize_equation_localization` | Summarize equation localization uncertainty. | `completed` | `equation_localization_summary` | `{"row_count": 174}` |
| `doctor_report` | Record active/backend Python and external backend capability provenance. | `available` | `doctor_report` | `{}` |
| `lean_readiness` | Record direct Lean, Lake, and LeanDojo readiness without promoting readiness to proof. | `ready_with_caveats` | `lean_readiness` | `{"root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05"}` |
| `audit_and_propose_fix` | Audit selected labels and propose concrete derivation/evidence repairs. | `no_proposal` | `high_level_workflow_result` | `{"labels": ["eq:coherent-rbf-factor", "eq:coherent-rbf-gram", "eq:coherent-retained-marginal", "eq:coherent-square-root", "eq:coherent-student-density", "eq:coherent-student-scale", "eq:coherent-sv-bound", "eq:coherent-sv-envelope", "eq:coherent-target-normalizer", "eq:coherent-total-weight-gradient", "eq:coherent-true-target", "eq:corr-increment", "eq:count-rms", "eq:decomp", "eq:frozen-apf-scalar", "eq:generic-affine-contract", "eq:generic-complete-mixture", "eq:generic-cv-estimator", "eq:generic-cv-tangent", "eq:generic-cv-unbiased", "eq:generic-cv-variance", "eq:generic-defensive-bound", "eq:generic-dmis-expectation", "eq:generic-finite-cv", "eq:generic-finite-estimator"], "reused_exact_evidence": false, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "target_file": "attempt05_n4_failure_analysis.tex", "validate_proposed_fixes": true}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-rbf-factor`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-rbf-factor", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-rbf-gram`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-rbf-gram", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-retained-marginal`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-retained-marginal", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-square-root`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-square-root", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-student-density`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-student-density", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-student-scale`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-student-scale", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-sv-bound`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-sv-bound", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-sv-envelope`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-sv-envelope", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-target-normalizer`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-target-normalizer", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-total-weight-gradient`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-total-weight-gradient", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-true-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-true-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:corr-increment`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:corr-increment", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:count-rms`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:count-rms", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:decomp`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:decomp", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:frozen-apf-scalar`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:frozen-apf-scalar", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-affine-contract`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-affine-contract", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-complete-mixture`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-complete-mixture", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-cv-estimator`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-cv-estimator", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-cv-tangent`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-cv-tangent", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-cv-unbiased`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-cv-unbiased", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-cv-variance`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-cv-variance", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-defensive-bound`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-defensive-bound", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-dmis-expectation`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-dmis-expectation", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-finite-cv`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-finite-cv", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-finite-estimator`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-finite-estimator", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | `diagnostic_only` | `high_level_workflow_result` | `{"evidence_count": 25, "question": "Audit selected document labels for mathematical rigor gaps and proposed repairs", "source": {"file": "attempt05_n4_failure_analysis.tex", "labels": ["eq:coherent-rbf-factor", "eq:coherent-rbf-gram", "eq:coherent-retained-marginal", "eq:coherent-square-root", "eq:coherent-student-density", "eq:coherent-student-scale", "eq:coherent-sv-bound", "eq:coherent-sv-envelope", "eq:coherent-target-normalizer", "eq:coherent-total-weight-gradient", "eq:coherent-true-target", "eq:corr-increment", "eq:count-rms", "eq:decomp", "eq:frozen-apf-scalar", "eq:generic-affine-contract", "eq:generic-complete-mixture", "eq:generic-cv-estimator", "eq:generic-cv-tangent", "eq:generic-cv-unbiased", "eq:generic-cv-variance", "eq:generic-defensive-bound", "eq:generic-dmis-expectation", "eq:generic-finite-cv", "eq:generic-finite-estimator"], "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745"}}` |
| `validate_proposed_fixes` | Attach deterministic backend-attempt accountability to concrete proposed fixes. | `completed` | `proposal_fix_validation_summary` | `{"backend_order": ["sympy", "lean"], "detail_count": 4, "policy": "require_attempt_when_encodable"}` |

## Concrete Repair Ledger

### 1. `eq:coherent-student-density`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `proof_audit_v2:eq:coherent-student-density:obligation_1`

### 2. `eq:generic-affine-contract`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:generic-affine-contract`

### 3. `eq:generic-defensive-bound`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:generic-defensive-bound`


## Diagnostic Abstention Ledger

- No diagnostic abstentions were recorded.

## Gap And Proposal Ledger

### 1. `eq:coherent-rbf-factor`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Basis ladder and reference-measure boundary > eq:coherent-rbf-factor > line 2327`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:coherent-rbf-factor:obligation_1`

### 2. `eq:coherent-retained-marginal`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Recursive moment-derived coordinates > eq:coherent-retained-marginal > line 2139`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:coherent-retained-marginal:obligation_2`

### 3. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Complete deterministic-mixture importance sampling > eq:coherent-student-density > line 1804`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `proof_audit_v2:eq:coherent-student-density:obligation_1`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 4. `eq:coherent-total-weight-gradient`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Analytical-gradient contract for the combined program > eq:coherent-total-weight-gradient > line 2409`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:coherent-total-weight-gradient:obligation_1`

### 5. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Executable recursive construction > eq:generic-affine-contract > line 2092`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:generic-affine-contract`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 6. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-defensive-bound > line 1973`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:generic-defensive-bound`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

## Non-Claims

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
