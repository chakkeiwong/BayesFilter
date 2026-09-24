# Math Document Rigor Audit

Target: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`

## Executive Summary

- Coverage status: `partial_coverage`
- Selected targets: 24 / 122 labeled equation rows
- Gaps: 5
- Proposals: 2
- Concrete repairs: 2
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
| `audit_and_propose_fix` | Audit selected labels and propose concrete derivation/evidence repairs. | `no_proposal` | `high_level_workflow_result` | `{"labels": ["eq:actual-transition-factor", "eq:actual-transition-scale", "eq:actual-ukf-joint-update", "eq:actual-ukf-moments", "eq:actual-ukf-points", "eq:actual-ukf-update", "eq:actual-ukf-weights", "eq:actual-unnormalized-retained", "eq:als", "eq:apf-conditional-mean", "eq:apf-log-weight", "eq:apf-score", "eq:branch-energy", "eq:branch-target", "eq:coherent-apf-proposal", "eq:coherent-apf-ratio", "eq:coherent-apf-target", "eq:coherent-defensive-bound", "eq:coherent-dmis-identity", "eq:coherent-dmis-variance", "eq:coherent-error-recursion", "eq:coherent-filter-k", "eq:coherent-filter-operator", "eq:coherent-finite-normalizer"], "reused_exact_evidence": false, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "target_file": "attempt05_n4_failure_analysis.tex", "validate_proposed_fixes": true}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-transition-factor`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-transition-factor", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-transition-scale`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-transition-scale", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-joint-update`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-joint-update", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-moments`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-moments", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-points`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-points", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-update`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-update", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-weights`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-weights", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-unnormalized-retained`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-unnormalized-retained", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:als`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:als", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:apf-conditional-mean`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:apf-conditional-mean", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:apf-log-weight`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:apf-log-weight", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:apf-score`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:apf-score", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:branch-energy`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:branch-energy", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:branch-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:branch-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-apf-proposal`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-apf-proposal", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-apf-ratio`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-apf-ratio", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-apf-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-apf-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-defensive-bound`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-defensive-bound", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-dmis-identity`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-dmis-identity", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-dmis-variance`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-dmis-variance", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-error-recursion`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-error-recursion", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-filter-k`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-filter-k", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-filter-operator`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-filter-operator", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-finite-normalizer`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-finite-normalizer", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | `diagnostic_only` | `high_level_workflow_result` | `{"evidence_count": 24, "question": "Audit selected document labels for mathematical rigor gaps and proposed repairs", "source": {"file": "attempt05_n4_failure_analysis.tex", "labels": ["eq:actual-transition-factor", "eq:actual-transition-scale", "eq:actual-ukf-joint-update", "eq:actual-ukf-moments", "eq:actual-ukf-points", "eq:actual-ukf-update", "eq:actual-ukf-weights", "eq:actual-unnormalized-retained", "eq:als", "eq:apf-conditional-mean", "eq:apf-log-weight", "eq:apf-score", "eq:branch-energy", "eq:branch-target", "eq:coherent-apf-proposal", "eq:coherent-apf-ratio", "eq:coherent-apf-target", "eq:coherent-defensive-bound", "eq:coherent-dmis-identity", "eq:coherent-dmis-variance", "eq:coherent-error-recursion", "eq:coherent-filter-k", "eq:coherent-filter-operator", "eq:coherent-finite-normalizer"], "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745"}}` |
| `validate_proposed_fixes` | Attach deterministic backend-attempt accountability to concrete proposed fixes. | `completed` | `proposal_fix_validation_summary` | `{"backend_order": ["sympy", "lean"], "detail_count": 9, "policy": "require_attempt_when_encodable"}` |

## Concrete Repair Ledger

### 1. `eq:actual-ukf-update`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:actual-ukf-update`

### 2. `eq:coherent-defensive-bound`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `proof_audit_v2:eq:coherent-defensive-bound:obligation_1`


## Diagnostic Abstention Ledger

- No diagnostic abstentions were recorded.

## Gap And Proposal Ledger

### 1. `eq:actual-transition-scale`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-transition-scale > line 400`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-transition-scale:obligation_1`

### 2. `eq:actual-ukf-joint-update`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-joint-update > line 589`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-ukf-joint-update:obligation_1`, `proof_audit_v2:eq:actual-ukf-joint-update:obligation_2`, `proof_audit_v2:eq:actual-ukf-joint-update:obligation_3`, `proof_audit_v2:eq:actual-ukf-joint-update:obligation_4`, `proof_audit_v2:eq:actual-ukf-joint-update:obligation_5`

### 3. `eq:actual-ukf-points`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-points > line 549`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-ukf-points:obligation_6`

### 4. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-update > line 575`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:actual-ukf-update`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 5. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Complete deterministic-mixture importance sampling > eq:coherent-defensive-bound > line 1845`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `proof_audit_v2:eq:coherent-defensive-bound:obligation_1`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

## Non-Claims

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
