# Math Document Rigor Audit

Target: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`

## Executive Summary

- Coverage status: `partial_coverage`
- Selected targets: 24 / 122 labeled equation rows
- Gaps: 6
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
| `audit_and_propose_fix` | Audit selected labels and propose concrete derivation/evidence repairs. | `no_proposal` | `high_level_workflow_result` | `{"labels": ["eq:generic-log-cv-tangent", "eq:generic-mixture-responsibility", "eq:generic-recursive-program", "eq:hermite-antiderivative", "eq:hermite-kr-cdf", "eq:hermite-product", "eq:incomplete-hermite-closed", "eq:incomplete-hermite-gram", "eq:initial-gamma-score", "eq:is-identity", "eq:l2-bound", "eq:lyapunov", "eq:lyapunov-dot", "eq:obs-log", "eq:observation-xi-score", "eq:proposal-map", "eq:q-normalizer", "eq:q-physical", "eq:rms", "eq:score-recursion", "eq:shift-invariance", "eq:telescope", "eq:transition-gamma-score", "eq:zh"], "reused_exact_evidence": false, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "target_file": "attempt05_n4_failure_analysis.tex", "validate_proposed_fixes": true}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-log-cv-tangent`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-log-cv-tangent", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-mixture-responsibility`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-mixture-responsibility", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:generic-recursive-program`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:generic-recursive-program", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:hermite-antiderivative`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:hermite-antiderivative", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:hermite-kr-cdf`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:hermite-kr-cdf", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:hermite-product`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:hermite-product", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:incomplete-hermite-closed`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:incomplete-hermite-closed", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:incomplete-hermite-gram`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:incomplete-hermite-gram", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:initial-gamma-score`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:initial-gamma-score", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:is-identity`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:is-identity", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:l2-bound`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:l2-bound", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:lyapunov`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:lyapunov", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:lyapunov-dot`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:lyapunov-dot", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:obs-log`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:obs-log", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:observation-xi-score`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:observation-xi-score", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:proposal-map`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:proposal-map", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:q-normalizer`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:q-normalizer", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:q-physical`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:q-physical", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:rms`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:rms", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:score-recursion`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:score-recursion", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:shift-invariance`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:shift-invariance", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:telescope`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:telescope", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:transition-gamma-score`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:transition-gamma-score", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:zh`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:zh", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | `diagnostic_only` | `high_level_workflow_result` | `{"evidence_count": 24, "question": "Audit selected document labels for mathematical rigor gaps and proposed repairs", "source": {"file": "attempt05_n4_failure_analysis.tex", "labels": ["eq:generic-log-cv-tangent", "eq:generic-mixture-responsibility", "eq:generic-recursive-program", "eq:hermite-antiderivative", "eq:hermite-kr-cdf", "eq:hermite-product", "eq:incomplete-hermite-closed", "eq:incomplete-hermite-gram", "eq:initial-gamma-score", "eq:is-identity", "eq:l2-bound", "eq:lyapunov", "eq:lyapunov-dot", "eq:obs-log", "eq:observation-xi-score", "eq:proposal-map", "eq:q-normalizer", "eq:q-physical", "eq:rms", "eq:score-recursion", "eq:shift-invariance", "eq:telescope", "eq:transition-gamma-score", "eq:zh"], "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745"}}` |
| `validate_proposed_fixes` | Attach deterministic backend-attempt accountability to concrete proposed fixes. | `completed` | `proposal_fix_validation_summary` | `{"backend_order": ["sympy", "lean"], "detail_count": 5, "policy": "require_attempt_when_encodable"}` |

## Concrete Repair Ledger

### 1. `eq:q-physical`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `proof_audit_v2:eq:q-physical:obligation_1`

### 2. `eq:transition-gamma-score`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:transition-gamma-score`


## Diagnostic Abstention Ledger

- No diagnostic abstentions were recorded.

## Gap And Proposal Ledger

### 1. `eq:generic-log-cv-tangent`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-log-cv-tangent > line 2016`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:generic-log-cv-tangent:obligation_1`

### 2. `eq:generic-recursive-program`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Executable recursive construction > eq:generic-recursive-program > line 2069`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:generic-recursive-program:obligation_1`

### 3. `eq:hermite-antiderivative`

- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > Exact Gaussian--Hermite conditional CDFs > eq:hermite-antiderivative > line 1330`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:hermite-antiderivative:obligation_2`, `proof_audit_v2:eq:hermite-antiderivative:obligation_3`

### 4. `eq:initial-gamma-score`

- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > Manual score for the C2 SV parameterization > eq:initial-gamma-score > line 1572`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:initial-gamma-score:obligation_1`

### 5. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > The retained marginal as a proposal > eq:q-physical > line 1245`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `proof_audit_v2:eq:q-physical:obligation_1`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 6. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > Frozen-TT Proposal Correction with an Analytical Score > Manual score for the C2 SV parameterization > eq:transition-gamma-score > line 1589`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:transition-gamma-score`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

## Non-Claims

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
