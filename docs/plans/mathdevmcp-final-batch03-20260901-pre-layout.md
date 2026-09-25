# Math Document Rigor Audit

Target: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`

## Executive Summary

- Coverage status: `partial_coverage`
- Selected targets: 25 / 122 labeled equation rows
- Gaps: 3
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
| `audit_and_propose_fix` | Audit selected labels and propose concrete derivation/evidence repairs. | `no_proposal` | `high_level_workflow_result` | `{"labels": ["eq:coherent-finite-target", "eq:coherent-fitted-normalizer", "eq:coherent-floor-normalizer", "eq:coherent-frozen-gradient", "eq:coherent-gaussian-density", "eq:coherent-hermite-ladder", "eq:coherent-initial-pullback", "eq:coherent-initial-target", "eq:coherent-joint-covariance", "eq:coherent-linear-moments", "eq:coherent-linear-transition", "eq:coherent-lipschitz", "eq:coherent-map", "eq:coherent-mixture-density", "eq:coherent-moment-cov", "eq:coherent-moment-g0", "eq:coherent-moment-g1", "eq:coherent-moment-g2", "eq:coherent-moment-mean", "eq:coherent-moment-z", "eq:coherent-nonlinear-moments", "eq:coherent-normalizer-error", "eq:coherent-product-student-reference", "eq:coherent-pullback", "eq:coherent-pullback-integral"], "reused_exact_evidence": false, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "target_file": "attempt05_n4_failure_analysis.tex", "validate_proposed_fixes": true}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-finite-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-finite-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-fitted-normalizer`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-fitted-normalizer", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-floor-normalizer`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-floor-normalizer", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-frozen-gradient`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-frozen-gradient", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-gaussian-density`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-gaussian-density", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-hermite-ladder`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-hermite-ladder", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-initial-pullback`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-initial-pullback", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-initial-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-initial-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-joint-covariance`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-joint-covariance", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-linear-moments`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-linear-moments", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-linear-transition`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-linear-transition", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-lipschitz`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-lipschitz", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-map`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-map", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-mixture-density`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-mixture-density", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-moment-cov`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-moment-cov", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-moment-g0`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-moment-g0", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-moment-g1`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-moment-g1", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-moment-g2`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-moment-g2", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-moment-mean`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-moment-mean", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-moment-z`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-moment-z", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-nonlinear-moments`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-nonlinear-moments", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-normalizer-error`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-normalizer-error", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-product-student-reference`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-product-student-reference", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-pullback`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-pullback", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:coherent-pullback-integral`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:coherent-pullback-integral", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | `diagnostic_only` | `high_level_workflow_result` | `{"evidence_count": 25, "question": "Audit selected document labels for mathematical rigor gaps and proposed repairs", "source": {"file": "attempt05_n4_failure_analysis.tex", "labels": ["eq:coherent-finite-target", "eq:coherent-fitted-normalizer", "eq:coherent-floor-normalizer", "eq:coherent-frozen-gradient", "eq:coherent-gaussian-density", "eq:coherent-hermite-ladder", "eq:coherent-initial-pullback", "eq:coherent-initial-target", "eq:coherent-joint-covariance", "eq:coherent-linear-moments", "eq:coherent-linear-transition", "eq:coherent-lipschitz", "eq:coherent-map", "eq:coherent-mixture-density", "eq:coherent-moment-cov", "eq:coherent-moment-g0", "eq:coherent-moment-g1", "eq:coherent-moment-g2", "eq:coherent-moment-mean", "eq:coherent-moment-z", "eq:coherent-nonlinear-moments", "eq:coherent-normalizer-error", "eq:coherent-product-student-reference", "eq:coherent-pullback", "eq:coherent-pullback-integral"], "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745"}}` |
| `validate_proposed_fixes` | Attach deterministic backend-attempt accountability to concrete proposed fixes. | `completed` | `proposal_fix_validation_summary` | `{"backend_order": ["sympy", "lean"], "detail_count": 6, "policy": "require_attempt_when_encodable"}` |

## Concrete Repair Ledger

### 1. `eq:coherent-gaussian-density`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `proof_audit_v2:eq:coherent-gaussian-density:obligation_1`

### 2. `eq:coherent-map`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:coherent-map`


## Diagnostic Abstention Ledger

- No diagnostic abstentions were recorded.

## Gap And Proposal Ledger

### 1. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Complete deterministic-mixture importance sampling > eq:coherent-gaussian-density > line 1800`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `proof_audit_v2:eq:coherent-gaussian-density:obligation_1`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 2. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Recursive moment-derived coordinates > eq:coherent-map > line 2204`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:coherent-map`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 3. `eq:coherent-moment-cov`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > Recursive moment-derived coordinates > eq:coherent-moment-cov > line 2161`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:coherent-moment-cov:obligation_1`

## Non-Claims

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
