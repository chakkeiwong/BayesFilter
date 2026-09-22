# Math Document Rigor Audit

Target: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`

## Executive Summary

- Coverage status: `partial_coverage`
- Selected targets: 30 / 122 labeled equation rows
- Gaps: 10
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
| `audit_and_propose_fix` | Audit selected labels and propose concrete derivation/evidence repairs. | `no_proposal` | `high_level_workflow_result` | `{"labels": ["eq:obs-log", "eq:actual-initial-map", "eq:actual-initial-target", "eq:actual-initial-scale", "eq:actual-als-local-solve", "eq:actual-initial-retention", "eq:actual-joint-hint", "eq:actual-current-map", "eq:actual-previous-map", "eq:actual-old-reexpression", "eq:actual-conversion", "eq:actual-transition-factor", "eq:actual-previous-energy", "eq:actual-transition-scale", "eq:actual-transition-branch-target", "eq:actual-transition-branch-closure", "eq:actual-gh-nodes", "eq:actual-gh-update", "eq:actual-gh-prediction", "eq:actual-gh-slope", "eq:actual-gh-joint", "eq:actual-loop-summary", "eq:actual-ukf-points", "eq:actual-ukf-weights", "eq:actual-ukf-moments", "eq:actual-ukf-update", "eq:actual-ukf-joint-update", "eq:actual-raw-ukf-zero-gain", "eq:actual-log-square-guide", "eq:actual-unnormalized-retained"], "reused_exact_evidence": false, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "target_file": "attempt05_n4_failure_analysis.tex", "validate_proposed_fixes": true}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:obs-log`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:obs-log", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-initial-map`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-initial-map", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-initial-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-initial-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-initial-scale`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-initial-scale", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-als-local-solve`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-als-local-solve", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-initial-retention`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-initial-retention", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-joint-hint`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-joint-hint", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-current-map`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-current-map", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-previous-map`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-previous-map", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-old-reexpression`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-old-reexpression", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-conversion`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-conversion", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-transition-factor`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-transition-factor", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-previous-energy`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-previous-energy", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-transition-scale`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-transition-scale", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-transition-branch-target`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-transition-branch-target", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-transition-branch-closure`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-transition-branch-closure", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-gh-nodes`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-gh-nodes", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-gh-update`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-gh-update", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-gh-prediction`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-gh-prediction", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-gh-slope`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-gh-slope", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-gh-joint`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-gh-joint", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-loop-summary`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-loop-summary", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-points`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-points", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-weights`. | `inconclusive` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-weights", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-moments`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-moments", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-update`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-update", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-ukf-joint-update`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-ukf-joint-update", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-raw-ukf-zero-gain`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-raw-ukf-zero-gain", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-log-square-guide`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-log-square-guide", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `audit_derivation_v2_label` | Generate local derivation audit evidence for `eq:actual-unnormalized-retained`. | `unverified` | `proof_audit_v2_result` | `{"after": 1, "backend": "sympy", "before": 4, "file": "attempt05_n4_failure_analysis.tex", "label": "eq:actual-unnormalized-retained", "paragraph_context": true, "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745", "summary_only": true, "task_context": "symbolic_exposition"}` |
| `propose_fix` | Translate audit evidence into conservative repair proposals. | `diagnostic_only` | `high_level_workflow_result` | `{"evidence_count": 30, "question": "Audit selected document labels for mathematical rigor gaps and proposed repairs", "source": {"file": "attempt05_n4_failure_analysis.tex", "labels": ["eq:obs-log", "eq:actual-initial-map", "eq:actual-initial-target", "eq:actual-initial-scale", "eq:actual-als-local-solve", "eq:actual-initial-retention", "eq:actual-joint-hint", "eq:actual-current-map", "eq:actual-previous-map", "eq:actual-old-reexpression", "eq:actual-conversion", "eq:actual-transition-factor", "eq:actual-previous-energy", "eq:actual-transition-scale", "eq:actual-transition-branch-target", "eq:actual-transition-branch-closure", "eq:actual-gh-nodes", "eq:actual-gh-update", "eq:actual-gh-prediction", "eq:actual-gh-slope", "eq:actual-gh-joint", "eq:actual-loop-summary", "eq:actual-ukf-points", "eq:actual-ukf-weights", "eq:actual-ukf-moments", "eq:actual-ukf-update", "eq:actual-ukf-joint-update", "eq:actual-raw-ukf-zero-gain", "eq:actual-log-square-guide", "eq:actual-unnormalized-retained"], "root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "source_digest": "c1cbef24192b9d5fae8ba858a2c257538ee11917d464b2a5f67a418aab5cf745"}}` |
| `validate_proposed_fixes` | Attach deterministic backend-attempt accountability to concrete proposed fixes. | `completed` | `proposal_fix_validation_summary` | `{"backend_order": ["sympy", "lean"], "detail_count": 9, "policy": "require_attempt_when_encodable"}` |

## Concrete Repair Ledger

### 1. `eq:actual-old-reexpression`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:actual-old-reexpression`

### 2. `eq:actual-gh-slope`

- Location: ``
- Problem: 
- Why mathematically problematic: 
- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Assumption statement:

```latex
State a condition ensuring that the displayed inverse operand is invertible.
```
- Backend evidence: `not_requested_for_exposition_patch` - The patch states a standard sufficient condition; it is not a proof certificate.
- Evidence refs: `document_exposition:eq:actual-gh-slope`

### 3. `eq:actual-ukf-update`

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


## Diagnostic Abstention Ledger

- No diagnostic abstentions were recorded.

## Gap And Proposal Ledger

### 1. `eq:actual-initial-map`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > The initial step $t=0$ > eq:actual-initial-map > line 255`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-initial-map:obligation_1`

### 2. `eq:actual-initial-scale`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > The initial step $t=0$ > eq:actual-initial-scale > line 271`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-initial-scale:obligation_2`

### 3. `eq:actual-als-local-solve`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > The initial step $t=0$ > eq:actual-als-local-solve > line 288`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-als-local-solve:obligation_1`

### 4. `eq:actual-initial-retention`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > The initial step $t=0$ > eq:actual-initial-retention > line 318`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-initial-retention:obligation_2`, `proof_audit_v2:eq:actual-initial-retention:obligation_3`

### 5. `eq:actual-previous-map`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-previous-map > line 360`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-previous-map:obligation_1`

### 6. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > A transition step $t\geq1$ > eq:actual-old-reexpression > line 368`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:actual-old-reexpression`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 7. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > How the current GH9 hint is produced > eq:actual-gh-slope > line 477`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:actual-gh-slope`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

### 8. `eq:actual-gh-joint`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > How the current GH9 hint is produced > eq:actual-gh-joint > line 492`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-gh-joint:obligation_4`

### 9. `eq:actual-ukf-points`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-points > line 549`
- Classification: `diagnostic_abstention`
- Problem: The equation role or derivation remains unresolved.
- Why mathematically problematic: The source does not yet distinguish a definition, assumption, identity, or derived claim clearly enough for routing.
- Evidence refs: `proof_audit_v2:eq:actual-ukf-points:obligation_6`, `proof_audit_v2:eq:actual-ukf-points:obligation_4`, `proof_audit_v2:eq:actual-ukf-points:obligation_5`

### 10. `invertibility_required`

- Location: `attempt05_n4_failure_analysis.tex > The Actual Program: A Time-Step Trace > What a UKF guide would do > eq:actual-ukf-update > line 575`
- Classification: `concrete_repair`
- Problem: The displayed inverse/series lacks required local exposition conditions.
- Why mathematically problematic: Inverse notation requires an invertible operand. The displayed Neumann series additionally requires a convergence condition; positive definiteness is only one structured sufficient condition.
- Evidence refs: `document_exposition:eq:actual-ukf-update`

- Proposed fix: State a condition ensuring that the displayed inverse operand is invertible.
- Backend evidence: `not_requested_for_exposition_patch`

## Non-Claims

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
