# Compact Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `43cf6bb7763445b77f8ce677278538b9e6dd4aa6bd1ee51e8d0f1c620de0841f`
Coverage: `partial_coverage`; targets `24`; gaps `9`; concrete repairs `2`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `71be89dd80e779b7170777506d6583d926bcb490c9e63e8a6d4b942c12ed9c06` (1920639 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:obs-log` | `equality` | `unsupported_or_ambiguous` | 78 |
| `eq:actual-initial-map` | `equality` | `definition` | 255 |
| `eq:actual-initial-target` | `equality` | `definition` | 264 |
| `eq:actual-initial-scale` | `unavailable` | `unsupported_or_ambiguous` | 271 |
| `eq:actual-als-local-solve` | `unavailable` | `unsupported_or_ambiguous` | 288 |
| `eq:actual-initial-retention` | `unavailable` | `unsupported_or_ambiguous` | 318 |
| `eq:actual-joint-hint` | `unavailable` | `unsupported_or_ambiguous` | 348 |
| `eq:actual-current-map` | `equality` | `definition` | 358 |
| `eq:actual-previous-map` | `equality` | `definition` | 360 |
| `eq:actual-old-reexpression` | `equality` | `definition` | 368 |
| `eq:actual-conversion` | `equality` | `definition` | 383 |
| `eq:actual-transition-factor` | `equality` | `definition` | 395 |
| `eq:actual-previous-energy` | `equality` | `definition` | 398 |
| `eq:actual-transition-scale` | `unavailable` | `unsupported_or_ambiguous` | 400 |
| `eq:actual-transition-branch-target` | `unavailable` | `unsupported_or_ambiguous` | 411 |
| `eq:actual-transition-branch-closure` | `equality` | `definition` | 422 |
| `eq:actual-gh-nodes` | `unavailable` | `unsupported_or_ambiguous` | 453 |
| `eq:actual-gh-update` | `unavailable` | `unsupported_or_ambiguous` | 462 |
| `eq:actual-gh-prediction` | `unavailable` | `unsupported_or_ambiguous` | 469 |
| `eq:actual-gh-slope` | `equality` | `definition` | 477 |
| `eq:actual-gh-joint` | `unavailable` | `unsupported_or_ambiguous` | 492 |
| `eq:actual-loop-summary` | `unavailable` | `unsupported_or_ambiguous` | 532 |
| `eq:actual-ukf-points` | `unavailable` | `unsupported_or_ambiguous` | 549 |
| `eq:actual-ukf-weights` | `unavailable` | `unsupported_or_ambiguous` | 553 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:actual-initial-map` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-initial-map:obligation_1` |
| `eq:actual-initial-scale` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-initial-scale:obligation_2` |
| `eq:actual-als-local-solve` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-als-local-solve:obligation_1` |
| `eq:actual-initial-retention` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-initial-retention:obligation_2, proof_audit_v2:eq:actual-initial-retention:obligation_3` |
| `eq:actual-previous-map` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-previous-map:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:actual-old-reexpression` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:actual-gh-slope` |
| `eq:actual-gh-joint` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-gh-joint:obligation_4` |
| `eq:actual-ukf-points` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-ukf-points:obligation_6, proof_audit_v2:eq:actual-ukf-points:obligation_4, proof_audit_v2:eq:actual-ukf-points:obligation_5` |

## Role-Specific Obligation Ledger

### `eq:obs-log`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-initial-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-initial-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-initial-scale`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-als-local-solve`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-initial-retention`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-joint-hint`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-current-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-previous-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-old-reexpression`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-conversion`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-transition-factor`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-previous-energy`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-transition-scale`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-transition-branch-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-transition-branch-closure`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-gh-nodes`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-gh-update`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-gh-prediction`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-gh-slope`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-gh-joint`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-loop-summary`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-ukf-points`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-ukf-weights`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
