# Compact Math Document Rigor Audit

Source: `attempt05_observation_aware_tt_algorithm_note.tex`
Source SHA-256: `1b8cba0c5ac6f37322e78a0115e304ef2275531c50ce1ff2927a5de3f3284be2`
Coverage: `partial_coverage`; targets `11`; gaps `4`; concrete repairs `2`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `203719c168f657e5e005dab8e51ab0bbc872789536601c8be313d769c03ff125` (639844 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:robust-signed-weights` | `unavailable` | `unsupported_or_ambiguous` | 5163 |
| `eq:robust-margin` | `unavailable` | `unsupported_or_ambiguous` | 5224 |
| `eq:robust-sv-potential` | `unavailable` | `unsupported_or_ambiguous` | 5254 |
| `eq:robust-mode-derivatives` | `equality` | `theorem_proposition` | 5269 |
| `eq:robust-change-measure` | `equality` | `theorem_proposition` | 5302 |
| `eq:robust-chart` | `unavailable` | `unsupported_or_ambiguous` | 5342 |
| `eq:robust-initializer-joint` | `unavailable` | `unsupported_or_ambiguous` | 5388 |
| `eq:robust-mixture` | `unavailable` | `unsupported_or_ambiguous` | 5405 |
| `eq:robust-importance` | `equality` | `theorem_proposition` | 5424 |
| `eq:robust-second-moment` | `unavailable` | `unsupported_or_ambiguous` | 5450 |
| `eq:robust-mixture-score` | `unavailable` | `unsupported_or_ambiguous` | 5496 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:robust-signed-weights` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:robust-signed-weights:obligation_1, proof_audit_v2:eq:robust-signed-weights:obligation_3` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:robust-sv-potential:obligation_2` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:robust-mode-derivatives` |
| `eq:robust-chart` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:robust-chart:obligation_1, proof_audit_v2:eq:robust-chart:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:robust-signed-weights`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-margin`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-sv-potential`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-mode-derivatives`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-change-measure`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-chart`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-initializer-joint`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-mixture`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-importance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-second-moment`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:robust-mixture-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
