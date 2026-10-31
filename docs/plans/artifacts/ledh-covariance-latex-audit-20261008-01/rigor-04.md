# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `3`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `93c75bcbcbfa845ac10e67dd59f22ac976a366262ed1a920b979e5f573eb25f1` (371623 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:affine-composition` | `unavailable` | `unsupported_or_ambiguous` | 350 |
| `eq:sv-score` | `equality` | `unsupported_or_ambiguous` | 378 |
| `eq:mapped-density` | `unavailable` | `unsupported_or_ambiguous` | 404 |
| `eq:importance-ratio` | `unavailable` | `unsupported_or_ambiguous` | 435 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:affine-composition` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:affine-composition:obligation_1, proof_audit_v2:eq:affine-composition:obligation_2, proof_audit_v2:eq:affine-composition:obligation_3` |
| `eq:sv-score` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:sv-score:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:mapped-density` |

## Role-Specific Obligation Ledger

### `eq:affine-composition`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:sv-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:mapped-density`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:importance-ratio`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
