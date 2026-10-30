# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `4`; concrete repairs `2`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `034265d9d45c370edcffc8d46cd946c892c1a5bf4245ad7340cedd305db4ff8b` (399087 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:global-tangent` | `unavailable` | `unsupported_or_ambiguous` | 937 |
| `eq:weight-tangent` | `unavailable` | `unsupported_or_ambiguous` | 958 |
| `eq:weighted-moment-tangent` | `equality` | `unsupported_or_ambiguous` | 963 |
| `eq:cayley-tangent` | `unavailable` | `unsupported_or_ambiguous` | 1009 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:global-tangent:obligation_2` |
| `eq:weight-tangent` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:weight-tangent:obligation_1` |
| `eq:weighted-moment-tangent` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:weighted-moment-tangent:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:cayley-tangent:obligation_1, proof_audit_v2:eq:cayley-tangent:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:global-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:weight-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:weighted-moment-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:cayley-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
