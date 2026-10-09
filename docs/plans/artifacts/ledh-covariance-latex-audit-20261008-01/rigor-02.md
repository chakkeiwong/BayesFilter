# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `4`; concrete repairs `3`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `ceda834e5062989ad5aacfd4019ea44fea3322719f6b629494b337edbca3a0d1` (450224 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:ukf-cross` | `unavailable` | `unsupported_or_ambiguous` | 180 |
| `eq:ukf-update` | `unavailable` | `unsupported_or_ambiguous` | 189 |
| `eq:global-map` | `unavailable` | `unsupported_or_ambiguous` | 207 |
| `eq:global-covariance` | `unavailable` | `unsupported_or_ambiguous` | 221 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:ukf-cross:obligation_1` |
| `eq:ukf-update` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:ukf-update:obligation_2, proof_audit_v2:eq:ukf-update:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:global-map` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:global-covariance:obligation_2, proof_audit_v2:eq:global-covariance:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:ukf-cross`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:ukf-update`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:global-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:global-covariance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
