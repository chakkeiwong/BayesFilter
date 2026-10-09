# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `2`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `3e74cb4143a3c0f0718729f4bf04df6a1530434e0467da622a5c44b03589dfd6` (374642 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:barycentric-loss` | `equality` | `definition` | 525 |
| `eq:transform-constraints` | `unavailable` | `unsupported_or_ambiguous` | 537 |
| `eq:riccati-correction` | `equality` | `definition` | 553 |
| `eq:sinkhorn-iterations` | `equality` | `definition` | 573 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:transform-constraints` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:transform-constraints:obligation_2, proof_audit_v2:eq:transform-constraints:obligation_1` |
| `eq:riccati-correction` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:riccati-correction:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:barycentric-loss`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:transform-constraints`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:riccati-correction`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:sinkhorn-iterations`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
