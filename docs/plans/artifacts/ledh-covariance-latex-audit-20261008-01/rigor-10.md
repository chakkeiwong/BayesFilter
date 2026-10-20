# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `4`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `ecc87afbdd337ede0f7d2fe65efd479a6cd49df2a44b258625cc12897a017454` (325564 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:total-score` | `unavailable` | `unsupported_or_ambiguous` | 837 |
| `eq:responsibilities` | `unavailable` | `unsupported_or_ambiguous` | 852 |
| `eq:mixture-differentials` | `unavailable` | `unsupported_or_ambiguous` | 860 |
| `eq:inverse-tangent` | `unavailable` | `unsupported_or_ambiguous` | 872 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:total-score` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:total-score:obligation_2` |
| `eq:responsibilities` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:responsibilities:obligation_1` |
| `eq:mixture-differentials` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:mixture-differentials:obligation_2` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:inverse-tangent:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:total-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:responsibilities`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:mixture-differentials`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:inverse-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
