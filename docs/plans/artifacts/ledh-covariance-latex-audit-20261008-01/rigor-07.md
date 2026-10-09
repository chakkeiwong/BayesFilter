# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `4`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `4072b180eff1181fd828fdbcfb2de6bbf1f4de36eef1e4de25154d3c40500bd1` (560113 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:actual-barycentres` | `unavailable` | `unsupported_or_ambiguous` | 589 |
| `eq:cholesky-reset` | `equality` | `theorem_proposition` | 602 |
| `eq:reset-covariance` | `unavailable` | `unsupported_or_ambiguous` | 611 |
| `eq:orthogonal-repair` | `unavailable` | `unsupported_or_ambiguous` | 657 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:actual-barycentres` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-barycentres:obligation_4, proof_audit_v2:eq:actual-barycentres:obligation_1, proof_audit_v2:eq:actual-barycentres:obligation_2, proof_audit_v2:eq:actual-barycentres:obligation_3` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:cholesky-reset` |
| `eq:reset-covariance` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:reset-covariance:obligation_1` |
| `eq:orthogonal-repair` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:orthogonal-repair:obligation_1, proof_audit_v2:eq:orthogonal-repair:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:actual-barycentres`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:cholesky-reset`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:reset-covariance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:orthogonal-repair`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
