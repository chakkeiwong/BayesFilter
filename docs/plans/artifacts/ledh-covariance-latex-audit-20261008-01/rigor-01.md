# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `3`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `67de13e760897ac0eb5a30cf4e45ed86fa820634be2dad9b31e7fe330628ac40` (442691 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:ar1-total` | `equality` | `unsupported_or_ambiguous` | 74 |
| `eq:finite-target` | `unavailable` | `unsupported_or_ambiguous` | 104 |
| `eq:mixture-moments` | `unavailable` | `unsupported_or_ambiguous` | 139 |
| `eq:ukf-observation` | `unavailable` | `unsupported_or_ambiguous` | 176 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:ar1-total` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:ar1-total:obligation_1` |
| `eq:finite-target` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:finite-target:obligation_2` |
| `eq:ukf-observation` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:ukf-observation:obligation_4, proof_audit_v2:eq:ukf-observation:obligation_2, proof_audit_v2:eq:ukf-observation:obligation_3` |

## Role-Specific Obligation Ledger

### `eq:ar1-total`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:finite-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:mixture-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:ukf-observation`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
