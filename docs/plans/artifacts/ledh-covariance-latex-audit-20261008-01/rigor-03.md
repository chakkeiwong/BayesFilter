# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `3`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `f9c5848b9043e84ac4379bafc551563b92b9178bd5b6ec2d0bf213df2b05d06d` (353149 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:quadratic-guide` | `unavailable` | `unsupported_or_ambiguous` | 278 |
| `eq:bridge-moments` | `unavailable` | `unsupported_or_ambiguous` | 310 |
| `eq:bridge-derivatives` | `unavailable` | `unsupported_or_ambiguous` | 319 |
| `eq:local-ode` | `unavailable` | `unsupported_or_ambiguous` | 329 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:quadratic-guide` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:quadratic-guide:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:bridge-moments:obligation_2` |
| `eq:local-ode` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:local-ode:obligation_3` |

## Role-Specific Obligation Ledger

### `eq:quadratic-guide`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:bridge-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:bridge-derivatives`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:local-ode`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
