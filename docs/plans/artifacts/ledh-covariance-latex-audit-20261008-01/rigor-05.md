# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `3`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `eaa63956c8e861442dd788bfd96890d152420351c81a0f87c0304454333e3d77` (363299 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:is-identity` | `unavailable` | `unsupported_or_ambiguous` | 448 |
| `eq:defensive-bound` | `unavailable` | `unsupported_or_ambiguous` | 455 |
| `eq:log-increment` | `unavailable` | `unsupported_or_ambiguous` | 477 |
| `eq:weighted-moments` | `unavailable` | `unsupported_or_ambiguous` | 502 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:is-identity` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:is-identity:obligation_1` |
| `eq:log-increment` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:log-increment:obligation_2` |
| `eq:weighted-moments` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:weighted-moments:obligation_2, proof_audit_v2:eq:weighted-moments:obligation_3, proof_audit_v2:eq:weighted-moments:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:is-identity`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:defensive-bound`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:log-increment`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:weighted-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
