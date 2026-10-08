# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `2ee36c54a341e3f020a993e8d87705b9361de031aecbded7e7b075b6ba89fa9f`
Coverage: `partial_coverage`; targets `4`; gaps `2`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `f7d0c36c384cfdb0b2cf2ca4554d0d58d2cebf42c616d7dc24cf26f858fd3587` (364158 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:bounded-skew` | `unavailable` | `unsupported_or_ambiguous` | 734 |
| `eq:moment-objective` | `unavailable` | `unsupported_or_ambiguous` | 760 |
| `eq:fixed-optimizer` | `unavailable` | `unsupported_or_ambiguous` | 778 |
| `eq:weak-displacement` | `unavailable` | `unsupported_or_ambiguous` | 803 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:moment-objective:obligation_3, proof_audit_v2:eq:moment-objective:obligation_1, proof_audit_v2:eq:moment-objective:obligation_2` |
| `eq:weak-displacement` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:weak-displacement:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:bounded-skew`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:moment-objective`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:fixed-optimizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:weak-displacement`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
