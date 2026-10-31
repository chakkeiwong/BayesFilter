# Compact Math Document Rigor Audit

Source: `ledh_covariance_proposal_20261008.tex`
Source SHA-256: `928a5c72def00110b13c741f9e1ff0df6366ee779d6678824f66320b1a704a25`
Coverage: `partial_coverage`; targets `4`; gaps `4`; concrete repairs `4`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `79aac167c90b821fa323d05406ebc7208cd9f8f0e9238263e1f6fe819fe416a7` (404370 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:ukf-cross` | `unavailable` | `unsupported_or_ambiguous` | 202 |
| `eq:cholesky-reset` | `equality` | `definition` | 642 |
| `eq:cayley` | `unavailable` | `unsupported_or_ambiguous` | 731 |
| `eq:cayley-tangent` | `unavailable` | `unsupported_or_ambiguous` | 1062 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:ukf-cross:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:cholesky-reset` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:cayley` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:cayley-tangent:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:ukf-cross`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:cholesky-reset`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:cayley`

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
