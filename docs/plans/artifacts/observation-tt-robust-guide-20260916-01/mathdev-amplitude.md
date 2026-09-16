# Compact Math Document Rigor Audit

Source: `attempt05_observation_aware_tt_algorithm_note.tex`
Source SHA-256: `0bd0fc97ef1fdfa3a405561f1ecade649039506a7aa41bd85381fc14dd6ae19c`
Coverage: `partial_coverage`; targets `1`; gaps `1`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `7255b67ef6943721e0b4cef323e1cf1edca3b05a9ef2951c9b86bc84ce1e03cb` (191172 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:robust-amplitude-scale` | `equality` | `theorem_proposition` | 5588 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:robust-amplitude-scale` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:robust-amplitude-scale:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:robust-amplitude-scale`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
