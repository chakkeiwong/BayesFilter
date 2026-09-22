# Compact Math Document Rigor Audit

Source: `ledh_younis_kdm_score.tex`
Source SHA-256: `5a7562526417f1fd54d285d6d592b5243eff4ed69bcfbb2de7addf9fe465736c`
Coverage: `partial_coverage`; targets `2`; gaps `2`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `4043e14c4af6ff87ba80d864e9528505f6dc8e345531a6081c8d095c457f0ee7` (245731 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:fd-cubic-weights` | `unavailable` | `unsupported_or_ambiguous` | 1605 |
| `eq:fd-mse` | `equality` | `theorem_proposition` | 1640 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:fd-cubic-weights` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:fd-cubic-weights:obligation_2` |
| `eq:fd-mse` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:fd-mse:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:fd-cubic-weights`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:fd-mse`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
