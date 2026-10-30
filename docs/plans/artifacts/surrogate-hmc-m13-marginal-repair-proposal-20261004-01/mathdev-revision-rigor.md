# Compact Math Document Rigor Audit

Source: `surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex`
Source SHA-256: `f615d5438eb909eab0374482dc360ed0d7397d558315cca08255126392836a7d`
Coverage: `partial_coverage`; targets `3`; gaps `3`; concrete repairs `3`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `61644c3bd1e22b54dfbfb905eb77988130f495e68cbd677951ce63f8c89afdb2` (286939 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:affine-density` | `unavailable` | `unsupported_or_ambiguous` | 116 |
| `eq:inverse-tangent` | `equality` | `unsupported_or_ambiguous` | 316 |
| `eq:gaussian-score` | `equality` | `unsupported_or_ambiguous` | 336 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:affine-density:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:inverse-tangent` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:gaussian-score` |

## Role-Specific Obligation Ledger

### `eq:affine-density`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:inverse-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:gaussian-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
