# Compact Math Document Rigor Audit

Source: `surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex`
Source SHA-256: `af8af3a90b834832a28ade6ede21b95e4548d7fc1c090833837ba5fa154a6a87`
Coverage: `partial_coverage`; targets `11`; gaps `5`; concrete repairs `3`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `e8d586b1f456600b52cca8f288c4f9457ff57fd0f4cdb5b2a1240a0194e7abaf` (667868 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:affine-density` | `unavailable` | `unsupported_or_ambiguous` | 112 |
| `eq:integration-proof` | `unavailable` | `unsupported_or_ambiguous` | 168 |
| `eq:rb-proof` | `unavailable` | `unsupported_or_ambiguous` | 198 |
| `eq:variance-decomposition` | `equality` | `theorem_proposition` | 206 |
| `eq:two-component-square` | `equality` | `theorem_proposition` | 212 |
| `eq:counterexample` | `unavailable` | `unsupported_or_ambiguous` | 241 |
| `eq:mixture-score` | `equality` | `definition` | 278 |
| `eq:inverse-tangent` | `equality` | `unsupported_or_ambiguous` | 310 |
| `eq:gaussian-score` | `equality` | `unsupported_or_ambiguous` | 328 |
| `eq:defensive-bound` | `equality` | `unsupported_or_ambiguous` | 381 |
| `eq:defensive-variance` | `unavailable` | `unsupported_or_ambiguous` | 389 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:affine-density:obligation_1` |
| `eq:integration-proof` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:integration-proof:obligation_1` |
| `eq:rb-proof` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:rb-proof:obligation_3, proof_audit_v2:eq:rb-proof:obligation_2` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:inverse-tangent` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:gaussian-score:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:affine-density`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:integration-proof`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:rb-proof`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:variance-decomposition`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:two-component-square`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:counterexample`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:mixture-score`

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

### `eq:defensive-bound`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:defensive-variance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
