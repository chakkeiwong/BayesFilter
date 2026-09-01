# Compact Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `43cf6bb7763445b77f8ce677278538b9e6dd4aa6bd1ee51e8d0f1c620de0841f`
Coverage: `partial_coverage`; targets `25`; gaps `7`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `0fe775c40b25c24bc6375387abf188d48615891aedf2581312d96743e49d7369` (1343077 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:incomplete-hermite-gram` | `equality` | `definition` | 1315 |
| `eq:hermite-product` | `equality` | `theorem_proposition` | 1320 |
| `eq:hermite-antiderivative` | `unavailable` | `unsupported_or_ambiguous` | 1330 |
| `eq:incomplete-hermite-closed` | `equality` | `definition` | 1358 |
| `eq:hermite-kr-cdf` | `equality` | `unsupported_or_ambiguous` | 1372 |
| `eq:apf-log-weight` | `equality` | `definition` | 1424 |
| `eq:frozen-apf-scalar` | `equality` | `definition` | 1435 |
| `eq:apf-conditional-mean` | `unavailable` | `unsupported_or_ambiguous` | 1449 |
| `eq:score-recursion` | `unavailable` | `unsupported_or_ambiguous` | 1481 |
| `eq:apf-score` | `unavailable` | `unsupported_or_ambiguous` | 1493 |
| `eq:lyapunov` | `equality` | `definition` | 1539 |
| `eq:lyapunov-dot` | `equality` | `definition` | 1550 |
| `eq:initial-gamma-score` | `equality` | `definition` | 1572 |
| `eq:transition-gamma-score` | `equality` | `definition` | 1589 |
| `eq:observation-xi-score` | `equality` | `definition` | 1597 |
| `eq:coherent-true-target` | `equality` | `definition` | 1655 |
| `eq:coherent-finite-target` | `equality` | `definition` | 1663 |
| `eq:coherent-finite-normalizer` | `equality` | `definition` | 1668 |
| `eq:coherent-initial-target` | `unavailable` | `unsupported_or_ambiguous` | 1672 |
| `eq:coherent-apf-target` | `equality` | `definition` | 1683 |
| `eq:coherent-apf-proposal` | `equality` | `definition` | 1686 |
| `eq:coherent-apf-ratio` | `equality` | `definition` | 1689 |
| `eq:coherent-pullback` | `equality` | `definition` | 1730 |
| `eq:coherent-square-root` | `unavailable` | `unsupported_or_ambiguous` | 1736 |
| `eq:coherent-fitted-normalizer` | `equality` | `theorem_proposition` | 1741 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:incomplete-hermite-gram` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:incomplete-hermite-gram:obligation_1` |
| `eq:hermite-antiderivative` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:hermite-antiderivative:obligation_2, proof_audit_v2:eq:hermite-antiderivative:obligation_3` |
| `eq:score-recursion` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:score-recursion:obligation_1` |
| `eq:initial-gamma-score` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:initial-gamma-score:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:transition-gamma-score` |
| `eq:coherent-initial-target` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:coherent-initial-target:obligation_2` |
| `eq:coherent-apf-proposal` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:coherent-apf-proposal:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:incomplete-hermite-gram`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:hermite-product`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:hermite-antiderivative`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:incomplete-hermite-closed`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:hermite-kr-cdf`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:apf-log-weight`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:frozen-apf-scalar`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:apf-conditional-mean`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:score-recursion`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:apf-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:lyapunov`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:lyapunov-dot`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:initial-gamma-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:transition-gamma-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:observation-xi-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-true-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-finite-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-finite-normalizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-initial-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-apf-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-apf-proposal`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-apf-ratio`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-pullback`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-square-root`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-fitted-normalizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
