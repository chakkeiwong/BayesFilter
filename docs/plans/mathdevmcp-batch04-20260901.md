# Compact Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `43cf6bb7763445b77f8ce677278538b9e6dd4aa6bd1ee51e8d0f1c620de0841f`
Coverage: `partial_coverage`; targets `25`; gaps `5`; concrete repairs `5`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `92c161f38dcb8c0669697158fec53b2cba1735180f438fa5aa18f3003151e02b` (1107260 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:coherent-target-normalizer` | `equality` | `theorem_proposition` | 1745 |
| `eq:coherent-pullback-integral` | `unavailable` | `unsupported_or_ambiguous` | 1753 |
| `eq:coherent-initial-pullback` | `unavailable` | `unsupported_or_ambiguous` | 1774 |
| `eq:coherent-floor-normalizer` | `unavailable` | `unsupported_or_ambiguous` | 1783 |
| `eq:coherent-gaussian-density` | `equality` | `unsupported_or_ambiguous` | 1796 |
| `eq:coherent-student-density` | `equality` | `unsupported_or_ambiguous` | 1800 |
| `eq:coherent-student-scale` | `equality` | `unsupported_or_ambiguous` | 1808 |
| `eq:coherent-mixture-density` | `unavailable` | `unsupported_or_ambiguous` | 1816 |
| `eq:coherent-dmis-identity` | `unavailable` | `unsupported_or_ambiguous` | 1829 |
| `eq:coherent-dmis-variance` | `equality` | `theorem_proposition` | 1834 |
| `eq:coherent-defensive-bound` | `unavailable` | `unsupported_or_ambiguous` | 1841 |
| `eq:generic-complete-mixture` | `equality` | `definition` | 1878 |
| `eq:generic-dmis-expectation` | `unavailable` | `unsupported_or_ambiguous` | 1890 |
| `eq:generic-finite-estimator` | `equality` | `statistical_estimator` | 1898 |
| `eq:generic-cv-estimator` | `equality` | `statistical_estimator` | 1927 |
| `eq:generic-cv-unbiased` | `equality` | `statistical_estimator` | 1933 |
| `eq:generic-cv-variance` | `equality` | `statistical_estimator` | 1935 |
| `eq:generic-finite-cv` | `equality` | `statistical_estimator` | 1941 |
| `eq:generic-defensive-bound` | `unavailable` | `unsupported_or_ambiguous` | 1967 |
| `eq:generic-mixture-responsibility` | `unavailable` | `unsupported_or_ambiguous` | 1994 |
| `eq:generic-cv-tangent` | `equality` | `statistical_estimator` | 2000 |
| `eq:generic-log-cv-tangent` | `equality` | `definition` | 2007 |
| `eq:generic-recursive-program` | `unavailable` | `unsupported_or_ambiguous` | 2060 |
| `eq:generic-affine-contract` | `unavailable` | `unsupported_or_ambiguous` | 2083 |
| `eq:coherent-retained-marginal` | `unavailable` | `unsupported_or_ambiguous` | 2129 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:coherent-gaussian-density:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:coherent-student-density:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:coherent-defensive-bound` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:generic-defensive-bound` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:generic-affine-contract` |

## Role-Specific Obligation Ledger

### `eq:coherent-target-normalizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-pullback-integral`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-initial-pullback`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-floor-normalizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-gaussian-density`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-student-density`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-student-scale`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-mixture-density`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-dmis-identity`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-dmis-variance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-defensive-bound`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-complete-mixture`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-dmis-expectation`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-finite-estimator`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-cv-estimator`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-cv-unbiased`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-cv-variance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-finite-cv`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-defensive-bound`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-mixture-responsibility`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-cv-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-log-cv-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-recursive-program`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:generic-affine-contract`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-retained-marginal`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
