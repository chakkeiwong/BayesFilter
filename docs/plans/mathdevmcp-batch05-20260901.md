# Compact Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `43cf6bb7763445b77f8ce677278538b9e6dd4aa6bd1ee51e8d0f1c620de0841f`
Coverage: `partial_coverage`; targets `24`; gaps `5`; concrete repairs `1`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `0c4d5db4766e768b8e65090d82a1c65027646afa3e3ca7201baab437b79e8bab` (1044686 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:coherent-moment-g0` | `equality` | `theorem_proposition` | 2135 |
| `eq:coherent-moment-g1` | `equality` | `theorem_proposition` | 2137 |
| `eq:coherent-moment-g2` | `equality` | `theorem_proposition` | 2139 |
| `eq:coherent-moment-z` | `equality` | `theorem_proposition` | 2149 |
| `eq:coherent-moment-mean` | `equality` | `theorem_proposition` | 2150 |
| `eq:coherent-moment-cov` | `equality` | `theorem_proposition` | 2151 |
| `eq:coherent-linear-transition` | `equality` | `theorem_proposition` | 2172 |
| `eq:coherent-linear-moments` | `unavailable` | `unsupported_or_ambiguous` | 2177 |
| `eq:coherent-nonlinear-moments` | `unavailable` | `unsupported_or_ambiguous` | 2184 |
| `eq:coherent-map` | `equality` | `definition` | 2193 |
| `eq:coherent-joint-covariance` | `equality` | `theorem_proposition` | 2198 |
| `eq:coherent-filter-k` | `equality` | `definition` | 2231 |
| `eq:coherent-filter-operator` | `unavailable` | `unsupported_or_ambiguous` | 2233 |
| `eq:coherent-normalizer-error` | `equality` | `theorem_proposition` | 2238 |
| `eq:coherent-lipschitz` | `unavailable` | `unsupported_or_ambiguous` | 2246 |
| `eq:coherent-error-recursion` | `equality` | `theorem_proposition` | 2252 |
| `eq:coherent-sv-bound` | `unavailable` | `unsupported_or_ambiguous` | 2276 |
| `eq:coherent-sv-envelope` | `equality` | `theorem_proposition` | 2288 |
| `eq:coherent-hermite-ladder` | `equality` | `unsupported_or_ambiguous` | 2308 |
| `eq:coherent-rbf-factor` | `unavailable` | `unsupported_or_ambiguous` | 2316 |
| `eq:coherent-rbf-gram` | `equality` | `theorem_proposition` | 2324 |
| `eq:coherent-product-student-reference` | `unavailable` | `unsupported_or_ambiguous` | 2355 |
| `eq:coherent-frozen-gradient` | `unavailable` | `unsupported_or_ambiguous` | 2391 |
| `eq:coherent-total-weight-gradient` | `equality` | `statistical_estimator` | 2398 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:coherent-moment-g1` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:coherent-moment-g1:obligation_1` |
| `eq:coherent-moment-z` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:coherent-moment-z:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:coherent-map` |
| `eq:coherent-filter-operator` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:coherent-filter-operator:obligation_1` |
| `eq:coherent-total-weight-gradient` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:coherent-total-weight-gradient:obligation_1` |

## Role-Specific Obligation Ledger

### `eq:coherent-moment-g0`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-moment-g1`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-moment-g2`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-moment-z`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-moment-mean`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-moment-cov`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-linear-transition`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-linear-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-nonlinear-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-joint-covariance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-filter-k`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-filter-operator`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-normalizer-error`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-lipschitz`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-error-recursion`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-sv-bound`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-sv-envelope`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-hermite-ladder`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-rbf-factor`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-rbf-gram`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-product-student-reference`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-frozen-gradient`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:coherent-total-weight-gradient`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
