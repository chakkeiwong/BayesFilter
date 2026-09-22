# Compact Math Document Rigor Audit

Source: `attempt05_n4_failure_analysis.tex`
Source SHA-256: `43cf6bb7763445b77f8ce677278538b9e6dd4aa6bd1ee51e8d0f1c620de0841f`
Coverage: `partial_coverage`; targets `29`; gaps `6`; concrete repairs `2`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `4141c73f42bbb40b01674a7e2e2aa64452a7f54dec7c50e6d713451a810330e5` (1494478 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:actual-ukf-moments` | `unavailable` | `unsupported_or_ambiguous` | 563 |
| `eq:actual-ukf-update` | `unavailable` | `unsupported_or_ambiguous` | 575 |
| `eq:actual-ukf-joint-update` | `unavailable` | `unsupported_or_ambiguous` | 589 |
| `eq:actual-raw-ukf-zero-gain` | `unavailable` | `unsupported_or_ambiguous` | 613 |
| `eq:actual-log-square-guide` | `unavailable` | `unsupported_or_ambiguous` | 624 |
| `eq:actual-unnormalized-retained` | `unavailable` | `unsupported_or_ambiguous` | 642 |
| `eq:actual-reference-moments` | `unavailable` | `unsupported_or_ambiguous` | 649 |
| `eq:actual-recursive-moments` | `unavailable` | `unsupported_or_ambiguous` | 659 |
| `eq:actual-recursive-prediction` | `unavailable` | `unsupported_or_ambiguous` | 665 |
| `eq:branch-target` | `equality` | `definition` | 770 |
| `eq:branch-energy` | `equality` | `theorem_proposition` | 779 |
| `eq:als` | `unavailable` | `unsupported_or_ambiguous` | 808 |
| `eq:rms` | `equality` | `definition` | 827 |
| `eq:count-rms` | `equality` | `definition` | 834 |
| `eq:zh` | `unavailable` | `unsupported_or_ambiguous` | 846 |
| `eq:corr-increment` | `equality` | `theorem_proposition` | 872 |
| `eq:telescope` | `equality` | `theorem_proposition` | 877 |
| `eq:shift-invariance` | `equality` | `definition` | 976 |
| `eq:l2-bound` | `unavailable` | `unsupported_or_ambiguous` | 999 |
| `eq:decomp` | `equality` | `statistical_estimator` | 1150 |
| `eq:proposal-map` | `equality` | `definition` | 1231 |
| `eq:q-normalizer` | `equality` | `definition` | 1244 |
| `eq:q-physical` | `equality` | `definition` | 1245 |
| `eq:is-identity` | `equality` | `theorem_proposition` | 1271 |
| `eq:incomplete-hermite-gram` | `equality` | `definition` | 1315 |
| `eq:hermite-product` | `equality` | `theorem_proposition` | 1320 |
| `eq:hermite-antiderivative` | `unavailable` | `unsupported_or_ambiguous` | 1330 |
| `eq:incomplete-hermite-closed` | `equality` | `definition` | 1358 |
| `eq:hermite-kr-cdf` | `equality` | `unsupported_or_ambiguous` | 1372 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:actual-ukf-moments` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-ukf-moments:obligation_1, proof_audit_v2:eq:actual-ukf-moments:obligation_2, proof_audit_v2:eq:actual-ukf-moments:obligation_3` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `document_exposition:eq:actual-ukf-update` |
| `eq:actual-ukf-joint-update` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:actual-ukf-joint-update:obligation_4, proof_audit_v2:eq:actual-ukf-joint-update:obligation_5` |
| `eq:als` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:als:obligation_1` |
| `invertibility_required` | `concrete_repair` | The displayed inverse/series lacks required local exposition conditions. | `proof_audit_v2:eq:q-physical:obligation_1` |
| `eq:hermite-antiderivative` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:hermite-antiderivative:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:actual-ukf-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-ukf-update`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-ukf-joint-update`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-raw-ukf-zero-gain`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-log-square-guide`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-unnormalized-retained`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-reference-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-recursive-moments`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:actual-recursive-prediction`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:branch-target`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:branch-energy`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:als`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:rms`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:count-rms`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:zh`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:corr-increment`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:telescope`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:shift-invariance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:l2-bound`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:decomp`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:proposal-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:q-normalizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:q-physical`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:is-identity`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

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


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
