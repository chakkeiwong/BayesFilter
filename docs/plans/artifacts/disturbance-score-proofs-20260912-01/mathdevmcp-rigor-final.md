# Compact Math Document Rigor Audit

Source: `ledh_younis_kdm_score.tex`
Source SHA-256: `8131d744d744827143ec5dde48ef366497a1bdba05fc172cf9386b37ab6c5fad`
Coverage: `partial_coverage`; targets `30`; gaps `3`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `7e893994b31eee277e19060cbf9a32fb55c8f79798907a6b8a0c9722dee95408` (1875936 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:disturbance-state-map` | `unavailable` | `unsupported_or_ambiguous` | 1413 |
| `eq:disturbance-model` | `unavailable` | `unsupported_or_ambiguous` | 1416 |
| `eq:disturbance-joint-integrand` | `equality` | `definition` | 1445 |
| `eq:disturbance-tangent` | `equality` | `definition` | 1468 |
| `eq:disturbance-shock-total-derivative` | `equality` | `definition` | 1474 |
| `eq:disturbance-complete-score` | `unavailable` | `unsupported_or_ambiguous` | 1484 |
| `eq:disturbance-score-identity` | `equality` | `theorem_proposition` | 1497 |
| `eq:disturbance-differentiate-integral` | `equality` | `theorem_proposition` | 1511 |
| `eq:disturbance-proposal-law` | `equality` | `theorem_proposition` | 1543 |
| `eq:disturbance-importance-pair` | `unavailable` | `unsupported_or_ambiguous` | 1550 |
| `eq:disturbance-importance-identities` | `unavailable` | `unsupported_or_ambiguous` | 1558 |
| `eq:disturbance-apf-weight` | `unavailable` | `unsupported_or_ambiguous` | 1592 |
| `eq:disturbance-apf-conditional-normalizer` | `equality` | `theorem_proposition` | 1612 |
| `eq:disturbance-history-kernel` | `equality` | `theorem_proposition` | 1645 |
| `eq:disturbance-particle-measure` | `unavailable` | `unsupported_or_ambiguous` | 1665 |
| `eq:disturbance-particle-unbiasedness` | `equality` | `theorem_proposition` | 1674 |
| `eq:disturbance-particle-derivative` | `unavailable` | `unsupported_or_ambiguous` | 1679 |
| `eq:disturbance-particle-induction-step` | `unavailable` | `unsupported_or_ambiguous` | 1704 |
| `eq:disturbance-analytical-score-recursion` | `equality` | `theorem_proposition` | 1728 |
| `eq:ratio-bias-counterexample` | `unavailable` | `unsupported_or_ambiguous` | 1764 |
| `eq:control-known-integral` | `equality` | `definition` | 1798 |
| `eq:disturbance-control-variate` | `unavailable` | `unsupported_or_ambiguous` | 1804 |
| `eq:control-variate-coefficient` | `equality` | `theorem_proposition` | 1811 |
| `eq:control-variate-variance` | `equality` | `theorem_proposition` | 1828 |
| `eq:disturbance-reference-control` | `unavailable` | `unsupported_or_ambiguous` | 1873 |
| `eq:disturbance-reference-residual` | `equality` | `statistical_estimator` | 1887 |
| `eq:disturbance-conditional-score` | `equality` | `definition` | 1945 |
| `eq:disturbance-rao-blackwell` | `unavailable` | `unsupported_or_ambiguous` | 1954 |
| `eq:disturbance-moving-line-score` | `unavailable` | `unsupported_or_ambiguous` | 1995 |
| `eq:disturbance-moving-line-likelihood` | `equality` | `theorem_proposition` | 2008 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:disturbance-importance-pair` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-importance-pair:obligation_3` |
| `eq:disturbance-analytical-score-recursion` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-analytical-score-recursion:obligation_1` |
| `eq:disturbance-control-variate` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-control-variate:obligation_2` |

## Role-Specific Obligation Ledger

### `eq:disturbance-state-map`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-model`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-joint-integrand`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-tangent`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-shock-total-derivative`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-complete-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-score-identity`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-differentiate-integral`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-proposal-law`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-importance-pair`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-importance-identities`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-apf-weight`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-apf-conditional-normalizer`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-history-kernel`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-particle-measure`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-particle-unbiasedness`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-particle-derivative`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-particle-induction-step`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-analytical-score-recursion`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:ratio-bias-counterexample`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:control-known-integral`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-control-variate`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:control-variate-coefficient`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:control-variate-variance`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-reference-control`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-reference-residual`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-conditional-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-rao-blackwell`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-moving-line-score`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.

### `eq:disturbance-moving-line-likelihood`

- Local obligations: `[]`
- Downstream-only integration obligations: `[]`
- Boundary: the source role selects relevant checks but does not establish their assumptions or truth.


## Boundaries

- `document_rigor_audit_not_document_proof`: The report is a rigor gap/proposal ledger, not a proof of the document.
- `partial_coverage_not_exhaustive`: Limited target selection is not an exhaustive full-document audit.
- `leandojo_not_certificate`: LeanDojo proof search is not a certificate unless the reconstructed Lean source passes direct Lean checking without placeholders.
