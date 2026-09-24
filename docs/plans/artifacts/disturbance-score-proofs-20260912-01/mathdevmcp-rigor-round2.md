# Compact Math Document Rigor Audit

Source: `ledh_younis_kdm_score.tex`
Source SHA-256: `071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a`
Coverage: `partial_coverage`; targets `30`; gaps `6`; concrete repairs `0`; diagnostic abstentions `0`

This is a bounded transport summary. Exact detailed records remain available through `resolve_agent_report`.
Detailed artifact: `7c33c90326a92f09f36750cbbcb8dde61048bf1b4b26ccaccb282707874cd073` (1830798 bytes; state `verified`).

| Label | Relation | Source role | Line |
| --- | --- | --- | ---: |
| `eq:disturbance-state-map` | `unavailable` | `unsupported_or_ambiguous` | 1412 |
| `eq:disturbance-model` | `unavailable` | `unsupported_or_ambiguous` | 1415 |
| `eq:disturbance-joint-integrand` | `equality` | `definition` | 1440 |
| `eq:disturbance-tangent` | `equality` | `definition` | 1463 |
| `eq:disturbance-shock-total-derivative` | `equality` | `definition` | 1469 |
| `eq:disturbance-complete-score` | `unavailable` | `unsupported_or_ambiguous` | 1479 |
| `eq:disturbance-score-identity` | `equality` | `theorem_proposition` | 1492 |
| `eq:disturbance-differentiate-integral` | `equality` | `theorem_proposition` | 1506 |
| `eq:disturbance-proposal-law` | `equality` | `theorem_proposition` | 1538 |
| `eq:disturbance-importance-pair` | `unavailable` | `unsupported_or_ambiguous` | 1545 |
| `eq:disturbance-importance-identities` | `unavailable` | `unsupported_or_ambiguous` | 1551 |
| `eq:disturbance-apf-weight` | `unavailable` | `unsupported_or_ambiguous` | 1585 |
| `eq:disturbance-apf-conditional-normalizer` | `equality` | `theorem_proposition` | 1605 |
| `eq:disturbance-history-kernel` | `equality` | `theorem_proposition` | 1638 |
| `eq:disturbance-particle-measure` | `unavailable` | `unsupported_or_ambiguous` | 1658 |
| `eq:disturbance-particle-unbiasedness` | `equality` | `theorem_proposition` | 1667 |
| `eq:disturbance-particle-derivative` | `unavailable` | `unsupported_or_ambiguous` | 1672 |
| `eq:disturbance-particle-induction-step` | `unavailable` | `unsupported_or_ambiguous` | 1697 |
| `eq:disturbance-analytical-score-recursion` | `equality` | `theorem_proposition` | 1720 |
| `eq:ratio-bias-counterexample` | `unavailable` | `unsupported_or_ambiguous` | 1756 |
| `eq:control-known-integral` | `equality` | `definition` | 1788 |
| `eq:disturbance-control-variate` | `unavailable` | `unsupported_or_ambiguous` | 1794 |
| `eq:control-variate-coefficient` | `equality` | `theorem_proposition` | 1801 |
| `eq:control-variate-variance` | `equality` | `theorem_proposition` | 1818 |
| `eq:disturbance-reference-control` | `unavailable` | `unsupported_or_ambiguous` | 1863 |
| `eq:disturbance-reference-residual` | `equality` | `statistical_estimator` | 1877 |
| `eq:disturbance-conditional-score` | `equality` | `definition` | 1935 |
| `eq:disturbance-rao-blackwell` | `unavailable` | `unsupported_or_ambiguous` | 1944 |
| `eq:disturbance-moving-line-score` | `unavailable` | `unsupported_or_ambiguous` | 1985 |
| `eq:disturbance-moving-line-likelihood` | `equality` | `theorem_proposition` | 1998 |

## Gap Ledger

| Label | Classification | Problem | Evidence |
| --- | --- | --- | --- |
| `eq:disturbance-state-map` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-state-map:obligation_2, proof_audit_v2:eq:disturbance-state-map:obligation_1` |
| `eq:disturbance-model` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-model:obligation_1` |
| `eq:disturbance-complete-score` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-complete-score:obligation_2` |
| `eq:disturbance-importance-pair` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-importance-pair:obligation_3` |
| `eq:disturbance-importance-identities` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-importance-identities:obligation_2` |
| `eq:disturbance-analytical-score-recursion` | `diagnostic_abstention` | The equation role or derivation remains unresolved. | `proof_audit_v2:eq:disturbance-analytical-score-recursion:obligation_1` |

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
