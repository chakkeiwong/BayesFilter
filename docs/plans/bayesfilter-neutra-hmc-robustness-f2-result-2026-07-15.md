# NeuTra HMC Robustness Phase F2 Result

Date: 2026-07-15  
Decision: `PASS_F2_NEW_FIXTURE_NEUTRA`

## Outcome

The exact F1 frozen candidate was tuned and sampled on target signature
`312d2f4ceb5d65bf18251fa53ae1276781c62fd2daefaba0bda8dc3d46a5d283`.
Step size `0.8` was nominated at acceptance `0.625` with no configuration health
veto. Shared-controller admission retained 2,000 warm-up samples per chain and
extended retained sampling from 1,000 (`max modern R-hat 1.01177`, fail) to
2,000 (`1.00498`, pass).

Independent confirmation used fresh seed roots, retained 2,000 warm-up samples
per chain, and collected 4,000 posterior draws per chain. It passed:

| Gate | Result |
| --- | ---: |
| max modern R-hat | 1.0036022359 |
| min bulk ESS | 4073.1309 |
| min tail ESS | 3154.1754 |
| max F0 comparator difference, combined MCSE | 1.7740 |
| max truth distance, posterior SD | 1.3420 |
| health/status/divergence vetoes | none |

At 2,000 retained confirmation draws the full diagnostic was already finite and
within R-hat/ESS thresholds, but the predeclared 4,000 minimum correctly forced
another independent retained chunk. Warm-up was archived and excluded from
posterior summaries.

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Primary F2 criterion | passed downstream HMC/comparator/recovery contract |
| Hard veto screen | passed |
| New-fixture NeuTra viability | supported for this candidate and fixture |
| S1 same-fixture additional-seed arm | passed independently |
| Joint program robustness answer | passed narrowly for one additional seed and one additional fixture |
| Statistically supported ranking | none |
| Descriptive-only differences | acceptance, runtime, ESS, losses, and metric magnitudes |
| Broad robustness/default readiness | not established |

The strongest alternative explanation is that both fixtures belong to the same
favorable 18D LGSSM family and the learned architecture may not transfer to
other models, dimensions, data regimes, or posterior geometries. A valid
failure on another model family would overturn any broader claim. This result
does not establish sampler superiority, calibration, population reliability,
production readiness, or universal NeuTra reliability.

## Artifacts And Handoff

- Admission: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f2/tuning-attempt-01/result.json`
- Confirmation: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f2/confirmation-attempt-01/result.json`

Proceed to Phase A terminal omission/drift audit. Any report or policy claim
must preserve the narrow two-fixture/additional-seed scope.
