# NeuTra HMC Robustness Phase F0 Result

Date: 2026-07-15  
Decision: `PASS_F0_NEW_FIXTURE_AND_COMPARATOR`

## Outcome

Simulation seed `(20260715, 701)` produced a deterministic T=120 observation
fixture with target signature
`312d2f4ceb5d65bf18251fa53ae1276781c62fd2daefaba0bda8dc3d46a5d283`.
The model, prior, truth, horizon, and parameter order match the original
fixture, while the config, fixture, observations, and target hashes differ.

The exact CPU/XLA value-score gate and target-specific geometry/mass gates
passed. The mass condition number was `2.13`, no eigenvalue clipping or
fallback occurred, and factor reconstruction error was `1.4e-17`.

The first comparator grid failed before sequential sampling: one `0.8` kernel
had declared energy-error events and the healthy grid did not resolve the
acceptance bracket. A fresh bounded repair grid selected step size `0.3` at
acceptance `0.75`. The shared controller then retained 2,000 warm-up samples
per chain and collected 4,000 posterior draws per chain. Results were:

| Gate | Result |
| --- | ---: |
| max modern R-hat | 1.0053980349 |
| min bulk ESS | 1758.5945 |
| min tail ESS | 3982.9331 |
| max truth distance, posterior SD | 1.3839 |
| health/status/divergence vetoes | none for admitted kernel |

At 2,000 retained draws, R-hat and tail ESS passed but minimum bulk ESS was
`870.27`; the controller correctly extended to 4,000. Warm-up was archived and
excluded from posterior summaries.

## Decision And Inference Status

| Item | Status |
| --- | --- |
| New fixture identity | passed |
| Exact target/XLA/geometry/mass | passed |
| Plain-HMC comparator | passed after one localized tuning-grid repair |
| Hard veto evidence | step size 0.8 rejected; admitted step 0.3 had none |
| Statistically supported ranking | none |
| Descriptive-only differences | fixture spread, acceptance, runtime, and metric magnitudes |
| NeuTra evidence | none yet on this fixture |
| Next justified action | F1 target-specific training screen |

The new fixture has descriptively larger observation spread in three of four
coordinates and a larger maximum observation, but this does not prove it is a
harder posterior. This phase establishes a valid comparator, not NeuTra
quality, sampler superiority, calibration, broad robustness, production, or
default readiness.

## Artifacts

- Identity ledger: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f0/fixture_identity.json`
- Failed grid: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f0/plain-hmc/comparator/result.json`
- Admitted comparator: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f0/plain-hmc/comparator-repair-attempt-02/result.json`

## Handoff

F1 may use the new exact target and target-specific factor. It must not reuse
the truth-centered comparator center as the NeuTra training center and must not
reuse screen weights in the long run.
