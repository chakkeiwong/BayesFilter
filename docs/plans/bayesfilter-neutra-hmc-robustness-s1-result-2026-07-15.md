# NeuTra HMC Robustness Phase S1 Result

Date: 2026-07-15  
Decision: `PASS_S1_SAME_FIXTURE_THIRD_SEED`

## Outcome

An independently initialized `wide_2x_lr5e3` transport was trained for 5,000
batched GPU/XLA steps with seed `(20260715, 1203)`. The training used one
compiled `tf_while_loop`, TensorFlow memory growth, no full-device
preallocation, no repository NumPy/host callback, no prior weights, all-valid
target status, and exact frozen reload/value-score parity.

The fresh downstream HMC arm nominated step size `0.8` from acceptance but
admitted it only through the shared sequential controller. Admission retained
2,000 warm-up samples per chain, excluded them from posterior draws, and
extended retained sampling from 1,000 (`max modern R-hat 1.01744`, fail) to
2,000 (`1.00665`, pass).

Independent confirmation used fresh seeds, retained 2,000 warm-up samples per
chain, and collected 4,000 posterior draws per chain. It passed with:

| Gate | Result |
| --- | ---: |
| max modern R-hat | 1.0027965461 |
| min bulk ESS | 5394.2862 |
| min tail ESS | 4381.2278 |
| max plain-HMC difference, combined MCSE | 1.9669 |
| max truth distance, posterior SD | 1.6456 |
| health/status/divergence vetoes | none |

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Primary criterion | passed same-fixture downstream HMC contract |
| Hard veto screen | passed; no nonfinite, invalid status, immobility, or declared energy-error event |
| Same-fixture additional-seed viability | supported for this one seed |
| Statistically supported ranking | none; no method or seed ranking was tested |
| Descriptive-only differences | acceptance, training trajectory, runtime, and point metric differences |
| Joint robustness | not established until the new-fixture F2 arm also passes |
| Default readiness | engineering default path is consolidated; broad scientific default readiness is not established |

The strongest alternative explanation is that this favorable observation
fixture is unusually easy and the selected architecture is seed-stable only
here. A failed valid new-fixture arm would overturn any two-fixture robustness
claim. This result does not establish calibration, superiority, population
reliability, broad robustness, production readiness, or universal NeuTra
validity.

## Artifacts

- Training result:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/s1/training-attempt-01/phase4/training_jobs/dense_seed1203/attempt_1_graph_native/result.json`
- Tuning/admission result:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/s1/hmc-tuning-attempt-01/result.json`
- Confirmation result:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/s1/confirmation-attempt-01/result.json`

## Handoff

Proceed to F0 using the predeclared new simulation seed `(20260715, 701)`.
The old comparator cannot be reused because observations and target identity
will differ.
