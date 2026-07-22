# Structural UKF NeuTra Phase 2 Training Result

Date: 2026-07-17

Decision: `ADMIT_FRESH_5000_STEP_TRANSPORT_CONTINUE_HMC`

## Outcome

All four target-specific 500-step GPU/XLA screens completed with finite values,
valid target telemetry, batch-native execution, and frozen/trainable parity.
The common-heldout reverse-KL means were:

| Recipe | Hidden width | Learning rate | Mean heldout reverse KL |
| --- | --- | ---: | ---: |
| `dim3_lr1e3` | `(15,15)` | `0.001` | `138.3472` |
| `dim3_lr5e3` | `(15,15)` | `0.005` | `137.9174` |
| `dim6_lr1e3` | `(30,30)` | `0.001` | `138.1008` |
| `dim6_lr5e3` | `(30,30)` | `0.005` | `137.8506` |

The frozen paired rule nominated `dim6_lr5e3`. This is nomination evidence,
not a statistically supported ranking or a transport-quality claim. The
selection record SHA-256 is
`980776ec5e3d53159cf57d618f02adf0afd39b64c68d9962edac404fc404f4f5`.

A fresh selected 5,000-step run then completed without reusing screen weights.
The first unsegmented final attempt was lost in a managed-session restart and
produced no checkpoint or scientific result. Attempt 2 used the reviewed
infrastructure-resume mechanism in five deterministic 1,000-step XLA segments.
Each segment restored all trainable variables and both Adam moments from a
hash-checked state in a fresh directory; the target, seed, optimizer, schedule,
and step trajectory did not change.

The final transport passed target health, GPU placement, XLA control-flow,
frozen/trainable value/log-determinant/pullback parity, and common-heldout
checks. Its heldout reverse-KL mean was `137.7344`, compared with `195.5907`
for the exact zero/identity source-prior baseline. This difference is
descriptive training evidence only.

## Binding artifacts

- final result SHA-256:
  `fe2ff521b65a43e62ab23f32691e579ed8c8202cbc5ec91872a000995f860777`;
- final run-manifest SHA-256:
  `23df1d6fbfdf408859695bdb0aa66317405a2ab7e08488ec31ddbd5dca8b4756`;
- frozen transport SHA-256:
  `58d626f77d7cc668fe9d5959d0f73d9dd13b5a78c208fa9ac125b0b1ea0bd0d6`;
- step-5000 checkpoint SHA-256:
  `939c6c27874e5a3f0c443071d8512c79bca0bcf80a04df34ed0b160dd4c55eec`;
- recursive-hash ledger SHA-256:
  `f4fd5a9513dd7872c36dd50ff8797d834828d14514cfaeebde17c65d2f1b944d`;
- terminal training state hash:
  `42e2934d645f0dec9c554befbf92f4cd0f5716b897ff8977a34b8966d13e4513`.

The five successful segment wall times sum to about `747.0` seconds, including
repeated process startup, identity replay, XLA compilation, checkpoint writing,
and final heldout/parity checks. The failed attempt consumed about nine GPU
minutes but supplied no weights or scientific evidence.

## Handoff review

Phase 3 entry conditions are satisfied. The HMC harness must load only the
selected final payload above, replay its exact typed target identity, run a
compiled physical-coordinate canary, tune on separate probe and verification
seeds, retain warm-up separately, and use physical-coordinate modern R-hat,
ESS, health, and truth-tail gates. Training or heldout loss may not substitute
for any sampler gate.

No HMC convergence, truth recovery, filter exactness, calibration, superiority,
or readiness claim is made by Phase 2.
