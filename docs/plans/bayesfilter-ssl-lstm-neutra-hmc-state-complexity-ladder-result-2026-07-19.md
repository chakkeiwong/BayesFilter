# SSL-LSTM NeuTra-HMC State-Complexity Ladder Result

Date: 2026-07-19  
Decision: `PHASE1_TARGET_PASS_PHASE2_RESOURCE_STOP_NO_MATERIAL_TRAINING`

## Scope

The executed work repaired the dimension-general four-coordinate target and
selected-direction analytic score, then ran trusted TensorFlow/XLA target
preflights for `q=1,2,5,10,20` and full NeuTra/HMC mechanics canaries for
`q=1,2`. The 64 GiB host-RAM ceiling was applied. No material NeuTra training,
Optuna study, HMC tuning, retained posterior acquisition, or predictive
validation was executed.

## Target And Score Result

The new target estimates the same four homologous coordinates at every rung:
`latent_mean_weight.0.0`, `latent_mean_bias.0`,
`observation_weight.0.0`, and `observation_bias.0`. The remaining chart
coordinates are fixed by a deterministic rung fixture. This avoids pretending
that 3,862 q=20 parameters can be estimated from 30 scalar observations.

The selected score path constructs derivative surfaces with leading dimension
four instead of the full `9q^2+13q+2` chart dimension. Focused checks passed:

- existing full-route and structural adapter tests: `19 passed`;
- selected-target focused suite: `7 passed, 5 deselected`;
- q=1 target parity with the locked scalar target: passed;
- q=2 selected score versus finite differences: passed;
- q=1,2,5,10,20 directional derivative surfaces have leading dimension 4;
- deterministic fixture and synthetic observation replay: passed;
- isolated q=20 target/XLA preflight: finite repeated score, four-coordinate
  score, host RSS `3.03 GiB`, GPU allocator peak `0.31 GiB`.

The q-ladder target receipt is
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/target-preflight.json`.
The isolated q=20 receipt is
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/q20-isolated-target-preflight.json`.

## Mechanics Canary Result

Each canary used the real 32x32 three-stage dense-IAF trainer, two independent
seeds, batch size 480, 10 training steps per seed, frozen transport reload,
and a two-draw transformed-HMC mechanics call. These are mechanics/timing
artifacts only.

| q | Canary wall | Host RSS | Warm NeuTra step | HMC samples | Status |
|---:|---:|---:|---:|---|---|
| 1 | 510.17 s | 8.48 GiB | 6.79 s | finite `[2,4,4]` | mechanics passed; resource projection veto |
| 2 | 630.78 s | 8.72 GiB | 12.06 s | finite `[2,4,4]` | mechanics passed; resource projection veto |

Receipts:

- `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/canary-q1.json`;
- `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/canary-q2.json`.

The q=1 two-seed 5,000-step training projection is `74,271 s` before the
declared 50% margin and `111,407 s` with margin, approximately `30.9 GPU-hours`.
The q=2 projection is `123,436 s` before margin and `185,155 s` with margin,
approximately `51.4 GPU-hours`. Neither projection includes Optuna trials,
HMC tuning/confirmation, four-chain retained draws, or predictive validation.

The q=1/q=2 canary execution cost was `1,140.95 s` wall in fresh trusted GPU
processes. The q=5, q=10, and q=20 target preflights were already complete, but
their material canaries were correctly skipped after the q=1 resource gate.

## Decision Table

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
| --- | --- | --- | --- | --- |
| Admit target implementation for further planning | q=1 parity, q=2 finite differences, all-rung four-direction shapes, q=20 isolated memory/finite score | No target or host-RAM veto | Retain the generalized target and directional-score repair | Posterior correctness or HMC convergence |
| Admit q=1/q=2 mechanics | Two-seed trainer and minimal HMC calls finite with valid artifacts | No mechanics veto | Use timing to size a new GPU budget | Training quality, sampler validity, or convergence |
| Stop material ladder | q=1 projected 30.9 GPU-hours before downstream phases; current authority is 1.0 GPU-hour | Resource continuation veto | Request explicit GPU-hour authorization and refresh the ledger | Scientific failure, NeuTra failure, or HMC failure |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Target and mechanics passed; resource continuation veto fired |
| Statistically supported ranking | None |
| Descriptive-only differences | q-dependent wall, warm-step cost, RSS, GPU allocator telemetry, and training-loss rows |
| Default readiness | Not established |
| Next evidence needed | New GPU budget, full 5,000-step two-seed training, Optuna nomination, transformed-target preflight, tuned four-chain HMC, and predictive-moment validation |

## Why Execution Stopped

The new 64 GiB host-RAM limit is not the bottleneck. Isolated q=20 target RSS
was only `3.03 GiB`; q=1/q=2 canary RSS was below `9 GiB`. The limiting cost is
GPU/XLA execution of the 480-row reverse-KL trainer and HMC graph. Continuing
without a materially larger explicit GPU-hour budget would produce a partial
ladder and recreate the earlier scope error.

## Nonclaims And Red Team

No posterior oracle was used. The canaries do not establish NeuTra quality,
HMC convergence, posterior correctness, model adequacy, full-parameter
estimability, predictive validity, or superiority over plain HMC. The strongest
alternative explanation for the high cost is compilation and graph shape
overhead rather than steady-state per-step cost; a future budget refresh should
measure reusable compiled continuation separately. The conclusion would be
overturned as a resource decision if a validated smaller-shape batch or reused
compiled runner reduced the complete two-seed training and downstream workflow
to the newly authorized envelope.
