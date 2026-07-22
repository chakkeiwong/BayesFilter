# P5 Target-Design Attempt 01 Repair Record

Date: 2026-07-16

Attempt root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/target-design/attempt-01`

Classification: `HARNESS_EVIDENCE_PERSISTENCE`.

The CPU-only target-design process emitted only TensorFlow initialization
messages. After more than two minutes the Python command was no longer visible,
the PTY did not return, and the attempt root remained empty. The session was
terminated. No design row, prior-predictive result, negative-control result,
dataset, target signature, or scientific decision is reconstructed or claimed
from this attempt.

The original runner wrote artifacts only after all three information rows and
the 4,096-by-200 prior-predictive batch. That made a process/session failure
indistinguishable from long computation and discarded completed rung evidence.
The repair writes `attempt_status.json` before computation, writes every design
seed immediately, persists the combined design, prior-predictive, and
negative-control rungs separately, and flushes bounded progress tokens. The
fixed `[11,5]` by `[200,1]` information computation is now CPU-XLA compiled.

The scientific target, design seeds, source points, horizons, thresholds,
prior-predictive count, final seed, hardware class, and phase budget are
unchanged. Attempt 02 must use a fresh output root.
