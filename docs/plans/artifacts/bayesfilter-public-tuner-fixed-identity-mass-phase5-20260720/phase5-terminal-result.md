# Phase 5 Terminal Result

Date: `2026-07-20`

## Decision

`PHASE5_CLOSED_RUNTIME_BOUNDARY_AND_TUNING_VETO`

The Phase 5 execution plan and observability repair were implemented and
verified. The campaign did not produce four complete scientific cell results
under the current managed execution boundary. The preserved artifacts support
the following classifications.

| Cell | Durable evidence | Classification |
| --- | --- | --- |
| `PP-UKF` | Complete training, manifest, streamed log, and `result.json` | `TUNING_FAILED`; hard veto `bootstrap_screen_error`, repair trigger `ValueError`; no HMC sampling authorized |
| `PP-SGQF` | Complete training, manifest, streamed log, and `result.json` | `TUNING_FAILED`; hard veto `bootstrap_screen_error`, repair trigger `ValueError`; no HMC sampling authorized |
| `SIR-SGQF` | Three of five final-training segments, GPU/XLA metadata, streamed log; no terminal result | `INTERRUPTED_RUNTIME_BEFORE_FINAL_TRAINING_COMPLETION`; no sampler or truth-tail evidence |
| `STR-UKF` | GPU/XLA initialization, first screen state, streamed log; no terminal result | `INTERRUPTED_RUNTIME_DURING_RECIPE_SCREEN`; no final training, sampler, or truth-tail evidence |

The complete Phase 4 LGSSM two-seed result remains authoritative and is not
part of this Phase 5 campaign.

## Engineering Evidence

- Phase 5 plan audit: `PASS_FOR_IMPLEMENTATION_AND_EXECUTION`.
- Focused NeuTra contract suite: `29 passed, 2 warnings`.
- Python compilation and focused `git diff --check`: passed.
- Trusted RTX 4080 SUPER preflight: passed.
- TensorFlow 2.19.1 GPU/XLA compilation: passed.
- TensorFlow memory-growth policy: verified before logical-device creation.
- New per-cell `run_state.json` and streamed `launch-logs/<cell>.log` were
  written for all launched cells.

## Interpretation

The two predator-prey cells demonstrate that training, artifacting, and GPU/XLA
execution completed, but the public tuner rejected both at its bootstrap
screen. This is candidate/tuning failure evidence, not evidence against the
target or NeuTra research direction.

The SIR and structural cells were interrupted by the managed process boundary
before terminal artifacts. Their partial training/screen files are engineering
and infrastructure evidence only and must not be interpreted as transport,
sampler, convergence, or truth-tail results.

No cross-cell ranking is supported. No cell is default-ready or production
ready. The blocked registry inventory was not launched and remains separate
from candidate failures.

## Next Justified Action

Do not rerun under the same managed boundary. A future continuation needs a
host terminal, SSH session, host-level `tmux`, or host service whose process
lifetime is independent of the Codex managed PID namespace. It should use fresh
roots and the unchanged scientific contract. Before that continuation, inspect
the shared `bootstrap_screen_error` payload for the two predator-prey cells and
decide whether a localized tuner repair is justified; do not retune on claim
data or silently promote a cross-model setting.
