# LGSSM NeuTra Sequential HMC Repair Phase R0 Result

Date: 2026-07-15  
Decision: `PASS_R0_CONTROLLER_AND_PLAN_REPAIR`

## Outcome

The fixed 1,000/1,000 admission design is superseded without modifying its
historical artifacts. The TensorFlow-only NeuTra campaign now has a bounded
sequential controller that:

- retains and separately archives every warm-up chunk;
- excludes warm-up from posterior draws;
- checks the latest warm-up window with max(rank-normalized split R-hat,
  folded rank-normalized split R-hat);
- continues cumulative retained sampling under the same modern diagnostic;
- stops each stage only at its pass criterion, health veto, or declared cap;
  and
- enforces 10,000-per-chain maximums for both warm-up and retained sampling.

The real route uses four chains in one TensorFlow/TFP batch, float64, CPU-hidden
execution, and XLA. It reuses a compiled fixed-size program across equal-sized
chunks and writes TensorFlow tensor archives without NumPy.

## Checks And Review

| Check | Result |
| --- | --- |
| Python compile | pass |
| Focused controller/campaign tests | `15 passed` |
| TensorFlow-only import closure | pass inside focused suite |
| `git diff --check` | pass |
| Claude health probe | `CLAUDE_PROBE_OK` |
| Bounded one-path material plan review | `VERDICT: AGREE` |

The deterministic tests cover warm-up extension, retained extension, separate
warm-up/retained tensors and archives, warm-up-cap failure before posterior
sampling, retained-cap failure, and both 10,000 caps.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit R1 execution | controller behavior and focused checks pass | no implementation, diagnostic, artifact, or review veto | how many chunks each real candidate needs | run both fresh sequential candidate verifications | no posterior convergence, correctness, recovery, or superiority claim |

## Handoff

Execute
`docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-r1-subplan-2026-07-15.md`.
The parent target, frozen transports, selected step size `0.8`, leapfrog count
10, thresholds, hardware class, and total six-hour budget remain unchanged.
