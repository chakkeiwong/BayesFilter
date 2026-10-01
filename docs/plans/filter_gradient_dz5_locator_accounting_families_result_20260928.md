# Locator accounting-family result

All three isolated CPU/XLA probes and the saved-evidence/policy readback pass.
The unchanged r1, real-one-iteration fixture retains identical first-two-row
inputs, values and validity bytes. Source/AST intervention counts are5/4/2;
each worker has one trace, three optimizer callback batches and no host
callbacks. The readback verifies source, HLO and callback-array hashes for
the new probes and saved controls04584/04585/04635/04636.

| Run | Integer dtype intervention | First-two-row score bytes |
|---|---|---|
| 04642 | Index/calls family (5 attributes) | Identical to unmodified candidate; original difference1.5276668818842154e-13 remains |
| 04643 | Progress family (4 attributes) | Identical to original; differs from unmodified candidate |
| 04644 | Invalid-row family (2 attributes) | Identical to unmodified candidate; original difference unchanged |
| 04645 | Saved evidence and policy |164 checks pass; no new policy exception |

The unit used4/6 workers and675.876765/1800 CPU process-seconds. Close it
without spending the two harness-retry slots. CPU is an explicit diagnostic
exception; no GPU or runtime numerical implementation was changed.

This result localizes a sufficient intervention to the progress family:
`attempts`, `optimizer_calls`, `round_calls`, `replays`. The first, second and
fourth are reported bookkeeping; `round_calls` also controls the callback
budget. Separate that one control counter from the three reporting counters
in a newly allocated bounded unit before proposing a storage-compatible remedy.

Skeptical result review: the family test does not prove a wrong analytical
formula, a specific compiler pass, or that dtype changes in another context
are harmless. Integer resources at int32 remain ineligible for this GPU route.
All121 strict full-trajectory differences, unconverged optimizers and other
master gaps remain open. Do not rerun full trajectories or claim a repair from
this first-score result. Primary-agent review only; no independent reviewer.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Localize to progress counters | Only that family restores original first-score bytes | No GPU-compatible storage remedy | Whole-graph context and family interaction | Separate round-budget and reporting counters | Formula error, full-record equivalence or admission |
