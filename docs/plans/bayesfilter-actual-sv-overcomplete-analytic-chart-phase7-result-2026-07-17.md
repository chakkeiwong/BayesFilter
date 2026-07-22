# Actual-SV Overcomplete Analytical Chart Phase 7 Result

Date: 2026-07-17

Status: `PASS_PHASE_7_TRUSTED_GPU_XLA`

Plan:
`docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md`

## Result

The selected `T=1000,K=23` Actual-SV overcomplete analytical chart completed
the trusted TensorFlow GPU/XLA score run on the first charged attempt.
TensorFlow configured exactly one logical RTX 4080 SUPER GPU with the frozen
`8192 MiB` memory limit before device initialization; memory growth was not
enabled.  XLA compiled the float64 manual-total-JVP route.

Artifact:
`docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-07-gpu/attempt-01-t1000-k23-trusted-gpu-xla-result.json`

## Binding Gates

| Gate | Result |
| --- | --- |
| All five center/FD chart evaluations valid | pass |
| Same-scalar manual score versus FD | pass; relative error `6.037244233113808e-8` versus frozen FD-only tolerance `0.07071067811865477` |
| Same-point warm replay | bitwise identical, including manual score |
| Source and graph topology | pass; source guard clean and concrete graph has three functional `StatelessWhile` nodes |
| Device placement | all recorded claim outputs on `/device:GPU:0` |
| Logical-device cap | one logical GPU, `8192 MiB` |
| Memory growth | disabled |

The center GPU value was `-2286.226010065916` and its manual total score was
`(5.664478697508284,-2.56604922190062)`.  The weakest evaluated chart remained
positive at `5.396496430930526e-165` (`log_beta_minus`, time zero).  The
recorded TensorFlow allocator peak was `1,995,776` bytes; this is TensorFlow
allocator telemetry and does not include the CUDA context, libraries, or
driver allocations.

Total harness time was `69.5032` seconds: trace `2.9250` seconds, first compile
and evaluation `15.8266` seconds, same-point replay `9.2562` seconds, and four
remaining warm evaluations `41.0227` seconds total.  This is engineering
timing from one run, not a statistically supported performance comparison.

## CPU/GPU Diagnostic

The Phase 5 CPU artifact was bound by SHA-256 and preparation identity.  All
validity flags agreed.  Across the five points, objective absolute differences
were at most `9.094947017729282e-13`, and manual-score absolute differences
were at most `1.4408030324375432e-11`.  Condition and residual telemetry also
remained inside the binding validity gates.  These differences are descriptive
only; no post-hoc CPU/GPU equivalence threshold was introduced.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 7 | Frozen full-horizon GPU/XLA chart, FD, replay, graph, device, and cap gates all pass | No OOM, native abort, wrong-device, graph-expansion, chart, or budget veto | One GPU run is engineering certification, not performance statistics or scientific accuracy proof | Write terminal ledgers and update `CE-07` narrowly | No HMC, canonical Contract E--Chol, default, leaderboard, or superiority claim |

## Failure And Retry Ledger

| Attempt | Classification | Repair | Wall time | Budget effect |
| --- | --- | --- | ---: | --- |
| Preflight `T=2` | Successful CPU-hidden harness smoke; not a charged GPU attempt | Confirmed serialization, exception handling, placement, FD, and source audit | `7.50 s` | CPU diagnostic only |
| GPU attempt 1 | pass | None required | `69.50 s` | 1 of 2 attempts; well within 1 GPU-hour budget |

## Post-Run Red Team

The strongest alternative explanation is that output-device strings and a low
TensorFlow allocator peak could coexist with unreported CUDA context memory.
That does not overturn the binding cap: TensorFlow itself created the logical
device with `8192 MB` and all claim outputs were GPU-resident.  It does limit
what the allocator telemetry can claim.  A failure at another parameter point
would also not contradict this run; the certified region is only the frozen
center plus four `1e-5` FD endpoints.
