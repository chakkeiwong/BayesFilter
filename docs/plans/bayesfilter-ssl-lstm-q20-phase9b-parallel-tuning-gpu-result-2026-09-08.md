# Phase 9B four-GPU-hour parallel tuning validation

Date: 2026-09-08  
Status: `PASS_PARALLEL_TUNING_VALIDATION_P1_P2_NOT_OPENED`  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md`  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`

The user authorized a fresh total of 14,400 aggregate GPU-seconds. This
campaign validates process-parallel cold-scope tuning, not full P1 sampling.
The old 5,200-second campaign and its 183.799655-second balance are preserved.
Initial reservations are 600 aggregate seconds for a two-worker GPU/XLA smoke,
7,200 for a fresh factor/strict tuning wave, and a 6,600-second repair reserve.
Unused allocations are released; the ceiling is not a target to spend.

The September 8 skeptical audit keeps the complete candidate grid, fresh
heldout verification, original target/bridge, float64, TF32 and XLA settings,
and all scientific vetoes unchanged. Parallel workers never share mutable
TensorFlow state, output roots, or ledger writes. Fresh source hashes account
for pre-existing unrelated worktree changes. Smoke evidence cannot be used as
tuning evidence. The historical first/steady controller forecast remains a
separate P1 blocker.

The GPU startup smoke and fresh two-worker tuning wave both passed. The
campaign consumed **844.217169 aggregate worker-seconds (0.234505 GPU-hours)**,
leaving **13,555.782831 seconds (3.765495 GPU-hours)** unused. No repair or retry
was needed. Both workers exited, all reservations are settled, and the display
GPU was not used. P1/P2 are not opened by this tuning-only validation.

## Campaign and startup

The new ledger is at
`docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/campaign-4gpuh-20260908T090200Z/campaign_budget_ledger.json`.
`campaign_start.json` beside it records the authorization, source closure,
plan hash, old-ledger checksum, and exact smoke command. The directory suffix
is an output label; actual start times are recorded inside the manifests.

Before launch, the refreshed CPU suite passed **137 tests** with 191 dependency
warnings in 13.21 seconds; compilation and `git diff --check` passed. The suite
is the previous 135-test command plus two smoke-versus-tuning isolation tests.
The log is at
`docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/authorized-preflight-r1/cpu-tests.log`.

The two-process GPU smoke passed on non-display physical GPU 1 (factor worker)
and non-display physical GPU 0 (strict worker). Their UUIDs are respectively
`GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c` and
`GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3`. The repository inventory identifies
GPU 2 as the display device; it was not selected. The two worker intervals
overlapped and each verified memory growth before initialization, one trace,
XLA HLO, and two repeated numerical calls. CPU-reference maximum absolute
error was 3.552713678800501e-15 for each call, below the smoke-only 1e-10
tolerance. Smoke consumption was **9.29150758497417 aggregate worker-seconds**,
leaving 14,390.708492415026 seconds before tuning reservation. This validates
only process/GPU startup mechanics, not the q=20 target or tuning computation.

The fresh tuning attempt is `tuning-r1/` under the same campaign root. Its
launch receipt is `tuning-r1-launch.json`. The actual command is:

```bash
timeout --signal=TERM --kill-after=35s 7300s env \
  TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/campaign-4gpuh-20260908T090200Z/tuning-r1 \
  --ledger docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/campaign-4gpuh-20260908T090200Z/campaign_budget_ledger.json \
  --worker-seconds 3600
```

The external 7,300-second cap derives from two queued 3,600-second workers plus
a convenience 100-second coordinator margin. Per-worker deadlines include
30-second termination grace; the outer 35-second kill grace is a supervisor
fallback, not an extra scientific-work allocation. The smoke command uses the
same entrypoint with `--smoke-only`, `--worker-seconds 300`, fresh `smoke-r1`,
and a 700-second outer cap (two queued 300-second workers plus the same margin).

Tuning workers PID 367848 (factor, GPU 1) and PID 367849 (strict, GPU 0) both
persisted verified memory-growth receipts before fresh chart construction.
Their launch environments and seed roots are in their own `run_start.json`.
No existing display or unrelated process was stopped. Both tuning workers have
now exited and the parent reconciled and settled their complete receipts.

## Tuning outcome

The final structured inspection is `validated_result_summary.json` under the
campaign root. It verifies manifest and nested artifact checksums, current
source/plan identity, actual overlapping process intervals, complete grid and
heldout receipts, old-ledger preservation, and summed worker-time accounting.
The frozen parallel plan was not edited during or after the workers ran.

| Diagnostic | Factor, non-display GPU 1 | Strict, non-display GPU 0 |
|---|---:|---:|
| Parent-measured worker lifetime, including startup and exit | 342.403281 s | 492.522381 s |
| Fresh chart construction | 44.034960 s | 51.473545 s |
| Complete scope tuning | 291.446535 s | 434.130373 s |
| Declared/measured grid pairs | 8 / 8 | 8 / 8 |
| Candidates passing the hard screen | Indices 0 and 2 | Indices 0 and 2 |
| Selected candidate | Index 0: epsilon 0.055, L=3 | Index 0: epsilon 0.055, L=3 |
| Fresh heldout verification | Passed | Passed |
| Disjoint tuner seed records | 13 | 13 |
| TensorFlow allocator peak | 268,897,792 bytes | 268,897,792 bytes |

The tuning wave took **493.456773 seconds of elapsed wall time** (about
8 minutes 13 seconds), consuming **834.925661 aggregate worker-seconds**.
The two worker intervals overlapped for **342.399843 seconds**, so this is
observed process-parallel execution, not just a launcher configured for two
workers. TensorFlow was 2.20.0 and TensorFlow Probability 0.25.0. Worker
manifests record the unchanged float64 q=20 protocol, GPU/XLA, TF32, verified
on-demand memory allocation, batch-native chart training, and private fresh
seed/checkpoint/tuning roots. The smallest observed free memory across the
selected GPUs was **31,302 MiB**, above the required 5,120 MiB headroom.

Each arm rejects candidate indices 1, 3, 4, 5, 6, and 7 because a chain did not
move; low acceptance is recorded as a repair trigger. These are candidate-level
vetoes, not a failure of the entire tuning grid or evidence against either
backend. All eight pairs were measured, and both remaining viable candidates
received the declared replicated selection checks. The tuner nominated index
0 and a fresh heldout stream passed before issuing its verified handoff. The
short selection diagnostics do not statistically rank the viable candidates.

The claimed engineering quantity is concurrent execution of the same complete
scope-tuning procedure, with preserved grid/seed/artifact semantics. The
actually executed quantity is fresh-chart, cold-scope (chart 0, beta 1) tuning
in separate factor and strict workers, through the existing public tuner.
Code-path checks, overlapping PIDs, eight-pair receipts, heldout checks, and
checksums support that narrow match. This is not a paired numerical-equivalence
or performance experiment: fresh charts/seeds and physical device context
differ between arms. The observed timing differences are descriptive only;
there is no statistically supported speedup or factor-versus-strict ranking.

## Accounting and cleanup

| Budget item | Aggregate worker-seconds |
|---|---:|
| User-authorized cap | 14,400.000000 |
| Two-worker startup smoke | 9.291508 |
| Two-worker fresh tuning | 834.925661 |
| Total consumed | 844.217169 |
| Remaining, not spent | 13,555.782831 |
| Active reservations | 0.000000 |

There are exactly four unique settlements: two smoke workers and two tuning
workers. The consumed balance equals the sum of their individual lifetimes;
overlapping time is not discounted. The old ledger remains byte-for-byte
unchanged at SHA-256
`1961d07cf421b06abfb6ec4b62830b44e4052849505d72cd3cf0cea087160415`.
No additional allocation, environment mutation, retry, or target change was
made. The cap is a maximum, not a reason to run unnecessary work.

`gpu-after.json` records the final trusted NVIDIA inventory. The four campaign
worker PIDs (366435, 366436, 367848, 367849) have exited. GPU 1 returned to
18 MiB used and GPU 0 to 337 MiB, while the unrelated/display processes remain
present. The existing lightweight compute context on GPU 0 was permitted by
the owner's utilization/headroom rule; this run did not require exclusivity
or stop that process.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Process/GPU tuning execution | Passed: real overlapping workers, growth/XLA startup and complete accounting | No process, memory, source, or artifact veto fired | Only this device/scope/batch combination was exercised | Retain the process-parallel tuning route for this diagnostic | No broad multi-GPU scaling or default-readiness claim |
| Factor and strict cold-scope tuning | Passed: eight measured pairs per arm, viable candidates, fresh heldout and verified handoffs | Six candidate-level movement vetoes per arm; selected candidates pass | Short streams and different fresh chart/seed realizations | Preserve both tuning results; keep candidate rejection distinct from research-direction rejection | No sampler ranking, posterior convergence, or whitening |
| M4-P0 full-controller readiness | Not passed by this run | Old incomplete strict controller timing and trace/startup diagnostic gaps remain | Per-transition target/score/status cost at the full schedule | Continue P0-I/J/K/L instrumentation and bounded cost localization | A tuning handoff is not a full readiness closeout |
| P1 and P2 | Remain blocked | Original factor forecast exceeds its P1 arm cap; posterior/chart/uncertainty gates remain open | Affordability and downstream scientific behavior | Establish an affordable, complete controller schedule before P1; P2 stays separate | This allocation/run does not launch or fund complete P1 sampling |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Six grid candidates per arm reject on lack of all-chain movement. Both selected candidates and their independent heldout checks pass; no resource or artifact veto fires. |
| Statistically supported ranking | None. The two viable grid candidates per arm are not ranked scientifically by these short diagnostics. |
| Descriptive-only differences | Per-arm chart/tuning/runtime observations and candidate acceptance/efficiency metrics; no uncertainty-aware comparison was run. |
| Default-readiness | Not established; this validates the optional Phase 0 execution route for one cold scope. |
| Next evidence needed | Complete sequential-controller health/cost measurements, then downstream posterior/reference checks and predeclared multi-seed uncertainty evidence for any ranking. |

## Post-run red team and next step

The strongest alternative explanation for the timing difference is different
chart/seed realization or device/host context, not an algorithmic advantage.
The weakest evidence for broader execution readiness is that no 500-transition
controller chunk or full six-chunk P1 schedule was executed here. GPU startup
and tuning cannot substitute for that missing measurement. A changed source,
failed checksum, missing grid pair, failed independent heldout check, or
downstream trace-health failure would invalidate the corresponding conclusion.

The current question is answered without using the reserve: tuning does run
in multiple isolated GPU processes, both required arms finish, and cost stays
within the authorized aggregate budget. There is no candidate or research-
direction rejection. The next program work is the remaining full-controller
instrumentation and performance localization, under an explicit bounded
subplan that preserves the remaining allocation. Do not launch the old P1
command or reinterpret this tuning manifest as a Phase 0 full-readiness pass.
