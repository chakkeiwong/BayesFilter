# Phase 9B recovery, health, and two-arm runtime result

Date: 2026-09-09 (Asia/Shanghai).  
Status: `M4_P0_RECOVERY_HEALTH_VALIDATED_STRICT_AFFORDABILITY_BLOCKED`.  
Governing master: `bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.  
Executed plan: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-plan-2026-09-09.md`.  
Repair record: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-grid-repair-2026-09-09.md`.  
Component plan: `bayesfilter-ssl-lstm-q20-phase9b-component-profile-plan-2026-09-09.md`.

## Outcome

The requested cap change, interruption-recovery repair and canary, complete
two-arm timing measurement, and sampled-state health checks are complete.
The active ceiling is **14,400 seconds per arm**, within the owner's separate
**36,000 aggregate GPU-worker-second** allocation. Both arms ran in independent
processes on non-display GPUs; display GPU 2 was not used.

Both hard-interruption canaries reproduce their uninterrupted references
exactly. Both full runtime arms complete two actual 500-transition calls through
the shared controller, with all required health checks passing. These are
warmup diagnostics, not completed P1 sampling or posterior evidence.

**Phase 0 has not passed.** Strict's six-chunk forecast is 14,040.61 seconds
without interruption reserve and **16,202.12 seconds with the declared reserve**,
above the four-hour ceiling. The conservative fresh two-arm reserve-inclusive
forecast also exceeds the remaining campaign balance. Component profiling
identifies repeated target-status evaluation as a concrete optimization
hypothesis, not an implemented speedup. The old serial P1 entrypoint additionally
needs checkpoint integration and refreshed source/readiness checks. P1/P2 were
not launched; no campaign GPU worker remains.

## Evidence and provenance

The campaign root, abbreviated `campaign/` in this note, is:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z/`

Its sibling `preflight/` directory contains launch and regression logs.
`campaign-start.json` records Git commit
`301a521bedd50819b54a5e926154fdf2e3c606f5`, the dirty worktree, original command,
interpreter, plan and result paths, source hashes, and allocation. The original
plan remains frozen at SHA-256
`b6e3ac4ef9aa25e5a403e1ce59defb75dc584cd90d9b91525735967173479395`;
its prelaunch/pending wording is historical, superseded by this result.

`source-migration-r1.json` and `source-migration-r2.json` record bounded
orchestration repairs, preserved stream identities, and unrelated concurrent
work. No ledger debit was reset. The final campaign source identity is
`568437ae40ed94e72fa93856ac905bd113378637ef9e3b87bc2cf0252f66663d`.
All 155 files in that recorded closure still match at terminal verification.
The component sidecar has separate script/plan hashes and did not alter the
runtime numerical source closure.

| Evidence | Location under `campaign/` |
|---|---|
| Exact canary equality and same-GPU bindings | `canary-result.json`, `interrupt-complete.json`, `resume-complete.json` |
| Full runtime, health, setup, seeds, source and allocator receipts | `launches/runtime-ad2ac13153/{factor,strict}/run_manifest.json` |
| Independent checkpoint and health verification | `postrun-verification-r1.json` |
| Completed component profiles and input banks | `launches/component-profile-daee5a058c/{factor,strict}/` |
| Final spending, outstanding reservations and attempt history | `campaign_budget_ledger.json` |
| Cleanup and trusted device inventory | `process-cleanup-verification.json`, `gpu-after.json` |
| Compact final summary including profiling debits | `terminal-summary.json` |

`validated-result.json` is the successful runtime-stage summary. Its budget
snapshot predates component profiling and is preserved, not overwritten or
misrepresented as the terminal balance.

Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, TensorFlow 2.20.0,
TFP 0.25.0, float64, four chains with four parameters, XLA enabled, TF32 enabled,
and verified memory growth before logical-device initialization. Factor used
physical GPU 1 (`GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`); strict used physical
GPU 0 (`GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3`). Both are RTX 4080 SUPER
devices. Per-worker TensorFlow allocator peak during full runtime was
34,567,168 bytes; this is not the same quantity as process reservation in
`nvidia-smi`. Startup, admission and headroom snapshots are retained per launch.
The owner policy remains utilization at most 40%, estimated worker memory plus
5 GiB free headroom, non-display first, and display only if no non-display
device is eligible. Unrelated contexts were neither stopped nor treated as an
exclusivity veto.

The q=20 target, observations and chart identities are bound by each manifest's
source closure and `setup.profile` / `setup.chart_source_identity`; no external
dataset or new observation bank was introduced by the repairs or profiling.
Complete training, initialization, preflight and tuning seed namespaces are
preserved there. Full-runtime warmup roots are `[866397959, 20260909]` for
factor and `[72698284, 20260909]` for strict. The corresponding declared retained
roots end in `20261909`, but no retained sampling occurred.

The principal executed commands, from the repository root, were:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py \
--campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z \
--canary-only

TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py \
--campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z \
--resume

TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/profile_ssl_lstm_q20_phase9b_components_2026_09_09.py \
--campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z
```

These are executed-command records, not instructions to rerun completed work.
Canary retries used `--resume --canary-only`; the bounded reference-timeout
repair is recorded in `reference-timeout-repair-launch.json` and its sibling
preflight log. Per-launch job, supervision and start records preserve actual
child commands, external timeouts, timestamps and environment.

## Recovery implementation and canary

`bayesfilter/runtime/durable_tensor_checkpoint.py` stores TensorFlow tensor
trees without NumPy or pickle. Completed bundles are checksummed and atomically
published after synchronized writes; partial attempts remain separate. Input
and source identities must match, and committed corruption fails closed rather
than silently regenerating evidence. The optional checkpoint-aware shared
controller route in `bayesfilter/inference/neutra_hmc.py` preserves the existing
compiled kernel, stateless seed derivation, and warmup/retained rules.

The launcher commits chart construction, verified tuning, sampling chunks and
archives separately. An external timeout bounds each child even after parent
loss; parent reconciliation prevents double charging. A queue repair preserves
completed siblings and selects the pending arm's recorded GPU rather than
mistaking another free GPU for evidence that its required device is unavailable.

For each arm, a separate process was **SIGKILLed during its second compiled
eight-transition call**, after its first chunk was committed, archived and
health-checked. A fresh process reused that first chunk, recomputed the
interrupted second chunk, and reproduced the uninterrupted **sample and full
trace tensor bytes exactly**, on the same GPU. Both canaries pass. This is not
merely a pause between completed calls.

Recovery is from the last committed unit, not a live CUDA instruction pointer.
An interruption during a 500-transition production-sized diagnostic call can
require replay of that entire unfinished call. An unfinished chart/training or
tuning stage is replayed as a stage; individual internal tuning-call logs are
not resumable tuner checkpoints. The GPU canary exercises eight-transition
calls, while full-sized calls separately validate the actual 500-transition
path. These tests do not prove every possible failure point, power-loss or
disk-loss recovery, cross-device bitwise equality, or universal recovery of
every legacy entrypoint. The old P1 consumer has not acquired recovery merely
because the shared controller now offers it.

CPU checks include actual SIGKILL at five checkpoint publication boundaries,
partial writes, corruption, input identity, writer exclusion, real shared-
controller deterministic replay, full health vetoes, lost-parent accounting,
and successful-sibling/queue behavior. The terminal CPU suite has **152 passes,
191 dependency warnings, 21.95 seconds**. GPUs were intentionally hidden for
these reference tests. Compilation and whitespace checks pass.

Independent post-run verification checks **34 committed bundles and 372
serialized tensors**, including hashes, dtype/shape and identities. It repeats
both exact-equality checks, recomputes all four full-call health checks on CPU,
and confirms each first archive completed before the second call started.

## Tuning and numerical health

The inherited eight-pair grid, epsilon `(0.055, 0.06, 0.07, 0.075)` times
L `(3, 8)`, failed on both fresh charts with
`verification_chain_without_movement`. This is a tuning-candidate veto, not
evidence that either backend or the research direction is invalid. The charts
were retained rather than retrained or seed-shopped. Halving the failed minimum
0.055 four times gives the repair grid
`(0.0275, 0.01375, 0.006875, 0.0034375)` times `(3, 8)`.
Disjoint repaired tuning roots add 10,000 to the original roots' second integer;
the public measured-grid tuner and independent heldout verification are
unchanged. Verified selected handoffs are factor epsilon **0.01375, L=3** and
strict epsilon **0.0275, L=3**. These are scope-bound selections, not defaults
for another chart, seed, or task, and not a statistically supported ranking.

Each full call contains 500 transitions times four chains, or **2,000 sampled
states**. All four calls pass finite state, accepted target, log acceptance,
`delta_H = -log_accept_ratio`, required sampled-state target status and
valid-score flags, required eigenvalue/floor checks, and movement in every
chain. Movement now means some movement within the chunk, not merely a final
state different from the initial state. Explicit signatures are state
`[4,4] float64` and seed `[2] int32`; one trace and HLO hashes are recorded.

| Diagnostic, descriptive unless a stated veto | Factor first / repeat | Strict first / repeat |
|---|---:|---:|
| Required full health | Pass / pass | Pass / pass |
| Sampled states checked | 2,000 / 2,000 | 2,000 / 2,000 |
| Observed acceptance | 0.9635 / 0.8240 | 0.5765 / 0.5610 |
| Maximum absolute energy error | 6.9939 / 740.4365 | 2,681.8273 / 2,631.5522 |
| Finite log-acceptance values below -1000 | 0 / 0 | 39 / 12 |

The inherited -1000 count is explicitly explanatory only under the predeclared
policy. It is **not** a native divergence count or a newly imposed continuation
veto. Native TFP divergence counts and proposed-state scores are not exposed by
this shared trace; they are not reported as zero or checked. Acceptance and
finite extreme energies do not establish convergence or posterior validity.

One telemetry field, `min_innovation_eigen_gap`, is positive infinity because
q=20 has observation dimension one: there is no pair of innovation eigenvalues.
This sentinel is defined in
`bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py:839`, with
observation dimension in `bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py:90`.
It is not a nonfinite target, score, eigenvalue, or energy. The first component
inspector incorrectly rejected it. Its repair permits only this named positive-
infinity gap with verified scalar observation dimension, retains raw telemetry,
and continues to reject NaN, negative infinity and other required nonfinite
quantities. Six focused cases test that exact boundary.

## Complete-controller timing and affordability

Both workers ran concurrently. The full-runtime wave took **4,280.09 seconds
elapsed** and consumed **7,101.66 aggregate worker-seconds**. This establishes
process overlap, not a controlled parallel-versus-serial speedup estimate.

| Seconds | Factor | Strict |
|---|---:|---:|
| First actual 500-transition call | 1,398.5691 | 2,140.3011 |
| Repeated actual 500-transition call | 1,413.2088 | 2,129.1275 |
| Parent-measured worker lifetime | 2,822.0831 | 4,279.5734 |
| Measured fresh setup used in forecast | 822.2040 | 1,236.2784 |
| Observed per-chunk external overhead | 2.4054 | 2.3751 |
| Six-chunk estimate including setup/startup/overhead | **9,305.4221** | **14,040.6141** |
| With one repeated chunk, its overhead and 30 s grace reserved | **10,751.0363** | **16,202.1166** |
| Reserve-inclusive estimate fits 14,400 s | **Yes** | **No** |

The six chunks are the declared bounded P1-shaped schedule of four warmup
chunks and two retained chunks, not a claim that warmup or retained R-hat gates
will pass. Actual diagnostics deliberately stop before the third warmup call.
The estimate uses first-call cost plus five repeats, measured setup/startup and
observed external overhead. The one-chunk interruption reserve and 30-second
grace were declared before measurement, not chosen after seeing the result.

Factor's nominal estimate is about **2 h 35 min**, or **2 h 59 min** with reserve.
Strict's is about **3 h 54 min**, or **4 h 30 min** with reserve: **1,802.12 seconds
over the arm cap**. Together the reserve-inclusive fresh-setup estimates total
**26,953.15 seconds**, versus **24,647.44 seconds remaining** in this allocation.
Thus removing the strict arm-cap issue alone would not automatically fund a
fresh unchanged two-arm run plus its declared reserves. Do not drop reserves,
increase the owner's ceiling, spend prior campaigns' balances, reuse diagnostic
draws as untouched validation, or promise convergence to make the arithmetic fit.

These are descriptive forecasts from one repeated call per arm, not confidence
bounds. Similar first/repeated costs do not support a compile-only explanation.
Different charts, seeds, tuned epsilon and GPU context prevent interpreting the
factor/strict difference as a statistically supported backend ranking.

## Component localization and next repair

The bounded sidecar profiles the exact initial and terminal four-chain banks
from the completed runtime arm, with unchanged target, chart, dtype and XLA
mode. Each component has one trace, retained HLO hash and four warmed repeats
per bank; repeated outputs agree exactly.

| Mean seconds per four-chain bank, initial / terminal | Factor | Strict |
|---|---:|---:|
| Transport forward and log determinant | 0.000576 / 0.000465 | 0.000569 / 0.000462 |
| Physical target value/score | 0.579620 / 0.716080 | 0.914297 / 1.067367 |
| Transformed target value/score | 0.582472 / 0.718056 | 0.914767 / 1.064443 |
| Sampled-state target status | 0.578927 / 0.716974 | 0.914768 / 1.064462 |

Target evaluation dominates transport on these banks. Status costs about
another target evaluation, consistent with
`bayesfilter/inference/tempered_target_tf.py:602`, which calls the bridge's
value/score/status operation again, and
`bayesfilter/inference/batched_value_score.py:828`, which forwards transformed
state telemetry. Isolated XLA timings are not an additive decomposition of the
fused controller; subtracting them from full runtime would not prove a speedup.

The next justified repair is to investigate carrying already-computed
accepted-state value/score/status through the controller rather than evaluating
status again. **No such optimization is implemented or validated here.** Its
evidence contract must preserve accepted-versus-rejected state selection,
status semantics, numerical outputs and stateless trajectory, full health,
batch-native GPU/XLA execution and interruption recovery. Inspect the existing
status-carrying mechanics before introducing another route. Use an optional
source-bound path, focused rejection/status equivalence tests, a short GPU
parity/recovery canary, then fresh full-call measurements on both arms under
the same remaining ledger. Retune if numerical or tuning-scope identity changes;
never silently restamp an old handoff.

Only a new reserve-inclusive forecast that fits both arm caps and the then-
remaining aggregate balance can support P0-L closure. After that, integrate the
checkpoint-aware path into the actual P1 consumer and refresh its executable
plan/source/readiness checks. A current source audit alone cannot substitute
for those engineering and numerical checks. P2's chart/posterior/reference and
uncertainty requirements remain separate later scientific work.

## Attempt and budget ledger

Accounting is the sum of parent-measured GPU-worker lifetimes, including setup,
compilation, failed attempts and cleanup. It is neither utilization-weighted
kernel time nor the elapsed time of parallel waves.

| Attempt | Disposition | Aggregate seconds |
|---|---|---:|
| `reference-177b233307` | Inherited-grid movement veto on both fresh charts | 551.492915 |
| `reference-76dd2d4e9a` | Factor completes; strict unfinished tuning hits setup timeout | 2,008.324226 |
| `reference-d2ecbe5728` | Factor committed work reused; strict completes within repaired setup cap | 1,281.620865 |
| `interrupt-8851aaacb7` | Both intentional mid-call SIGKILL injections recorded | 106.508829 |
| `resume-dab326f736` | Factor resumes; strict queue matching bug prevents launch | 49.188418 |
| `resume-c124e1671c` | Strict-only resume after queue/sibling repair | 64.067426 |
| `runtime-ad2ac13153` | Both full runtime/health measurements complete | 7,101.656477 |
| `component-profile-f5a9a88b58` | Inspector incorrectly rejects scalar-gap sentinel | 86.179962 |
| `component-profile-daee5a058c` | Both corrected component profiles complete | 103.516467 |

Final consumption is **11,352.55558313601 seconds = 3.153487662 GPU-hours**.
Remaining is **24,647.44441686399 seconds = 6.846512338 GPU-hours**; reserved
time is **zero**. Every attempt and arm settlement reconciles with its measured
lifetime. No unused reservation is a new allocation. Both the older readiness
and September 8 tuning ledgers retain their previously recorded SHA-256 hashes.

`process-cleanup-verification.json` checks 32 recorded process receipts and
reports `PASS_NO_CAMPAIGN_GPU_WORKERS_REMAIN`. The trusted post-run inventory at
`2026-09-09T09:15:56Z` shows GPU 1 back at 18 MiB and GPU 0 at 337 MiB; unrelated
GPU 0/display contexts remain untouched. No further GPU work is necessary to
establish this result, and the remaining budget is preserved for a discriminating
repair rather than spent repeating successful diagnostics.

One secondary shell console log, `preflight/canary-repair-r1-launch.log`, was
inadvertently reused by a later redirect. This loss is recorded in the grid-
repair note. All worker logs, immutable jobs/results/summaries, tensors,
checkpoints and budget evidence remain available. No scientific receipt was
reconstructed from that overwritten console log.

## Decision table

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close P0-I recovery checks for the checkpoint-aware route | Exact two-arm killed/resumed equality; committed work reused | Identity/corruption/timeout/accounting checks pass | Coverage is bounded, not every failure mode | Preserve protocol; test P1 integration separately | Live CUDA-state, power-loss or universal legacy-route recovery |
| Close P0-J sampled-state health measurement | All four full calls pass complete declared health | No required finite/status/movement veto | Unexposed proposed scores/native divergences; strict finite extremes | Retain telemetry; preserve semantics in repairs | Posterior validity or absence of native divergences |
| Complete P0-K localization | Two-bank component receipts on both arms | Output-repeat/signature/memory checks pass | Isolated kernels need not add to controller cost | Test accepted-state status reuse | Implemented speedup or superiority |
| Keep P0-L and P1 blocked | Strict reserve-inclusive forecast fails; fresh two-arm total exceeds remaining balance | Resource forecast veto, not target invalidity | Single-repeat forecasts and future repair cost | Repair, remeasure, reconcile budget and actual P1 consumer | Phase 0 pass, completed P1, or research-direction rejection |

## Inference status

| Evidence class | Finding |
|---|---|
| Hard veto screen | Initial grid candidates fail movement; repaired handoffs and four full diagnostic calls pass their required health screen. Strict fails the reserve-inclusive resource screen. |
| Statistically supported ranking | None: no predeclared paired uncertainty analysis establishes a backend or tuning ranking. |
| Descriptive-only differences | Acceptance, energy extremes, first/repeat runtime, component means, allocator peaks and the six-chunk forecasts. |
| Default-readiness | Not established; no sampler, transport, tuning, posterior or generic backend default is promoted. |
| Next evidence needed | Source-equivalent status-reuse and recovery checks, complete-controller timing with reserve, valid P1 integration, and later independent sequential/posterior comparisons with uncertainty. |

## Post-run red team

The strongest alternative explanation for a predicted optimization benefit is
that standalone status cost is not removable from the fused controller, or
carrying it changes rejection-state diagnostics. Only numerical/status parity
and full-call measurements can resolve that. Two endpoint banks underrepresent
the trajectory and one steady call does not bound tail runtime. The four-hour
resource conclusion could change with a measured source-equivalent repair;
it cannot change merely because the nominal strict forecast is under four hours.

Recovery evidence is strongest for committed-bundle integrity and the tested
same-GPU mid-call interruption. Its weakest boundary is untested hardware/disk
failure and unfinished setup-stage granularity. The successful mechanics and
health checks do not repair the older chart-quality or posterior-evidence gaps.
What failed is the current runtime-affordability screen and earlier localized
tuning/harness attempts, not the target, mathematical model, or research
direction. Continue with the specific cost repair rather than abandoning the
program or prematurely opening P1/P2.
