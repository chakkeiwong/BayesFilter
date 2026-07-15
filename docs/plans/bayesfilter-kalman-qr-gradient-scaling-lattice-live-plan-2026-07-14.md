# Kalman QR Gradient Scaling Lattice Live Plan

Date: 2026-07-14

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `R6_BLOCKED_GPU_RESOURCE_NOT_EXCLUSIVE`

## Question

What descriptive XLA first-call, warm-call, and graph-size behavior do the
repaired true-batched analytical and reverse-mode autodiff QR score methods
show across the originally requested LGSSM lattice?

The exact lattice is `T=120`, `D in {10,20,30}`, `P in {50,150}`,
`B in {1,4,16}`, CPU TensorFlow thread limits `{1,4,16}` in `float32`, and GPU
dtypes `{float32,float64}`. The two methods are
`batch_native_analytical_qr_score` and
`batch_native_autodiff_qr_score` only.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Exact question above. |
| Comparator | True-batched analytical versus true-batched autodiff on the same deterministic nested fixture and parameter batch within every cell. |
| Candidate mechanism | Analytical forward sensitivity versus one reverse-mode VJP; both use the same QR likelihood target. |
| Expected failure mode | XLA compile timeout/crash, non-finite output, wrong dtype/shape, parity failure, source drift, or resource contamination. |
| Primary pass criterion | All 180 method records complete and pass finite/dtype/shape/direct-output checks; all 90 analytical/autodiff cell pairs pass comparator parity. |
| Promotion veto | Any invalid method record or comparator parity failure. A failed schedule is retained and stops expansion. |
| Continuation veto | Source drift; missing/corrupt structured evidence; CPU-schedule foreign CPU work above 16 logical CPUs or load above 64 after one retry; GPU-schedule foreign GPU 1 PID, ownership/cleanup breach, or load above 64 after one retry; GPU 1 not exclusive within two hours; or failure to preserve process cleanup. Foreign CPU samples alone are explanatory on GPU schedules. |
| Repair trigger | A device-relevant resource veto retries the same schedule once: CPU schedules use the foreign-CPU/load limits, while GPU schedules use load/GPU-ownership/cleanup limits. Method failure localizes by stage and stops the lattice for inspection. |
| Explanatory diagnostics | Trace time, first executable call, five synchronized warm calls, GraphDef nodes/bytes, CPU thread limit, GPU dtype, batch size, and process/device snapshots. |
| Must not be concluded | No statistically supported speed ranking, physical-core pinning, universal hardware/framework result, HMC/posterior correctness, production/default readiness, or scientific validity. |

## Evidence Contract

- Scientific/engineering question: the descriptive scaling question above.
- Exact baseline: the analytical true-batched score method; autodiff is the
  comparator on identical fixture, parameter rows, dtype, device, JIT, and
  TF32 policy.
- Primary criterion: correctness/completeness, not runtime rank.
- Hard vetoes: invalid output, failed parity, timeout/crash, source drift,
  missing provenance, or unresolved resource overlap.
- Explanatory only: all timing and graph-size values. Five warm calls share one
  compiled process and are not independent statistical replications.
- Not concluded even on pass: any superior/faster/best claim. Claim-grade
  ranking would require prospectively paired independent-process replications
  and uncertainty analysis after this lattice.
- Result preservation: incremental raw schedule directories plus
  `docs/benchmarks/kalman_qr_gradient_scaling_lattice_r6_2026-07-14/status.json`,
  compact `summary.json`, and a result note.

## Execution Contract

- Sequential schedules only; no concurrent CPU/GPU benchmark schedules.
- Each schedule contains six `D/P` cells and two isolated method children.
- Each child receives a 600-second timeout and records trace, first XLA call,
  five synchronized warm calls, output materialization, parity, and GraphDef.
- CPU sets `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and records the
  requested TensorFlow/OpenMP thread limit. A thread limit is not physical CPU
  affinity or a physical-core claim.
- GPU exposes physical GPU 1 as logical `/GPU:0`. Before each GPU schedule it
  requires no compute process, at most 512 MiB used, and at most 5%
  utilization. During a schedule, GPU 1 compute PIDs must belong to the
  benchmark-owned process group; the benchmark's own memory and utilization
  are expected and are not contention. Any other GPU 1 PID is a veto. Trust
  basis is
  `owner_designated_managed_session_visible_gpu_trusted`.
- GPU/process census failures fail closed. During a live child, GPU 1 must show
  either an owned compute PID or remain below the idle-memory threshold. After
  the schedule runner exits, GPU 1 must have no compute PID and must return to
  at most 512 MiB used; a lingering owned context is a cleanup veto.
- Parent `XLA_FLAGS` is unset. Direct GPU/XLA method children apply and record
  the verified benchmark-local
  `--xla_gpu_enable_triton_gemm=false` policy. CPU remains unchanged.
- Before each CPU timing schedule, wait for no other known benchmark worker
  consuming more than 16 logical CPUs in aggregate, and require one-minute
  host load at most 64 on this 256-CPU machine. During GPU schedules, foreign
  CPU usage is explanatory only because timings are not promotion evidence;
  every five-second sample is preserved, and host load above 64 remains a
  veto. GPU schedules additionally require strict GPU 1 ownership and cleanup.
  If a device-relevant veto fires during a schedule, retain that attempt and
  retry once. Never kill or displace another lane.
- The supervisor hashes all execution-affecting Kalman sources, runners, and
  this plan before every schedule and stops on drift.

### R6 Continuation Boundary

`r4` is terminal `complete_with_failures`. Its nine CPU schedules passed and
its first GPU schedule completed twice with all 12 structured XLA/parity
records valid, but both GPU attempts were excluded by a harness bug: the
strict prelaunch GPU-idle predicate was reused during execution, so the
benchmark's own physical-GPU-1 PID, memory allocation, and utilization were
misclassified as foreign contention. This invalidates the GPU admission
decision, not the method, GPU placement, XLA compilation, or numerical output.

`r6` may inherit only the nine accepted CPU schedules from
`docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json`.
Before writing `r6/status.json`, the supervisor must verify:

- all nine exact CPU schedule specifications are present and passed;
- only each schedule's final accepted attempt is inherited;
- each accepted attempt has return code zero, no overlap veto, structured
  validity, CPU-only environment, exact thread limit, and input `XLA_FLAGS`
  unset;
- each referenced schedule status still passes all 12 method, five-warm-call,
  aggregate, and comparator checks;
- hashes of exactly these six measurement-affecting files are identical to the
  `r4` manifest:
  `bayesfilter/linear/kalman_qr_tf.py`,
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`,
  `bayesfilter/linear/qr_factor_tf.py`,
  `scripts/kalman_qr_benchmark_contract.py`,
  `scripts/benchmark_kalman_qr_parameter_count_scaling.py`, and
  `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`.

Exactly two control-plane files may differ from `r4`:
`docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py` and
this plan. Their new hashes are frozen for `r6` and checked for drift before
every new GPU schedule. They do not alter an inherited CPU method record or
its measurement. No other execution-affecting file may drift. No `r4` GPU
timing from `r4` or `r5` is inherited. All six GPU schedules rerun under the
repaired device-relevant gate. The combined `r6` summary is valid only if it contains
the 108 revalidated CPU records plus 72 new GPU records.

## Skeptical Pre-Execution Audit

Status: `PASSED_AFTER_CORRECTING_RANKING_AND_RESOURCE_CONTAMINATION`.

- Wrong baseline: corrected by excluding obsolete scalar/Python row-loop arms.
- Proxy promotion: graph size and timing are explanatory; output validity and
  pair parity carry the pass gate.
- Missing stop conditions: per-child timeout, source drift, artifact validity,
  overlap retry, GPU wait budget, and cleanup stops are explicit.
- Unfair comparison: both methods use the same fixture, parameter rows, dtype,
  device, XLA setting, synchronization, and warm-call count within each cell.
- Hidden assumption: CPU thread count is only a TensorFlow/OpenMP limit, not
  core affinity. GPU float32 and float64 are separate descriptive arms.
- Runtime GPU ownership: corrected after `r4` showed that an idle-only
  predicate necessarily rejects the benchmark's own GPU process. Runtime
  exclusivity is now process-group ownership, while strict idle requirements
  remain unchanged at prelaunch.
- GPU host contention: corrected after `r5` showed that a transient 18.6-core
  foreign CPU sample vetoed an otherwise exclusive GPU schedule even though
  one-minute host load stayed below 10. CPU usage remains a hard veto for CPU
  schedules; on GPU schedules it is preserved as explanatory context while
  load 64, GPU ownership, and GPU cleanup remain hard vetoes.
- Environment mismatch: CPU hides GPU before import; GPU requires the managed
  trust basis and exclusive physical GPU 1.
- Misleading pass risk: five within-process warm calls could look precise while
  ignoring between-process variability. Therefore no ranking is permitted.
- Misleading failure risk: another lane can consume CPU/GPU resources. The
  supervisor records overlap and one clean retry instead of attributing it to
  a method.
- Artifact adequacy: child stage records distinguish trace, first XLA call,
  warm execution, timeout, compiler failure, and numerical failure.

## Pre-Mortem

The lattice could pass while misleading us if compiler caches, thermal state,
CPU scheduling, or within-process autocorrelation make timings unusually
stable. Raw calls and device/process snapshots are preserved, but timings stay
descriptive. It could fail because another lane is active rather than because
the algorithm is invalid; the overlap census and one clean retry distinguish
that explanation. A default-Triton failure is not retested here because the
previous discriminator already selected the no-Triton route for this stack.

The first supervisor launch wrote only a pre-run `waiting_for_resources`
status under
`docs/benchmarks/kalman_qr_gradient_scaling_lattice_2026-07-14/status.json`;
no benchmark child launched and no timing was collected. That attempt exposed
an over-conservative any-process gate. It is invalid prelaunch evidence. The
historical `r2` gate then replaced it before that run's measurement.

The first `r2` schedule attempt passed all 12 method/parity records but was
rejected because the continuous monitor still used the superseded any-process
rule. A second attempt was interrupted before completion while correcting that
monitor. Neither attempt is included in the result lattice. The clean `r3`
root uses the same prospective aggregate CPU/load thresholds both before and
during every schedule, plus process-group cleanup on interruption.

The `r3` run accepted the clean `threads=1,B=1` schedule. Both `B=4` attempts
also passed every method and parity check, but were excluded because transient
unrelated tests used 4.08 and 8.41 logical CPUs during five-second samples.
That four-CPU veto was disproportionate on a 256-CPU host and stopped `r3` as
`complete_with_failures`. The clean `r4` run prospectively raises only the CPU
contention threshold to 16 logical CPUs and load 64. It does not reuse `r3`
timings. Strict GPU 1 exclusivity is unchanged.

The `r4` run admitted all nine CPU schedules. `cpu-t16-b4` and
`cpu-t16-b16` each needed one contention retry; their final accepted attempts
passed. The first GPU float32 `B=1` schedule then completed twice with return
code zero, all 12 records structurally valid, logical `/GPU:0` placement, and
the no-Triton policy recorded. Both were falsely vetoed because the runtime
monitor applied the prelaunch idle test to the benchmark's own GPU process.
`r4` therefore stops as a partial harness-blocker artifact with 108 admitted
CPU rows. The repair trigger is `r5`, not a claim that `r4` passed the full
lattice.

The `r5` ownership repair worked: both `gpu-b1-float32` attempts returned zero,
completed all 12 records, passed every aggregate and comparator check, placed
all work on logical `/GPU:0`, recorded the managed-session trust basis and
no-Triton XLA policy, and released physical GPU 1 to 18 MiB with no process.
Neither attempt self-vetoed its owned GPU PID. They were nevertheless excluded
by single five-second foreign-CPU samples of `3367.8%` and `1858.4%`, while
one-minute loads were only `8.28` and `9.90`. This is a GPU timing-harness
policy failure, not XLA, numerical, placement, cleanup, target, or method
failure. `r5` is terminal `complete_with_failures`; no `r5` GPU timing is
promoted or inherited. The repair trigger is the fresh `r6` root.

## Bounded Review Question

READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else:
`docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-live-plan-2026-07-14.md`.
Do not edit, run commands, launch agents, or authorize any boundary crossing.
Question: Is the prospective `r6` repair and continuation contract internally
consistent, correct, feasible, artifact-complete, and boundary-safe, especially
the device-specific split in which CPU schedules retain the 16-core
foreign-work veto while GPU schedules treat foreign CPU usage as explanatory
under a load-64 veto and retain strict process-group GPU ownership/cleanup?
End with `VERDICT: AGREE` or `VERDICT: REVISE`.

## Prelaunch Check And Review Record

- Focused supervisor tests after the ownership/inheritance repair: `10 passed`.
- Python compilation, supervisor self-check, and `git diff --check`: passed.
- Real `r4` inheritance check: exactly 9 passed CPU schedules, 108 method
  records, and the exact six-file measurement hash set validated.
- Claude Opus max-effort review round 1: `REVISE`; it found ambiguity between
  measurement-source hashes and repaired control-plane hashes.
- Repair: enumerated the exact six `r4`-matching measurement files and the
  exact two continuation-root-frozen control files above; their current freeze
  target is `r6`.
- Claude Opus max-effort review round 2: `AGREE` via the primary review path.
- Native post-review audit added fail-closed GPU/process census handling and a
  post-run process/memory release gate; focused checks were rerun and passed.
- Native provenance audit additionally pins the terminal `r4` master status
  and every inherited CPU schedule status by SHA-256, revalidating them before
  and after each new GPU schedule.
- Claude Opus max-effort review round 3 after those material additions:
  `AGREE` via the primary review path.
- A broader mixed historical Phase-6 suite reported `143 passed, 8 failed`:
  one stale `/tmp/kalman_qr_phase6_cpu_xla_gateb_r3` assumption and seven
  pre-existing R2 archive-hash failures. Those failures are outside this
  lattice supervisor and are not treated as `r5` evidence. This lane's focused
  supervisor suite remains `10 passed` and the real 9-schedule/108-record
  inheritance check passes.
- `r5` terminal blocker checks: 9 inherited CPU schedules remain passed; both
  GPU attempts have return code zero and structured validity; each has one
  foreign-CPU overlap sample; no GPU/XLA/numerical/placement/cleanup veto
  fired. `r5/summary.json` correctly remains at 108 admitted CPU rows.
- Claude round 4 returned `REVISE` for a stale `r5/status.json` label; repaired
  to `r6/status.json` and focused checks passed.
- Claude round 5, the configured final round, returned `REVISE` for the
  unscoped ledger phrase `foreign compute overlap` and a historical `r2`
  sentence. Both documentation findings were repaired above. No sixth Claude
  round is permitted; final consistency is carried by focused checks and the
  native skeptical audit. This is not represented as Claude convergence.
- Final native audit after round-5 repair: `10 passed`; compilation,
  supervisor self-check, and whitespace validation passed; a direct policy
  discriminator rejected `3367.8%` foreign CPU for CPU timing, admitted it for
  an idle GPU arm at load `9.9`, and rejected the GPU arm at load `65`; the
  pinned 9-schedule/108-record inheritance set revalidated. The plan and
  implementation now agree on all round-5 findings.
- Final skeptical audit: passed. `r4` invalidated the GPU harness decision, not
  the target or methods; `r5` preserves the same research question and
  correctness gates while repairing only ownership classification and
  inherited-artifact provenance.

## Exact Command

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py \
  --output-root \
    docs/benchmarks/kalman_qr_gradient_scaling_lattice_r6_2026-07-14 \
  --inherit-passed-cpu-from \
    docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json \
  --method-timeout-seconds 600 \
  --resource-wait-seconds 7200 \
  --resource-poll-seconds 30
```

## Stop And Handoff

Stop at the first non-overlap schedule failure and write a stage-localized
result; do not skip the failed cell. A resource-exclusivity timeout is a
resource blocker, not an XLA or algorithm failure. On completion, validate
strict JSON, record exact commands/environment/source hashes/wall time, write
the result with decision and inference-status tables, and state whether any
ranking is statistically supported. The next justified action is either a
focused repair for a failed stage or an independent-process replication plan
for only the scientifically interesting comparisons nominated by the
descriptive lattice.

## R6 Close Record

`r6` ran from `2026-07-13T19:25:02.025465+00:00` through
`2026-07-13T21:25:29.590599+00:00` and exited 2 with terminal status
`blocked_resource_not_exclusive`. Physical GPU 1 remained occupied by a
foreign compute context throughout the 7,200-second prelaunch window. No GPU
benchmark child launched, no GPU timing was admitted, and no XLA, numerical,
placement, or cleanup failure occurred.

The terminal summary contains 108 revalidated CPU rows from nine passed
schedules; all have five synchronized warm calls and all schedule aggregate
checks pass. The full primary gate remains incomplete at 108/180 records. The
phase result is
`docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-result-2026-07-14.md`.

Next-phase handoff: retain the frozen execution contract and sources. Only
after trusted census shows physical GPU 1 exclusive, start a fresh result root,
inherit the same pinned 108 CPU records, and rerun all six GPU schedules. Do
not interpret this resource blocker as an XLA or algorithm failure and do not
relax GPU ownership, cleanup, placement, parity, or JIT gates.
