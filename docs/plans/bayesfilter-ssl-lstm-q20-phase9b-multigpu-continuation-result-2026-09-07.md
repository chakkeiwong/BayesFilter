# Phase 9B multi-GPU continuation result

Date: 2026-09-07
Status: `M4_P0_BUDGET_INFEASIBLE_P1_BLOCKED_P2_BLOCKED`
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-2026-09-07.md`

## Decision

GPU placement is no longer the blocker. The diagnostic executed on an eligible
non-display GPU and completed the factor arm's two 500-transition calls. The
steady-state call took 1,390.70 seconds, similar to the first call's 1,360.86
seconds. The previous slow-chunk observation cannot be explained away as a
one-time compilation cost on this run.

Under the plan's first-call-plus-five-steady-calls extrapolation, the minimum
six-chunk factor schedule requires 8,314.36 seconds **before** chart construction,
tuning, serialization, and reserve. That is incompatible with the unchanged
2,600-second arm cap. This is a descriptive budget forecast, not a statistical
bound on all future runtimes. It is sufficient to refuse the planned P1 launch;
it is not evidence that either sampler is scientifically wrong.

The diagnostic was stopped while the strict arm was underway. There is no
complete strict-arm timing, two-arm forecast, or passing M4-P0 closeout. P1 was
not launched. The shared 5,200-second campaign now has about **183.80 seconds**
remaining, which cannot fund the required fresh diagnostic and P1 schedule.

## Execution and provenance

Fresh attempt root:
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/runtime-diagnostic-20260907T143204Z/`

Its `run_manifest.json` is a **non-passing interrupted-run closeout**, reconstructed
from preserved worker artifacts and the external process-exit observation. It
does not masquerade as the worker's missing complete diagnostic result. It
contains the actual command, environment, launch Git state, target/data identity,
seed namespaces, source hashes, archive checksums, timings, budget settlement,
and evidence limits. The actual start was September 7 at 22:34:25.180687 +08:00;
the directory suffix is only an attempt label. Exit was confirmed at
23:27:28.771055 +08:00. No unrelated dirty work or historical receipt was reverted.

Actual command, with trusted GPU access:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/diagnose_ssl_lstm_q20_phase9b_executable_readiness_2026_09_06.py \
  --output-dir \
  docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/runtime-diagnostic-20260907T143204Z \
  --max-seconds 3307.39
```

The launcher selected physical GPU 1, UUID
`GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`, PCI bus `00000000:21:00.0`.
Selection recorded display disabled/not attached, 0% utilization, and 32,212
MiB free. The UUID was set before TensorFlow import. The independent trusted
TensorFlow probe passed on this GPU with memory growth enabled. This probe was
console evidence, not a substitute for this worker's own memory-policy receipt.

The worker required growth in its launch environment and called the fail-closed
repository growth helper before logical-device initialization. Its returned
memory-policy object was not persisted before the forced stop; that evidence
gap is explicit. The completed factor arm did preserve allocator telemetry:
peak 268,898,048 bytes, with 31,616 MiB free at its boundary check. NVIDIA process
reservation is not treated as live tensor memory. After stopping, trusted
inventory again showed GPU 1 at 18 MiB used and 32,212 MiB free. Existing display
and remote-desktop processes on the other GPUs were left intact.

## Results and accounting

| Quantity | Observed result | Role |
|---|---|---|
| Factor chart construction and cold-scope tuning | Completed; selected epsilon 0.055, L=3 | Fresh tuning candidate, not posterior evidence |
| Factor first 500-transition call | 1,360.858696 s | First-call timing, including any compilation |
| Factor repeated same-shape call | 1,390.700522 s | Steady-state timing; single replication |
| Graph/XLA | One trace; compiler-IR receipt preserved | Engineering check |
| Movement | All four chains moved in both calls | Hard movement screen only |
| Archives | Two warmup-only archives; checksums verified at settlement | Diagnostic evidence, not a P1 retained stream |
| Strict arm | Begun, interrupted; no complete readiness receipt | Unmeasured comparator |
| Factor six-chunk extrapolation | 8,314.361307 s excluding setup/overhead | P1 budget veto, not a runtime lower bound |
| Diagnostic worker wall debit | 3,183.590345 s | Monotonic supervisor exit observation; roughly one-second polling precision |
| Prior spend | 1,832.61 s | Historical timestamp estimate, not retroactively measured |
| Total campaign debit | 5,016.200345 s | Includes historical estimate |
| Remaining / reserved | 183.799655 s / 0 s | No renewed allowance |

The tuning artifact's aggregate veto list includes rejected grid candidates.
The selected candidate was eligible and the tuning result passed; rejected
alternatives do not invalidate that fact. Neither this selection nor observed
acceptance establishes a statistical ranking or posterior validity.

## Interrupted-run audit

The planned external timeout was absent from the actual launch command. A
deadline supervisor was attached at 2,454.33 seconds, before the 3,307.39-second
workload deadline. It started no GPU work and did not alter the running source.
When the factor measurements showed P1 budget infeasibility, TERM was sent to
our PID 203299 at 23:26:29 +08:00. The Python handler did not finish during
compiled work. KILL followed at 23:27:28; exit code was 137. The stop occurred
before the allocation deadline. The manual grace was about 59 seconds, not
the planned 30 seconds; all of it is charged. Neither deadline nor total
campaign spending was exceeded.

The external settlement script preserved ledger snapshots before and after
settling this attempt exactly once. The ledger's `elapsed_wall_seconds` at
administrative closeout includes later bookkeeping latency; the charged worker
duration is the supervisor's exit observation, not that administrative field.

Source inspection found two further reporting limitations. The diagnostic only
writes its factor receipt after both calls, so it cannot distinguish a current
first call from a current second call through durable stage files. Earlier
console progress descriptions claiming the first call was still running were
therefore too strong. The diagnostic also did not preserve full per-step target
status, finite log-acceptance, and declared energy-veto checks in the arm receipt.
Its `PASS_READINESS_ARM` label establishes only the narrower checks listed here;
it does **not** establish full P1 health readiness.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Placement | Passed on non-display GPU 1 | No load/headroom veto at launch or factor boundary | Future shared load can change | Re-probe at every future launch | No exclusive reservation |
| M4-P0 | Not passed | Infeasible arm forecast; incomplete strict runtime and required telemetry | Per-transition cost and strict timing | Repair diagnostic instrumentation, then budgeted profiling | No complete runtime clearance |
| P1 | Blocked | Only 183.80 s remain; factor minimum forecast exceeds its cap | Cost after a verified performance repair | No launch or automatic cap expansion | No candidate rejection |
| P2 | Blocked independently | Chart thresholds and downstream reference/uncertainty gates remain open | Transport quality and posterior mixing | Preserve the separate scientific plan | No posterior or default claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Budget feasibility fails; complete two-arm readiness and full trace validation are missing. Movement, finite states/targets, XLA, and factor archive checks pass. |
| Statistically supported ranking | None. The strict arm is incomplete; there is one diagnostic seed stream per measured call. |
| Descriptive-only differences | First-call and steady-state times, allocator values, tuning diagnostics, and acceptance. |
| Default-readiness | Not established. Prior narrow numerical-backend admission is unchanged. |
| Next evidence needed | Complete trace-valid runtime measurements, then fresh sequential sampling and downstream uncertainty/reference checks under an affordable declared campaign. |

## Post-run static verification

The placement helper now rejects unknown display attachment instead of
silently classifying it as non-display. P1 forecast validation rejects missing
physical identity and nonfinite capacity; a source-audit file missing at launch
is reported as a P1 contract error before import. Synthetic entrypoint tests
initialize CPU execution with GPUs explicitly hidden before simulating the
selection environment; no real GPU telemetry is used by those tests.

Post-run audit `r14` passes with no findings. The unified focused CPU suite
passed **95 tests** with 191 dependency deprecation warnings in 8.94 seconds;
the separate route-policy suite passed **6 tests** in 2.23 seconds. Python
compilation and `git diff --check` passed. Ledger verification confirmed exactly
one attempt settlement, zero reserved seconds, consistent remaining-budget
arithmetic, and all preserved manifest artifact checksums. These checks do not
close the remaining runtime/telemetry work.

The focused command and its durable log are:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
  tests/test_display_gpu_policy.py tests/test_campaign_budget_ledger.py \
  tests/test_ssl_lstm_q20_phase9b_p1_canary.py tests/test_ssl_lstm_q20_phase9b_readiness.py \
  tests/test_neutra_hmc.py tests/test_q20_bridge_identity.py \
  tests/test_ssl_lstm_q20_phase9a_repair_runner.py tests/test_gpu_probe_contract.py
```

`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/validation-20260907-r14/focused-cpu-tests.log`

The separate route command uses the same CPU-hiding environment and interpreter
with `-m pytest -q tests/test_neutra_hmc_route_policy.py`. The r14 audit is at
`docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260907-r14/run_manifest.json`.
It describes post-run source, not a retroactive replacement for the launch's
r13 receipt. Historical source checksums and run outputs remain preserved.

## Next Phase 0 work

1. Preserve this interrupted attempt. Never reuse its charts, tuning, or draws as
   fresh P1 evidence, and never combine it with a later strict arm into a pass.
2. Persist startup memory verification and per-call start/completion records;
   archive a completed call before the next expensive call. Validate the full
   trace with the shared health rules rather than interpreting movement alone.
3. Use an external process-level timeout from launch, with parent settlement
   after forced termination. Python checks between compiled calls are insufficient.
4. Prepare a bounded per-transition profile of target/score, transport, status
   evaluation, and the controller, without changing the target, precision, XLA,
   chain shape, or required P1 schedule. Test any implementation repair first.
5. Obtain an explicit compute allocation before another serious GPU campaign.
   The present remainder cannot pay for a fresh complete measurement. Do not
   guess a larger cap from the factor result alone: strict costs remain unknown.

Post-run red team: the strongest alternative explanation is target/score or
status-evaluation work repeated inside the controller, not necessarily an
intrinsic limitation of HMC. The exact bottleneck has not been profiled. The
single steady call weakens the one-time-compilation explanation but does not
prove a universal cost. A source-equivalent, trace-valid runtime repair could
overturn the budget conclusion. The weakest evidence is missing strict timing
and complete health telemetry, not GPU availability.
