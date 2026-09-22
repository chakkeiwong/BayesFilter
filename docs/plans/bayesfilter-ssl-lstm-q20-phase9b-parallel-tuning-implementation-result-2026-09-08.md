# Phase 9B process-parallel tuning implementation result

Date: 2026-09-08  
Status: `CPU_INTEGRATED_GPU_VALIDATION_UNFUNDED`  
Program state: `M4_P0_BUDGET_INFEASIBLE_P1_BLOCKED_P2_BLOCKED`  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md`  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`

## Outcome

The new tuning-only entrypoint launches the factor and strict cold-scope jobs
in separate Python processes. It calls the existing fresh-chart builder and
`_tune_scope`; the public measured-joint-grid tuner still owns candidate
measurement, selection, and fresh heldout verification. Its parent coordinator
uses separate supervision directories, pins one GPU UUID per worker, manages
timeouts and cleanup, checks complete worker results, and is the only writer of
the campaign ledger.

This is working CPU-tested integration, not a completed q=20 GPU experiment.
No new GPU inventory, TensorFlow GPU smoke, tuning run, or P1/P2 run was
launched in this continuation. The old Phase 9A, readiness, and P1 entrypoints
remain serial. Candidate pairs within each new worker also remain serial;
candidate-level process parallelism is possible in principle but is not part
of this bounded change. Sequential HMC transitions cannot be made independent
simply by adding processes.

Implementation paths:

- `bayesfilter/runtime/parallel_tuning.py`
- `bayesfilter/runtime/display_gpu_policy.py`
- `docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py`
- `tests/test_parallel_tuning.py`
- `tests/test_display_gpu_policy.py`
- `tests/test_ssl_lstm_q20_phase9b_parallel_tuning.py`

## Review findings and repairs

The earlier 45-test pass did not establish executable integration. Review found
that the coordinator had no actual tuning consumer, pre-created roots that
existing launchers reject, treated file existence as success, could report a
successful wave after a child failure, and charged fast children until a slower
sibling exited. Its display policy also filled an extra worker slot using the
display GPU while non-display GPUs were eligible.

These defects are repaired. The process tests require both subprocesses to
reach a shared readiness barrier before either can finish, so a serial
implementation cannot pass the overlap test. Additional checks exercise
partial launch failure, missing/malformed artifacts, ignored TERM followed by
KILL, interruption, worker/control-root overlap, incomplete measured grids,
device/memory receipt mismatch, and preservation of completed siblings.
Coordinator integration tests use real CPU subprocesses and synthetic devices
to verify exactly one settlement per worker. A simulated parent record-write
failure preserves completed worker charges instead of charging zero. A
synthetic framework test verifies that the actual worker entrypoint calls the
existing chart builder and scope tuner after persisting startup memory policy.
That synthetic test is not GPU evidence.

One intermediate integration run failed at Python parsing due to an unmatched
parenthesis in the new job constructor. It was corrected before the final
suite. This was an implementation error, not a target, numerical, or resource
failure. All work in this continuation is routine implementation/CPU validation;
no serious-campaign debit or budget expansion was made.

## Resource and evidence boundaries

With two eligible non-display GPUs, the two required tuning jobs run together.
With only one eligible non-display GPU, the second job queues; the manifest
records actual concurrency rather than calling a one-worker wave parallel.
The display GPU is eligible only when no non-display GPU passes the resource
checks, not merely to fill a third slot. No unrelated or display process is
terminated.

Admission requires load at most 40% and free memory for the inherited 4 GiB
allocator-peak estimate plus 5 GiB headroom. Workers verify growth before
logical-device initialization and retain the startup receipt before chart or
tuning work. Parent telemetry checks live free memory every five seconds;
missing telemetry or less than 5 GiB free stops the wave. This is periodic
resource supervision, not a hard allocator limit or a guarantee that free
memory cannot briefly drop between observations. GPU validation must test the
estimate and supervision on real hardware.

Worker timeouts include termination grace. The parent accounts individual
process lifetimes, including import, setup, tuning, and cleanup; overlapping
lifetimes are summed as worker-seconds. It reserves both complete worker caps
before detecting GPUs and never creates or increases a campaign ledger. A
concurrent run may lower elapsed wall time without lowering total compute.
Source/plan/profile identity, job-specific stateless seed roots, original
candidate-grid coverage, startup memory evidence, and tuning artifact checksums
must reconcile before a worker passes. No worker result issues a P1 closeout.

## Validation

The final CPU suite passed **135 tests**, with **191 dependency deprecation
warnings**, in pytest-reported **13.44 seconds**. This includes the six NeuTra
route-policy tests and the existing Phase 9A/P1/readiness regression suites;
do not add their previous standalone counts to this total. Python compilation
and `git diff --check` passed. Plan-only execution passed without loading an
accelerator framework or detecting devices.

All durable records are under:
`docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/validation-r1/`.

| Record | Contents |
|---|---|
| `cpu-tests.log` | Final 135-test suite |
| `compile-whitespace.log` | Compilation and whitespace result |
| `plan-only.json` | Framework-free executable inspection and source/plan identity |
| `historical-ledger-before.sha256` and `historical-ledger-after.sha256` | Identical old-ledger checksum |
| `historical-ledger-summary.json` | Preserved balance and zero reserved seconds |

Actual test command, with GPUs intentionally hidden before framework import:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
  tests/test_parallel_tuning.py tests/test_display_gpu_policy.py \
  tests/test_ssl_lstm_q20_phase9b_parallel_tuning.py tests/test_campaign_budget_ledger.py \
  tests/test_ssl_lstm_q20_phase9b_p1_canary.py tests/test_ssl_lstm_q20_phase9b_readiness.py \
  tests/test_neutra_hmc.py tests/test_q20_bridge_identity.py \
  tests/test_ssl_lstm_q20_phase9a_repair_runner.py tests/test_gpu_probe_contract.py \
  tests/test_neutra_hmc_route_policy.py
```

Compilation used the same interpreter with `-m py_compile` on the three new or
modified runtime/entrypoint files and their three focused test files, followed
by `git diff --check`. The plan-only command is recorded in the execution plan.
Synthetic subprocesses import only standard-library modules and never
initialize an accelerator framework; synthetic UUIDs are not GPU observations.

## Budget preservation

The original campaign ledger is unchanged byte-for-byte. SHA-256 before and
after this continuation:
`1961d07cf421b06abfb6ec4b62830b44e4052849505d72cd3cf0cea087160415`.

| Original campaign field | Seconds |
|---|---:|
| Total allocation | 5,200.0 |
| Consumed | 5,016.200344768003 |
| Reserved | 0.0 |
| Remaining | 183.79965523199735 |

The historical remainder is insufficiently supported for a fresh complete
two-worker validation schedule. No larger allocation or runtime timeout has
been invented. The next serious launch needs a plain-language allocation
decision and an existing funded ledger; no special approval wording or token
is required. Source audit r14 remains a historical September 7 static record,
not clearance for these new sources.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Process coordinator engineering | Passed CPU process/integration tests | Synthetic crash, timeout, incomplete receipt and accounting checks pass | Real GPU startup and resource behavior untested | Bounded trusted GPU smoke after allocation planning | No q=20 GPU speedup or numerical equivalence |
| Parallel q=20 tuning diagnostic | Wired to original builder/tuner; no real target execution | No funded complete schedule or real worker receipts | Startup/compile time and hardware contention | New bounded allocation, then complete factor/strict tuning diagnostic | No tuning admission from synthetic tests |
| Full P0/P1 readiness | Not passed | Prior full trace/startup gaps, incomplete strict controller timing, and infeasible factor forecast remain | Cost of the actual sequential controller | Continue P0-I/J/K/L instrumentation and profiling under a funded plan | Tuning parallelism does not resolve the roughly 8,314-second factor sampling forecast |
| P2 scientific validation | Blocked separately | Chart-quality, reference/posterior and uncertainty gates remain open | Transport quality and posterior mixing | Preserve the scientific validation sequence | No posterior, whitening, ranking, or default promotion |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Engineering tests reject invalid launches and receipts; no new target-level numerical veto is measured |
| Statistically supported ranking | None |
| Descriptive-only differences | Synthetic process timings and historical single-stream controller timings; neither ranks methods |
| Default-readiness | Not established; this is an optional Phase 0 tuning diagnostic |
| Next evidence needed | Trusted multi-GPU validation, complete budgeted target runs, and a separate paired uncertainty-aware timing design if a speedup claim is desired |

## Post-run red team and next step

The strongest remaining alternative explanation for a future parallel failure
is host/GPU contention or process initialization rather than the q=20 method.
The weakest present evidence is the complete absence of a real multi-GPU
validation run. The tests establish process wiring and bookkeeping, not the
TensorFlow numerical behavior of that wiring. A source mismatch, wrong worker
placement, missing memory-growth receipt, incomplete grid, or failed real
downstream check would overturn execution readiness.

This continuation rejects neither the factor candidate nor the research
direction. It repairs serial orchestration and stops at the unchanged compute
boundary. Next: allocate bounded worker-seconds for trusted GPU validation,
validate the two-worker tuning route, and separately repair/profile the full
controller before attempting P1. Do not rerun the old campaign command or use
the new tuning-only manifest as a full Phase 0 closeout.
