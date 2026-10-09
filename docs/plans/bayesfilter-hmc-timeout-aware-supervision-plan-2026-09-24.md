# HMC timeout-aware supervision repair plan

Status: bounded engineering repair executed and reviewed, 2026-09-24. The
[result](bayesfilter-hmc-timeout-aware-supervision-result-2026-09-24.md) records
passing tests, the GPU extension/termination check and remaining campaign gaps.

## Question and scope

The C2 confirmation workers were making numerical tuning progress when their
890-second process caps expired. The supervisor did not pass the remaining
deadline into native tuning, did not record machine-load evidence at the kill
boundary, and stopped the rest of a cell after every non-complete child. This
repair makes the execution boundary progress-aware and workload-aware while
preserving the candidate-set and statistical contracts.

The scope is the isolated validation executor and its public ordinary tuning
caller. It covers deadline propagation, durable progress observation, machine
workload diagnostics, bounded adaptive grace, failure classification, and
continuation of independent fits. It does not change epsilon/L admission,
R-hat policy, posterior estimators, confirmation denominators, or the frozen
C2 source/design. A future continuation must use a fresh versioned output root
and retain the original cumulative fit and cell budgets.

## Evidence contract

The engineering question is whether a slow but productive worker can reach its
declared budget without being killed solely because the host is contended, and
whether an unproductive or invalid worker still terminates promptly. The
baseline is the current isolated supervisor and native ordinary tuner.

Promotion criteria are deterministic focused tests showing that:

* a fit deadline reaches native preparation and candidate-set tuning;
* durable numerical-file progress is detected without treating a changing log
  or heartbeat alone as numerical progress;
* a declared extension is granted only when progress and trusted contention
  evidence are both present, and never past the cell deadline;
* a stalled worker is terminated with an explicit reason;
* a timed-out fit is retained as unavailable evidence while later independent
  fits continue; failed, corrupt, or missing-assessment workers still stop the
  affected cell; and
* cumulative prior elapsed time is charged across retries and cannot be reset.

Promotion vetoes are unexpected source/design drift, invalid checkpoint or
assessment, missing required telemetry for a policy that requires it, a
process surviving termination, or any test that permits a timeout extension
past the outer cell budget. Runtime speed, GPU utilization, and the number of
verified candidates are explanatory diagnostics, not evidence of sampler
quality or statistical power.

The result artifacts are this plan, the updated master program, focused test
receipts, and per-attempt supervision records. Focused CPU runs test engineering
mechanics only. After they pass, a bounded GPU/XLA canary may test the real
telemetry and full public ordinary route under the evidence contract below.

## Timeout policy

`fit_process_timeout_seconds` remains the base cumulative fit allocation. A
design may explicitly declare a `timeout_policy` with:

* a polling interval;
* a no-progress stall interval;
* a finite extension pool; and
* a load threshold for trusted host/GPU contention.

The extension pool is a declared budget, not an automatic right. At a polling
boundary the supervisor may extend the base deadline only if a new numerical
checkpoint/evidence/chunk file was observed, the workload probe reports trusted
contention, and extension remains inside the design's outer cell deadline. A
worker that is busy but not advancing does not receive more time. A worker
advancing on an unloaded machine receives no contention extension. Every
decision, observation, and reason is written to the process exit receipt.

The default policy for existing designs is monitoring with no extension, which
preserves their serialized contract. A continuation that wants adaptive grace
must declare it in a new design identity. The extension and stall numbers are
reviewed execution hypotheses, not scientific defaults; the plan records them
per design and charges all extension seconds to the same cell budget.

Machine telemetry is advisory and fail-closed for extension: normalized CPU
load is a host-wide heuristic, read without importing a framework. The trusted
supervisor queries bounded `nvidia-smi` device and process records, matched to
an explicitly selected GPU UUID. Numeric CUDA ordinals are ambiguous and remain
diagnostic-only for extension decisions. Missing or
untrusted GPU telemetry records `load_unknown` and cannot justify extra time.
The worker's own process is excluded when compute-process PIDs are available.

## Skeptical pre-execution audit

The audit found four material risks and their controls:

1. **A proxy could be mistaken for progress.** File mtime alone is insufficient;
   progress requires a new completed preparation operation or committed tuning
   or posterior work. Preparation events are parsed and deduplicated by work
   identity; tuning and posterior control bundles also require valid checksums.
   The numerical resume loader verifies tensor payloads. Logs are context only.
2. **Load could be mistaken for external contention.** GPU utilization from the
   worker itself must not grant grace. Extensions require trusted telemetry and,
   a foreign compute PID on the selected GPU. CPU runs use the declared host
   saturation heuristic. Unknown telemetry never extends a cap.
3. **Grace could silently invalidate the campaign budget.** The cell deadline is
   passed to the supervisor and is an absolute ceiling. Base plus extension
   seconds are charged cumulatively; a retry cannot recover an exhausted slot.
4. **A timeout could be treated as scientific failure or stop all evidence.** A
   local timeout is an unavailable fit and continuation trigger. Source,
   artifact, missing-assessment, and nonzero-exit failures remain cell-stop
   conditions until diagnosed.

The resumed implementation audit found three further risks. Changing a remaining
deadline inside a frozen numerical configuration changes identity and may change
random streams; instead a context-local execution allowance composes with the
existing numerical limits, and an atomic parent-owned record communicates any
extension. Partial posterior summaries must not be cached as terminal results;
committed chunks remain resumable. Finally, the outer supervisor must supply
its actual monotonic deadline and reserve reporting time, so starting the child
cannot reset the cell clock. These repairs are required before promotion.

The revised plan passes this skeptical review for bounded implementation and
tests. Its baseline, scientific thresholds, full denominators, source identity
checks, and continuation vetoes remain explicit. A test success establishes
execution mechanics, not the cause of the historical slowdown or confirmation
size/power. Corrupt progress and process leaks veto engineering completion.

## Execution phases

1. Add a framework-free timeout-policy and workload-observation module with
   strict validation and deterministic fake-probe support.
2. Apply a dynamic execution scope to ordinary preparation/native search,
   resumed candidate-set tuning, and posterior chunks, preserving numerical
   configuration, seeds, and partial checkpoints.
3. Replace the unconditional process wait with progress-aware supervision,
   bounded contention grace, explicit receipts, and cumulative accounting.
4. Continue after local timeouts while stopping on invalid or unknown failures;
   preserve all denominators, candidate evidence, and prior receipts.
5. Add unit tests for policy validation, progress classification, trusted-load
   extension, hard outer deadlines, process termination, continuation, and
   retry accounting. Add a small isolated integration test that exercises the
   actual public pipeline without GPU or long sampling.
6. Update the official book, aligned reference and master, run focused checks
   plus import/compile checks, and record limitations. Preserve the frozen C2
   design and exhausted slots. A new diagnostic is not a confirmation retry.

## Default and numerical assumption audit

The base 890-second cap and C2 statistical design are inherited historical
allocations, not new defaults. Extensions default to zero and stalls default to
disabled, preserving existing design allocations. Polling at 1 second, telemetry
at 30 seconds, a 1-second timeout for each of two GPU queries, a 60-second recent
progress window and at most 5 seconds (also capped at one tenth of an allocation)
for cooperative shutdown are convenience engineering hypotheses. They affect
observability and interruption latency, not inference. Fake-clock tests check
boundaries; real child tests check delivery and cleanup. A host normalized load
threshold of 1 is a saturation heuristic, not proof of foreign contention.
Per-campaign changes require recorded prices and remain within declared budgets.

## Bounded telemetry canary after focused tests

Question: do UUID-matched trusted telemetry, shared execution allowance and the
ordinary public pipeline operate together on the default GPU/XLA route? Reuse
one development normal-conjugate fit as a numerical comparator, preserving its
model, tuning and posterior settings; use a new diagnostic design and output
directory. Maximum expenditure is 600 GPU worker-seconds from the unreserved
balance, with at most two attempts sharing that ceiling. This is a convenience
debugging allocation, not a measured price or a confirmation allocation. Do not
manufacture contention by launching competing workloads. The pass criteria are
valid telemetry, a normal or correctly classified budget exit, preserved tuning
and process records, and no surviving worker. Missing trusted GPU/memory-growth
provenance, corrupted evidence or a budget overrun veto the canary. Acceptance,
runtime and posterior diagnostics are descriptive; no ranking, posterior
promotion, default timeout calibration or historical-cause claim follows. Store
the manifest, exact command/source, receipts and result below
`artifacts/hmc-timeout-supervision-2026-09-24/`; charge actual enclosing worker
elapsed time once. A failure triggers a localized repair and the remaining
attempt only if the same contract and total allocation still hold.

The concrete canary uses a 450-second base, 90-second extension pool and
570-second cell ceiling inside a 585-second enclosing diagnostic cap. These are
convenience allocations within 600 seconds, leaving 15 seconds outside the
enclosing cap for launch/receipt overhead; they are not proposed defaults or a
claim that one fit needs 450 seconds. Stall detection stays disabled because
its compilation/chunk threshold has not been calibrated on this shared GPU.

## Stop and next-step rules

Stop numerical promotion and repair on a failed focused test, an unexpected
identity/checkpoint mismatch, or an extension past the cell deadline. After the
repair passes, the next campaign phase is a bounded telemetry/pricing canary on
fresh output. Only after that can a C2 continuation be planned; it must use the
remaining original allocation, never reset timed-out fits, and report all
planned missingness.
