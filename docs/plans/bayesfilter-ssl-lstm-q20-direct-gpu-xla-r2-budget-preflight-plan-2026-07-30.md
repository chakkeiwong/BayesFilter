# q=20 Direct GPU/XLA r2 Budget Preflight Plan

Date: 2026-07-30
Status: `REVIEWED_READY_FOR_PREFLIGHT`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Does the repaired q=20 target issue one byte-identical v2 target payload on CPU and trusted GPU, and what wall budget is required for a fresh direct batch-native GPU/XLA training campaign? |
| Mechanism under test | Static target fixture/observations are constructed explicitly on `/CPU:0`; a receipt-heavy diagnostic separates target/trainer construction, initial validation compile/execution, first optimizer compile/execution, warm optimizer updates, warm validation, status, support, and export costs. |
| Exact comparator | CPU-hidden and trusted-GPU SHA-256 hashes of the complete target signature payload; for timing, repeated calls to the same compiled `(32,32)`, batch-100, `lr=2e-4` program in one process. |
| Promotion criterion | Complete CPU/GPU payload hashes, target signatures, and adapter signatures match exactly; at least three post-first-call optimizer-update receipts and one post-first-call validation receipt complete with finite/status-valid outputs; a conservative full-campaign budget can be derived without using the timed-out r1 arm as an unmeasured extrapolation. |
| Promotion veto | Any CPU/GPU identity mismatch or invalid/missing payload hash. This blocks a new campaign until target identity is repaired again. |
| Continuation veto | No trusted GPU, memory-growth failure, non-XLA execution, invalid bound status, nonfinite loss/gradient/parameter, missing per-operation receipt, 12,000-second preflight material cap, or inability to obtain at least three warm-update timings. |
| Repair trigger | A localized diagnostic artifact/receipt defect may be fixed and retried if target, program, hardware class, and total budget remain unchanged. A target or method change requires a new preflight identity. |
| Explanatory only | Loss, gradient, clipping, allocator bytes, compiler time, update times, validation times, support time, and derived campaign-duration scenarios. |
| Must not be concluded | No tuning selection, candidate rejection, training improvement, convergence, HMC readiness, posterior validity, architecture ranking, or default readiness. |

## Evidence Contract

| Item | Contract |
| --- | --- |
| CPU identity artifact | q=20 v2 target payload, canonical payload SHA-256, target signature, adapter signature, construction policy, fixture device, observation device, source hashes. No target value/score call. |
| GPU identity artifact | Same fields, constructed in a trusted GPU process with verified memory growth before logical initialization. Exact canonical equality with the CPU artifact is required. No target value/score call. |
| Timing target | Repository-issued binding for `batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="compiled_custom_op")`, accessed only through the bound trainer proxy. |
| Timing transports | Separate `(32,32)` and `(64,64)` processes, batch 100, `lr=2e-4`, initialization scale `0.01`, clip `10`, three ELU IAF stages. These are r1 warm-start mechanics settings, not promoted defaults or tuning results. Learning rate does not change graph shape, so it is not duplicated for timing. |
| Timing sequence | Per architecture: construction receipt; initial 64-row validation; first optimizer update; warm updates 2 through 6; second 64-row validation; status probe on a two-row fixed batch; first and warm frozen export/reload/support probes; first and warm 256-row audit-shape validations; HLO extraction. Persist a receipt atomically after every operation. |
| Primary timing estimate | Median and maximum of optimizer updates 2 through 6 if at least three complete. First update/validation remain separate compile-inclusive costs. |
| Conservative campaign projection | Price the prior full protocol (four 100-update tuning arms across both capacities and two 1,000-update final streams) for both possible selected final architectures. Sum capacity-specific process construction/compile overhead, warm updates, 64-row validations, first/warm support/export, first/warm 256-row audits, HLO extraction, and 25% contingency. Report a range using warm medians and maxima; this estimates required compute but does not authorize that protocol. |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r1/`. |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Remaining authorization `13,184.690720226998 s` | Derived from user-authorized 20,000 seconds minus the closed r1 ledger charge | Treating unused r1 time as permission to resume r1 | New root, new preflight-only ledger, no r1 mutation |
| Preflight material cap `12,000 s` | Convenience cap below the authorized remainder, reserving `1,184.690720226998 s` for CPU tests, artifact validation, and reporting | Diagnostic consumes all headroom | Check ledger before every operation and stop with receipt |
| Six optimizer updates per architecture | Convenience diagnostic ladder | Too few for long-run thermal/runtime drift | Require five possible warm timings per capacity, report uncertainty and no performance guarantee |
| Three warm timings minimum | Smallest number supporting a median/spread rather than one anecdote | Still underreplicated | Report every timing; projection range uses maximum and contingency |
| `(32,32)` and `(64,64)`, `lr=2e-4` | Prior hard-valid capacity hypotheses; timing only | Capacity can change compile and transport costs | Time each in an isolated process and use its own receipts |
| Batch 100 | r1 mechanics setting | Different batch changes runtime and memory | Bind estimate to batch 100 only |
| 25% contingency | Convenience planning margin, not measured uncertainty | Insufficient for environmental variability | Show unbuffered and buffered totals; final campaign plan must review it |

## Skeptical Pre-Execution Audit

- Wrong baseline: corrected. Identity compares complete canonical payloads, not
  target-signature strings alone. Timing compares calls within one fixed graph
  and separates first-call compilation from warm execution.
- Proxy promotion: timing and mechanics cannot select a candidate or establish
  training quality.
- Missing stop conditions: trusted GPU, memory growth, XLA, status, finite,
  receipt, and wall-budget vetoes are explicit.
- Stale context: r1 is closed; none of its tuning artifacts can be resumed or
  upgraded. Its mechanics timing is historical context only.
- Environment mismatch: CPU and GPU construction artifacts include source,
  TensorFlow, device, and policy provenance.
- Artifact adequacy: every operation writes start/completion timestamps and the
  last completed operation, so timeout no longer hides progress.
- Misleading pass: hardware-invariant identity plus timing still does not show
  that any training protocol is scientifically adequate.

Audit decision: `PASS_FOR_PREFLIGHT_ONLY`.

## Budget And Stop Contract

- Authorized remaining headroom: `13,184.690720226998 s`.
- New preflight material cap: `12,000 s`.
- Reserve outside material cap: `1,184.690720226998 s`.
- Maximum serious operations: one GPU identity construction and two timing
  processes. Each timing process has six optimizer updates, two 64-row
  validations, one two-row status probe, two support/export probes, two
  256-row audit-shape validations, and one HLO extraction.
- No tuning, final training, HMC, package mutation, or external action.
- An outer timeout is a continuation veto, not authority to omit operations or
  extrapolate from an unknown update count.

## Pre-Mortem

| Failure | Discriminator |
| --- | --- |
| CPU/GPU signatures match but payload differs | Compare complete canonical payload hash and equality, not only signatures. |
| First compilation is mistaken for warm update cost | Separate receipts for every call; exclude update 1 from the warm summary. |
| Timeout again hides progress | Atomically replace `progress.json` before and after every operation. |
| Status checks double every update and distort the estimate | Training timing uses the bound proxy's same-call hard status enforcement; separate status telemetry is timed and reported as a validation/support cost. |
| Six updates look stable by chance | Report the small sample, median/max spread, and 25% convenience contingency; make no performance guarantee. |

## Planned Commands

1. CPU-hidden identity artifact.
2. Trusted GPU identity artifact on an idle physical GPU with memory growth.
3. CPU-hidden identity comparison and fail-closed parity artifact.
4. Trusted receipt-heavy timing diagnostic on the same hardware class.
5. CPU-hidden projection/result generation and artifact validation.

Exact commands and realized devices will be recorded in the result artifacts.
