# q=20 Direct GPU/XLA r2 Budget Preflight Recovery Plan

Date: 2026-07-30
Status: `REVIEWED_READY_FOR_BOUNDED_RECOVERY`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | After repaired q=20 CPU/GPU target identity parity passed, what wall budget is required for the previously declared direct batch-native GPU/XLA training campaign? |
| Candidate or mechanism | Direct q=20 bound target, `(32,32)` and `(64,64)` three-stage ELU IAF transports, batch 100, `lr=2e-4`, XLA JIT, and compiled custom principal square root. |
| Promotion criterion | Preserve at least three finite/status-valid warm update receipts for each architecture; directly measure a static-shape 256-row audit; derive the complete prior-protocol budget with every estimate provenance labeled. |
| Promotion veto | CPU/GPU v2 identity mismatch or source-closure mismatch. |
| Continuation veto | Invalid target status, nonfinite result, memory-growth or GPU/XLA failure, missing receipt, source change after identity issuance, or exhaustion of the exact remaining preflight cap. |
| Repair trigger | A localized harness shape/receipt defect may be repaired under a fresh artifact root when target, method, hardware class, criteria, and total campaign budget are unchanged. |
| Explanatory only | Loss, gradient, clipping, allocator bytes, compile time, update time, validation time, support time, audit time, HLO hash, and campaign-duration projection. |
| Must not be concluded | No tuning selection, candidate rejection, convergence, posterior validity, HMC readiness, architecture ranking, or default readiness. |

## Preserved Evidence And Failure Audit

The first r2 attempt is immutable under
`docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r1/`.
It established exact repaired CPU/GPU target parity and preserved these direct
`(32,32)` receipts before a harness failure:

- target and trainer construction;
- compile-inclusive and warm 64-row validation;
- optimizer updates 1 through 6, including five warm updates;
- two-row hard-status probe; and
- first and repeated support/export probes.

The first 256-row audit failed before target execution. TensorFlow's
`reduce_retracing=True` validation wrapper generalized the leading dimension
after seeing 64 and then 256 rows, while the bound target correctly requires a
static batch size. This is a harness shape failure. It does not invalidate the
target, completed update receipts, numerical status, transport support, or the
NeuTra candidate.

The original plan was materially under-budgeted. The preserved `(32,32)` core
charged `5,006.831091717002 s`, and a full repeated `(64,64)` ladder plus two
256-row audits cannot fit the remaining authorization. Repeating that command
would spend compute on a known failure and would not answer the budget question.

## Revised Evidence Contract

| Item | Contract |
| --- | --- |
| Identity | Reissue CPU-hidden and trusted-GPU identity under this recovery runner's source closure. Complete payload, target signature, adapter signature, static CPU placement, and source hashes must match exactly. |
| Preserved `(32,32)` timing | Read the prior progress artifact by SHA-256. Require completed optimizer updates 1-6, both 64-row validations, status, and both support receipts. Never overwrite or resume its trainer state. |
| New `(64,64)` timing | In one trusted GPU process: construction; optimizer updates 1-4, giving three warm receipts; one compile-inclusive 64-row validation; one two-row status probe; one support/export probe; one explicitly `TensorSpec([256,4])` compile-inclusive audit; and HLO extraction. |
| Static audit repair | Compile the audit through a dedicated XLA function with exact input signature `[256,4]`. Do not reuse the shape-generalizing validation wrapper. |
| Projection | Price four 100-update tuning arms and two 1,000-update final streams exactly as declared in r1. Use architecture-specific measured update costs. For `(64,64)`, use its single compile-inclusive 64-row validation for both first and later validation calls, and its single support call for both first and later support calls. Use the directly measured compile-inclusive `(64,64)` 256-row audit for both architectures and both audit calls. These substitutions are conservative planning estimates, not measured warm-call claims. |
| Contingency | Report unbuffered median/max projections and a 25% convenience contingency. The requested budget is the larger buffered-max architecture scenario. |
| New artifact root | `docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r2/`. |

## Numerical Provenance And Budget

| Number | Provenance | Status and risk |
| --- | --- | --- |
| Original r2 material cap `12,000 s` | Reviewed convenience cap inside user-authorized remainder | Fixed total cap across r1 and r2 artifacts |
| r1 charge `5,007.6122855830035 s` | Material ledger, including identity and failed timing attempt | Measured |
| r2 recovery material cap `6,992.0 s` | Derived floor below `12,000 - 5,007.6122855830035` | Leaves `0.3877144169965 s` unallocated rather than exceeding the total |
| Total authorization remaining `8,177.078434643995 s` | `13,184.690720226998 - 5,007.6122855830035` | Includes nonmaterial reporting reserve |
| Recovery reserve `1,185.0784346439948 s` | Total remaining minus new material cap | Tests, startup, validation, notes, and reporting |
| Three warm `(64,64)` updates | Minimum already declared by the original r2 plan | Small timing sample; report every value and no performance guarantee |
| One 256-row audit | Budget-bounded direct measurement | Compile-inclusive value is reused conservatively; no measured warm-audit claim |
| 25% contingency | Original convenience planning margin | Not a statistical interval or hardware guarantee |

## Skeptical Pre-Execution Audit

- Wrong baseline: no. The preserved and new runs bind the same repaired v2
  target and the recovery reissues identity after the runner/plan source change.
- Proxy promotion: no. Timing and mechanics remain explanatory only.
- Missing stop conditions: no. Identity, source, status, finite, GPU/XLA,
  receipt, and exact remaining-budget vetoes are explicit.
- Unfair comparison: architecture-specific optimizer timings are measured.
  Shared audit pricing uses the larger transport and is labeled as an upper
  planning estimate rather than an architecture comparison.
- Hidden shape assumption: repaired by a dedicated exact `[256,4]` input
  signature and a focused CPU shape test before launch.
- Artifact adequacy: every new operation writes start/completion receipts; the
  preserved prior artifact is hash-bound into the recovery result.
- Misleading success: even a completed projection says only how much the prior
  protocol would cost. It does not establish that the protocol is sufficient
  for a scientific or downstream claim.

Audit decision: `PASS_FOR_BOUNDED_RECOVERY_ONLY`.

## Pre-Mortem

| Failure | Earliest discriminator |
| --- | --- |
| Source changes between identity and timing | Compare live source closure to both identity artifacts at timing start. |
| Exact audit still loses the static leading dimension | Focused graph trace test requires `[256,4]`; audit receipt records static batch size. |
| `(64,64)` costs exceed the remainder | Ledger check before every operation and outer timeout below the recovery cap. |
| One audit timing understates later calls | Reuse the compile-inclusive value for every audit call and label it a conservative planning substitution. |
| Environmental load changes update cost | Report all per-update values, use warm maximum, then apply the separate 25% convenience margin. |

## Planned Commands

1. Focused CPU tests for recovery surface, exact-shape audit tracing, receipt
   parsing, and projection arithmetic.
2. CPU-hidden identity issuance in the r2 root.
3. Trusted GPU identity issuance with memory growth on an idle device.
4. CPU-hidden fail-closed identity comparison.
5. Trusted `(64,64)` recovery timing with an outer timeout no larger than the
   recovery material cap.
6. CPU-hidden projection, artifact/hash validation, result note, and reset memo.

No tuning, final training, or HMC is authorized by this recovery plan.
