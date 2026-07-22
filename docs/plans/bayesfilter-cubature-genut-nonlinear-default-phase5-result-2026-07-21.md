# Cubature/GenUT Nonlinear Default Program: Phase 5 Result

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_GPU_XLA_MECHANICS_ONLY`

> **Correction, 2026-07-22:** The arbitrary transformed observation fixture is
> not an SV dataset.  GPU/XLA placement, finite output, memory, and CPU/GPU
> finite-program parity remain engineering evidence; no SV evidence survives.

## Outcome

The candidate finite value/recursive-score core executes on the trusted RTX
4080 SUPER GPU with TensorFlow float32, TF32 enabled, XLA compilation, and the
repository memory-growth policy. The repaired smoke uses a genuinely CPU-pinned
non-XLA reference and a GPU-pinned XLA candidate on identical fixed inputs.

The accepted manifest-bearing artifact is:

`docs/benchmarks/artifacts/cubature_genut_gpu_xla_smoke_20260721/attempt06_manifest/result.json`

Attempt 05 remains valid numerical evidence, but it predates the complete run
manifest and is retained as historical evidence rather than the accepted
claim-bearing artifact.

It reports:

| Quantity | Result |
|---|---:|
| Reference placement | CPU:0 |
| Candidate placement | GPU:0 |
| Value absolute difference | `0.0` |
| Maximum score absolute difference | `5.9009e-6` |
| Finite outputs | `true` |
| XLA compiled | `true` |
| TF32 | `true` |
| TensorFlow allocator peak | `184,556,544` bytes |
| Hard-valid smoke | `true` |

The fixture is exact transformed-SV with `d=1`, `N=12`, and `T=2`. It is a
placement/compilation smoke only. It does not establish high-dimensional
scaling, target-horizon precision, FP64 agreement, or model-row admission.

The candidate identity repair adds process-local issuance sealing and explicit
callable/source dependency fields. Those dependency fields are still supplied
by the experimental caller and are not yet tied to an inspectable repository
callable closure. This is sufficient for tamper detection in the candidate
tests, but not sufficient for canonical/admission identity.

## Harness Repairs

Attempts 02 and 03 were not valid CPU/GPU parity evidence. The original
reference was not explicitly device-pinned. Attempt 03 and attempt 04 showed
that TensorFlow soft placement moved the reference graph to the GPU despite an
inner CPU context. The harness now disables soft placement before graph
construction, builds shared inputs on CPU, wraps the reference invocation in a
CPU device context, and records both result devices. An unsupported CPU kernel
therefore fails closed instead of silently changing the comparator.

## Checks

| Check | Result |
|---|---|
| Candidate identity, filter, and adapter tests | `11 passed` CPU-hidden |
| GPU memory growth | Verified before logical-device initialization |
| GPU/XLA smoke | Passed as `attempt06_manifest` |
| Failed attempts | Preserved under the same artifact root; not used as evidence |
| Canonical Contract E route | Unmodified |
| NumPy in candidate runtime | None added |

## Decision Table

| Decision | Status |
|---|---|
| Tiny GPU/XLA execution feasibility | Passed |
| CPU/GPU placement contract | Passed after fail-closed repair |
| High-dimensional scaling | Not tested |
| FP64/reference precision gate | Not tested |
| Full-horizon nonlinear evidence | Not established |
| Default or leaderboard readiness | False; policy unchanged |
| Next justified action | Use the completed scalar scaling diagnostic to prioritize staged XLA graph repair and a real `d>1` adapter |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Tiny smoke passes; prior invalid placement attempts vetoed |
| Statistically supported ranking | None |
| Descriptive-only differences | CPU/GPU numerical difference is descriptive smoke evidence only |
| Default readiness | Not eligible |
| Evidence needed next | Staged/loop-native scaling, FP64/reference arm, model-scope tuning, and same-target Contract E comparisons |

## Nonclaims

This result does not establish exact nonlinear filtering, unbiasedness, score
precision at the target horizon, method superiority, high-dimensional
feasibility, HMC readiness, leaderboard admission, or a NAWM result.
