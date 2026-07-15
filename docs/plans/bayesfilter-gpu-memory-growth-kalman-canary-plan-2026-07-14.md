# GPU Memory Growth Kalman Canary Plan

Date: 2026-07-14

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `CLOSED_PASS_MEMORY_GROWTH_REPAIR`

## Question And Evidence Contract

Question: does the repaired Kalman QR GPU/XLA child use bounded TensorFlow
allocator memory instead of reserving almost all VRAM for the smallest exact
lattice cell `D=10,T=120,P=50,B=1,float32`?

Baseline: prior identical-cell children without memory growth, whose TensorFlow
logs created a 30,421 MiB device allocation and whose NVIDIA process footprint
was about 30,724 MiB. Candidate: the same analytical/autodiff methods and
no-Triton XLA policy with `TF_FORCE_GPU_ALLOW_GROWTH=true` plus explicit
`set_memory_growth(..., True)` before logical-device initialization.

Primary pass criterion: both methods complete on logical `/GPU:0`, XLA is
enabled, output finite/dtype/shape and analytical/autodiff parity checks pass,
each child records memory growth enabled and TensorFlow allocator current/peak
bytes, and the process releases its GPU context after completion.

Vetoes: prelaunch physical GPU 0 utilization at or above 50%; a new foreign
non-display GPU 0 compute process; memory-growth configuration failure; OOM,
timeout, CPU placement, missing telemetry, numerical/parity failure, or context
not released. The pre-existing `gnome-remote-desktop-daemon` and `nxnode.bin`
contexts are authorized shared-device occupants and remain untouched.

Explanatory only: first/warm-call timings, GraphDef size, allocator byte
magnitudes, and shared-device NVIDIA utilization. No performance comparison is
valid on this shared display GPU.

Nonclaims: this canary does not establish the largest-cell memory capacity,
the complete GPU lattice, speed superiority, production/default readiness,
HMC/posterior correctness, or scientific validity.

Artifact: `docs/benchmarks/kalman_qr_gpu_memory_growth_canary_2026-07-14/`
and a result section appended to this plan.

## Skeptical Pre-Execution Audit

Passed. The baseline is the identical smallest cell, not a different problem.
NVIDIA reservation is not treated as live memory; TensorFlow allocator peak is
the primary resource measurement. The canary cannot silently promote timing,
and it stops before larger cells. GPU 0 is allowed only under the owner's
prospective `<50%` utilization authorization, with existing display contexts
preserved and no other lane displaced.

## Command

The focused method-isolated runner will execute dimensions `10`, parameter
counts `50`, `T=120`, `B=1`, `float32`, both primary methods, five warm calls,
GPU/XLA/TF32, no resume, and a 600-second child timeout. GPU 0 is exposed as
logical `/GPU:0`; the exact command and environment will be recorded in the
result.

## Close Record

The canary completed with both method records passing. TensorFlow allocator
peaks were 564,992 bytes for analytical and 2,064,128 bytes for autodiff.
Independent NVIDIA monitoring showed about a 276 MiB increase above the shared
display baseline during execution and a return to baseline afterward, instead
of the prior approximately 30.7 GiB process reservation. The result is
`docs/plans/bayesfilter-gpu-memory-growth-kalman-canary-result-2026-07-14.md`.

Handoff: retain required memory growth and allocator telemetry. Run the next
capacity cell prospectively; do not infer largest-cell capacity or timing rank
from this shared-GPU smallest-cell canary.
