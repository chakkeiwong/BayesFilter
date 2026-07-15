# Kalman QR Gradient Scaling Lattice GPU 0 Plan

Date: 2026-07-14

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `AUTHORIZED_AFTER_SKEPTICAL_AUDIT`

## Question And Research Intent

Complete the repaired LGSSM true-batched analytical/autodiff QR gradient
lattice at `T=120`, `D={10,20,30}`, `P={50,150}`, `B={1,4,16}`, CPU thread
limits `{1,4,16}` in `float32`, and GPU dtypes `{float32,float64}`. Every
method child uses XLA and five synchronized warm calls.

The comparator is `batch_native_analytical_qr_score` versus
`batch_native_autodiff_qr_score` on the same deterministic fixture and
parameter batch. The primary gate is completeness and correctness: 180 method
records and 90 method pairs must pass finite/dtype/shape/direct-output and
analytical/autodiff parity checks. Timing, graph size, scaling, and allocator
magnitudes are explanatory only.

## Evidence Contract

- Exact CPU baseline: the nine passed schedules and 108 records from
  `docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json`.
  Inheritance revalidates the unchanged Kalman math files, QR factorization,
  benchmark contract, each accepted status hash, CPU-only environment, XLA,
  five warm calls, and aggregate checks. Drift in the two GPU harness files is
  explicitly classified as memory-growth/admission-only and preserved.
- GPU candidate: all six schedules rerun fresh on physical GPU 0, exposed as
  logical `/GPU:0`; no prior GPU or canary timing is inherited.
- GPU policy: `TF_FORCE_GPU_ALLOW_GROWTH=true`, explicit TensorFlow memory
  growth before logical-device initialization, no full-device preallocation,
  and allocator current/peak bytes required in every method record.
- Shared-device admission: prelaunch GPU 0 must have only PIDs 5955 and 6575,
  at most 2,048 MiB used, utilization below 50%, and one-minute host load at
  most 64. During execution only those display PIDs plus benchmark-process-
  group PIDs are allowed. Post-run GPU 0 must return to the display-only PID
  set and at most 2,048 MiB.
- XLA policy: input `XLA_FLAGS=UNSET`; GPU children apply and record
  `--xla_gpu_enable_triton_gemm=false`; TF32 is enabled only for float32.
- Hard vetoes: source/inherited artifact drift, timeout/crash, nonfinite or
  wrong dtype/shape, parity failure, missing growth/allocator evidence, new
  foreign GPU 0 PID, host load above 64 after one retry, OOM, CPU placement,
  or failure to release the benchmark context.
- Explanatory only: all timing, graph, shared-device utilization, and allocator
  values. GPU 0 display activity makes timing unsuitable for ranking.
- Nonclaims: no statistically supported speed/superiority ranking, physical-
  core pinning, exclusive-GPU timing, universal framework/hardware conclusion,
  HMC/posterior/default/production readiness, or scientific validity.

Artifact root:
`docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14/`.
Result:
`docs/plans/bayesfilter-kalman-qr-gradient-scaling-lattice-gpu0-result-2026-07-14.md`.

## Prior Attempt And Repair Trigger

The `gpu0_r1` attempt produced 108 revalidated CPU records and 48 admissible
GPU records, then rejected both `gpu-b4-float64` executions because an
unrelated process entered physical GPU 0 after each schedule began. The method
children themselves completed with valid structured status; this is a shared-
device overlap veto, not XLA, numerical, parity, OOM, or memory-growth evidence.
Because the one-retry contract was exhausted, `gpu0_r1` is terminal and must
not be relabeled complete. This `gpu0_r2` run starts from a fresh evidence root,
inherits only the validated CPU schedules, waits for the same prelaunch gate,
and reruns all six GPU schedules without weakening any gate.

## Skeptical Pre-Execution Audit

Passed subject to focused tests and a final trusted prelaunch census. The
baseline is not a weak proxy: it is the exact CPU lattice and only unchanged
CPU numerical evidence is inherited. The GPU canary is not promoted into the
lattice; all six schedules rerun. Memory growth fixes allocator reservation
but does not prove largest-cell capacity, so each fresh schedule can still veto
on OOM or missing telemetry. The `<50%` rule is a prelaunch foreign-load gate;
the benchmark's own runtime utilization is expected. Shared display activity
prevents timing rank claims but does not invalidate correctness/parity or
allocator telemetry. Stop on a genuine structured failure rather than skipping
the cell.

## Exact Command

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py \
  --output-root \
    docs/benchmarks/kalman_qr_gradient_scaling_lattice_gpu0_r2_2026-07-14 \
  --inherit-passed-cpu-from \
    docs/benchmarks/kalman_qr_gradient_scaling_lattice_r4_2026-07-14/status.json \
  --method-timeout-seconds 600 \
  --resource-wait-seconds 7200 \
  --resource-poll-seconds 30
```

## Stop And Handoff

On complete, require 15 passed schedules, 180 rows, 108 inherited CPU records,
72 new GPU records, all aggregate/comparator checks true, five warm calls per
record, logical `/GPU:0` placement, growth/allocator telemetry on every GPU
record, and display-only GPU context after exit. Write the result with decision,
inference-status, manifest, resource caveats, and post-run red-team sections.
On a true veto, preserve the failed stage and stop without weakening the
growth, parity, placement, PID, load, or cleanup gates.
