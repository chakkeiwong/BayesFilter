# Matched Kalman QR CPU Process GPU Comparison Plan

Date: 2026-07-15

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `AUTHORIZED_AFTER_SKEPTICAL_AUDIT`

## Research Intent And Evidence Contract

Question: for the same analytical QR Kalman value-and-gradient work at
`(D,P,T,B)=(30,150,12,16)`, what steady-state throughput and memory are
observed for:

- `cpu_native_b16_xla`: one CPU process, one native `B=16` XLA graph, pinned
  to physical CPUs `16..31`;
- `cpu_processes_16xb1_xla`: 16 CPU processes, each with one `B=1` XLA graph
  and one distinct physical CPU from `16..31`;
- `gpu_native_b16_xla`: one native `B=16` XLA graph on physical GPU 0.

All arms use TensorFlow float32, the same deterministic fixture and 16-row
parameter cloud, `build_batch_native_analytic_fn`, XLA JIT, two untimed warm
calls, five measured calls per fresh block, and scalar-sentinel device
synchronization. Compilation, process startup, and output materialization are
excluded from steady-state timing and reported separately. The GPU arm uses
`TF_FORCE_GPU_ALLOW_GROWTH=true`, explicit TensorFlow memory growth, and
`--xla_gpu_enable_triton_gemm=false` because the current repository GPU/XLA QR
path requires that benchmark-local workaround.

Exact baseline: `cpu_native_b16_xla`. Candidates are the same computation
partitioned across 16 CPU/XLA processes and the same native batch on GPU/XLA.
The primary engineering pass criterion is six fresh, parity-valid blocks for
every arm with no placement, XLA, device, memory-growth, resource, cleanup, or
finite-output failure. Pairwise block log-ratio bootstrap intervals and sign
tests are reported. Timing is descriptive unless those predeclared uncertainty
summaries exclude one; the benchmark does not predeclare a promotion threshold
or change a default.

Hard vetoes:

- any arm does not cover proposal IDs `0..15` exactly once;
- value or score shape, dtype, finiteness, or cross-arm float32 parity fails;
- an arm is not XLA-compiled or is placed on the wrong device;
- CPU worker/task affinity or NUMA placement fails;
- GPU memory growth or allocator telemetry is missing, GPU 0 has at least 50%
  prelaunch utilization, or a foreign GPU process overlaps the run;
- aggregate CPU RSS exceeds 16 GiB, GPU allocator peak exceeds 16 GiB, a
  timeout occurs, or child cleanup fails.

Explanatory diagnostics: compilation/ready time, every warm-call duration,
CPU-seconds and effective cores, aggregate CPU RSS, GPU allocator current/peak,
device census, and graph/output shapes. These do not veto unless they cross a
declared resource limit or expose an invalid timing boundary.

Artifact root:
`docs/benchmarks/kalman_qr_matched_cpu_process_gpu_2026-07-15/`.
Result:
`docs/plans/bayesfilter-kalman-qr-matched-cpu-process-gpu-comparison-result-2026-07-15.md`.

## Skeptical Pre-Execution Audit

Passed. The name `XLA` is not treated as an arm: all three arms use XLA, so the
comparison distinguishes native CPU batching, process sharding, and native GPU
batching. The harness runs fresh evidence instead of combining earlier runs
with different schedules and timing boundaries. It times synchronized worker
kernels rather than supervisor IPC, gives both CPU architectures the same 16
physical cores, and validates the exact 16 row outputs. GPU and CPU hardware
are necessarily different resource budgets, so the result is a matched-work
latency/throughput comparison, not an equal-cost or universal compiler test.
The small `T=12` workload can underfill the GPU; that is part of this workload's
answer, not evidence about larger workloads.

Pre-mortem: a fast arm could be invalid because it skipped rows, returned
before device completion, reused a different fixture, or timed compilation
differently. Exact row identity, scalar synchronization, source fingerprints,
fresh-process blocks, separated cold/steady timing, and rowwise output parity
distinguish those failures. CPU contention or GPU overlap could also create a
false ranking; targeted CPU and GPU admission/overlap telemetry is recorded,
and contaminated blocks fail rather than being selected.

## Commands And Stops

Focused checks:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_matched_cpu_process_gpu.py
```

Trusted execution:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_matched_cpu_process_gpu_2026_07_15.py
```

Stop immediately on a hard veto. Preserve partial structured artifacts and do
not weaken parity, XLA, device, growth, contamination, timeout, or memory gates.
If complete, write the result with the exact manifest, raw timings, pairwise
uncertainty, decision table, inference status, limitations, and post-run
red-team note.

## Nonclaims

No universal CPU/GPU/XLA superiority, equal-hardware-cost comparison, larger
`T` extrapolation, default-policy change, HMC/posterior readiness, scientific
validity, or production readiness. This benchmark does not compare XLA to
non-XLA execution because every arm uses XLA.
