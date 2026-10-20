# HMC chain batches, processes, and machine benchmarks

`HMCChainExecutor` runs filtering-backed HMC through the shared
`ReusableFullChainHMCRunner` and TFP integrator. It supports a native tensor batch
of chains, one process per chain, and batches distributed across persistent
processes. Applications supply an importable target factory rather than writing
their own process lifecycle and TensorFlow initialization code.

The defaults are GPU execution and XLA compilation. CPU execution is an explicit
reference or machine-comparison choice. A non-XLA run requires a diagnostic
reason. Neither executor nor benchmark qualifies a tuning artifact or declares
convergence: use the public HMC tuner and the applicable sequential posterior
controller for those decisions. This interface runs fixed-kernel chunks; it does
not automatically replace process management in existing applications or wire
process layouts into the high-level tuner and NeuTra controller. Existing
topology-bound tuning artifacts cannot be silently reused for a new topology.

## Choose a layout

For eight chains, these examples mean:

| Layout | Native batch size | Processes | Logical CPU cores per process |
|---|---:|---:|---:|
| `HMCChainLayout(8, workers=1, cores_per_worker=8)` | 8 | 1 | 8 |
| `HMCChainLayout.one_process_per_chain(8, cores_per_chain=2)` | 1 | 8 | 2 |
| `HMCChainLayout(4, workers=2, cores_per_worker=4)` | 4 | 2 | 4 |
| `HMCChainLayout(2, workers=2, cores_per_worker=4)` | 2 | 2 | 4 |

The last case assigns two batches to each process. Workers process their batches
in order and reuse one compiled runner per distinct batch shape. A shorter final
batch is supported. Outputs retain the initial chain order regardless of worker
completion order.

The process backend is tested on Linux and uses POSIX process groups and pipe
selectors. It is not a Windows multiprocessing backend.

Core budgets apply **per process**. Batched chains share that budget; no library
can promise a particular chain an exclusive core while those chains execute one
batched tensor operation. One chain per process provides a per-chain budget.
On Linux the default pins workers to disjoint slices of the caller's allowed
logical CPU affinity and sets TensorFlow intra-op threads accordingly, with one
inter-op thread. Oversubscribed budgets and more workers than batches are
rejected. Affinity is not a reservation against other applications. On systems
without affinity support, explicitly use `pin_cpu=False`; thread settings then
limit numerical parallelism without confining the process to particular CPUs.

GPU workers enable and verify memory growth before device initialization. With
no `gpu_devices`, the shared display-aware policy chooses one eligible GPU,
shared by all workers; automatic selection rejects a display fallback. Pass
`gpu_devices=("GPU-uuid",)` to select a shared GPU,
or one ID per worker to distribute processes explicitly. Separate processes
have separate CUDA contexts and allocators, even on the same GPU. Memory growth
is not a hard memory cap. Benchmark the allocation costs as well as throughput.

## Target contract and execution

A factory is an importable `module:callable` with JSON keyword arguments. It must
return an adapter with:

- `log_prob_and_grad(theta)`: native batched TensorFlow values `[B]` and
  analytical scores `[B, D]` for states `[B, D]`, preserving the requested dtype;
- `adapter_signature()`: a stable identity binding the target, prepared data,
  numerical settings and relevant implementation;
- `value_score_capability()`: the existing reviewed full-chain XLA capability,
  with its target scope and evidence; and
- optionally `target_status_telemetry(theta)`, using the shared status fields.

Keep the factory in an importable module, not a notebook closure. Pass input
paths or ordinary JSON configuration, not TensorFlow objects. Child processes
import the factory after setting devices, CPU affinity, threads and memory
policy. The parent execution API imports no TensorFlow.

Chains must be independent along the leading dimension. Shape checks cannot
prove that an arbitrary target uses native batching or that its score is the
derivative of its value. The executor never supplies a scalar-map fallback.
The benchmark tests target agreement against singleton batches at the provided
states; this can catch cross-chain coupling but does not replace target review
and analytical-gradient tests. A scalar-only filter must acquire a native batch
adapter before it can use this interface, including its size-one batches.

```python
from bayesfilter.inference import (
    HMCChainExecutor, HMCChainKernel, HMCChainLayout, HMCChainTarget,
)

target = HMCChainTarget("my_application.targets:make_filter_target",
                        {"observations_path": "/data/observations.json"})
kernel = HMCChainKernel(
    num_results=128, step_size=0.02, num_leapfrog_steps=12,
    dtype="float64", target_scope="my-reviewed-target-scope",
)
layout = HMCChainLayout(4, workers=2, cores_per_worker=4)
initial = [[0.1, -1.0]] * 8  # Replace with your declared chain initialization.

with HMCChainExecutor(target, initial, kernel, layout,
                       output_dir="runs/filter-chains-001") as executor:
    values = executor.value_and_score(initial)
    first = executor.run()
    second = executor.run()  # Continues from each chain's last returned state.
    replay = executor.run(current_state=initial, seed=(123, 456))

samples = second.load()  # TensorFlow tensor [draws, chains, parameters].
```

Output directories must be new. Every call records ordered tensor shards,
checksums, actual seeds and step size, timings, health, device/CPU placement,
thread configuration and allocator observations. `load()` verifies shard hashes
and assembles tensors at a host output boundary. Raw trace tensors remain in
each shard's `trace` records. `value_and_score()` without an argument evaluates
the original initial bank; it does not advance or inspect the evolving chain
state. Output loading can initialize TensorFlow in the parent; workers remain
independent fresh processes.

Calls accept new starts, seeds and scalar step sizes without retracing. Chunk
length, leapfrog count, target, dtype, batch partition and JIT policy are static.
Automatic seeds fold the HMC call index and then the batch's start index into
the configured root seed. Explicit seeds use call index zero for replay.
Changing worker count preserves streams for an unchanged partition on the same
backend. Changing batch size changes TFP random tensor shapes and therefore
trajectories. Bitwise cross-device or cross-version replay is not promised.

By default, available target status is checked for both accepted and proposed
states. Nonfinite states, target values, scores, momenta, acceptance ratios,
invalid status or a reported native divergence fail the call and retain its
diagnostic tensors. Missing native divergence/status telemetry is recorded as
unavailable, not zero failures. Movement and maximum absolute log acceptance
are reported for caller assessment; the benchmark additionally applies its
movement and declared energy-error vetoes. The executor does not assess ESS or
R-hat. Use the context manager (or `close()`) to reap all workers, including after
errors. Concurrent calls on one executor are rejected.

## Standard benchmark

The built-in example uses the existing batched QR Kalman filter and analytical
gradient on a small float64 linear-Gaussian fixture. It is useful for verifying
installation and the benchmark harness. To choose resources for your model,
benchmark your own factory, prepared data, chain count and fixed HMC kernel.

```bash
python -m bayesfilter.inference.chain_benchmark \
  --chains 8 --layout 8:1:8 --layout 1:8:1 --layout 4:2:4 \
  --draws 32 --leapfrog-steps 3 --step-size 0.03 \
  --repetitions 3 --warm-calls 5 --output-dir runs/kalman-layouts-001
```

Each `--layout` is `CHAINS_PER_BATCH:WORKERS:CORES_PER_WORKER`. Add `--device CPU`
for a CPU comparison; set `CUDA_VISIBLE_DEVICES=-1` before importing TensorFlow
in a deliberately CPU-only parent. The default example requires float64. Custom
factories can use `--dtype float32` with declared `--atol`/`--rtol` suitable for
their numerical contract. No tolerance is relaxed automatically.

```bash
python -m bayesfilter.inference.chain_benchmark \
  --target-factory my_application.targets:make_filter_target \
  --target-kwargs target-config.json --initial-state chain-starts.json \
  --target-scope my-reviewed-target-scope \
  --layout 8:1:8 --layout 4:2:4 --layout 1:8:1 \
  --draws 128 --leapfrog-steps 12 --step-size 0.02 \
  --output-dir runs/my-filter-layouts-001
```

The Python equivalent is
`benchmark_hmc_chain_execution(target, initial, kernel, layouts, output_dir=...)`.
The CLI and function use the same implementation. `--no-jit-compile` requires
`--non-xla-reason` and records a debug/reference exception. It does not qualify
the non-XLA route as the default.

Each repetition creates fresh workers and rotates layout order. The benchmark
compares values/scores against singleton evaluation and requires fixed-state,
fixed-seed HMC replay within a layout, finite/valid health, movement in every
chain, and maximum absolute log acceptance within `max_energy_error` (default
100, a diagnostic veto rather than a tuned scientific threshold). Failures are
retained and excluded from advice. Any failed repetition makes its layout
ineligible. Singleton-reference failure prevents comparisons and produces a
failure report. The benchmark repeats the same HMC work; its chunks must not be
combined and treated as posterior draws.

`report.md` and `result.json` report:

- Worker startup separately from first target/HMC calls. First-call times
  include tracing, compilation and execution, not isolated compiler time.
- Warm target and HMC timings, repetition medians and observed spread.
  End-to-end timings include synchronization, IPC and tensor artifact writing;
  per-worker call times exclude artifact writing but include runner diagnostics.
- Worker host RSS and lifetime peak RSS, and per-call GPU allocator current/peak
  bytes. Sums of worker high-water marks are upper bounds, not simultaneous live
  memory; host RSS can count shared pages in multiple processes. GPU allocator
  counts exclude driver/context and other processes' memory. They are not
  `nvidia-smi` reservation measurements.

The fastest observed eligible layout is descriptive machine advice based on
warm end-to-end time. Check memory, compile/startup cost and timing variability
before choosing it. A larger filter, different horizon, chain count, dtype or
hardware load can change the answer. This benchmark does not establish the
best HMC tuning, effective samples per second, convergence, or posterior quality.
