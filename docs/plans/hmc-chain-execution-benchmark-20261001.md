# Shared HMC chain execution and machine benchmarking

The question is whether applications can use one library interface to run
filtering-backed HMC in chain batches, separate processes, or batches distributed
across processes, and measure the hardware tradeoff without changing the target
or HMC tuning procedure. Existing `ReusableFullChainHMCRunner` supports a batched
target, `IndependentChainHMCRunner` supports scalar serial/threaded execution,
and specialized harnesses own their own processes. There is no common persistent
process executor with CPU budgets and a reusable machine benchmark.

Implement on `feat/hmc-chain-execution-benchmark-20261001`. Reuse the existing
TFP HMC runner and analytical value/score adapter; do not implement another
integrator or tuner. The public tuning authorities remain `tune_hmc_kernel` and
`tune_fixed_transport_hmc_kernel` as listed in
`HMC_TUNING_INTERFACE_CAPABILITIES`. Changing execution topology must not silently
replay an old topology-bound tuning artifact. The new executor is fixed-kernel
execution infrastructure, with no authority to issue tuning or convergence
claims. NeuTra posterior stopping remains with the shared sequential controller.

The interface accepts an importable target factory plus JSON configuration,
initial states, a fixed kernel configuration, chain batch size, worker count,
CPU cores per worker, and device placement. One chain per process makes the
worker core budget the per-chain budget; a batched worker shares its cores.
Use fresh persistent Python subprocesses whose device/thread environment is set
before TensorFlow or application imports. Validate affinity and oversubscription;
apply and verify memory growth for GPUs. GPU/XLA remains the default; CPU is an
explicit user-selected comparison backend. Non-XLA requires a diagnostic reason.
Reuse static-signature runners and permit live starts/seed/step-size inputs.
No scalar mapping fallback may masquerade as native batching. Factory and target
identity, batch partition, stream policy and runtime provenance are recorded.

Numerical state/recurrences remain TensorFlow/TFP. Python schedules processes,
formats artifacts and assembles completed results only. Each chunk has a unique
output directory and serialized TensorFlow tensors. A worker exception, malformed
result, deadline or parent cancellation cleans up all owned processes and retains
logs; partial chains are not silently accepted. Changes to worker count preserve
the stream for an unchanged batch partition. Changing batch size changes TFP
random tensor shapes; do not require identical trajectories across different
batch partitions. Compare target values/scores at identical states instead.

Provide a CLI and Python benchmark API using user target factories, plus a small
Kalman/analytical-score example that calls an existing batched filter authority.
Benchmark complete HMC chunks and target value/score evaluation. Separate process
startup, first compiled call, synchronized warm calls, end-to-end throughput,
host RSS and TF GPU allocator current/peak; process reservation is distinct.
Keep seeds, target, starts, dtype, fixed epsilon/L and work counts identical.
Rotate layout order across fresh-process repetitions. Record every failure and
hardware/software identity. A fastest observed layout is descriptive hardware
advice, never an ESS, convergence or tuning qualification. Invalid values/scores,
failed health, different target identities or wrong placement veto nomination.

Validation: standard-library config/serialization/lifecycle tests; real CPU/XLA
batched-versus-singleton target values/scores; deterministic replay and unchanged
partition streams across worker counts; dynamic input/one-trace reuse; uneven
chain partitions and ordered outputs; timeout/worker failure cleanup; original
runner parity; one trusted GPU/XLA Kalman smoke; a small CLI benchmark readback.
Use unique artifacts under `docs/plans/artifacts/hmc-chain-execution-20261001`.
Bound this engineering validation to 1800 CPU and 900 GPU process-seconds, at most
two concurrent numerical test workers, and no training, serious posterior run,
package installation or external application edits. Stop a failed unit, fix the
concrete implementation/harness cause and rerun only its affected checks.

Skeptical review: thread counts are not guaranteed per-chain cores when chains
share a process; document and enforce per-worker affinity. Parent TensorFlow
initialization must not leak into child setup. A nominal batch shape does not
prove arbitrary application callbacks are vectorized; require batch-capable
targets, check shapes/independence at frozen inputs and expose that limitation.
Random-stream differences must not be mislabeled numerical regressions. Timing
must synchronize outputs and include IPC separately from kernel time; repeated
calls inside one process are not independent statistical replications. Reusing
the existing runner preserves kernel/authority checks; the benchmark cannot
invent a production tuning artifact. These controls answer the implementation
question within a small engineering budget without reopening the completed
filter-rewrite campaign.

## Implementation checkpoint

The public executor, fresh-process worker, benchmark API/CLI and reference guide
are implemented. Review found and repaired missing rejected-proposal health,
cleanup after broken pipes, blocking input writes outside deadlines, redundant
graphs for identical batch shapes and missing structured reference-failure
reports. Available target telemetry is checked by default. The existing reusable
runner gains an explicit `state_dtype` option; its historical default remains
float64, while the new executor passes the requested dtype explicitly. No kernel,
filter, tuner or posterior stopping algorithm is duplicated.

An initial CPU smoke exposed an omitted diagnostic target scope in the example
factory. The existing authority check rejected it before execution; the fixture
and callers now bind the same diagnostic scope. The first process test then
passed partition/replay/resource checks but its direct-runner comparator lacked
the required seed argument. This test-harness omission was repaired. Float32
execution exposed float64-only host diagnostic conversions in the shared runner;
they now use its existing explicit casting helper.

The CPU/XLA feature and existing runner/status regression suite passed all 52
checks (`cpu-validation-r4`). Three further checks cover automatic display-GPU
refusal, HMC-only dtype enforcement and explicit non-XLA reporting. The expanded
feature suite passed all 12 checks as `cpu-validation-r5`, giving 55 distinct
focused checks across both suites. GPU Kalman CLI revision 1
correctly retained a failure: strict placement caught CPU-only tensor
serialization inside the GPU device scope. Moving serialization to an explicit
CPU artifact boundary repaired it. GPU revision 2 passed singleton/batch target
parity (maximum error 4.44e-16), repeated HMC replay and health, with verified
memory growth and XLA on GPU3. Both process layouts passed. Shared GPU load means
the measured speed difference remains descriptive only.

Terminal review added a dtype check on the HMC callback itself, so calling HMC
without a preceding target evaluation cannot hide an adapter dtype conversion.
Workers and benchmark layouts also compare execution-source hashes to detect
code changes during an experiment. Final GPU revision 3 passed those changes,
the same numerical and health checks, and both layouts. Source and artifact
checksums, commands, environment and compute accounting are preserved in
`artifacts/hmc-chain-execution-20261001/validation-manifest.json`. Validation used
less than a conservative 800 CPU/250 GPU worker-wall-second bound, below the
1800 CPU/900 GPU process-second budgets;
no more than two numerical workers ran concurrently. No scientific claim follows
from these engineering fixtures. High-level tuner/controller topology wiring
and migration of individual applications remain outside this fixed-kernel
execution feature; the reference guide states that boundary explicitly.

Implementation and validation are complete. Terminal review and measured limits
are recorded in [the result note](hmc-chain-execution-benchmark-results-20261001.md).
Feature commit `5189c6771` was pushed on its feature branch. A concurrent remote
main update to `e7d5b480e` caused the first main push to be rejected. It merged
without conflicts, including the shared HMC reference guide. Upstream changes
include the Kalman fixture's persistence-cap parameter and shape preservation,
so merge commit `67b51a3bf` was revalidated: all 55 focused CPU checks passed and
the bounded GPU Kalman benchmark passed both layouts with maximum target error
4.44e-16. `artifacts/hmc-chain-execution-20261001/merged-validation.json` records
commands, hashes and evidence. Cumulative conservative worker-wall bounds are
1050 CPU and 350 GPU seconds, within the original budgets. Implementation,
integration and validation are complete. The previous filter-rewrite campaign
remains closed.
