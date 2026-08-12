# q=20 CPU-XLA 32x1-Chain NeuTra-HMC Canary

Date: 2026-08-01
Tier: bounded CPU mechanics/performance canary
Status: `READY_TO_EXECUTE`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Can 32 one-chain CPU/XLA NeuTra-HMC processes run concurrently under one dedicated supervisor core, and what wall time does that imply for the policy-minimum HMC transition count? |
| Exact input | Seed A best trainer state at step 1,500 from the completed q=20 CPU-XLA training campaign. |
| Exact comparator | One process with one chain, using the same checkpoint, target, transport, HMC graph shape, and warm-call count as each p32 worker. |
| Primary pass | The p1 and p32 arms restore the exact checkpoint, compile with XLA, see no GPU, emit finite samples/traces, preserve declared affinity, and exit successfully. |
| Vetoes | Checkpoint/hash or target/transport mismatch, visible GPU, missing XLA receipt, nonfinite output, worker crash, affinity mismatch, artifact failure, aggregate worker RSS above 64 GiB, or 1,200-second cap. |
| Explanatory diagnostics | Cold compile time, synchronized warm window, aggregate chain transitions/second, speedup, parallel efficiency, and RSS. |
| Derived estimate | `p32 warm window / 3 * 3,000 transitions * 4 leapfrog steps`; this is a linear workload estimate, not a measured long-chain runtime. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-chain-hmc-canary-2026-08-01/r1/summary.json` plus per-arm results and worker logs. |
| Nonclaims | No HMC tuning, convergence, posterior correctness, CPU default, GPU comparison, or proof of linear timing in chain length/leapfrog count. |

## Frozen Topology And Mechanics

- Supervisor: CPU `32`, pinned by the launch command.
- Workers: 32 independent persistent processes on physical CPUs `0..31`.
- Matched baseline: one persistent process on CPU `0`.
- Each worker runs one chain with its own seed and a start selected cyclically
  from the four declared q=20 starting offsets.
- Each timed call executes one burn-in plus two returned transitions, one
  leapfrog step per transition. One call is therefore three chain-transition
  leapfrogs.
- One cold compile-plus-execute call and two synchronized warm calls are
  recorded per worker.
- TensorFlow, OpenMP, and BLAS threads are fixed to one. CUDA is hidden before
  TensorFlow import. The q=20 batch-native target and HMC wrapper both use XLA.
- The checkpoint is restored, frozen, and reloaded independently in every
  worker. No scalar target or sample-wise map is used.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
| --- | --- | --- | --- |
| 32 workers + one supervisor | User-selected topology | Uses 33 physical cores without supervisor contention | NUMA or memory contention; synchronized p32 window and RSS expose it |
| One chain per process | Derived from final-validation parallelism | Chains are independent while transitions within a chain are sequential | Batch-size-one target may be slower than four-chain batching; matched p1 baseline measures exact shape |
| Two warm repetitions | Convenience canary choice | Enough to detect gross feasibility/contention within a short run | Insufficient for statistical ranking; timing remains descriptive |
| Three-transition call | Smallest existing HMC mechanics shape with burn-in and returned draws | Bounds canary cost while exercising full HMC mechanics | Fixed overhead may distort long-chain projection; estimate is explicitly linear and unproven |
| Four leapfrog steps in projection | Existing q=20 kernel hypothesis | Matches the planned validation kernel hypothesis | Fresh tuning may select another count; recompute estimate after tuning |
| 3,000 minimum transitions | Repository sequential policy | 2,000 warm-up plus 1,000 retained per chain | More draws may be required by R-hat/ESS; report minimum only |

## Skeptical Pre-Execution Audit

- Wrong baseline: no; p1 and p32 use the same new one-chain worker executable.
- Proxy promotion: no; finite short calls and speed do not establish sampler
  validity.
- Missing stop: no; startup/round timeouts, 64 GiB worker-RSS veto, and a
  1,200-second total cap are declared.
- Hidden thread multiplication: controlled by affinity and all framework/BLAS
  thread settings fixed to one.
- Compile contamination: cold time is separate; p32 warm calls begin only after
  all 32 workers are ready.
- Unfair timing: each warm round uses a common future monotonic start time.
- Environment mismatch: the worker fails if a GPU is visible and records
  TensorFlow version, XLA, affinity, checkpoint, target, and transport identity.
- Misleading extrapolation: the long-run estimate assumes linear scaling with
  transition and leapfrog work; diagnostics, chunk I/O, tuning, and extra
  transitions are excluded.
- Resource feasibility: the previous p4 canary measured about `0.99 GiB` RSS
  per worker, projecting about `31.7 GiB` for p32 on a 251 GiB host.

Audit decision: `PASS_FOR_BOUNDED_32X1_CHAIN_CPU_XLA_HMC_CANARY`.

## Command

```bash
taskset -c 32 timeout 1200 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/benchmark_ssl_lstm_q20_cpu_xla_32x1_chain_hmc_2026_08_01.py \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-chain-hmc-canary-2026-08-01/r1
```

