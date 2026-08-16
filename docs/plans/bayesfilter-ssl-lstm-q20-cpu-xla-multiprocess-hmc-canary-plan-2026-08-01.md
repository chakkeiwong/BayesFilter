# q=20 CPU-XLA Multiprocess NeuTra-HMC Canary

Date: 2026-08-01
Tier: bounded CPU mechanics/performance canary
Status: `READY_TO_EXECUTE`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Can independent CPU processes execute the same XLA-compiled q=20 NeuTra-HMC mechanics concurrently, and what aggregate warm throughput is observed at 1, 2, and 4 processes? |
| Exact input | Seed A best trainer state at step 1,500 from the completed parallel CPU-XLA training campaign. |
| Comparator | One process on one physical core, using the same checkpoint, target, transport, HMC shape, and warm-call count as every concurrent process. |
| Primary pass | Every process hides CUDA before TensorFlow import, records no visible GPU, uses `jit_compile=True`, returns finite HMC samples, and exits successfully. |
| Vetoes | Checkpoint/hash failure, target-signature mismatch, visible GPU, non-XLA configuration, nonfinite sample/trace, worker crash, affinity mismatch, missing artifact, or 1,800-second wall cap. |
| Explanatory diagnostics | Process startup, first-call compile-plus-execute time, warm-call time, RSS, per-topology wall time, aggregate transitions/second, and scaling efficiency. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-2026-08-01/r1/summary.json` plus worker logs/results. |
| Nonclaims | No HMC tuning, convergence, posterior correctness, CPU default, GPU comparison, topology superiority, or transport promotion. |

## Frozen Mechanics

- Checkpoint: Seed A `checkpoint-1500.json`, using its bound best trainer state.
- Target: q=20 batch-native TensorFlow target with `tensorflow_eigh` principal
  square root and XLA enabled.
- Transport: restored `(32,32)` dense-IAF state, frozen and reloaded inside each
  process before HMC runner construction.
- HMC: four chains, two retained transitions, one burn-in transition, one
  leapfrog step, fixed step size `0.01`, standard trace, no tuning/adaptation.
- Timing: one first call followed by three warm calls using the same compiled
  runner and distinct stateless seeds.
- Topologies: `1`, `2`, and `4` independent persistent processes. Each process
  is pinned to one distinct physical core on NUMA node 0 with TensorFlow intra-
  and inter-op thread counts fixed to one.
- GPU: intentionally hidden with `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
  import in every child.

Each call performs three HMC transitions per chain (`1` burn-in plus `2`
returned transitions), or twelve chain-transitions across four chains. The
summary reports both process-call throughput and chain-transition throughput;
the former is the clean measure for independent-process scaling.

## Skeptical Pre-Execution Audit

- Wrong baseline: no; all topology arms use the same exact worker command and
  differ only in concurrent process count, affinity, and seed.
- Proxy promotion: no; timing and short mechanics are explanatory and cannot
  establish sampler validity.
- Missing stop: no; worker startup and each arm are timeout-bounded, and the
  complete canary has an 1,800-second wall cap.
- Hidden thread multiplication: controlled by one-core affinity and TensorFlow,
  OpenMP, and BLAS thread variables set to one.
- Compile contamination: first call is reported separately; aggregate warm
  throughput uses synchronized post-compilation calls only.
- Unfair concurrency: workers remain persistent and receive a common future
  start timestamp for each warm round.
- Environment mismatch: CPU workers fail if any TensorFlow GPU is visible and
  record XLA, TensorFlow version, affinity, and target/checkpoint signatures.
- Misleading pass: feasibility and throughput do not make these CPU-trained
  artifacts claim-bearing HMC inputs.

Audit decision: `PASS_FOR_BOUNDED_CPU_XLA_MULTIPROCESS_HMC_CANARY`.

## Launch Repair

The first supervisor launch failed before starting a worker because the new
per-topology output directory was not created before opening its stderr log.
The launcher now creates that directory immediately on entry. This is a local
artifact-path repair: the target, checkpoint, HMC mechanics, topology ladder,
criteria, cap, and output root are unchanged.

## Command

```bash
timeout 1800 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/benchmark_ssl_lstm_q20_cpu_xla_multiprocess_hmc_2026_08_01.py \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-2026-08-01/r1
```
