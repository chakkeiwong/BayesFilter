# TFP NUTS Gaussian Benchmark

Date: 2026-05-03

Purpose: record a minimal, reproducible test for the recurring proposal that
BayesFilter should use TensorFlow Probability NUTS as the default solution to
HMC pathologies. The benchmark deliberately uses a static-shape standard
Gaussian target, not a filtering likelihood. This makes the result a lower-bound
implementation test: any overhead observed here comes before SVD filters, DSGE
models, custom gradients, missing-data logic, or model-specific failure
handling are added.

Script:
- `docs/benchmarks/benchmark_tfp_nuts_gaussian.py`

Command used for the recorded run:

```bash
python docs/benchmarks/benchmark_tfp_nuts_gaussian.py \
  --num-results 128 \
  --num-burnin-steps 64 \
  --repeats 4 \
  --output docs/benchmarks/tfp_nuts_gaussian_benchmark_2026-05-03.json
```

The script defaults are intentionally smaller than the recorded run so future
agents can rerun a quick smoke benchmark before proposing NUTS as a remedy.

## Interpretation Rule

This benchmark does not test statistical superiority of NUTS over fixed-step
HMC. It only tests whether TFP NUTS should remain a default implementation
hypothesis when the project needs a transparent, fully compiled, filter-aware
HMC backend.

If NUTS is materially slower than fixed-step HMC on this Gaussian target, then
NUTS may still be useful as a diagnostic or reference sampler, but it should
not be treated as the default answer to BayesFilter performance, compilation,
or numerical-debugging problems.

## Recorded Result

Environment:
- TensorFlow: 2.19.1
- TensorFlow Probability: 0.25.0
- Logical devices: CPU and NVIDIA GeForce RTX 4080 SUPER GPU in this run.

Configuration:
- Dimension: 4
- Chains: 2
- Results per run: 128
- Burn-in steps per run: 64
- Repeats: 4
- HMC leapfrog steps: 3
- NUTS max tree depth: 4
- Step size: 0.25

Result summary from
`docs/benchmarks/tfp_nuts_gaussian_benchmark_2026-05-03.json`:

| Sampler | Mode | First call seconds | Mean warm repeat seconds | Seconds per draw | Status |
| --- | --- | ---: | ---: | ---: | --- |
| HMC | eager | 4.223 | 3.110 | 0.01215 | ok |
| HMC | graph | 0.904 | 0.647 | 0.00253 | ok |
| HMC | XLA | 1.543 | 0.070 | 0.00027 | ok |
| NUTS | eager | 66.832 | 72.868 | 0.28464 | ok |
| NUTS | graph | 14.837 | 15.704 | 0.06134 | ok |
| NUTS | XLA | 3.823 | 1.013 | 0.00396 | ok |

The JSON file is the authoritative timing artifact. The key expected diagnostic
is the ratio between NUTS and fixed-step HMC under each execution mode, with
first-call time interpreted as tracing/compilation plus execution for graph/XLA
modes.

Warm per-draw ratios in this run:
- NUTS eager was about 23.4 times slower than HMC eager.
- NUTS graph was about 24.3 times slower than HMC graph.
- NUTS XLA was about 14.6 times slower than HMC XLA.

## Decision

The BayesFilter monograph should document the following bounded decision:

- TFP NUTS is a useful reference implementation and may be useful for small
  diagnostic experiments.
- TFP NUTS should not be the default production backend for BayesFilter.
- The production path should prioritize a small, inspectable HMC transition
  whose target, gradient, failure handling, adaptation, diagnostics, and JIT/XLA
  contracts are owned by BayesFilter.

This decision remains especially important for SVD-filter and DSGE work because
the Gaussian benchmark omits the hard parts: spectral derivative pathologies,
large state dimension, boundary transforms, missing data, and filter failure
policies.
