# Generic LEDH Transport Optimization Run Manifest

- Campaign: `ledh-generic-transport-optimization-20260719`
- Git commit: `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`
- Plan: `docs/plans/bayesfilter-ledh-generic-transport-optimization-master-plan-2026-07-19.md`
- Result: `docs/plans/bayesfilter-ledh-generic-transport-optimization-phase1-result-2026-07-19.md`
- Environment: conda `tf-gpu`; TensorFlow `2.19.1`
- Hardware: NVIDIA GeForce RTX 4080 SUPER; driver `591.86`
- Execution: trusted escalated GPU, XLA JIT, float32, TF32 enabled, 8192 MiB logical-device limit
- Data: repository LGSSM diagnostic generator, dataset seed `81100`
- Model role: performance witness only; no LGSSM-specific operation enters the shared cache helper
- Output policy: unique files under this directory; no prior artifact overwritten

## Serious Runs

| Artifact | Arm | B | N | T | P | Seeds | Warm repeats | Wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `lgssm_t50_n1024_b1_baseline_attempt02.json` | streamed baseline | 1 | 1024 | 50 | 5 | `94101` | 5 | `39.760685` |
| `lgssm_t50_n1024_b1_cached_attempt02.json` | cached geometry/JVP | 1 | 1024 | 50 | 5 | `94101` | 5 | `19.438852` |
| `lgssm_t50_n1024_b16_baseline_attempt01.json` | streamed baseline | 16 | 1024 | 50 | 5 | `94101:94116` | 3 | `127.628427` |
| `lgssm_t50_n1024_b16_cached_attempt01.json` | cached geometry/JVP | 16 | 1024 | 50 | 5 | `94101:94116` | 3 | `30.923875` |
| `lgssm_t50_n1024_b1_cached_batched_reset_attempt03.json` | rejected batched reset candidate | 1 | 1024 | 50 | 5 | `94101` | 5 | `18.945381` |

All serious runs used `sinkhorn_steps=20`, `balance_steps=8`, epsilon `0.5`,
scaling `0.9`, the repository exact-divisor chunk policy with `K=N=1024`, and
the all-active Contract E-Chol reset mask.

## Smoke

- `smoke_t2_n128_cached_attempt01.json`
- B=1, N=128, T=2, P=5, seed `94101`
- Wall time: `15.272698 s`
- Purpose: XLA compilation, graph identity, memory, replay, finite, marginal,
  and reset hard gates before the serious paired runs

## Commands

Every JSON artifact records the exact argument vector in its `command` field.
The executable was invoked as:

```text
/home/chakwong/anaconda3/bin/conda run -n tf-gpu python docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py <artifact command arguments>
```

## Integrity

SHA-256 values:

```text
96817d8030e1121ed7e06efa156b03daff4b97a99962d70ec16f06173393d26a  lgssm_t50_n1024_b1_baseline_attempt02.json
c00d37da179adb92a6c9b086b72a42af640ca317882ca8682304ffc9ad8f394f  lgssm_t50_n1024_b1_cached_attempt02.json
47edb04d6267a7ad280ced4776b7cdf3f52367f62c19cd790e46e6e30329d6ef  lgssm_t50_n1024_b16_baseline_attempt01.json
d745f1fac5471040030aca9ddc7ee9824eecd35af6c628b39dfc5742ab2b0d6c  lgssm_t50_n1024_b16_cached_attempt01.json
158cce498a56f900b742010f0759f1add8d0972c06b5f8951b4ef1793470a703  lgssm_t50_n1024_b1_cached_batched_reset_attempt03.json
1072784659a51b8a58d9a43be04505739e40eb9990c870f81fb62d84a8141e4f  smoke_t2_n128_cached_attempt01.json
```

## Nonclaims

This manifest does not establish a universal speedup, nonlinear-model benefit,
HMC readiness, posterior correctness, leaderboard readiness, or a universal
memory threshold for enabling dense same-cloud caching.
