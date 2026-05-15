# BayesFilter V1 Model B/C BC6 GPU/XLA Scaling Result

## Date

2026-05-15

## Controlling Documents

Master program:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

Phase plan:

```text
docs/plans/bayesfilter-v1-model-bc-bc6-gpu-xla-scaling-plan-2026-05-14.md
```

## Plan Tightening And Drift Check

BC6 required shape-specific GPU/XLA timing statements for Model B/C after
branch-gated shape evidence.  The existing GPU harness was Model B-only, so a
BC6-specific benchmark-only harness was added.  It times Model B and default
Model C value filters and records score branch gates for the same
model/filter/horizon cells before interpreting timings.

No drift found:

- GPU/CUDA/XLA commands were run only with escalated permissions;
- the harness is under `docs/benchmarks` and changes no production behavior;
- Model C score branch metadata uses `allow_fixed_null_support=True`;
- timing claims remain shape-specific and diagnostic.

## Independent Plan Audit

The plan is safe to execute because BC1 and BC3 already provide the shape
envelope for the tested `T in {8, 16}` cells.  BC6 does not depend on HMC
readiness or convergence.  The benchmark skips timing rows if the value or
score branch gate fails, so timing evidence cannot outrun correctness
diagnostics.

## Escalated Device Evidence

Escalated `nvidia-smi` reported:

- GPU: NVIDIA GeForce RTX 4080 SUPER;
- driver: 591.86;
- CUDA: 13.1;
- visible memory: 16376 MiB.

Escalated TensorFlow probe reported:

- TensorFlow: 2.19.1;
- physical devices: CPU and GPU.

## Execution

Benchmark harness:

```text
docs/benchmarks/benchmark_bayesfilter_v1_model_bc_gpu_xla.py
```

Artifacts:

```text
docs/benchmarks/bayesfilter-v1-model-bc-gpu-xla-scaling-2026-05-15.json
docs/benchmarks/bayesfilter-v1-model-bc-gpu-xla-scaling-2026-05-15.md
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarks/benchmark_bayesfilter_v1_model_bc_gpu_xla.py --device-scope visible --timesteps 8,16 --models model_b_nonlinear_accumulation,model_c_autonomous_nonlinear_growth --backends tf_svd_cubature,tf_svd_ukf,tf_svd_cut4 --devices cpu,gpu --modes graph,xla --repeats 2 --warmup-calls 1 --output docs/benchmarks/bayesfilter-v1-model-bc-gpu-xla-scaling-2026-05-15.json --markdown-output docs/benchmarks/bayesfilter-v1-model-bc-gpu-xla-scaling-2026-05-15.md
```

## Results

The benchmark produced 48 timing rows:

- 2 models;
- 3 filters;
- 2 horizons: `T=8` and `T=16`;
- 2 requested devices: CPU and GPU;
- 2 execution modes: graph and XLA.

All 48 rows had `status = ok`.  Every row had score branch metadata
`3/3`, and every Model C row used structural fixed support.

Shape-specific timing interpretation:

- GPU graph mode was slower than CPU graph mode for these small `T=8` and
  `T=16` scalar-observation shapes.
- GPU XLA was much faster than GPU graph for every tested shape.
- GPU XLA was competitive with CPU XLA on several Model C rows and faster than
  CPU XLA for Model C `T=16` cubature/UKF in this diagnostic run.
- CPU graph remained the most consistently fast path for these small shapes.

## Veto Audit

No veto diagnostic fired:

- GPU evidence came from escalated commands;
- no non-escalated GPU failure was interpreted;
- no broad speedup claim is made;
- benchmark code stayed under `docs/benchmarks`;
- production semantics were not changed.

## Interpretation

BC6 passes.  The evidence is useful for shape-specific planning:

- for tiny and small scalar Model B/C shapes, GPU graph overhead dominates;
- XLA can reduce GPU overhead substantially;
- larger horizons, batched parameter grids, or batched independent panels are
  still the appropriate next hypotheses for finding a real GPU payoff.

This result does not certify broad GPU speedup, GPU HMC performance, or
production deployment policy.

## Continuation Decision

BC7 is justified.  BC6 did not name a nonlinear Hessian consumer; it only
strengthened the evidence that performance questions should be handled through
shape/batch benchmarks rather than second-order derivative implementation.
