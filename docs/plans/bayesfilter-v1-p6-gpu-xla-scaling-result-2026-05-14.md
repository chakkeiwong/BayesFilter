# BayesFilter V1 P6 GPU/XLA Scaling Result

## Date

2026-05-14

## Governing Plan

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-plan-2026-05-14.md
```

## Phase Scope

P6 tests whether real GPU/XLA execution improves the cost of a nonlinear
SVD-CUT4 value-filter row on a stable branch box.  This is a diagnostic timing
phase only.  It does not change production behavior and does not certify broad
GPU speedups.

## Plan Tightening

The phase plan was tightened operationally as follows:

- use Model B nonlinear accumulation and SVD-CUT4 as the diagnostic target,
  because P4 selected this as the first nonlinear HMC target and P2/P3 already
  showed stable branch metadata for it;
- compare CPU eager, CPU graph, CPU XLA, GPU eager, GPU graph, and GPU XLA on
  one fixed shape;
- write a benchmark-only harness under `docs/benchmarks`, not production code;
- run GPU-visible commands only after escalated `nvidia-smi`.

No master-plan edit was needed.

## Independent Audit

Audit question:

```text
Does P6 provide truthful hardware diagnostics without promoting them to
general claims?
```

Audit findings:

- GPU/CUDA visibility was checked with escalated `nvidia-smi`.
- The GPU-visible TensorFlow run reported both CPU and GPU logical devices.
- The CPU-hidden control used `CUDA_VISIBLE_DEVICES=-1` and reported only CPU.
- The new benchmark harness is confined to `docs/benchmarks` and imports the
  existing testing/model helpers.
- No production filter behavior or public API was changed.
- All timing rows report branch metadata; the Model B branch was `3/3` with no
  active floors, weak spectral gaps, or nonfinite rows.

Veto diagnostics checked:

| Veto | Result |
| --- | --- |
| Non-escalated GPU failure treated as real CUDA failure | clear; GPU probe/run used escalation |
| Broad speedup claim from one tiny artifact | clear; result is limited to one fixed Model B shape |
| Benchmark changes production behavior | clear; only benchmark/docs artifacts added |
| Branch instability ignored in timing rows | clear; branch rows are `3/3` with no blockers |

## Artifacts

Benchmark harness:

```text
docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py
```

CPU-hidden control:

```text
docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.json
docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.md
```

GPU-visible diagnostic:

```text
docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json
docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.md
```

## Commands

Escalated GPU probe:

```bash
nvidia-smi
```

CPU-hidden control:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py \
  --device-scope cpu \
  --repeats 2 \
  --warmup-calls 1 \
  --timesteps 24 \
  --backends tf_svd_cut4 \
  --modes eager,graph,xla \
  --devices cpu \
  --output docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.md
```

Escalated GPU-visible diagnostic:

```bash
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py \
  --device-scope visible \
  --repeats 2 \
  --warmup-calls 1 \
  --timesteps 24 \
  --backends tf_svd_cut4 \
  --modes eager,graph,xla \
  --devices cpu,gpu \
  --output docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.md
```

Documentation/code checks:

```bash
python -m py_compile docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py
git diff --check
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"
```

## Results

Hardware probe:

- NVIDIA GPU visible: RTX 4080 SUPER class device;
- TensorFlow GPU-visible run created `/device:GPU:0`;
- XLA compiled at least one cluster in the GPU-visible run.

CPU-hidden control, Model B SVD-CUT4, `T = 24`, `14` points:

| Device | Mode | Branch OK | Mean steady seconds | Status |
| --- | --- | ---: | ---: | --- |
| CPU | eager | 3/3 | 0.088768 | ok |
| CPU | graph | 3/3 | 0.004820 | ok |
| CPU | XLA | 3/3 | 0.000391 | ok |

GPU-visible diagnostic, same shape:

| Device | Mode | Branch OK | Mean steady seconds | Status |
| --- | --- | ---: | ---: | --- |
| CPU | eager | 3/3 | 0.086636 | ok |
| CPU | graph | 3/3 | 0.005458 | ok |
| CPU | XLA | 3/3 | 0.022035 | ok |
| GPU | eager | 3/3 | 0.305209 | ok |
| GPU | graph | 3/3 | 0.061104 | ok |
| GPU | XLA | 3/3 | 0.022038 | ok |

## Interpretation

P6 confirms that the nonlinear SVD-CUT4 Model B diagnostic can execute under
GPU-visible TensorFlow and XLA on this machine.  For this tiny fixed shape,
GPU execution is not faster than the fastest CPU control row.  The only safe
performance statement is therefore:

```text
GPU/XLA is operational for the tested Model B SVD-CUT4 shape, but this artifact
does not support a broad GPU speedup claim.
```

This is consistent with the expectation that a 2-dimensional, 24-step,
14-point diagnostic is too small to amortize GPU launch and compilation costs.
Future GPU work should use a shape ladder with larger state dimension, longer
horizons, batched parameter points, or batched independent filtering problems.

## Gate Result

P6 primary gate passes:

- the artifact states whether GPU/XLA helps for the tested shape;
- the claim is explicitly limited to that shape;
- no veto diagnostic fired.

## Next Phase Justification

P7 remains optional.  P3 still labels full exact nonlinear likelihood for
Models B-C as blocked, but no current V1 claim depends on such a reference.
P7 should therefore be a decision/result phase unless a stronger nonlinear
reference is needed for a specific public claim.
