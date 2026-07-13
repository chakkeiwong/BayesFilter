# Phase 15 Result: Manual-Score LGSSM XLA Compile Gate

Date: 2026-07-08

## Scope

This result closes the Phase 15 manual-score LGSSM XLA compile gate. The phase
tested the current no-`GradientTape` LGSSM affine NeuTra objective with
`jit_compile=True` on trusted GPU and recorded compile-time and compilation-size
proxies.

The phase did not run `jit_compile=false`, NeuTra training, optimizer updates,
HMC sampling/tuning, external sample generation, DSGE/c603, route ranking,
default-policy changes, or scientific/product/readiness claims.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `PASS_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE` |
| Primary criterion status | Passed: trusted GPU diagnostic compiled and executed with `jit_compile=True`, finite value/gradient diagnostics, timing, and size proxies. |
| Veto diagnostic status | Passed: no `jit_compile=false` runtime run, no CPU runtime evidence, no optimizer update/training, no HMC, no sample generation, current signatures matched. |
| Main uncertainty | This is a compile gate only. It does not establish training quality, posterior correctness, HMC convergence, or production readiness. |
| Next justified action | A future subplan may authorize GPU NeuTra training with `jit_compile=True` only, using these signatures and preserving no-HMC/no-sample boundaries. |
| What is not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, broad XLA readiness beyond this exact compile gate, or scientific validity. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the no-tape LGSSM affine NeuTra objective compile and execute under trusted GPU XLA with `jit_compile=True`, and what are the compile-time/size diagnostics? |
| Baseline/comparator | Phase 14A manual-score target signatures and previous Phase 13 taped-XLA blocker as stale failure context only. |
| Primary criterion | Trusted GPU diagnostic compiles with `jit_compile=True`, executes two calls with finite value/gradient diagnostics, and records compile-time and size proxies. |
| Veto diagnostics | Any `jit_compile=false` runtime run, CPU runtime evidence, hidden optimizer update/training, hidden HMC sampling/tuning, hidden sample generation, stale signature reuse, nonfinite diagnostics, unsupported readiness/scientific/product claims. |
| Explanatory diagnostics | First/second call timing, compile-time proxy, concrete graph byte size, HLO text byte size, device manifest, target/adapter signatures. |
| Artifact | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-diagnostic-2026-07-08.json` |

## Run Manifest

| Field | Value |
| --- | --- |
| Trusted GPU probe | `nvidia-smi` passed: NVIDIA GeForce RTX 4080 SUPER, driver `591.86`, CUDA `13.1`. |
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_xla_repair_tf` |
| Runtime evidence | Trusted GPU/XLA only. |
| JIT policy | `jit_compile=True`; `jit_compile=false` runtime was not run. |
| Target signature | `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038` |
| Adapter signature | `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900` |
| Old taped target signature | `290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb` |
| Batch size | `16` |
| Optimizer update | Not run. |
| NeuTra training | Not run. |
| HMC | Not run. |
| External sample generation | Not run. |
| TensorFlow XLA log | TensorFlow logged `Compiled cluster using XLA!` |
| Diagnostic JSON SHA-256 | `3b3df0592e47f5503e931e66f70f8dd6648839b3d99e7428ee5f03a62016231a` |

## Compile Diagnostics

| Diagnostic | Value |
| --- | ---: |
| First call wall time | `52.11530358507298` seconds |
| Second call wall time | `0.07660454418510199` seconds |
| Compile-time proxy | `52.03869904088788` seconds |
| Concrete graph serialized size | `3164271` bytes |
| Compiler IR HLO text size | `20368338` bytes |
| Total diagnostic elapsed time | `75.12749586394057` seconds |

Finite checks:

```text
first_loss_finite = true
first_shift_gradient_finite = true
first_raw_scale_gradient_finite = true
second_loss_finite = true
second_shift_gradient_finite = true
second_raw_scale_gradient_finite = true
```

All recorded outputs were on `/GPU:0`; TensorFlow soft device placement was
disabled during the diagnostic.

## Local Checks

- `python -m py_compile bayesfilter/testing/neutra_xla_repair_tf.py tests/test_neutra_xla_repair_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_xla_repair_tf.py tests/test_batched_value_score.py tests/test_lgssm_generic_target_adapter_tf.py -q`: passed, `36 passed, 1 skipped, 2 warnings`. These were source/config checks only, not runtime evidence.
- `python -m json.tool docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-diagnostic-2026-07-08.json`: passed.
- Trusted `nvidia-smi`: passed.
- Trusted GPU/XLA compile diagnostic: passed.

## Nonclaims

- No `jit_compile=false` runtime diagnostic was run.
- No NeuTra training was run.
- No optimizer update was run.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No DSGE/c603 target was used.
- No HMC readiness, posterior correctness, sampler quality, transport
  superiority, production readiness, default-policy change, broad XLA readiness,
  or scientific validity is claimed.

## Next Handoff

The next phase may plan GPU NeuTra training only with `jit_compile=True`. It
must not introduce a `jit_compile=false` fallback run. It must preserve the
current manual-score target and adapter signatures, and it must keep HMC
sampling/tuning and external sample generation out of scope unless a later
reviewed subplan explicitly authorizes them.
