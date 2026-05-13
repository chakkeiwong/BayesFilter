# Policy: BayesFilter v1 CI And Runtime Tiers

## Date

2026-05-11

## Purpose

This policy separates evidence tiers so BayesFilter developers can run the
right tests without accidentally requiring external projects, GPU access, or
long HMC jobs.

## Tier 1: Fast Local CI

Purpose:
- protect imports and core deterministic contracts.

Examples:

```text
tests/test_v1_public_api.py
selected low-runtime unit tests
```

Policy:
- no external checkouts;
- no GPU requirement;
- no HMC;
- target runtime should stay small enough for frequent local use.

## Tier 2: Focused Local Regression

Purpose:
- protect v1 QR/SVD value, derivative, and compiled parity behavior.

Current command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Observed runtime:

```text
31 passed, 2 warnings in 84.50s
```

Policy:
- run before commits that touch filtering internals;
- CPU-only by default;
- no external checkouts.

## Tier 3: Extended CPU Diagnostics

Purpose:
- characterize benchmark scaling, graph warmup, process memory, branch
  frequency, and derivative/Hessian behavior.

Examples:

```text
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
tests/test_svd_cut_branch_diagnostics_tf.py
QR score/Hessian shape ladders
```

Policy:
- not required on every commit;
- artifacts must state CPU-only scope;
- failures should be recorded per backend/shape rather than hidden.
- tests in this tier should be opt-in when they live under `tests/`.

Current opt-in command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

## Tier 4: Optional Live External

Purpose:
- certify compatibility with observed external project checkouts.

Current MacroFinance result:

```text
/home/chakwong/MacroFinance
0e81988957ef1f8b520014929bea32ffee3881f4
tests/test_macrofinance_linear_compat_tf.py: 4 passed, 2 warnings in 74.54s
```

Policy:
- external source edits are forbidden;
- tests must skip cleanly if checkout is absent;
- dirty external checkout status must be recorded when present;
- passing evidence does not authorize client default switch-over.

## Tier 5: Escalated GPU/XLA-GPU

Purpose:
- establish device availability and matching-shape GPU behavior.

Current evidence:

```text
nvidia-smi sees NVIDIA GeForce RTX 4080 SUPER
TensorFlow 2.19.1 sees CPU and GPU under escalation
small GPU-visible smoke benchmark rows status = ok
tiny XLA-visible linear value rows status = ok
```

Policy:
- GPU probes and GPU benchmarks require escalated sandbox permissions;
- non-escalated GPU failures are sandbox evidence only;
- CPU/GPU shape matching is required before performance comparison;
- small-shape GPU-visible and XLA-visible success is not a broad GPU speedup,
  QR derivative XLA, or HMC readiness claim.

## Tier 6: HMC Readiness

Purpose:
- certify target-specific sampler readiness.

First target:

```text
linear_qr_score_hessian_static_lgssm
```

Policy:
- target-specific only;
- requires value, score, Hessian, compiled parity, nonfinite, and sampler
  diagnostics;
- SVD-CUT and DSGE HMC claims remain blocked until their separate gates pass.
- tests in this tier must be opt-in and must not claim convergence unless a
  separate convergence diagnostic is explicitly run and documented.

Current opt-in command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
```

Current result:

```text
tests/test_hmc_linear_qr_readiness_tf.py: 3 passed, 2 warnings in 39.00s
```

Claim scope:
- target-specific finite-diagnostic and tiny HMC smoke only;
- no convergence, production sampler, GPU/XLA, MacroFinance, DSGE, or SVD-CUT
  HMC readiness claim.
