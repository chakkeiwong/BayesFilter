# Phase 1 Focused Check Log

Date: 2026-07-13

Execution target: deliberate CPU-only float64 reference/debug check with
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import.

## Commands And Results

```text
python -m json.tool docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-numerical-statistical-design-freeze-2026-07-13.json
exit: 0

CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_contract_e_phase1_normative_math.py tests/test_contract_e_cholesky_ridge_reset.py
14 passed in 3.91s
exit: 0

python -m py_compile docs/benchmarks/contract_e_reset_tf.py tests/test_contract_e_phase1_normative_math.py
exit: 0

git diff --check
exit: 0
```

The focused tests checked fixed-ridge manual VJP versus TensorFlow autodiff for
source cloud, normalized weights, dense reference matrix, residual design, and
ridge; per-input directional central-difference ladders; the probability-to-
logit pullback; direct source moment and weight paths; the ridged covariance
identity and raw residual formula; and row-quotient JVP/VJP duality.

TensorFlow 2.19.1 reported only a CPU physical device after GPU hiding. CUDA
plugin-registration and `cuInit` messages appeared during import despite
`CUDA_VISIBLE_DEVICES=-1`; they are recorded as environment noise and do not
support any GPU-health conclusion.

## Interpretation

The deterministic tiny fixture supports the checked finite-program algebra and
the diagnostic helper's VJP on that chart. It does not prove production
implementation correctness, floating-point adequacy on GPU/XLA/TF32, streaming
composition, LGSSM agreement, nonlinear validity, or HMC readiness.
