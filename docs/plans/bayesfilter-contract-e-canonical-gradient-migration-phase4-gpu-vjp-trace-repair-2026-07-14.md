# Phase 4 GPU VJP Trace Repair

Date: 2026-07-14

Status: `REPAIRED_BEFORE_NUMERICAL_OUTPUT`

The first VJP-mode launch initialized the trusted GPU but failed during
TensorFlow tracing, before XLA compilation or analytic VJP execution:

```text
KeyError: 'finite'
```

The preflight harness incorrectly treated
`_contract_e_streaming_vjp_core(...)["reset"]` as the cloud forward diagnostic
dictionary. In the VJP API that field is instead the reset cotangent
decomposition and has no `finite` or `factor_diagonal_positive` keys. GPU
allocator peak at failure was under one MiB and the log contains no XLA compile
line, so this is a pre-execution harness schema defect, not derivative,
feasibility, numerical, or scientific evidence.

The repair defines the VJP hard predicate from
`quotient["valid_chart"]` plus the already-declared finiteness check over every
returned analytic cotangent. The immutable successful forward artifact already
establishes the unchanged fixture's finite positive Contract E Cholesky chart.
No fixture, shape, dtype, transport setting, chunk, ridge, upstream cotangent,
threshold, claim, or production implementation is changed. After syntax,
CPU-hidden tracing at a tiny shape is not available without changing the frozen
static constants, so the unchanged trusted-GPU VJP launch is retried only after
source inspection and Python compilation pass.
