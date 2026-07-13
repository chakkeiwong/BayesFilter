# Phase 8 Closeout Result: Dtype And Batched Analytical Score Program

Date: 2026-07-09

## Decision

`BLOCKER_CLOSEOUT_GPU_TENSORFLOW_VISIBILITY`

The dtype cleanup and batch-native analytical score implementation reached the
CPU-hidden correctness and smoke gates.  The refreshed benchmark harness can
time `batch_native_analytical_qr_score`, `scalar_analytical_row_loop`, and
`autodiff_row_loop_qr_score` with JIT, dtype, device, and batch provenance.

The full CPU/GPU benchmark ladder is blocked in the current runtime because
TensorFlow reports no physical or logical GPU, even though `nvidia-smi` sees
two NVIDIA GPUs.  No GPU timing table was produced.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| `BLOCKER_CLOSEOUT_GPU_TENSORFLOW_VISIBILITY` | Dtype, batch score, and CPU-hidden smoke gates passed; full CPU/GPU ladder did not run | GPU provenance veto fired because TensorFlow logical GPU list is empty | Whether a repaired TensorFlow/CUDA runtime exposes GPUs | Repair TensorFlow GPU visibility, rerun Phase 7B provenance gate, then launch full ladder commands | No GPU benchmark, CPU/GPU comparison, speed ranking, production/default readiness, HMC readiness, posterior correctness, or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for CPU-hidden dtype/batch score/smoke gates; GPU full-ladder veto fired. |
| Statistically supported ranking | Not assessed. |
| Descriptive-only differences | CPU-hidden smoke timing only; one row and one repeat. |
| Default-readiness | Not assessed. |
| Next evidence needed | TensorFlow-visible GPU provenance plus full CPU/GPU ladder artifacts; replicated uncertainty before any ranking claim. |

## Reached Artifacts

- Master program: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md`
- Runbook: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-gated-execution-runbook-2026-07-09.md`
- Ledger: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-execution-ledger-2026-07-09.md`
- Stop handoff: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-stop-handoff-2026-07-09.md`
- Phase 7 result: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-result-2026-07-09.md`
- Phase 7B subplan: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7b-full-ladder-subplan-2026-07-09.md`
- Phase 7B blocker: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7b-full-ladder-blocker-result-2026-07-09.md`
- CPU-hidden smoke JSON: `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.json`
- CPU-hidden smoke Markdown: `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.md`
- GPU smoke failure log: `docs/benchmarks/logs/kalman_qr_batched_score_smoke_float32_gpu_xla_2026-07-09.log`

## Implementation Summary

- Added dtype-polymorphic TensorFlow QR value/score infrastructure across the
  reached phases.
- Implemented `tf_qr_sqrt_kalman_score_batched_static`.
- Added batched analytical score tests and dtype contract tests.
- Updated the parameter-count benchmark harness with batch-size support,
  JIT-default compiled timing arms, observed dtype checks, device provenance,
  and nonclaims.
- Kept the batch-static autodiff value-gradient route diagnostic-only after it
  returned nonfinite scores for the synthetic benchmark fixture.

## Checks

| Check | Result |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py` | `19 passed` |
| `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` | passed |
| `git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests` | passed |
| CPU-hidden FP32 XLA smoke with `--batch-size 2` | passed |
| TensorFlow GPU provenance probe | blocked; `physical_gpu=[]`, `logical_gpu=[]` |

## Review Status

Claude review was not used in this run because the runbook records an earlier
approval-reviewer rejection for external-disclosure risk.  Fresh bounded Codex
substitute reviews were used for the Phase 7B subplan and harness repair.  This
review path is weaker than Claude review and does not authorize GPU/runtime,
scientific, product, or default-policy claims.

## Post-Run Red Team

The strongest alternative explanation for the GPU blocker is an environment
configuration mismatch between the NVIDIA driver visible to `nvidia-smi` and
the TensorFlow runtime's CUDA initialization path.  A result that would
overturn the blocker is a trusted TensorFlow probe in this repo environment
showing at least one logical GPU, followed by a passing `--device gpu` smoke
artifact with managed-session trust basis recorded.

The weakest part of the timing evidence is that only a CPU-hidden smoke row was
run after the harness repair.  It validates wiring and parity, not a runtime
ranking.
