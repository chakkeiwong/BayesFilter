# Kalman QR Dtype And Batched Score Visible Stop Handoff

Date: 2026-07-09

## Status

`STOP_PHASE_7B_BLOCKED_GPU_TENSORFLOW_VISIBILITY`

## Current Phase

Phase 7B: Full CPU/GPU ladder.

## Active Boundaries

- Codex is supervisor and executor.
- Claude is read-only reviewer only.
- No detached or nested execution is authorized by this runbook.
- GPU runtime is deferred until a later reviewed phase and requires trusted
  provenance.
- No default-policy, HMC-readiness, posterior-correctness, or speed-superiority
  claim is authorized.
- Phase 2 CPU-hidden value checks passed but do not support benchmark,
  GPU/default-readiness, or speed claims.
- Phase 3 CPU-hidden analytical-score dtype checks passed but do not support
  benchmark, GPU/default-readiness, batch-native-score, or speed claims.
- Phase 4 CPU-hidden benchmark dtype smoke passed for `float32` and `float64`,
  with requested and observed analytical/autodiff value/score dtypes recorded.
- Phase 5 contract passed with bounded Codex substitute review.
- Phase 6 implemented `tf_qr_sqrt_kalman_score_batched_static` and passed
  CPU-hidden correctness, dtype, source-contract, and XLA smoke checks.
- Phase 7 refreshed benchmark commands and artifacts to include the
  batch-native analytical score arm, scalar analytical row-loop comparator, and
  autodiff row-loop comparator.
- The CPU-hidden FP32 XLA harness smoke passed for one `(10, 10)`, 50-parameter,
  batch-size-2, 8-timestep row.  It records the batch-static autodiff
  value-gradient route as diagnostic-only because it returned nonfinite scores
  for this synthetic lower-triangular benchmark fixture.
- Full CPU/GPU ladders remain deferred until exact-grid commands are refreshed
  and GPU runtime approval/provenance is confirmed.
- Phase 7B exact-grid commands were drafted and reviewed with bounded Codex
  substitute review.
- `nvidia-smi` sees two NVIDIA GPUs, but TensorFlow 2.20.0 reports
  `physical_gpu=[]` and `logical_gpu=[]` in the current runtime.
- A tiny `--device gpu` benchmark smoke failed before benchmark row creation
  with `RuntimeError: requested GPU benchmark but no logical GPU is visible`.
- Full CPU/GPU ladders are blocked until TensorFlow GPU visibility is repaired
  in a trusted environment.
- Phase 7 timings are descriptive only unless replicated with uncertainty; no
  statistical speed ranking or production-readiness claim is authorized.

## If Stopped

Resume by reading:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-gated-execution-runbook-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-execution-ledger-2026-07-09.md`
- the current phase subplan/result.
