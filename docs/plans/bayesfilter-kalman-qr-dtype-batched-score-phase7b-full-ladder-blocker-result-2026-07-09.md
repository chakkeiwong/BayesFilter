# Phase 7B Blocker Result: Full CPU/GPU Ladder

Date: 2026-07-09

## Decision

`BLOCKED_GPU_TENSORFLOW_VISIBILITY`

Phase 7B full CPU/GPU ladder was not launched.  The subplan review found
fixable plan issues, those were repaired, and local preflight checks passed for
script compilation and diff hygiene.  The GPU provenance gate failed because
TensorFlow 2.20.0 reports no physical or logical GPU in the active runtime,
even though `nvidia-smi` sees two NVIDIA GPUs.  The benchmark harness correctly
refuses `--device gpu` when TensorFlow has no logical GPU.

This is an environment/runtime blocker for GPU benchmark artifacts, not
evidence against the batch-native analytical score implementation.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the full CPU/GPU descriptive ladder be launched with trusted GPU provenance? |
| Baseline/comparator | Phase 7 CPU-hidden smoke plus Phase 7B TensorFlow GPU provenance gate. |
| Primary criterion | TensorFlow reports at least one logical GPU and benchmark JSON can record `/GPU:0`, physical/logical GPU lists, JIT, dtype, TF32 flag, and managed-session trust basis before GPU ladders run. |
| Veto diagnostics | TensorFlow physical/logical GPU list empty, `--device gpu` benchmark raises before artifact row, or GPU provenance cannot be recorded. |
| Explanatory diagnostics | `nvidia-smi` device visibility, TensorFlow GPU lists, CUDA initialization error, and failed GPU smoke log. |
| Not concluded | No GPU runtime result, CPU/GPU speed comparison, production/default readiness, HMC readiness, posterior correctness, or statistical ranking. |
| Artifacts | This blocker result, Phase 7B subplan, CPU smoke artifacts, GPU smoke failure log. |

## Review And Repair

Fresh bounded Codex substitute review of the Phase 7B subplan returned
`VERDICT: REVISE` with four findings:

- required log artifacts were listed but commands did not redirect output to
  those paths;
- the GPU provenance gate was referenced but not operationally defined;
- comparator naming should exactly identify `autodiff_row_loop_qr_score`;
- Phase 8 handoff needed to distinguish normal pass closeout from blocker
  closeout.

Repairs applied:

- added stdout/stderr redirection to CPU/GPU ladder commands;
- added an exact TensorFlow GPU provenance command and pass condition;
- tightened comparator naming in the evidence contract;
- clarified pass closeout versus blocker closeout.

Fresh bounded Codex substitute review of the harness returned `VERDICT:
REVISE` because one JSON key still said
`descriptive_batched_autodiff_over_batch_native_analytical_median_ratio`.
That stale key was removed and the CPU-hidden FP32 smoke artifact was
regenerated.

Claude review was not used because the current runbook records a prior
approval-reviewer rejection for external-disclosure risk.  Codex substitute
review is weaker than Claude review.

## Checks Run

| Check | Result |
| --- | --- |
| `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` | passed |
| `git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests` | passed |
| `nvidia-smi` | passed; two NVIDIA GeForce RTX 4080-class GPUs visible to the driver |
| TensorFlow GPU provenance probe | failed GPU gate; `physical_gpu=[]`, `logical_gpu=[]`, `cuda_visible_devices=UNSET` |
| GPU smoke benchmark | failed before benchmark rows with `RuntimeError: requested GPU benchmark but no logical GPU is visible` |
| Refreshed CPU-hidden FP32 smoke after stale-key repair | passed |

## Artifact Paths

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7b-full-ladder-subplan-2026-07-09.md`
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.json`
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.md`
- `docs/benchmarks/logs/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.log`
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_gpu_xla_2026-07-09.json` was not written because the GPU command failed before payload creation.
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_gpu_xla_2026-07-09.md` was not written because the GPU command failed before payload creation.
- `docs/benchmarks/logs/kalman_qr_batched_score_smoke_float32_gpu_xla_2026-07-09.log`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| `BLOCKED_GPU_TENSORFLOW_VISIBILITY` | failed for GPU provenance; CPU-hidden smoke remains passed | GPU logical device list empty in TensorFlow; harness correctly refused GPU benchmark | Whether a different trusted TensorFlow/CUDA environment exposes GPUs | Repair TensorFlow GPU visibility or run in a known GPU-enabled environment, then rerun Phase 7B provenance gate and full ladders | No GPU benchmark, CPU/GPU comparison, speed ranking, production/default readiness, HMC readiness, or posterior correctness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | GPU provenance veto fired. |
| Statistically supported ranking | Not assessed. |
| Descriptive-only differences | CPU-hidden smoke timing only. |
| Default-readiness | Not assessed. |
| Next evidence needed | TensorFlow-visible GPU provenance followed by full CPU/GPU ladder artifacts. |

## Handoff

Proceed to Phase 8 blocker closeout unless the runtime is changed so that
TensorFlow sees at least one logical GPU.  If GPU visibility is repaired, resume
from the Phase 7B subplan, rerun the preflight provenance gate, and then launch
the full ladder commands with log redirection.
