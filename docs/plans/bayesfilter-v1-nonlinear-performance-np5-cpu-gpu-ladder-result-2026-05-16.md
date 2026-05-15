# BayesFilter V1 Nonlinear Performance NP5 CPU/GPU Ladder Result

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP4 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`
- NP5 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-plan-2026-05-15.md`
- Benchmark JSON: `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json`
- Benchmark Markdown: `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md`

## Phase purpose

Execute the accepted narrow NP5 scope only: trusted matched CPU/GPU timing for NP4-supported static Model B value cells with `timesteps=3`, `dtype=tf.float64`, `return_filtered=False`, backends `tf_svd_cubature`, `tf_svd_ukf`, and `tf_svd_cut4`, comparing graph and XLA on CPU and GPU without expanding to score paths, Model C, broader ladders, or any HMC claim.

## Skeptical plan audit

Audit target: whether the accepted narrow NP5 command and artifact answer the matched CPU/GPU timing question without silently broadening beyond NP4-certified support cells.

Findings:

- Wrong-cell risk: NP4 certified only static-shape Model B and Model C value cells for graph-vs-XLA parity boundaries, with GPU support remaining unclaimed there. The accepted NP5 scope therefore had to stay on Model B value rows only and explicitly inherit the NP4 support-cell boundary rather than treating this benchmark as a general GPU readiness proof.
- Mismatched-comparator risk: CPU/GPU rows would be uninterpretable if shape, dtype, model, return contract, or backend differed across rows. The executed command fixed `model=model_b_nonlinear_accumulation`, `timesteps=3`, `dtype=float64`, `return_filtered=False`, and the backend set before comparing device/mode rows.
- Trust-label risk: any GPU-visible timing row collected without trusted execution would be sandbox evidence only. This run used `--device-scope visible` under `escalated_sandbox`, and the required trusted pre-probes from Codex are recorded below as provenance.
- Overclaim risk: one tiny static row per backend can show only exact tested-cell timing. It cannot justify broad GPU speedup, default backend changes, score-path conclusions, HMC readiness, or dynamic-horizon claims.
- Branch-gate risk: timing rows are invalid if branch metadata differs across compared rows. The benchmark emits value and score branch gate counts for every row, and all emitted rows showed matching `3/3` branch counts for this exact shape.
- Warmup conflation risk: compile/warmup time can dominate tiny rows. The executed harness kept warmup separate via `warmup_calls=1`, then reported first-call and steady-state timings.

Audit outcome: pass for the accepted narrow NP5 scope. The command and artifacts answer only the matched static Model B value timing question and preserve the required non-claim boundaries.

## Evidence contract

Question:

- For NP4-supported static Model B value cells at `timesteps=3`, which of CPU graph, CPU XLA, GPU graph, or GPU XLA is fastest for each tested backend after compile/warmup separation?

Baseline/comparators:

- Comparator family `np5-model-b-t3-return-filtered-false-f64` with exact rows:
  - `model_b_t3_tf64_rf0_tf_svd_cubature_cpu_graph`
  - `model_b_t3_tf64_rf0_tf_svd_cubature_cpu_xla`
  - `model_b_t3_tf64_rf0_tf_svd_cubature_gpu_graph`
  - `model_b_t3_tf64_rf0_tf_svd_cubature_gpu_xla`
  - `model_b_t3_tf64_rf0_tf_svd_ukf_cpu_graph`
  - `model_b_t3_tf64_rf0_tf_svd_ukf_cpu_xla`
  - `model_b_t3_tf64_rf0_tf_svd_ukf_gpu_graph`
  - `model_b_t3_tf64_rf0_tf_svd_ukf_gpu_xla`
  - `model_b_t3_tf64_rf0_tf_svd_cut4_cpu_graph`
  - `model_b_t3_tf64_rf0_tf_svd_cut4_cpu_xla`
  - `model_b_t3_tf64_rf0_tf_svd_cut4_gpu_graph`
  - `model_b_t3_tf64_rf0_tf_svd_cut4_gpu_xla`
- NP4 support boundary for admissibility: Model B value rows with static observations shape `(3, 1)`, `return_filtered=False`, and backends `tf_svd_cubature`, `tf_svd_ukf`, `tf_svd_cut4`.

Primary criterion:

- Each timing statement is admissible only when compared rows have identical model equations, branch metadata, static shape, timestep count, dtype, backend, and return contract, differing only by requested device and execution mode.

Veto diagnostics:

- trusted GPU pre-probe provenance missing;
- benchmark not run with trusted/escalated GPU-visible execution;
- any compared row has non-`ok` status;
- branch metadata differs across compared rows;
- CPU/GPU rows differ in model, shape, dtype, backend, or return contract;
- exact tested-cell timings are promoted to broad speedup or policy claims.

Explanatory diagnostics only:

- warmup time;
- first-call time;
- RSS deltas;
- observed logical/physical device lists.

What is not concluded:

- broad GPU speedup across shapes or models;
- GPU or CPU default backend policy;
- score-path speed or XLA readiness;
- dynamic-horizon support;
- Model C performance;
- HMC performance, convergence, or readiness.

Artifact:

- `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md`
- `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json`
- `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md`

## Trusted pre-probe provenance from Codex supervisor

These pre-probes were required and were already run by Codex before this bounded worker execution:

1. `nvidia-smi` at `2026-05-16 02:39:29` showed an NVIDIA GeForce RTX 4080-class GPU with driver `591.86` and CUDA `13.1`.
2. Trusted TensorFlow device probe printed TensorFlow `2.19.1` and physical devices `CPU:0` and `GPU:0`.

This NP5 worker treated those pre-probes as authoritative provenance and did not rerun separate probe commands.

## Exact benchmark command

```text
python docs/benchmarks/benchmark_bayesfilter_v1_model_bc_gpu_xla.py --device-scope visible --models model_b_nonlinear_accumulation --timesteps 3 --backends tf_svd_cubature,tf_svd_ukf,tf_svd_cut4 --modes graph,xla --devices cpu,gpu --repeats 2 --warmup-calls 1 --output /home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json --markdown-output /home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md
```

Trust label for the benchmark command: `escalated_sandbox`.

## Comparator admissibility checks

Identical across every compared row:

- model: `model_b_nonlinear_accumulation`
- path: value only
- NP4 support-cell boundary: static Model B value cell, `return_filtered=False`
- observations shape / horizon: `(3, 1)` / `timesteps=3`
- dtype: `tf.float64`
- branch metadata: `value_branch_ok_count=3`, `value_branch_total_count=3`, `score_branch_ok_count=3`, `score_branch_total_count=3`
- benchmark repeats / warmup policy: `repeats=2`, `warmup_calls=1`
- branch status: all rows `status=ok`
- git branch: `main`

Backend-specific constants preserved within each comparator family:

- `tf_svd_cubature`: `point_count=6`
- `tf_svd_ukf`: `point_count=7`
- `tf_svd_cut4`: `point_count=14`

Requested-vs-actual device observations:

- CPU rows ran on `/device:CPU:0`
- GPU rows ran on `/device:GPU:0`
- Artifact environment recorded both logical and physical CPU/GPU devices visible during the trusted benchmark run.

## Runtime rows

| Comparator id | Backend | Device | Mode | Warmup s | First call s | Steady mean s | Status | Interpretation label |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `model_b_t3_tf64_rf0_tf_svd_cubature_cpu_graph` | `tf_svd_cubature` | CPU | graph | 1.182206 | 0.002341 | 0.001646 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cubature_cpu_xla` | `tf_svd_cubature` | CPU | xla | 1.066668 | 0.001150 | 0.000819 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cubature_gpu_graph` | `tf_svd_cubature` | GPU | graph | 0.241731 | 0.005086 | 0.005359 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cubature_gpu_xla` | `tf_svd_cubature` | GPU | xla | 0.773783 | 0.001069 | 0.000811 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_ukf_cpu_graph` | `tf_svd_ukf` | CPU | graph | 0.227435 | 0.001787 | 0.001629 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_ukf_cpu_xla` | `tf_svd_ukf` | CPU | xla | 0.989236 | 0.001374 | 0.001394 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_ukf_gpu_graph` | `tf_svd_ukf` | GPU | graph | 0.235348 | 0.007553 | 0.004964 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_ukf_gpu_xla` | `tf_svd_ukf` | GPU | xla | 0.792625 | 0.001695 | 0.001396 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cut4_cpu_graph` | `tf_svd_cut4` | CPU | graph | 0.361893 | 0.002479 | 0.002160 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cut4_cpu_xla` | `tf_svd_cut4` | CPU | xla | 0.967769 | 0.002431 | 0.002218 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cut4_gpu_graph` | `tf_svd_cut4` | GPU | graph | 0.257009 | 0.007676 | 0.007809 | `ok` | continuation_candidate |
| `model_b_t3_tf64_rf0_tf_svd_cut4_gpu_xla` | `tf_svd_cut4` | GPU | xla | 0.830728 | 0.002454 | 0.002190 | `ok` | continuation_candidate |

Note: the CUT4 GPU XLA warmup time stored in the JSON artifact is `0.8307277409985545`. Timing interpretation here uses the authoritative JSON rows; the acceptance boundary is row status plus matched metadata, not any single warmup number.

The JSON artifact was post-audit hygiene-patched by the Codex supervisor to narrow generated `Model B/C` claim wording to the exact Model B T=3 run and to add row-level audit fields required by the master program: comparator id, path, dtype, static shape, seed policy, tolerance, finite/shape status, trust label, CPU/GPU policy, promotion/continuation/repair labels, command, environment, artifact path, and non-implication text. No timing values were changed.

## Per-backend timing interpretation

- `tf_svd_cubature`: GPU XLA (`0.000811 s`) and CPU XLA (`0.000819 s`) are effectively tied at this repeat depth, both faster than CPU graph (`0.001646 s`) and much faster than GPU graph (`0.005359 s`). This supports only an exact tested-cell statement for the static Model B row.
- `tf_svd_ukf`: CPU XLA (`0.001394 s`) and GPU XLA (`0.001396 s`) are again effectively tied and both faster than CPU graph (`0.001629 s`) and GPU graph (`0.004964 s`). No broader policy conclusion follows from this tiny row.
- `tf_svd_cut4`: CPU graph (`0.002160 s`) is slightly faster than GPU XLA (`0.002190 s`) and CPU XLA (`0.002218 s`), while GPU graph (`0.007809 s`) is slowest. This is exactly the kind of backend-specific exception that blocks any broad “GPU wins” narrative.

## CPU/GPU policy statement

This NP5 result follows the project GPU/CUDA policy by using trusted GPU-visible execution only for the actual benchmark command and by recording Codex-provided trusted pre-probe provenance. No nontrusted GPU row was used for timing interpretation.

## NP4 support-cell boundary carried into NP5

This phase used only the NP4-supported static Model B value cells for `tf_svd_cubature`, `tf_svd_ukf`, and `tf_svd_cut4` at `return_filtered=False`. It did not open score cells, dynamic horizon, `return_filtered=True`, Model C timing claims, or any GPU-XLA certification beyond the fact that the trusted benchmark executed on `/device:GPU:0` for these exact rows.

## Run manifest

| Field | Value |
| --- | --- |
| Git commit | `81c647b157612a233dde1a9fbd847647a5b03b8f` |
| Dirty worktree | `true` |
| Command | `python docs/benchmarks/benchmark_bayesfilter_v1_model_bc_gpu_xla.py --device-scope visible --models model_b_nonlinear_accumulation --timesteps 3 --backends tf_svd_cubature,tf_svd_ukf,tf_svd_cut4 --modes graph,xla --devices cpu,gpu --repeats 2 --warmup-calls 1 --output /home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json --markdown-output /home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md` |
| Trust label | `escalated_sandbox` |
| Environment | `local CLI session` |
| Python | `3.11.14` |
| TensorFlow | `2.19.1` |
| TensorFlow Probability | `N/A; not separately reported by this harness` |
| CPU/GPU status | `trusted visible CPU/GPU benchmark run` |
| GPU intentionally hidden | `no` |
| Device visibility | `logical: /device:CPU:0 and /device:GPU:0; physical: CPU:0 and GPU:0` |
| Dtype | `tf.float64` |
| Model/backend/shape/horizon/parameter dimension/point count | `Model B nonlinear accumulation; backends tf_svd_cubature/tf_svd_ukf/tf_svd_cut4; observations shape (3,1); horizon 3; parameter dimension N/A for value timing row interpretation; point counts 6/7/14 by backend` |
| Random seeds | `N/A` |
| Warmup policy and wall time | `warmup_calls=1; repeats=2; compile/warmup separated in artifact; total command wall time not separately recorded outside row timings` |
| Output artifacts | `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json`; `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md` |
| Governing phase plan | `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-plan-2026-05-15.md` |
| Result file | `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md` |
| Derivation/proof-obligation artifact | `N/A; no algebraic production change was made` |

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Record exact trusted CPU/GPU timing rows for the three NP4-supported static Model B value backends and carry them forward as backend-specific evidence only | passed: every compared row was `status=ok` with matched model/backend/shape/dtype/branch metadata and trusted GPU-visible execution | passed: trusted pre-probe provenance recorded, benchmark run under `escalated_sandbox`, no row mismatch detected, no broad claim promoted | repeat-depth is tiny and only one static shape was run, so small differences near tie level may reflect measurement noise | continue to NP6/NP7 using these rows as exact tested-cell evidence only; if broader CPU/GPU claims are needed, run a separately planned wider ladder | no broad GPU speedup claim, no default-policy claim, no score-path claim, no Model C claim, no HMC claim |

## Promotion / continuation / repair labels

- Promotion label: `not_promoted_beyond_exact_tested_cells`
- Continuation label: `continue_with_shape_specific_backend_evidence_only`
- Repair label: `no_repair_needed_within_accepted_narrow_scope`

## Non-implications

This NP5 result does not show a general GPU advantage for BayesFilter nonlinear filters. It does not justify enabling GPU by default, does not certify score or HMC behavior, does not establish dynamic-horizon performance, and does not turn trusted execution of these rows into a broad XLA or production-readiness claim. The CUT4 row, where CPU graph slightly beats both XLA rows, is direct evidence against collapsing these exact-cell results into a single device-wide story.

## Files changed

- `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md`
- `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json`
- `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md`

## Commands run

```text
git rev-parse HEAD
git status --short docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md
python docs/benchmarks/benchmark_bayesfilter_v1_model_bc_gpu_xla.py --device-scope visible --models model_b_nonlinear_accumulation --timesteps 3 --backends tf_svd_cubature,tf_svd_ukf,tf_svd_cut4 --modes graph,xla --devices cpu,gpu --repeats 2 --warmup-calls 1 --output /home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json --markdown-output /home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md
```

## Phase exit label

`NP5_NARROW_MODEL_B_T3_CPU_GPU_ROWS_RECORDED_NO_BROAD_PROMOTION`
