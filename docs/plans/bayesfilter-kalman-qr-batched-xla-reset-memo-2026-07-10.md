# Reset memo: Kalman QR batched score benchmark and XLA compilation blockers

## Date
2026-07-10

## Context
The current Kalman QR benchmark lane was started to compare analytical QR score
computation against autodiff for linear Gaussian Kalman filtering under XLA JIT.
The requested grid was:

- `T = 120`
- state/measurement dimensions `(10, 10)`, `(20, 20)`, `(30, 30)`
- parameter counts `50` and `150`
- parameter batch sizes `1`, `4`, `16`
- CPU thread/core settings `1`, `4`, `16`
- GPU `float32` versus `float64`
- analytical versus autodiff score paths

The run did not produce a valid full CPU/GPU timing table.  The benchmark
harness itself exposed XLA compilation and graph-size problems that must be
treated as the primary current result.

## Decision / policy
- Do not use the overnight batch-size-16 artifacts to rank analytical versus
  autodiff runtime.  Those rows are dominated by XLA compile/codegen failures.
- Do not rerun the full 3 x 2 x 3 CPU/GPU grid with the current harness as-is.
  It will likely spend hours recompiling or fail in LLVM code generation again.
- Treat current timings as descriptive/debug artifacts only.  They do not
  support statistical ranking, production readiness, HMC readiness, posterior
  correctness, or a default-policy change.
- Next work should first repair the benchmark graph structure and compile-size
  instrumentation, then run a small correctness and compile-size smoke before
  any overnight grid.

## What changed
- File: `scripts/benchmark_kalman_qr_parameter_count_scaling.py`
  - Added dtype/device/JIT controls, CPU thread recording, batch-size support,
    row subprocess isolation, and benchmark result provenance.
  - Added methods:
    - `batch_native_analytical_qr_score`
    - `scalar_analytical_row_loop`
    - `autodiff_row_loop_qr_score`
    - diagnostic-only `batched_static_autodiff_probe`
  - Important limitation: the benchmark still constructs batched model tensors
    with a Python loop over batch rows, and both scalar/autodiff comparators use
    Python row loops inside `tf.function`.
- File: `bayesfilter/linear/kalman_qr_derivatives_tf.py`
  - Added or modified batched-static analytical QR score support.
  - Important limitation: the batched-static score kernel uses `tf.while_loop`
    over time, but helper derivative functions still use Python loops over the
    static parameter dimension, which enlarges the compiled graph as `P` grows.
- Files: `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-*`
  - Created phase plans/results for dtype controls and batched score work.
- File: `docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py`
  - Created overnight runner for the requested CPU/GPU grid.
- File: `docs/benchmarks/kalman_qr_core_batch_grid_overnight_status_2026-07-09.json`
  - Records final overnight status: `complete_with_failures`.

## Current artifact map
- Main status:
  - `docs/benchmarks/kalman_qr_core_batch_grid_overnight_status_2026-07-09.json`
- CPU batch grid artifacts:
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads1_batch1_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads1_batch4_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads1_batch16_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads4_batch1_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads4_batch4_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads4_batch16_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads16_batch1_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads16_batch4_xla_2026-07-09.json`
  - `docs/benchmarks/kalman_qr_core_batch_grid_cpu_threads16_batch16_xla_2026-07-09.json`
- GPU preflight artifact:
  - `docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_overnight_2026-07-09.json`
- Main result note:
  - `docs/plans/bayesfilter-kalman-qr-core-batch-dtype-grid-result-2026-07-09.md`
- XLA dump diagnostic:
  - Raw dump path used before reboot: `/tmp/kalman_xla_dump_single_b1p50`
  - This path is in `/tmp`; it may not survive reboot.  The key measurements
    are recorded below.

## XLA compilation evidence
Single smallest CPU XLA diagnostic, batch-native analytical only:

- Case: `dim = 10`, `parameter_count = 50`, `T = 120`, `batch_size = 1`,
  `dtype = float32`, CPU with GPUs hidden.
- Traced TensorFlow graph size before XLA:
  - `batch_native_analytical`: 5,798 GraphDef nodes, 1.36 MiB GraphDef.
  - `scalar_analytical_row_loop`: 5,304 GraphDef nodes, 1.89 MiB GraphDef.
  - `autodiff_row_loop`: 368 GraphDef nodes, 0.24 MiB GraphDef.
- XLA dump for `batch_native_analytical`, same case:
  - total dump directory: about 103 MiB
  - file count: 1,571
  - largest files:
    - `cpu_after_optimizations.txt`: 8,703,009 bytes
    - `cpu_after_optimizations-buffer-assignment.txt`: 7,372,889 bytes
    - `before_optimizations.txt`: 5,837,098 bytes

Batch-size growth trace diagnostic:

- Case: `dim = 10`, `parameter_count = 50`, `T = 120`, `batch_size = 4`
  - `batch_native_analytical`: 5,912 nodes, 1.37 MiB GraphDef.
  - `scalar_analytical_row_loop`: 21,150 nodes, 7.75 MiB GraphDef.
- Case: `dim = 10`, `parameter_count = 50`, `T = 120`, `batch_size = 16`
  - `batch_native_analytical`: 6,368 nodes, 1.41 MiB GraphDef.
  - `scalar_analytical_row_loop`: 84,534 nodes, 31.43 MiB GraphDef.
- Case: `dim = 10`, `parameter_count = 150`, `T = 120`, `batch_size = 16`
  - `batch_native_analytical`: 16,568 nodes, 3.95 MiB GraphDef.
  - The scalar row-loop graph was not allowed to finish tracing because it was
    already clearly in the compile-blowup regime.

Interpretation:

- The batch-native analytical wrapper is relatively stable in GraphDef size
  across batch size for `P=50`, though it still grows with parameter count.
- The scalar analytical row-loop comparator scales roughly linearly with batch
  size at trace time and becomes a compile-size hazard.
- The current benchmark path is not a clean "no Python loop" batched benchmark.
  It contains Python loops that are statically unrolled into the graph.

## Python loop / NumPy audit
Python loops in the current timed or compiled benchmark path:

- `scripts/benchmark_kalman_qr_parameter_count_scaling.py::_batched_model_tensors`
  uses:
  - `per_row = [_model_tensors(fixture, params_batch[index]) for index in range(batch_size)]`
- `build_scalar_analytic_row_loop_fn` loops over `range(batch_size)`.
- `build_autodiff_row_loop_fn` loops over `range(batch_size)`.
- `bayesfilter/linear/kalman_qr_derivatives_tf.py` batched helper functions
  loop over static `parameter_dim` in:
  - `_batched_stack_qr_lower_factor_first_derivatives`
  - `_batched_cholesky_factor_first_derivatives`
  - `_batched_factor_covariance_first_derivatives`

TensorFlow loops:

- `tf_qr_sqrt_kalman_score_batched_static` uses `tf.while_loop` over time.
- This is good for avoiding a Python loop over `T`, but it does not eliminate
  parameter-axis Python loops in helper derivative routines.

NumPy status:

- No NumPy computation was found in the timed TensorFlow graph.
- `.numpy()` is used only after the timed call returns, inside `_materialize`,
  to pull TensorFlow outputs into JSON/reporting structures.

## Bugs / blockers resolved
- The original question "was CPU JIT compiled?" is resolved:
  - yes, the CPU benchmark commands used `--jit-compile`;
  - CPU artifacts intentionally set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
    import for CPU-only runs.
- The original question "is GPU 32 bit or 64 bit?" is partially resolved:
  - the attempted GPU preflight was `float32`;
  - the full requested GPU `float32` versus `float64` grid did not run because
    GPU XLA preflight failed.

## Open blockers
### CPU batch-size-16 XLA compile/codegen failure
Final systemd unit:

- Unit: `kalman-qr-core-batch-grid-20260709.service`
- Command:
  - `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py`
- Final state:
  - failed with exit status 1 at 2026-07-10 13:41:42 CST
  - consumed 23h 59min 6.362s CPU time

CPU artifact status from the overnight status JSON:

- `threads=1, batch=1`: passed/skipped existing complete
- `threads=1, batch=4`: passed
- `threads=1, batch=16`: failed
- `threads=4, batch=1`: passed
- `threads=4, batch=4`: passed
- `threads=4, batch=16`: failed
- `threads=16, batch=1`: passed
- `threads=16, batch=4`: passed
- `threads=16, batch=16`: failed

Representative CPU failure symptoms:

- `batch_size=16`, `dim=10`, `parameter_count=150`: row timed out after
  3600 seconds while compiling.
- `batch_size=16`, larger dimensions: child rows often failed with
  `MissingChildRows` after child return codes `-6` or `-11`.
- stderr tails included:
  - `Very slow compile?`
  - `LLVM compilation error: Cannot allocate memory`
  - `allocateMappedMemory failed with error: Cannot allocate memory`
  - `LLVM ERROR: Unable to allocate section memory!`
  - `Resource tracker ... became defunct`

Current conclusion:

- These are XLA CPU compile/codegen resource failures.  They are not evidence
  that the Kalman math is numerically wrong.
- The current full-grid CPU benchmark design is invalid for the user's intended
  speed question because compile failures and static unrolled comparators
  dominate the result.

### GPU XLA preflight failure
GPU preflight status:

- Artifact:
  `docs/benchmarks/kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_overnight_2026-07-09.json`
- Case:
  - `dim=10`, `parameter_count=50`, `T=120`, `batch_size=1`, `dtype=float32`
- GPU visibility:
  - TensorFlow saw two RTX 4080 SUPER logical GPU devices in trusted managed
    session context.
- Failure:
  - `FailedPreconditionError: Can not combine dim orders and requirements.`
  - The failure occurred in `__inference_autodiff_row_loop_score`.
- Status recorded in overnight JSON:
  - `gpu_status = blocked_gpu_xla_autodiff_preflight`

Current conclusion:

- No full GPU timing grid exists.
- The GPU failure is tied to the current XLA/autodiff row-loop preflight path.
  It does not yet establish that the batch-native analytical GPU path is
  unusable.

## Verification already run
```bash
systemctl --user status kalman-qr-core-batch-grid-20260709.service --no-pager
```

Observed:

- Unit is failed, not active.
- Exit status 1, CPU consumed about 24 hours.

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'kalman_qr|run_kalman_qr|benchmark_kalman|XLA_FLAGS|tfgpu/bin/python'
```

Observed:

- No active Kalman QR benchmark worker was present; only the inspection command
  matched.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 PYTHONUNBUFFERED=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -c '<trace GraphDef sizes>'
```

Observed:

- GraphDef-size diagnostics listed in the XLA compilation evidence section.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 XLA_FLAGS=--xla_dump_to=/tmp/kalman_xla_dump_single_b1p50 \
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -c '<compile one batch-native case>'
```

Observed:

- Smallest batch-native analytical CPU XLA compile completed.
- Output shape/dtype was `(1,)`, `(1, 50)`, `float32`, `float32`.
- XLA dump size measurements are recorded above.

## Current policy
- BayesFilter TensorFlow/TFP algorithmic, gradient-bearing, benchmark, and
  production-target paths default to XLA JIT.
- CPU-only runs must hide GPU devices before TensorFlow import and label CPU as
  explicit reference/debug/benchmark evidence, not the production default.
- GPU benchmark or CUDA probing should be trusted/escalated or covered by the
  managed-session GPU trust policy, and artifacts must record provenance.
- Scientific/statistical claims require their own evidence gates; current
  benchmark artifacts are compile/debug evidence only.

## Known limitations / cautions
- The repository worktree is very dirty and includes unrelated HMC/scalar SSM
  files from adjacent work.  Do not reset or revert unrelated changes.
- Many Kalman QR files and benchmark artifacts are untracked; inspect carefully
  before committing.
- `/tmp/kalman_xla_dump_single_b1p50` may be lost after reboot.  The memo
  records the important measurements, but not the raw dump contents.
- The current benchmark compares a batch-native analytical path against scalar
  and autodiff row-loop comparators.  That is not a fair compile-size comparison
  at large batch size.
- The batch-native analytical path still has parameter-axis Python loops in
  helper derivative routines, so increasing `parameter_count` can still enlarge
  the XLA graph.
- The current GPU preflight uses the autodiff row-loop path as part of the
  combined parity benchmark; this can block GPU exploration even if the
  analytical path itself might compile.

## Suggested next steps
1. Write a focused repair subplan before more long runs.
2. Add graph-size and compile-size instrumentation to the benchmark output:
   GraphDef node count, GraphDef byte size, first-call compile time, warm-call
   time, and method-specific failure stage.
3. Replace `_batched_model_tensors` with true batched TensorFlow algebra:
   broadcast bases over `[B, ...]` and use `tf.einsum`/batched matmul instead
   of a Python loop over `batch_size`.
4. Vectorize batched covariance-factor derivative construction in the fixture
   path where possible, so covariance derivatives are built as batched tensor
   algebra rather than row loops.
5. Treat the scalar analytical row-loop and autodiff row-loop comparators as
   small-batch correctness references only.  Do not compile them for
   `batch_size=16` unless the research question is specifically compile-stress.
6. Create a GPU analytical-only preflight that does not require the autodiff
   row-loop path to compile.  Use it to determine whether the batch-native
   analytical route is viable on GPU.
7. Investigate a true batched autodiff comparator separately.  Previous
   diagnostic `batched_static_autodiff_probe` was not yet established as a
   valid replacement for row-loop autodiff.
8. After repairs, rerun only a tiny grid first:
   - `dim=10`, `parameter_count=50`, `T=120`, `batch_size=1,4,16`
   - CPU one thread
   - analytical-only compile-size diagnostics plus small-batch correctness
     reference.
9. Only after that smoke passes, relaunch the requested CPU/GPU grid under a
   reviewed overnight plan.

## Reboot handoff
- No Kalman QR benchmark worker is currently active.
- The failed systemd unit is already stopped/failed.
- Safe after reboot to resume from this memo and the artifact map above.
- First post-reboot action should be:

```bash
cd /home/ubuntu/python/BayesFilter
git status --short
systemctl --user status kalman-qr-core-batch-grid-20260709.service --no-pager
```

- Then read this memo and avoid rerunning the full overnight grid until the
  benchmark graph repair plan is written.
