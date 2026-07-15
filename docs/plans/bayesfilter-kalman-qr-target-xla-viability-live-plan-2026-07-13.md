# Kalman QR Target-Scale XLA Viability Live Plan

Date: 2026-07-13

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `CLOSED_ENGINEERING_REPAIR_PASSED_TESTED_BOUNDARIES`

## Question

Do the repaired true-batched analytical and autodiff QR score methods compile
and execute with XLA at the benchmark horizon, or was XLA viability left
unknown because the historical supervisor pruned runtime on an unrelated
cross-shape GraphDef-equality gate?

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Whether the current true-batched pair is viable under CPU/XLA at `T=120`, beginning with `D=10`, `P=50`, and `B=1`. |
| Baseline/comparator | `batch_native_analytical_qr_score` and `batch_native_autodiff_qr_score` on the same deterministic fixture and parameter batch. The obsolete Python row-loop methods are excluded. |
| Mechanism under test | Direct per-cell XLA compilation/execution of the current builders, independent of cross-cell GraphDef topology equality. |
| Expected failure mode | A method-local compile timeout/crash, TensorList/XLA error, LLVM OOM, non-finite result, invalid shape/dtype, or analytical/autodiff disagreement. |
| Promotion criterion | In separate fresh processes, both methods complete trace, first XLA call, and warm call; return finite `float32` outputs of shapes `[B]` and `[B,P]`; preserve first/warm parity; and agree within existing `float32` value/score tolerances. |
| Promotion veto | Any missing method record, timeout/crash, compiler error, non-finite output, wrong dtype/shape, or parity failure. Peak RSS and elapsed time do not rank methods. |
| Continuation veto | Invalid fixture/comparator, corrupted or incomplete stage evidence, uncontrolled process cleanup, or failure of the analytical baseline that prevents attribution to autodiff. A failed autodiff candidate alone is not a research-direction veto. |
| Repair trigger | An autodiff first-call failure triggers a bounded `T` ladder to test reverse-loop/TensorList scaling before checkpointing or custom-gradient work. A harness-only failure triggers harness repair and rerun without reusing target evidence. |
| Explanatory diagnostics | GraphDef node/byte counts, generated-loop structure, compiler stderr, first/warm durations, peak RSS, and cross-shape topology differences. |
| Must not be concluded | No GPU viability, full `D/P/B` lattice readiness, speed ranking, memory superiority, HMC/posterior correctness, production readiness, or scientific validity from the CPU canary. |

## Code Trace And Working Diagnosis

1. `scripts/benchmark_kalman_qr_parameter_count_scaling.py` builds the
   analytical path with one true-batched forward-sensitivity loop carrying
   `[B,P,...]` derivatives.
2. The same file builds the autodiff path by applying one reverse-mode VJP to
   the batched value loop in `bayesfilter/linear/kalman_qr_tf.py`.
3. TensorFlow differentiates that value loop into a saved-intermediate forward
   loop and a reverse loop. The fixed `maximum_iterations=n_timesteps` bound is
   present, repairing the earlier XLA TensorList-size compatibility failure.
4. The historical Phase 6 supervisor evaluates normalized GraphDef equality
   across static `B/P` specializations. It prunes all XLA calls when
   `trace_common_valid` is false. Static-shape specialization is expected, so
   that predicate does not establish per-cell XLA invalidity.
5. Historical CPU LLVM and GPU layout failures name obsolete row-loop methods,
   not the repaired true-batched pair. The only direct repaired-path XLA
   evidence is tiny or `T=8`; target-horizon viability has not been measured.

## Execution

Use fresh processes, CPU only, one TensorFlow thread, GPU deliberately hidden,
`float32`, XLA enabled, and a 300-second limit per method:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_target_xla_viability_canary_2026_07_13.py \
  --dimension 10 --timesteps 120 --parameter-count 50 --batch-size 1 \
  --method-timeout-seconds 300 \
  --output docs/benchmarks/kalman_qr_target_xla_viability_cpu_d10_t120_p50_b1_2026-07-13.json
```

If and only if the cell passes, repeat with `B=4`, then `B=16`, each in a new
artifact. Stop the batch ladder at the first failed cell and localize that
failure. Do not launch the full dimension/parameter grid under this plan.

## Required Checks

Before target execution:

- Python compile of the canary harness;
- harness self-check covering valid, missing/failed-method, and parity-failure
  aggregation;
- Git diff/whitespace check limited to this lane;
- confirm no Kalman benchmark worker is live.

For each target cell:

- separate child process and peak-RSS record per method;
- durable last-entered stage, return code, timeout status, stderr tail/hash,
  GraphDef summary, first and warm output records;
- aggregate finite/dtype/shape, first/warm parity, and cross-method parity;
- exact command, Git commit, Python/TensorFlow environment, CPU/GPU/JIT/thread
  status, wall time, fixture identity, and output path.

## Resource And Stop Rules

- Maximum 300 seconds plus 10-second termination grace per method.
- At most the three `D=10,P=50,T=120` batch cells in this plan.
- Run methods sequentially; do not retain a failed child process group.
- Stop for invalid evidence or analytical-baseline failure. An autodiff failure
  selects localization rather than ending the repair direction.
- GPU execution, dependency changes, mathematical changes, and broader sweeps
  require a refreshed evidence contract.

## Skeptical Pre-Execution Audit

Status: `PASSED_AFTER_REMOVING_CROSS_SHAPE_TRACE_EQUALITY_AS_A_LAUNCH_VETO`.

- Wrong baseline: controlled by selecting only the repaired true-batched pair.
- Proxy promotion: GraphDef structure, peak RSS, and timing are explanatory;
  actual XLA completion plus numerical checks carry the viability gate.
- Missing stop conditions: per-method timeout, sequential batch escalation, and
  evidence-validity stops are explicit.
- Unfair comparison: fixture, parameter rows, dtype, thread count, device, JIT,
  and tolerances are identical; methods run in fresh processes.
- Hidden assumption: CPU viability does not imply GPU viability and is labeled
  accordingly.
- Stale context: the current source contains the TensorList bound repair, while
  the historical target pilot records zero Kalman/XLA invocations.
- Artifact adequacy: stage journals and child resource records distinguish
  trace, first-call compile/execution, warm execution, timeout, and crash.

## Artifacts

- Harness:
  `docs/benchmarks/run_kalman_qr_target_xla_viability_canary_2026_07_13.py`
- First result:
  `docs/benchmarks/kalman_qr_target_xla_viability_cpu_d10_t120_p50_b1_2026-07-13.json`
- Result note:
  `docs/plans/bayesfilter-kalman-qr-target-xla-viability-result-2026-07-13.md`

## Pilot Result And Handoff

The predeclared `D=10`, `P=50`, `T=120`, `B in {1,4,16}` CPU/XLA ladder
passed for both repaired true-batched methods. Every first and warm result was
finite with the expected dtype/shape, and the largest analytical/autodiff score
residual was `1.6391277313232422e-07` at `B=16`.

This falsifies blanket CPU/XLA incompatibility for the repaired pair and shows
that the former Phase 6 pilot was pruned without testing its selected cells. It
does not establish viability at `D in {20,30}`, `P=150`, or on GPU. The exact
evidence and resource measurements are recorded in the result note above.

Next handoff: refresh this live plan with a bounded maximum-shape CPU cell
(`D=30`, `P=150`, `B=16`, `T=120`) and then a managed-session GPU/XLA canary if
the CPU boundary passes. The refresh must state why a maximum-shape pass is a
boundary canary rather than proof that every static specialization passes.

## Maximum-Shape CPU Boundary Refresh

Status: `SKEPTICAL_AUDIT_PASSED_FOR_ONE_CELL`.

Question: does the largest requested static cell compile and execute for both
repaired methods under the same CPU/XLA contract, or does dimension/parameter
growth expose the reverse-loop compiler-memory risk not seen at `D=10,P=50`?

Exact command:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_target_xla_viability_canary_2026_07_13.py \
  --dimension 30 --timesteps 120 --parameter-count 150 --batch-size 16 \
  --method-timeout-seconds 300 \
  --output docs/benchmarks/kalman_qr_target_xla_viability_cpu_d30_t120_p150_b16_2026-07-13.json
```

The baseline, primary criterion, vetoes, stage evidence, timeout, and nonclaims
are unchanged. This cell is deliberately the maximum requested shape because
it directly stresses the largest derivative tensors and saved activations. A
pass supports viability at that exact boundary and weakens a monotone
shape-memory failure hypothesis; it does not prove every intermediate static
specialization or GPU backend. A failure is localized by method and stage. No
other CPU lattice cell is authorized by this refresh.

Observed result: both methods passed the maximum CPU cell. Analytical/autodiff
value and score parity passed; the maximum score residual was
`2.980232238769531e-07`. The analytical child used 1,022,344 KiB peak RSS and
the autodiff child used 993,200 KiB. Single-run timing/RSS remains explanatory.

## Managed GPU/XLA Pilot Refresh

Status: `SKEPTICAL_AUDIT_PASSED_FOR_ONE_GPU_CELL`.

Question: does the repaired pair compile and execute on the repository's
default GPU/XLA target at the smallest benchmark-horizon pilot cell, with
actual GPU placement and device provenance verified?

GPU preflight found two RTX 4080 SUPER devices. Physical GPU 1 was idle with
18 MiB used and no compute process; GPU 0 carried desktop services. The pilot
therefore exposes only physical GPU 1 as logical `GPU:0` inside each fresh
child. Each child must record exactly one physical and logical GPU, TF32
enabled, output tensors placed on `device:GPU:0`, GPU peak memory, XLA compiler
stderr, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`. Any CPU fallback fails
the gate.

Exact command:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_target_xla_viability_canary_2026_07_13.py \
  --device gpu --cuda-visible-devices 1 \
  --dimension 10 --timesteps 120 --parameter-count 50 --batch-size 1 \
  --method-timeout-seconds 300 \
  --output docs/benchmarks/kalman_qr_target_xla_viability_gpu_d10_t120_p50_b1_2026-07-13.json
```

The primary criterion and numerical vetoes remain unchanged. Additional hard
vetoes are missing/ambiguous GPU identity, TF32 disabled, output placement off
logical `GPU:0`, missing device-memory provenance, or another compute process
appearing on physical GPU 1 before launch. A pilot pass authorizes a refreshed
maximum-shape GPU boundary plan, not the launch itself. No other GPU cell is
authorized here.

First attempt classification: `INVALID_NONTRUSTED_SANDBOX_GPU_VISIBILITY`.
Both children stopped at `before_tensorflow_import`/device discovery with zero
visible TensorFlow GPUs and `CUDA_ERROR_NO_DEVICE`; neither fixture, method,
trace, nor XLA call ran. Concurrent trusted `nvidia-smi` still showed physical
GPU 1 healthy, idle, and with 18 MiB used. Under the repository GPU policy this
attempt cannot establish a machine, driver, TensorFlow, Kalman, or XLA failure.
Rerun the exact command in a trusted/elevated context and let that result
replace the public pilot artifact; the invalid attempt remains diagnostic-only.

Trusted result: the analytical method passed with verified GPU placement,
TF32, XLA compilation, finite first/warm outputs, and 184,710,656 bytes peak
TensorFlow GPU allocation. The autodiff method aborted with signal 6 during its
first XLA call. XLA's GPU GEMM fusion autotuner reported:

```text
FAILED_PRECONDITION: Can not combine dim orders and requirements
Failure occured when compiling fusion gemm_fusion_dot.124
```

The fused HLO is anchored to `gradient_tape/MatMul_2` and
`gradient_tape/matmul_2/MatMul_1`. Source order maps `MatMul_2` to the third
factor covariance in `_batched_model_tensors`, the initial covariance
`initial_factor @ initial_factor^T`, before the Kalman time loop. The failing
fusion combines batch-one bitcasts/transposes and the reverse matmul. This is
not the earlier TensorList bound error, a timeout, an OOM, or a runtime
reverse-loop failure.

## GPU GEMM-Fusion Discriminator Refresh

Status: `SKEPTICAL_AUDIT_PASSED_FOR_TWO_SEQUENTIAL_FLAG_PROBES`.

Question: is the current GPU failure caused by the GEMM autotuner/fusion route
rather than the mathematical graph being generally uncompilable?

Run the exact failing `D=10,P=50,B=1,T=120` pair with one XLA control changed
at a time:

1. `XLA_FLAGS=--xla_gpu_autotune_level=0`. This is already used by the legacy
   overnight harness, but its retained failure named the obsolete row-loop
   method, so the current true-batched path must be tested directly.
2. Only if probe 1 fails with a layout/fusion compiler error,
   `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, forcing the non-Triton GEMM
   route while leaving XLA, GPU, TF32, fixture, method, and tolerance intact.

Each probe writes a separate artifact. Pass requires both methods and every
existing numerical/device/provenance check. A probe that passes nominates the
flag as a bounded compiler workaround; it does not authorize a default flag or
establish performance. If both fail, the next discriminator is a code-local
value-only covariance-construction arm on the same cell. Do not run a timestep
ladder unless needed to confirm that this pre-loop fusion failure is invariant
to `T`; current HLO attribution already weakens the horizon/TensorList
hypothesis.

Stop after the first fully valid flag probe. Do not combine flags in these
probes, do not change TensorFlow packages, and do not launch a larger GPU cell.

Observed discriminator results:

- `--xla_gpu_autotune_level=0` did not repair the failure. It changed the fatal
  autotuner abort into a recoverable `FailedPreconditionError`, but the same
  dimension-order conflict remained in the autodiff first call.
- `--xla_gpu_enable_triton_gemm=false` passed. Both methods compiled with XLA,
  executed on GPU, recorded TF32 and device memory, produced finite correctly
  shaped outputs, preserved first/warm parity, and agreed within tolerance.
  The maximum score residual was `1.6391277313232422e-07`.

Conclusion at the pilot cell: the failure is specific to XLA's Triton GEMM
fusion/lowering for the batch-one covariance-gradient layout. It is not a
general GPU/XLA incompatibility of the repaired graph. The non-Triton GEMM
route is a viable workaround candidate.

## No-Triton Maximum GPU Boundary Refresh

Status: `SKEPTICAL_AUDIT_PASSED_FOR_ONE_BOUNDARY_CELL`.

Question: does the single-flag workaround remain viable at the maximum
requested `D=30,P=150,B=16,T=120` shape, including device memory, output
validity, and analytical/autodiff parity?

Exact command:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_target_xla_viability_canary_2026_07_13.py \
  --device gpu --cuda-visible-devices 1 \
  --dimension 30 --timesteps 120 --parameter-count 150 --batch-size 16 \
  --method-timeout-seconds 300 \
  --output docs/benchmarks/kalman_qr_target_xla_viability_gpu_no_triton_gemm_d30_t120_p150_b16_2026-07-13.json
```

The same method, numerical, device, TF32, provenance, process-isolation, and
timeout gates apply. The preflight must again show physical GPU 1 idle. A pass
nominates the no-Triton flag for the benchmark launcher and authorizes focused
launcher/contract repair; it is not a universal TensorFlow policy or speed
claim. A failure stops expansion and is localized by method/stage. No
intermediate or full GPU lattice is authorized by this refresh.

Observed result: both repaired methods passed the maximum GPU cell under the
single no-Triton flag. Peak TensorFlow device allocation was 137,465,344 bytes
for analytical and 271,683,072 bytes for autodiff, well below the 32 GiB device
capacity. The maximum score residual was `0.00014348328113555908`, within the
predeclared tolerance. This establishes maximum-cell GPU/XLA viability for the
workaround, but it does not yet justify making a process-wide compiler flag the
benchmark default.

## Code-Local Triton Repair Discriminator Refresh

Status: `SKEPTICAL_AUDIT_PASSED_FOR_EXISTING_PARITY_TESTED_VARIANT`.

A process-wide no-Triton flag could change the performance comparison being
measured. Before promoting that policy, test the existing
`build_batch_native_autodiff_value_only_explicit_fn` under default Triton at
the exact failing pilot cell. This builder preserves the same batched Kalman
value and reverse-mode VJP but builds only the eight value tensors the
likelihood consumes and explicitly broadcasts static bases to `[B,...]`.
Earlier CPU diagnostics established deterministic value/score parity and a
smaller GraphDef; they did not test this GPU layout failure.

The canary may expose `--autodiff-variant full_helper|value_only_explicit` only
as selection plumbing. Run the analytical baseline and
`value_only_explicit` autodiff at `D=10,P=50,B=1,T=120` with default
`XLA_FLAGS` unset. The same numerical/device/provenance checks apply, and the
artifact must record the selected builder. If it passes, repeat that variant at
the maximum GPU cell before changing production benchmark dispatch. If it
fails, retain the no-Triton workaround and do not test speculative rewrites in
this phase. No timing ranking is supported by either outcome.

Observed result: `value_only_explicit` failed under default Triton with the
same batch-one `gradient_tape/MatMul_2` / `matmul_2/MatMul_1` fusion and
dimension-order conflict. Removing unused derivative construction and making
bases explicitly batch-shaped therefore does not repair this compiler defect.
The no-Triton workaround remains selected; no numerical/model construction
change is promoted.

## Benchmark-Default Policy Verification Refresh

Status: `SKEPTICAL_AUDIT_PASSED_FOR_SCOPED_EXECUTION_POLICY_EDIT`.

Implementation scope:

- `scripts/benchmark_kalman_qr_parameter_count_scaling.py` applies
  `--xla_gpu_enable_triton_gemm=false` before TensorFlow import only for direct
  GPU/auto + XLA benchmark invocations when the caller has not already made an
  explicit Triton choice.
- CPU and non-JIT execution remain unchanged. Unrelated existing XLA flags are
  preserved and the no-Triton flag is appended. An explicit caller Triton
  choice wins.
- The benchmark manifest records action, requested device, JIT state, input
  flags, effective flags, and evidence basis.
- No BayesFilter library API, Kalman math, global shell environment, TensorFlow
  package, or repository-wide XLA policy changes.

Focused unit checks must cover GPU default, auto default, CPU exclusion,
non-JIT exclusion, caller override, and unrelated-flag preservation. Then run
the actual method-isolated supervisor with `XLA_FLAGS` unset:

```bash
env -u XLA_FLAGS CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --dimensions 10 --parameter-counts 50 --timesteps 120 --batch-size 1 \
  --dtype float32 --device gpu --repeats 1 --timeout-seconds 300 \
  --methods batch_native_analytical_qr_score \
    batch_native_autodiff_qr_score \
  --output-dir /tmp/kalman_qr_target_xla_default_policy_gpu_d10_p50_b1 \
  --no-resume --jit-compile --tf32-enabled
```

The supervisor must complete both methods, parity, dtype/shape/finite, and
method-isolation checks. Each direct child must record effective
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. This is implementation
verification, not a new performance comparison. A failure triggers policy
plumbing repair; it does not invalidate the already passing explicit-flag
canaries.

## Closure

The real method-isolated GPU supervisor completed both methods with parent
`XLA_FLAGS` unset. Every aggregate identity, record, finite, dtype/shape,
pair-completeness, and comparator-parity check passed. Each child recorded the
benchmark-local no-Triton action, effective flag, actual `/GPU:0` output
placement, and managed-session GPU trust basis. The raw status was copied to
`docs/benchmarks/kalman_qr_target_xla_default_policy_gpu_d10_t120_p50_b1_status_2026-07-13.json`.

The focused QR numerical suite subsequently completed with `115 passed`. The
policy matrix and full parameter-count harness suite passed with `6` and `56`
tests respectively, the canary fail-closed self-check passed all six checks,
and no benchmark worker remained live. A bounded read-only Claude Opus review
of the benchmark source returned `VERDICT: AGREE` with no policy defect. Its
residual risks were limited to conservative pre-import handling of `auto` when
that request later selects CPU and token-based parsing of unusually quoted or
malformed `XLA_FLAGS`; neither affects the verified explicit-GPU path.

Final conclusion: CPU/XLA is viable as-is for the tested pilot ladder and
maximum requested cell. GPU/XLA is viable for the tested pilot and maximum
cells on the TensorFlow 2.20 / RTX 4080 SUPER stack when the benchmark uses
`--xla_gpu_enable_triton_gemm=false`. Default Triton remains vetoed for the
autodiff method on this stack because its covariance-gradient fusion fails with
the recorded dimension-order conflict. The benchmark-local pre-import policy
is the selected repair; the caller can explicitly override it.

This closes the XLA engineering blocker and hands off to the static benchmark
lattice. It does not establish every specialization, any performance or memory
ranking, posterior/HMC correctness, production readiness outside this
benchmark, or scientific validity. The complete evidence and provenance caveat
are in
`docs/plans/bayesfilter-kalman-qr-target-xla-viability-result-2026-07-13.md`.
