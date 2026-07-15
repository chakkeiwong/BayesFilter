# Kalman QR Target-Scale XLA Viability Result

Date: 2026-07-13

Status: `CLOSED_ENGINEERING_REPAIR_PASSED_TESTED_BOUNDARIES`

Plan:
`docs/plans/bayesfilter-kalman-qr-target-xla-viability-live-plan-2026-07-13.md`.

## Final Result

The current true-batched analytical and autodiff QR score methods are viable
with XLA on the tested CPU and GPU boundary cells. GPU viability on the tested
TensorFlow 2.20 / RTX 4080 SUPER stack requires the benchmark-local compiler
policy `--xla_gpu_enable_triton_gemm=false` unless the caller explicitly makes
a different Triton choice.

The policy is applied before TensorFlow import only for direct benchmark
invocations that request GPU or auto device selection with XLA enabled. It is
not applied to CPU or non-JIT runs, it preserves unrelated `XLA_FLAGS`, and an
explicit caller Triton setting wins. No Kalman equation, library API,
TensorFlow package, global shell setting, or repository-wide XLA policy was
changed.

This closes the XLA engineering blocker for launching the requested benchmark
under the recorded no-Triton policy. It does not prove that every static
`D/P/B` specialization passes, and it provides no speed or memory ranking.

## Causal Trace

Three distinct problems had been conflated.

### 1. Earlier TensorList incompatibility

`build_batch_native_autodiff_fn` performs one reverse-mode VJP through
`tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop`. TensorFlow saves
loop intermediates in TensorLists for the reverse pass. The earlier batched
value loop did not provide a fixed list bound, so XLA rejected the generated
reverse loop.

The existing source repair adds `maximum_iterations=n_timesteps` to the
`tf.while_loop` in `bayesfilter/linear/kalman_qr_tf.py`. The condition remains
`t < n_timesteps`, so the bound supplies compiler shape information without
changing the Kalman recursion or number of executed time steps. CPU/XLA now
passes at both the pilot and maximum requested shapes.

### 2. Historical harness false veto

The historical Phase 6 supervisor required normalized GraphDef identity across
different static `B/P` signatures. Static signatures legitimately specialize
TensorFlow graphs, so cross-shape graph identity is not a per-cell XLA
compileability condition. When `trace_common_valid=false`, the supervisor
pruned the target XLA calls. Its retained result therefore showed that the
launch gate failed, not that either repaired target method failed XLA.

Historical LLVM/GPU failures for `scalar_row_loop_score` and
`autodiff_row_loop_score` also concerned obsolete row-loop methods, not the
current true-batched pair.

### 3. Current GPU Triton GEMM compiler failure

Direct execution exposed a real, different GPU compiler defect:

1. `_batched_model_tensors` builds transition, observation, and initial
   covariance matrices from parameterized factors.
2. The third factor product is
   `initial_factor @ matrix_transpose(initial_factor)`.
3. `GradientTape` differentiates that batched matrix product before the Kalman
   time loop.
4. On the TensorFlow 2.20 default GPU route, XLA's Triton GEMM fusion compiler
   aborts while compiling the associated reverse matmul. The HLO names
   `gradient_tape/MatMul_2` and
   `gradient_tape/matmul_2/MatMul_1` and reports
   `FAILED_PRECONDITION: Can not combine dim orders and requirements`.

At `B=1`, the failing fusion contains bitcasts between `[1,D,D]` and `[D,D]`,
transposes, and the reverse matmul with incompatible layout requirements. The
failure occurs during the first XLA call, before executing the differentiated
Kalman loop. It is not the old TensorList error, an OOM, a timeout, numerical
instability, or evidence that the Kalman mathematics is invalid.

The discriminator results isolate the backend route:

- Default Triton GEMM: analytical passed; autodiff aborted in the named fusion.
- `--xla_gpu_autotune_level=0`: the abort became a recoverable
  `FailedPreconditionError`, but the same layout conflict remained.
- Existing `value_only_explicit` model construction: failed in the same fusion,
  rejecting unused derivative construction and implicit base broadcasting as
  sufficient repairs.
- `--xla_gpu_enable_triton_gemm=false`: the unchanged analytical and autodiff
  computations both compiled and passed through XLA's non-Triton GPU GEMM
  route.

The supported diagnosis is therefore a TensorFlow 2.20 GPU Triton
fusion/lowering defect for this covariance-gradient layout on the tested
stack. It is not a general GPU/XLA incompatibility of the graph. This does not
prove the precise upstream compiler implementation defect beyond the captured
HLO and flag discriminator.

## Boundary Evidence

| Device/policy | Shape `(D,P,B,T)` | Analytical | Autodiff | Score max residual | Value max residual | Result |
| --- | --- | --- | --- | ---: | ---: | --- |
| CPU/XLA, GPU hidden | `(10,50,1,120)` | passed | passed | `1.6391e-07` | `0` | passed |
| CPU/XLA, GPU hidden | `(10,50,4,120)` | passed | passed | `1.1921e-07` | `0` | passed |
| CPU/XLA, GPU hidden | `(10,50,16,120)` | passed | passed | `1.6391e-07` | `0` | passed |
| CPU/XLA, GPU hidden | `(30,150,16,120)` | passed | passed | `2.9802e-07` | `0.0003662` | passed |
| GPU/XLA, default Triton | `(10,50,1,120)` | passed | compiler abort | N/A | N/A | failed discriminator |
| GPU/XLA, autotune level 0 | `(10,50,1,120)` | passed | compiler error | N/A | N/A | failed discriminator |
| GPU/XLA, no Triton GEMM | `(10,50,1,120)` | passed | passed | `1.6391e-07` | `0` | passed |
| GPU/XLA, no Triton GEMM | `(30,150,16,120)` | passed | passed | `0.00014348` | `0.0446777` | passed |
| GPU/XLA, explicit value-only builder | `(10,50,1,120)` | passed | compiler abort | N/A | N/A | failed discriminator |

All passing cells satisfied the predeclared `rtol=atol=2e-4` comparison,
finite `float32` output, shape, first/warm parity, and single-trace checks. The
maximum GPU cell recorded peak TensorFlow device allocation of `137,465,344`
bytes for analytical and `271,683,072` bytes for autodiff. These one-run memory
measurements are descriptive only.

The maximum-shape cells are boundary canaries. They weaken a simple monotone
shape or memory failure hypothesis but are not proof for every intermediate
static specialization.

## Production-Path Verification

The real method-isolated supervisor was run with parent `XLA_FLAGS` unset at
`D=10,P=50,B=1,T=120`. It completed both methods and all six aggregate checks:
identity integrity, record integrity, finite output metadata, expected
dtype/shape, primary-pair completeness, and comparator parity.

Each direct child recorded:

- `action=benchmark_default_no_triton_applied`;
- `input_xla_flags=UNSET`;
- `effective_xla_flags=--xla_gpu_enable_triton_gemm=false`;
- `jit_compile=true`;
- actual output placement on `/GPU:0`;
- trust basis `owner_designated_managed_session_visible_gpu_trusted`.

The copied status artifact preserves the outer supervisor record verbatim.
Its outer schedule correctly records that the parent input `XLA_FLAGS` was
unset, but it also retains a stale historical
`trust_basis=gpu_hidden_cpu_debug_reference`. Runtime trust and placement are
therefore taken from the two child records, which both contain the correct GPU
trust basis and output devices. This metadata caveat does not change the
compiler or numerical result.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept tested CPU/XLA viability | Both methods passed pilot ladder and maximum cell | No CPU hard veto fired | Untested intermediate specializations | Use CPU only for explicit reference/debug work | Full lattice or performance ranking |
| Accept tested GPU/XLA viability under scoped no-Triton policy | Both methods passed pilot, maximum cell, and real supervisor | No workaround-path hard veto fired | Other shapes and performance effect of the compiler route | Launch the benchmark with the policy recorded in every artifact | Universal TensorFlow fix, default-Triton viability, or full lattice proof |
| Reject default Triton for this benchmark stack | Autodiff first XLA call reproducibly failed; analytical remained valid | Hard compiler veto fired | Whether a later TensorFlow release fixes it | Keep caller override and retest after stack changes | Mathematical or algorithmic failure |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Passed focused source, harness, process-isolation, policy, and supervisor checks. |
| Numerical validity | Passed deterministic analytical/autodiff parity at the listed CPU and no-Triton GPU cells. |
| XLA viability | Established only for the listed cells and policies. |
| Memory/performance | Descriptive only; no replicated comparison or ranking. |
| Scientific interpretation | No scientific claim was tested. |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Default Triton is vetoed for current GPU autodiff runs; no hard veto fired on CPU or no-Triton GPU boundary runs. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Trace time, call time, graph size, RSS, and GPU allocation. |
| Default-readiness | Ready as a benchmark-local execution policy, not as a repository-wide TensorFlow policy. |
| Next evidence needed | Execute the requested static benchmark lattice with identical recorded compiler policy before analyzing performance. |

## Checks

- Policy matrix: `6 passed`.
- Full parameter-count harness suite: `56 passed`.
- Canary self-check: all six fail-closed/provenance checks passed.
- Focused QR numerical suite:
  `115 passed` in `201.79s` with `CUDA_VISIBLE_DEVICES=-1`.
- Python compilation and lane-limited whitespace checks: passed.
- Process cleanup: no Kalman benchmark worker remained live.
- Bounded read-only Claude Opus source review: `VERDICT: AGREE`; no defect
  found in direct-invocation scoping, explicit CPU/non-JIT exclusion, caller
  override, or unrelated-flag preservation.

The reviewer noted two residual implementation risks. An `auto` request must
be classified before TensorFlow device discovery, so the GPU-specific flag can
remain present if `auto` later selects CPU; explicit CPU requests remain
unchanged. Triton-choice detection also assumes normally whitespace-tokenized
`XLA_FLAGS` rather than unusually quoted or malformed input. Neither risk
affects the verified explicit-GPU supervisor path.

The focused numerical command was:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_kalman_qr_batch_native_autodiff.py
```

The warnings were TensorFlow AutoGraph/gast deprecations under Python 3.13;
they did not fail a test.

## Artifacts

| Evidence | Artifact | SHA-256 |
| --- | --- | --- |
| CPU pilot `B=1` | `docs/benchmarks/kalman_qr_target_xla_viability_cpu_d10_t120_p50_b1_2026-07-13.json` | `32aec706255f428a5d80773c22916eccedb39c821ac133362c4355e5f2afffb4` |
| CPU pilot `B=4` | `docs/benchmarks/kalman_qr_target_xla_viability_cpu_d10_t120_p50_b4_2026-07-13.json` | `bf5fc97cb3b42f977f73cf884279072b14ca60e9e42285817ad075ee685acafe` |
| CPU pilot `B=16` | `docs/benchmarks/kalman_qr_target_xla_viability_cpu_d10_t120_p50_b16_2026-07-13.json` | `e6901cbab1a6dc756c4b1ba22fd95426d720a1401b9ee7e02d830f8276df727f` |
| CPU maximum | `docs/benchmarks/kalman_qr_target_xla_viability_cpu_d30_t120_p150_b16_2026-07-13.json` | `9ef815d807b199d6c5ec718a220ef794e0754de6f08e3e3429871f0d8ca18f21` |
| GPU default failure | `docs/benchmarks/kalman_qr_target_xla_viability_gpu_d10_t120_p50_b1_2026-07-13.json` | `ffaf7fcd8f28e6ac5164006f3c6c6bd8798bbc109cd271f01c85f1008e2a691b` |
| GPU autotune-zero failure | `docs/benchmarks/kalman_qr_target_xla_viability_gpu_autotune0_d10_t120_p50_b1_2026-07-13.json` | `7fff37004bc4fe5d10e4eb1423f22a5a03d396c3beb7c3c25518733c8813cd45` |
| GPU no-Triton pilot | `docs/benchmarks/kalman_qr_target_xla_viability_gpu_no_triton_gemm_d10_t120_p50_b1_2026-07-13.json` | `f3169888881a1b5bb65059478198021a02ce3fda420a42da458bca76dc64113c` |
| GPU no-Triton maximum | `docs/benchmarks/kalman_qr_target_xla_viability_gpu_no_triton_gemm_d30_t120_p150_b16_2026-07-13.json` | `984caded42522cbfad895513403cc83ed9cfccff856e1f04179f0e433d9841e5` |
| GPU explicit-builder failure | `docs/benchmarks/kalman_qr_target_xla_viability_gpu_value_only_explicit_d10_t120_p50_b1_2026-07-13.json` | `9aefb160b49838902af30dc8b2720a8c8035146ceb778b988cda4b48120dd54f` |
| Real supervisor status | `docs/benchmarks/kalman_qr_target_xla_default_policy_gpu_d10_t120_p50_b1_status_2026-07-13.json` | `49b79f0d593cafd46ac129bc4773d0d5cb677561248483cf96ce65fd3df0916b` |

Each canary JSON embeds the exact command, commit, environment, device/JIT/TF32
settings, deterministic fixture identity, stage records, compiler stderr,
resource observations, output checks, and artifact path.

## Post-Run Red Team

The strongest alternative explanation is narrower than the result: the
no-Triton route could still fail at an untested static specialization, and it
could change relative performance even while preserving correctness. The full
benchmark lattice is the next discriminating evidence for coverage, and its
performance comparisons must keep the compiler policy identical and visible
across methods.

A future TensorFlow/CUDA/compiler-stack change could remove or alter the
failure. That would justify retesting default Triton; it would not retroactively
invalidate these artifacts. The weakest current evidence is performance. The
strongest evidence is the one-flag backend discriminator on an unchanged graph,
the maximum-shape boundary pass, and the real launcher pass with child-level
GPU/XLA provenance.
