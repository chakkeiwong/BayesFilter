# Contract E--TP Phase 9 GPU/XLA Scaling Result

metadata_date: 2026-07-15
status: PASS_LGSSM_FLOAT64_GPU_XLA_WITH_COMPILE_SCALING_BLOCKER
plan: `docs/plans/bayesfilter-contract-e-tp-phase9-gpu-xla-scaling-plan-2026-07-15.md`
artifact_root: `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase9_gpu_xla_20260715/`

## Outcome

The actual successful LGSSM `finite_lookahead=8` Contract E--TP recursion
compiled and executed on the trusted NVIDIA RTX 4080 SUPER under TensorFlow
XLA at `T=10` and `T=50`. Both rungs used float64, remained on GPU, returned
finite values and total scores, passed every explicit flow/chart predicate, and
agreed with the controlling CPU executions at floating-point roundoff.

Phase 9 therefore passes the scoped float64 LGSSM GPU/XLA engineering gate. It
does not pass the repository's float32/TF32 production-target gate. The
recursive implementation is hard-coded float64, and the scalar-SV and
predator--prey recursive paths do not expose checked XLA-default factories.

## Results

| Rung | CPU/GPU value difference | Largest absolute score difference | Compile plus first execution | Warmed execution | Graph operations | Peak allocator memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `T=10` | `-2.8422e-14` | `1.0303e-13` | `51.2585 s` | `0.2330 s` | `24,787` | `16,889,344` bytes |
| `T=50` | `0` | `2.3448e-13` | `1,694.9407 s` | `1.5158 s` | `140,307` | `67,278,592` bytes |

Both artifacts report `chart_valid=true`. At `T=50`, the minimum retained
weight is `1.7782e-4`, the largest chart condition number is `2228.10`, and the
largest absolute feature residual is `9.9476e-14`.

Controlling artifacts and SHA-256 values:

- `lgssm_t10_float64_result.json`:
  `514a7056bbd064d311d0de48015c46b5aac322970d40d2b6838552c6cdf2e92e`;
- `lgssm_t50_float64_result.json`:
  `b8c3765e730c8829f1d9cc6723b6fbfe889420f4ce03692dafd8cb04c5936e86`.

## Compile-Scaling Finding

The `T=50` command did not hang. XLA remained active and eventually emitted a
slow-compilation diagnostic. Compilation took approximately 28.25 minutes,
and `ptxas` reported register spills, while the warmed execution took only
1.52 seconds. The current Python-static time loop creates a monolithic unrolled
graph whose operation count grows from 24,787 at `T=10` to 140,307 at `T=50`.

This is an engineering scaling defect in graph construction, not evidence of a
wrong value or score. A production-oriented repair should replace the
monolithic static unroll with a staged or loop-native compilation strategy and
must recheck total-gradient and fail-closed semantics. The campaign does not
silently treat the fast warmed runtime as sufficient when compile latency is
material.

## XLA Assertion Audit

XLA logged that TensorFlow `Assert` operators were ignored. The pass does not
depend on those operators alone:

- LEDH flow validity is an explicit Boolean tensor from finite and positive
  chart predicates;
- Contract E--TP input, index, row-scale, rank, roundoff, residual, and
  positive-weight conditions contribute to `valid_chart`;
- invalid reset outputs are NaN-poisoned in the compiled numerical graph; and
- the Phase 9 runner checks `valid_history`, objective/score finiteness, GPU
  placement, and preparation identity after compiled execution.

Phase 2 separately established the negative compiled fixture: an invalid chart
returns `valid_chart=false` and NaN-poisoned carried outputs. The ignored
assertion warnings are therefore visible diagnostics, not the only fail-closed
mechanism.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | dirty pre-existing research worktree; unrelated changes preserved |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu`, TensorFlow `2.19.1` |
| GPU | NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9 |
| Driver/CUDA reported by preflight | driver `591.86`; CUDA `13.1` |
| XLA / dtype / TF32 | XLA true; float64; TF32 enabled but irrelevant to float64 arithmetic |
| Data seed | deterministic LGSSM seed `81100` |
| Preparations | exact Phase 8B `T=10,50` chart artifacts, hashes bound in each result |
| Attempts | one successful attempt per rung |
| Hardware trust | `owner_designated_managed_session_visible_gpu_trusted` |
| Total measured compile/execute time | approximately `1,747.95 s`, excluding preflight and startup |

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | pass for both LGSSM rungs |
| Float64 LGSSM GPU/XLA engineering gate | pass |
| CPU/GPU numerical classification | execution-equivalent at roundoff; not a statistical claim |
| Compile scalability | blocker for a production-quality monolithic `T=50` graph |
| Float32/TF32 readiness | blocked by dtype-generic refactor requirement |
| Nonlinear GPU/XLA readiness | blocked by missing checked recursive XLA factories |
| HMC/canonical/default/leaderboard readiness | false |
| Next justified action | terminal synthesis; future staged compilation and dtype-generic work require a new scoped plan |

## Post-Run Red Team

The strongest alternative explanation for the CPU/GPU agreement is that both
execute the same finite approximation and could share a scientific modeling
error. That concern is answered only by the earlier Kalman comparison and
same-scalar derivative checks, not by GPU parity. The weakest engineering
evidence is compile scalability: one center and two horizons do not establish
general scaling. A device fallback, invalid chart, nonfinite output, or
meaningful CPU/GPU discrepancy would overturn this result; none occurred.

No float32/TF32 production readiness, nonlinear full-horizon readiness,
canonical admission, default selection, leaderboard contribution, or HMC
readiness is concluded.
