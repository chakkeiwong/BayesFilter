# q=20 GPU-Native Eigh Localization Plan

Date: 2026-07-31
Status: `REVIEWED_READY_FOR_LOCALIZATION_ONLY`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Is the measured q=20 `~568 s` update primarily an artifact of the current hybrid host-callback principal-square-root backend, and can the existing device-side `tensorflow_eigh_strict` backend preserve the same target value/score/status while materially reducing update time? |
| Mechanism under test | Replace only `principal_sqrt_backend="compiled_custom_op"` with `"tensorflow_eigh_strict"`; retain the repaired v2 target, q=20 model, analytic four-coordinate score, batch-native route, XLA, batch 100, transport, optimizer, dtype, seeds, and GPU hardware class. |
| Exact comparator | Identical two-row q=20 theta batch evaluated by separately compiled custom-callback and strict TensorFlow-eigh programs; then a `(32,32)` trainer using only the parity-passed strict backend. |
| Promotion criterion | Exact target signature equality; finite and hard-valid rows for both backends; maximum value difference `<=1e-8`; score differences satisfy absolute `<=1e-7` or relative `<=1e-7`; strict-backend trainer completes one first and at least one warm batch-100 update with finite receipts and HLO extraction. |
| Promotion veto | Target signature mismatch, invalid status, nonfinite output, parity tolerance failure, non-XLA execution, or missing source/operation receipt. |
| Continuation veto | No trusted GPU, memory-growth failure, strict backend compile/execute failure, material cap, or outer timeout. |
| Repair trigger | A receipt-only defect may be repaired under the same root and total budget. A target/math/backend change beyond the declared strict TensorFlow implementation requires a new plan. |
| Explanatory only | Compile time, warm target time, update time, loss, gradients, clipping, allocator bytes, and speed ratio. |
| Must not be concluded | No tuning selection, training improvement, convergence, architecture ranking, posterior validity, HMC readiness, campaign budget, or default readiness. |

## Root-Cause Evidence Before Execution

The prior update was XLA-compiled: TensorFlow logged `Compiled cluster using
XLA!`, the trainer used `tf.function(..., jit_compile=True)`, and HLO extraction
completed. The critical custom calls were not GPU-native numerical kernels:

- the CUDA callbacks call `cudaDeviceSynchronize()`;
- copy the complete matrices from device to host;
- call serial C++ loops over batch rows using Eigen
  `SelfAdjointEigenSolver`;
- copy results back to device; and
- synchronize the stream again.

For q=20, each update carries batch 100 through 30 sequential filter steps.
The state dimension is 60, innovation dimension is 20, the augmented
principal square root is 80 by 80, and the unscented rule has 161 points.
The existing code also has `tensorflow_eigh_strict`, which implements the same
strict input formulas using TensorFlow `tf.linalg.eigh` and tensor algebra on
the selected device. It is an eligible localization candidate, not a promoted
replacement.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Tiny parity | Two fixed rows, batch-native q=20 target, separate XLA functions, first and warm calls for both backends, full value/score/status comparison. |
| Native update timing | Fresh `(32,32)` trainer, batch 100, `lr=2e-4`, initialization scale `0.01`, clip `10`, updates 1 through 3 maximum. Update 1 is compile-inclusive; updates 2-3 are warm receipts. |
| XLA proof | Extract HLO from the strict-backend trainer and require nonempty text containing `ENTRY`. |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-gpu-native-eigh-localization-2026-07-31/r1/`. |
| Baseline artifact | Prior custom-backend warm medians remain `565.9442223530059 s` for `(32,32)` and `568.1131881010078 s` for `(64,64)`. They are historical same-hardware timing comparators, not rerun in this budget. |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| `tensorflow_eigh_strict` | Existing device-side implementation; repair hypothesis | Numerical differences accumulate over the recursion | Direct value/score/status parity before trainer construction |
| Two parity rows | Convenience smallest discriminating batch | Misses batch-size-dependent layout defect | Require exact static batch; treat pass as nomination only, not full batch-100 parity proof |
| Three trainer updates | Bounded localization maximum | Too few for thermal or training claims | Report every call; no campaign projection or training claim |
| `(32,32)` | Lower-capacity prior mechanics-viable arm | Transport-specific overhead differs | Target dominates prior `(32,32)`/`(64,64)` timings; report scope exactly |
| Parity tolerances | Inherited repository custom-versus-TensorFlow-eigh tests | Too loose for this target | Report raw maximum differences; failure vetoes promotion |

## Skeptical Pre-Execution Audit

- Wrong baseline: no. Target/model/score/transport settings remain fixed; only
  the square-root execution backend changes.
- Proxy promotion: no. Tiny parity and three updates can nominate a repair only.
- Missing stop conditions: parity, finite/status, XLA, GPU memory policy,
  receipt, material cap, and timeout vetoes are explicit.
- Hidden environment mismatch: both backends execute in one trusted GPU process
  with the same TensorFlow version and visible device.
- Artifact adequacy: every first/warm target call and update has an atomic
  receipt; timeout preserves the active operation.
- Misleading pass: a speedup does not prove broad parity, training quality, or
  a new campaign budget. A fresh batch-100 parity and longer timing preflight
  would still be required before campaign execution.

Audit decision: `PASS_FOR_LOCALIZATION_ONLY`.

## Budget

- Remaining user authorization before this diagnostic:
  `2,152.4224067549985 s`.
- Material cap: `1,800 s`.
- Nonmaterial reserve: `352.4224067549985 s`.
- One trusted GPU process and one outer timeout of `1,800 s` maximum.
- No tuning arm, final stream, HMC, package build, custom-op rebuild, or source
  implementation change is authorized.

## Planned Commands

1. CPU-hidden focused surface and arithmetic tests.
2. Trusted GPU localization on an idle device with TensorFlow memory growth.
3. CPU-hidden artifact/hash validation and result/reset-note correction.

