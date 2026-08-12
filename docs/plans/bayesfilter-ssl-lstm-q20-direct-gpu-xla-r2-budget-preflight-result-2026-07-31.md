# q=20 Direct GPU/XLA r2 Budget Preflight Result

Date: 2026-07-31
Status: `HYBRID_HOST_CALLBACK_TIMING_COMPLETE_GPU_NATIVE_BUDGET_INVALID`

## Result

The repaired q=20 target passed exact CPU/trusted-GPU identity parity. XLA
operation receipts were obtained for both `(32,32)` and `(64,64)` transports.
The measured route was not GPU-native: its principal-square-root and Sylvester
CUDA callbacks synchronize the device, copy tensors to the host, run serial
Eigen eigensolvers, copy results back to the device, and synchronize again.

The prior projection of `1,755,960.4640257251 s` (`20.3236 days`) is therefore
a valid budget for this exact hybrid host-staged implementation only. It is
`wrong relative to a GPU-native campaign-budget claim` and must not be used as
the requested budget for the intended GPU route. A new budget is `not checked`
until the existing device-side `tensorflow_eigh_strict` repair candidate or a
true CUDA solver implementation passes parity and timing diagnostics.

This preflight did not launch tuning, final training, or HMC. No material
campaign budget is currently defensible for the intended GPU-native route.

## XLA Versus GPU-Native Verdict

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Trainer update was XLA-compiled | `correct` | `tf.function(..., jit_compile=True)`, TensorFlow log `Compiled cluster using XLA!`, and nonempty HLO SHA-256 `be00ecf5de4b4063542e7667e3d4f72273049fa8ce1cab4a29b99994a05f2460`. |
| Critical square-root/Sylvester work was GPU-native | `wrong relative to that claim` | Both CUDA callbacks call `cudaDeviceSynchronize`, copy device-to-host, execute host `solve_batch`/`principal_sqrt_batch`, copy host-to-device, and synchronize the stream. |
| Nine-minute update is optimized GPU performance | `unsupported` | The measured route serializes host callbacks inside the 30-step filter recursion. No GPU-native backend timing was obtained. |
| 20-day projection is the required GPU campaign budget | `wrong relative to that claim` | It prices the hybrid host-staged backend and its synchronization/copy overhead. |
| 20-day projection prices the exact measured implementation | `correct` | Direct warm receipts and exact prior-protocol call counts support that narrower statement. |

At q=20, batch 100, the state dimension is 60, innovation dimension is 20,
the augmented principal square root is `80 x 80`, and the unscented rule has
161 points. The active analytic score uses a sequential 30-step
forward-sensitivity recursion. At minimum, each step invokes the augmented
principal-square-root and four-direction Sylvester operations through the
host-staged callbacks; diagnostic eigendecompositions add further device-side
work. The host loops process all 100 batch rows serially.

The existing `tensorflow_eigh_strict` path implements the strict square root
and Sylvester formulas with `tf.linalg.eigh` and TensorFlow tensor algebra on
the selected device. It is a repair hypothesis, not yet parity- or
performance-promoted for q=20. A bounded localization plan was prepared, but
two trusted launch approval reviews timed out before process creation and the
non-elevated sandbox hid the GPU. Therefore its q=20 result is `not checked`.

## Target And Computation Verdict

| Item | Verdict |
| --- | --- |
| Claimed target | Repaired q=20 v2 SSL-LSTM complexity target with static fixture and observations constructed explicitly on `/CPU:0` and accessed through the repository-issued direct batch-native status-bearing binding. |
| Quantity actually compared | Complete canonical CPU-hidden and trusted-GPU signature payloads, payload SHA-256, target signature, adapter signature, static-data devices, and source closure. |
| Equality verdict | `correct`: every identity field matched exactly. Target signature and canonical payload SHA-256 are both `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; adapter signature is `c990a3a97d62a2557f0466a7ca1f0e009d5e35708156aeca1ce801257db48c73`. |
| Timing quantity | Direct elapsed wall time per construction, optimizer update, validation, status, support/export, exact `[256,4]` audit, and HLO extraction operation on trusted GPU 1. |
| Relation to campaign budget | Derived projection of the exact r1 protocol counts for the current hybrid host-staged backend only. It is ineligible as a GPU-native budget. |
| Not proved | Training quality, convergence, superiority, posterior correctness, HMC readiness, predictive validity, production/default readiness, or adequacy of the r1 protocol for any scientific claim. |

## Timing Evidence

| Diagnostic | `(32,32)` | `(64,64)` | Role |
| --- | ---: | ---: | --- |
| First optimizer update | `575.5342889290041 s` | `581.1347850020102 s` | compile-inclusive explanatory timing |
| Warm update timings | `565.5443`, `566.2002`, `566.4672`, `565.7587`, `565.9442 s` | `568.1132`, `569.5429`, `566.3587 s` | primary budget input |
| Warm update median | `565.9442223530059 s` | `568.1131881010078 s` | primary projection |
| Warm update maximum | `566.4671974140074 s` | `569.5429083190102 s` | conservative projection |
| 64-row first/compile-inclusive validation | `746.6088576859911 s` | `741.1236495470075 s` | validity and budget |
| 64-row warm validation | `731.0182942809915 s` | not separately measured; compile-inclusive cost reused | validity and budget |
| Support/export | `54.3624` first, `54.2480 s` repeat | `54.556780951999826 s` single call | support veto and budget |
| Exact 256-row audit | prior shape-generalized call failed before target execution | `2926.5343564180075 s` | direct validity and conservative budget |
| Peak TensorFlow allocator | `1,166,137,600 bytes` before the failed audit | `4,778,124,288 bytes` after completed audit | resource provenance |

Every completed optimizer update produced finite loss and gradients. Gradient
clipping was active on every timed update; this is explanatory only. Both
architectures' validation and support rows were finite, hard-status-valid, and
had zero numerical floors. The completed 256-row audit had minimum innovation
eigenvalue `0.36475295349420545`, zero floors, and exact static input shape
`[256,4]`. The `(64,64)` frozen support round-trip maximum absolute residual
was `1.7763568394002505e-15`, with maximum inverse radius
`4.000000000000002`.

## Projection

The priced protocol is exactly four 100-update tuning arms, with three
64-row validation calls and one support call per arm, followed by two
1,000-update final streams of the selected architecture, each with eleven
64-row validations, twelve support calls, and two 256-row audit calls.

| Selected final architecture | Unbuffered median | Unbuffered max | Buffered median | Buffered max |
| --- | ---: | ---: | ---: | ---: |
| `(32,32)` | `1,396,982.130265894 s` | `1,398,413.6681030176 s` | `1,746,227.6628323677 s` | `1,748,017.085128772 s` |
| `(64,64)` | `1,401,525.1565597688 s` | `1,404,768.37122058 s` | `1,751,906.445699711 s` | `1,755,960.4640257251 s` |

The larger buffered-max scenario is `1,755,960.4640257251 s`. Do not request or
authorize that amount as the intended GPU-native campaign budget. It is kept
only as the cost of the exact hybrid implementation measured on the RTX 4080
SUPER hardware class. Parallel execution could reduce its calendar time but
would not repair the host-staged numerical path.

The projection conservatively uses the direct compile-inclusive `(64,64)`
64-row validation cost for both first and later `(64,64)` validation calls, its
single support cost for first and later support calls, and the direct
compile-inclusive `(64,64)` 256-row audit cost for both architectures and every
audit call. These substitutions avoid an unsupported warm-call discount; they
are planning estimates, not measured repeat-call claims.

## Budget Accounting

| Ledger | Charged | Remaining |
| --- | ---: | ---: |
| Original r2 root | `5,007.6122855830035 s` | `6,992.3877144169965 s` of the 12,000-second material cap at failure |
| Recovery r2 root | `6,024.656027888996 s` | `967.3439721110044 s` of its 6,992-second recovery cap |
| Combined preflight material | `11,032.268313472 s` | `967.7316865280009 s` of the original 12,000-second material cap |
| User-authorized r2 remainder | `13,184.690720226998 s` | `2,152.4224067549985 s` unused |

The unused `2,152.4224067549985 s` is not enough to start any declared
100-update arm and is not being used for training.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` with a dirty concurrent worktree |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| Hardware | physical GPU 1 exposed as logical `/device:GPU:0`; NVIDIA GeForce RTX 4080 SUPER |
| Execution | TensorFlow `float64`, TF32 platform setting enabled, XLA JIT confirmed, soft placement disabled |
| GPU allocation policy | `TF_FORCE_GPU_ALLOW_GROWTH=true`; repository helper verified growth before logical initialization; no full-device preallocation |
| Target route | direct batch-native status-bearing TensorFlow target; no scalar fallback, row mapping, or sample-axis Python loop |
| Batch size | 100 for optimizer updates; exact 64 and 256 rows for validation/audit |
| Seeds | stateless timing seeds recorded in the recovery runner; no tuning/final seed was consumed |
| Data version | q=20 target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| r1 preserved 32x32 receipt | `docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r1/timing/32x32/progress.json`, SHA-256 `7815f618e6d7ac96dd23b73b1c9d46cb19d5b2b2822d5950b4c73c53306644b2` |
| r2 64x64 result | `docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r2/timing/64x64/result.json`, SHA-256 `d5fbd064ccba228458c23c4a46c2682e32521d3b56b18ef9fe69a7834f3ec7d7` |
| Projection | `docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r2/projection.json`, SHA-256 `ec90eabae6df92abe10c56246d8254c8e2421f3092e93c062bd17ec6e269f0ec` |
| Plans | original r2 preflight plan and bounded recovery plan under `docs/plans/` |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close target-identity blocker | Passed exact full-payload parity | No identity veto | none for the compared payload/source closure | retain repaired v2 identity | no target-value correctness theorem |
| Reject hybrid timing as GPU-native budget | Hybrid implementation measured completely | execution-backend veto: critical CUDA callbacks stage through host serial eigensolvers | GPU-native strict-eigh parity/performance not checked | run bounded strict-eigh localization when trusted GPU launch is available | no material campaign budget |
| Preserve both architectures as mechanics-viable | Both produced finite, status-valid direct updates | no hard mechanics veto | only 5 and 3 warm timing receipts; no quality experiment | evaluate only inside an authorized tuning campaign | no ranking or selection |
| Withhold training and HMC | Not authorized and far beyond remaining budget | campaign-budget continuation veto | campaign may stop early, but cannot be budgeted on that hope | stop after notes/tests | no convergence, posterior, or HMC claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Repaired identity, XLA/device, finite, numerical-floor, status, support, and exact-audit screens passed. The first audit harness failed before target execution and was repaired under the unchanged target/method/budget boundary. |
| Statistically supported ranking | None. No architecture or optimizer ranking is supported. |
| Descriptive-only differences | Warm update medians/maxima, validation/audit times, loss, gradients, clipping, support residuals, and allocator use. |
| Viable candidates | Both `(32,32)` and `(64,64)` remain mechanics-viable. Neither is tuning-selected or claim-ready. |
| Default readiness | Not established. |
| Next evidence needed | First, q=20 custom-versus-`tensorflow_eigh_strict` value/score/status parity and batch-100 GPU/XLA timing. Only then derive a new campaign budget; training evidence remains later. |

## Negative-Result Classification

| Failure class | Verdict |
| --- | --- |
| Implementation failure | The original shared validation wrapper generalized the batch dimension and was wrong for the target's required static-shape contract. The dedicated exact-shape XLA wrapper repaired this local harness defect. |
| Tuning failure | Not evaluated. |
| Diagnostic failure | Original 256-row r2 audit attempt failed before target execution; all prior receipts remained valid and the fresh-root recovery completed the direct audit. |
| Evidence against NeuTra | None. |
| Evidence against the prior campaign budget | Strong for the intended GPU-native target: the 20.32-day estimate prices a hybrid host-staged implementation and is invalid for GPU-native planning. |

## Post-Run Red Team

The strongest explanation for the long timings is not transient load but the
explicit host staging in both critical custom callbacks. Tight warm timings
show that the behavior is repeatable; they do not make it representative of a
GPU-native backend.

The result that would permit any new request is a parity-passed GPU-native
backend timing. Protocol reduction is a separate scientific decision and must
not be used to hide implementation overhead. The weakest current evidence is
that `tensorflow_eigh_strict` has not been run at q=20 in a trusted GPU process.

## Nonclaims

This result does not establish training improvement, convergence, posterior
correctness, HMC readiness, architecture or learning-rate superiority,
predictive validity, production/default readiness, or scientific validity. It
does not authorize the projected campaign.
