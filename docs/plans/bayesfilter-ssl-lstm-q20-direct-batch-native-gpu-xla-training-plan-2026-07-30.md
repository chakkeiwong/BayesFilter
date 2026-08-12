# q=20 Direct Batch-Native GPU/XLA NeuTra Training Plan

Date: 2026-07-30
Status: `STOPPED_UNDER_BUDGETED_AND_TARGET_IDENTITY_INVALID`

## Research Intent Ledger

| Role | Predeclared contract |
| --- | --- |
| Main question | Can the q=20 SSL-LSTM target and a dense-IAF NeuTra optimizer execute as one direct batch-native TensorFlow GPU/XLA program, then produce two independently trained transports that improve untouched reverse-KL audit loss over their corresponding untrained transports without a numerical, support, or artifact veto? |
| Candidate mechanism | Direct leading-axis vectorization of the q=20 principal-square-root UKF value, analytic score, numerical status, transport forward/log determinant, reverse-KL reduction, gradient, and Adam update. No CPU value/score pool and no scalar or row-mapped fallback are allowed. |
| Exact comparator | For each final training seed, the same architecture, hyperparameters, initialization seed, and untouched audit base draws evaluated before versus after training. The historical CPU and CPU-worker/GPU artifacts are context and warm starts, not claim comparators. |
| Promotion criterion | Both frozen final seeds pass every hard veto and have a paired trained-minus-untrained untouched-audit one-sided 95% upper bound below zero. This promotes only the label `GPU_XLA_TRAINING_SCREEN_PASSED`. |
| Promotion veto | Either final seed fails the paired untouched-audit criterion, frozen reload/support checks, or required manifest/artifact validation. A promotion veto rejects the current frozen protocol, not NeuTra as a research direction. |
| Continuation veto | Binding cannot be repository-issued; status cannot be preserved and hard-enforced; direct and legacy batch value/score disagree; GPU memory growth or XLA cannot be verified; batch size is one; a scalar/row-mapped fallback is observed; any nonfinite value, score, status-critical diagnostic, loss, gradient, parameter, support result, or artifact appears; campaign cap is exhausted; or required disjoint seeds/artifacts are unavailable. |
| Repair trigger | A local source, interface, export, or checkpoint defect with unchanged target, method, criteria, hardware class, privacy boundary, and campaign budget may be repaired once and retried with a new attempt receipt. A failed tuning candidate triggers selection among remaining valid candidates, not criterion relaxation. |
| Explanatory diagnostics | Calibration and validation losses, gradient norms, clipping, scale-log/hidden tails, numerical-status values, runtime, compiler warm-up, allocator current/peak bytes, and TF32 state. They do not establish convergence, ranking, posterior correctness, or HMC readiness. |
| What must not be concluded | No HMC or MCMC convergence claim; no posterior, predictive, or scientific-validity claim; no architecture superiority; no global hyperparameter optimum; no default or production-readiness promotion. HMC is not authorized by this plan. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Required evidence before its status can advance |
| --- | --- |
| Engineering correctness | Focused CPU-hidden tests; repository-issued batch binding; exact value/score parity; one direct compiled GPU optimizer update; frozen export/reload; source hashes captured at import. |
| Numerical validity | Per-row target status is produced by the same batch-native kernel call and hard-enforced for every optimizer update; no active floor, roundoff repair, classified-invalid row, nonpositive/nonfinite minimum innovation eigenvalue, or nonfinite value/score is admitted; support round trip is at most `1e-9`. |
| Scientific interpretation | Frozen candidate selection uses heldout tuning validation; final seeds are new; the final audit is untouched; paired uncertainty is reported; loss remains a training-screen endpoint, not a posterior or HMC endpoint. |

## Evidence Contract

| Item | Contract |
| --- | --- |
| Target | `batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="compiled_custom_op")`, bound by `require_batch_native_neutra_target`. |
| Backend | TensorFlow/TFP, direct batch-native `float64` target and transport computation, GPU visible, XLA JIT on. No NumPy runtime computation is allowed in the runner; Python standard-library arithmetic is allowed at the artifact boundary. |
| Training batch | 100. This is inherited from the completed CPU diagnostic and earlier q=20 studies as a warm-start hypothesis; the mechanics gate must prove static shape and resource viability. It is not promoted as a universal batch default. |
| Capacity candidates | `(32,32)` and `(64,64)`. Both passed the prior fixed-protocol hard screen, while their ranking was unresolved. They are bounded capacity hypotheses, not defaults. |
| Optimizer candidates | Initial learning rates `2e-4` and `4e-4`, crossed with the two capacities. `4e-4` is the completed CPU warm start; halving it is the smallest predeclared stability alternative. Adam `(beta1=0.9, beta2=0.999, epsilon=1e-7)`, initialization scale `0.01`, per-variable clip `10`, ELU, three IAF stages, and scale cap `1` remain inherited procedure controls held fixed for this bounded campaign and are explicitly not optimized. |
| Tuning ladder | Four candidates, one tuning initialization/training seed each, 100 updates, heldout validation at steps 0, 50, and 100. Rank only candidates without a hard veto by paired step-100-minus-step-0 heldout mean; ties within `0.05` mean loss are unresolved and select the lower-capacity/lower-rate arm as the compute-conservative representative. The `0.05` band is a convenience indifference band, not statistical equivalence. |
| Final seeds | Two new streams not used in tuning. Each uses the frozen selected controls, recurring controller validation every 100 updates, 200-update patience, at most one factor-0.5 learning-rate repair, two post-repair no-improvement cycles, and 1,000 updates maximum. |
| Untouched audit | 256 stateless base draws per final stream from a seed never used for tuning, optimizer updates, recurring validation, repair, stopping, or checkpoint selection. Compare the frozen selected-best state with its corresponding untrained state on identical draws. |
| Support probe | Origin plus positive/negative radius-4 coordinate points in the NeuTra base chart; require finite target/transformed score, frozen forward/inverse/forward round trip at most `1e-9`, and recovered inverse radius at most `4.30`. |
| Required artifact | Versioned root `docs/plans/artifacts/ssl-lstm-q20-direct-batch-native-gpu-xla-training-2026-07-30/r1/`, containing preflight, mechanics, tuning arms, repository-issued tuning selection, final seed results/checkpoints/frozen payloads, terminal summary, exact commands, source hashes, and result note. |

The one-sided critical value `1.6694022215079607` is inherited from the
64-row paired validation convention (Student-t, 63 degrees of freedom). The
256-row audit will use the appropriate Student-t 255-degree critical value
recorded by the runner. Neither value is a convergence threshold.

## Seed Partition

| Role | Seeds | Status |
| --- | --- | --- |
| Mechanics | initialization `(20260730, 5101)`, update `(20260730, 5201)` | Debug-only, excluded from tuning and final claims |
| Tuning | initialization `(20260730, 6101)`, updates `(20260730, 6201)`, validation `(20260730, 6301)` | Candidate calibration/selection only |
| Final seed A | initialization `(20260730, 7101)`, updates `(20260730, 7201)`, recurring validation `(20260730, 7301)`, audit `(20260730, 7401)` | Claim-bearing replication 1 |
| Final seed B | initialization `(20260730, 7102)`, updates `(20260730, 7202)`, recurring validation `(20260730, 7302)`, audit `(20260730, 7402)` | Claim-bearing replication 2 |

These are convenience-chosen stateless seeds declared before execution. They
provide reproducibility and partition separation, not proof of population-wide
seed robustness.

## Default And Assumption Audit

| Choice | Provenance and status | Justification | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- | --- |
| Direct q=20 target | Current repository candidate route; hypothesis pending engineering gate | It is the only existing q=20 leading-axis TensorFlow target intended to avoid row mapping | Missing authority/status surface could make a nominally batch-native run ineligible | Binding issuance, status parity, AST/source-closure payload, CPU-hidden focused tests |
| Compiled custom principal square root | Repository GPU/XLA default for this target; reviewed default for the mechanics gate | Exercises the intended GPU/XLA kernel rather than the CPU `tensorflow_eigh` exception | Custom-op/XLA/device failure or classified-invalid rows | One-update mechanics artifact and per-row status |
| Batch 100 | Inherited warm start from prior q=20 runs | Directly tests the established training scale | Excess memory or poor GPU utilization; inherited batch may not be optimal | Allocator before/after/peak and canary wall time |
| `(32,32)` / `(64,64)` | Prior unresolved fixed-protocol candidates | Preserves the only q=20 capacity comparison with hard-valid historical arms | Search is narrow and could miss a better architecture | Report as bounded search; no architecture superiority claim |
| LR `2e-4` / `4e-4` | One derived half-rate and one CPU warm start | Tests the most immediate cross-backend optimizer sensitivity | Both can be too small/large | 100-update loss and numerical-status trajectories |
| Initialization `0.01`, clip `10` | Inherited warm-start controls, not reviewed GPU optima | Keeps the bounded search within the authorized compute | Could confound capacity/LR selection | Gradient/clipping and scale/hidden telemetry; list as unresolved default debt |
| 100 tuning updates | Convenience budget ladder | Gives two heldout checkpoints per candidate before expensive final replication | May select on transient behavior | Preserve full validation trajectory; final streams can veto but cannot retune |
| 1,000 final updates | Convenience cap at half the completed CPU maximum | Fits two independent final streams within the authorized wall budget | Plateau may not be reached | Stop reason and terminal-minus-best loss; do not claim convergence |
| Two final seeds | Minimum bounded replication | Detects gross seed instability | Underpowered for broad ranking | Require same-direction paired audit pass; state remaining uncertainty |
| TF32 enabled | Repository GPU execution metadata convention | Records the platform mode, though the claim-bearing tensors are `float64` | Could be mistaken for an FP32/TF32 numerical claim | Record dtype and state; make no TF32 performance/accuracy claim |

## Budget And Stop Contract

- The user supplied 20,000 additional seconds of headroom. This plan caps all
  material GPU mechanics, tuning, final training, validation, audit, and one
  in-scope repair/retry at 18,000 cumulative wall seconds. The remaining 2,000
  seconds are reserved for focused engineering tests and artifact validation.
- The update ceilings are one mechanics update, 400 total tuning updates, and
  2,000 total final updates. Reaching a wall or update ceiling stops the phase;
  it does not relax a criterion.
- Check budget before every update and retain a progress artifact at every
  validation boundary. Never overwrite a completed attempt or arm.
- Stop before tuning if mechanics does not pass. Stop before final training if
  no tuning arm is hard-valid or if the tuning selection artifact is missing,
  stale, caller-stamped, or does not match the exact target/binding/source
  scope.
- Stop the affected arm immediately on any hard veto. Another valid candidate
  may continue only if the campaign boundary and remaining budget are intact.
- Do not launch HMC under any outcome.

## Pre-Mortem

| Misleading outcome | Distinguishing check |
| --- | --- |
| The command succeeds while bypassing numerical status | Bind the exact callable, require the source/dependency closure, make invalid status poison the same optimizer update, and persist binding/status evidence. |
| Loss improves because tuning and audit draws overlap | Persist all stateless seed pairs and fail closed on overlap; final audit seeds are never passed to training or controller code. |
| A GPU transport update still evaluates targets row by row | Binding source audit forbids Python loops, `tf.map_fn`, `tf.vectorized_map`, callbacks, and scalar delegation; artifact records all fallback flags false. |
| A CPU or non-XLA path is mislabeled | Require visible logical GPU, verified memory growth before device initialization, XLA on, compiled custom-op backend, and GPU device placement in the mechanics artifact. |
| A proxy loss pass is called HMC readiness | Promotion label is restricted to GPU/XLA training-screen viability and every result table must preserve the HMC/posterior nonclaims. |
| The run fails because inherited optimizer controls are poor | The bounded LR/capacity screen and gradient/status trajectories distinguish immediate tuning failure; fixed initialization/clip remain explicit rescue hypotheses. |

## Skeptical Pre-Execution Audit

- Wrong baseline: corrected. The comparator is each final candidate's own
  untrained transport under identical untouched audit draws, not the CPU run
  or a historically weaker architecture.
- Proxy promotion: corrected. Reverse-KL audit loss can promote only a training
  screen. It cannot establish posterior correctness, HMC readiness, or a new
  repository default.
- Missing stop conditions: corrected through binding, numerical, device,
  artifact, update, wall-time, support, and seed-overlap vetoes.
- Unfair architecture comparison: the tuning grid crosses both capacities with
  both learning rates, but fixed initialization and clipping remain a stated
  limitation. No architecture ranking will be made.
- Hidden inherited defaults: batch, optimizer moments, initialization, clip,
  activation, stage count, scale cap, and TF32 state are all classified above.
- Stale context: the 2026-07-30 CPU result is complete and diagnostic-only; the
  older GPU route uses CPU value/score workers and is historical engineering
  evidence, not the direct target under test.
- Environment mismatch: mechanics must establish the managed-session visible
  GPU, memory-growth policy, compiled custom op, XLA, device placement, and
  allocator telemetry before material tuning.
- Artifact adequacy: the planned binding, status, untrained/trained paired
  audits, frozen payloads, source hashes, progress records, and run manifest can
  answer the narrow training-screen question.

Audit decision: `PASS_AFTER_STATUS_CONTRACT_ENGINEERING_GATE`. The plan may
execute the target-contract repair and focused tests now. Material tuning and
final training remain conditional on a passing GPU mechanics artifact.

## Planned Commands

The exact commands will use the repository environment, set
`TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import, select one available
physical GPU without preempting another lane, and invoke the new runner in
`contract`, `mechanics`, `tune`, then `final` modes. Every executed command and
realized `CUDA_VISIBLE_DEVICES` value will be written to the run manifest.

## Terminal Result Requirements

The result note must include a decision table, inference-status table, hard
vetoes, viable candidates, whether any ranking is statistically supported,
descriptive-only differences, default-readiness, run manifest, negative-result
classification if applicable, and a post-run red team. It must state the
claimed target, quantity actually computed, their relationship, supporting
artifact, and what remains unproved.

## Close Record

The direct GPU/XLA mechanics gate passed, but material training stopped before
the first tuning arm reached its step-50 validation receipt. That arm exhausted
its 3,600-second outer bound, proving the predeclared four-arm tuning and
two-stream final protocol cannot fit the remaining campaign budget.

A post-run provenance audit also found that the pre-repair q=20 target's
synthetic observations differed between CPU and GPU construction at about
`1e-16`, yielding different target signatures. Therefore the `r1` contract and
mechanics artifacts do not bind one hardware-invariant scientific target and
are engineering evidence only. Static target-data construction is now pinned
to `/CPU:0` under a new v2 identity policy for a future campaign, but trusted
GPU signature parity could not be rerun because the approval reviewer timed
out twice. No tuning selection, final stream, or HMC run was launched.

Terminal interpretation and the next preflight are recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-result-2026-07-30.md`.
