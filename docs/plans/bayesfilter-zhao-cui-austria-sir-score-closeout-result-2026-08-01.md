# Zhao-Cui Austria SIR Score Closeout Result

Date: 2026-08-01

Status: `STOPPED_T1_EXACT_REPLAY_CONTINUATION_VETO`

## Direct Verdict

The current T1 exact-replay score issuer candidate is rejected. Trusted
TensorFlow GPU/XLA execution succeeded, but the final budgeted T1 launch missed
the frozen parent cores by a maximum absolute residual of
`2.73558953e-13`. The admission threshold is exactly `0`, so this is a hard
veto even though the residual is numerically small.

No T1 tangent issuer artifact was issued, T2 was not run, and the analytical
total score remains unadmitted. The failure rejects the current issuer
implementation relative to the exact finite-program target. It does not reject
the Zhao-Cui Austria SIR score research direction.

## Claimed And Computed Quantities

The claimed T1 target was

\[
  \left.\nabla_\theta \ell_1(\theta)\right|_{\theta=0}
\]

for the frozen admitted Lane-B training and normalizer-calibration program.
The run first computed the origin primal replay and compared every resulting
core to the admitted parent. That computed replay differed from the parent by
`2.73558953e-13`, so derivative issuance stopped before any score could be
admitted. Equality of the replayed and admitted finite programs is therefore
wrong under the exact criterion.

The admitted value baselines remain unchanged:

| Quantity | Admitted evidence |
|---|---|
| T1 identity | `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59` |
| T1 value | `-31.1290512231882` |
| T2 identity | `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e` |
| T2 increment | `-35.154752282413156` |
| T1:T2 cumulative value | `-66.28380350560136` |
| Analytical total score | Not admitted |

## Plan Audit And Repair

The pre-execution skeptical audit passed after correcting one stale sentence
that described memory growth instead of the reviewed 6,144 MiB logical-device
hard cap. Baselines, recurrence terms, proxy diagnostics, stop conditions,
artifact paths, hardware class, and nonclaims were otherwise coherent.

T1 launch 1 exposed a localized graph-tracing defect: the parameterized SIR
model constructor performed eager `.numpy()` validation inside the XLA-traced
loss. The repair introduced a claim-local graph-native FP64 row value assembled
from the repository Austria XLA transition and isotropic-Gaussian primitives.
`bayesfilter/highdim/models.py` was added to the issuer source closure.

Before retry, focused tests established graph-native/eager row-value parity,
all-three-coordinate graph-native/analytical-score parity, real nested XLA
reverse-gradient plus outer forward-JVP execution, strict issuer closure, and
the fixed GPU memory-policy contract. This changed only the evaluation harness
for the same declared scalar; data, optimizer, seeds, thresholds, hardware, and
the total attempt budget were unchanged.

## Execution Attempts

| Attempt | Outcome | Classification |
|---|---|---|
| T1 `t1-training-jvp-01` | CUDA GPU initialized under the hard cap, then XLA tracing failed with `AttributeError: 'SymbolicTensor' object has no attribute 'numpy'`. The directory contains no files. | Localized harness failure before replay gates; repair trigger. |
| T1 `t1-training-jvp-02` | CUDA/XLA compiled and the origin replay ran. Maximum parent-core residual was `2.73558953e-13` versus required `0`. The directory contains no files. | Exact-replay hard veto; current candidate rejected. |
| T2 | Not launched; output directory remains absent. | Correctly closed because strict T1 admission did not occur. |

Both T1 launches in the plan budget were consumed. Approval-review timeouts
before process creation were infrastructure events and did not count as
scientific launches.

## GPU And Memory Evidence

Trusted preflight observed one NVIDIA GeForce RTX 4080 SUPER, UUID
`GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, with `16376 MiB` total memory.
TensorFlow created one logical GPU in `fixed_logical_device_limit` mode with
`memory_limit_mib_per_physical_device=6144`, `hard_allocator_cap=true`, memory
growth disabled, and policy configuration before logical-device initialization.

The second T1 process logged CUDA XLA service initialization, cuDNN 91700, and
successful XLA cluster compilation. The replay veto occurred before the
runner's terminal allocator report and before any tensor files were written.

## Verification

CPU-only mechanics tests intentionally hid GPUs with
`CUDA_VISIBLE_DEVICES=-1`; they are engineering/reference evidence, not claim
execution evidence.

| Check | Result |
|---|---|
| Pre-repair hard-cap/issuer suite | `15 passed, 2 warnings in 48.73s` |
| Post-repair target, nested-XLA, memory, T1, and T2 suite | `17 passed, 2 warnings in 77.70s` |
| Nested production-pattern regression alone | `1 passed, 2 warnings in 28.02s` |
| Terminal full T1/T2 score, issuer, child, and memory suite | `38 passed, 2 warnings in 234.33s` |
| `py_compile` and focused `git diff --check` before launch | Passed |

The warnings were TensorFlow Probability `distutils` deprecations.

## Failure Classification

The admitted parent, frozen clouds, and identities loaded successfully, and
the repaired route passed eager/graph value-score parity. The supported
failure class is backend/operation-order mismatch at exact origin replay, not
data corruption or a proven wrong recurrence.

The leading repair hypothesis is that zero-parameter graph-native log targets
are materialized outside the CUDA/XLA optimizer graph while active log targets
are evaluated inside it. Mathematically equal origin values can differ in
last-bit arithmetic, perturb normalized training weights, and produce the
observed residual. A future plan may test computing both active and zero
reference values inside the same compiled graph. This is a hypothesis, not
demonstrated evidence.

The tolerance must remain exactly zero. Replacing clouds, accepting an ULP
allowance, or stamping a failed replay as admitted is forbidden.

## Decision Table

| Field | Decision |
|---|---|
| Decision | Reject the current T1 issuer candidate and stop this T1:T2 closeout plan. |
| Primary criterion | Failed: maximum parent-core residual `2.73558953e-13`, required `0`. |
| Veto diagnostics | Exact-replay veto fired; derivative and T2 gates were ineligible. |
| Main uncertainty | Whether same-graph active/origin evaluation restores bitwise replay without changing the scalar. |
| Next justified action | Write a fresh bounded T1 repair plan with a prelaunch exact-origin-ratio diagnostic and a new versioned output root. |
| Not concluded | Correct analytical score, exact physical likelihood, arbitrary-theta correctness, T2 score, T3+, HMC readiness, source-faithful parameter estimation, production readiness, or superiority. |

## Inference Status

| Field | Status |
|---|---|
| Hard veto screen | Supports a hard veto against the current T1 issuer candidate. |
| Viable candidates | No issuer is admitted; same-graph origin evaluation is an untested repair hypothesis. |
| Statistically supported ranking | None; no stochastic comparison was run. |
| Descriptive-only differences | Residual magnitude and runtimes describe failures but do not weaken the exact veto. |
| Default readiness | No. |
| Next evidence needed | Fresh-plan T1 residual `0`, then manual/JVP/FD parity and strict reload before T2. |

## Evidence Ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Focused repair tests pass; the claim runner executes on CUDA/XLA. |
| Numerical validity | Failed exact origin parent-core replay. |
| Scientific interpretation | Current candidate rejected; direction remains viable; no score or comparative claim is supported. |

## Post-Run Red Team

The strongest alternative explanation is that the graph-native value is
algebraically correct but not bitwise identical to the admitted-parent
arithmetic. The result needed to overturn this verdict is zero-residual replay
under the unchanged finite program, followed by all score and loader gates.
The weakest evidence is the untested same-graph origin-ratio hypothesis.

## Run Manifest

| Field | Value |
|---|---|
| Git commit | `fb9a0679adb7c731ff2ac42551f39bdcc15222a1` plus preserved dirty-worktree implementation |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu` |
| Python | `3.11.14` |
| TensorFlow / TFP | `2.19.1` / `0.25.0` |
| Device | RTX 4080 SUPER, trusted CUDA/XLA, 6,144 MiB logical-device cap |
| Dtype / JIT | `float64`; `jit_compile=True` |
| Seeds | Frozen admitted artifact/cloud seeds; no new stochastic data |
| Plan | `docs/plans/bayesfilter-zhao-cui-austria-sir-score-closeout-plan-2026-08-01.md` |
| Attempt directories | `t1-training-jvp-01`, `t1-training-jvp-02`; both preserved empty |
| T2 directory | Absent |
| Result artifacts | This note; no issuer `result.json` or tangent tensors |
