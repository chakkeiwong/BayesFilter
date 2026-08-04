# Zhao-Cui Austria SIR Score Closeout Plan

Date: 2026-08-01

Status: `STOPPED_T1_EXACT_REPLAY_CONTINUATION_VETO`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | Does manual paired-core differentiation equal the origin total derivative of the admitted Lane-B T1:T2 finite value program? |
| Candidate | Exact replay of the frozen T1 and T2 training plus normalizer-calibration algorithms, differentiated offline by TensorFlow forward-over-reverse JVP. |
| Expected failure | Backend or operation-order drift prevents exact parent-core replay; a disconnected or unstable training JVP fails same-replay finite differences. |
| Promotion criterion | Exact origin parent-core replay, manual/JVP score parity, same-replay centered-FD parity, strict issuer reload, and peak TensorFlow allocation at most 6 GiB at both T1 and T2. |
| Promotion veto | Any failed criterion above, stale or mismatched input/source identity, nonfinite score, omitted carried T1 marginal derivative, runtime autodiff/FD, or score/value scalar mismatch. |
| Continuation veto | Invalid admitted parent/input artifact, proven wrong recurrence, irreparable exact-replay mismatch under the frozen backend, 6 GiB breach, or two failed launches for the same issuer. |
| Repair trigger | A localized runner, serialization, source-closure, or operation-order failure with unchanged scalar and evidence contract. |
| Explanatory only | Physical Fisher score, UKF/GenUT/SGQF values or scores, validation residuals, runtime, and allocator utilization below the cap. |
| Must not be concluded | Exact physical likelihood, arbitrary-theta correctness, HMC readiness, T5+, source-faithful parameter estimation, method ranking, or scientific superiority. |

## Claimed Scalar And Score

The admitted origin value is the fixed Lane-B retained-object program

\[
  L_{1:2}(0) = \ell_1(0)+\ell_2(0), \qquad
  \ell_t(\theta)=\log Z_t(\theta)-c_t,
\]

where training points, proposal objects, coordinate frames, shifts `c_t`,
defensive masses `tau`, ranks, bases, optimizer schedule, and calibration rows
are frozen. For each `theta`, the definition reruns that same deterministic
training and calibration program with its target factors evaluated at `theta`.
Only its value and total derivative at `theta=0` are sought:

\[
  s_{1:2}(0)=\left.\nabla_\theta L_{1:2}(\theta)\right|_{\theta=0}.
\]

At T2 the active fixed-row target is

\[
 \log q_2^\theta(z_2,z_1)
 = \log \widehat p_1^\theta(z_1)
 + \log f_\theta(z_2\mid z_1)
 + \log g_\theta(y_2\mid z_2),
\]

so its row score is

\[
 \nabla_\theta\log q_2^θ
 = s_{\widehat p_1}(z_1)
 + s_f(z_2\mid z_1)
 + s_g(y_2\mid z_2).
\]

The proposal density and its sampled rows are frozen at the origin and have
zero derivative. The carried marginal term is mandatory. The compact runtime
child uses

\[
 G_{t,k}(\theta)=G_{t,k}(0)
 + \sum_{a=1}^3\theta_a\dot G_{t,k,a}
\]

and manual paired-core contractions. This child represents the first-order
origin value/score only; it is not an arbitrary-`theta` replay oracle.

## Source Grounding And Classification

The inspected Zhao-Cui source anchors are:

- sequential posterior and adjacent-state targets: paper Eqs. (9)-(12),
  Eq. (15), and Algorithm 2 in
  `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:344-721`;
- squared-TT marginal construction: paper Proposition 2 at the same local
  source `:594-655` and author
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25-85`;
- conditional proposal and correction context: paper Eqs. (20)-(23) and
  Algorithm 3 at `:812-924`;
- author event order, previous marginal, fit, and log-normalizer accumulation:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72-135`.

Classification:

| Operation | Classification | Reason |
|---|---|---|
| Previous marginal times transition times observation | `source_faithful` operation inside the local route | It matches paper Eq. (15), Algorithm 2(a), and `full_sol.m:72-80,132-135`. |
| Squared-TT mass and prefix marginal contraction | `source_faithful` operation inside the local route | It matches Proposition 2 and `marginalise.m:25-85`. |
| Frozen proposal rows, frames, ranks, bases, shifts, and schedules | `fixed_hmc_adaptation` mechanics, without HMC authorization | Random/adaptive inputs are frozen for a deterministic derivative. |
| External three-coordinate parameter input, optimizer JVP, and linear tangent-core child | `extension_or_invention` | The author program estimates parameters jointly and does not define this conditional origin-score issuer. |

No result from this plan may call the assembled parameter-score route
source-faithful.

## Evidence Contract

- Exact baseline T1 identity:
  `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`,
  value `-31.1290512231882`.
- Exact baseline T2 identity:
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`,
  increment `-35.154752282413156`, cumulative value
  `-66.28380350560136`.
- T1 pass: zero maximum residual over every replayed parent core; manual
  value/score equals replay value/JVP; each coordinate passes centered FD with
  step `1e-4`, absolute tolerance `2e-4`, and relative tolerance `2e-4`.
- T2 pass: the same zero core residual and manual/JVP parity; cumulative score
  passes centered FD with step `1e-4`, absolute tolerance `3e-4`, and relative
  tolerance `3e-4`.
- Both issuers must bind the admitted parents, exact prepared inputs, optimizer
  semantics, source closure, tensor hashes, hard-gate evidence, runtime manual
  backend, XLA use, and the upstream issuer chain.
- Runtime score uses no autodiff or finite differences. Offline JVP and FD are
  issuance and diagnostic tools only.
- Versioned output root:
  `docs/plans/artifacts/zhao-cui-austria-sir-score-closeout-20260801/`.

## Skeptical Plan Audit

Audit verdict: `PASS_FOR_BOUNDED_T1_THEN_T2_EXECUTION`.

- Wrong baseline: rejected. The plan loads only the admitted Lane-B fixed
  parents above; APF, generic retained grids, original adaptive TT-cross/ALS,
  ratio bridges, and the Fisher-calibrated tangent are excluded.
- Proxy promotion: rejected. Physical Fisher, UKF, GenUT, SGQF, validation
  loss, and runtime cannot admit the score.
- Hidden recurrence term: rejected. Tests require the T1 normalized retained-
  marginal score in every T2 row and tie that row score to diagnostic autodiff.
- Environment mismatch: controlled. The original T1 cloud is GPU-backend
  bound, so claim execution uses the repository TensorFlow GPU/XLA route with
  the reviewed 6,144 MiB logical-device hard cap configured before device
  initialization. CPU is used only for focused mechanics/reference checks with
  `CUDA_VISIBLE_DEVICES=-1`.
- T2 optimizer mismatch: rejected by audit and test. The admitted program uses
  4096 rows, 256-row microbatches, 16 equally accumulated gradients, one global
  clip and Adam transition per update, and 96 updates. The functional replay
  matches one admitted Keras/XLA full-cloud update exactly.
- Tail mismatch: not active for issuance. Both selected T2 training and
  calibration inputs record `zero_target_count=0`; the unrelated untouched
  value row with extended-real zero density is not consumed by this replay.
- Artifact insufficiency: rejected. Strict loaders revalidate inputs, current
  source closure, evidence thresholds, all 108 tangent tensors at T2, and child
  identities. A Boolean `PASS` field alone is insufficient.
- Missing stop conditions: repaired below.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| Rank 4, order 5, L1 `1e-9`, LR `3e-4` | Admitted T1/T2 selected parent; frozen baseline, not a new default | Retuning would change the scalar | Exact parent-core replay |
| 96 updates and Keras-3 Adam constants/order | Admitted runners; reviewed default for this frozen replay | ULP or optimizer-state drift | One-step and T2 full-cloud exact parity tests |
| T2 256-row microbatch | Admitted T2 runner; frozen baseline | Different accumulation changes cores and tangent | Exact full-cloud parity and origin replay |
| Frozen shift and `tau` derivative | Definition of the selected finite program; hypothesis made explicit | Including their recalibration derivative changes the target | Same-replay FD of the declared scalar |
| Centered-FD step/tolerances | Diagnostic hypothesis | Step error could mask a wrong JVP | Three coordinate residuals retained in issuer |
| Linear tangent-core runtime child | First-order representation | Misuse away from the origin | Nonclaim plus origin-only loader parity |
| 6 GiB allocator cap | Campaign resource bound | Full-history/autodiff retention blows memory | A 6,144 MiB logical-device limit before initialization plus peak TensorFlow allocator evidence before tensor writes |

## Pre-Mortem

- A run could pass while differentiating the wrong quantity if it compared to a
  Fisher score. Prevention: only same-replay JVP, manual contraction, and FD
  are promotion evidence.
- A run could reproduce the value while omitting the T1 marginal tangent at
  T2. Prevention: rowwise sum decomposition and active-weight JVP tests.
- A run could write a plausible artifact after stale code or prepared data.
  Prevention: repository-issued source/input/tensor closure and strict reload.
- A run could fail for infrastructure rather than mathematics through GPU
  denial, XLA compilation, or serialization. Such a failure triggers one
  localized repair/retry and does not reject the score construction.
- A run could exceed memory during nested autodiff. The issuer stops above 6
  GiB before writing tangents; no full history is serialized or retained at
  runtime.

## Execution Sequence And Budget

1. Run CPU-hidden focused mechanics, optimizer parity, issuer round-trip, and
   tamper tests. Require all pass.
2. Run escalated `nvidia-smi`, then an escalated TensorFlow GPU device probe
   that verifies the explicit 6,144 MiB logical-device limit before
   initialization. This is the repository-documented hard-cap exception to
   memory growth.
3. Run the T1 issuer in a fresh directory with exact origin tolerance. Budget:
   at most two launches, including one localized repair/retry, and at most 90
   minutes total wall time.
4. Strictly reload the resulting T1 issuer. If it passes, run the T2 issuer in
   a fresh directory. Budget: at most two launches and at most 120 minutes
   total wall time.
5. Run the complete focused T1/T2 suite and `git diff --check`; refresh the
   result note with exact commands, wall times, devices, scores, residuals,
   identities, and inference status.
6. Only if both issuer artifacts pass may a fresh plan define the T3 and later
   recurrence. Do not jump directly to T5/T10/T20.

Stop at the first scientific continuation veto. A platform rejection before
process launch is an execution blocker, not a failed candidate, and must not be
circumvented.

## Execution Attempt Ledger

| Attempt | Classification | Result | Repair/next action |
|---|---|---|---|
| T1 launch 1, `t1-training-jvp-01` | Localized harness/graph-tracing failure before replay gates | The GPU process started under the 6,144 MiB cap, then `SpatialSIRSSM.__post_init__` called `.numpy()` because the parameter-scaled model was constructed inside the XLA-traced loss. The fresh directory exists but contains no files. | Preserve the empty directory. Use a claim-local graph-native FP64 value assembled from the existing Austria XLA transition and isotropic-Gaussian primitives, require parity with the eager value and analytical score, and use the one remaining T1 launch at `t1-training-jvp-02`. The scalar, data, optimizer, thresholds, hardware class, and total budget are unchanged. |
| T1 launch 2, `t1-training-jvp-02` | Exact-replay hard veto | The repaired CUDA/XLA route ran under the 6,144 MiB hard cap and reached the parent-core gate. Maximum core residual was `2.73558953e-13`, not the required exact `0`, so no tangent or result files were issued. | Preserve the empty directory and stop this plan. T2 remains closed. A future freshly budgeted plan may test computing both active and origin log targets inside the same compiled CUDA/XLA graph; it must not relax the zero tolerance or reuse either failed output path. |

The two-launch T1 budget is exhausted. The second result rejects the current
issuer candidate under the exact finite-program criterion; it does not reject
the score research direction. No third T1 launch and no T2 launch are permitted
under this plan.
