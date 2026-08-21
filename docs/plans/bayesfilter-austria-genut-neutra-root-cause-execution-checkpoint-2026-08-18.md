# Austria GenUT NeuTra Root-Cause Execution Checkpoint

Date: 2026-08-18

Status: `XLA_T20_NAN_LOCALIZED_TF32_SEEDED_STAGE_D_BLOWUP_GUARD_CORRECT_TF32OFF_XLA_FINITE`
(see "XLA NaN Localization Campaign, 2026-08-19/20" below)

Superseded former statuses (preserved):
`GRAPH_INEQUALITY_GRAPPLER_CONSISTENT_METAOFF_BITWISE_RESTORED_XLA_T20_NONFINITE_INVALID_HARD_VETO`
`GPU_EAGER_PASS_GPU_GRAPH_WITHIN_MODE_IDENTITY_FAIL_XLA_AND_INVARIANTS_NOT_RUN`
`CPU_REPAIR_AND_DERIVATIVE_AUTHORITY_COMPLETE_GPU_CONFIRMATION_BLOCKED_BY_APPROVAL_502`

Plan:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md`

Independent review:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-second-review-reply-2026-08-17.md`

Review verdict: `AGREE`

## Research Question

Why do the Austria tangent-free and tangent-carrying batch GenUT endpoints
return different finite scalar values, and is the mismatch initiated by the
redundant iteration-start standardization in the JVP primal path?

## Evidence Contract

| Role | Declaration |
|---|---|
| Exact baseline | Current tangent-free batch endpoint on the frozen Austria `T=20`, `N=1008` target. |
| Primary criterion | Locate the first unequal particle-path tensor and show forward and reverse causal ordering arms that reproduce the two current primal kernels exactly. |
| Hard veto | Changed source/input hash, intrusive graph instrumentation, nonfinite current baseline, unequal correction-zero endpoint, or a production edit before causal confirmation. |
| Explanatory only | Condition number, coefficient size, CPU/GPU gap magnitude, runtime, and approximate score behavior. |
| Nonclaims | No dual-cap verdict, derivative correctness, NeuTra readiness, HMC readiness, tuning validity, posterior correctness, or default promotion. |

The skeptical pre-run audit passed because the baseline, primary identity
criterion, stop conditions, frozen inputs, stale-tuning prohibition, and
diagnostic-only status were explicit before execution.

## Implemented Diagnostic Lane

- Runner:
  `docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py`
- Focused tests:
  `tests/highdim/test_genut_batch_primal_parity.py`
- Route identity artifact:
  `docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attempt01/route_identity.json`

No production GenUT source was edited by this execution lane.

The subsequent shared-primal score repair is recorded in
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`.
Its terminal CPU authority artifact is
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attempt04/derivative_cpu.json`:
public value and independent forward-autodiff value agree exactly at
`T=1,2,20`, and public scores agree exactly at all three horizons. The
focused graph/XLA and existing batch tests pass. The only remaining gate is a
trusted RTX 5080 endpoint run; the launch was rejected before process creation
because the approval service returned HTTP 502.

> **Correction 2026-08-19:** the "only remaining gate" statement above was
> true when written and is now stale. The RTX 5080 endpoint runs were executed
> on 2026-08-18/19: eager PASSED, graph FAILED within-mode identity, and XLA
> returned nonfinite/invalid output at `T=20` (hard veto). See the status line
> at the top of this file and
> `docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-result-2026-08-19.md`.
> The remaining gates are now the XLA nonfiniteness localization (problem #1)
> and the graph-mode scope decision (problem #2), not a single endpoint run.

## Frozen Identity

- Git commit: `dae37183bf4421682b2ad991e2dc0d0f3c53f260`
- Target signature:
  `4845e7322685e19650024e5886e47d89c8b9c4b70c5d36a639c9b1218d39b5c3`
- Route classification: `batch_diagonal_candidate`
- TensorFlow: `2.20.0-dev0+selfbuilt`
- Frozen shapes: observations `[20,9]`, initial noise `[1008,18]`, process
  noise `[20,1008,18]`, design `[1008,18]`

The callable contains diagonal third/fourth-moment correction and final affine
restoration. It does not contain pairwise moment correction, pairwise radial
cap `2`, or coordinate cap `b=0.98,p=8`. This execution cannot establish a
failure of the promoted dual-cap route.

## CPU-Hidden Diagnostic Results

These runs explicitly used `CUDA_VISIBLE_DEVICES=-1`. They are engineering
diagnostics only and are not substitutes for the reviewed GPU campaign.

### Endpoint And Localization

- `T=20`, four corrections: value-only `-680.7359009`; value carried by the
  score endpoint `-680.6416016`; gap `0.0942993`. Both were finite and valid.
- `T=1`, zero corrections: the endpoint values were bitwise identical.
- Zero-tangent Sinkhorn particles were bitwise identical.
- Zero-tangent Contract-E restored particles were bitwise identical.
- Contract-E minimum-gap scalars differed by `1.19e-6`, while particles and
  validity agreed. This is the predeclared H3A validity-only asymmetry.
- Four-correction zero-tangent higher-moment particles differed by `0.07015`.
- The diagnostic value-order projection matched the current value kernel
  bitwise.
- The diagnostic redundant-order projection matched the current JVP primal
  kernel bitwise.
- The first unequal particle-path tensor was the standardized cloud after the
  JVP path's redundant iteration-start standardization.

This supports H1 causally in the CPU-hidden diagnostic arithmetic: the two
current kernels are exactly reproduced by the two declared operation orders.
It does not yet authorize a production repair because the reviewed GPU
confirmation has not run.

### Conditioning

Holding the same first-step `J,r` fixed, direct least squares closely matched
the float64 reference and had a much smaller equation residual than the current
normal-equation solve. The normal-equation coefficient error is therefore an
amplifier after the operation-order mismatch, not evidence that it initiates
the mismatch. Solver selection remains unpromoted.

### Invariants

- Injected tangent-only invalidity returned a NaN value/score pair with
  `program_valid=false`: fail-closed behavior passed.
- Adding another posterior row did not change row 0's value, score, or validity
  at the tested point: the batch-composition invariant passed.

## Verification

```text
python -m py_compile diagnostic runner and focused test: PASS
pytest tests/highdim/test_genut_batch_primal_parity.py: 3 passed
focused existing batch endpoint/FD tests plus new tests: 5 passed
git diff --check for execution-lane files: PASS
```

## GPU Execution Blocker

GPU occupancy was read successfully under trusted access:

```text
GPU 0: RTX 5080, 16303 MiB total, 5947 MiB used
GPU 1: RTX 4080 SUPER, 16376 MiB total, 14495 MiB used
```

GPU 0 was selected to avoid the nearly full GPU 1. Four bounded launch attempts
were then made using the `tftwogpu` environment, deterministic operations,
TF32, memory growth, and a fresh output path. The permission reviewer timed out
before process creation for every attempt. No TensorFlow GPU diagnostic ran and
no failed scientific attempt should be counted.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve H1 as the leading initiating-defect hypothesis; do not edit production source yet. | Passed exactly in CPU-hidden diagnostic arithmetic; GPU confirmation pending. | No scientific veto fired. Trusted GPU execution is externally blocked before process creation. | Whether GPU eager/graph/XLA endpoint arithmetic reproduces the same exact causal ordering. | Obtain explicit trusted GPU execution, run eager endpoint plus eager localization, then endpoint-only graph/XLA arms. | No production repair, dual-cap result, derivative validity, NeuTra/HMC readiness, or tuning claim. |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Current CPU-hidden baseline finite; correction-zero parity, fail-closed, and batch-composition checks passed. GPU screen not run. |
| Statistically supported ranking | Not applicable; this is deterministic localization, not a stochastic method comparison. |
| Descriptive-only differences | Endpoint gap magnitude, condition numbers, coefficient magnitudes, and CPU/GPU differences are explanatory only. |
| Default readiness | Not ready. The tested route is not dual-cap, production source is unchanged, and GPU/derivative/cross-model/tuning phases remain. |
| Next evidence needed | Trusted GPU eager reproduction and localization, endpoint-only graph/XLA replication, then shared-primal production repair and independent derivative validation if H1 confirms. |

## Post-Run Red Team

The strongest alternative explanation is that CPU oneDNN arithmetic makes the
operation-order defect look causally sufficient while GPU/XLA fusion exposes an
additional primal duplication. A GPU result in which either diagnostic
projection fails to reproduce its corresponding current kernel exactly would
overturn the current sufficiency conclusion and require finer localization.
The weakest evidence is therefore execution-mode generality, not the CPU-local
mapping between each current kernel and its operation order.

## GPU Resume Audit, 2026-08-18

The GPU confirmation was re-audited against the restart memo before resuming.
The audit passes for a bounded retry because the runner guards the exact frozen
target signature, adapter signature, and tensor hashes; compares the value-only
and value-plus-score endpoints for the same finite program; treats finite output
and `program_valid=true` as veto checks rather than promotion evidence; records
TensorFlow, TF32, GPU placement, memory growth, source hashes, command, commit,
and wall time; and preserves the explicit dual-cap, NeuTra, HMC, tuning, and
posterior nonclaims. The old `repair_validation_attempt06` GPU artifact is stale
for the current worktree because its recorded
`bayesfilter/highdim/cubature_genut_batch_tf.py` SHA-256 does not match the
current source after a later shared-worktree edit.

Resume budget: one trusted occupancy probe, then at most three endpoint
processes (eager, graph, and XLA), one fresh output directory per process and a
combined wall-time ceiling of 15 minutes. Run eager first. Stop without running
later modes if the frozen identity changes, output is nonfinite or invalid, or
the within-mode value identity fails. After all three modes, cross-mode scalar
drift is a confirmation veto under the restart memo. Run the invariant phase
only if all endpoint gates, including cross-mode identity, pass.

Material-default audit: `T=20`, `N=1008`, FP32, TF32 enabled, four diagonal
moment-correction steps, the repository target controls, deterministic ops, GPU
0, and the `tftwogpu` environment are frozen baseline choices from the reviewed
plan and memo, not newly promoted defaults. Their main failure mode is
mode-sensitive ill-conditioned arithmetic; the earliest diagnostic is the
separate eager/graph/XLA endpoint comparison. No tuning or cross-scope transfer
is performed in this confirmation.

The trusted occupancy probe succeeded immediately before the resume attempt:
GPU 0 was an RTX 5080 with 16,303 MiB total, 4,270 MiB used, and 8% utilization;
GPU 1 was an RTX 4080 SUPER with 16,376 MiB total, 11 MiB used, and 0%
utilization. The frozen GPU-0 eager command was then rejected before process
creation because the approval reviewer returned HTTP 502. No
`repair_validation_attempt07` process artifact was created, no campaign compute
was consumed, and this is an infrastructure failure rather than evidence about
the endpoint. Per the sandbox rejection, the same GPU action requires fresh
explicit user approval before retrying. Graph, XLA, and invariants were not run.

## Current-Source GPU Endpoint Confirmation, 2026-08-18 (Claude Code)

Executed per the restart memo's exact resume procedure under Claude Code's own
trusted GPU boundary. The prior approval-reviewer 404/502 blocker did not recur.

Pre-launch: commit `dae37183bf4421682b2ad991e2dc0d0f3c53f260` unchanged; all
six relevant source SHA-256 values matched the memo table
(`cubature_genut_batch_tf.py` = `ae8cbfb486fc90a4a38257702cf025569ca7651bb058558d341ad602cf8a976e`);
trusted occupancy probe: GPU 0 RTX 5080 16303 MiB total / 5170 MiB used / 4%,
GPU 1 RTX 4080 SUPER 11 MiB used; `repair_validation_attempt07` absent.

Commands run (exactly the memo's step-4 and step-5 commands):

```text
TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 \
MPLCONFIGDIR=/tmp/bayesfilter-matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py \
--device gpu --gpu-index 0 --phase endpoint --endpoint-modes eager \
--output .../repair_validation_attempt07/endpoint_gpu0_eager.json
# then --endpoint-modes graph --output .../repair_validation_attempt08/endpoint_gpu0_graph.json
```

Results:

| Arm | Artifact | Status | T=1 (0 steps) | T=20 (4 steps) | Gate |
|---|---|---|---|---|---|
| eager | `repair_validation_attempt07/endpoint_gpu0_eager.json` | COMPLETE, 90.8 s | `-31.12767029`, exact | value-only = score-value = `-683.0018921`, exact_equal, 0 ULP | PASS |
| graph | `repair_validation_attempt08/endpoint_gpu0_graph.json` | COMPLETE, 2342.7 s | `-31.12767029`, exact, bitwise equal to eager | value-only `-683.0575562` vs score-value `-682.4954834`; gap `0.56207275`, rel `8.23e-4`, ~9209 ULP; finite/valid but NOT equal | FAIL (within-mode identity veto) |
| xla | not run | — | — | — | stopped per step-5 rule |
| invariants | not run | — | — | — | gated on all endpoint passes |

Both processes recorded: frozen identity guard PASS, matching current source
hashes, TensorFlow `2.20.0-dev0+selfbuilt`, execution device `/GPU:0`
(RTX 5080), memory-growth configured before logical initialization and
verified on all physical GPUs, TF32 enabled, deterministic-ops env `1`, CPU
target construction. No artifact was overwritten.

Explanatory observations (no tolerance created):

- Graph T=20 score range `[-2123.37, -463.27]` versus eager
  `[404.53, 2976.94]`: sign-level disagreement of the JVP under graph tracing.
- Eager vs graph value-only cross-mode diff `0.0557`, matching the stale
  attempt06 eager/graph pair at printed precision — the graph-mode value drift
  reproduced across the source change; the new observation is that within-mode
  value/score-value identity itself fails in graph mode on the current source.
- Graph T=1 zero-correction control is bitwise mode-stable, localizing the
  divergence to the four-step higher-moment correction under graph tracing.
  *(Correction 2026-08-19: this localization is wrong relative to that claim —
  the inequality also reproduces at `T=3, steps=0` with no correction; the
  `T=1` control was too weak. See the "Graph-Mode Localization Campaign,
  2026-08-19" section below and the localization result note.)*

Classification: compiler-mode confirmation failure (current source). Not a
research-direction rejection; not evidence against the exact CPU derivative
authority or the eager GPU pass. Budget deviation: the graph arm alone
(~39 min, dominated by tracing the Python-unrolled T=20 recursion) exceeded
the 15-minute combined ceiling; it was allowed to reach a terminal artifact
rather than being killed into an uninterpretable `RUNNING` state, then the
stop condition was honored — no further processes were launched.

Next justified action: a fresh bounded reviewed plan to localize the
graph-mode value-vs-JVP primal divergence at `T=20` (e.g., graph-mode
correction-step bisection 0→4 at the endpoint level, then first-unequal-tensor
localization inside the traced shared correction core), plus a decision on the
graph-arm wall-time budget. XLA, invariants, cross-model, tuning, NeuTra, HMC,
dual-cap, and default phases remain blocked.

## Graph-Mode Localization Campaign, 2026-08-19 (Claude Code)

Executed under
`docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-plan-2026-08-18.md`
after a skeptical pre-run audit (recorded in the plan). Full result:
`docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-result-2026-08-19.md`.

Status update:
`GRAPH_INEQUALITY_GRAPPLER_CONSISTENT_METAOFF_BITWISE_RESTORED_XLA_T20_NONFINITE_INVALID_HARD_VETO`

Key findings, frozen scope and current source (`ae8cbf...`) throughout:

1. The non-XLA graph within-mode inequality reproduces at `T=2, steps=1` and
   at `T=3, steps=0` — i.e. WITHOUT any higher-moment correction. The earlier
   statement that the divergence "localizes to the four-step higher-moment
   correction" is wrong relative to that claim; the `T=1` control was too
   weak. Both the horizon recursion and the correction loop are affected;
   the gap grows with horizon/steps (34 ULP at `(3,0)` to 9209 ULP at
   `(20,4)`).
2. Setting `disable_meta_optimizer=True` before tracing restores exact
   bitwise within-mode identity in 4/4 probed failing cases, spanning both
   lanes. Classification: consistent with grappler graph rewrites
   transforming the value-only and JVP-carrying graphs differently; not
   consistent with ForwardAccumulator emitting a different primal op
   sequence. Per-pass isolation not run.
3. The XLA arm (memo step-6 command, `repair_validation_attempt09/`)
   COMPLETEd but `T=20` returned nonfinite value AND score with
   `program_valid=[false]` on both endpoints (fail-closed coherent;
   `finite_pattern_equal=true`). `T=1` was exact and bitwise equal to
   eager/graph. HARD VETO for XLA at the frozen claim scope on current
   source. The stale attempt06 XLA value was finite, so this is
   current-source/current-XLA-specific; cause not checked. Score-graph XLA
   compile took >40 min with register-spill warnings; wall 69.5 min.

Budget: 4 processes, ~83 min total wall, within the plan's 3.5 h ceiling; no
launch failures; no artifact overwritten; no tolerance created; nothing
promoted.

Open human decisions before further execution: (a) whether non-XLA graph mode
remains a claim-bearing confirmation arm (given eager passes and meta_off
restores identity); (b) scope and budget for XLA nonfiniteness localization,
which now blocks the repository-default execution target at the frozen scope.
NeuTra, HMC, tuning, cross-model, dual-cap, and default phases remain blocked.

## XLA NaN Localization Campaign, 2026-08-19/20 (Claude Code)

Executed under
`docs/plans/bayesfilter-austria-genut-xla-nan-localization-plan-2026-08-19.md`
(skeptical audit recorded there). Terminal result:
`docs/plans/bayesfilter-austria-genut-xla-nan-localization-result-2026-08-20.md`.

Status update:
`XLA_T20_NAN_LOCALIZED_TF32_SEEDED_STAGE_D_BLOWUP_GUARD_CORRECT_TF32OFF_XLA_FINITE`

Key findings (frozen scope, current source `ae8cbf...`):

1. Reproduction boundary: XLA+TF32 NaN occurs ONLY at `T=20, steps=4`.
   `(20,0)`, `(10,4)`, and all smaller scopes pass valid/finite under XLA.
   The failure is the correction-loop x late-horizon interaction.
2. Failing stage directly identified from serialized endpoint diagnostics:
   all Stage A/B/C aggregates finite and healthy in the failing run; all
   four Stage D (higher-moment correction) aggregates NaN. Candidate raw-NaN
   sites: unridged Choleskys (`cubature_genut_batch_tf.py:1273,:1288,:746`)
   and the 2x2 normal-equation solve (`:705-714`; LM branch inactive).
3. TF32 is the seed: with `enable_tensor_float_32_execution(False)` and
   everything else identical, XLA `T=20, steps=4` is FINITE and VALID on
   both endpoints (value `-680.6786`, Stage D diagnostics healthy). XLA
   miscompilation is refuted; the fail-closed guard system worked correctly.
4. Explanatory, TF32-independent: XLA also shows the within-mode value/JVP
   `exact_equal=false` program-split (like non-XLA graph mode), including
   TF32-off. Separate issue; not fixed by any TF32 decision.

The campaign changed no source, no tolerance, no default. Open owner
decision recorded in the result note: harden Stage D numerically (only
option addressing the mechanism; edits production source), vs TF32-off for
claim-bearing Austria XLA (contradicts the repo TF32 default directive), vs
scope exclusion. This decision, plus the standing graph-mode scope decision,
are the two human gates before the three-mode confirmation contract can be
revisited. NeuTra, HMC, tuning, cross-model, dual-cap remain blocked.

Infrastructure note: ~14 launch rejections from a harness permission-
classifier outage preceded P0; no process or budget consumed; same class as
historical approval 502/404 failures.
