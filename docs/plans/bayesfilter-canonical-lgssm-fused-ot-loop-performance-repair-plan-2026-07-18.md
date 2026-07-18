# Canonical LGSSM Fused OT And Loop Performance Repair Plan

Date: 2026-07-18
Campaign ID: `canonical-lgssm-fused-ot-loop-repair-20260718`
Status: `AUDITED_READY_FOR_EXECUTION`

## Research Intent Ledger

| Field | Frozen statement |
| --- | --- |
| Main question | Can the fixed finite Contract E--Chol LGSSM value and total-score program be evaluated without redundant OT work and without Python horizon unrolling, while preserving its value, total derivative, validity semantics, and Kalman comparison? |
| Mechanism under test | Reuse one finite-Sinkhorn/terminal-balance state per active reset; propagate its primal and five parameter tangents together; derive marginal telemetry from the same state/application; carry the filter through one TensorFlow horizon loop; and omit Contract E entirely for statically inactive reset routes. |
| Exact baseline | The pre-repair float64 `canonical_value_and_score_core` finite program with identical prepared tensors, 20 annealed Sinkhorn steps, explicit terminal balance count, exact chunk policy, reset mask, residual design, ridge, epsilon, scaling, and parameter coordinates. Frozen short-rung outputs and same-scalar derivatives are comparison evidence; differentiated Kalman filtering is the LGSSM scientific oracle. |
| Expected failure mode | The current route repeats Sinkhorn/balance/application work in primal, manual JVP, marginal diagnostics, inactive reset branches, and replay, while Python-unrolling the horizon. A repair can also fail by silently changing the finite scalar, omitting a Contract E total-derivative term, or retaining an uncounted full `N^2` diagnostic sweep. |
| Primary promotion criterion | Short-rung repaired value and total score agree with the frozen finite-program reference within dtype-justified bounds; same-scalar derivative checks pass; all validity gates agree; all-inactive graphs contain zero reachable OT solves; all-active execution performs one shared forward-with-JVP Sinkhorn/balance solve per time step; marginal telemetry performs no additional Sinkhorn/balance solve; the horizon is a TensorFlow functional loop; and trusted GPU/XLA warm performance makes the declared T=10 and T=50 ladder feasible. |
| Promotion veto | Wrong scalar or score, partial derivative, route/preparation identity drift, nonfinite output, invalid physical/flow/geometry/quotient/reset state, marginal failure, hidden CPU/non-XLA execution, wrong dtype/TF32 classification, wrong chunk policy, Python horizon unrolling, repeated OT solve hidden beneath a fused wrapper, or missing synchronized timing/traversal evidence. |
| Continuation veto | The frozen reference is invalid or cannot be reconstructed; repaired and baseline programs disagree and localization cannot resolve the cause within the attempt budget; T=2 trusted-GPU execution exceeds the resource cap after the planned repairs; a required GPU artifact is corrupt or incomplete; or the total campaign budget is exhausted. A candidate balance count failing its marginal screen is a repair/selection result, not a research-direction veto. |
| Repair trigger | A localized implementation, graph, instrumentation, harness, serialization, XLA, or resource failure under the unchanged target and budget triggers repair plus a fresh versioned attempt. |
| Explanatory diagnostics | Compile/first-call/warm timing, graph/HLO size, traversal counters, active reset count, exact chunk identity, peak allocator memory, per-step `TV_col` and `E_row`, Kalman value/score error, and float32/TF32 drift. These do not replace promotion gates. |
| Forbidden conclusions | No HMC readiness, nonlinear-model validity, posterior correctness, statistical superiority, production admission, default-readiness beyond this implementation route, or leaderboard completeness follows from this campaign alone. |

## Evidence Contract

### Claimed and computed quantities

The claimed implementation target is the same fixed finite scalar executed by
the pre-repair Contract E--Chol LGSSM route and its total derivative with
respect to the five physical parameters. The repaired route may change only
evaluation topology and reuse. It may not stop gradients, change the reset,
change the finite Sinkhorn schedule selected for a comparison, change the
residual design/ridge dependence, or replace the total derivative by a partial
derivative.

The independent scientific comparator remains the differentiated Kalman
likelihood for this LGSSM. Same-scalar parity proves implementation preservation;
Kalman comparison diagnoses finite-particle filtering accuracy. Neither implies
the other.

### OT work accounting

The runtime artifact must count, at minimum, per compiled invocation:

- finite Sinkhorn primal/JVP state constructions;
- terminal-balance primal/JVP schedules;
- transport application tile sweeps;
- marginal-only tile sweeps;
- Contract E reset forward/JVP factorizations;
- active and inactive reset rows/steps; and
- compiled callable invocations used for timing or replay.

`one shared OT solve` means one joint primal-plus-tangent Sinkhorn state and one
joint terminal-balance state per active filter step. A helper that calls a
primal solver and then a JVP solver that reconstructs the primal fails this
criterion. Reusing potentials but making an extra full `N^2` marginal sweep
must be reported separately and is not `zero-cost telemetry`. At `N=1024`, the
active chunk policy gives one `1024 x 1024` tile, so payload, row mass, column
mass, post-quotient column mass, and their required tangents should be
accumulated from the same transient tile traversal.

### Marginal definitions and selection

For row masses `r_i`, post-quotient column masses `q_j`, and normalized source
weights `w_j`, define

\[
 E_{\mathrm{row}}=\max_i |r_i-1|,
 \qquad
 TV_{\mathrm{col}}=\frac{1}{2N}\sum_j |q_j-Nw_j|.
\]

The design screen is `E_row <= 0.01` and `TV_col <= 1e-4` at every active reset.
These are owner-approved algorithmic probability-error tolerances, not
roundoff claims and not a theorem about accumulated likelihood/score error.
The terminal-balance design ladder is `0,1,2,3,5,8`; zero is explanatory only
under the current positive-balance canonical boundary. The smallest positive
candidate passing every frozen design case is selected. It is checked without
retuning on a disjoint audit set before use in longer horizons. If no candidate
passes, do not invent a larger default from Kalman results; inspect the residual
trajectory and revise the numerical plan visibly.

## Default And Assumption Audit

| Choice | Provenance and status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| Contract E--Chol reset and total derivative | Binding repository policy; canonical target | Only reset/gradient route eligible for this work | Partial or historical raw route could appear faster; source-closure and same-scalar derivative tests veto it |
| 20 annealed Sinkhorn steps | Existing finite program; frozen baseline, not newly promoted | Isolates terminal-balance and topology repairs | Could be excessive or insufficient, but changing it would confound scalar parity; record residuals and defer annealed-step tuning |
| Balance candidates `0,1,2,3,5,8` | Bounded diagnostic ladder motivated by observed rapid early convergence; hypothesis | Finds the smallest count under the new probability criteria without repeating the obsolete 50-step target | Grid may miss a required count; emit per-iteration trajectories and stop for visible redesign rather than extrapolate from Kalman output |
| `TV_col <= 1e-4`, `E_row <= 0.01` | Owner-approved tolerances; reviewed campaign criteria | Directly measure total column probability error and maximum row mass error | They may be too weak for downstream value/score accuracy; Kalman and same-scalar gates remain separate and no general default claim is made |
| Chunk policy `dpf_transport_exact_divisor_cap3000_v1` | Completed repository policy; reviewed default | For `N=1024`, `K=1024`, one block | Reintroducing small chunks multiplies tile count; fail-closed policy and artifact identity test catch it. No chunk tuning is authorized here |
| Float64 | Reference lane | Separates topology/derivative preservation from production precision | Slow and not TF32 evidence; artifact labels it reference-only |
| Float32 with TF32 enabled | Repository production target | Required production performance/precision lane | Drift can exceed useful bounds; paired float64 comparison and Kalman diagnostics veto unsupported promotion |
| Five simultaneous parameter tangents | Fixed LGSSM dimension `p=5`; target-specific implementation | Avoids five separate scalar traversals and preserves manual total JVP | Tangent batching may increase memory or alter indexing; per-coordinate AD/FD and peak-memory checks catch it |
| Static all-active/all-inactive specialization | Reset masks are prepared fixed inputs | Makes inactive OT absence provable and avoids executing discarded branches | Mixed schedules may require dynamic compaction that XLA handles poorly; test all-on, all-off, and mixed separately before choosing topology |
| 8 GiB GPU limit | Owner-approved existing campaign limit | Prevents machine-wide pressure and matches prior evidence | Fused tangent state may OOM; catch resource exceptions, record them, and stop or repair without raising the limit |

## Skeptical Pre-Execution Audit

Audit date: 2026-07-18. Verdict: `PASS_AFTER_REVISION`.

1. **Wrong baseline risk:** the first draft could compare only with Kalman and
   miss a changed finite particle scalar. Repaired by making frozen same-program
   parity the implementation gate and Kalman a separate scientific oracle.
2. **Proxy promotion risk:** graph loops, traversal counters, and faster timing
   cannot establish score correctness. Repaired by requiring value, total-score,
   same-scalar derivative, and validity parity before performance promotion.
3. **Hidden work risk:** reusing potentials alone still permits an additional
   `N^2` column diagnostic sweep. Repaired by counting tile sweeps separately
   and requiring same-tile accumulation for the current one-block `N=1024`
   production witness.
4. **Stale chunk recommendation:** increasing chunks was already completed and
   guarded. Repaired by freezing the current policy and forbidding a new ladder.
5. **Arbitrary balance count:** five iterations looked plausible but was not
   established. Repaired by a marginal-only design/audit ladder and a ban on
   selecting from Kalman accuracy.
6. **Unfair timing:** the old timer combines compilation, execution, and a
   second replay. Repaired by device synchronization and separate trace,
   compile-plus-first, and warm timings; first and warm outputs also supply the
   replay comparison.
7. **Inactive-branch ambiguity:** `tf.cond` with a fixed captured predicate can
   still retain both branch functions in a graph library. Repaired by factory
   specialization and reachable-function inspection, not a top-level op-name
   check alone.
8. **Dynamic mixed-mask risk:** compacting arbitrary active rows introduces
   shape and recompilation hazards. Repaired by evaluating schedule grouping or
   fixed-shape masked execution only after all-on/all-off correctness; mixed
   topology cannot block the homogeneous LGSSM evidence unless it changes the
   public prepared-input contract.
9. **Misleading long run:** a fast T=50 run could still answer the wrong
   question. Repaired by mandatory T=2 correctness/resource promotion before
   T=10, and T=10 before T=50.
10. **Environment mismatch:** GPU detection or execution inside a restricted
    sandbox can look broken. All GPU/CUDA commands use trusted/escalated
    execution; CPU checks deliberately set `CUDA_VISIBLE_DEVICES=-1`.

The commands and artifacts below answer the stated question after these
revisions. No material unexamined default remains; open choices are explicitly
classified as hypotheses or reference settings.

## Phases

### Phase 0: Frozen Reference And Work Instrumentation

Create a small immutable short-rung reference for `T=1,2`, float64, fixed
prepared inputs, all-active/all-inactive/mixed reset masks, and current 20+50
finite schedules. Add structural counters that distinguish state construction,
balance, transport, telemetry, reset, and invocation work. Counters must not
change numerical tensors or be implemented using stateful operations inside
the XLA claim route; structural test hooks or a separately traced instrumented
route are acceptable.

Exit: frozen values/scores/validity/telemetry exist; counter tests expose the
known redundant baseline; current chunk identity is recorded. Stop if the
reference itself is nonfinite or internally derivative-inconsistent.

### Phase 1: Shared OT State And Same-Pass Diagnostics

Refactor the streaming transport internals to expose one joint
primal-plus-tangent state: potentials and tangents, terminal-balance result,
transport numerator/mass and tangents, and direct marginal telemetry. Remove
the call from forward quotient telemetry back into finite Sinkhorn/balance.

For `K=N`, accumulate payload, row mass, raw column mass, and post-quotient
column mass in the same transient tile application. For a future multi-block
case, an additional coupling-reduction sweep may be retained only if explicitly
counted and shown not to rerun Sinkhorn/balance; it cannot satisfy the strict
one-tile witness gate by being mislabeled free.

Exit: primitive value/JVP parity passes against the pre-repair helpers; direct
`TV_col` and `E_row` agree with an independent dense reference at tiny `N`; one
joint state construction and no diagnostic solver reconstruction are observed.

### Phase 2: Fused Contract E And Filter Step

Add forward-with-JVP variants for the row quotient and Contract E reset. Reuse
forward moments, Cholesky factors, centered clouds, and affine state in the
reset JVP instead of calling the forward reset again for every parameter.
Create one LGSSM step that advances particles, particle tangents, log weights,
weight tangents, likelihood, score, validity, and telemetry together.

Exit: all five coordinate tangents equal the separated reference and
TensorFlow forward AD/FD on short rungs; one active step has one shared OT
state; no stopped or transported-cloud-only derivative is reachable.

### Phase 3: Reset Specialization And Functional Horizon

Replace both Python horizon loops and Python history lists with one
`tf.while_loop` and `TensorArray` telemetry. Issue factory-specialized
all-inactive and active-capable callables. The all-inactive route must have no
reachable Contract E/OT functions. Mixed masks must preserve exact results; use
schedule grouping or another fixed-shape XLA-compatible topology selected from
a short graph/compile comparison, not unreviewed dynamic shapes.

Replace horizon-dependent or iterative-solver Python unrolling reachable from
the production callable, including Sinkhorn-schedule telemetry. Fixed tiny
parameter-coordinate construction may remain only if graph-size evidence shows
it does not scale with `T`.

Exit: `T=1,2,5` loop/reference parity; AST/reachable-graph guard passes;
graph-operation and HLO-size growth from `T=2` to `T=10/50` is bounded and not
linear body duplication; inactive traversal count is zero.

### Phase 4: Harness And Marginal Schedule

Create a current non-archival driver that records run manifest, direct
`TV_col`, `E_row`, structural traversal counts, active rows, source hashes,
chunk policy, dtype, TF32, XLA, device, synchronized timing, HLO/graph metrics,
memory, and structured exceptions. Never overwrite an attempt directory.

Run the terminal-balance candidate ladder on frozen `T=2,N=128` float64 design
seeds `81300..81307`, then the selected positive count once on untouched audit
seeds `81320..81327`. Do not inspect Kalman values during selection. Bind the
selected count and marginal policy identifiers into preparation/route identity
before longer claim-bearing runs.

Exit: one positive count passes every design and audit reset. Otherwise write a
selection result with residual trajectories and stop before the production
ladder.

### Phase 5: Correctness And Trusted GPU/XLA T=2 Gate

Run CPU-hidden float64 primitive/short-rung tests, then trusted GPU probes and
the all-active/all-inactive T=2 route under the 8 GiB cap. Compare float64
repaired output to the frozen finite-program reference and differentiated
Kalman oracle. Run float32/TF32 against float64 with exact prepared-input
semantics and record drift without inventing a precision threshold after seeing
the result; float32 advances only if finite, valid, same-sign, free of order-one
coordinate disagreement, and the detailed drift is scientifically reviewable.

Performance promotion requires zero inactive OT state constructions, one joint
active construction per step, zero diagnostic solver reconstructions, and a
warm runtime projection below the T=10/T=50 node caps below.

Exit: all correctness, graph, resource, and traversal gates pass. A candidate
that is merely faster but wrong does not advance.

### Phase 6: T=10 And T=50 Ladder

Run trusted GPU/XLA with a fresh versioned directory at each horizon:

1. `T=10`, first as one-seed resource witness, then the frozen claim-bearing
   seed batch only if the witness passes;
2. `T=50`, first as one-seed resource witness, then the frozen claim-bearing
   seed batch only if the witness passes.

Node caps are 20 minutes for the T=10 witness/claim node and 45 minutes for the
T=50 witness/claim node. These are resource stop conditions, not scientific
thresholds. Each horizon must retain traversal, marginal, value/score, Kalman,
precision, replay, timing, and memory evidence. Stop on an invalid artifact,
resource exception, or cap breach; do not silently reduce `N`, seeds, horizon,
or reset work in the claim-bearing arm.

Exit: T=10 and T=50 artifacts pass their declared engineering and numerical
gates, or a precise blocker result records why the ladder stopped.

### Phase 7: Terminal Audit And Result

Run focused tests, source/graph audits, artifact schema validation, and a
post-run red team. Write a result containing separate engineering, numerical,
and scientific ledgers; decision and inference-status tables; exact manifest;
attempt/failure/repair history; and explicit nonclaims.

## Required Checks

CPU-only checks deliberately hide GPU devices:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m py_compile <touched Python paths>
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q <focused Contract E/LGSSM tests>
git diff --check
```

GPU preflight and all TensorFlow GPU/XLA runs require trusted/escalated
execution. Before the first run:

```bash
nvidia-smi
/home/chakwong/anaconda3/envs/tf-gpu/bin/python <framework GPU device probe>
```

Exact benchmark commands are frozen in the phase result after the repaired
driver exists; each command must specify or emit campaign ID, attempt ID,
horizon, particle count, seeds, balance count, chunk-policy ID, dtype, TF32,
JIT, GPU memory cap, timeout, plan path, and unique output root.

## Artifact Contract And Budget

Versioned root:
`docs/benchmarks/artifacts/canonical_lgssm_fused_ot_loop_repair_20260718/`.

Required outputs:

- `phase0_reference/` and traversal inventory;
- primitive and loop parity JSON;
- balance design/audit ledger;
- T=2 float64 and TF32 artifacts;
- T=10 and T=50 resource/claim artifacts if promoted;
- one attempt ledger with failure classification and remaining budget;
- terminal result at
  `docs/plans/bayesfilter-canonical-lgssm-fused-ot-loop-performance-repair-result-2026-07-18.md`.

Budget: at most six trusted-GPU launch attempts, four total GPU hours, and the
per-node caps above. Local CPU tests and compile checks are routine work. A
localized repair/retry consumes this budget and needs no renewed approval while
the target, data, method, criteria, GPU class, privacy boundary, and total
budget remain unchanged.

## Handoff And Stop Conditions

Every phase writes its status into the attempt ledger before the next phase.
Advance only when its exact exit conditions pass. Do not stop merely because a
balance candidate or performance candidate fails when the next declared phase
is designed to diagnose or repair that failure. Stop only on a stated
continuation veto, exhausted budget, a required change to the scientific
contract, or an external/irreversible boundary requiring human approval.
