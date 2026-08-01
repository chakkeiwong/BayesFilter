# Phase 8 Continuation Subplan: T=2, N=32 Lower-Rung Candidate Ladder

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Continuation ID: `contract-e-canonical-gradient-migration-continuation-20260714-115526`

Status: `REVIEWED_LOCAL_CLAUDE_UNAVAILABLE_EXECUTION_AUTHORIZED_BY_OWNER`

## Phase Objective

Evaluate the pre-result staged Contract E-Chol candidate graph on the frozen
LGSSM center with `T=2`, `N=32`, dataset seed `81100`, and estimator seed
`80920`. Select or reject one finite ridge/transport/chunk tuple using only the
predeclared chart and same-input downstream-stability rules. Emit one immutable
JSON result per graph node and one aggregate close record.

The owner-selected center-scoped gradient loss is

```text
max_k abs(g_ContractE,k - g_Kalman,k) / abs(g_Kalman,k) <= 0.05
```

This is a center-scoped effect-size screen. It is not the FD threshold, a
confidence level, a full-box HMC certificate, or a claim about primary-shape
leaderboard accuracy.

## Entry Conditions

- Phases 0-7 are closed under the canonical Contract E-Chol policy.
- The original campaign is expired and immutable; this is a fresh two-hour
  continuation, not an extension of that campaign.
- The handoff has been reconciled so center-scoped Phase 8 may defer full-box
  HMC readiness.
- Owner amendment selects `delta_grad=0.05` and exploratory audit count `16`.
- Existing focused Phase 8 union passes `32/32`; no candidate result from that
  union is used as a numerical setting.

## Required Artifacts

- this reviewed subplan;
- the exact node harness and ladder driver source hashes;
- one fresh versioned output directory under
  `docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/lower-rung/`;
- one JSON result per evaluated node, never overwritten;
- an aggregate ladder result with selection/no-selection reason;
- a run manifest containing command, git commit, environment, seeds, dtype,
  device, JIT/TF32 settings, timeout, node count, wall time, and artifact paths;
- focused tests, `py_compile`, and `git diff --check` records; and
- a phase close record that drafts the next subplan only if the handoff gates
  pass.

## Required Checks And Reviews

- bounded one-path read-only review of this subplan and the exact harness path;
- CPU-hidden environment check before TensorFlow graph execution;
- fixed observation hash and parameter-center identity;
- Contract E route and prepared-input identity checks;
- finite values, scores, charts, branches, Cholesky diagonals, covariance and
  mean residual telemetry, and row/column transport residual telemetry;
- no dense `N x N` production allocation in the canonical streaming path;
- deterministic repeated-call equality for every node;
- ridge hard checks in listed order, with no adaptive ridge or grid expansion;
- exact step comparator edges `10->20`, `20->40`, `40->80`;
- exact chunk comparator edge `16/16->8/8`;
- value and HMC-gradient edge metrics using the owner-selected `0.05` loss;
- final selected-tuple same-program FD ladder with representable dyadic steps;
- subprocess timeout of 300 seconds per node and aggregate continuation cap of
  7200 seconds; and
- post-run local checks before any next-phase handoff.

Claude review limitation: the health probe returned `CLAUDE_PROBE_OK`, but the
substantive one-path review produced no output across two bounded windows and
was terminated. Under the repository proportionality policy, this is recorded
as reviewer unavailability; it is not a scientific pass or execution
authority. The local skeptical audit below is the execution gate for this
trusted local continuation.

## Local Skeptical Audit

`PASS_WITH_CLAUDE_UNAVAILABLE_RECORDED`: Kalman is used only as the declared
LGSSM comparator; telemetry is explanatory unless an explicit edge rule assigns
it a role; the graph and node cap are finite; audit count `16` is not used by
this lower rung; no GPU, primary-shape, HMC, nonlinear, release, or leaderboard
command is present; and every failure has a stop or candidate-rejection path.

## Evidence Contract

Question: does a finite, reproducible lower-rung Contract E-Chol program remain
valid and downstream-stable under the predeclared graph at the frozen center?

Comparator: exact float64 Kalman value and HMC-coordinate gradient for the same
`T=2` observation prefix; same-input Contract E node outputs for refinement
edges; same-program FD for the selected final tuple.

Promotion criterion: a selected tuple must pass all hard chart/finiteness/identity
checks, every listed refinement edge must satisfy relative value drift `<=0.001`
and every HMC-gradient component drift `<=0.05`, and the final FD screen must
pass its separate heuristic threshold `0.05*sqrt(5)`.

Vetoes: nonfinite output, invalid chart, identity mismatch, endpoint branch
change, sign reversal against the Kalman oracle, value/gradient edge failure,
dense allocation, timeout, missing artifact, or any attempt to tune from audit
seeds.

Explanatory diagnostics: raw covariance residual, Cholesky condition proxies,
row/column marginal residuals, chunk drift, runtime, and memory. These cannot
promote a candidate by themselves.

Nonclaims: no full-box HMC readiness, no primary-shape (`T=50,N=10000`) result,
no statistical power from audit count `16`, no superiority/ranking claim, no
nonlinear validity, no leaderboard release, and no HMC execution.

## Frozen Candidate Graph

```text
ridge scale s0 = 0.1225
ridge exponents k = {-24,-20,-16,-12,-8,-4,0,4,8}, listed increasing
finite Sinkhorn steps = {10,20,40,80}
lower-rung chunk tilings = {16/16, 8/8}
baseline epsilon = 0.5
baseline scaling = 0.9
```

Evaluate ridge candidates at `80, 8/8`, stopping at the first hard-valid ridge.
Then evaluate steps `10,20,40` while reusing the selected `80,8/8` node, and
evaluate `16/16` while reusing the selected `8/8` node. Finally rerun the
selected tuple with the complete FD ladder. At most fourteen node attempts are
allowed; no interpolation, fallback, or result-dependent expansion is allowed.

## Forbidden Claims And Actions

- Do not call `0.05` a confidence interval or reuse it for FD.
- Do not use audit count `16` to claim adequate power or default readiness.
- Do not execute GPU, `T=50,N=10000`, HMC, nonlinear migration, leaderboard
  regeneration, release, or public export in this continuation.
- Do not select a ridge from center output alone for a full-box HMC claim.
- Do not import or invoke raw-barycentric routes or v1 artifacts.
- Do not overwrite prior evidence or reuse an expired campaign output root.

## Handoff Conditions

Draft the next Phase 8 primary-shape subplan only if the selected lower-rung
tuple and final FD screen pass, every required artifact is complete, and the
owner-approved center scope/nonclaims remain intact. If no tuple passes, write a
candidate-rejection result and stop without rejecting the Contract E research
direction. Phase 9, GPU, HMC, and leaderboard work require a separate reviewed
subplan and fresh budget.

## Stop Conditions

- any continuation veto or owner-boundary change;
- aggregate wall clock reaches two hours or 7200 seconds;
- fourteen node attempts are exhausted;
- no hard-valid ridge or no stable step/chunk survivor;
- final FD heuristic fails or is inconclusive;
- artifact serialization, source identity, or process environment fails; or
- a repair would change the frozen scientific target, loss, graph, or budget.

## Close Record Requirement

At close, record the exact command actually run, node attempts and failures,
selection decision, all hard/veto diagnostics, elapsed time, remaining budget,
review verdict, and the next justified action. Mark the result center-scoped and
exploratory until a later owner-approved statistical arm passes.

## Attempt Ledger

- `attempt1`: localized driver failure after four valid immutable node
  artifacts. The equal-length three-edge slices were not applied before
  `zip(..., strict=True)`, so the driver raised before edge evaluation. No
  selection or scientific decision was produced. Repair: zip
  `STEP_COUNTS[:-1]` with `STEP_COUNTS[1:]`; rerun focused tests and use a fresh
  `attempt2` output directory. The target, graph, seeds, loss, hardware class,
  and total continuation budget remain unchanged.
- `attempt2`: exposed a plan-invalidating identification defect. At `T=1`, the
  only likelihood increment and its gradient are computed before Contract E
  reset, so all step settings produced bitwise-identical objective/score hashes
  while final-particle hashes and row residuals differed. The planned downstream
  edge metric was therefore structurally zero and could not select transport
  accuracy. The driver also reversed the declared refined-child residual
  direction. Repair: use the minimal discriminating `T=2` prefix so the first
  reset is upstream of a second likelihood increment, restore child residual
  `<=` parent residual for edges `10->20`, `20->40`, `40->80`, select the child
  count, and preserve `T=1` artifacts as invalid-design evidence. This is a
  target-preserving repair within the owner request to get results and refine;
  `N`, data seed, estimator seed, loss, graph, hardware class, and total budget
  remain unchanged.
